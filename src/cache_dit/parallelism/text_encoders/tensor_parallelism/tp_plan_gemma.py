import torch
from typing import Union
from transformers.models.t5gemma.modeling_t5gemma import T5GemmaEncoder
from transformers import (
    GemmaModel,
    Gemma2Model,
    Gemma3Model,
    GemmaForCausalLM,
    Gemma2ForCausalLM,
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
)
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer
from transformers.models.t5gemma.modeling_t5gemma import T5GemmaEncoderLayer

from torch.distributed import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from cache_dit.logger import init_logger
from cache_dit.utils import maybe_empty_cache
from cache_dit.parallelism.config import ParallelismConfig

from .tp_plan_registers import (
    TextEncoderTensorParallelismPlanner,
    TextEncoderTensorParallelismPlannerRegister,
)

logger = init_logger(__name__)

_supported_gemma_classes = (
    T5GemmaEncoder,
    GemmaModel,
    Gemma2Model,
    Gemma3Model,
    GemmaForCausalLM,
    Gemma2ForCausalLM,
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
)


# Text Encoder Lumina-Image, prx series models.
@TextEncoderTensorParallelismPlannerRegister.register("T5GemmaEncoder")
@TextEncoderTensorParallelismPlannerRegister.register("GemmaModel")
@TextEncoderTensorParallelismPlannerRegister.register("Gemma2Model")
@TextEncoderTensorParallelismPlannerRegister.register("Gemma3Model")
@TextEncoderTensorParallelismPlannerRegister.register("GemmaForCausalLM")
@TextEncoderTensorParallelismPlannerRegister.register("Gemma2ForCausalLM")
@TextEncoderTensorParallelismPlannerRegister.register("Gemma3ForCausalLM")
@TextEncoderTensorParallelismPlannerRegister.register("Gemma3ForConditionalGeneration")
class GemmaTensorParallelismPlanner(TextEncoderTensorParallelismPlanner):
    def apply(
        self,
        text_encoder: torch.nn.Module,
        parallelism_config: ParallelismConfig,
        **kwargs,
    ) -> torch.nn.Module:
        assert isinstance(
            text_encoder, _supported_gemma_classes
        ), "GemmaTensorParallelismPlanner can only be applied to Gemma Language Models."
        tp_mesh = self.mesh(parallelism_config=parallelism_config)
        text_encoder = self.parallelize_text_encoder(
            text_encoder=text_encoder,
            tp_mesh=tp_mesh,
        )

        return text_encoder

    def parallelize_text_encoder(
        self,
        text_encoder: Union[
            T5GemmaEncoder,
            GemmaModel,
            Gemma2Model,
            Gemma3Model,
            GemmaForCausalLM,
            Gemma2ForCausalLM,
            Gemma3ForCausalLM,
            Gemma3ForConditionalGeneration,
        ],
        tp_mesh: DeviceMesh,
    ):

        # NOTE: Gemma3 can be used as a multi-modal backbone. In those cases the actual
        # language model is nested under `language_model` (and sometimes `language_model.model`).
        # We need to unwrap to the module that has `layers`.
        if isinstance(text_encoder, Gemma3ForConditionalGeneration) and hasattr(
            text_encoder, "language_model"
        ):
            model_container = getattr(text_encoder, "language_model")
            model = getattr(model_container, "model", model_container)
        elif isinstance(
            text_encoder,
            (
                GemmaForCausalLM,
                Gemma2ForCausalLM,
                Gemma3ForCausalLM,
            ),
        ):
            model = text_encoder.model
        else:
            model = text_encoder
            if not hasattr(model, "layers") and hasattr(model, "language_model"):
                model_container = getattr(model, "language_model")
                model = getattr(model_container, "model", model_container)

        if not hasattr(model, "layers"):
            raise AttributeError(
                f"{model.__class__.__name__} object has no attribute 'layers'. "
                "If this is a multi-modal Gemma3 model, expected the language model to be "
                "under `language_model` (and optionally `language_model.model`)."
            )

        assert isinstance(model, torch.nn.Module), "model must be a torch.nn.Module"
        for _, block in model.layers.named_children():
            assert isinstance(
                block,
                (
                    GemmaDecoderLayer,
                    Gemma2DecoderLayer,
                    Gemma3DecoderLayer,
                    T5GemmaEncoderLayer,
                ),
            ), (
                f"Unsupported layer type {block.__class__.__name__} for Gemma TP. "
                "Expected a GemmaDecoderLayer/Gemma2DecoderLayer/Gemma3DecoderLayer/T5GemmaEncoderLayer."
            )
            layer_plan = {
                "self_attn.q_proj": ColwiseParallel(),
                "self_attn.k_proj": ColwiseParallel(),
                "self_attn.v_proj": ColwiseParallel(),
                "self_attn.o_proj": RowwiseParallel(),
                "mlp.gate_proj": ColwiseParallel(),
                "mlp.up_proj": ColwiseParallel(),
                "mlp.down_proj": RowwiseParallel(),
            }

            parallelize_module(
                module=block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_plan,
            )

        if isinstance(
            text_encoder,
            (
                GemmaForCausalLM,
                Gemma2ForCausalLM,
                Gemma3ForCausalLM,
                Gemma3ForConditionalGeneration,
            ),
        ):
            # NOTE: Gemma3ForConditionalGeneration may store the LM under `language_model`.
            if isinstance(text_encoder, Gemma3ForConditionalGeneration) and hasattr(
                text_encoder, "language_model"
            ):
                language_model = getattr(text_encoder, "language_model")
                if hasattr(language_model, "model"):
                    language_model.model = model
                else:
                    text_encoder.language_model = model
            else:
                text_encoder.model = model
        else:
            text_encoder = model

        maybe_empty_cache()

        return text_encoder
