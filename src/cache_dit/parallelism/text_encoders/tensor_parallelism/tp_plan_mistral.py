import torch
from typing import Union
from transformers import (
    MistralModel,
    Mistral3Model,
    MistralForCausalLM,
    Mistral3ForConditionalGeneration,
)
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
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


_supported_mistral_classes = (
    MistralModel,
    Mistral3Model,
    MistralForCausalLM,
    Mistral3ForConditionalGeneration,
)


# Text Encoder for FLUX.2 series models.
@TextEncoderTensorParallelismPlannerRegister.register("MistralModel")
@TextEncoderTensorParallelismPlannerRegister.register("Mistral3Model")
@TextEncoderTensorParallelismPlannerRegister.register("MistralForCausalLM")
@TextEncoderTensorParallelismPlannerRegister.register("Mistral3ForConditionalGeneration")
class MistralTensorParallelismPlanner(TextEncoderTensorParallelismPlanner):
    def apply(
        self,
        text_encoder: torch.nn.Module,
        parallelism_config: ParallelismConfig,
        **kwargs,
    ) -> torch.nn.Module:
        assert isinstance(
            text_encoder, _supported_mistral_classes
        ), "MistralTensorParallelismPlanner can only be applied to Mistral Language Models."
        tp_mesh = self.mesh(parallelism_config=parallelism_config)
        text_encoder = self.parallelize_text_encoder(
            text_encoder=text_encoder,
            tp_mesh=tp_mesh,
        )

        return text_encoder

    def parallelize_text_encoder(
        self,
        text_encoder: Union[
            MistralModel,
            Mistral3Model,
            MistralForCausalLM,
            Mistral3ForConditionalGeneration,
        ],
        tp_mesh: DeviceMesh,
    ):

        if isinstance(
            text_encoder,
            (
                Mistral3Model,
                MistralForCausalLM,
                Mistral3ForConditionalGeneration,
            ),
        ):
            if isinstance(text_encoder, MistralForCausalLM):
                model = text_encoder.model
            else:
                # Mistral3ForConditionalGeneration, Mistral3Model
                model = text_encoder.language_model
        else:
            model = text_encoder

        for _, block in model.layers.named_children():
            assert isinstance(block, MistralDecoderLayer)
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
                Mistral3Model,
                MistralForCausalLM,
                Mistral3ForConditionalGeneration,
            ),
        ):
            if isinstance(text_encoder, MistralForCausalLM):
                text_encoder.model = model
            else:
                # Mistral3ForConditionalGeneration, Mistral3Model
                text_encoder.language_model = model
        else:
            text_encoder = model

        maybe_empty_cache()

        return text_encoder
