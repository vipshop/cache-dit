import torch
from typing import Union
from transformers import GlmModel, GlmForCausalLM, Glm4Model, Glm4ForCausalLM
from transformers.models.glm.modeling_glm import GlmDecoderLayer
from transformers.models.glm4.modeling_glm4 import Glm4DecoderLayer

from torch.distributed import DeviceMesh

from torch.distributed.tensor import Replicate
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


_supported_glm_classes = (
    GlmModel,
    GlmForCausalLM,
    Glm4Model,
    Glm4ForCausalLM,
)


# Text Encoder for CogView4 series models.
@TextEncoderTensorParallelismPlannerRegister.register("GlmModel")
@TextEncoderTensorParallelismPlannerRegister.register("Glm4Model")
@TextEncoderTensorParallelismPlannerRegister.register("GlmForCausalLM")
@TextEncoderTensorParallelismPlannerRegister.register("Glm4ForCausalLM")
class GlmTensorParallelismPlanner(TextEncoderTensorParallelismPlanner):
    def apply(
        self,
        text_encoder: torch.nn.Module,
        parallelism_config: ParallelismConfig,
        **kwargs,
    ) -> torch.nn.Module:
        assert isinstance(
            text_encoder, _supported_glm_classes
        ), "GlmTensorParallelismPlanner can only be applied to Glm Language Models."
        tp_mesh = self.mesh(parallelism_config=parallelism_config)
        text_encoder = self.parallelize_text_encoder(
            text_encoder=text_encoder,
            tp_mesh=tp_mesh,
        )

        return text_encoder

    def parallelize_text_encoder(
        self,
        text_encoder: Union[GlmModel, GlmForCausalLM, Glm4Model, Glm4ForCausalLM],
        tp_mesh: DeviceMesh,
    ):

        if isinstance(text_encoder, (GlmForCausalLM, Glm4ForCausalLM)):
            model = text_encoder.model
        else:
            model = text_encoder

        assert isinstance(model, (GlmModel, Glm4Model)), "model must be an instance of GlmModel."
        for _, block in model.layers.named_children():
            assert isinstance(block, (GlmDecoderLayer, Glm4DecoderLayer))
            layer_plan = {
                "self_attn.q_proj": ColwiseParallel(),
                "self_attn.k_proj": ColwiseParallel(),
                "self_attn.v_proj": ColwiseParallel(),
                "self_attn.o_proj": RowwiseParallel(),
                "mlp.gate_up_proj": ColwiseParallel(output_layouts=Replicate()),
                "mlp.down_proj": RowwiseParallel(output_layouts=Replicate()),
            }

            parallelize_module(
                module=block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_plan,
            )

        if isinstance(text_encoder, (GlmForCausalLM, Glm4ForCausalLM)):
            text_encoder.model = model
        else:
            text_encoder = model

        maybe_empty_cache()

        return text_encoder
