import torch
from transformers import GlmImageForConditionalGeneration

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


@TextEncoderTensorParallelismPlannerRegister.register("GlmImageForConditionalGeneration")
class GlmImageTensorParallelismPlanner(TextEncoderTensorParallelismPlanner):
    def apply(
        self,
        text_encoder: torch.nn.Module,
        parallelism_config: ParallelismConfig,
        **kwargs,
    ) -> torch.nn.Module:
        assert isinstance(
            text_encoder, GlmImageForConditionalGeneration
        ), "GlmImageTensorParallelismPlanner can only be applied to GlmImageForConditionalGeneration."
        tp_mesh = self.mesh(parallelism_config=parallelism_config)
        text_encoder = self.parallelize_text_encoder(
            text_encoder=text_encoder,
            tp_mesh=tp_mesh,
        )

        return text_encoder

    def parallelize_text_encoder(
        self,
        text_encoder: GlmImageForConditionalGeneration,
        tp_mesh: DeviceMesh,
    ):
        from transformers import GlmImageTextModel
        from transformers.models.glm_image.modeling_glm_image import GlmImageTextDecoderLayer

        model: GlmImageTextModel = text_encoder.model.language_model

        for _, block in model.layers.named_children():
            assert isinstance(block, GlmImageTextDecoderLayer)
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
        maybe_empty_cache()
