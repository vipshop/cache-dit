import torch
from transformers import Qwen2_5_VLForConditionalGeneration
from torch.distributed import DeviceMesh, init_device_mesh

from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from cache_dit.logger import init_logger
from cache_dit.utils import maybe_empty_cache
from cache_dit.parallelism.parallel_config import ParallelismConfig

from .tp_plan_registers import (
    TextEncoderTensorParallelismPlanner,
    TextEncoderTensorParallelismPlannerRegister,
)

logger = init_logger(__name__)


@TextEncoderTensorParallelismPlannerRegister.register("Qwen2_5_VLForConditionalGeneration")
class Qwen25VLTensorParallelismPlanner(TextEncoderTensorParallelismPlanner):
    def apply(
        self,
        text_encoder: torch.nn.Module,
        parallelism_config: ParallelismConfig,
        **kwargs,
    ) -> torch.nn.Module:
        assert isinstance(
            text_encoder, Qwen2_5_VLForConditionalGeneration
        ), "Qwen25VLTensorParallelismPlanner can only be applied to Qwen2_5_VLForConditionalGeneration"
        text_encoder_world_size = parallelism_config.text_encoder_world_size
        device_type = torch.accelerator.current_accelerator().type
        tp_mesh: DeviceMesh = init_device_mesh(
            device_type=device_type,
            mesh_shape=[text_encoder_world_size],
        )

        text_encoder = self.parallelize_text_encoder(
            text_encoder=text_encoder,
            tp_mesh=tp_mesh,
        )

        return text_encoder

    def parallelize_text_encoder(
        self,
        text_encoder: Qwen2_5_VLForConditionalGeneration,
        tp_mesh: DeviceMesh,
    ):
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer

        for _, block in text_encoder.model.language_model.layers.named_children():
            assert isinstance(block, Qwen2_5_VLDecoderLayer)
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
        maybe_empty_cache()

        return text_encoder
