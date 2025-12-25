import torch
from torch.distributed import DeviceMesh, init_device_mesh
from torch.distributed._tensor import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from diffusers import QwenImageTransformer2DModel
from cache_dit.parallelism.parallel_config import ParallelismConfig
from .tp_plan_registers import (
    TensorParallelismPlanner,
    TensorParallelismPlannerRegister,
)
from .tp_utils import shard_divisible_attr

from cache_dit.logger import init_logger

logger = init_logger(__name__)


@TensorParallelismPlannerRegister.register("QwenImageTransformer2DModel")
class QwenImageTensorParallelismPlanner(TensorParallelismPlanner):
    def apply(
        self,
        transformer: torch.nn.Module,
        parallelism_config: ParallelismConfig,
        **kwargs,
    ) -> torch.nn.Module:
        assert (
            parallelism_config.tp_size is not None and parallelism_config.tp_size > 1
        ), "parallel_config.tp_size must be set and greater than 1 for tensor parallelism"

        device_type = torch.accelerator.current_accelerator().type
        tp_mesh: DeviceMesh = init_device_mesh(
            device_type=device_type,
            mesh_shape=[parallelism_config.tp_size],
        )

        transformer = self.parallelize_transformer(
            transformer=transformer,
            tp_mesh=tp_mesh,
        )

        return transformer

    def parallelize_transformer(
        self,
        transformer: QwenImageTransformer2DModel,
        tp_mesh: DeviceMesh,
    ):
        from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock

        for _, block in transformer.transformer_blocks.named_children():
            assert isinstance(block, QwenImageTransformerBlock)
            shard_divisible_attr(
                block.attn,
                "heads",
                tp_mesh.size(),
                what="attn",
                context="QwenImageTensorParallelismPlanner",
            )
            layer_plan = {
                "attn.to_q": ColwiseParallel(),
                "attn.to_k": ColwiseParallel(),
                "attn.to_v": ColwiseParallel(),
                "attn.to_out.0": RowwiseParallel(),
                "img_mod.1": ColwiseParallel(output_layouts=Replicate()),
                "img_mlp.net.0.proj": ColwiseParallel(),
                "img_mlp.net.2": RowwiseParallel(),
                "attn.add_q_proj": ColwiseParallel(),
                "attn.add_k_proj": ColwiseParallel(),
                "attn.add_v_proj": ColwiseParallel(),
                "attn.to_add_out": RowwiseParallel(),
                "txt_mod.1": ColwiseParallel(output_layouts=Replicate()),
                "txt_mlp.net.0.proj": ColwiseParallel(),
                "txt_mlp.net.2": RowwiseParallel(),
            }
            parallelize_module(
                module=block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_plan,
            )
        return transformer
