import torch
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from torch.distributed import DeviceMesh
from diffusers import GlmImageTransformer2DModel
from cache_dit.parallelism.config import ParallelismConfig
from .tp_plan_registers import (
    TensorParallelismPlanner,
    TensorParallelismPlannerRegister,
)
from .tp_utils import shard_divisible_attr

from cache_dit.logger import init_logger

logger = init_logger(__name__)


@TensorParallelismPlannerRegister.register("GlmImageTransformer2DModel")
class GlmImageTensorParallelismPlanner(TensorParallelismPlanner):
    def apply(
        self,
        transformer: torch.nn.Module,
        parallelism_config: ParallelismConfig,
        **kwargs,
    ) -> torch.nn.Module:
        tp_mesh = self.mesh(parallelism_config=parallelism_config)
        transformer = self.parallelize_transformer(
            transformer=transformer,
            tp_mesh=tp_mesh,
        )

        return transformer

    def parallelize_transformer(
        self,
        transformer: GlmImageTransformer2DModel,
        tp_mesh: DeviceMesh,
    ):
        from diffusers.models.transformers.transformer_glm_image import GlmImageTransformerBlock

        for _, block in transformer.transformer_blocks.named_children():
            assert isinstance(block, GlmImageTransformerBlock)
            shard_divisible_attr(
                block.attn1,
                "heads",
                tp_mesh.size(),
                what="attn1",
                context="GlmImageTensorParallelismPlanner",
            )
            layer_plan = {
                "attn1.to_q": ColwiseParallel(),
                "attn1.to_k": ColwiseParallel(),
                "attn1.to_v": ColwiseParallel(),
                "attn1.to_out.0": RowwiseParallel(),
                "ff.net.0.proj": ColwiseParallel(),
                "ff.net.2": RowwiseParallel(),
            }
            parallelize_module(
                module=block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_plan,
            )
        return transformer
