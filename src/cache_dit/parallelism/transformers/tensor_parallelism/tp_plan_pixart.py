from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from cache_dit.logger import init_logger
from cache_dit.parallelism.config import ParallelismConfig

from .tp_plan_registers import (
    TensorParallelismPlanner,
    TensorParallelismPlannerRegister,
)
from .tp_utils import shard_divisible_attr

logger = init_logger(__name__)


@TensorParallelismPlannerRegister.register("PixArt")
class PixArtTensorParallelismPlanner(TensorParallelismPlanner):
    def apply(
        self,
        transformer: nn.Module,
        parallelism_config: ParallelismConfig,
        **_kwargs,
    ) -> nn.Module:
        tp_mesh = self.mesh(parallelism_config=parallelism_config)
        transformer = self.parallelize_transformer(
            transformer=transformer,
            tp_mesh=tp_mesh,
        )

        return transformer

    def parallelize_transformer(
        self,
        transformer: nn.Module,
        tp_mesh: DeviceMesh,
    ):
        """
        Parallelize PixArt transformer blocks.

        PixArt uses BasicTransformerBlock with:
        - Self-attention (attn1)
        - Cross-attention (attn2)
        - Feed-forward network (ff)
        - Standard normalization layers
        """
        for i, block in enumerate(transformer.transformer_blocks):
            # Split attention heads across TP devices
            tp_size = tp_mesh.size()
            shard_divisible_attr(
                block.attn1,
                "heads",
                tp_size,
                what="attn1",
                context="PixArtTensorParallelismPlanner",
            )
            shard_divisible_attr(
                block.attn2,
                "heads",
                tp_size,
                what="attn2",
                context="PixArtTensorParallelismPlanner",
            )

            # Create layer plan for tensor parallelism
            layer_plan = {
                # Self-attention projections (column-wise)
                "attn1.to_q": ColwiseParallel(),
                "attn1.to_k": ColwiseParallel(),
                "attn1.to_v": ColwiseParallel(),
                "attn1.to_out.0": RowwiseParallel(),
                # Cross-attention projections (column-wise)
                "attn2.to_q": ColwiseParallel(),
                "attn2.to_k": ColwiseParallel(),
                "attn2.to_v": ColwiseParallel(),
                "attn2.to_out.0": RowwiseParallel(),
                # Feed-forward network
                "ff.net.0.proj": ColwiseParallel(),
                "ff.net.2": RowwiseParallel(),
            }

            # Apply tensor parallelism to the block
            parallelize_module(
                module=block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_plan,
            )

            logger.debug(f"Parallelized PixArt block {i} with TP size {tp_mesh.size()}")

        return transformer
