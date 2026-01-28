from diffusers.models.transformers.hunyuan_transformer_2d import (
    HunyuanDiTBlock,
)
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


@TensorParallelismPlannerRegister.register("HunyuanDiT")
class HunyuanDiTTensorParallelismPlanner(TensorParallelismPlanner):
    def apply(
        self,
        transformer: nn.Module,
        parallelism_config: ParallelismConfig,
        **kwargs,
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
        Parallelize HunyuanDiT transformer blocks.

        HunyuanDiT has a unique architecture with:
        - Dual attention (self-attention and cross-attention)
        - Skip connections with skip_linear layer
        - Norm layers (norm1, norm2, norm3)
        - FFN layer
        - Long skip connections between blocks
        """

        for i, block in enumerate(transformer.blocks):
            assert isinstance(block, HunyuanDiTBlock)

            # Split attention heads across TP devices
            tp_size = tp_mesh.size()
            shard_divisible_attr(
                block.attn1,
                "heads",
                tp_size,
                what="attn1",
                context="HunyuanDiTTensorParallelismPlanner",
            )
            shard_divisible_attr(
                block.attn2,
                "heads",
                tp_size,
                what="attn2",
                context="HunyuanDiTTensorParallelismPlanner",
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

            logger.debug(f"Parallelized HunyuanDiT block {i} with TP size {tp_mesh.size()}")

        return transformer
