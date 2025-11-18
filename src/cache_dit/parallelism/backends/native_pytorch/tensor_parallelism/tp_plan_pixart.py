import torch
from torch import nn
from torch.distributed import DeviceMesh, init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from cache_dit.logger import init_logger
from cache_dit.parallelism.parallel_config import ParallelismConfig

from .tp_plan_registers import (
    TensorParallelismPlanner,
    TensorParallelismPlannerRegister,
)

logger = init_logger(__name__)


@TensorParallelismPlannerRegister.register("PixArt")
class PixArtTensorParallelismPlanner(TensorParallelismPlanner):
    def apply(
        self,
        transformer: nn.Module,
        parallelism_config: ParallelismConfig,
        **_kwargs,
    ) -> nn.Module:
        assert parallelism_config.tp_size is not None and parallelism_config.tp_size > 1, (
            "parallel_config.tp_size must be set and greater than 1 for " "tensor parallelism"
        )

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
            block.attn1.heads //= tp_mesh.size()
            block.attn2.heads //= tp_mesh.size()

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
