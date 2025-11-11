import torch
from torch import nn
from torch.distributed import DeviceMesh, init_device_mesh
from torch.distributed._tensor import Replicate
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


@TensorParallelismPlannerRegister.register("Mochi")
class MochiTensorParallelismPlanner(TensorParallelismPlanner):
    def apply(
        self,
        transformer: torch.nn.Module,
        parallelism_config: ParallelismConfig,
        **kwargs,
    ) -> torch.nn.Module:
        assert (
            parallelism_config.tp_size is not None
            and parallelism_config.tp_size > 1
        ), (
            "parallel_config.tp_size must be set and greater than 1 for "
            "tensor parallelism"
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
        # Mochi uses transformer_blocks similar to other DiT models
        # Apply tensor parallelism to each transformer block
        for _, block in transformer.transformer_blocks.named_children():
            # Adjust attention heads for tensor parallelism
            if hasattr(block, "attn") and hasattr(block.attn, "heads"):
                block.attn.heads //= tp_mesh.size()

            # Define parallelization plan for Mochi transformer blocks
            layer_plan = {
                # Self-attention layers
                "attn.to_q": ColwiseParallel(),
                "attn.to_k": ColwiseParallel(),
                "attn.to_v": ColwiseParallel(),
                "attn.to_out.0": RowwiseParallel(),
                # Feed-forward layers (assuming standard structure)
                "ff.net.0.proj": ColwiseParallel(),
                "ff.net.2": RowwiseParallel(),
                # Cross-attention layers (if present)
                "attn.add_q_proj": ColwiseParallel(),
                "attn.add_k_proj": ColwiseParallel(),
                "attn.add_v_proj": ColwiseParallel(),
                "attn.to_add_out": RowwiseParallel(),
                # Context feed-forward (if present)
                "ff_context.net.0.proj": ColwiseParallel(),
                "ff_context.net.2": RowwiseParallel(),
            }

            # Handle linear layer normalization (if present, similar to Flux)
            if (
                hasattr(block, "norm1")
                and hasattr(block.norm1, "linear")
                and block.norm1.linear is not None
            ):
                layer_plan["norm1.linear"] = ColwiseParallel(
                    output_layouts=Replicate()
                )

            if (
                hasattr(block, "norm1_context")
                and hasattr(block.norm1_context, "linear")
                and block.norm1_context.linear is not None
            ):
                layer_plan["norm1_context.linear"] = ColwiseParallel(
                    output_layouts=Replicate()
                )

            # Apply the parallelization plan to this block
            parallelize_module(
                module=block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_plan,
            )

        # Handle Mochi-specific layers if any
        # This may need adjustment based on actual Mochi architecture
        if hasattr(transformer, "proj_in") and transformer.proj_in is not None:
            parallelize_module(
                module=transformer,
                device_mesh=tp_mesh,
                parallelize_plan={"proj_in": ColwiseParallel()},
            )

        if (
            hasattr(transformer, "proj_out")
            and transformer.proj_out is not None
        ):
            parallelize_module(
                module=transformer,
                device_mesh=tp_mesh,
                parallelize_plan={"proj_out": RowwiseParallel()},
            )

        return transformer
