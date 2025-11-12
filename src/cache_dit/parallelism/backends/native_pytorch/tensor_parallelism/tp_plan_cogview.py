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


@TensorParallelismPlannerRegister.register("CogView3Plus")
@TensorParallelismPlannerRegister.register("CogView4")
@TensorParallelismPlannerRegister.register("CogVideoX")
class CogViewTensorParallelismPlanner(TensorParallelismPlanner):
    def apply(
        self,
        transformer: torch.nn.Module,
        parallelism_config: ParallelismConfig,
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
        for _, block in transformer.transformer_blocks.named_children():
            # Reduce attention heads for tensor parallelism
            block.attn1.heads //= tp_mesh.size()

            layer_plan = {
                # Self-attention projections
                "attn1.to_q": ColwiseParallel(),
                "attn1.to_k": ColwiseParallel(),
                "attn1.to_v": ColwiseParallel(),
                "attn1.to_out.0": RowwiseParallel(),
                # Feed-forward networks
                "ff.net.0.proj": ColwiseParallel(),
                "ff.net.2": RowwiseParallel(),
                "norm1.linear": ColwiseParallel(output_layouts=Replicate()),
            }

            # Add norm2.linear if present (CogVideoX)
            if hasattr(block, "norm2") and hasattr(block.norm2, "linear"):
                layer_plan["norm2.linear"] = ColwiseParallel(
                    output_layouts=Replicate()
                )

            parallelize_module(
                module=block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_plan,
            )

        return transformer
