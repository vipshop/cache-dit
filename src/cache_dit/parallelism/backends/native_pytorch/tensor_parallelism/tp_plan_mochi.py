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
        for _, block in transformer.transformer_blocks.named_children():
            block.attn1.heads //= tp_mesh.size()

            layer_plan = {
                "attn1.to_q": ColwiseParallel(),
                "attn1.to_k": ColwiseParallel(),
                "attn1.to_v": ColwiseParallel(),
                "attn1.to_out.0": RowwiseParallel(),
                "ff.net.0.proj": ColwiseParallel(),
                "ff.net.2": RowwiseParallel(),
                "norm1.linear": ColwiseParallel(output_layouts=Replicate()),
            }

            if getattr(block.attn1, "to_add_out", None) is not None:
                text_plan = {
                    "attn1.add_q_proj": ColwiseParallel(),
                    "attn1.add_k_proj": ColwiseParallel(),
                    "attn1.add_v_proj": ColwiseParallel(),
                    "attn1.to_add_out": RowwiseParallel(),
                }
                layer_plan.update(text_plan)
            if getattr(block.norm1_context, "linear", None) is not None:
                layer_plan["norm1_context.linear"] = ColwiseParallel(
                    output_layouts=Replicate()
                )
            if getattr(block.norm1_context, "linear_1", None) is not None:
                layer_plan["norm1_context.linear_1"] = ColwiseParallel(
                    output_layouts=Replicate()
                )
            if getattr(block, "ff_context", None) is not None:
                layer_plan["ff_context.net.0.proj"] = ColwiseParallel()
                layer_plan["ff_context.net.2"] = RowwiseParallel()

            parallelize_module(
                module=block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_plan,
            )

        return transformer
