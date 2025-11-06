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


@TensorParallelismPlannerRegister.register("Kandinsky5")
class Kandinsky5TensorParallelismPlanner(TensorParallelismPlanner):
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
        for _, block in transformer.visual_transformer_blocks.named_children():
            block.attn.heads //= tp_mesh.size()
            layer_plan = {
                "self_attention.to_query": ColwiseParallel(),
                "self_attention.to_key": ColwiseParallel(),
                "self_attention.to_value": ColwiseParallel(),
                "self_attention.out_layer": RowwiseParallel(),
                "cross_attention.to_query": ColwiseParallel(),
                "cross_attention.to_key": ColwiseParallel(),
                "cross_attention.to_value": ColwiseParallel(),
                "cross_attention.out_layer": RowwiseParallel(),
                "visual_modulation.out_layer": ColwiseParallel(
                    output_layouts=Replicate()
                ),
                "feed_forward.in_layer": ColwiseParallel(),
                "feed_forward.out_layer": RowwiseParallel(),
            }
            parallelize_module(
                module=block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_plan,
            )
        return transformer
