import torch
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed._tensor import Replicate
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


@TensorParallelismPlannerRegister.register("Kandinsky5")
class Kandinsky5TensorParallelismPlanner(TensorParallelismPlanner):
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
        transformer: nn.Module,
        tp_mesh: DeviceMesh,
    ):
        for _, block in transformer.visual_transformer_blocks.named_children():
            tp_size = tp_mesh.size()
            shard_divisible_attr(
                block.self_attention,
                "num_heads",
                tp_size,
                what="self_attention",
                context="Kandinsky5TensorParallelismPlanner",
            )
            shard_divisible_attr(
                block.cross_attention,
                "num_heads",
                tp_size,
                what="cross_attention",
                context="Kandinsky5TensorParallelismPlanner",
            )
            layer_plan = {
                "self_attention.to_query": ColwiseParallel(),
                "self_attention.to_key": ColwiseParallel(),
                "self_attention.to_value": ColwiseParallel(),
                "self_attention.out_layer": RowwiseParallel(),
                "cross_attention.to_query": ColwiseParallel(),
                "cross_attention.to_key": ColwiseParallel(),
                "cross_attention.to_value": ColwiseParallel(),
                "cross_attention.out_layer": RowwiseParallel(),
                "visual_modulation.out_layer": ColwiseParallel(output_layouts=Replicate()),
                "feed_forward.in_layer": ColwiseParallel(),
                "feed_forward.out_layer": RowwiseParallel(),
            }
            parallelize_module(
                module=block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_plan,
            )
        return transformer
