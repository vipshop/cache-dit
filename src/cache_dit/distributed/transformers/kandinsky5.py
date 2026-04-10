from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed._tensor import Replicate
from torch.distributed.tensor.parallel import (
  ColwiseParallel,
  ParallelStyle,
  RowwiseParallel,
  parallelize_module,
)

from ...logger import init_logger
from ..config import ParallelismConfig
from ..utils import shard_div_attr
from .register import (
  TensorParallelismPlanner,
  TensorParallelismPlannerRegister,
)

logger = init_logger(__name__)


@TensorParallelismPlannerRegister.register("Kandinsky5")
class Kandinsky5TensorParallelismPlanner(TensorParallelismPlanner):

  def _apply(
    self,
    transformer: torch.nn.Module,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    tp_mesh = self.mesh(parallelism_config=parallelism_config)
    transformer, layer_plans = self.parallelize_transformer(
      transformer=transformer,
      tp_mesh=tp_mesh,
    )

    return transformer, layer_plans

  def parallelize_transformer(
    self,
    transformer: nn.Module,
    tp_mesh: DeviceMesh,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    layer_plans = []
    for _, block in transformer.visual_transformer_blocks.named_children():
      tp_size = tp_mesh.size()
      shard_div_attr(block.self_attention, "num_heads", tp_size)
      shard_div_attr(block.cross_attention, "num_heads", tp_size)
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
      layer_plans.append(layer_plan)
    return transformer, layer_plans
