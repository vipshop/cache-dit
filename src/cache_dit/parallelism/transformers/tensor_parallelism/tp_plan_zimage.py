import torch
from typing import Dict, List, Tuple
from torch.distributed import DeviceMesh
from torch.distributed.tensor.parallel import (
  ParallelStyle,
  ColwiseParallel,
  RowwiseParallel,
  parallelize_module,
)

from ....logger import init_logger
from ...config import ParallelismConfig

from .tp_plan_registers import TensorParallelismPlanner, TensorParallelismPlannerRegister
from ...utils import shard_div_attr

logger = init_logger(__name__)


@TensorParallelismPlannerRegister.register("Lumina2")
@TensorParallelismPlannerRegister.register("ZImage")
class ZImageTensorParallelismPlanner(TensorParallelismPlanner):

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
    transformer: torch.nn.Module,
    tp_mesh: DeviceMesh,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    class_name = transformer.__class__.__name__

    attn_mod_name = "attention" if class_name.startswith("ZImage") else "attn"
    ff_linear_name = "w" if class_name.startswith("ZImage") else "linear_"

    def tp_shard_block(block, tp_size):
      attn = getattr(block, attn_mod_name)
      shard_div_attr(attn, "heads", tp_size)
      layer_plan = {
        f"{attn_mod_name}.to_q": ColwiseParallel(),
        f"{attn_mod_name}.to_k": ColwiseParallel(),
        f"{attn_mod_name}.to_v": ColwiseParallel(),
        f"{attn_mod_name}.to_out.0": RowwiseParallel(),
        f"feed_forward.{ff_linear_name}1": ColwiseParallel(),
        f"feed_forward.{ff_linear_name}3": ColwiseParallel(),
        f"feed_forward.{ff_linear_name}2": RowwiseParallel(),
        # saving more memory at the cost of more communication
        # "adaLN_modulation.0": ColwiseParallel(output_layouts=Replicate()),
      }

      parallelize_module(
        module=block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_plan,
      )
      return layer_plan

    tp_size = tp_mesh.get_group().size()
    layer_plans = []
    for _, block in transformer.noise_refiner.named_children():
      layer_plans.append(tp_shard_block(block, tp_size))
    for _, block in transformer.context_refiner.named_children():
      layer_plans.append(tp_shard_block(block, tp_size))
    for _, block in transformer.layers.named_children():
      layer_plans.append(tp_shard_block(block, tp_size))

    return transformer, layer_plans
