import torch
from diffusers.models.transformers.hunyuan_transformer_2d import (
  HunyuanDiTBlock, )
from typing import Dict, List, Tuple
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor.parallel import (
  ParallelStyle,
  ColwiseParallel,
  RowwiseParallel,
  parallelize_module,
)

from ....logger import init_logger
from ...config import ParallelismConfig

from .tp_plan_registers import (
  TensorParallelismPlanner,
  TensorParallelismPlannerRegister,
)
from ...utils import shard_div_attr

logger = init_logger(__name__)


@TensorParallelismPlannerRegister.register("HunyuanDiT")
class HunyuanDiTTensorParallelismPlanner(TensorParallelismPlanner):

  def _apply(
    self,
    transformer: nn.Module,
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
    """Parallelize HunyuanDiT transformer blocks.

    HunyuanDiT uses dual attention blocks with skip connections, multiple normalization layers,
    feed-forward submodules, and long skip paths across the network.

    :param transformer: Transformer module to process.
    :param tp_mesh: Tensor-parallel device mesh.
    :returns: The parallelized transformer and its per-layer sharding plan.
    """

    layer_plans = []
    for i, block in enumerate(transformer.blocks):
      assert isinstance(block, HunyuanDiTBlock)

      # Split attention heads across TP devices
      tp_size = tp_mesh.size()
      shard_div_attr(block.attn1, "heads", tp_size)
      shard_div_attr(block.attn2, "heads", tp_size)

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
      layer_plans.append(layer_plan)
    return transformer, layer_plans
