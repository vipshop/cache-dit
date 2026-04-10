from typing import Dict, List, Tuple

import torch
from diffusers import GlmImageTransformer2DModel
from diffusers.models.transformers.transformer_glm_image import GlmImageTransformerBlock
from torch.distributed import DeviceMesh
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


@TensorParallelismPlannerRegister.register("GlmImageTransformer2DModel")
class GlmImageTensorParallelismPlanner(TensorParallelismPlanner):

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
    transformer: GlmImageTransformer2DModel,
    tp_mesh: DeviceMesh,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    layer_plans = []
    for _, block in transformer.transformer_blocks.named_children():
      assert isinstance(block, GlmImageTransformerBlock)
      shard_div_attr(block.attn1, "heads", tp_mesh.size())
      layer_plan = {
        "attn1.to_q": ColwiseParallel(),
        "attn1.to_k": ColwiseParallel(),
        "attn1.to_v": ColwiseParallel(),
        "attn1.to_out.0": RowwiseParallel(),
        "ff.net.0.proj": ColwiseParallel(),
        "ff.net.2": RowwiseParallel(),
      }
      parallelize_module(
        module=block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_plan,
      )
      layer_plans.append(layer_plan)
    return transformer, layer_plans
