import torch
from typing import Dict, List, Tuple

from torch.distributed import DeviceMesh
from torch.distributed._tensor import Replicate
from torch.distributed.tensor.parallel import (
  ColwiseParallel,
  ParallelStyle,
  parallelize_module,
)

from ...logger import init_logger
from ...utils import maybe_empty_cache
from ..config import ParallelismConfig
from .register import (
  TextEncoderTensorParallelismPlanner,
  TextEncoderTensorParallelismPlannerRegister,
)

logger = init_logger(__name__)


def _mlp_plan(prefix: str) -> Dict[str, ParallelStyle]:
  # Replicated down_proj/router/attention consumers require a complete local activation, so each
  # sharded gate/up output is all-gathered instead of handing Shard(-1) to a replicated module.
  return {
    f"{prefix}.gate_proj": ColwiseParallel(output_layouts=Replicate()),
    f"{prefix}.up_proj": ColwiseParallel(output_layouts=Replicate()),
  }


@TextEncoderTensorParallelismPlannerRegister.register("MingImageMLLM")
class MingImageMLLMTensorParallelismPlanner(TextEncoderTensorParallelismPlanner):
  """Memory-oriented TP for the Bailing-MoE text tower.

  QKV, dense/MoE ``gate_proj`` and ``up_proj`` weights are sharded with replicated outputs, so
  every consumer of a sharded linear still sees a complete local activation. ``dense``/
  ``down_proj``, routers and the vision tower remain replicated.
  """

  def _apply(
    self,
    text_encoder: torch.nn.Module,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    tp_mesh = self.mesh(parallelism_config=parallelism_config)
    text_encoder, layer_plans = self.parallelize_text_encoder(
      text_encoder=text_encoder,
      tp_mesh=tp_mesh,
    )
    return text_encoder, layer_plans

  def parallelize_text_encoder(
    self,
    text_encoder: torch.nn.Module,
    tp_mesh: DeviceMesh,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    trunk = text_encoder.model.model
    layer_plans = []
    for block in trunk.layers:
      layer_plan = {
        "attention.query_key_value": ColwiseParallel(output_layouts=Replicate()),
      }
      if getattr(block, "is_moe", False):
        for index in range(len(block.mlp.experts)):
          layer_plan.update(_mlp_plan(f"mlp.experts.{index}"))
        if block.mlp.shared_experts is not None:
          layer_plan.update(_mlp_plan("mlp.shared_experts"))
      else:
        layer_plan.update(_mlp_plan("mlp"))
      parallelize_module(
        module=block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_plan,
      )
      layer_plans.append(layer_plan)
    maybe_empty_cache()
    return text_encoder, layer_plans
