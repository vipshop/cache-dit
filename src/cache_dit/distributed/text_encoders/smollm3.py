"""SmolLM3 text-encoder tensor-parallel planner.

SmolLM3ForCausalLM has identical layer structure to LlamaForCausalLM
(``model.layers[i].self_attn.{q_proj,k_proj,v_proj,o_proj}`` and
``mlp.{gate_proj,up_proj,down_proj}``) but is not a runtime subclass,
so the generic Llama planner does not match.
"""

import torch
from typing import Dict, List, Tuple
from torch.distributed import DeviceMesh
from torch.distributed.tensor.parallel import (
  ColwiseParallel,
  ParallelStyle,
  RowwiseParallel,
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


@TextEncoderTensorParallelismPlannerRegister.register("SmolLM3ForCausalLM")
class SmolLM3TensorParallelismPlanner(TextEncoderTensorParallelismPlanner):
  """TP planner for SmolLM3ForCausalLM.

  Reuses the standard Llama TP plan because the module layout (self_attn.q/k/v/o_proj +
  mlp.gate/up/down_proj) is identical.
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
    model = text_encoder.model  # SmolLM3Model — same layout as LlamaModel
    layer_plans = []
    for _, block in model.layers.named_children():
      layer_plan = {
        "self_attn.q_proj": ColwiseParallel(),
        "self_attn.k_proj": ColwiseParallel(),
        "self_attn.v_proj": ColwiseParallel(),
        "self_attn.o_proj": RowwiseParallel(),
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
      }
      parallelize_module(
        module=block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_plan,
      )
      layer_plans.append(layer_plan)

    text_encoder.model = model
    maybe_empty_cache()
    return text_encoder, layer_plans
