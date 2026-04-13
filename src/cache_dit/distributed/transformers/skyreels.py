from typing import Dict, List, Optional, Tuple

import torch
from diffusers.models.modeling_utils import ModelMixin
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed._tensor import Replicate
from torch.distributed.tensor.parallel import (
  ColwiseParallel,
  ParallelStyle,
  RowwiseParallel,
  parallelize_module,
)

from ...distributed.core import (
  _ContextParallelInput,
  _ContextParallelModelPlan,
  _ContextParallelOutput,
)
from ...logger import init_logger
from ..config import ParallelismConfig
from ..utils import shard_div_attr
from .register import (
  ContextParallelismPlanner,
  ContextParallelismPlannerRegister,
  TensorParallelismPlanner,
  TensorParallelismPlannerRegister,
)

logger = init_logger(__name__)


@ContextParallelismPlannerRegister.register("SkyReelsV2")
class SkyReelsV2ContextParallelismPlanner(ContextParallelismPlanner):

  def _apply(
    self,
    transformer: Optional[torch.nn.Module | ModelMixin] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:

    # NOTE: Diffusers native CP plan still not supported
    # for SkyReelsV2 now, use custom implementation.
    self._cp_planner_preferred_native_diffusers = False

    # SkyReelsV2 uses WanModel architecture (config: "_class_name": "WanModel")
    # Based on BlockAdapter, it uses Pattern_3 where encoder_hidden_states
    # will NEVER change in the blocks forward loop.
    # This is different from regular Wan which uses Pattern_2.
    _cp_plan = {
      # Pattern of rope, split_output=True (split output rather than input):
      #    un-split input
      #    -> keep input un-split
      #    -> rope
      #    -> splited output
      # SkyReelsV2 (like Wan) has rotary position embeddings
      "rope": {
        0: _ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
        1: _ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
      },
      # Pattern of blocks.0, split_output=False:
      #     un-split input -> split -> to_qkv/...
      #     -> all2all
      #     -> attn (local head, full seqlen)
      #     -> all2all
      #     -> splited output
      #     (only split hidden_states, not encoder_hidden_states)
      "blocks.0": {
        "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
      },
      # Pattern of blocks.*, split_output=False:
      #     splited input (previous splited output) -> to_qkv/...
      #     -> all2all
      #     -> attn (local head, full seqlen)
      #     -> all2all
      #     -> splited output
      # Since encoder_hidden_states never changes (Pattern_3), we need to
      # split it at ALL blocks by the inserted split hook.
      # hidden_states has been automatically split in previous block.
      "blocks.*": {
        "encoder_hidden_states":
        _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
      },
      # The final proj_out will gather the splited output.
      #     splited input (previous splited output)
      #     -> all gather
      #     -> un-split output
      "proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
    }
    return _cp_plan


@TensorParallelismPlannerRegister.register("SkyReelsV2")
class SkyReelsV2TensorParallelismPlanner(TensorParallelismPlanner):

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

    tp_size = tp_mesh.get_group().size()
    layer_plans = []

    # SkyReelsV2 uses a similar architecture to video transformers
    # We parallelize the attention and feedforward layers across blocks
    for name, block in transformer.blocks.named_children():
      # Reduce the number of attention heads per device
      if hasattr(block, "attn"):
        if hasattr(block.attn, "heads"):
          shard_div_attr(block.attn, "heads", tp_size)

      # Define parallelization plan for each block
      # This follows a standard pattern:
      # - Q, K, V projections: column-wise parallel
      # - Attention output: row-wise parallel
      # - FFN first layer: column-wise parallel
      # - FFN second layer: row-wise parallel
      layer_plan = {
        "attn.to_q": ColwiseParallel(),
        "attn.to_k": ColwiseParallel(),
        "attn.to_v": ColwiseParallel(),
        "attn.to_out.0": RowwiseParallel(),
      }

      # Add FFN parallelization if the block has a feedforward network
      if hasattr(block, "ff") or hasattr(block, "mlp"):
        ff_module = "ff" if hasattr(block, "ff") else "mlp"
        # Typical FFN structure: Linear -> Activation -> Linear
        layer_plan[f"{ff_module}.net.0"] = ColwiseParallel()
        layer_plan[f"{ff_module}.net.2"] = RowwiseParallel()

      # Add normalization layer parallelization with replicated output
      if hasattr(block, "norm1"):
        if hasattr(block.norm1, "linear"):
          layer_plan["norm1.linear"] = ColwiseParallel(output_layouts=Replicate())
      if hasattr(block, "norm2"):
        if hasattr(block.norm2, "linear"):
          layer_plan["norm2.linear"] = ColwiseParallel(output_layouts=Replicate())

      try:
        parallelize_module(
          module=block,
          device_mesh=tp_mesh,
          parallelize_plan=layer_plan,
        )
        layer_plans.append(layer_plan)
        logger.debug(f"Successfully parallelized block: {name}")
      except Exception as e:
        logger.debug(f"Could not parallelize block {name}: {e}")
        logger.debug("Block structure may differ from expected pattern. Skipping this block.")
    return transformer, layer_plans
