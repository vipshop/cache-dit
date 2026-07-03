"""JoyImage distributed parallelism planners (CP and TP)."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
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
from .register import (
  ContextParallelismPlanner,
  ContextParallelismPlannerRegister,
  TensorParallelismPlanner,
  TensorParallelismPlannerRegister,
)

logger = init_logger(__name__)


def _patch_rope_for_cp(transformer: nn.Module, parallelism_config: ParallelismConfig) -> None:
  """Monkey-patch get_rotary_pos_embed to chunk RoPE frequencies for CP.

  RoPE freqs are computed once before the block loop, but CP splits hidden_states at
  double_blocks.0.  Without this patch, each rank receives full-sequence RoPE but partial-sequence
  Q/K, causing misalignment.
  """
  cp_size = parallelism_config.ulysses_size or parallelism_config.ring_size or 1
  if cp_size <= 1:
    return

  orig_rope = transformer.get_rotary_pos_embed

  def patched_rope(self, vis_rope_size, txt_rope_size=None):
    vis_freqs, txt_freqs = orig_rope(vis_rope_size, txt_rope_size)
    rank = dist.get_rank() % cp_size
    vis_freqs = (
      torch.tensor_split(vis_freqs[0], cp_size, dim=0)[rank],
      torch.tensor_split(vis_freqs[1], cp_size, dim=0)[rank],
    )
    if txt_freqs is not None:
      txt_freqs = (
        torch.tensor_split(txt_freqs[0], cp_size, dim=0)[rank],
        torch.tensor_split(txt_freqs[1], cp_size, dim=0)[rank],
      )
    return vis_freqs, txt_freqs

  transformer.get_rotary_pos_embed = patched_rope.__get__(transformer)


def _patch_attn_processor_for_cp(transformer: nn.Module,
                                 parallelism_config: ParallelismConfig) -> None:
  """Set _parallel_config on each block's attention processor.

  JoyImageAttnProcessor passes self._parallel_config to dispatch_attention_fn, enabling Ulysses/Ring
  attention to use the correct CP group config.
  """
  for block in transformer.double_blocks:
    block.attn.processor._parallel_config = parallelism_config


@ContextParallelismPlannerRegister.register("JoyImageEditTransformer")
class JoyImageContextParallelismPlanner(ContextParallelismPlanner):

  def _apply(
    self,
    transformer: Optional[torch.nn.Module | ModelMixin] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:
    n_blocks = len(transformer.double_blocks)

    _patch_rope_for_cp(transformer, parallelism_config)
    _patch_attn_processor_for_cp(transformer, parallelism_config)

    _cp_plan = {
      "double_blocks.0": {
        "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3),
        "encoder_hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3),
      },
      f"double_blocks.{n_blocks - 1}": (
        _ContextParallelOutput(gather_dim=1, expected_dims=3),
        _ContextParallelOutput(gather_dim=1, expected_dims=3),
      ),
    }
    return _cp_plan


@TensorParallelismPlannerRegister.register("JoyImageEditTransformer")
class JoyImageTensorParallelismPlanner(TensorParallelismPlanner):

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
    # JoyImage uses FUSED QKV projections (img_attn_qkv / txt_attn_qkv)
    # where the weight layout is [Q_all, K_all, V_all].  Standard
    # ColwiseParallel() splits this linearly, breaking Q/K/V boundaries.
    # Strategy (per skill §3.4.2 Replicate pattern):
    #   - ColwiseParallel(output_layouts=Replicate()) on fused QKV:
    #     weights sharded (memory saved), output all-gathered so the
    #     attention processor sees complete Q/K/V.  Do NOT shard_div_attr
    #     because the processor sees the full heads count.
    #   - RowwiseParallel(input_layouts=Replicate()) on output proj:
    #     the input is a full replicated tensor from the attention
    #     processor, so input_layouts=Replicate() is REQUIRED (default
    #     Shard(-1) would misinterpret the full tensor and crash).
    layer_plans = []
    for _, block in transformer.double_blocks.named_children():
      layer_plan: Dict[str, ParallelStyle] = {
        "attn.img_attn_qkv": ColwiseParallel(output_layouts=Replicate()),
        "attn.txt_attn_qkv": ColwiseParallel(output_layouts=Replicate()),
        "attn.img_attn_proj": RowwiseParallel(input_layouts=Replicate()),
        "attn.txt_attn_proj": RowwiseParallel(input_layouts=Replicate()),
        "img_mlp.net.0.proj": ColwiseParallel(),
        "img_mlp.net.2": RowwiseParallel(),
        "txt_mlp.net.0.proj": ColwiseParallel(),
        "txt_mlp.net.2": RowwiseParallel(),
      }

      parallelize_module(
        module=block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_plan,
      )

      layer_plans.append(layer_plan)

    return transformer, layer_plans
