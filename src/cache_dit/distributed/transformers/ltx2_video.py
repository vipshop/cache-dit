from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor.parallel import (
  ColwiseParallel,
  ParallelStyle,
  RowwiseParallel,
  parallelize_module,
)
from diffusers.models.transformers.transformer_ltx2 import (
  LTX2AudioVideoAttnProcessor,
  LTX2PerturbedAttnProcessor,
)

from ...logger import init_logger
from ..config import ParallelismConfig
from ..utils import shard_div_attr
from .register import TensorParallelismPlanner, TensorParallelismPlannerRegister

logger = init_logger(__name__)

# Attention modules inside `LTX2VideoTransformerBlock` that are head-sharded under TP.
_SHARDED_ATTNS: Tuple[str, ...] = (
  "attn1",
  "attn2",
  "audio_attn1",
  "audio_attn2",
  "audio_to_video_attn",
  "video_to_audio_attn",
)

# LTX-2.5 checkpoints build blocks with `perturbed_attn=True`, which swaps in the standalone
# `LTX2PerturbedAttnProcessor`; both must be handled by the TP planner.
_LTX2_PROCESSORS: Tuple[type, ...] = (LTX2AudioVideoAttnProcessor, LTX2PerturbedAttnProcessor)


def _ltx2_block_layer_plan(block: nn.Module) -> Dict[str, ParallelStyle]:
  """Build the TP layer plan for one `LTX2VideoTransformerBlock`.

  Every attention projection is column-sharded on the head axis and every output / FFN projection
  row-sharded. Per-head gates (`gated_attn=True`, used by LTX-2.3/2.5) have to follow the same head
  axis as `attn.heads`, otherwise each rank emits full-width gate logits while the processor
  unflattens the attention output with the TP-sharded head count.

  Modulation modules (`prompt_adaln`, `keyframes_abs_pos_embedding`) are deliberately left out:
  TP only shards Linear weights, the residual stream keeps its full feature width, so their
  activations must stay replicated.

  :param block: One LTX-2 transformer block.
  :returns: Mapping from parameter path to parallel style.
  """
  layer_plan: Dict[str, ParallelStyle] = {}
  for attn_name in _SHARDED_ATTNS:
    layer_plan[f"{attn_name}.to_q"] = ColwiseParallel()
    layer_plan[f"{attn_name}.to_k"] = ColwiseParallel()
    layer_plan[f"{attn_name}.to_v"] = ColwiseParallel()
    layer_plan[f"{attn_name}.to_out.0"] = RowwiseParallel()
    if getattr(getattr(block, attn_name), "to_gate_logits", None) is not None:
      layer_plan[f"{attn_name}.to_gate_logits"] = ColwiseParallel()

  layer_plan["ff.net.0.proj"] = ColwiseParallel()
  layer_plan["ff.net.2"] = RowwiseParallel()
  layer_plan["audio_ff.net.0.proj"] = ColwiseParallel()
  layer_plan["audio_ff.net.2"] = RowwiseParallel()
  return layer_plan


class DistributedRMSNorm(nn.Module):

  def __init__(
    self,
    tp_mesh: DeviceMesh,
    normalized_shape: Union[int, list[int], torch.Size],
    eps: Optional[float],
    elementwise_affine: bool,
    weight: torch.nn.parameter.Parameter,
  ):
    super().__init__()
    self.tp_mesh = tp_mesh
    self.elementwise_affine = elementwise_affine
    self.normalized_shape = normalized_shape
    self.eps = eps
    if self.elementwise_affine:
      assert weight is not None
    self.weight = weight

  @classmethod
  def from_rmsnorm(cls, tp_mesh: DeviceMesh, rmsnorm: nn.RMSNorm):
    assert len(rmsnorm.normalized_shape) == 1
    if rmsnorm.weight is not None:
      tp_size = tp_mesh.get_group().size()
      tp_rank = tp_mesh.get_group().rank()
      weight = rmsnorm.weight.chunk(tp_size, dim=0)[tp_rank]
    else:
      weight = None
    norm = cls(
      tp_mesh=tp_mesh,
      normalized_shape=rmsnorm.normalized_shape,
      eps=rmsnorm.eps,
      elementwise_affine=rmsnorm.elementwise_affine,
      weight=weight,
    )
    return norm

  def forward(self, x):
    if self.elementwise_affine:
      assert x.shape[-1] == self.weight.shape[0]
    mean_square = torch.mean(x * x, dim=-1, keepdim=True)
    torch.distributed.all_reduce(
      mean_square,
      op=torch.distributed.ReduceOp.AVG,
      group=self.tp_mesh.get_group(),
    )
    root_mean_square = torch.sqrt(mean_square + self.eps)
    x_normed = x / root_mean_square
    if self.elementwise_affine:
      x_normed = x_normed * self.weight.to(device=x.device)
    assert x_normed.device.type != "cpu"
    return x_normed


class ShardRotaryEmbProcessor:
  """Shard query/key rotary embeddings to match TP-sharded heads/channels.

  - interleaved RoPE: `cos`/`sin` are shaped `(B, T, D)` and shard along the last dimension.
  - split RoPE: `cos`/`sin` are shaped `(B, H, T, D/2)` and shard along the head dimension.

  Both LTX-2 attention processors are supported. LTX-2.5 checkpoints set `perturbed_attn=True`,
  whose `LTX2PerturbedAttnProcessor` is a standalone class (not a subclass of
  `LTX2AudioVideoAttnProcessor`); its `perturbation_mask` / `all_perturbed` arguments are forwarded
  unchanged, because `LTX2Attention.forward` filters kwargs by
  `inspect.signature(processor.__call__)` and would silently drop them otherwise.
  """

  def __init__(self, processor: Union[LTX2AudioVideoAttnProcessor, LTX2PerturbedAttnProcessor],
               tp_size: int, tp_rank: int):
    self.processor = processor
    self.tp_size = tp_size
    self.tp_rank = tp_rank

  @classmethod
  def from_attn_processor(cls, processor: Union[LTX2AudioVideoAttnProcessor,
                                                LTX2PerturbedAttnProcessor], tp_size: int,
                          tp_rank: int) -> "ShardRotaryEmbProcessor":
    return cls(processor=processor, tp_size=tp_size, tp_rank=tp_rank)

  def _shard_rope(self, emb):
    if emb is None:
      return None
    cos, sin = emb
    if cos is None or sin is None:
      return emb
    # split rope: (B, H, T, D/2)
    if cos.ndim == 4:
      cos = torch.chunk(cos, self.tp_size, dim=1)[self.tp_rank]
      sin = torch.chunk(sin, self.tp_size, dim=1)[self.tp_rank]
    else:
      # interleaved rope: (B, T, D)
      cos = torch.chunk(cos, self.tp_size, dim=-1)[self.tp_rank]
      sin = torch.chunk(sin, self.tp_size, dim=-1)[self.tp_rank]
    return (cos, sin)

  def __call__(
    self,
    attn,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    query_rotary_emb=None,
    key_rotary_emb=None,
    perturbation_mask: Optional[torch.Tensor] = None,
    all_perturbed: Optional[bool] = None,
  ) -> torch.Tensor:
    query_rotary_emb = self._shard_rope(query_rotary_emb)
    key_rotary_emb = self._shard_rope(key_rotary_emb)
    kwargs = dict(
      encoder_hidden_states=encoder_hidden_states,
      attention_mask=attention_mask,
      query_rotary_emb=query_rotary_emb,
      key_rotary_emb=key_rotary_emb,
    )
    if isinstance(self.processor, LTX2PerturbedAttnProcessor):
      kwargs["perturbation_mask"] = perturbation_mask
      kwargs["all_perturbed"] = all_perturbed
    return self.processor(attn, hidden_states, **kwargs)


@TensorParallelismPlannerRegister.register("LTX2")
class LTX2VideoTensorParallelismPlanner(TensorParallelismPlanner):

  def _apply(
    self,
    transformer: torch.nn.Module,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    tp_mesh = self.mesh(parallelism_config=parallelism_config)
    transformer, layer_plans = self.parallelize_transformer(transformer=transformer,
                                                            tp_mesh=tp_mesh)
    return transformer, layer_plans

  def parallelize_transformer(
    self,
    transformer: nn.Module,
    tp_mesh: DeviceMesh,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    tp_size = tp_mesh.get_group().size()
    tp_rank = tp_mesh.get_group().rank()

    def _shard_attention(attn: nn.Module, what: str):
      shard_div_attr(attn, "heads", tp_size, what=what)

    def prepare_block(block: nn.Module):
      # Shard heads for all attention modules inside the block
      for attn_name in _SHARDED_ATTNS:
        _shard_attention(getattr(block, attn_name), attn_name)

      layer_plan = _ltx2_block_layer_plan(block)

      parallelize_module(module=block, device_mesh=tp_mesh, parallelize_plan=layer_plan)

      # Shard qk norms
      for attn_name in _SHARDED_ATTNS:
        attn = getattr(block, attn_name)
        attn.norm_q = DistributedRMSNorm.from_rmsnorm(tp_mesh, attn.norm_q)
        attn.norm_k = DistributedRMSNorm.from_rmsnorm(tp_mesh, attn.norm_k)
      return layer_plan

    # Shard RoPE frequencies for all attention processors in every block.
    # NOTE: This assumes rotary embedding head counts align with attention heads (true for LTX-2 configs).
    layer_plans = []
    for _, block in transformer.transformer_blocks.named_children():
      for attn_name in _SHARDED_ATTNS:
        attn = getattr(block, attn_name)
        if hasattr(attn, "processor") and isinstance(attn.processor, _LTX2_PROCESSORS):
          attn.processor = ShardRotaryEmbProcessor.from_attn_processor(
            processor=attn.processor,
            tp_size=tp_size,
            tp_rank=tp_rank,
          )
      layer_plans.append(prepare_block(block))

    return transformer, layer_plans
