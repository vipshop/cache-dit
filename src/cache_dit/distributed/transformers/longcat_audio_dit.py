"""LongCat-AudioDiT transformer distributed parallelism planners (CP + TP).

`LongCatAudioDiTTransformer` denoises a single 64-channel waveform-latent stream through
`transformer.blocks`, with a length of 58 (5 s) to 351 (30 s) frames. Everything a block consumes
except `hidden_states` — the text condition, the self/cross rotary tables, the per-step AdaLN
modulation and the duration mask — is computed once before the block loop, which is why the block
stack is cacheable and why the planners only have to deal with the latent sequence itself.

* CP shards the latent sequence at the transformer entry: `hidden_states` and `latent_cond` are
  concatenated along the feature axis inside `forward`, so both are split with the same boundaries,
  and `proj_out` gathers the shard back before the scheduler step. The rotary table is rebuilt from
  the (already sharded) length, so it would restart at position 0 on every rank: the planner swaps
  in a full-table wrapper and lets the attention processors slice their own position window.
* TP shards the self/cross attention projections and the FFN of every block. The attention
  processors reshape with `heads` and `inner_dim`, so both attributes are divided as well, and the
  QK RMSNorm moves to a distributed variant because the norm spans the pre-shard feature axis.
"""

import inspect
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
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


class _FullRotaryTable(nn.Module):
  """Rotary embedding wrapper that always returns the whole position table.

  The transformer derives the table length from the latent length it currently sees. Under CP that
  length is the local shard, so an unpatched table would place every rank at positions `0..S-1`.
  Returning the full table (the original embedding already caches 2048 positions) lets each rank
  slice `[offset:offset + seq_len]` for the latent stream and `[:text_len]` for the replicated text
  stream.

  :param rotary: Original `AudioDiTRotaryEmbedding` instance.
  """

  def __init__(self, rotary: nn.Module):
    super().__init__()
    self.rotary = rotary
    self.table_len = int(getattr(rotary, "max_position_embeddings", 2048))

  def forward(
    self,
    hidden_states: torch.Tensor,
    seq_len: Optional[int] = None,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the full `(cos, sin)` table, ignoring the local sequence length.

    :param hidden_states: Tensor the table is built for, only used for device and dtype.
    :param seq_len: Local sequence length, deliberately ignored.
    :returns: Full `(cos, sin)` tables of shape `(table_len, dim_head)`.
    """

    return self.rotary(hidden_states, self.table_len)


_OFFSET_CACHE: Dict[Tuple[int, Tuple[int, ...]], int] = {}


def _cp_rank_offset(cp_config, seq_len: int) -> int:
  """Return the global position offset of this rank's latent shard.

  :param cp_config: Cache-dit context parallel config attached to the transformer.
  :param seq_len: Local (already sharded) latent length.
  :returns: Offset of the first local frame inside the global sequence.
  """

  degree = int(cp_config.ulysses_degree or 1)
  rank = int(cp_config._ulysses_local_rank or 0)
  mesh = cp_config._ulysses_mesh
  uneven = bool(cp_config.ulysses_anything)
  if degree <= 1:
    degree = int(cp_config.ring_degree or 1)
    rank = int(cp_config._ring_local_rank or 0)
    mesh = cp_config._ring_mesh
    uneven = False
  if degree <= 1:
    return 0
  if not uneven:
    # Equipartition: every rank holds `seq_len` frames.
    return rank * seq_len
  # Ulysses-anything splits with `torch.tensor_split`, so the offset is the prefix sum of the
  # per-rank sizes. A local length alone does not identify the split (17 frames and 16 frames both
  # give rank 1 a length of 8), so the cache key is the whole size vector.
  sizes = [0] * degree
  dist.all_gather_object(sizes, int(seq_len), group=mesh.get_group())
  key = (degree, tuple(sizes))
  offset = _OFFSET_CACHE.get(key)
  if offset is None:
    offset = sum(sizes[:rank])
    _OFFSET_CACHE[key] = offset
  return offset


class _CPRotarySliceProcessor:
  """Attention processor wrapper that slices the full rotary table per rank.

  The wrapped processor keeps owning the attention call (and therefore the communication that the
  cache-dit CP config triggers through `dispatch_attention_fn`); this wrapper only replaces the
  full-length tables with the exact position windows before delegating.

  :param processor: Original `AudioDiTSelfAttnProcessor` / `AudioDiTCrossAttnProcessor`.
  """

  def __init__(self, processor: nn.Module):
    self._processor = processor
    # Set by cache-dit when CP is enabled.
    self._cp_config = None
    self._parallel_config = None

  @property
  def _attention_backend(self):
    """Forward the backend name to the wrapped processor so the two never diverge."""
    return getattr(self._processor, "_attention_backend", None)

  @_attention_backend.setter
  def _attention_backend(self, backend):
    self._processor._attention_backend = backend

  def __call__(
    self,
    attn: nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    post_attention_mask: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    audio_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    prompt_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
  ) -> torch.Tensor:
    """Slice both rotary tables and run the original processor.

    :param attn: Owning `AudioDiTAttention` module.
    :param hidden_states: Local latent shard.
    :param encoder_hidden_states: Replicated text condition for the cross attention.
    :param post_attention_mask: Post-attention mask of the cross attention.
    :param attention_mask: Optional self-attention mask; all-True masks are dropped by the planner.
    :param audio_rotary_emb: Full latent rotary table.
    :param prompt_rotary_emb: Full text rotary table.
    :returns: The processor output.
    """

    cp_config = self._cp_config
    # Self attention owns the sharded latent sequence, so it needs the collective attention. Cross
    # attention queries the local shard against the *replicated* text: an all-to-all there would
    # concatenate the same text on every rank and break the attention, so it runs locally.
    is_cross_attention = encoder_hidden_states is not None
    self._processor._cp_config = None if is_cross_attention else cp_config
    self._processor._parallel_config = None if is_cross_attention else self._parallel_config
    if audio_rotary_emb is not None:
      seq_len = hidden_states.shape[1]
      # Without CP the table already starts at position 0; with CP this rank owns the window
      # `[offset, offset + seq_len)` of the global sequence.
      offset = _cp_rank_offset(cp_config, seq_len) if cp_config is not None else 0
      audio_rotary_emb = (
        audio_rotary_emb[0][offset:offset + seq_len],
        audio_rotary_emb[1][offset:offset + seq_len],
      )
    if prompt_rotary_emb is not None and encoder_hidden_states is not None:
      # The text stream is replicated, so its positions always start at 0.
      text_len = encoder_hidden_states.shape[1]
      prompt_rotary_emb = (prompt_rotary_emb[0][:text_len], prompt_rotary_emb[1][:text_len])
    if encoder_hidden_states is None:
      return self._processor(
        attn,
        hidden_states,
        attention_mask=attention_mask,
        audio_rotary_emb=audio_rotary_emb,
      )
    return self._processor(
      attn,
      hidden_states,
      encoder_hidden_states=encoder_hidden_states,
      post_attention_mask=post_attention_mask,
      attention_mask=attention_mask,
      audio_rotary_emb=audio_rotary_emb,
      prompt_rotary_emb=prompt_rotary_emb,
    )


def _install_cp_rope(transformer: nn.Module) -> None:
  """Swap in the full-table rotary embedding and the rotary-slicing processors.

  :param transformer: `LongCatAudioDiTTransformer` instance.
  """

  if not isinstance(transformer.rotary_embed, _FullRotaryTable):
    transformer.rotary_embed = _FullRotaryTable(transformer.rotary_embed)
    logger.info("LongCat-AudioDiT CP: rotary embedding returns the full position table, the "
                "attention processors slice each rank's window.")
  for block in transformer.blocks:
    for attn in (block.self_attn, getattr(block, "cross_attn", None)):
      if attn is None or getattr(attn, "processor", None) is None:
        continue
      if not isinstance(attn.processor, _CPRotarySliceProcessor):
        attn.processor = _CPRotarySliceProcessor(attn.processor)


def _wrap_maskless_forward(transformer: nn.Module) -> None:
  """Drop the duration mask before `forward` runs under CP.

  `attention_mask` is a `(batch, duration)` boolean tensor derived from the requested duration, so
  it is either all-True or padded, and a position-indexed mask cannot be reused once the sequence
  is sharded (the self attention sees the full sequence after the all-to-all while `input_embed`
  and the trailing multiply see the local shard). An all-True mask is a no-op, so it is replaced by
  `None`; a padded mask is rejected instead of silently producing a wrong result.

  :param transformer: `LongCatAudioDiTTransformer` instance.
  """

  if getattr(transformer, "_cache_dit_maskless_forward", False):
    return
  original_forward = transformer.forward
  signature = inspect.signature(original_forward)

  def forward(self, *args, **kwargs):
    bound = signature.bind(*args, **kwargs)
    mask = bound.arguments.get("attention_mask")
    if mask is not None:
      if not getattr(transformer, "_cache_dit_mask_verified", False):
        if not bool(mask.all()):
          raise NotImplementedError(
            "Context parallelism for LongCat-AudioDiT requires an all-True `attention_mask` (the "
            "pipeline derives it from the requested duration and never pads), but the given mask "
            "carries masked positions. Pass `attention_mask=None` instead.")
        # The mask is derived from the duration and never changes between steps, so one host sync
        # per run is enough.
        transformer._cache_dit_mask_verified = True
      bound.arguments["attention_mask"] = None
    return original_forward(*bound.args, **bound.kwargs)

  # Cache-dit inspects the signature when it wraps `forward` for the cached blocks, and
  # `disable_cache` rebinds the stored callable with `__get__`, which needs a real method.
  forward.__signature__ = signature
  transformer.forward = forward.__get__(transformer, type(transformer))
  transformer._cache_dit_maskless_forward = True


@ContextParallelismPlannerRegister.register("LongCatAudioDiTTransformer")
class LongCatAudioDiTContextParallelismPlanner(ContextParallelismPlanner):

  def _apply(
    self,
    transformer: Optional[torch.nn.Module] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:
    """Build the LongCat-AudioDiT context parallel plan.

    :param transformer: `LongCatAudioDiTTransformer` instance.
    :param parallelism_config: Requested parallelism configuration.
    :returns: The context parallel plan.
    """

    if transformer is None or parallelism_config is None:
      return {}

    ring_size = int(parallelism_config.ring_size or 1)
    ulysses_size = int(parallelism_config.ulysses_size or 1)
    if ring_size > 1 and ulysses_size > 1:
      # The latent shard is a single flat chunk of the flattened mesh, so the position window of a
      # rank is `flat_rank * local_len`; the per-mesh offsets used below would repeat positions
      # across the ring dimension.
      raise NotImplementedError(
        f"LongCat-AudioDiT context parallelism does not support a mixed ring x ulysses mesh "
        f"(got ring_size={ring_size}, ulysses_size={ulysses_size}). Use either Ulysses or Ring.")

    _install_cp_rope(transformer)
    _wrap_maskless_forward(transformer)

    # `forward` returns `LongCatAudioDiTTransformerOutput`, so the sequence is sharded on the two
    # embedding outputs instead of the transformer entry: `hidden_states` and `latent_cond` are
    # concatenated along the feature axis afterwards, which requires identical shard boundaries.
    # Every downstream tensor (residual, rope, block outputs) is then already local, and `proj_out`
    # rebuilds the full sequence before the scheduler step.
    return {
      "input_embed": {
        0: _ContextParallelInput(split_dim=1, expected_dims=3, split_output=True)
      },
      "latent_embed": {
        0: _ContextParallelInput(split_dim=1, expected_dims=3, split_output=True)
      },
      "proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
    }


class _DistributedRMSNorm(nn.Module):
  """TP-aware RMSNorm: the global mean square is all-reduced before normalizing the local shard.

  Diffusers' `RMSNorm` normalizes over `dim` (`normalized_shape`) and `AudioDiTAttention` applies it
  to the full `inner_dim` projection, so after a column-wise shard each rank only sees part of the
  vector. Averaging the partial mean squares (all shards have the same size) restores the exact
  global statistic; the weight itself is sharded along the same axis.

  :param tp_mesh: Tensor parallel mesh.
  :param dim: Normalized shape of the original norm.
  :param eps: Epsilon of the original norm.
  :param weight: Sharded weight (a slice of the original parameter).
  """

  def __init__(
    self,
    tp_mesh: DeviceMesh,
    dim: torch.Size,
    eps: float,
    weight: Optional[nn.Parameter],
  ):
    super().__init__()
    self.tp_mesh = tp_mesh
    self.dim = dim
    self.eps = eps
    self.weight = weight

  @classmethod
  def from_rmsnorm(cls, tp_mesh: DeviceMesh, rmsnorm: nn.Module) -> "_DistributedRMSNorm":
    """Build the distributed norm from a diffusers `RMSNorm`.

    :param tp_mesh: Tensor parallel mesh.
    :param rmsnorm: Original `RMSNorm` module.
    :returns: The TP-aware replacement.
    """

    tp_size = tp_mesh.get_group().size()
    tp_rank = tp_mesh.get_group().rank()
    assert getattr(rmsnorm, "bias", None) is None, (
      "The distributed LongCat-AudioDiT RMSNorm only implements the bias-free diffusers RMSNorm.")
    weight = rmsnorm.weight
    if weight is not None:
      weight = nn.Parameter(weight.detach().chunk(tp_size, dim=0)[tp_rank].clone())
    return cls(tp_mesh=tp_mesh, dim=rmsnorm.dim, eps=rmsnorm.eps, weight=weight)

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Normalize with the globally reduced mean square.

    :param hidden_states: Sharded activation.
    :returns: Normalized activation of the same shape.
    """

    input_dtype = hidden_states.dtype
    if self.weight is not None:
      assert hidden_states.shape[-1] == self.weight.shape[0], (
        f"Expected the local feature width to match the sharded weight "
        f"({hidden_states.shape[-1]} vs {self.weight.shape[0]}).")
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    dist.all_reduce(variance, op=dist.ReduceOp.AVG, group=self.tp_mesh.get_group())
    hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
    if self.weight is not None:
      hidden_states = self.weight.to(hidden_states.dtype) * hidden_states
    return hidden_states.to(input_dtype)


def _shard_attention(attn: nn.Module, tp_mesh: DeviceMesh) -> None:
  """Divide the attributes the attention processors reshape with and the QK norms.

  :param attn: `AudioDiTAttention` module of one block.
  :param tp_mesh: Tensor parallel mesh.
  """

  tp_size = tp_mesh.get_group().size()
  # The processors recompute `head_dim = inner_dim // heads` and view with both, so both have to
  # follow the shard; dividing only `heads` would double the effective head dimension.
  shard_div_attr(attn, "heads", tp_size)
  shard_div_attr(attn, "inner_dim", tp_size)
  if getattr(attn, "qk_norm", False):
    attn.q_norm = _DistributedRMSNorm.from_rmsnorm(tp_mesh, attn.q_norm)
    attn.k_norm = _DistributedRMSNorm.from_rmsnorm(tp_mesh, attn.k_norm)


@TensorParallelismPlannerRegister.register("LongCatAudioDiTTransformer")
class LongCatAudioDiTTensorParallelismPlanner(TensorParallelismPlanner):

  def _apply(
    self,
    transformer: torch.nn.Module,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    """Build the LongCat-AudioDiT tensor parallel plan.

    :param transformer: `LongCatAudioDiTTransformer` instance.
    :param parallelism_config: Requested parallelism configuration.
    :returns: The parallelized transformer and the applied layer plans.
    """

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
    """Shard every block, leaving the step-invariant embedding and modulation tables replicated.

    :param transformer: `LongCatAudioDiTTransformer` instance.
    :param tp_mesh: Tensor parallel mesh.
    :returns: The parallelized transformer and the applied layer plans.
    """

    layer_plans = []
    for block in transformer.blocks:
      _shard_attention(block.self_attn, tp_mesh)
      if getattr(block, "cross_attn", None) is not None:
        _shard_attention(block.cross_attn, tp_mesh)
      layer_plan = {
        "self_attn.to_q": ColwiseParallel(),
        "self_attn.to_k": ColwiseParallel(),
        "self_attn.to_v": ColwiseParallel(),
        "self_attn.to_out.0": RowwiseParallel(),
        "ffn.ff.0": ColwiseParallel(),
        "ffn.ff.3": RowwiseParallel(),
      }
      if getattr(block, "cross_attn", None) is not None:
        layer_plan.update({
          "cross_attn.to_q": ColwiseParallel(),
          "cross_attn.to_k": ColwiseParallel(),
          "cross_attn.to_v": ColwiseParallel(),
          "cross_attn.to_out.0": RowwiseParallel(),
        })
      parallelize_module(
        module=block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_plan,
      )
      layer_plans.append(layer_plan)

    # `norm_out.linear` emits the scale/shift pair that is chunked right afterwards, so its output
    # has to stay replicated; `proj_out` and the modulation tables are tiny and stay replicated too.
    parallelize_module(
      module=transformer.norm_out,
      device_mesh=tp_mesh,
      parallelize_plan={
        "linear": ColwiseParallel(output_layouts=Replicate()),
      },
    )
    return transformer, layer_plans
