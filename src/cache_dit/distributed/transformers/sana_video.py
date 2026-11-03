"""SANA-Video (2B / 480p) distributed parallelism planners (CP + TP).

The transformer is diffusers' `SanaVideoTransformer3DModel`, so the planners rely on diffusers
attribute names (`transformer_blocks`, `attn1`, `attn2`, `ff`, `rope`).

Context parallelism is unusual for this model in two ways:

* The self-attention is **linear attention** (`SanaLinearAttnProcessor3_0`): the output is
  `(Σ_j v_j relu(k_rot_j)^T) @ relu(q_rot)` scaled by `1 / (Σ_j relu(k_j) · q)`. Both aggregates are
  plain sums over the token axis, so a rank holding a token shard only needs one all-reduce of
  the two local aggregates per layer to obtain the global ones (the denominator all-reduce stays in
  the input dtype, the numerator one is fp32, matching upstream) — no Ulysses head all-to-all, so
  `--ulysses` and `--ring` behave identically here. The cross-attention keeps its query local and
  its K/V (the complete prompt) replicated, so it runs locally with `cp_config=None`.
* The feed-forward is convolution-based (`GLUMBTempConv`), including a **3-tap temporal
  convolution**, and the block reshapes with `frames/height/width` integers that `forward()` derives
  from the (full) latent grid. A token-axis split would therefore desynchronise the reshape, so the
  plan shards the **frame axis** of the 5D latent grid: every downstream quantity
  (`post_patch_num_frames`, patch embedding, unpatchify) is then naturally local. Two consequences
  are handled explicitly:
  - `WanRotaryPosEmbed` builds its table from the tensor's own shape, so it is patched to build the
    **global** frame table and slice this rank's window (`cp.md` §2.8 RoPE drift).
  - `conv_temp` would lose one frame of context at each chunk boundary, so it is patched to
    exchange one halo frame with the neighbouring ranks; the two global ends keep upstream's zero
    padding, which makes the result numerically identical to the full-sequence convolution.

The frame split requires the **latent frame count to be divisible by the CP size** (the Wan VAE
compresses time by 4, so `frames = 4k + 1` becomes `k + 1` latent frames), unless
`--ulysses-anything` switches the sharder to an uneven split. An indivisible count is rejected up
front with the suggested counts instead of failing inside the sharder.
"""

import functools
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor.parallel import (
  ColwiseParallel,
  ParallelStyle,
  RowwiseParallel,
  parallelize_module,
)

from ...attention import _dispatch_attention_fn
from ...distributed.core import (
  _ContextParallelInput,
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


class _DistributedRMSNorm(nn.Module):
  """TP-aware equivalent of diffusers' `RMSNorm` for the SANA-Video Q/K norms.

  `qk_norm="rms_norm_across_heads"` normalises the **whole** `inner_dim`, so a per-head weight shard
  would normalise a partial vector. The variance is therefore all-reduced (mean) across the TP group
  before the scaling, which reproduces the full-width norm while the affine weight stays sharded.

  `wan.py::DistributedRMSNorm` cannot be reused: it is written against `torch.nn.RMSNorm`
  (`normalized_shape`), while SANA-Video's norms are `diffusers.models.normalization.RMSNorm`
  (`dim`, float32 variance, optional bias).

  :param num_features: Full (unsharded) width of the normalised axis.
  :param eps: Numerical epsilon of the upstream RMSNorm.
  :param tp_mesh: The TP device mesh.
  :param weight: Sharded affine weight, or `None` when the norm is not affine.
  :param bias: Sharded affine bias, or `None`.
  """

  def __init__(self, num_features: int, eps: float, tp_mesh: DeviceMesh,
               weight: Optional[nn.Parameter], bias: Optional[nn.Parameter]) -> None:
    super().__init__()
    self.num_features = num_features
    self.eps = eps
    self.tp_mesh = tp_mesh
    self.weight = weight
    self.bias = bias

  @classmethod
  def from_rmsnorm(cls, tp_mesh: DeviceMesh, norm: nn.Module) -> "_DistributedRMSNorm":
    """Shard a diffusers `RMSNorm` over the TP group.

    :param tp_mesh: The TP device mesh.
    :param norm: The upstream norm module.
    :returns: The TP-aware replacement.
    """

    tp_size = tp_mesh.size()
    tp_rank = tp_mesh.get_group().rank()

    def shard(param):
      if param is None:
        return None
      return nn.Parameter(param.detach().chunk(tp_size, dim=-1)[tp_rank].clone())

    return cls(
      num_features=int(norm.dim[0]),
      eps=float(norm.eps),
      tp_mesh=tp_mesh,
      weight=shard(norm.weight),
      bias=shard(getattr(norm, "bias", None)),
    )

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Normalise the local feature shard with the global mean-square.

    :param hidden_states: Local tensor whose last axis holds this rank's feature shard.
    :returns: The normalised tensor, same shape and dtype.
    """

    input_dtype = hidden_states.dtype
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    dist.all_reduce(variance, op=dist.ReduceOp.AVG, group=self.tp_mesh.get_group())
    hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
    if self.weight is None:
      return hidden_states.to(input_dtype)
    hidden_states = hidden_states.to(self.weight.dtype) * self.weight
    if self.bias is not None:
      hidden_states = hidden_states + self.bias
    return hidden_states


def _cp_is_armed(owner) -> bool:
  """Whether `owner` was armed by the CP runtime with a multi-rank config.

  :param owner: Transformer or attention processor carrying `_cp_config`.
  :returns: True when context parallelism is active on more than one rank.
  """

  cp_config = getattr(owner, "_cp_config", None)
  if cp_config is None:
    return False
  # The logical CP size, not the job's world size: a TP-only rank also reports `world_size > 1`
  # and would otherwise take the CP paths (extra collectives, cache-dit attention dispatch) while
  # the unarmed bit-identical fallback is what TP-only runs must keep.
  ring_degree, ulysses_degree = getattr(cp_config, "mesh_shape", (1, 1))
  return int(ring_degree) * int(ulysses_degree) > 1


def _cp_group(owner):
  """Return the flattened CP process group of an armed owner.

  :param owner: Transformer or attention processor carrying `_cp_config`.
  :returns: The `ProcessGroup` covering every CP rank.
  """

  return owner._cp_config._flattened_mesh.get_group()


def _cp_ranks(owner) -> Tuple[int, int]:
  """Return `(rank, world_size)` inside the flattened CP group.

  :param owner: Transformer or attention processor carrying `_cp_config`.
  :returns: This rank's index and the group size.
  """

  group = _cp_group(owner)
  return dist.get_rank(group), dist.get_world_size(group)


# The self-attention is linear attention: two global sums over the token axis are all a shard needs.
_ATTN_PATCH_ORIGINALS: Dict[type, object] = {}
_ATTN_PATCH_INSTALLED = False


def _upstream_call(processor) -> Callable:
  """Return the saved upstream `__call__` of a patched attention processor.

  Looked up along the MRO so a user subclass that inherits the processor without overriding
  `__call__` still resolves to the upstream implementation.

  :param processor: The attention processor instance.
  :returns: The upstream unbound `__call__`.
  """

  for klass in type(processor).__mro__:
    original = _ATTN_PATCH_ORIGINALS.get(klass)
    if original is not None:
      return original
  raise RuntimeError(
    f"{type(processor).__name__} has no saved upstream `__call__`; "
    "sana_video context parallelism patches the processors lazily and this subclass bypasses it.")


def _install_attention_patches() -> None:
  """Install the CP-aware attention processor patches once per process."""

  global _ATTN_PATCH_INSTALLED
  if _ATTN_PATCH_INSTALLED:
    return

  from diffusers.models.transformers.transformer_sana_video import (
    SanaAttnProcessor2_0,
    SanaLinearAttnProcessor3_0,
  )

  _ATTN_PATCH_ORIGINALS.setdefault(SanaLinearAttnProcessor3_0, SanaLinearAttnProcessor3_0.__call__)
  _ATTN_PATCH_ORIGINALS.setdefault(SanaAttnProcessor2_0, SanaAttnProcessor2_0.__call__)
  SanaLinearAttnProcessor3_0.__call__ = __patch__sana_linear_processor_call__
  SanaAttnProcessor2_0.__call__ = __patch__sana_cross_processor_call__
  _ATTN_PATCH_INSTALLED = True


def _apply_rotary(query, key, rotary_emb):
  """Apply SANA's interleaved real RoPE to `[B, S, H, D]` query/key.

  :param query: Query tensor of shape `[B, S, H, D]`.
  :param key: Key tensor of shape `[B, S, H, D]`.
  :param rotary_emb:`(freqs_cos, freqs_sin)` of shape `[1, S, 1, D]` each.
  :returns: The rotated query and key.
  """

  def rotate(hidden_states, freqs_cos, freqs_sin):
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)

  return rotate(query, *rotary_emb), rotate(key, *rotary_emb)


def __patch__sana_linear_processor_call__(
  self,
  attn,
  hidden_states: torch.Tensor,
  encoder_hidden_states: Optional[torch.Tensor] = None,
  attention_mask: Optional[torch.Tensor] = None,
  rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
  """CP-aware `SanaLinearAttnProcessor3_0.__call__`.

  The armed path replaces the two per-token reductions of upstream's linear attention with global
  sums over the CP group: `Σ_j relu(k_j)` (the denominator) and `Σ_j v_j relu(k_rot_j)^T` (the key
  of the numerator). Each sum keeps the dtype upstream uses for it — the denominator in the input
  dtype, the numerator in fp32 — because only the reduction range changes, not the arithmetic.
  The unarmed path delegates to the saved upstream `__call__`, so a cp-free run stays
  bit-identical.

  :param self: The processor instance.
  :param attn: The `Attention` module owning this processor.
  :param hidden_states: Local query stream of shape `[B, S_local, C]`.
  :param encoder_hidden_states: Cross-attention stream; `None` for the self-attention.
  :param attention_mask: Unused by the self-attention path.
  :param rotary_emb: `(freqs_cos, freqs_sin)` for the local token window.
  :returns: The attention output of shape `[B, S_local, C]`.
  """

  original = _upstream_call(self)
  if not _cp_is_armed(self) or encoder_hidden_states is not None:
    return original(
      self,
      attn,
      hidden_states,
      encoder_hidden_states=encoder_hidden_states,
      attention_mask=attention_mask,
      rotary_emb=rotary_emb,
    )

  original_dtype = hidden_states.dtype
  query = attn.to_q(hidden_states)
  key = attn.to_k(hidden_states)
  value = attn.to_v(hidden_states)

  if attn.norm_q is not None:
    query = attn.norm_q(query)
  if attn.norm_k is not None:
    key = attn.norm_k(key)

  query = query.unflatten(2, (attn.heads, -1))
  key = key.unflatten(2, (attn.heads, -1))
  value = value.unflatten(2, (attn.heads, -1))

  query = F.relu(query)
  key = F.relu(key)

  query_rotate = query
  key_rotate = key
  if rotary_emb is not None:
    query_rotate, key_rotate = _apply_rotary(query, key, rotary_emb)

  # [B, H, C, S_local]
  query = query.permute(0, 2, 3, 1)
  key = key.permute(0, 2, 3, 1)
  query_rotate = query_rotate.permute(0, 2, 3, 1)
  key_rotate = key_rotate.permute(0, 2, 3, 1)
  value = value.permute(0, 2, 3, 1)

  query_rotate, key_rotate, value = query_rotate.float(), key_rotate.float(), value.float()

  group = _cp_group(self)
  # Both aggregates are sums over the token axis, so the global value is one SUM all-reduce away.
  # Dtypes follow upstream: `denom` stays in the input dtype, `numerator` is already fp32.
  denom = key.sum(dim=-1, keepdim=True)
  numerator = torch.matmul(value, key_rotate.transpose(-1, -2))
  dist.all_reduce(denom, op=dist.ReduceOp.SUM, group=group)
  dist.all_reduce(numerator, op=dist.ReduceOp.SUM, group=group)

  z = 1 / (denom.transpose(-2, -1) @ query + 1e-15)
  hidden_states = torch.matmul(numerator, query_rotate) * z

  hidden_states = hidden_states.flatten(1, 2).transpose(1, 2)
  hidden_states = hidden_states.to(original_dtype)

  hidden_states = attn.to_out[0](hidden_states)
  hidden_states = attn.to_out[1](hidden_states)
  return hidden_states


def __patch__sana_cross_processor_call__(
  self,
  attn,
  hidden_states: torch.Tensor,
  encoder_hidden_states: Optional[torch.Tensor] = None,
  attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
  """CP-aware `SanaAttnProcessor2_0.__call__`.

  The cross-attention query is the local token shard while key/value are the complete prompt, so
  the call is exact without any collective: it goes through cache-dit's `_dispatch_attention_fn`
  with `cp_config=None`. The unarmed path delegates to the saved upstream `__call__`.

  :param self: The processor instance.
  :param attn: The `Attention` module owning this processor.
  :param hidden_states: Local query stream of shape `[B, S_local, C]`.
  :param encoder_hidden_states: Complete prompt embedding of shape `[B, S_text, C]`.
  :param attention_mask: Optional text-padding mask of shape `[B, S_text]`.
  :returns: The attention output of shape `[B, S_local, C]`.
  """

  original = _upstream_call(self)
  if not _cp_is_armed(self):
    return original(
      self,
      attn,
      hidden_states,
      encoder_hidden_states=encoder_hidden_states,
      attention_mask=attention_mask,
    )

  batch_size, sequence_length, _ = (hidden_states.shape if encoder_hidden_states is None else
                                    encoder_hidden_states.shape)

  if attention_mask is not None:
    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
    attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

  query = attn.to_q(hidden_states)
  if encoder_hidden_states is None:
    encoder_hidden_states = hidden_states

  key = attn.to_k(encoder_hidden_states)
  value = attn.to_v(encoder_hidden_states)

  if attn.norm_q is not None:
    query = attn.norm_q(query)
  if attn.norm_k is not None:
    key = attn.norm_k(key)

  head_dim = key.shape[-1] // attn.heads
  query = query.view(batch_size, -1, attn.heads, head_dim)
  key = key.view(batch_size, -1, attn.heads, head_dim)
  value = value.view(batch_size, -1, attn.heads, head_dim)

  # The query is sharded, the cross-attention K/V are complete: no CP communication is needed.
  hidden_states = _dispatch_attention_fn(
    query,
    key,
    value,
    attn_mask=attention_mask,
    dropout_p=0.0,
    is_causal=False,
    backend=self._attention_backend,
    cp_config=None,
  )
  hidden_states = hidden_states.flatten(2, 3)
  hidden_states = hidden_states.type_as(query)

  hidden_states = attn.to_out[0](hidden_states)
  hidden_states = attn.to_out[1](hidden_states)
  return hidden_states / attn.rescale_output_factor


def _patch_rope_for_cp(transformer: nn.Module) -> None:
  """Patch `WanRotaryPosEmbed.forward` to return this rank's global position window.

  `diffusers`' rope builds its table from the tensor's own shape. Under the frame-axis split that
  would restart positions at 0 on every rank, so the patched method builds the table for the
  **global** frame count and slices `[offset, offset + local_frames)`.

  :param transformer: The SANA-Video transformer whose `rope` module is patched.
  """

  rope = getattr(transformer, "rope", None)
  if rope is None or getattr(rope, "_cp_patched", False):
    return
  rope_type = type(rope)

  def patched_rope(hidden_states: torch.Tensor):
    if not _cp_is_armed(transformer) or hidden_states.dim() != 5:
      return rope_type.forward(rope, hidden_states)
    local_frames = hidden_states.shape[2]
    group = _cp_group(transformer)
    rank, world_size = _cp_ranks(transformer)
    local_sizes = [None] * world_size
    dist.all_gather_object(local_sizes, int(local_frames), group=group)
    offset = sum(local_sizes[:rank])
    total_frames = sum(local_sizes)
    full_shape = (hidden_states.shape[0], hidden_states.shape[1], total_frames,
                  hidden_states.shape[3], hidden_states.shape[4])
    freqs_cos, freqs_sin = rope_type.forward(rope, hidden_states.new_empty(full_shape))
    tokens_per_frame = freqs_cos.shape[1] // total_frames
    start = offset * tokens_per_frame
    length = local_frames * tokens_per_frame
    return freqs_cos[:, start:start + length], freqs_sin[:, start:start + length]

  rope.forward = patched_rope
  rope._cp_patched = True


def _cp_temporal_conv_forward(conv: nn.Conv2d, x: torch.Tensor, owner: nn.Module):
  """Temporal `conv_temp` with one halo frame per side, so a chunk boundary matches the full run.

  Upstream calls this convolution on `[B, C, F_local, H * W]` with `padding=(1, 0)`. Under the
  frame-axis split the first/last local frames would see upstream's zero padding instead of their
  real neighbours, so the neighbouring ranks' boundary frames are exchanged and the convolution
  runs with `padding=(0, 0)` on the padded sequence. The two global ends keep zeros, which is
  exactly what upstream's padding produces.

  :param conv: The `conv_temp` module.
  :param x: Local input of shape `[B, C, F_local, H * W]`.
  :param owner: Transformer carrying the active `_cp_config`.
  :returns: The convolved local frames, same shape as `x`.
  """

  # A single local frame is legitimate (a 2-frame latent grid split across 2 ranks), and it is the
  # case that depends on the halo most: without it both neighbours fall back to zero padding.
  if not _cp_is_armed(owner) or x.dim() != 4:
    return type(conv).forward(conv, x)

  group = _cp_group(owner)
  rank, world_size = _cp_ranks(owner)
  first = x[:, :, :1].contiguous()
  last = x[:, :, -1:].contiguous()
  gathered_first = [torch.empty_like(first) for _ in range(world_size)]
  gathered_last = [torch.empty_like(last) for _ in range(world_size)]
  dist.all_gather(gathered_first, first, group=group)
  dist.all_gather(gathered_last, last, group=group)

  prev = gathered_last[rank - 1] if rank > 0 else torch.zeros_like(first)
  nxt = gathered_first[rank + 1] if rank < world_size - 1 else torch.zeros_like(first)
  padded = torch.cat([prev, x, nxt], dim=2)
  out = F.conv2d(
    padded,
    conv.weight,
    conv.bias,
    stride=conv.stride,
    padding=(0, 0),
    dilation=conv.dilation,
    groups=conv.groups,
  )
  # Both halo slots are always present (zeros at the two global ends), so a kernel-3 convolution
  # over `F_local + 2` already yields exactly `F_local` frames.
  return out


def _patch_temporal_conv_for_cp(transformer: nn.Module) -> None:
  """Give every `conv_temp` in the feed-forward stack the halo-aware forward.

  :param transformer: The SANA-Video transformer whose blocks are patched.
  """

  for module in transformer.modules():
    conv = getattr(module, "conv_temp", None)
    if not isinstance(conv, nn.Conv2d) or getattr(conv, "_cp_patched", False):
      continue
    conv.forward = functools.partial(_cp_temporal_conv_forward, conv, owner=transformer)
    conv._cp_patched = True


def _validate_frame_split(transformer: nn.Module, hidden_states: torch.Tensor) -> None:
  """Reject a latent frame count that the CP sharder cannot split evenly.

  :param transformer: The SANA-Video transformer carrying the active `_cp_config`.
  :param hidden_states: The 5D latent grid passed to `transformer.forward()`.
  """

  if not _cp_is_armed(transformer):
    return
  if hidden_states.dim() != 5:
    raise ValueError(
      "SANA-Video context parallelism expects the 5D latent grid `[B, C, frames, H, W]`; got a "
      f"{hidden_states.dim()}D tensor. Pass `output_type='latent'`-style 5D latents, not a packed "
      "sequence.")
  _, world_size = _cp_ranks(transformer)
  frames = hidden_states.shape[2]
  cp_config = getattr(transformer, "_cp_config", None)
  if getattr(cp_config, "ulysses_anything", False):
    # `--ulysses-anything` switches the sharder to an uneven split, which the halo / RoPE patches
    # already handle because they work from per-rank lengths instead of assuming equipartition.
    if frames >= world_size:
      return
  elif frames % world_size == 0:
    return

  candidates = [f for f in range(13, 121, 4) if ((f - 1) // 4 + 1) % world_size == 0]
  raise ValueError(
    "SANA-Video context parallelism shards the frame axis, so either the latent frame count must "
    f"divide by the CP size, or `--ulysses-anything` must be enabled: the input has {frames} "
    f"latent frames and the CP size is {world_size}. The Wan VAE compresses time by 4 "
    f"(`frames = 4k + 1`), so `--num-frames {candidates[0]}` runs on the even-split path (frames "
    f"from {candidates[:6]}...). Add `--ulysses-anything` to keep {frames} frames on the uneven "
    "split instead, which gives up `--compile` because the uneven gather cannot be compiled.")


@ContextParallelismPlannerRegister.register("SanaVideoTransformer3DModel")
class SanaVideoContextParallelismPlanner(ContextParallelismPlanner):
  """Context parallelism for `SanaVideoTransformer3DModel`.

  Shards the **frame axis** of the 5D latent grid, which keeps every shape the block derives from
  the latent grid local. See the module docstring for the linear-attention all-reduce, the RoPE
  window and the temporal-convolution halo.

  The output gather sits on the transformer itself, so it consumes the `(sample,)` tuple the
  pipelines ask for (`return_dict=False`); a `return_dict=True` call under CP is rejected by the
  gather hook instead of silently returning a partial grid.
  """

  def _apply(
    self,
    transformer: Optional[nn.Module] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ):
    """Build the CP plan for one SANA-Video transformer.

    :param transformer: The SANA-Video transformer.
    :param parallelism_config: The resolved parallelism configuration.
    :returns: The hook plan consumed by the CP runtime.
    """

    _install_attention_patches()
    _patch_rope_for_cp(transformer)
    _patch_temporal_conv_for_cp(transformer)
    if not getattr(transformer, "_cp_frame_guard", False):
      # Runs before the framework's split hook: the frame count has to be validated on the full
      # grid, otherwise the sharder fails with a bare divisibility assert.
      def _frame_guard(module, args, kwargs):
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        if hidden_states is not None:
          _validate_frame_split(module, hidden_states)

      transformer.register_forward_pre_hook(_frame_guard, with_kwargs=True, prepend=True)
      transformer._cp_frame_guard = True

    _cp_plan = {
      "": [
        {
          "hidden_states": _ContextParallelInput(split_dim=2, expected_dims=5, split_output=False),
          # I2V builds a per-frame 5D timestep (`conditioning_mask` zeroes the first frame, then
          # `expand` gives `[B, 1, F', h', w']`), which the transformer flattens to the same
          # `F' * h' * w'` token grid as `hidden_states`. T2V passes a 1D `[B]` timestep; the
          # `expected_dims` guard skips it.
          "timestep": _ContextParallelInput(split_dim=2, expected_dims=5, split_output=False),
        },
        # The pipelines call the transformer with `return_dict=False`, so the gather hook sees the
        # `(sample,)` tuple.
        _ContextParallelOutput(gather_dim=2, expected_dims=5),
      ],
    }
    return _cp_plan


@TensorParallelismPlannerRegister.register("SanaVideoTransformer3DModel")
class SanaVideoTensorParallelismPlanner(TensorParallelismPlanner):
  """Tensor parallelism for `SanaVideoTransformer3DModel`.

  Strategy: shard the attention projections only. The feed-forward is `GLUMBTempConv`, i.e. Conv2d
  blocks (1x1 + 3x3 depthwise + 1x1, plus a temporal 3x1), which `torch.distributed.tensor.parallel`
  cannot shard, so it stays replicated. Attention is worth sharding because the projections carry
  the full `inner_dim = 2240` on both sides, and `20` heads divide by 2 and 4.

  `qk_norm="rms_norm_across_heads"` normalises the **whole** `inner_dim`, so the per-head shard
  would normalise a partial vector: `norm_q` / `norm_k` are replaced with `_DistributedRMSNorm`,
  whose mean-square is all-reduced across the TP group (same recipe as the Wan planner).
  """

  def _apply(
    self,
    transformer: nn.Module,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> Tuple[nn.Module, List[Dict[str, ParallelStyle]]]:
    """Build the TP plan for one SANA-Video transformer.

    :param transformer: The SANA-Video transformer.
    :param parallelism_config: The resolved parallelism configuration.
    :returns: The parallelized transformer and its per-block layer plans.
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
  ) -> Tuple[nn.Module, List[Dict[str, ParallelStyle]]]:
    """Shard every block's attention projections across the TP mesh.

    :param transformer: The SANA-Video transformer.
    :param tp_mesh: The TP device mesh.
    :returns: The parallelized transformer and its per-block layer plans.
    """

    tp_size = tp_mesh.size()

    def prepare_block(block: nn.Module) -> Dict[str, ParallelStyle]:
      layer_plan = {
        "attn1.to_q": ColwiseParallel(),
        "attn1.to_k": ColwiseParallel(),
        "attn1.to_v": ColwiseParallel(),
        "attn1.to_out.0": RowwiseParallel(),
        "attn2.to_q": ColwiseParallel(),
        "attn2.to_k": ColwiseParallel(),
        "attn2.to_v": ColwiseParallel(),
        "attn2.to_out.0": RowwiseParallel(),
      }
      shard_div_attr(block.attn1, "heads", tp_size)
      shard_div_attr(block.attn2, "heads", tp_size)
      parallelize_module(
        module=block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_plan,
      )
      for attn in (block.attn1, block.attn2):
        if getattr(attn, "norm_q", None) is not None:
          attn.norm_q = _DistributedRMSNorm.from_rmsnorm(tp_mesh, attn.norm_q)
        if getattr(attn, "norm_k", None) is not None:
          attn.norm_k = _DistributedRMSNorm.from_rmsnorm(tp_mesh, attn.norm_k)
      return layer_plan

    layer_plans = []
    for _, block in transformer.transformer_blocks.named_children():
      layer_plans.append(prepare_block(block))
    return transformer, layer_plans
