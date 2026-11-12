"""Fused-op call sites of the vendored Boogu-Image model.

Three normalizations dominate the block cost, and each one is a candidate for a
fused CUDA launch from ``cache_dit._C_ops``:

``apply_scaled_rms_norm``
  ``rms_norm(x) * (1 + scale)``, the modulation epilogue of every block.
``apply_gated_rms_norm``
  ``residual + tanh(gate) * rms_norm(x)``, the gated residual update.
``apply_qk_norm_rope``
  per-head ``norm_q`` / ``norm_k`` followed by interleaved RoPE.

Every helper keeps the pre-fusion expression as its fallback and only takes the
kernel path when all shape/dtype/alignment preconditions hold, so an unsupported
input (fp32 weights, DTensor shards, batch > 1, non-complex RoPE table, missing
extension, ...) silently produces the same numbers as before the fusion.

``CACHE_DIT_FUSED_OPS`` is the single switch (see :mod:`..env`), the same one the
FLUX and Qwen-Image families read. This family has no separate install step: the
helpers are called straight from the vendored transformer, so constructing
:class:`~.transformer_boogu.BooguImageTransformer2DModel` - which only cache-dit's
own ``BooguImagePipeline`` does - is the activation point. ``unset`` therefore
leaves the kernels on for every Boogu pipeline user, and ``0`` is the escape hatch
that bisects a suspected kernel problem (or reproduces the pre-fusion chain)
without editing code. ``set_boogu_fused_ops_enabled()`` stays as the in-process
handle used by the parity test.
"""

import torch

from ....kernels import cuda_ext_is_available
from ....kernels.models.boogu import boogu_fused_qk_norm_rope
from ....kernels.models.boogu import boogu_gated_rms_norm
from ....kernels.models.boogu import boogu_scaled_rms_norm
from ..env import fused_ops_default, fused_ops_enabled_on_install
from .rms_norm import RMSNorm

#: Norm classes whose forward is a plain RMS (`x * rsqrt(mean(x^2) + eps) * w`).
#: `diffusers.models.normalization.RMSNorm` is what `Attention(qk_norm="rms_norm")`
#: builds, so the QK fusion must accept it as well as the vendored norm.
_RMS_NORM_TYPES: tuple[type, ...] = (RMSNorm, )

try:  # diffusers always ships this, but keep the import defensive.
  from diffusers.models.normalization import RMSNorm as _DiffusersRMSNorm

  _RMS_NORM_TYPES = (RMSNorm, _DiffusersRMSNorm)
except ImportError:  # pragma: no cover - diffusers is a hard dependency
  pass

# The kernels pack 8 bf16 lanes per vec unit and cap the per-row channel count.
_MAX_FUSED_CHANNELS = 8192


def activate_fused_ops() -> None:
  """Turn the operator family on for a constructed Boogu transformer.

  Called from `BooguImageTransformer2DModel.__init__`, the cache-dit-owned object these helpers
  are reached from, so a `BooguImagePipeline` gets the kernels without the user setting anything.

  An explicit ``CACHE_DIT_FUSED_OPS=0`` wins over this, and a later
  :func:`set_boogu_fused_ops_enabled` call wins over the construction.
  """

  global FUSED_OPS_ENABLED
  if fused_ops_enabled_on_install():
    FUSED_OPS_ENABLED = True


FUSED_OPS_ENABLED = fused_ops_default()


def _boogu_kernels_usable() -> bool:
  """Return whether the Boogu fused path may run in this process.

  :returns: `True` when the family switch is on and `cache_dit._C_ops` is importable.
  """

  return FUSED_OPS_ENABLED and cuda_ext_is_available()


def set_boogu_fused_ops_enabled(enabled: bool) -> None:
  """Enable or disable the Boogu fused kernel path for this process.

  :param enabled: Whether the Boogu kernels may be used.
  """

  global FUSED_OPS_ENABLED
  FUSED_OPS_ENABLED = enabled


def _aligned16(t: torch.Tensor) -> bool:
  """Return whether `t` starts on a 16-byte boundary.

  The kernels load with `uint4` (16 B), so a tensor whose first element is not
  16 B-aligned would fault. Only the offset needs checking: the caching
  allocator hands out 512 B-aligned blocks, so `offset * itemsize % 16` fully
  decides alignment. Sliced views (e.g. the `chunk(4, dim=1)` modulation terms)
  are routinely offset but still aligned, which is why this must be an
  alignment test rather than `storage_offset() == 0`.

  :param t: Tensor about to be handed to a kernel.
  :returns: `True` when the first byte is 16 B-aligned.
  """

  return (t.storage_offset() * t.element_size()) % 16 == 0


def _rms_norm_convention(norm) -> bool | None:
  """Return the zero-centered weight flag when `norm` is a plain RMSNorm.

  The fused kernels only do an RMS reduction, so a module that subtracts the
  mean (`nn.LayerNorm`, or any custom norm exposing `weight`/`eps`) must never
  reach them. Type-based detection is the only reliable marker here: the
  vendored RMSNorm carries `zero_centered_weight`, while diffusers' RMSNorm does
  not, so an attribute-presence test alone would either drop the QK fusion
  (`Attention(qk_norm="rms_norm")` builds diffusers' class) or accept a
  LayerNorm.

  :param norm: Candidate norm module.
  :returns: The zero-centered flag, or `None` when the module is not an RMSNorm.
  """

  if not isinstance(norm, _RMS_NORM_TYPES):
    return None
  return bool(getattr(norm, "zero_centered_weight", False))


def _fusable_norm(x: torch.Tensor, norm, batch_one: bool) -> bool:
  """Return whether a norm module can be fused for this input.

  :param x: Input activations `[B, S, N]`.
  :param norm: Norm module providing `weight` / `eps` (and the zero-centered flag).
  :param batch_one: Whether the kernel requires `B == 1` (per-channel modulation).
  :returns: `True` when every precondition of the fused kernel holds.
  """

  if not _boogu_kernels_usable() or norm is None:
    return False
  weight = getattr(norm, "weight", None)
  if weight is None or getattr(norm, "bias", None) is not None:
    return False
  # A plain nn.LayerNorm passes the weight/bias checks while requiring a mean
  # subtraction the kernel does not perform, so gate on the norm class.
  if _rms_norm_convention(norm) is None:
    return False
  if not x.is_cuda or x.dtype is not torch.bfloat16 or x.dim() != 3:
    return False
  if batch_one and x.shape[0] != 1:
    return False
  channels = x.shape[-1]
  if channels == 0 or channels % 8 != 0 or channels > _MAX_FUSED_CHANNELS:
    return False
  # `weight.is_cuda` matters under layerwise offload: the norm weight can still
  # be on the CPU when the forward runs, and the kernel would reject it.
  return (weight.is_cuda and weight.dtype is torch.bfloat16 and weight.is_contiguous()
          and weight.numel() == channels and _aligned16(weight) and _aligned16(x)
          and x.is_contiguous())


def _fusable_scale(scale, channels: int) -> bool:
  """Return whether a `[B, D]` modulation term reduces to a per-channel vector.

  :param scale: Modulation tensor, or `None`.
  :param channels: Channel count of the tensor it scales.
  :returns: `True` when `scale` is a single-row bf16 contiguous CUDA `[1, D]`
    tensor with no storage offset.
  """

  if scale is None:
    return False
  return (scale.dim() == 2 and scale.shape[0] == 1 and scale.shape[1] == channels
          and scale.dtype is torch.bfloat16 and scale.is_cuda and scale.is_contiguous()
          and _aligned16(scale))


def apply_scaled_rms_norm(x: torch.Tensor, norm, scale=None) -> torch.Tensor:
  """Compute ``norm(x) * (1 + scale.unsqueeze(1))``, kernel-accelerated when possible.

  :param x: Input activations `[B, S, D]`.
  :param norm: RMSNorm module (needs `weight` / `eps`, no bias).
  :param scale: Optional modulation term `[1, D]`.
  :returns: Normalized (and optionally modulated) activations `[B, S, D]`.
  """

  if _fusable_norm(x, norm, batch_one=True):
    flat_scale = scale[0] if _fusable_scale(scale, x.shape[-1]) else None
    if scale is None or flat_scale is not None:
      return boogu_scaled_rms_norm(
        x,
        norm.weight,
        norm.eps,
        flat_scale,
        bool(_rms_norm_convention(norm)),
      )
  out = norm(x)
  if scale is not None:
    out = out * (1 + scale.unsqueeze(1))
  return out


def apply_gated_rms_norm(
  x: torch.Tensor,
  norm,
  gate: torch.Tensor,
  residual: torch.Tensor,
) -> torch.Tensor:
  """Compute ``residual + gate.unsqueeze(1).tanh() * norm(x)``, fused when possible.

  :param x: Input activations `[B, S, D]`.
  :param norm: RMSNorm module (needs `weight` / `eps`, no bias).
  :param gate: Per-channel gate `[B, D]`.
  :param residual: Residual stream `[B, S, D]`.
  :returns: The gated residual output `[B, S, D]`.
  """

  if (_fusable_norm(x, norm, batch_one=True) and _fusable_scale(gate, x.shape[-1])
      and residual.shape == x.shape and residual.dtype is x.dtype and residual.is_cuda
      and residual.is_contiguous() and _aligned16(residual)):
    return boogu_gated_rms_norm(
      x,
      norm.weight,
      norm.eps,
      gate[0],
      residual,
      bool(_rms_norm_convention(norm)),
    )
  return residual + gate.unsqueeze(1).tanh() * norm(x)


def _fusable_qk_norm_rope(x: torch.Tensor, rotary, norm) -> bool:
  """Return whether the QK-norm + RoPE fusion applies to one tensor.

  :param x: Query or key states `[B, S, H, D]`.
  :param rotary: Complex rotary table `[B, S, D/2]`.
  :param norm: Norm module applied per head.
  :returns: `True` when every precondition of the fused kernel holds.
  """

  if not _boogu_kernels_usable() or rotary is None or norm is None:
    return False
  weight = getattr(norm, "weight", None)
  if weight is None or getattr(norm, "bias", None) is not None:
    return False
  # Only plain RMSNorm modules may be fused, and the kernel has no zero-centered
  # flag (`weight + 1` would be dropped), so reject that convention outright.
  if _rms_norm_convention(norm) is not False:
    return False
  if x.dtype is not torch.bfloat16 or not x.is_cuda or x.dim() != 4 or not x.is_contiguous():
    return False
  if x.shape[-1] % 2 != 0 or weight.numel() != x.shape[-1]:
    return False
  if (not weight.is_cuda or weight.dtype is not torch.bfloat16 or not weight.is_contiguous()):
    return False
  if rotary.dtype not in (torch.complex64, torch.complex128) or not rotary.is_contiguous():
    return False
  return rotary.dim() == 3 and rotary.shape == (x.shape[0], x.shape[1], x.shape[3] // 2)


def apply_qk_norm_rope(
  query: torch.Tensor | None,
  key: torch.Tensor | None,
  rotary,
  norm_q=None,
  norm_k=None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
  """Apply per-head QK RMSNorm followed by interleaved RoPE.

  :param query: Query states `[B, S, Hq, D]`, or `None`.
  :param key: Key states `[B, S, Hkv, D]`, or `None`.
  :param rotary: Complex rotary table `[B, S, D/2]`, or `None`.
  :param norm_q: Norm module applied to `query`.
  :param norm_k: Norm module applied to `key`.
  :returns: The normalized and rotated `(query, key)` pair.
  """

  if (_fusable_qk_norm_rope(query, rotary, norm_q)
      and (key is None or _fusable_qk_norm_rope(key, rotary, norm_k))):
    if query is not None:
      query = boogu_fused_qk_norm_rope(query, rotary, norm_q.weight, norm_q.eps)
    if key is not None:
      key = boogu_fused_qk_norm_rope(key, rotary, norm_k.weight, norm_k.eps)
    return query, key

  if query is not None and norm_q is not None:
    query = norm_q(query)
  if key is not None and norm_k is not None:
    key = norm_k(key)
  if rotary is not None:
    from .embeddings import apply_rotary_emb

    if query is not None:
      query = apply_rotary_emb(query, rotary, use_real=False)
    if key is not None:
      key = apply_rotary_emb(key, rotary, use_real=False)
  return query, key
