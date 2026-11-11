"""Memory-bound fused operators for the FLUX.2-Klein transformer.

Every operator here replaces an expression the stock diffusers module already evaluates, so
the fused branch and the eager branch must produce the same values. The eager branch is
therefore not an approximation: wherever the upstream call is a single torch function it is
that function (`F.rms_norm` and `diffusers.models.embeddings.apply_rotary_emb`), and the
modulation chain is the literal sequence of bf16 tensor ops the block performs.

Which operators actually ship enabled is a measured decision, not a stylistic one: under
`torch.compile` inductor already fuses most of these chains, and NCU showed two of them sitting
at the DRAM ceiling with the traffic of the theoretical minimum. :data:`ACTIVE_OPS` therefore
defaults to the operators that measured faster, and the rest stay available behind
`CACHE_DIT_FLUX2_KLEIN_OPS` for A/B runs.

The kernels live in :mod:`cache_dit.kernels.models` (the shared ``cache_dit._C_ops``
extension). When the extension is missing or a precondition fails (dtype, device, alignment,
shape, batch, offload), the eager branch runs instead. That is the only fallback, and it is
never silent at the operator-selection level, because :data:`FUSED_OPS_ENABLED` and the
per-operator preconditions are asserted by the tests.
"""

import os
from typing import Optional

import torch
import torch.nn.functional as F

from ....logger import init_logger

logger = init_logger(__name__)

_ENV_SWITCH = "CACHE_DIT_FLUX2_KLEIN_FUSED_OPS"
_ENV_OPS = "CACHE_DIT_FLUX2_KLEIN_OPS"
_ENV_ROPE_CACHE = "CACHE_DIT_FLUX2_KLEIN_ROPE_CACHE"
_ENV_JOIN = "CACHE_DIT_FLUX2_KLEIN_JOIN"
_FALSEY = ("0", "false", "off", "no")
ALL_OPS = ("ln_modulate", "residual_ln_modulate", "qk_norm_rope", "gated_residual", "silu_mul")
# The measured winners under `compile` (see `.tmp/klein2-flux2/`): the two normalisation
# operators remove a full read of the residual stream and of the normalised tensor, and the
# q/k operator removes the redundant second load of every row. `gated_residual` and
# `silu_mul` only re-implement chains inductor already fuses at the DRAM ceiling, so they stay
# off by default and are kept for the A/B runs that produced these numbers.
DEFAULT_OPS = ("ln_modulate", "residual_ln_modulate", "qk_norm_rope")
_FUSED_HEAD_DIM = 128


def _switch_from_env() -> bool:
  """Read the operator switch once, treating only explicit falsey values as "off".

  :returns: Whether fused operators are enabled.
  """

  value = os.environ.get(_ENV_SWITCH)
  if value is None:
    return True
  return value.strip().lower() not in _FALSEY


FUSED_OPS_ENABLED = _switch_from_env()


def _rope_cache_from_env() -> bool:
  """Read the rotation-table cache switch once.

  `Flux2PosEmbed` rebuilds its float64 `cos`/`sin` tables on every forward; `img_ids` and
  `txt_ids` do not change between denoising steps, so the tables can be reused. Cache-dit
  measured that chain at 0.58 ms/step under compile, which is a pure constant-folding win.

  :returns: Whether the tables are cached per position-id tensor.
  """

  value = os.environ.get(_ENV_ROPE_CACHE)
  if value is None:
    # Measured neutral under `compile_repeated_blocks()` (189.43 ms/step with the cache,
    # 189.45 without, i.e. inside the noise of the run), so it does not ship enabled: the cache
    # keys on the position-id tensor's address, and keeping a feature that buys nothing while
    # it can hold a tensor alive is not worth it.
    return False
  return value.strip().lower() not in _FALSEY


ROPE_CACHE_ENABLED = _rope_cache_from_env()


def _join_from_env() -> bool:
  """Read the joined-stream switch once.

  The double block normalises a text and an image stream and then concatenates them; the
  joined operator writes both into one buffer so the concatenation never runs. Measured against
  two calls plus `torch.cat` on the compiled arm it is neutral: -0.41 ms/step with one build
  order and +0.11 ms with the other, and in both cases whichever pipeline was built *second* in
  the process ran 0.3-0.5 ms faster. The comparison was therefore measuring the process, not the
  copy. It ships disabled so there is one path to reason about, not because the copy is free.

  :returns: Whether the joined variant is used.
  """

  value = os.environ.get(_ENV_JOIN)
  if value is None:
    return False
  return value.strip().lower() not in _FALSEY


JOIN_ENABLED = _join_from_env()


def _ops_from_env() -> frozenset[str]:
  """Return the per-operator allow-list.

  The default is the measured winner set, because an opaque custom operator that replaces a
  chain inductor already fuses usually loses: the fusion boundary moves and the traffic does
  not. `all` or a comma separated list re-enables any subset for A/B runs.

  :returns: Names of the operators allowed to dispatch to CUDA.
  """

  raw = os.environ.get(_ENV_OPS)
  if raw is None:
    return frozenset(DEFAULT_OPS)
  value = raw.strip().lower()
  if value in ("", "all"):
    return frozenset(ALL_OPS)
  if value in _FALSEY:
    return frozenset()
  return frozenset(part.strip() for part in value.split(",") if part.strip())


ACTIVE_OPS = _ops_from_env()

# The kernels are built with `--use_fast_math`, so the one operator that needs a transcendental
# (`silu`'s sigmoid) cannot be bit-identical to the eager expression and is gated here. The
# mul/add operators are exact and are compared with `torch.equal`; the normalisations differ
# only in their reduction tree (measured rel-L2 1.4e-5, one bf16 ulp).
REL_L2_TOL = 1e-2
NORM_REL_L2_TOL = 1e-4


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
  """Relative L2 distance, used to gate the operators that are not bit-exact.

  :param actual: Kernel output.
  :param expected: Eager reference.
  :returns: `||actual - expected|| / ||expected||` in fp64.
  """

  delta = (actual.double() - expected.double()).norm()
  return (delta / expected.double().norm().clamp_min(1e-12)).item()


def set_flux2_klein_fused_ops_enabled(enabled: bool) -> None:
  """Override the operator switch at runtime, for parity tests and A/B runs.

  :param enabled: Whether :func:`apply_*` may dispatch to the CUDA kernels.
  """

  global FUSED_OPS_ENABLED
  FUSED_OPS_ENABLED = bool(enabled)


def set_flux2_klein_active_ops(ops) -> None:
  """Override the per-operator allow-list at runtime.

  :param ops: Iterable of operator names, or `None` to disable every operator.
  """

  global ACTIVE_OPS
  ACTIVE_OPS = frozenset() if ops is None else frozenset(ops)


def fused_ops_extension_available() -> bool:
  """Return whether the shared CUDA extension can be imported.

  :returns: `True` when ``cache_dit._C_ops`` and its ops are usable.
  """

  from ....kernels import cuda_ext_is_available

  return cuda_ext_is_available()


def _fusable_tensor(x: torch.Tensor) -> bool:
  """Return whether `x` satisfies the dense layout every fused kernel assumes.

  :param x: Candidate tensor.
  :returns: `True` for a dense CUDA bf16 tensor with a whole number of 16-byte vectors.
  """

  return (x.is_cuda and x.is_contiguous() and x.dtype in (torch.bfloat16, torch.float16)
          and x.numel() % 8 == 0)


def _fusable_qk(x: torch.Tensor) -> bool:
  """Return whether `x` can be read in place by the q/k kernel.

  The kernel addresses rows through the tensor's own strides, so a chunk of a fused projection
  is valid input: only the innermost dimension has to be dense and the row strides have to keep
  the 16-byte vector load aligned (the byte-level alignment is re-checked in C++, which cannot
  be inspected while tracing).

  :param x: Candidate `[1, S, H, D]` tensor.
  :returns: `True` when the kernel may read it directly.
  """

  return (x.is_cuda and x.dtype in (torch.bfloat16, torch.float16) and x.dim() == 4
          and x.shape[0] == 1 and x.shape[3] == _FUSED_HEAD_DIM and x.stride(-1) == 1
          and x.stride(-2) % 8 == 0 and x.stride(-3) % 8 == 0)


def _fusable_rows(rows: torch.Tensor, dim: int) -> bool:
  """Return whether a `[R, D]` modulation tensor can feed the kernels.

  :param rows: Candidate `[R, D]` tensor.
  :param dim: Expected hidden size.
  :returns: `True` for a dense CUDA bf16 tensor of the expected width.
  """

  # A non-zero storage offset is normal: the modulation rows are chunks of one narrow tensor,
  # and every kernel addresses them through `data_ptr()`, which already includes the offset.
  return (rows.is_cuda and rows.is_contiguous() and rows.dtype == torch.bfloat16 and rows.dim() == 2
          and rows.shape[-1] == dim)


def _enabled(name: str) -> bool:
  """Return whether one operator is enabled.

  :param name: Operator name from :data:`ALL_OPS`.
  :returns: `True` when the operator may dispatch to CUDA.
  """

  return FUSED_OPS_ENABLED and name in ACTIVE_OPS


def modulation_rows(params: torch.Tensor, dim: int) -> torch.Tensor:
  """View a block modulation chunk as the kernel's folded `[R, D]` rows.

  `Flux2Modulation.split` returns tensors shaped `[B, 1, D]`, i.e. one row per sample; the
  kernels index them by sample instead of broadcasting them over the token axis, which is the
  materialisation this avoids.

  :param params: `[B, 1, D]` (or `[B, D]`) modulation chunk.
  :param dim: Expected hidden size.
  :returns: A contiguous `[B, D]` view.
  """

  rows = params[..., :dim]
  return rows.reshape(rows.shape[0], dim)


def _eager_ln_modulate(
  x: torch.Tensor,
  one_plus: torch.Tensor,
  shift: torch.Tensor,
  eps: float,
) -> torch.Tensor:
  """Literal transcription of the block's `(1 + scale) * norm(x) + shift`.

  :param x:`[B, S, D]` block input.
  :param one_plus:`[R, D]` folded `1 + scale` rows, already rounded to bf16 by torch.
  :param shift:`[R, D]` shift rows.
  :param eps: LayerNorm epsilon.
  :returns:`[B, S, D]` modulated normalisation.
  """

  normed = F.layer_norm(x, (x.shape[-1], ), None, None, eps)
  # `[R, D] -> [B, 1, D]` is exactly the shape the upstream broadcast uses.
  return one_plus.unsqueeze(1) * normed + shift.unsqueeze(1)


def apply_ln_modulate(
  x: torch.Tensor,
  one_plus: torch.Tensor,
  shift: torch.Tensor,
  eps: float,
) -> torch.Tensor:
  """Apply LayerNorm and the folded modulation in one pass over `x`.

  :param x: `[B, S, D]` block input.
  :param one_plus: `[R, D]` folded `1 + scale` rows, `R` in `{1, B}`.
  :param shift: `[R, D]` shift rows.
  :param eps: LayerNorm epsilon.
  :returns: `[B, S, D]` modulated normalisation.
  """

  dim = x.shape[-1]
  rows_ok = _fusable_rows(one_plus, dim) and _fusable_rows(shift, dim)
  if (_enabled("ln_modulate") and rows_ok and _fusable_tensor(x) and dim % 128 == 0
      and x.shape[0] in (1, one_plus.shape[0])):
    from ....kernels import flux2_klein_ln_modulate

    return flux2_klein_ln_modulate(x, one_plus, shift, eps)
  return _eager_ln_modulate(x, one_plus, shift, eps)


def _eager_residual_ln_modulate(
  hidden: torch.Tensor,
  gate: torch.Tensor,
  delta: torch.Tensor,
  one_plus: torch.Tensor,
  shift: torch.Tensor,
  eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Literal transcription of `hidden + gate * delta` followed by the modulation.

  :param hidden:`[B, S, D]` residual stream.
  :param gate:`[R, D]` gate rows.
  :param delta:`[B, S, D]` block output.
  :param one_plus:`[R, D]` folded `1 + scale` rows.
  :param shift:`[R, D]` shift rows.
  :param eps: LayerNorm epsilon.
  :returns:`(new residual stream, modulated normalisation of it)`.
  """

  updated = hidden + gate.unsqueeze(1) * delta
  return updated, _eager_ln_modulate(updated, one_plus, shift, eps)


def apply_residual_ln_modulate(
  hidden: torch.Tensor,
  gate: torch.Tensor,
  delta: torch.Tensor,
  one_plus: torch.Tensor,
  shift: torch.Tensor,
  eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Add the gated residual and normalise the result in one pass over `hidden`.

  The eager chain reads `hidden` for the add and twice more inside the layer norm; the fused
  operator reads it once. This is the operator that removes the most traffic of the family,
  because every block performs it right before its second normalisation.

  :param hidden: `[B, S, D]` residual stream.
  :param gate: `[R, D]` gate rows.
  :param delta: `[B, S, D]` block output.
  :param one_plus: `[R, D]` folded `1 + scale` rows.
  :param shift: `[R, D]` shift rows.
  :param eps: LayerNorm epsilon.
  :returns: `(new residual stream, modulated normalisation of it)`.
  """

  dim = hidden.shape[-1]
  rows_ok = (_fusable_rows(one_plus, dim) and _fusable_rows(shift, dim)
             and _fusable_rows(gate, dim))
  if (_enabled("residual_ln_modulate") and rows_ok and _fusable_tensor(hidden)
      and _fusable_tensor(delta) and hidden.shape == delta.shape and dim % 128 == 0
      and hidden.shape[0] in (1, one_plus.shape[0])):
    from ....kernels import flux2_klein_residual_ln_modulate

    return flux2_klein_residual_ln_modulate(hidden, gate, delta, one_plus, shift, eps)
  return _eager_residual_ln_modulate(hidden, gate, delta, one_plus, shift, eps)


def _eager_qk_norm_rope(
  x: torch.Tensor,
  weight: torch.Tensor,
  cos: torch.Tensor,
  sin: torch.Tensor,
  rope_row_offset: int,
  eps: float,
  norm: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
  """The upstream chain itself: the norm module followed by `apply_rotary_emb`.

  Calling the very functions diffusers calls makes the fallback identical to the unoptimised
  path by construction, which is what a fallback has to be. The norm is invoked as a *module*
  when one is given: under `--module-layerwise-cpu-offload` its weight still lives on the CPU
  until accelerate's pre-forward hook moves it, so reading `.weight` directly would raise
  "weight is on cpu" instead of falling back cleanly.

  :param x: `[1, S, H, D]` query or key tensor.
  :param weight: `[D]` scale of the `torch.nn.RMSNorm(head_dim)` module.
  :param cos: `[rope_rows, D]` fp32 cosine table.
  :param sin: `[rope_rows, D]` fp32 sine table.
  :param rope_row_offset: First rotation row of this stream.
  :param eps: RMSNorm epsilon.
  :param norm: The `torch.nn.RMSNorm` module, used by the fallback when available.
  :returns: Rotated tensor with the same shape and dtype as `x`.
  """

  from diffusers.models.embeddings import apply_rotary_emb

  if norm is not None:
    normed = norm(x)
  else:
    normed = F.rms_norm(x, (x.shape[-1], ), weight, eps)
  table = (cos[rope_row_offset:rope_row_offset + x.shape[1]],
           sin[rope_row_offset:rope_row_offset + x.shape[1]])
  return apply_rotary_emb(normed, table, sequence_dim=1)


def apply_qk_norm_rope(
  x: torch.Tensor,
  weight: torch.Tensor,
  cos: torch.Tensor,
  sin: torch.Tensor,
  rope_row_offset: int,
  eps: float,
  norm: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
  """Apply the per-head RMSNorm and the interleaved RoPE in one pass over `x`.

  :param x: `[1, S, H, D]` query or key tensor.
  :param weight: `[D]` `torch.nn.RMSNorm` scale.
  :param cos: `[rope_rows, D]` fp32 cosine table.
  :param sin: `[rope_rows, D]` fp32 sine table.
  :param rope_row_offset: First rotation row of this stream.
  :param eps: RMSNorm epsilon.
  :param norm: The `torch.nn.RMSNorm` module, used by the fallback when available.
  :returns: Rotated tensor with the same shape and dtype as `x`.
  """

  if (_enabled("qk_norm_rope") and _fusable_qk(x) and weight.is_cuda and weight.dtype == x.dtype
      and weight.is_contiguous() and weight.shape[-1] == _FUSED_HEAD_DIM
      and cos.dtype == torch.float32 and cos.is_cuda and cos.is_contiguous()
      and sin.dtype == torch.float32 and sin.is_cuda and sin.is_contiguous()
      and cos.shape[-1] == _FUSED_HEAD_DIM and sin.shape == cos.shape):
    from ....kernels import flux2_klein_qk_norm_rope

    return flux2_klein_qk_norm_rope(x, weight, cos, sin, rope_row_offset, eps).unsqueeze(0)
  return _eager_qk_norm_rope(x, weight, cos, sin, rope_row_offset, eps, norm=norm)


def apply_qk_norm_rope_join(
  image: torch.Tensor,
  image_weight: torch.Tensor,
  text: torch.Tensor,
  text_weight: torch.Tensor,
  cos: torch.Tensor,
  sin: torch.Tensor,
  eps: float,
  image_norm: Optional[torch.nn.Module] = None,
  text_norm: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
  """Normalise and rotate both attention streams into one joint buffer.

  Equivalent to `cat([norm_rope(text), norm_rope(image)], dim=1)` - the double blocks build
  exactly that - but the joint tensor is written once instead of being produced, then read and
  copied by a cat.

  :param image: `[1, S_img, H, D]` image-stream tensor.
  :param image_weight: `[D]` scale of the image stream.
  :param text: `[1, S_txt, H, D]` text-stream tensor.
  :param text_weight: `[D]` scale of the text stream.
  :param cos: `[S_txt + S_img, D]` fp32 cosine table, text rows first.
  :param sin: `[S_txt + S_img, D]` fp32 sine table, text rows first.
  :param eps: RMSNorm epsilon.
  :returns: `[1, S_txt + S_img, H, D]` joined tensor.
  """

  total = text.shape[1] + image.shape[1]
  if (_enabled("qk_norm_rope") and _fusable_qk(image) and _fusable_qk(text)
      and image.shape[0] == text.shape[0] and image.shape[2] == text.shape[2]
      and image.dtype == text.dtype and image_weight.is_cuda and text_weight.is_cuda
      and image_weight.dtype == image.dtype and text_weight.dtype == text.dtype
      and image_weight.is_contiguous() and text_weight.is_contiguous()
      and cos.dtype == torch.float32 and cos.is_cuda and cos.is_contiguous()
      and sin.dtype == torch.float32 and sin.is_cuda and sin.is_contiguous()
      and cos.shape == (total, _FUSED_HEAD_DIM) and sin.shape == cos.shape):
    from ....kernels import flux2_klein_qk_norm_rope_join

    return flux2_klein_qk_norm_rope_join(image, image_weight, text, text_weight, cos, sin,
                                         eps).unsqueeze(0)
  rotated_text = _eager_qk_norm_rope(text, text_weight, cos, sin, 0, eps, norm=text_norm)
  rotated_image = _eager_qk_norm_rope(image,
                                      image_weight,
                                      cos,
                                      sin,
                                      text.shape[1],
                                      eps,
                                      norm=image_norm)
  return torch.cat([rotated_text, rotated_image], dim=1)


def _eager_gated_residual(
  hidden: torch.Tensor,
  gate: torch.Tensor,
  delta: torch.Tensor,
) -> torch.Tensor:
  """Literal transcription of `hidden + gate * delta`.

  :param hidden:`[B, S, D]` residual stream.
  :param gate:`[R, D]` gate rows.
  :param delta:`[B, S, D]` block output.
  :returns: The new residual stream.
  """

  return hidden + gate.unsqueeze(1) * delta


def apply_gated_residual(
  hidden: torch.Tensor,
  gate: torch.Tensor,
  delta: torch.Tensor,
) -> torch.Tensor:
  """Add the gated residual, out of place.

  Out of place is a hard requirement: cache-dit derives a block's residual as
  `output - input`, so an in-place add would corrupt the cache.

  :param hidden: `[B, S, D]` residual stream.
  :param gate: `[R, D]` gate rows.
  :param delta: `[B, S, D]` block output.
  :returns: The new residual stream.
  """

  dim = hidden.shape[-1]
  if (_enabled("gated_residual") and _fusable_rows(gate, dim) and _fusable_tensor(hidden)
      and _fusable_tensor(delta) and hidden.shape == delta.shape and hidden.shape[0] == 1
      and gate.shape[0] == 1 and dim % 8 == 0):
    from ....kernels import flux2_klein_gated_residual

    return flux2_klein_gated_residual(hidden, gate.reshape(dim), delta)
  return _eager_gated_residual(hidden, gate, delta)


def _eager_silu_mul(gate_up: torch.Tensor) -> torch.Tensor:
  """Literal transcription of the SwiGLU activation.

  :param gate_up:`[B, S, 2D]` packed gate and value projections.
  :returns:`[B, S, D]` gated activation.
  """

  half = gate_up.shape[-1] // 2
  return F.silu(gate_up[..., :half]) * gate_up[..., half:]


def apply_silu_mul(gate_up: torch.Tensor) -> torch.Tensor:
  """Apply the SwiGLU activation in one pass over the packed projections.

  :param gate_up:`[B, S, 2D]` packed gate and value projections.
  :returns:`[B, S, D]` gated activation.
  """

  # Read in place through the row stride: the caller hands over the MLP half of a fused
  # QKV+MLP projection, whose rows are packed in `(2D)` but whose row stride spans the whole
  # fused width.
  if (_enabled("silu_mul") and gate_up.is_cuda and gate_up.dtype == torch.bfloat16
      and gate_up.dim() == 3 and gate_up.shape[0] == 1 and gate_up.shape[-1] % 16 == 0
      and gate_up.stride(-1) == 1 and gate_up.stride(-2) % 8 == 0):
    from ....kernels import flux2_klein_silu_mul

    return flux2_klein_silu_mul(gate_up)
  return _eager_silu_mul(gate_up)


__all__ = [
  "ACTIVE_OPS",
  "ALL_OPS",
  "DEFAULT_OPS",
  "FUSED_OPS_ENABLED",
  "ROPE_CACHE_ENABLED",
  "apply_gated_residual",
  "apply_ln_modulate",
  "apply_qk_norm_rope",
  "apply_qk_norm_rope_join",
  "apply_residual_ln_modulate",
  "apply_silu_mul",
  "fused_ops_extension_available",
  "modulation_rows",
  "relative_l2",
  "set_flux2_klein_active_ops",
  "set_flux2_klein_fused_ops_enabled",
]
