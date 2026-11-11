"""Memory-bound fused operators for the FLUX.3 Klein transformer.

Every operator here replaces an expression the stock diffusers module already evaluates, so
the fused branch and the reference branch must agree to the tolerance stated in
:mod:`cache_dit.kernels.models.flux3_klein`: pure ``mul``/``add`` operators are bit-exact,
the ones built on ``rsqrtf`` are accepted at :data:`REL_L2_TOL` because the extension is
compiled with ``--use_fast_math``.

Numerical contract. The kernels keep the chain in fp32 and round to bf16 on the store. Where
upstream materialises an intermediate tensor the rounding is kept: the branch sum of the gated
residual is ``attn_output + ff_output`` (``transformer_flux3.py:284``), a real bf16 tensor that
stock+compile emits as its own ``add_24`` kernel, so the kernel rounds it too. That makes the
fused branch match the *compiled* arm, not the eager chain, which rounds at every ``aten``
boundary. ``_eager_*`` below is the fallback and therefore implements the same chain as the
kernel - a fallback that changed the numerics would make the precondition *be* the algorithm.

The kernels live in :mod:`cache_dit.kernels.models` (the shared ``cache_dit._C_ops``
extension). When the extension is missing, or a precondition fails (dtype, device, alignment,
shape, CP sharding, offload), the eager branch runs instead: that is the only fallback, and it
is never silent on the operator-selection level because :data:`FUSED_OPS_ENABLED` and the
per-operator preconditions are asserted in the tests.
"""

import os
from typing import Optional

import torch
import torch.nn.functional as F

from ....logger import init_logger

logger = init_logger(__name__)

_ENV_SWITCH = "CACHE_DIT_FLUX3_KLEIN_FUSED_OPS"
_ENV_OPS = "CACHE_DIT_FLUX3_KLEIN_OPS"
_FALSEY = ("0", "false", "off", "no")
ALL_OPS = (
  "qk_norm_rope",
  "ln_modulate",
  "gated_residual",
)
# The checkpoint's head width: the fused q/k norm + RoPE kernel is written for one 16-byte
# lane per (head_dim / 8) channels, i.e. for 128 channels.
_FUSED_HEAD_DIM = 128
# The extensions are built with `--use_fast_math`, which turns `rsqrtf` into the approximate
# SFU form. Operators that fuse it therefore cannot be bit-identical to the eager expression;
# they are accepted at this relative L2 error instead, with end-to-end PSNR as the real gate.
REL_L2_TOL = 1e-2


def _switch_from_env() -> bool:
  """Read the operator switch once, treating only explicit falsey values as "off".

  :returns: Whether fused operators are enabled.
  """

  value = os.environ.get(_ENV_SWITCH)
  if value is None:
    return True
  return value.strip().lower() not in _FALSEY


FUSED_OPS_ENABLED = _switch_from_env()


def _ops_from_env() -> frozenset[str]:
  """Return the per-operator allow-list.

  A pointwise chain that inductor already fuses can be *slower* as an opaque custom op,
  because the fusion boundary moves. ``CACHE_DIT_FLUX3_KLEIN_OPS=all`` or a comma separated
  list re-enables any subset for A/B runs; the default is the measured winner set.

  Measured on an RTX PRO 5000 (torch 2.15, 1024x1024, 16 steps, warmup then median of
  CUDA-event timings, one transformer forward): the eager chain is 290.3 ms and ``all`` is
  244.7 ms; under ``torch.compile`` the stock graph is 236.1 ms and ``all`` is 233.4 ms. Per
  operator, eager / compiled delta against the same arm: ``qk_norm_rope`` -45.4 / -2.7 ms,
  ``gated_residual`` -6.3 / -1.1 ms, ``ln_modulate`` -6.7 / -1.1 ms. The compiled arm's per
  operator differences sit inside the ~0.4% compilation spread, and no operator is negative,
  so every operator stays in the default set.

  :returns: Names of the operators allowed to dispatch to CUDA.
  """

  raw = os.environ.get(_ENV_OPS)
  if raw is None:
    return frozenset(ALL_OPS)
  value = raw.strip().lower()
  if value in ("", "all"):
    return frozenset(ALL_OPS)
  if value in _FALSEY:
    return frozenset()
  return frozenset(part.strip() for part in value.split(",") if part.strip())


ACTIVE_OPS = _ops_from_env()


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
  """Relative L2 distance, used to gate the fast-math-sensitive kernels.

  :param actual: Kernel output.
  :param expected: Reference output.
  :returns: `||actual - expected|| / ||expected||` in fp64.
  """

  delta = (actual.double() - expected.double()).norm()
  return (delta / expected.double().norm().clamp_min(1e-12)).item()


def set_flux3_klein_fused_ops_enabled(enabled: bool) -> None:
  """Override the operator switch at runtime, for parity tests and A/B runs.

  :param enabled: Whether :func:`apply_*` may dispatch to the CUDA kernels.
  """

  global FUSED_OPS_ENABLED
  FUSED_OPS_ENABLED = bool(enabled)


def fused_ops_extension_available() -> bool:
  """Return whether the shared CUDA extension can be imported.

  :returns: `True` when ``cache_dit._C_ops`` and its ops are usable.
  """

  from ....kernels import cuda_ext_is_available

  return cuda_ext_is_available()


_OP_ENTRY_POINTS = (
  "flux3_klein_qk_norm_rope",
  "flux3_klein_ln_modulate",
  "flux3_klein_ln_modulate_cat",
  "flux3_klein_gated_residual",
  "flux3_klein_gated_residual_split",
)
_LN_ROW_MULTIPLE = 1024  # one CTA covers a 128-thread row of 8-channel vectors
_LN_MAX_CHANNELS = 8192  # the kernel instantiates up to 8 vectors per thread
_ops_ready: Optional[bool] = None
_OPS: dict = {}


def _kernel_ops_ready() -> bool:
  """Return whether the shared extension exports this family's entry points.

  Checked once: "the extension loads" and "the extension was built with this model's kernels"
  are different facts, and an older wheel would otherwise raise inside the operator instead of
  falling back. The miss is logged once, because a silent fallback is what makes a benchmark
  report operator numbers for a path that never ran.

  :returns: `True` when every entry point of this family is present in ``_C_ops.ops``.
  """

  global _ops_ready
  if _ops_ready is not None:
    return _ops_ready

  from ....kernels import cuda_ext_is_available

  if not cuda_ext_is_available():
    _ops_ready = False
    return _ops_ready

  # Importing the bridge is what registers the `cache_dit_cuda_ops` schemas; the handles are then
  # cached so the hot path never imports again. A lazy `import_module` inside a block forward is a
  # dynamo graph break (measured: 39 graphs instead of 1, and torch.compile degenerated to eager),
  # so the resolution has to happen exactly once, outside any traced region.
  from ....kernels import flux3_klein_gated_residual  # noqa: F401 - registers the schemas
  import cache_dit._C_ops as extension

  namespace = torch.ops.cache_dit_cuda_ops
  _OPS.update({
    "qk_norm_rope": namespace.flux3_klein_qk_norm_rope,
    "ln_modulate": namespace.flux3_klein_ln_modulate,
    "ln_modulate_cat": namespace.flux3_klein_ln_modulate_cat,
    "gated_residual": namespace.flux3_klein_gated_residual,
    "gated_residual_split": namespace.flux3_klein_gated_residual_split,
  })

  missing = [name for name in _OP_ENTRY_POINTS if not hasattr(extension.ops, name)]
  if missing:
    logger.warning("The installed cache_dit._C_ops extension does not export "
                   f"{missing}; FLUX.3 Klein fused ops fall back to eager. Rebuild the "
                   "extension from this checkout "
                   "(CACHE_DIT_BUILD_CUDA=1 python -m pip install -e . "
                   "--no-build-isolation --no-deps).")
    _ops_ready = False
    return _ops_ready

  _ops_ready = True
  return _ops_ready


_HITS: dict[str, int] = {name: 0 for name in ALL_OPS}
_FALLBACKS: dict[str, int] = {name: 0 for name in ALL_OPS}


def fused_ops_stats() -> dict[str, dict[str, int]]:
  """Return how often each operator reached a kernel and how often it fell back.

  A run can look fused and still take the eager path on every call, because the preconditions are
  per tensor and there are several of them. Benchmarks and the API tests assert on this counter so
  "installed" can never be reported without "dispatched".

  :returns:`{"hits": {op: count}, "fallbacks": {op: count}}`.
  """

  return {"hits": dict(_HITS), "fallbacks": dict(_FALLBACKS)}


def reset_fused_ops_stats() -> None:
  """Zero the counters, so a benchmark can measure one arm at a time."""

  for name in ALL_OPS:
    _HITS[name] = 0
    _FALLBACKS[name] = 0


def _record_hit(op: str) -> None:
  _HITS[op] += 1


def _record_fallback(op: str) -> None:
  _FALLBACKS[op] += 1


def _fusable_tensor(x: torch.Tensor) -> bool:
  """Return whether `x` satisfies the layout every fused kernel assumes.

  A non-zero `storage_offset` is fine: `data_ptr()` already includes it, so a contiguous
  slice is a valid input. Only dtype, dense layout and a whole number of 16-byte vectors
  matter here; the C++ side re-checks byte alignment, which cannot be inspected while
  tracing.

  Nothing is inspected while `torch.compile` traces: `element_size()` / `is_contiguous()`
  based branches were splitting every block into its own graph (measured: 39 graphs and a
  whole-step regression). Under tracing the shapes are static, and the C++ entry point
  re-checks layout and raises instead of degrading, so the check is redundant there.

  :param x: Candidate tensor.
  :returns: `True` for a dense CUDA tensor of a supported dtype and vector-friendly size.
  """

  if torch.compiler.is_compiling():
    return True
  # bf16 only: the kernels and their vector loads are written for it, and the Python gate has to
  # be a subset of the C++ checks or a "supported" tensor raises instead of falling back.
  return (x.is_cuda and x.is_contiguous() and x.dtype == torch.bfloat16 and x.numel() % 8 == 0)


def _row_or_token(x: torch.Tensor, like: torch.Tensor) -> bool:
  """Return whether `x` is a per-token or a per-sample modulation tensor for `like`.

  Per-sample rows (`(B, 1, D)`) are accepted because the whole modulation chain broadcasts
  over the token axis; the kernel takes explicit row strides and therefore reads either form.

  A row view such as `temb_mod.chunk(3, dim=-1)[1]` is **not** contiguous - its row stride is
  three times its width - and requiring contiguity here silently sent every real block through
  the eager fallback. Only the inner axis has to be dense; the row stride is passed to the
  kernel, which needs a 16-byte aligned row start.

  Strides are not inspected while `torch.compile` traces, for the reason documented on
  :func:`_fusable_tensor`.

  :param x: Modulation tensor, `(B, S, D)` or `(B, 1, D)`.
  :param like: Tensor the modulation is applied to, `(B, S, D)`.
  :returns: `True` when the strides can be reproduced inside the kernel.
  """

  if torch.compiler.is_compiling():
    return True
  if not (x.is_cuda and x.dtype == like.dtype and x.dim() == 3 and x.stride(-1) == 1):
    return False
  if x.shape[0] != like.shape[0] or x.shape[2] != like.shape[2]:
    return False
  if x.shape[1] not in (1, like.shape[1]):
    return False
  # The kernels read 16 bytes at a time from a row start. Only the strides are checked here,
  # because `data_ptr()` is not traceable; the C++ entry point re-checks the base pointer.
  if (x.stride(0) * x.element_size()) % 16:
    return False
  return x.shape[1] == 1 or (x.stride(1) * x.element_size()) % 16 == 0


def _eager_qk_norm_rope(
  x: torch.Tensor,
  weight: torch.Tensor,
  cos: torch.Tensor,
  sin: torch.Tensor,
  eps: float,
) -> torch.Tensor:
  """Literal transcription of diffusers' fp32 RMSNorm followed by interleaved RoPE.

  Upstream is `attn.norm_q(query.float())` (fp32 in, fp32 weight, no bf16 rounding) then
  `apply_rotary_emb(..., sequence_dim=1)` then `.to(value.dtype)`.

  :param x: `(B, S, H, Dh)` q or k tensor in the model dtype.
  :param weight: `(Dh,)` fp32 RMSNorm scale.
  :param cos: `(S, Dh)` fp32 cosine table.
  :param sin: `(S, Dh)` fp32 sine table.
  :param eps: RMSNorm epsilon.
  :returns: Rotated tensor with the same shape and dtype as `x`.
  """

  width = x.shape[-1]
  normed = F.rms_norm(x.float(), (width, ), weight, eps)
  x_real, x_imag = normed.reshape(*normed.shape[:-1], -1, 2).unbind(-1)
  rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
  out = normed * cos.unsqueeze(1).unsqueeze(0) + rotated * sin.unsqueeze(1).unsqueeze(0)
  return out.to(x.dtype)


_EAGER_BYPASS = {
  "qk_norm_rope":
  lambda x, weight, cos, sin, eps: _eager_qk_norm_rope(x, weight, cos, sin, eps),
  "ln_modulate":
  lambda x, scale, shift, eps: _eager_ln_modulate(x, scale, shift, eps),
  "ln_modulate_cat":
  lambda hidden, encoder, scale, shift, c_scale, c_shift, eps: _eager_ln_modulate_cat(
    hidden, encoder, scale, shift, c_scale, c_shift, eps),
  "gated_residual":
  lambda hidden, gate, attn, ff: _eager_gated_residual(hidden, gate, attn, ff),
  "gated_residual_split":
  lambda hidden, encoder, gate, c_gate, attn, ff, prefix: _eager_gated_residual_split(
    hidden, encoder, gate, c_gate, attn, ff, prefix),
}


def apply_qk_norm_rope(
  x: torch.Tensor,
  weight: torch.Tensor,
  cos: torch.Tensor,
  sin: torch.Tensor,
  eps: float,
) -> torch.Tensor:
  """Apply q/k RMSNorm and interleaved RoPE in one kernel when possible.

  This is the single seam with measurable headroom in the compiled arm: inductor emits a
  persistent kernel that runs at 71.8% of DRAM peak (two elements per thread, 64 threads per
  block), while the whole chain only moves `x` and its bf16 output.

  :param x: `(B, S, H, Dh)` q or k tensor.
  :param weight: `(Dh,)` fp32 RMSNorm scale.
  :param cos: `(S, Dh)` fp32 cosine table.
  :param sin: `(S, Dh)` fp32 sine table.
  :param eps: RMSNorm epsilon.
  :returns: The rotated tensor.
  """

  if torch.compiler.is_compiling():
    # Inductor already fuses this chain into one kernel, and an opaque custom op *blocks*
    # the surrounding fusion: measured on FLUX.3, dispatching inside the compiled graph
    # fragmented it into 39 graphs and made the whole step 1.9% slower than stock. Handing
    # the compiled graph the literal expression keeps it at stock parity, while the eager
    # path keeps the kernels.
    return _EAGER_BYPASS["qk_norm_rope"](x, weight, cos, sin, eps)

  if (FUSED_OPS_ENABLED and "qk_norm_rope" in ACTIVE_OPS and _kernel_ops_ready()
      and _fusable_tensor(x) and x.dim() == 4 and x.shape[-1] == _FUSED_HEAD_DIM and weight.is_cuda
      and weight.is_contiguous() and weight.dtype == torch.float32
      and weight.numel() == _FUSED_HEAD_DIM and cos.is_cuda and sin.is_cuda
      and cos.dtype == torch.float32 and sin.dtype == torch.float32 and cos.is_contiguous()
      and sin.is_contiguous() and cos.shape == sin.shape and cos.shape[0] == x.shape[1]
      and cos.shape[1] == x.shape[-1] and cos.numel() % 8 == 0):
    _record_hit("qk_norm_rope")
    return _OPS["qk_norm_rope"](x, weight, cos, sin, eps)

  _record_fallback("qk_norm_rope")
  return _eager_qk_norm_rope(x, weight, cos, sin, eps)


def _eager_ln_modulate(
  x: torch.Tensor,
  scale: torch.Tensor,
  shift: torch.Tensor,
  eps: float,
) -> torch.Tensor:
  """Literal transcription of `(1 + mod_scale) * norm(x) + mod_shift`.

  The norm has no affine parameters upstream (`elementwise_affine=False`).

  :param x:`(B, S, D)` block input.
  :param scale:`(B, S, D)` or `(B, 1, D)` modulation scale.
  :param shift:`(B, S, D)` or `(B, 1, D)` modulation shift.
  :param eps: LayerNorm epsilon.
  :returns: The modulated normalized input.
  """

  normed = F.layer_norm(x, (x.shape[-1], ), None, None, eps)
  return (1 + scale) * normed + shift


def apply_ln_modulate(
  x: torch.Tensor,
  scale: torch.Tensor,
  shift: torch.Tensor,
  eps: float,
) -> torch.Tensor:
  """Apply LayerNorm and the `(1 + scale) * x + shift` modulation in one kernel.

  :param x:`(B, S, D)` block input.
  :param scale:`(B, S, D)` or `(B, 1, D)` modulation scale.
  :param shift:`(B, S, D)` or `(B, 1, D)` modulation shift.
  :param eps: LayerNorm epsilon.
  :returns: The modulated normalized input.
  """

  if torch.compiler.is_compiling():
    # Inductor already fuses this chain into one kernel, and an opaque custom op *blocks*
    # the surrounding fusion: measured on FLUX.3, dispatching inside the compiled graph
    # fragmented it into 39 graphs and made the whole step 1.9% slower than stock. Handing
    # the compiled graph the literal expression keeps it at stock parity, while the eager
    # path keeps the kernels.
    return _EAGER_BYPASS["ln_modulate"](x, scale, shift, eps)

  fusable = (FUSED_OPS_ENABLED and "ln_modulate" in ACTIVE_OPS and _kernel_ops_ready()
             and _fusable_tensor(x) and x.dim() == 3 and x.shape[-1] % _LN_ROW_MULTIPLE == 0
             and x.shape[-1] <= _LN_MAX_CHANNELS and _row_or_token(scale, x)
             and _row_or_token(shift, x) and scale.shape == shift.shape)
  if not fusable:
    _record_fallback("ln_modulate")
    return _eager_ln_modulate(x, scale, shift, eps)

  _record_hit("ln_modulate")
  return _OPS["ln_modulate"](x, scale, shift, eps, None)


def _eager_ln_modulate_cat(
  hidden_states: torch.Tensor,
  encoder_hidden_states: torch.Tensor,
  scale: torch.Tensor,
  shift: torch.Tensor,
  context_scale: torch.Tensor,
  context_shift: torch.Tensor,
  eps: float,
) -> torch.Tensor:
  """Literal transcription of the joint block's two-stream normalize + concat.

  :param hidden_states:`(B, Sc, D)` content tokens.
  :param encoder_hidden_states:`(B, Se, D)` text tokens.
  :param scale: Content modulation scale.
  :param shift: Content modulation shift.
  :param context_scale: Text modulation scale.
  :param context_shift: Text modulation shift.
  :param eps: LayerNorm epsilon.
  :returns:`(B, Se + Sc, D)` with the text tokens first, as upstream concatenates them.
  """

  normed_context = _eager_ln_modulate(encoder_hidden_states, context_scale, context_shift, eps)
  normed_content = _eager_ln_modulate(hidden_states, scale, shift, eps)
  return torch.cat([normed_context, normed_content], dim=1)


def apply_ln_modulate_cat(
  hidden_states: torch.Tensor,
  encoder_hidden_states: torch.Tensor,
  scale: torch.Tensor,
  shift: torch.Tensor,
  context_scale: torch.Tensor,
  context_shift: torch.Tensor,
  eps: float,
) -> torch.Tensor:
  """Normalize and modulate both joint-block streams straight into one concatenated buffer.

  Upstream materializes two normalized tensors and then copies both again in `torch.cat`;
  writing the text half first and the content half at its offset removes those copies.

  :param hidden_states: `(B, Sc, D)` content tokens.
  :param encoder_hidden_states: `(B, Se, D)` text tokens.
  :param scale: Content modulation scale.
  :param shift: Content modulation shift.
  :param context_scale: Text modulation scale.
  :param context_shift: Text modulation shift.
  :param eps: LayerNorm epsilon.
  :returns: `(B, Se + Sc, D)` concatenated modulated tokens.
  """

  if torch.compiler.is_compiling():
    # Inductor already fuses this chain into one kernel, and an opaque custom op *blocks*
    # the surrounding fusion: measured on FLUX.3, dispatching inside the compiled graph
    # fragmented it into 39 graphs and made the whole step 1.9% slower than stock. Handing
    # the compiled graph the literal expression keeps it at stock parity, while the eager
    # path keeps the kernels.
    return _EAGER_BYPASS["ln_modulate_cat"](hidden_states, encoder_hidden_states, scale, shift,
                                            context_scale, context_shift, eps)

  fusable = (FUSED_OPS_ENABLED and "ln_modulate" in ACTIVE_OPS and _kernel_ops_ready()
             and _fusable_tensor(hidden_states) and _fusable_tensor(encoder_hidden_states)
             and hidden_states.shape[0] == encoder_hidden_states.shape[0]
             and hidden_states.shape[-1] == encoder_hidden_states.shape[-1]
             and hidden_states.dim() == 3 and hidden_states.shape[-1] % _LN_ROW_MULTIPLE == 0
             and hidden_states.shape[-1] <= _LN_MAX_CHANNELS
             and _row_or_token(scale, hidden_states) and _row_or_token(shift, hidden_states)
             and _row_or_token(context_scale, encoder_hidden_states)
             and _row_or_token(context_shift, encoder_hidden_states))
  if not fusable:
    _record_fallback("ln_modulate")
    return _eager_ln_modulate_cat(hidden_states, encoder_hidden_states, scale, shift, context_scale,
                                  context_shift, eps)

  _record_hit("ln_modulate")
  return _OPS["ln_modulate_cat"](hidden_states, encoder_hidden_states, scale, shift, context_scale,
                                 context_shift, eps)


def _eager_gated_residual(
  hidden_states: torch.Tensor,
  gate: torch.Tensor,
  attn_output: torch.Tensor,
  ff_output: torch.Tensor,
) -> torch.Tensor:
  """Fallback expression, numerically identical to the kernel.

  Upstream materialises `attn_output + ff_output` as a bf16 tensor and then evaluates
  `hidden + gate * output`, which stock+compile fuses into one fp32 kernel. This function does
  the same thing explicitly, so a failed precondition cannot change the result - only the speed.

  :param hidden_states: Residual stream.
  :param gate:`(B, S, D)` or `(B, 1, D)` modulation gate.
  :param attn_output: Attention branch output.
  :param ff_output: Feed-forward branch output.
  :returns: The new residual stream.
  """

  return _compiled_gated_residual(hidden_states, gate, attn_output, ff_output)


def _compiled_gated_residual(
  hidden_states: torch.Tensor,
  gate: torch.Tensor,
  attn_output: torch.Tensor,
  ff_output: torch.Tensor,
) -> torch.Tensor:
  """The chain the *compiled* step runs, which is the oracle the kernel is tested against.

  `output = attn_output + ff_output` is a materialised tensor upstream
  (`transformer_flux3.py:284`) and stock+compile keeps it that way - inductor emits it as its
  own `triton_poi_..._add_24` kernel - while the gated add is recomputed in fp32. So there are
  exactly two rounding points, and the kernel reproduces both.

  :param hidden_states: Residual stream.
  :param gate: `(B, S, D)` or `(B, 1, D)` modulation gate.
  :param attn_output: Attention branch output.
  :param ff_output: Feed-forward branch output.
  :returns: The new residual stream.
  """

  branch = (attn_output + ff_output).to(hidden_states.dtype)
  return (hidden_states.float() + gate.float() * branch.float()).to(hidden_states.dtype)


def apply_gated_residual(
  hidden_states: torch.Tensor,
  gate: torch.Tensor,
  attn_output: torch.Tensor,
  ff_output: torch.Tensor,
) -> torch.Tensor:
  """Fuse the branch sum and the gated residual into one kernel.

  Upstream materializes `attn_output + ff_output` and then reads it back for the gated add;
  computing both in registers removes one write and one read of a `(B, S, D)` tensor per
  block. Out-of-place is a hard requirement: cache-dit derives a block's residual as
  `output - input`, so an in-place update would corrupt the cache.

  :param hidden_states: Residual stream.
  :param gate: `(B, S, D)` or `(B, 1, D)` modulation gate.
  :param attn_output: Attention branch output.
  :param ff_output: Feed-forward branch output.
  :returns: The new residual stream.
  """

  if torch.compiler.is_compiling():
    # Inductor already fuses this chain into one kernel, and an opaque custom op *blocks*
    # the surrounding fusion: measured on FLUX.3, dispatching inside the compiled graph
    # fragmented it into 39 graphs and made the whole step 1.9% slower than stock. Handing
    # the compiled graph the literal expression keeps it at stock parity, while the eager
    # path keeps the kernels.
    return _EAGER_BYPASS["gated_residual"](hidden_states, gate, attn_output, ff_output)

  if (FUSED_OPS_ENABLED and "gated_residual" in ACTIVE_OPS and _kernel_ops_ready()
      and _fusable_tensor(hidden_states) and _fusable_tensor(attn_output)
      and _fusable_tensor(ff_output) and hidden_states.dim() == 3
      and attn_output.shape == hidden_states.shape and ff_output.shape == hidden_states.shape
      and _row_or_token(gate, hidden_states)):
    _record_hit("gated_residual")
    return _OPS["gated_residual"](hidden_states, gate, attn_output, ff_output)

  _record_fallback("gated_residual")
  return _eager_gated_residual(hidden_states, gate, attn_output, ff_output)


def _eager_gated_residual_split(
  hidden_states: torch.Tensor,
  encoder_hidden_states: torch.Tensor,
  gate: torch.Tensor,
  context_gate: torch.Tensor,
  attn_output: torch.Tensor,
  ff_output: torch.Tensor,
  prefix_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Literal transcription of the joint block's split + two gated residuals.

  :param hidden_states: `(B, Sc, D)` content residual stream.
  :param encoder_hidden_states: `(B, Se, D)` text residual stream.
  :param gate: Content modulation gate.
  :param context_gate: Text modulation gate.
  :param attn_output: `(B, Se + Sc, D)` attention output over the joint sequence.
  :param ff_output: `(B, Se + Sc, D)` feed-forward output over the joint sequence.
  :param prefix_len: `Se`, the number of text tokens at the front of the joint sequence.
  :returns: `(encoder_hidden_states, hidden_states)` in that order, as upstream returns them.
  """

  return _compiled_gated_residual_split(hidden_states, encoder_hidden_states, gate, context_gate,
                                        attn_output, ff_output, prefix_len)


def _compiled_gated_residual_split(
  hidden_states: torch.Tensor,
  encoder_hidden_states: torch.Tensor,
  gate: torch.Tensor,
  context_gate: torch.Tensor,
  attn_output: torch.Tensor,
  ff_output: torch.Tensor,
  prefix_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Compiled-semantics reference for the joint block: one rounding on the branch sum, one on the
  gated add. See :func:`_compiled_gated_residual`.

  :param hidden_states: `(B, Sc, D)` content residual stream.
  :param encoder_hidden_states: `(B, Se, D)` text residual stream.
  :param gate: Content modulation gate.
  :param context_gate: Text modulation gate.
  :param attn_output: `(B, Se + Sc, D)` attention output over the joint sequence.
  :param ff_output: `(B, Se + Sc, D)` feed-forward output over the joint sequence.
  :param prefix_len: `Se`, the number of text tokens at the front of the joint sequence.
  :returns: `(encoder_hidden_states, hidden_states)` in that order.
  """

  branch = (attn_output + ff_output).to(hidden_states.dtype)
  context_branch, content_branch = torch.split(branch, [prefix_len, hidden_states.shape[1]], dim=1)
  encoder = (encoder_hidden_states.float() + context_gate.float() * context_branch.float()).to(
    hidden_states.dtype)
  content = (hidden_states.float() + gate.float() * content_branch.float()).to(hidden_states.dtype)
  return encoder, content


def apply_gated_residual_split(
  hidden_states: torch.Tensor,
  encoder_hidden_states: torch.Tensor,
  gate: torch.Tensor,
  context_gate: torch.Tensor,
  attn_output: torch.Tensor,
  ff_output: torch.Tensor,
  prefix_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Fuse the joint block's branch sum, split and two gated residuals into one kernel.

  :param hidden_states: `(B, Sc, D)` content residual stream.
  :param encoder_hidden_states: `(B, Se, D)` text residual stream.
  :param gate: Content modulation gate.
  :param context_gate: Text modulation gate.
  :param attn_output: `(B, Se + Sc, D)` attention output over the joint sequence.
  :param ff_output: `(B, Se + Sc, D)` feed-forward output over the joint sequence.
  :param prefix_len: `Se`, the number of text tokens at the front of the joint sequence.
  :returns: `(encoder_hidden_states, hidden_states)` in that order.
  """

  if torch.compiler.is_compiling():
    # Inductor already fuses this chain into one kernel, and an opaque custom op *blocks*
    # the surrounding fusion: measured on FLUX.3, dispatching inside the compiled graph
    # fragmented it into 39 graphs and made the whole step 1.9% slower than stock. Handing
    # the compiled graph the literal expression keeps it at stock parity, while the eager
    # path keeps the kernels.
    return _EAGER_BYPASS["gated_residual_split"](hidden_states, encoder_hidden_states, gate,
                                                 context_gate, attn_output, ff_output, prefix_len)

  fusable = (FUSED_OPS_ENABLED and "gated_residual" in ACTIVE_OPS and _kernel_ops_ready()
             and _fusable_tensor(hidden_states) and _fusable_tensor(encoder_hidden_states)
             and _fusable_tensor(attn_output) and _fusable_tensor(ff_output)
             and hidden_states.dim() == 3 and attn_output.shape == ff_output.shape
             and attn_output.shape[0] == hidden_states.shape[0]
             and attn_output.shape[2] == hidden_states.shape[-1]
             and attn_output.shape[1] == prefix_len + hidden_states.shape[1]
             and encoder_hidden_states.shape[1] == prefix_len
             and 0 <= prefix_len <= attn_output.shape[1] and _row_or_token(gate, hidden_states)
             and _row_or_token(context_gate, encoder_hidden_states))
  if not fusable:
    _record_fallback("gated_residual")
    return _eager_gated_residual_split(hidden_states, encoder_hidden_states, gate, context_gate,
                                       attn_output, ff_output, prefix_len)

  _record_hit("gated_residual")
  return _OPS["gated_residual_split"](hidden_states, encoder_hidden_states, gate, context_gate,
                                      attn_output, ff_output, prefix_len)


__all__ = [
  "ACTIVE_OPS",
  "ALL_OPS",
  "FUSED_OPS_ENABLED",
  "REL_L2_TOL",
  "apply_gated_residual",
  "apply_gated_residual_split",
  "apply_ln_modulate",
  "apply_ln_modulate_cat",
  "apply_qk_norm_rope",
  "fused_ops_extension_available",
  "fused_ops_stats",
  "relative_l2",
  "reset_fused_ops_stats",
  "set_flux3_klein_fused_ops_enabled",
]
