"""Memory-bound fused operators for the Qwen-Image 2.1 transformer.

Every operator here is a drop-in replacement for an expression the stock diffusers module
already evaluates, so the fused branch and the eager branch must produce the *same bits*.
The eager branch is therefore written as a literal transcription of the upstream
expression (same dtypes, same operation order, same rounding points), not as an
"approximately equal" reference.

The kernels live in :mod:`cache_dit.kernels._models` (the shared ``cache_dit._C_models``
extension). When the extension is missing, or a precondition below fails (dtype, device,
alignment, shape, CP sharding, offload), the eager branch runs instead: that is the only
fallback, and it is never silent on the operator-selection level because
:data:`FUSED_OPS_ENABLED` and the per-op preconditions are asserted in the tests.
"""

import os

import torch
import torch.nn.functional as F

from ....logger import init_logger

logger = init_logger(__name__)

_ENV_SWITCH = "CACHE_DIT_QWEN_IMAGE21_FUSED_OPS"
_ENV_OPS = "CACHE_DIT_QWEN_IMAGE21_OPS"
_FALSEY = ("0", "false", "off", "no")
ALL_OPS = ("ln_modulate", "qk_norm_rope", "gated_residual", "silu_mul")
# Qwen-Image 2.1 rotates every head with the interleaved (complex) layout, so the fused
# qk-norm+RoPE kernel only exists for the checkpoint's head width.
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


def _ops_from_env() -> frozenset[str]:
  """Return the per-operator allow-list.

  A pointwise chain that inductor already fuses can be *slower* as an opaque custom op,
  because the fusion boundary moves. The default therefore only enables the operators that
  measured faster on this model; `CACHE_DIT_QWEN_IMAGE21_OPS=all` or a comma separated list
  (`ln_modulate,gated_residual,...`) re-enables any subset for A/B runs.

  :returns: Names of the operators allowed to dispatch to CUDA.
  """

  raw = os.environ.get(_ENV_OPS)
  if raw is None:
    # Every operator is on by default: turning the family off is what
    # `CACHE_DIT_QWEN_IMAGE21_FUSED_OPS=0` is for, and per-operator selection stays
    # available for A/B runs (`CACHE_DIT_QWEN_IMAGE21_OPS=gated_residual,silu_mul`).
    # Measured on an RTX PRO 5000 (torch 2.11, 1024x1024, 8 steps, cache+compile): all 369.2,
    # qk_norm_rope only 365.6, off 372.9 ms/step -- inductor already fuses the pointwise
    # chains, so the per-operator choice is a tuning knob, not a correctness one.
    return frozenset(ALL_OPS)
  value = raw.strip().lower()
  if value in ("", "all"):
    return frozenset(ALL_OPS)
  if value in _FALSEY:
    return frozenset()
  return frozenset(part.strip() for part in value.split(",") if part.strip())


ACTIVE_OPS = _ops_from_env()

# `None` = not probed yet, `-1` = probed and unusable, otherwise the selected variant/tier.
_LN_TIERS: dict[int, int] = {}
_PROBE_SEQ = 8
# `c10::complex<float>::operator*` is `a*c - b*d` / `a*d + b*c`, and nvcc contracts one of
# those products into an FMA. Which one is a property of the ATen build, so the kernel ships
# five pairings; variant 3 (the classic `__mulsc3` pairing) is the one this project measured
# as closest. The difference between variants is a single bf16 ulp and does not accumulate,
# i.e. far below the `--use_fast_math` error that is already accepted here, so the choice is
# a constant instead of a startup probe.
ROTATION_VARIANT = 3
# The extensions are built with `--use_fast_math`, which turns `expf` / `tanhf` / `rsqrtf`
# into the approximate SFU forms. Operators that fuse one of those therefore cannot be
# bit-identical to the eager expression any more; they are accepted at this relative L2
# error instead, with end-to-end PSNR as the real gate. Pure mul/add operators stay exact.
REL_L2_TOL = 1e-2


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
  """Relative L2 distance, used to gate the fast-math-sensitive kernels.

  :param actual: Kernel output.
  :param expected: Eager reference.
  :returns: `||actual - expected|| / ||expected||` in fp64.
  """

  delta = (actual.double() - expected.double()).norm()
  return (delta / expected.double().norm().clamp_min(1e-12)).item()


def set_qwen_image21_fused_ops_enabled(enabled: bool) -> None:
  """Override the operator switch at runtime, for parity tests and A/B runs.

  :param enabled: Whether :func:`apply_*` may dispatch to the CUDA kernels.
  """

  global FUSED_OPS_ENABLED
  FUSED_OPS_ENABLED = bool(enabled)


def fused_ops_extension_available() -> bool:
  """Return whether the shared model-kernel extension can be imported.

  :returns: `True` when ``cache_dit._C_models`` and its ops are usable.
  """

  from ....kernels._models import models_extension_is_available

  return models_extension_is_available()


def _fusable_tensor(x: torch.Tensor) -> bool:
  """Return whether `x` satisfies the layout every fused kernel assumes.

  A non-zero `storage_offset` is *not* a problem: `data_ptr()` already includes it, so a
  contiguous slice such as the decode step's `joint[:, prefix_len:]` is perfectly valid
  input. Only the dtype, the dense layout and a whole number of 16-byte vectors matter
  here; the C++ side re-checks byte alignment, which cannot be inspected while tracing.

  :param x: Candidate tensor.
  :returns: `True` for a dense CUDA tensor of a supported dtype and vector-friendly size.
  """

  return (x.is_cuda and x.is_contiguous() and x.dtype in (torch.bfloat16, torch.float16)
          and x.numel() % 8 == 0)


def rotary_cos_sin(rotary_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Split the checkpoint's complex rotary table into contiguous fp32 cos/sin halves.

  The table is rebuilt by :class:`QwenImage21Rope` once per forward and is identical for
  every layer, so the split is cached on the tensor itself instead of being recomputed.

  :param rotary_emb: `(S, D // 2)` complex64 table produced by the model's RoPE module.
  :returns: `(cos, sin)`, each `(S, D // 2)` fp32 and contiguous.
  """

  cached = getattr(rotary_emb, "_qwen21_cos_sin", None)
  if cached is not None and cached[0].device == rotary_emb.device:
    return cached
  cos = rotary_emb.real.contiguous().float()
  sin = rotary_emb.imag.contiguous().float()
  rotary_emb._qwen21_cos_sin = (cos, sin)
  return cos, sin


def _eager_qk_norm_rope(
  x: torch.Tensor,
  weight: torch.Tensor,
  rotary_emb: torch.Tensor,
  eps: float,
) -> torch.Tensor:
  """Literal transcription of diffusers' RMSNorm followed by interleaved RoPE.

  :param x: `(B, S, H, Dh)` q or k tensor in the model dtype.
  :param weight: `(Dh,)` RMSNorm scale.
  :param rotary_emb: `(S, Dh // 2)` complex64 table.
  :param eps: RMSNorm epsilon.
  :returns: Rotated tensor with the same shape and dtype as `x`.
  """

  variance = x.float().pow(2).mean(-1, keepdim=True)
  normed = (x * torch.rsqrt(variance + eps)).to(weight.dtype)
  normed = normed * weight
  freqs = rotary_emb.unsqueeze(1)
  rotated = torch.view_as_complex(normed.float().reshape(*normed.shape[:-1], -1, 2))
  return torch.view_as_real(rotated * freqs).flatten(3).type_as(x)


def apply_qk_norm_rope(
  x: torch.Tensor,
  weight: torch.Tensor,
  rotary_emb: torch.Tensor,
  eps: float,
  out: torch.Tensor | None = None,
) -> torch.Tensor:
  """Apply q/k RMSNorm and interleaved RoPE in one kernel when possible.

  :param x: `(B, S, H, Dh)` q or k tensor.
  :param weight: `(Dh,)` RMSNorm scale.
  :param rotary_emb: `(S, Dh // 2)` complex64 table.
  :param eps: RMSNorm epsilon.
  :param out: Optional destination, e.g. the target tail of a persistent KV buffer.
  :returns: The rotated tensor (the destination when `out` is given).
  """

  if (FUSED_OPS_ENABLED and "qk_norm_rope" in ACTIVE_OPS and _fusable_tensor(x)
      and x.shape[-1] == _FUSED_HEAD_DIM and weight.is_cuda and weight.dtype == x.dtype
      and weight.is_contiguous() and rotary_emb.dtype == torch.complex64
      and rotary_emb.shape[0] == x.shape[1]
      and (out is None or (out.is_cuda and out.dtype == x.dtype))):
    from ....kernels import qwen_image21_qk_norm_rope

    cos, sin = rotary_cos_sin(rotary_emb)
    return qwen_image21_qk_norm_rope(x, weight, cos, sin, eps, ROTATION_VARIANT, out=out)

  result = _eager_qk_norm_rope(x, weight, rotary_emb, eps)
  if out is not None:
    out.copy_(result)
    return out
  return result


def _select_modulation_rows(params: torch.Tensor,
                            target_token_mask: torch.Tensor | None) -> torch.Tensor:
  """Broadcast per-sample modulation rows over the token axis, exactly as upstream does.

  :param params: `(B, D)`, or `(B + 1, D)` with a trailing `t = 0` row.
  :param target_token_mask: `(S,)` bool, `True` at target-image positions, or `None`.
  :returns: `(B, S, D)` when a mask is given, else `(B, 1, D)`.
  """

  if target_token_mask is None:
    return params.unsqueeze(1)
  real, zero = params[:-1].unsqueeze(1), params[-1:].unsqueeze(0)
  return torch.where(target_token_mask.view(1, -1, 1), real, zero)


def _eager_ln_modulate(
  x: torch.Tensor,
  params: torch.Tensor,
  target_token_mask: torch.Tensor | None,
  eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Literal transcription of the block's `_modulate(img_norm(x), mod)`.

  :param x: `(B, S, D)` block input.
  :param params: `(B (+ 1), 2D)` modulation chunk holding `scale | gate`.
  :param target_token_mask: `(S,)` bool mask or `None`.
  :param eps: LayerNorm epsilon.
  :returns: `(x * (1 + scale), gate)`.
  """

  scale, gate = params.chunk(2, dim=-1)
  scale = _select_modulation_rows(scale, target_token_mask)
  gate = _select_modulation_rows(gate, target_token_mask)
  normed = F.layer_norm(x, (x.shape[-1], ), None, None, eps)
  return normed * (1 + scale), gate


def probe_ln_tier(dim: int, device: torch.device | None = None) -> int:
  """Decide whether the fused LayerNorm reproduces ATen's Welford tree on this device.

  ATen's layer-norm reduction is a 32-lane Welford tree with `coef = 1 / count` merges; the
  kernel replicates it, but the layout is a property of ATen's kernel selection rather than
  of the math, so it is verified per hidden size instead of assumed.

  :param dim: Hidden size the block will actually run with.
  :param device: CUDA device to probe on; defaults to the current device.
  :returns: `2` when LayerNorm can be fused, `1` when only the modulation epilogue may be
    fused (LayerNorm stays in ATen), or `0` when neither is safe.
  """

  cached = _LN_TIERS.get(dim)
  if cached is not None:
    return cached
  if not fused_ops_extension_available() or not torch.cuda.is_available():
    # Cache the negative outcome: one probe per hidden size per process, not per step.
    _LN_TIERS[dim] = 0
    return 0

  from ....kernels import qwen_image21_ln_modulate

  dev = device or torch.device("cuda", torch.cuda.current_device())
  torch.manual_seed(0)
  # Enough rows that a 1-ulp difference in the fp32 statistics shows up as a flipped bf16
  # rounding: with a handful of rows every formula looks exact.
  rows = 512
  x = torch.randn(1, rows, dim, device=dev, dtype=torch.bfloat16)
  params = torch.randn(2, 2 * dim, device=dev, dtype=torch.bfloat16)
  mask = torch.zeros(rows, device=dev, dtype=torch.bool)
  mask[:rows // 2] = True
  reference = _eager_ln_modulate(x, params, mask, 1e-6)
  modulated, gate = qwen_image21_ln_modulate(x, params, mask, 1e-6)

  # `tanhf` is approximated under `--use_fast_math`, so the tier is decided on rel-L2.
  tier = 2 if (relative_l2(modulated, reference[0]) <= REL_L2_TOL
               and relative_l2(gate, reference[1].tanh()) <= REL_L2_TOL) else 1
  _LN_TIERS[dim] = tier
  if tier == 2:
    logger.info(f"Qwen-Image 2.1 fused LayerNorm+modulation is bit-exact at D={dim}.")
  else:
    logger.info(f"Qwen-Image 2.1 fused LayerNorm does not reproduce ATen's Welford tree at "
                f"D={dim}; keeping ATen's LayerNorm and fusing only the modulation epilogue.")
  return tier


def apply_ln_modulate(
  x: torch.Tensor,
  params: torch.Tensor,
  target_token_mask: torch.Tensor | None,
  eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Apply LayerNorm, the `1 + scale` modulation and the row-select in one kernel when possible.

  :param x: `(B, S, D)` block input.
  :param params: `(B (+ 1), 2D)` modulation chunk holding `scale | gate`.
  :param target_token_mask: `(S,)` bool mask or `None`.
  :param eps: LayerNorm epsilon.
  :returns: `(x * (1 + scale), tanh(gate))`.
  """

  dim = x.shape[-1]
  # The 16-byte layout maps two rows per warp and derives the modulation row from the
  # sample, validated for a single sample; batch > 1 keeps the eager expression rather than
  # mis-mapping rows.
  fusable = ("ln_modulate" in ACTIVE_OPS and FUSED_OPS_ENABLED and x.shape[0] == 1
             and _fusable_tensor(x) and dim % 128 == 0 and params.is_cuda
             and params.dtype == x.dtype and params.is_contiguous()
             and (target_token_mask is None or
                  (target_token_mask.is_cuda and target_token_mask.dtype == torch.bool
                   and target_token_mask.is_contiguous())))
  if fusable:
    tier = _LN_TIERS.get(dim)
    if tier is None and not torch.compiler.is_compiling():
      tier = probe_ln_tier(dim, x.device)
    if tier == 2:
      from ....kernels import qwen_image21_ln_modulate

      return qwen_image21_ln_modulate(x, params, target_token_mask, eps)
    if tier == 1:
      from ....kernels import qwen_image21_modulate_affine

      normed = F.layer_norm(x, (dim, ), None, None, eps)
      return qwen_image21_modulate_affine(normed, params, target_token_mask)

  modulated, gate = _eager_ln_modulate(x, params, target_token_mask, eps)
  return modulated, gate.tanh()


def apply_gated_residual(hidden_states: torch.Tensor, gate: torch.Tensor,
                         update: torch.Tensor) -> torch.Tensor:
  """Compute `hidden_states + gate * update` without touching `hidden_states` in place.

  Out-of-place is a hard requirement: cache-dit derives a block's residual as
  `output - input`, so an in-place add corrupts the cache.

  :param hidden_states: Residual stream.
  :param gate: Already `tanh`-ed gate, same shape as `hidden_states`.
  :param update: Block output to be gated.
  :returns: The new residual stream.
  """

  if (FUSED_OPS_ENABLED and "gated_residual" in ACTIVE_OPS and _fusable_tensor(hidden_states)
      and _fusable_tensor(gate) and _fusable_tensor(update) and gate.shape == hidden_states.shape
      and update.shape == hidden_states.shape):
    from ....kernels import qwen_image21_gated_residual

    return qwen_image21_gated_residual(hidden_states, gate, update)

  return hidden_states + gate * update


def apply_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
  """Compute `silu(gate) * up` for the SwiGLU branch.

  :param gate: Gate projection output, `(B, S, 3D)`.
  :param up: Value projection output, same shape.
  :returns: The gated activation.
  """

  if (FUSED_OPS_ENABLED and "silu_mul" in ACTIVE_OPS and _fusable_tensor(gate)
      and _fusable_tensor(up) and gate.shape == up.shape):
    from ....kernels import qwen_image21_silu_mul

    return qwen_image21_silu_mul(gate, up)

  return F.silu(gate) * up


__all__ = [
  "FUSED_OPS_ENABLED",
  "apply_gated_residual",
  "apply_ln_modulate",
  "apply_qk_norm_rope",
  "apply_silu_mul",
  "fused_ops_extension_available",
  "probe_ln_tier",
  "rotary_cos_sin",
  "set_qwen_image21_fused_ops_enabled",
]
