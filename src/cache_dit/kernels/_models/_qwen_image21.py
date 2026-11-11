"""Python bridge for the Qwen-Image 2.1 operators in ``cache_dit._C_models``.

Same six-section layout as :mod:`._boogu`, and the same shared extension: the C++ side
only contributes ``qwen_image21_*`` entries to the existing ``ops`` submodule, so the
loader, the failure cache and the "extension missing" message are reused rather than
duplicated.
"""

from typing import Optional

import torch

from ._boogu import _get_required_ops_module

__all__ = [
  "qwen_image21_gated_residual",
  "qwen_image21_ln_modulate",
  "qwen_image21_qk_norm_rope",
  "qwen_image21_silu_mul",
]


def _call_qk_norm_rope(
  x: torch.Tensor,
  weight: torch.Tensor,
  cos: torch.Tensor,
  sin: torch.Tensor,
  eps: float,
  variant: int,
  out: torch.Tensor,
) -> torch.Tensor:
  """Invoke the C++ out-param entry point.

  :param x: `(B, S, H, 128)` query or key tensor.
  :param weight: `(128,)` RMSNorm scale.
  :param cos: `(S, 64)` fp32 cosine table.
  :param sin: `(S, 64)` fp32 sine table.
  :param eps: RMSNorm epsilon.
  :param variant: FMA contraction pattern of the rotation, as probed by `fused_ops`.
  :param out: Destination buffer, possibly a KV-buffer tail with non-trivial strides.
  :returns: `out`.
  """

  _get_required_ops_module().qwen_image21_qk_norm_rope(x, weight, cos, sin, eps, variant, out)
  return out


def _call_ln_modulate(
  x: torch.Tensor,
  one_plus: torch.Tensor,
  mask: Optional[torch.Tensor],
  eps: float,
  x_mod: torch.Tensor,
) -> None:
  """Invoke the fused LayerNorm + modulation entry point.

  :param x: `(B, S, D)` block input.
  :param one_plus: `(B (+ 1), D)` folded `1 + scale` rows.
  :param mask: `(S,)` bool target-token mask, or `None`.
  :param eps: LayerNorm epsilon.
  :param x_mod: Destination for `x * (1 + scale)`.
  """

  _get_required_ops_module().qwen_image21_ln_modulate(x, one_plus, mask, eps, x_mod)


def _call_gated_residual(
  hidden_states: torch.Tensor,
  gate_rows: torch.Tensor,
  mask: Optional[torch.Tensor],
  update: torch.Tensor,
  out: torch.Tensor,
  seq_len: int,
) -> torch.Tensor:
  """Invoke the gated residual entry point.

  :param hidden_states: Residual stream.
  :param gate_rows: `(B (+ 1), D)` already `tanh`-ed gate rows.
  :param mask: `(S,)` bool target-token mask, or `None`.
  :param update: Block output.
  :param out: Destination buffer.
  :param seq_len: Sequence length, needed to select a gate row per token.
  :returns: `out`.
  """

  _get_required_ops_module().qwen_image21_gated_residual(hidden_states, gate_rows, mask, update,
                                                         out, seq_len)
  return out


def _call_silu_mul(
  gate: torch.Tensor,
  up: torch.Tensor,
  out: torch.Tensor,
) -> torch.Tensor:
  """Invoke the SwiGLU gate entry point.

  :param gate: Gate projection output.
  :param up: Value projection output.
  :param out: Destination buffer.
  :returns: `out`.
  """

  _get_required_ops_module().qwen_image21_silu_mul(gate, up, out)
  return out


torch.library.define(
  "cache_dit_cuda_ops::qwen_image21_qk_norm_rope",
  "(Tensor x, Tensor weight, Tensor cos, Tensor sin, float eps, int variant) -> Tensor",
)
torch.library.define(
  "cache_dit_cuda_ops::qwen_image21_ln_modulate",
  "(Tensor x, Tensor one_plus, Tensor? target_token_mask, float eps) -> Tensor",
)
torch.library.define(
  "cache_dit_cuda_ops::qwen_image21_gated_residual",
  "(Tensor hidden_states, Tensor gate_rows, Tensor? target_token_mask, Tensor update, int seq_len)"
  " -> Tensor",
)
torch.library.define(
  "cache_dit_cuda_ops::qwen_image21_silu_mul",
  "(Tensor gate, Tensor up) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::qwen_image21_qk_norm_rope", "CUDA")
def _qk_norm_rope_impl(
  x: torch.Tensor,
  weight: torch.Tensor,
  cos: torch.Tensor,
  sin: torch.Tensor,
  eps: float,
  variant: int,
) -> torch.Tensor:
  if x.dim() != 4:
    raise ValueError(f"qwen_image21_qk_norm_rope expects a 4D tensor, got {tuple(x.shape)}")
  return _call_qk_norm_rope(x, weight, cos, sin, eps, variant, torch.empty_like(x))


@torch.library.register_fake("cache_dit_cuda_ops::qwen_image21_qk_norm_rope")
def _qk_norm_rope_fake(
  x: torch.Tensor,
  weight: torch.Tensor,
  cos: torch.Tensor,
  sin: torch.Tensor,
  eps: float,
  variant: int,
) -> torch.Tensor:
  return torch.empty_like(x)


@torch.library.impl("cache_dit_cuda_ops::qwen_image21_ln_modulate", "CUDA")
def _ln_modulate_impl(
  x: torch.Tensor,
  one_plus: torch.Tensor,
  target_token_mask: Optional[torch.Tensor],
  eps: float,
) -> torch.Tensor:
  if x.dim() != 3:
    raise ValueError(f"qwen_image21_ln_modulate expects a 3D tensor, got {tuple(x.shape)}")
  x_mod = torch.empty_like(x)
  _call_ln_modulate(x, one_plus, target_token_mask, eps, x_mod)
  return x_mod


@torch.library.register_fake("cache_dit_cuda_ops::qwen_image21_ln_modulate")
def _ln_modulate_fake(
  x: torch.Tensor,
  one_plus: torch.Tensor,
  target_token_mask: Optional[torch.Tensor],
  eps: float,
) -> torch.Tensor:
  return torch.empty_like(x)


@torch.library.impl("cache_dit_cuda_ops::qwen_image21_gated_residual", "CUDA")
def _gated_residual_impl(
  hidden_states: torch.Tensor,
  gate_rows: torch.Tensor,
  target_token_mask: Optional[torch.Tensor],
  update: torch.Tensor,
  seq_len: int,
) -> torch.Tensor:
  return _call_gated_residual(hidden_states, gate_rows, target_token_mask, update,
                              torch.empty_like(hidden_states), seq_len)


@torch.library.register_fake("cache_dit_cuda_ops::qwen_image21_gated_residual")
def _gated_residual_fake(
  hidden_states: torch.Tensor,
  gate_rows: torch.Tensor,
  target_token_mask: Optional[torch.Tensor],
  update: torch.Tensor,
  seq_len: int,
) -> torch.Tensor:
  return torch.empty_like(hidden_states)


@torch.library.impl("cache_dit_cuda_ops::qwen_image21_silu_mul", "CUDA")
def _silu_mul_impl(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
  return _call_silu_mul(gate, up, torch.empty_like(gate))


@torch.library.register_fake("cache_dit_cuda_ops::qwen_image21_silu_mul")
def _silu_mul_fake(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
  return torch.empty_like(gate)


def qwen_image21_qk_norm_rope(
  x: torch.Tensor,
  weight: torch.Tensor,
  cos: torch.Tensor,
  sin: torch.Tensor,
  eps: float,
  variant: int = 0,
  out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
  """Per-head RMSNorm plus interleaved RoPE, bit-identical to the eager chain.

  :param x: `(B, S, H, 128)` query or key tensor, contiguous bf16 CUDA.
  :param weight: `(128,)` RMSNorm scale.
  :param cos: `(S, 64)` fp32 cosine table.
  :param sin: `(S, 64)` fp32 sine table.
  :param eps: RMSNorm epsilon.
  :param variant: FMA contraction pattern of the rotation, as probed by `fused_ops`.
  :param out: Optional destination, e.g. the target tail of a persistent KV buffer. The
    kernel itself is functional, so the caller copies when a destination is given.
  :returns: The rotated tensor (or `out`).
  """

  result = torch.ops.cache_dit_cuda_ops.qwen_image21_qk_norm_rope(x, weight, cos, sin, eps, variant)
  if out is not None and out.data_ptr() != result.data_ptr():
    out.copy_(result)
    return out
  return result


def qwen_image21_ln_modulate(
  x: torch.Tensor,
  one_plus: torch.Tensor,
  target_token_mask: Optional[torch.Tensor],
  eps: float,
) -> torch.Tensor:
  """Fused LayerNorm plus the folded `1 + scale` modulation.

  :param x: `(B, S, D)` block input.
  :param one_plus: `(B (+ 1), D)` folded `1 + scale` rows.
  :param target_token_mask: `(S,)` bool mask selecting the `t = 0` row, or `None`.
  :param eps: LayerNorm epsilon.
  :returns: `x * (1 + scale)`.
  """

  return torch.ops.cache_dit_cuda_ops.qwen_image21_ln_modulate(x, one_plus, target_token_mask, eps)


def qwen_image21_gated_residual(
  hidden_states: torch.Tensor,
  gate_rows: torch.Tensor,
  target_token_mask: Optional[torch.Tensor],
  update: torch.Tensor,
  seq_len: int,
) -> torch.Tensor:
  """Out-of-place `hidden_states + bf16(gate * update)` with a per-token gate row.

  :param hidden_states: `(1, S, D)` residual stream.
  :param gate_rows: `(1 or 2, D)` already `tanh`-ed gate rows.
  :param target_token_mask: `(S,)` bool mask selecting the `t = 0` row, or `None`.
  :param update: Block output.
  :param seq_len: Sequence length.
  :returns: The new residual stream.
  """

  return torch.ops.cache_dit_cuda_ops.qwen_image21_gated_residual(hidden_states, gate_rows,
                                                                  target_token_mask, update,
                                                                  seq_len)


def qwen_image21_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
  """`bf16(silu_fp32(gate)) * up`.

  :param gate: Gate projection output.
  :param up: Value projection output.
  :returns: The gated activation.
  """

  return torch.ops.cache_dit_cuda_ops.qwen_image21_silu_mul(gate, up)
