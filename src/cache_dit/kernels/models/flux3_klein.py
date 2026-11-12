"""Python bridge for the FLUX.3 Klein operators in ``cache_dit._C_ops``.

Same six-section layout as :mod:`._qwen_image21`, and the same shared extension: the C++ side
only contributes ``flux3_klein_*`` entries to the existing ``ops`` submodule, so the loader, the
failure cache and the "extension missing" message are reused rather than duplicated.
"""

from typing import Optional

import torch

from ..cuda_ext import _get_required_ops_module

__all__ = [
  "flux3_klein_gated_residual",
  "flux3_klein_gated_residual_split",
  "flux3_klein_ln_modulate",
  "flux3_klein_ln_modulate_cat",
  "flux3_klein_qk_norm_rope",
]


def _call_qk_norm_rope(
  x: torch.Tensor,
  weight: torch.Tensor,
  cos: torch.Tensor,
  sin: torch.Tensor,
  eps: float,
  out: torch.Tensor,
) -> torch.Tensor:
  """Invoke the C++ out-param entry point.

  :param x: `(B, S, H, 128)` query or key tensor.
  :param weight: `(128,)` fp32 RMSNorm scale.
  :param cos: `(S, 128)` fp32 cosine table.
  :param sin: `(S, 128)` fp32 sine table.
  :param eps: RMSNorm epsilon.
  :param out: Destination buffer.
  :returns: `out`.
  """

  _get_required_ops_module().flux3_klein_qk_norm_rope(x, weight, cos, sin, eps, out)
  return out


def _call_ln_modulate(
  x: torch.Tensor,
  scale: torch.Tensor,
  shift: torch.Tensor,
  eps: float,
  out: torch.Tensor,
) -> None:
  """Invoke the fused LayerNorm + modulation entry point.

  :param x:`(B, S, D)` block input.
  :param scale:`(B, S, D)` or `(B, 1, D)` modulation scale.
  :param shift:`(B, S, D)` or `(B, 1, D)` modulation shift.
  :param eps: LayerNorm epsilon.
  :param out: Destination for the modulated input.
  """

  _get_required_ops_module().flux3_klein_ln_modulate(x, scale, shift, eps, out)


def _call_ln_modulate_cat(
  hidden_states: torch.Tensor,
  encoder_hidden_states: torch.Tensor,
  scale: torch.Tensor,
  shift: torch.Tensor,
  context_scale: torch.Tensor,
  context_shift: torch.Tensor,
  eps: float,
  out: torch.Tensor,
) -> torch.Tensor:
  """Invoke the two-stream concatenating entry point.

  :param hidden_states: `(B, Sc, D)` content tokens.
  :param encoder_hidden_states: `(B, Se, D)` text tokens.
  :param scale: Content modulation scale.
  :param shift: Content modulation shift.
  :param context_scale: Text modulation scale.
  :param context_shift: Text modulation shift.
  :param eps: LayerNorm epsilon.
  :param out: Destination, `(B, Se + Sc, D)`.
  :returns: `out`.
  """

  _get_required_ops_module().flux3_klein_ln_modulate_cat(hidden_states, encoder_hidden_states,
                                                         scale, shift, context_scale, context_shift,
                                                         eps, out)
  return out


def _call_gated_residual(
  hidden_states: torch.Tensor,
  gate: torch.Tensor,
  attn_output: torch.Tensor,
  ff_output: torch.Tensor,
  out: torch.Tensor,
) -> torch.Tensor:
  """Invoke the fused branch-sum + gated residual entry point.

  :param hidden_states: Residual stream.
  :param gate: `(B, S, D)` or `(B, 1, D)` modulation gate.
  :param attn_output: Attention branch output.
  :param ff_output: Feed-forward branch output.
  :param out: Destination buffer.
  :returns: `out`.
  """

  _get_required_ops_module().flux3_klein_gated_residual(hidden_states, gate, attn_output, ff_output,
                                                        out)
  return out


def _call_gated_residual_split(
  hidden_states: torch.Tensor,
  encoder_hidden_states: torch.Tensor,
  gate: torch.Tensor,
  context_gate: torch.Tensor,
  attn_output: torch.Tensor,
  ff_output: torch.Tensor,
  prefix_len: int,
  out_hidden: torch.Tensor,
  out_encoder: torch.Tensor,
) -> None:
  """Invoke the joint-block split entry point.

  :param hidden_states: `(B, Sc, D)` content residual stream.
  :param encoder_hidden_states: `(B, Se, D)` text residual stream.
  :param gate: Content modulation gate.
  :param context_gate: Text modulation gate.
  :param attn_output: `(B, Se + Sc, D)` attention output.
  :param ff_output: `(B, Se + Sc, D)` feed-forward output.
  :param prefix_len: `Se`, the number of text tokens at the front of the joint sequence.
  :param out_hidden: Destination for the content stream.
  :param out_encoder: Destination for the text stream.
  """

  _get_required_ops_module().flux3_klein_gated_residual_split(hidden_states, encoder_hidden_states,
                                                              gate, context_gate, attn_output,
                                                              ff_output, prefix_len, out_hidden,
                                                              out_encoder)


torch.library.define(
  "cache_dit_cuda_ops::flux3_klein_qk_norm_rope",
  "(Tensor x, Tensor weight, Tensor cos, Tensor sin, float eps) -> Tensor",
)
torch.library.define(
  "cache_dit_cuda_ops::flux3_klein_ln_modulate",
  "(Tensor x, Tensor scale, Tensor shift, float eps, Tensor? out) -> Tensor",
)
torch.library.define(
  "cache_dit_cuda_ops::flux3_klein_ln_modulate_cat",
  "(Tensor hidden_states, Tensor encoder_hidden_states, Tensor scale, Tensor shift, "
  "Tensor context_scale, Tensor context_shift, float eps) -> Tensor",
)
torch.library.define(
  "cache_dit_cuda_ops::flux3_klein_gated_residual",
  "(Tensor hidden_states, Tensor gate, Tensor attn_output, Tensor ff_output) -> Tensor",
)
torch.library.define(
  "cache_dit_cuda_ops::flux3_klein_gated_residual_split",
  "(Tensor hidden_states, Tensor encoder_hidden_states, Tensor gate, Tensor context_gate, "
  "Tensor attn_output, Tensor ff_output, int prefix_len) -> (Tensor, Tensor)",
)


@torch.library.impl("cache_dit_cuda_ops::flux3_klein_qk_norm_rope", "CUDA")
def _qk_norm_rope_impl(
  x: torch.Tensor,
  weight: torch.Tensor,
  cos: torch.Tensor,
  sin: torch.Tensor,
  eps: float,
) -> torch.Tensor:
  if x.dim() != 4:
    raise ValueError(f"flux3_klein_qk_norm_rope expects a 4D tensor, got {tuple(x.shape)}")
  return _call_qk_norm_rope(x, weight, cos, sin, eps, torch.empty_like(x))


@torch.library.register_fake("cache_dit_cuda_ops::flux3_klein_qk_norm_rope")
def _qk_norm_rope_fake(
  x: torch.Tensor,
  weight: torch.Tensor,
  cos: torch.Tensor,
  sin: torch.Tensor,
  eps: float,
) -> torch.Tensor:
  return torch.empty_like(x)


@torch.library.impl("cache_dit_cuda_ops::flux3_klein_ln_modulate", "CUDA")
def _ln_modulate_impl(
  x: torch.Tensor,
  scale: torch.Tensor,
  shift: torch.Tensor,
  eps: float,
  out: Optional[torch.Tensor],
) -> torch.Tensor:
  if x.dim() != 3:
    raise ValueError(f"flux3_klein_ln_modulate expects a 3D tensor, got {tuple(x.shape)}")
  destination = torch.empty_like(x) if out is None else out
  _call_ln_modulate(x, scale, shift, eps, destination)
  return destination


@torch.library.register_fake("cache_dit_cuda_ops::flux3_klein_ln_modulate")
def _ln_modulate_fake(
  x: torch.Tensor,
  scale: torch.Tensor,
  shift: torch.Tensor,
  eps: float,
  out: Optional[torch.Tensor],
) -> torch.Tensor:
  return torch.empty_like(x)


@torch.library.impl("cache_dit_cuda_ops::flux3_klein_ln_modulate_cat", "CUDA")
def _ln_modulate_cat_impl(
  hidden_states: torch.Tensor,
  encoder_hidden_states: torch.Tensor,
  scale: torch.Tensor,
  shift: torch.Tensor,
  context_scale: torch.Tensor,
  context_shift: torch.Tensor,
  eps: float,
) -> torch.Tensor:
  shape = (hidden_states.shape[0], hidden_states.shape[1] + encoder_hidden_states.shape[1],
           hidden_states.shape[2])
  combined = torch.empty(shape, dtype=hidden_states.dtype, device=hidden_states.device)
  return _call_ln_modulate_cat(hidden_states, encoder_hidden_states, scale, shift, context_scale,
                               context_shift, eps, combined)


@torch.library.register_fake("cache_dit_cuda_ops::flux3_klein_ln_modulate_cat")
def _ln_modulate_cat_fake(
  hidden_states: torch.Tensor,
  encoder_hidden_states: torch.Tensor,
  scale: torch.Tensor,
  shift: torch.Tensor,
  context_scale: torch.Tensor,
  context_shift: torch.Tensor,
  eps: float,
) -> torch.Tensor:
  shape = (hidden_states.shape[0], hidden_states.shape[1] + encoder_hidden_states.shape[1],
           hidden_states.shape[2])
  return torch.empty(shape, dtype=hidden_states.dtype, device=hidden_states.device)


@torch.library.impl("cache_dit_cuda_ops::flux3_klein_gated_residual", "CUDA")
def _gated_residual_impl(
  hidden_states: torch.Tensor,
  gate: torch.Tensor,
  attn_output: torch.Tensor,
  ff_output: torch.Tensor,
) -> torch.Tensor:
  return _call_gated_residual(hidden_states, gate, attn_output, ff_output,
                              torch.empty_like(hidden_states))


@torch.library.register_fake("cache_dit_cuda_ops::flux3_klein_gated_residual")
def _gated_residual_fake(
  hidden_states: torch.Tensor,
  gate: torch.Tensor,
  attn_output: torch.Tensor,
  ff_output: torch.Tensor,
) -> torch.Tensor:
  return torch.empty_like(hidden_states)


@torch.library.impl("cache_dit_cuda_ops::flux3_klein_gated_residual_split", "CUDA")
def _gated_residual_split_impl(
  hidden_states: torch.Tensor,
  encoder_hidden_states: torch.Tensor,
  gate: torch.Tensor,
  context_gate: torch.Tensor,
  attn_output: torch.Tensor,
  ff_output: torch.Tensor,
  prefix_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  out_hidden = torch.empty_like(hidden_states)
  out_encoder = torch.empty_like(encoder_hidden_states)
  _call_gated_residual_split(hidden_states, encoder_hidden_states, gate, context_gate, attn_output,
                             ff_output, prefix_len, out_hidden, out_encoder)
  return out_encoder, out_hidden


@torch.library.register_fake("cache_dit_cuda_ops::flux3_klein_gated_residual_split")
def _gated_residual_split_fake(
  hidden_states: torch.Tensor,
  encoder_hidden_states: torch.Tensor,
  gate: torch.Tensor,
  context_gate: torch.Tensor,
  attn_output: torch.Tensor,
  ff_output: torch.Tensor,
  prefix_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  return torch.empty_like(encoder_hidden_states), torch.empty_like(hidden_states)


def flux3_klein_qk_norm_rope(
  x: torch.Tensor,
  weight: torch.Tensor,
  cos: torch.Tensor,
  sin: torch.Tensor,
  eps: float,
  out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
  """Per-head fp32 RMSNorm plus interleaved RoPE, one bf16 rounding.

  :param x: `(B, S, H, 128)` query or key tensor, contiguous bf16 CUDA.
  :param weight: `(128,)` fp32 RMSNorm scale.
  :param cos: `(S, 128)` fp32 cosine table.
  :param sin: `(S, 128)` fp32 sine table.
  :param eps: RMSNorm epsilon.
  :param out: Optional destination; a copy is issued when it is not the returned buffer.
  :returns: The rotated tensor (or `out`).
  """

  result = torch.ops.cache_dit_cuda_ops.flux3_klein_qk_norm_rope(x, weight, cos, sin, eps)
  if out is not None and out.data_ptr() != result.data_ptr():
    out.copy_(result)
    return out
  return result


def flux3_klein_ln_modulate(
  x: torch.Tensor,
  scale: torch.Tensor,
  shift: torch.Tensor,
  eps: float,
  out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
  """Fused LayerNorm plus the `(1 + scale) * x + shift` modulation.

  :param x:`(B, S, D)` block input.
  :param scale:`(B, S, D)` or `(B, 1, D)` modulation scale.
  :param shift:`(B, S, D)` or `(B, 1, D)` modulation shift.
  :param eps: LayerNorm epsilon.
  :param out: Optional destination.
  :returns: The modulated normalized input.
  """

  return torch.ops.cache_dit_cuda_ops.flux3_klein_ln_modulate(x, scale, shift, eps, out)


def flux3_klein_ln_modulate_cat(
  hidden_states: torch.Tensor,
  encoder_hidden_states: torch.Tensor,
  scale: torch.Tensor,
  shift: torch.Tensor,
  context_scale: torch.Tensor,
  context_shift: torch.Tensor,
  eps: float,
) -> torch.Tensor:
  """Normalize and modulate both joint-block streams into one `[text, content]` buffer.

  :param hidden_states:`(B, Sc, D)` content tokens.
  :param encoder_hidden_states:`(B, Se, D)` text tokens.
  :param scale: Content modulation scale.
  :param shift: Content modulation shift.
  :param context_scale: Text modulation scale.
  :param context_shift: Text modulation shift.
  :param eps: LayerNorm epsilon.
  :returns:`(B, Se + Sc, D)` concatenated modulated tokens.
  """

  return torch.ops.cache_dit_cuda_ops.flux3_klein_ln_modulate_cat(hidden_states,
                                                                  encoder_hidden_states, scale,
                                                                  shift, context_scale,
                                                                  context_shift, eps)


def flux3_klein_gated_residual(
  hidden_states: torch.Tensor,
  gate: torch.Tensor,
  attn_output: torch.Tensor,
  ff_output: torch.Tensor,
) -> torch.Tensor:
  """Out-of-place `hidden_states + gate * (attn_output + ff_output)`.

  :param hidden_states: Residual stream.
  :param gate:`(B, S, D)` or `(B, 1, D)` modulation gate.
  :param attn_output: Attention branch output.
  :param ff_output: Feed-forward branch output.
  :returns: The new residual stream.
  """

  return torch.ops.cache_dit_cuda_ops.flux3_klein_gated_residual(hidden_states, gate, attn_output,
                                                                 ff_output)


def flux3_klein_gated_residual_split(
  hidden_states: torch.Tensor,
  encoder_hidden_states: torch.Tensor,
  gate: torch.Tensor,
  context_gate: torch.Tensor,
  attn_output: torch.Tensor,
  ff_output: torch.Tensor,
  prefix_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Joint-block branch sum, split and two gated residuals in one kernel.

  :param hidden_states: `(B, Sc, D)` content residual stream.
  :param encoder_hidden_states: `(B, Se, D)` text residual stream.
  :param gate: Content modulation gate.
  :param context_gate: Text modulation gate.
  :param attn_output: `(B, Se + Sc, D)` attention output.
  :param ff_output: `(B, Se + Sc, D)` feed-forward output.
  :param prefix_len: `Se`, the number of text tokens at the front of the joint sequence.
  :returns: `(encoder_hidden_states, hidden_states)` in that order.
  """

  return torch.ops.cache_dit_cuda_ops.flux3_klein_gated_residual_split(
    hidden_states, encoder_hidden_states, gate, context_gate, attn_output, ff_output, prefix_len)
