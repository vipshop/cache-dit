"""Python bridge for the FLUX.2-Klein operators in ``cache_dit._C_ops``.

Same six-section layout as :mod:`._qwen_image21`, and the same shared extension: the C++ side
only contributes ``flux2_klein_*`` entries to the existing ``ops`` submodule, so the loader,
the failure cache and the "extension missing" message are reused instead of duplicated.

Every operator here is functional at the ``torch.library`` level even though the C++ entry
points take out-params. That is deliberate: an operator declared as mutating an aliased buffer
loses its write when it is traced into a ``fullgraph=True`` graph on this torch build
(measured, 2.15), and these operators have to survive ``compile_repeated_blocks()``.
"""

import torch

from ..cuda_ext import _get_required_ops_module

__all__ = [
  "flux2_klein_gated_residual",
  "flux2_klein_ln_modulate",
  "flux2_klein_qk_norm_rope",
  "flux2_klein_qk_norm_rope_join",
  "flux2_klein_residual_ln_modulate",
  "flux2_klein_silu_mul",
]


def _call_qk_norm_rope(
  x: torch.Tensor,
  weight: torch.Tensor,
  cos: torch.Tensor,
  sin: torch.Tensor,
  rope_row_offset: int,
  eps: float,
  out: torch.Tensor,
  out_token_offset: int,
) -> torch.Tensor:
  """Invoke the C++ out-param entry point for one stream.

  :param x: `(1, S, H, 128)` query or key tensor.
  :param weight: `(128,)` RMSNorm scale.
  :param cos: `(rope_rows, 128)` fp32 cosine table.
  :param sin: `(rope_rows, 128)` fp32 sine table.
  :param rope_row_offset: First rotation row of this stream inside `cos`/`sin`.
  :param eps: RMSNorm epsilon.
  :param out: Destination `(S_total, H, 128)` buffer.
  :param out_token_offset: First destination token of this stream.
  :returns: `out`.
  """

  _get_required_ops_module().flux2_klein_qk_norm_rope(x, weight, cos, sin, rope_row_offset, eps,
                                                      out, out_token_offset)
  return out


def _call_qk_norm_rope_join(
  image: torch.Tensor,
  image_weight: torch.Tensor,
  text: torch.Tensor,
  text_weight: torch.Tensor,
  cos: torch.Tensor,
  sin: torch.Tensor,
  eps: float,
  out: torch.Tensor,
) -> torch.Tensor:
  """Invoke the C++ entry point that writes both streams into one buffer.

  :param image: `(1, S_img, H, 128)` image-stream query or key tensor.
  :param image_weight: `(128,)` scale of the image stream.
  :param text: `(1, S_txt, H, 128)` text-stream query or key tensor.
  :param text_weight: `(128,)` scale of the text stream.
  :param cos: `(S_txt + S_img, 128)` fp32 cosine table, text rows first.
  :param sin: `(S_txt + S_img, 128)` fp32 sine table, text rows first.
  :param eps: RMSNorm epsilon.
  :param out: Destination `(S_txt + S_img, H, 128)` buffer.
  :returns: `out`.
  """

  _get_required_ops_module().flux2_klein_qk_norm_rope_join(image, image_weight, text, text_weight,
                                                           cos, sin, eps, out)
  return out


def _call_ln_modulate(
  x: torch.Tensor,
  one_plus: torch.Tensor,
  shift: torch.Tensor,
  eps: float,
  out: torch.Tensor,
) -> None:
  """Invoke the fused LayerNorm + modulation entry point.

  :param x:`(B, S, D)` block input.
  :param one_plus:`(R, D)` folded `1 + scale` rows.
  :param shift:`(R, D)` shift rows.
  :param eps: LayerNorm epsilon.
  :param out: Destination for `(1 + scale) * layer_norm(x) + shift`.
  """

  _get_required_ops_module().flux2_klein_ln_modulate(x, one_plus, shift, eps, out)


def _call_residual_ln_modulate(
  hidden: torch.Tensor,
  gate: torch.Tensor,
  delta: torch.Tensor,
  one_plus: torch.Tensor,
  shift: torch.Tensor,
  eps: float,
  hidden_out: torch.Tensor,
  mod_out: torch.Tensor,
) -> None:
  """Invoke the fused residual add + LayerNorm + modulation entry point.

  :param hidden:`(B, S, D)` residual stream.
  :param gate:`(R, D)` gate rows.
  :param delta:`(B, S, D)` block output to be gated.
  :param one_plus:`(R, D)` folded `1 + scale` rows.
  :param shift:`(R, D)` shift rows.
  :param eps: LayerNorm epsilon.
  :param hidden_out: Destination for the updated residual stream.
  :param mod_out: Destination for the modulated normalisation.
  """

  _get_required_ops_module().flux2_klein_residual_ln_modulate(hidden, gate, delta, one_plus, shift,
                                                              eps, hidden_out, mod_out)


def _call_gated_residual(
  hidden: torch.Tensor,
  gate: torch.Tensor,
  update: torch.Tensor,
  out: torch.Tensor,
) -> torch.Tensor:
  """Invoke the gated residual entry point.

  :param hidden: `(1, S, D)` residual stream.
  :param gate: `(D,)` gate row.
  :param update: `(1, S, D)` block output.
  :param out: Destination buffer.
  :returns: `out`.
  """

  _get_required_ops_module().flux2_klein_gated_residual(hidden, gate, update, out)
  return out


def _call_silu_mul(gate_up: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
  """Invoke the SwiGLU entry point.

  :param gate_up: `(B, S, 2D)` packed gate and value projections.
  :param out: Destination buffer, `(B, S, D)`.
  :returns: `out`.
  """

  _get_required_ops_module().flux2_klein_silu_mul(gate_up, out)
  return out


torch.library.define(
  "cache_dit_cuda_ops::flux2_klein_qk_norm_rope",
  "(Tensor x, Tensor weight, Tensor cos, Tensor sin, int rope_row_offset, float eps,"
  " int out_token_offset) -> Tensor",
)
torch.library.define(
  "cache_dit_cuda_ops::flux2_klein_qk_norm_rope_join",
  "(Tensor image, Tensor image_weight, Tensor text, Tensor text_weight, Tensor cos, Tensor sin,"
  " float eps) -> Tensor",
)
torch.library.define(
  "cache_dit_cuda_ops::flux2_klein_ln_modulate",
  "(Tensor x, Tensor one_plus, Tensor shift, float eps) -> Tensor",
)
torch.library.define(
  "cache_dit_cuda_ops::flux2_klein_residual_ln_modulate",
  "(Tensor hidden, Tensor gate, Tensor delta, Tensor one_plus, Tensor shift, float eps)"
  " -> (Tensor, Tensor)",
)
torch.library.define(
  "cache_dit_cuda_ops::flux2_klein_gated_residual",
  "(Tensor hidden, Tensor gate, Tensor update) -> Tensor",
)
torch.library.define(
  "cache_dit_cuda_ops::flux2_klein_silu_mul",
  "(Tensor gate_up) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::flux2_klein_qk_norm_rope", "CUDA")
def _qk_norm_rope_impl(
  x: torch.Tensor,
  weight: torch.Tensor,
  cos: torch.Tensor,
  sin: torch.Tensor,
  rope_row_offset: int,
  eps: float,
  out_token_offset: int,
) -> torch.Tensor:
  if x.dim() != 4:
    raise ValueError(f"flux2_klein_qk_norm_rope expects a 4D tensor, got {tuple(x.shape)}")
  total = out_token_offset + x.shape[1]
  out = torch.empty((total, x.shape[2], x.shape[3]), dtype=x.dtype, device=x.device)
  _call_qk_norm_rope(x, weight, cos, sin, rope_row_offset, eps, out, out_token_offset)
  return out


@torch.library.register_fake("cache_dit_cuda_ops::flux2_klein_qk_norm_rope")
def _qk_norm_rope_fake(
  x: torch.Tensor,
  weight: torch.Tensor,
  cos: torch.Tensor,
  sin: torch.Tensor,
  rope_row_offset: int,
  eps: float,
  out_token_offset: int,
) -> torch.Tensor:
  total = out_token_offset + x.shape[1]
  return torch.empty((total, x.shape[2], x.shape[3]), dtype=x.dtype, device=x.device)


@torch.library.impl("cache_dit_cuda_ops::flux2_klein_qk_norm_rope_join", "CUDA")
def _qk_norm_rope_join_impl(
  image: torch.Tensor,
  image_weight: torch.Tensor,
  text: torch.Tensor,
  text_weight: torch.Tensor,
  cos: torch.Tensor,
  sin: torch.Tensor,
  eps: float,
) -> torch.Tensor:
  total = text.shape[1] + image.shape[1]
  out = torch.empty((total, image.shape[2], image.shape[3]), dtype=image.dtype, device=image.device)
  _call_qk_norm_rope_join(image, image_weight, text, text_weight, cos, sin, eps, out)
  return out


@torch.library.register_fake("cache_dit_cuda_ops::flux2_klein_qk_norm_rope_join")
def _qk_norm_rope_join_fake(
  image: torch.Tensor,
  image_weight: torch.Tensor,
  text: torch.Tensor,
  text_weight: torch.Tensor,
  cos: torch.Tensor,
  sin: torch.Tensor,
  eps: float,
) -> torch.Tensor:
  total = text.shape[1] + image.shape[1]
  return torch.empty((total, image.shape[2], image.shape[3]),
                     dtype=image.dtype,
                     device=image.device)


@torch.library.impl("cache_dit_cuda_ops::flux2_klein_ln_modulate", "CUDA")
def _ln_modulate_impl(
  x: torch.Tensor,
  one_plus: torch.Tensor,
  shift: torch.Tensor,
  eps: float,
) -> torch.Tensor:
  if x.dim() != 3:
    raise ValueError(f"flux2_klein_ln_modulate expects a 3D tensor, got {tuple(x.shape)}")
  out = torch.empty_like(x)
  _call_ln_modulate(x, one_plus, shift, eps, out)
  return out


@torch.library.register_fake("cache_dit_cuda_ops::flux2_klein_ln_modulate")
def _ln_modulate_fake(
  x: torch.Tensor,
  one_plus: torch.Tensor,
  shift: torch.Tensor,
  eps: float,
) -> torch.Tensor:
  return torch.empty_like(x)


@torch.library.impl("cache_dit_cuda_ops::flux2_klein_residual_ln_modulate", "CUDA")
def _residual_ln_modulate_impl(
  hidden: torch.Tensor,
  gate: torch.Tensor,
  delta: torch.Tensor,
  one_plus: torch.Tensor,
  shift: torch.Tensor,
  eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  hidden_out = torch.empty_like(hidden)
  mod_out = torch.empty_like(hidden)
  _call_residual_ln_modulate(hidden, gate, delta, one_plus, shift, eps, hidden_out, mod_out)
  return hidden_out, mod_out


@torch.library.register_fake("cache_dit_cuda_ops::flux2_klein_residual_ln_modulate")
def _residual_ln_modulate_fake(
  hidden: torch.Tensor,
  gate: torch.Tensor,
  delta: torch.Tensor,
  one_plus: torch.Tensor,
  shift: torch.Tensor,
  eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  return torch.empty_like(hidden), torch.empty_like(hidden)


@torch.library.impl("cache_dit_cuda_ops::flux2_klein_gated_residual", "CUDA")
def _gated_residual_impl(
  hidden: torch.Tensor,
  gate: torch.Tensor,
  update: torch.Tensor,
) -> torch.Tensor:
  return _call_gated_residual(hidden, gate, update, torch.empty_like(hidden))


@torch.library.register_fake("cache_dit_cuda_ops::flux2_klein_gated_residual")
def _gated_residual_fake(
  hidden: torch.Tensor,
  gate: torch.Tensor,
  update: torch.Tensor,
) -> torch.Tensor:
  return torch.empty_like(hidden)


@torch.library.impl("cache_dit_cuda_ops::flux2_klein_silu_mul", "CUDA")
def _silu_mul_impl(gate_up: torch.Tensor) -> torch.Tensor:
  shape = list(gate_up.shape)
  shape[-1] = shape[-1] // 2
  out = torch.empty(shape, dtype=gate_up.dtype, device=gate_up.device)
  return _call_silu_mul(gate_up, out)


@torch.library.register_fake("cache_dit_cuda_ops::flux2_klein_silu_mul")
def _silu_mul_fake(gate_up: torch.Tensor) -> torch.Tensor:
  shape = list(gate_up.shape)
  shape[-1] = shape[-1] // 2
  return torch.empty(shape, dtype=gate_up.dtype, device=gate_up.device)


def flux2_klein_qk_norm_rope(
  x: torch.Tensor,
  weight: torch.Tensor,
  cos: torch.Tensor,
  sin: torch.Tensor,
  rope_row_offset: int = 0,
  eps: float = 1e-6,
  out_token_offset: int = 0,
) -> torch.Tensor:
  """Per-head RMSNorm plus interleaved RoPE, bit-identical to the eager chain.

  :param x: `(1, S, H, 128)` query or key tensor, contiguous bf16 CUDA.
  :param weight: `(128,)` scale of a `torch.nn.RMSNorm(head_dim)` module.
  :param cos: `(rope_rows, 128)` fp32 cosine table.
  :param sin: `(rope_rows, 128)` fp32 sine table.
  :param rope_row_offset: First rotation row of this stream inside `cos`/`sin`.
  :param eps: RMSNorm epsilon.
  :param out_token_offset: First destination token, for writing into a prefix region.
  :returns: `(out_token_offset + S, H, 128)` rotated tensor.
  """

  return torch.ops.cache_dit_cuda_ops.flux2_klein_qk_norm_rope(x, weight, cos, sin, rope_row_offset,
                                                               eps, out_token_offset)


def flux2_klein_qk_norm_rope_join(
  image: torch.Tensor,
  image_weight: torch.Tensor,
  text: torch.Tensor,
  text_weight: torch.Tensor,
  cos: torch.Tensor,
  sin: torch.Tensor,
  eps: float = 1e-6,
) -> torch.Tensor:
  """Normalise and rotate both attention streams into one joint buffer.

  Equivalent to ``torch.cat([norm_rope(text), norm_rope(image)], dim=1)``, but the joint
  tensor is written once instead of being allocated, filled and then copied by a cat.

  :param image: `(1, S_img, H, 128)` image-stream tensor.
  :param image_weight: `(128,)` scale of the image stream.
  :param text: `(1, S_txt, H, 128)` text-stream tensor.
  :param text_weight: `(128,)` scale of the text stream.
  :param cos: `(S_txt + S_img, 128)` fp32 cosine table, text rows first.
  :param sin: `(S_txt + S_img, 128)` fp32 sine table, text rows first.
  :param eps: RMSNorm epsilon.
  :returns: `(S_txt + S_img, H, 128)` joined tensor.
  """

  return torch.ops.cache_dit_cuda_ops.flux2_klein_qk_norm_rope_join(image, image_weight, text,
                                                                    text_weight, cos, sin, eps)


def flux2_klein_ln_modulate(
  x: torch.Tensor,
  one_plus: torch.Tensor,
  shift: torch.Tensor,
  eps: float = 1e-6,
) -> torch.Tensor:
  """`(1 + scale) * layer_norm(x) + shift` in one pass over `x`.

  :param x: `(B, S, D)` block input.
  :param one_plus: `(R, D)` folded `1 + scale` rows, `R` in `{1, B}`.
  :param shift: `(R, D)` shift rows.
  :param eps: LayerNorm epsilon.
  :returns: `(B, S, D)` modulated normalisation.
  """

  return torch.ops.cache_dit_cuda_ops.flux2_klein_ln_modulate(x, one_plus, shift, eps)


def flux2_klein_residual_ln_modulate(
  hidden: torch.Tensor,
  gate: torch.Tensor,
  delta: torch.Tensor,
  one_plus: torch.Tensor,
  shift: torch.Tensor,
  eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Residual add plus the normalisation that follows it, in one pass over `hidden`.

  :param hidden:`(B, S, D)` residual stream.
  :param gate:`(R, D)` gate rows.
  :param delta:`(B, S, D)` block output to be gated.
  :param one_plus:`(R, D)` folded `1 + scale` rows.
  :param shift:`(R, D)` shift rows.
  :param eps: LayerNorm epsilon.
  :returns:`(hidden + gate * delta, (1 + scale) * layer_norm(hidden_new) + shift)`.
  """

  return torch.ops.cache_dit_cuda_ops.flux2_klein_residual_ln_modulate(
    hidden, gate, delta, one_plus, shift, eps)


def flux2_klein_gated_residual(
  hidden: torch.Tensor,
  gate: torch.Tensor,
  update: torch.Tensor,
) -> torch.Tensor:
  """Out-of-place `hidden + gate * update` with a per-sample gate row.

  :param hidden:`(1, S, D)` residual stream.
  :param gate:`(D,)` gate row.
  :param update:`(1, S, D)` block output.
  :returns: The new residual stream.
  """

  return torch.ops.cache_dit_cuda_ops.flux2_klein_gated_residual(hidden, gate, update)


def flux2_klein_silu_mul(gate_up: torch.Tensor) -> torch.Tensor:
  """`silu(gate) * up` over the two halves of `gate_up`'s last dimension.

  :param gate_up:`(B, S, 2D)` packed gate and value projections.
  :returns:`(B, S, D)` gated activation.
  """

  return torch.ops.cache_dit_cuda_ops.flux2_klein_silu_mul(gate_up)
