"""Fused forwards for the two FLUX.2-Klein block types.

The block classes and their module trees are left untouched: only each instance's `forward` is
rebound, so `_repeated_blocks = ["Flux2TransformerBlock", "Flux2SingleTransformerBlock"]`,
cache-dit's block wrapper and the forward-pattern signature checks keep seeing the class they
expect. The parameter lists below therefore have to match the diffusers signatures exactly,
positional order included, because `ForwardPattern.Pattern_1` / `Pattern_3` validate them.

Two fusions are installed here:

* `ln_modulate` replaces `(1 + scale) * layer_norm(x) + shift`. One pass over `x` instead of
  the statistics pass plus the apply pass, and the modulation rows are the folded `[R, D]`
  rows instead of a `(B, S, D)` broadcast.
* `residual_ln_modulate` additionally performs the block's residual add on the way in, so the
  residual stream is read once instead of three times. Every block runs it right before its
  second normalisation, which is why it moves more bytes than the first one.
"""

from typing import Any, Optional

import torch
from diffusers.models.transformers.transformer_flux2 import Flux2Modulation

from .fused_ops import (
  apply_gated_residual,
  apply_ln_modulate,
  apply_residual_ln_modulate,
  modulation_rows,
)


def fused_double_block_forward(
  self: torch.nn.Module,
  hidden_states: torch.Tensor,
  encoder_hidden_states: torch.Tensor,
  temb_mod_img: torch.Tensor,
  temb_mod_txt: torch.Tensor,
  image_rotary_emb: Optional[tuple] = None,
  joint_attention_kwargs: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Run one double-stream block with fused normalisation and residuals.

  Mirrors `Flux2TransformerBlock.forward` statement by statement; only the normalisation calls
  and the residual adds are replaced, and the attention call is passed through unchanged.

  :param self: The patched `Flux2TransformerBlock`.
  :param hidden_states: `(B, S_img, D)` image stream.
  :param encoder_hidden_states: `(B, S_txt, D)` text stream.
  :param temb_mod_img: `(B, 6 * D)` image-stream modulation chunk.
  :param temb_mod_txt: `(B, 6 * D)` text-stream modulation chunk.
  :param image_rotary_emb: `(cos, sin)` tables for the joint sequence.
  :param joint_attention_kwargs: Forwarded to the attention module.
  :returns: `(text stream, image stream)`.
  """

  joint_attention_kwargs = joint_attention_kwargs or {}
  dim = hidden_states.shape[-1]

  (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp,
                                     gate_mlp) = Flux2Modulation.split(temb_mod_img, 2)
  (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp,
                                           c_gate_mlp) = Flux2Modulation.split(temb_mod_txt, 2)

  norm_hidden_states = apply_ln_modulate(hidden_states, modulation_rows(1 + scale_msa, dim),
                                         modulation_rows(shift_msa, dim), self.norm1.eps)
  norm_encoder_hidden_states = apply_ln_modulate(encoder_hidden_states,
                                                 modulation_rows(1 + c_scale_msa, dim),
                                                 modulation_rows(c_shift_msa, dim),
                                                 self.norm1_context.eps)

  attention_outputs = self.attn(
    hidden_states=norm_hidden_states,
    encoder_hidden_states=norm_encoder_hidden_states,
    image_rotary_emb=image_rotary_emb,
    **joint_attention_kwargs,
  )
  attn_output, context_attn_output = attention_outputs

  hidden_states, norm_hidden_states = apply_residual_ln_modulate(
    hidden_states, modulation_rows(gate_msa, dim), attn_output, modulation_rows(1 + scale_mlp, dim),
    modulation_rows(shift_mlp, dim), self.norm2.eps)
  ff_output = self.ff(norm_hidden_states)
  hidden_states = apply_gated_residual(hidden_states, modulation_rows(gate_mlp, dim), ff_output)

  encoder_hidden_states, norm_encoder_hidden_states = apply_residual_ln_modulate(
    encoder_hidden_states, modulation_rows(c_gate_msa, dim), context_attn_output,
    modulation_rows(1 + c_scale_mlp, dim), modulation_rows(c_shift_mlp, dim),
    self.norm2_context.eps)
  context_ff_output = self.ff_context(norm_encoder_hidden_states)
  encoder_hidden_states = apply_gated_residual(encoder_hidden_states,
                                               modulation_rows(c_gate_mlp, dim), context_ff_output)

  if encoder_hidden_states.dtype == torch.float16:
    encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

  return encoder_hidden_states, hidden_states


def fused_single_block_forward(
  self: torch.nn.Module,
  hidden_states: torch.Tensor,
  encoder_hidden_states: Optional[torch.Tensor],
  temb_mod: torch.Tensor,
  image_rotary_emb: Optional[tuple] = None,
  joint_attention_kwargs: Optional[dict] = None,
  split_hidden_states: bool = False,
  text_seq_len: Optional[int] = None,
) -> Any:
  """Run one single-stream block with fused normalisation and residual.

  Mirrors `Flux2SingleTransformerBlock.forward`; the attention module still owns the fused
  QKV/MLP projection, the SwiGLU and the output projection.

  :param self: The patched `Flux2SingleTransformerBlock`.
  :param hidden_states: `(B, S, D)` joint stream, or the image stream when a text stream is
    passed separately.
  :param encoder_hidden_states: `(B, S_txt, D)` text stream, or `None` when already merged.
  :param temb_mod: `(B, 3 * D)` modulation chunk.
  :param image_rotary_emb: `(cos, sin)` tables for the joint sequence.
  :param joint_attention_kwargs: Forwarded to the attention module.
  :param split_hidden_states: Whether to return the two streams separately.
  :param text_seq_len: Length of the text prefix, taken from `encoder_hidden_states` when not
    given.
  :returns: The updated joint stream, or `(text stream, image stream)`.
  """

  if encoder_hidden_states is not None:
    text_seq_len = encoder_hidden_states.shape[1]
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

  dim = hidden_states.shape[-1]
  mod_shift, mod_scale, mod_gate = Flux2Modulation.split(temb_mod, 1)[0]

  norm_hidden_states = apply_ln_modulate(hidden_states, modulation_rows(1 + mod_scale, dim),
                                         modulation_rows(mod_shift, dim), self.norm.eps)

  joint_attention_kwargs = joint_attention_kwargs or {}
  attn_output = self.attn(
    hidden_states=norm_hidden_states,
    image_rotary_emb=image_rotary_emb,
    **joint_attention_kwargs,
  )

  hidden_states = apply_gated_residual(hidden_states, modulation_rows(mod_gate, dim), attn_output)
  if hidden_states.dtype == torch.float16:
    hidden_states = hidden_states.clip(-65504, 65504)

  if split_hidden_states:
    encoder_hidden_states, hidden_states = hidden_states[:, :
                                                         text_seq_len], hidden_states[:,
                                                                                      text_seq_len:]
    return encoder_hidden_states, hidden_states
  return hidden_states


def install_fused_modules(transformer: torch.nn.Module) -> None:
  """Bind the fused forwards on every block and install the attention seam.

  Idempotent. The attention processors are replaced per instance (never on the class), so the fusion
  does not leak to another FLUX.2 user in the same process.

  :param transformer: FLUX.2-Klein transformer whose modules should be rebound.
  """

  from . import fused_ops
  from .attention import install_attention_patch

  # This is the only place the operators are reached from, so it is also the only place that
  # turns them on: `CACHE_DIT_FUSED_OPS=0` is the sole opt-out, and a later
  # `set_flux2_klein_fused_ops_enabled()` call overrides the install.
  fused_ops.activate_fused_ops()

  for block in transformer.transformer_blocks:
    block.forward = fused_double_block_forward.__get__(block)
  for block in transformer.single_transformer_blocks:
    block.forward = fused_single_block_forward.__get__(block)

  # Read through the module, not a local binding: `activate_fused_ops()` has just written it.
  if fused_ops.FUSED_OPS_ENABLED and "qk_norm_rope" in fused_ops.ACTIVE_OPS:
    install_attention_patch(transformer)


__all__ = [
  "fused_double_block_forward",
  "fused_single_block_forward",
  "install_fused_modules",
]
