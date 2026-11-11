"""Fused forwards for the FLUX.3 Klein transformer blocks.

The block classes are left untouched: only each instance's `forward` is rebound. That keeps
`_repeated_blocks = ["Flux3ModeTransformerBlock", "Flux3SingleTransformerBlock"]`,
cache-dit's cached-blocks wrapper and diffusers' `check_forward_pattern` validation (which
inspects `inspect.signature(block.forward)`, both the parameter names and the `torch.Tensor`
count of the return annotation) seeing the contract they expect.

The two block shapes differ:

* `Flux3ModeTransformerBlock` runs one token mode through attention and FFN in parallel and
  returns a single tensor;
* `Flux3SingleTransformerBlock` rebuilds the joint `[text, content]` sequence inside the block,
  runs both branches over it, splits the result and returns `(encoder_hidden_states,
  hidden_states)` - which is `ForwardPattern.Pattern_1`.
"""

from typing import Any

import torch

from .fused_ops import (apply_gated_residual, apply_gated_residual_split, apply_ln_modulate,
                        apply_ln_modulate_cat)


def fused_mode_block_forward(
  self: torch.nn.Module,
  hidden_states: torch.Tensor,
  temb_mod: torch.Tensor,
  image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
  """Run one per-mode block with the fused modulate and gated residual.

  Same expression as upstream (`Flux3ModeTransformerBlock.forward`): the attention and FFN
  calls are passed through unchanged, so the attention processor, the FFN and the rotary table
  are exactly the ones the stock block would use.

  :param self: The patched block.
  :param hidden_states: `(B, S, D)` residual stream.
  :param temb_mod: `(B, S, 3 * D)` per-token modulation.
  :param image_rotary_emb: `(cos, sin)` fp32 tables of shape `(S, D_head)`.
  :returns: The updated residual stream.
  """

  mod_shift, mod_scale, mod_gate = torch.chunk(temb_mod, 3, dim=-1)
  norm_hidden_states = apply_ln_modulate(hidden_states, mod_scale, mod_shift, self.norm.eps)
  attn_output = self.attn(hidden_states=norm_hidden_states, image_rotary_emb=image_rotary_emb)
  ff_output = self.ff(norm_hidden_states)
  return apply_gated_residual(hidden_states, mod_gate, attn_output, ff_output)


def fused_single_block_forward(
  self: torch.nn.Module,
  hidden_states: torch.Tensor,
  encoder_hidden_states: torch.Tensor,
  temb_mod: torch.Tensor,
  temb_mod_context: torch.Tensor,
  image_rotary_emb: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
  """Run one joint block with the fused normalize+concat and gated residual.

  :param self: The patched block.
  :param hidden_states:`(B, Sc, D)` content tokens.
  :param encoder_hidden_states:`(B, Se, D)` text tokens.
  :param temb_mod:`(B, Sc, 3 * D)` content modulation.
  :param temb_mod_context:`(B, Se, 3 * D)` text modulation.
  :param image_rotary_emb:`(cos, sin)` fp32 tables over the joint sequence.
  :returns:`(encoder_hidden_states, hidden_states)`, as upstream returns them.
  """

  mod_shift, mod_scale, mod_gate = torch.chunk(temb_mod, 3, dim=-1)
  c_mod_shift, c_mod_scale, c_mod_gate = torch.chunk(temb_mod_context, 3, dim=-1)

  combined = apply_ln_modulate_cat(hidden_states, encoder_hidden_states, mod_scale, mod_shift,
                                   c_mod_scale, c_mod_shift, self.norm.eps)
  attn_output = self.attn(hidden_states=combined, image_rotary_emb=image_rotary_emb)
  ff_output = self.ff(combined)
  return apply_gated_residual_split(hidden_states, encoder_hidden_states, mod_gate, c_mod_gate,
                                    attn_output, ff_output, encoder_hidden_states.shape[1])


def _mode_blocks(transformer: torch.nn.Module) -> list[torch.nn.Module]:
  """Collect every per-mode block stack of the transformer.

  :param transformer: FLUX.3 transformer.
  :returns: The text stack followed by each content stack.
  """

  stacks = [transformer.txt_mode_blocks]
  stacks.extend(transformer.mode_blocks.values())
  return stacks


def install_fused_modules(transformer: torch.nn.Module) -> None:
  """Rebind the fused forwards on every block and swap in the fused attention processor.

  Idempotent. The attention processors are replaced per **instance** (never on the class), so a
  second FLUX.3 transformer in the same process keeps the stock processor.

  :param transformer: FLUX.3 transformer whose modules should be rebound.
  """

  from .attention import install_fused_processors

  for stack in _mode_blocks(transformer):
    for block in stack:
      block.forward = fused_mode_block_forward.__get__(block)
  for block in transformer.single_transformer_blocks:
    block.forward = fused_single_block_forward.__get__(block)

  install_fused_processors(transformer)

  if getattr(transformer, "_old_forward", None) is not None:
    # accelerate hooks captured the pre-patch forward and `mock_transformer` prefers
    # `_old_forward` when `_hf_hook` is set, so it has to follow every rewrite.
    transformer._old_forward = transformer.forward

  transformer._flux3_klein_fused_installed = True


def restore_fused_modules(transformer: torch.nn.Module) -> None:
  """Put the stock forwards and processors back, for parity tests and A/B runs.

  :param transformer: FLUX.3 transformer to restore.
  """

  from .attention import restore_processors

  for stack in _mode_blocks(transformer):
    for block in stack:
      block.__dict__.pop("forward", None)
  for block in transformer.single_transformer_blocks:
    block.__dict__.pop("forward", None)

  restore_processors(transformer)
  transformer._flux3_klein_fused_installed = False


def fused_modules_installed(transformer: torch.nn.Module) -> bool:
  """Return whether the fused forwards are currently bound.

  :param transformer: FLUX.3 transformer to inspect.
  :returns: `True` when every block runs the fused forward.
  """

  blocks = list(transformer.single_transformer_blocks)
  for stack in _mode_blocks(transformer):
    blocks.extend(stack)
  if not blocks:
    return False
  expected = {
    "Flux3ModeTransformerBlock": fused_mode_block_forward,
    "Flux3SingleTransformerBlock": fused_single_block_forward,
  }
  for block in blocks:
    want: Any = expected.get(type(block).__name__)
    if want is None or getattr(block.forward, "__func__", None) is not want:
      return False
  return True


__all__ = [
  "fused_mode_block_forward",
  "fused_modules_installed",
  "fused_single_block_forward",
  "install_fused_modules",
  "restore_fused_modules",
]
