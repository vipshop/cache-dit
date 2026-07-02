"""Patch BriaFiboTransformer2DModel — every loop-body operation is absorbed into individual block
forwards, following the HunyuanDiTPatchFunctor pattern.

Pattern_1 cache passes ``hidden_states, encoder_hidden_states`` as positional
args 1-2, then ``*args, **kwargs`` from the transformer forward call.
"""

import torch
from typing import Any, Optional

from diffusers.models.transformers.transformer_bria_fibo import (
  BriaFiboTransformer2DModel,
  BriaFiboTransformerBlock,
  BriaFiboSingleTransformerBlock,
  Transformer2DModelOutput,
)

from .functor_base import PatchFunctor
from ...logger import init_logger

logger = init_logger(__name__)


class BriaFiboPatchFunctor(PatchFunctor):

  def _apply(
    self,
    transformer: BriaFiboTransformer2DModel,
    **kwargs,
  ) -> BriaFiboTransformer2DModel:
    half_dim = transformer.inner_dim // 2
    block_id = 0

    # Save originals to inject _ptls without cache interference.
    orig_double = list(transformer.transformer_blocks)
    orig_single = list(transformer.single_transformer_blocks)
    transformer._orig_double_blocks = orig_double
    transformer._orig_single_blocks = orig_single

    for block in orig_double:
      assert isinstance(block, BriaFiboTransformerBlock)
      block._orig_forward = block.forward
      block._half_dim = half_dim
      block._block_id = block_id
      block.forward = _patched_double_block_forward.__get__(block)
      block_id += 1

    for block in orig_single:
      assert isinstance(block, BriaFiboSingleTransformerBlock)
      block._orig_forward = block.forward
      block._half_dim = half_dim
      block._block_id = block_id
      block.forward = _patched_single_block_forward.__get__(block)
      block_id += 1

    transformer.forward = _patched_transformer_forward.__get__(transformer)
    transformer._is_patched = True
    return transformer


def _patched_double_block_forward(
  self: BriaFiboTransformerBlock,
  hidden_states: torch.Tensor,
  encoder_hidden_states: torch.Tensor,
  temb: torch.Tensor,
  image_rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
  joint_attention_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  ptl = self._ptls[self._block_id]
  encoder_hidden_states = torch.cat([encoder_hidden_states[:, :, :self._half_dim], ptl], dim=-1)
  return self._orig_forward(hidden_states, encoder_hidden_states, temb, image_rotary_emb,
                            joint_attention_kwargs)


def _single_block_orig_impl(
  self: BriaFiboSingleTransformerBlock,
  hidden_states: torch.Tensor,
  temb: torch.Tensor,
  image_rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
  joint_attention_kwargs: Optional[dict[str, Any]] = None,
) -> torch.Tensor:
  residual = hidden_states
  norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
  mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
  joint_attention_kwargs = joint_attention_kwargs or {}
  attn_output = self.attn(
    hidden_states=norm_hidden_states,
    image_rotary_emb=image_rotary_emb,
    **joint_attention_kwargs,
  )
  hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
  gate = gate.unsqueeze(1)
  hidden_states = gate * self.proj_out(hidden_states)
  hidden_states = residual + hidden_states
  if hidden_states.dtype == torch.float16:
    hidden_states = hidden_states.clip(-65504, 65504)
  return hidden_states


def _patched_single_block_forward(
  self: BriaFiboSingleTransformerBlock,
  hidden_states: torch.Tensor,
  encoder_hidden_states: torch.Tensor,
  temb: torch.Tensor,
  image_rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
  joint_attention_kwargs: Optional[dict[str, Any]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
  ptl = self._ptls[self._block_id]
  encoder_hidden_states = torch.cat([encoder_hidden_states[:, :, :self._half_dim], ptl], dim=-1)
  text_seq_len = encoder_hidden_states.shape[1]
  block_input = torch.cat([encoder_hidden_states, hidden_states], dim=1)
  block_output = _single_block_orig_impl(self, block_input, temb, image_rotary_emb,
                                         joint_attention_kwargs)
  encoder_hidden_states = block_output[:, :text_seq_len, ...]
  hidden_states = block_output[:, text_seq_len:, ...]
  return encoder_hidden_states, hidden_states


def _patched_transformer_forward(
  self: BriaFiboTransformer2DModel,
  hidden_states: torch.Tensor,
  encoder_hidden_states: torch.Tensor = None,
  text_encoder_layers: list = None,
  pooled_projections: torch.Tensor = None,
  timestep: torch.LongTensor = None,
  img_ids: torch.Tensor = None,
  txt_ids: torch.Tensor = None,
  guidance: torch.Tensor = None,
  joint_attention_kwargs: dict[str, Any] | None = None,
  return_dict: bool = True,
) -> torch.FloatTensor | Transformer2DModelOutput:
  hidden_states = self.x_embedder(hidden_states)
  timestep = timestep.to(hidden_states.dtype)
  if guidance is not None:
    guidance = guidance.to(hidden_states.dtype)
  else:
    guidance = None
  temb = self.time_embed(timestep, dtype=hidden_states.dtype)
  if guidance is not None:
    temb += self.guidance_embed(guidance, dtype=hidden_states.dtype)
  encoder_hidden_states = self.context_embedder(encoder_hidden_states)
  if len(txt_ids.shape) == 3:
    txt_ids = txt_ids[0]
  if len(img_ids.shape) == 3:
    img_ids = img_ids[0]
  ids = torch.cat((txt_ids, img_ids), dim=0)
  image_rotary_emb = self.pos_embed(ids)

  projected_text_layers = [
    self.caption_projection[i](text_encoder_layers[i]) for i in range(len(text_encoder_layers))
  ]
  # Inject projected text layers onto original blocks before cache wraps them.
  for block in self._orig_double_blocks:
    block._ptls = projected_text_layers
  for block in self._orig_single_blocks:
    block._ptls = projected_text_layers

  for index_block, block in enumerate(self.transformer_blocks):
    if torch.is_grad_enabled() and self.gradient_checkpointing:
      encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
        block, hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs)
    else:
      encoder_hidden_states, hidden_states = block(hidden_states,
                                                   encoder_hidden_states,
                                                   temb,
                                                   image_rotary_emb=image_rotary_emb,
                                                   joint_attention_kwargs=joint_attention_kwargs)

  for index_block, block in enumerate(self.single_transformer_blocks):
    if torch.is_grad_enabled() and self.gradient_checkpointing:
      encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
        block, hidden_states, encoder_hidden_states, temb, image_rotary_emb, joint_attention_kwargs)
    else:
      encoder_hidden_states, hidden_states = block(hidden_states,
                                                   encoder_hidden_states,
                                                   temb,
                                                   image_rotary_emb=image_rotary_emb,
                                                   joint_attention_kwargs=joint_attention_kwargs)

  hidden_states = self.norm_out(hidden_states, temb)
  output = self.proj_out(hidden_states)
  if not return_dict:
    return (output, )
  return Transformer2DModelOutput(sample=output)
