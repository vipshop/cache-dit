"""Patch MotifVideoTransformer3DModel — split the sliced single-stream loop into stable lists.

Motif-Video's forward iterates `single_transformer_blocks` twice through *slices*:
the encoder part `[: len - num_decoder_layers]` and the DDT decoder part
`[-num_decoder_layers:]`. cache-dit replaces a block list with a single
`ModuleList([CachedBlocks])` wrapper, so both slices would resolve to the same wrapper:
the encoder pass would run all 24 blocks and the decoder pass would run them again —
corrupting the cache bookkeeping. The DDT decoder pass also feeds a `decoder_hidden_states`
cloned right after patchify (full sequence) as its query stream, whose length no longer
matches the local (split) sequence under context parallelism.

The patch splits the list into two dedicated sub-module attributes and mirrors the upstream
forward over them. Two CP-only insertions keep the sequence layout consistent under Ulysses:
the joint attention mask is permuted into all-to-all order `[H_0, E_0, H_1, E_1, ...]`
(full length), and the decoder query stream is chunked to this rank's local segment.
"""

import torch
import torch.nn.functional as F
from typing import Any, Dict, Optional, Union

from .functor_base import PatchFunctor
from ...logger import init_logger

logger = init_logger(__name__)


class MotifVideoPatchFunctor(PatchFunctor):
  """Split Motif-Video's sliced single-stream loops into cacheable block lists."""

  def _apply(
    self,
    transformer: torch.nn.Module,
    **kwargs,
  ) -> torch.nn.Module:
    from diffusers.models.transformers.transformer_motif_video import (
      MotifVideoTransformer3DModel, )

    assert isinstance(transformer, MotifVideoTransformer3DModel)

    num_decoder = transformer.config.num_decoder_layers
    num_single = len(transformer.single_transformer_blocks)
    assert 0 <= num_decoder <= num_single, (
      f"num_decoder_layers {num_decoder} out of range for {num_single} single blocks")
    # Block objects are shared by reference: `modules()` de-duplicates them, so
    # offload / quantization / `compile_repeated_blocks` stay alias-safe. Only
    # `state_dict()` gains duplicate keys (inference is unaffected).
    transformer._motif_single_encoder_blocks = torch.nn.ModuleList(
      list(transformer.single_transformer_blocks[:num_single - num_decoder]))
    transformer._motif_single_decoder_blocks = torch.nn.ModuleList(
      list(transformer.single_transformer_blocks[num_single - num_decoder:]))

    transformer._orig_forward = transformer.forward
    transformer.forward = _patched_transformer_forward.__get__(transformer)
    if hasattr(transformer, "_old_forward"):
      # accelerate's offload dispatches through `_old_forward` captured before this patch.
      transformer._old_forward = transformer.forward
    transformer._is_patched = True
    return transformer


# Adapted from diffusers' Motif-Video transformer implementation:
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_motif_video.py
def _patched_transformer_forward(
  self: torch.nn.Module,
  hidden_states: torch.Tensor,
  timestep: torch.LongTensor,
  encoder_hidden_states: torch.Tensor,
  encoder_attention_mask: Optional[torch.Tensor] = None,
  image_embeds: Optional[torch.Tensor] = None,
  attention_kwargs: Optional[Dict[str, Any]] = None,
  return_dict: bool = True,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
  from diffusers.utils import USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
  from diffusers.models.modeling_outputs import Transformer2DModelOutput

  if attention_kwargs is not None:
    attention_kwargs = attention_kwargs.copy()
    lora_scale = attention_kwargs.pop("scale", 1.0)
  else:
    lora_scale = 1.0

  if USE_PEFT_BACKEND:
    scale_lora_layers(self, lora_scale)
  else:
    if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
      logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend "
                     "is ineffective.")

  batch_size, _, num_frames, height, width = hidden_states.shape
  p, p_t = self.config.patch_size, self.config.patch_size_t
  post_patch_num_frames = num_frames // p_t
  post_patch_height = height // p
  post_patch_width = width // p

  cp_config = getattr(self, "_cp_config", None)
  use_cp = cp_config is not None and (cp_config._world_size or 0) > 1

  # 1. RoPE (full sequence; the CP rope output-split hook hands each rank its image segment)
  image_rotary_emb = self.rope(hidden_states)

  # 2. Conditional embeddings
  temb = self.time_text_embed(timestep)
  hidden_states = self.x_embedder(hidden_states)
  encoder_hidden_states = self.context_embedder(encoder_hidden_states)

  # First frame conditioning: Image embeddings from vision encoder
  if image_embeds is not None:
    image_embeds = self.image_embedder(image_embeds)
    encoder_hidden_states = torch.cat([image_embeds, encoder_hidden_states], dim=1)
    if encoder_attention_mask is not None:
      image_mask = torch.ones(
        image_embeds.shape[0],
        image_embeds.shape[1],
        device=encoder_attention_mask.device,
        dtype=encoder_attention_mask.dtype,
      )
      encoder_attention_mask = torch.cat([image_mask, encoder_attention_mask], dim=1)

  # image_embed_seq_len: used by cross-attention blocks to slice text from encoder_hidden_states
  image_embed_seq_len = image_embeds.shape[1] if image_embeds is not None else 0

  if self.num_decoder_layers > 0:
    decoder_hidden_states = hidden_states.clone()

  if encoder_attention_mask is not None:
    attention_mask = F.pad(
      encoder_attention_mask.to(torch.bool),
      (hidden_states.shape[1], 0),
      value=True,
    )
    # NOTE: Permute attention_mask into all-to-all order under context parallelism.
    # For example, if world size = 2: [H, E] -> [H_0, E_0, H_1, E_1] (length unchanged).
    if use_cp:
      world_size = cp_config._world_size
      hidden_mask = attention_mask[:, :hidden_states.shape[1]]
      encoder_mask = attention_mask[:, hidden_states.shape[1]:]
      hidden_mask_splits = torch.chunk(hidden_mask, world_size, dim=1)
      encoder_mask_splits = torch.chunk(encoder_mask, world_size, dim=1)
      new_attention_mask_splits = []
      for i in range(world_size):
        new_attention_mask_splits.append(hidden_mask_splits[i])
        new_attention_mask_splits.append(encoder_mask_splits[i])
      attention_mask = torch.cat(new_attention_mask_splits, dim=1)
    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
  else:
    attention_mask = None

  # 3. Dual stream transformer blocks
  if torch.is_grad_enabled() and self.gradient_checkpointing:
    for block in self.transformer_blocks:
      hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
        block,
        hidden_states,
        encoder_hidden_states,
        temb,
        attention_mask,
        image_rotary_emb,
        image_embed_seq_len,
      )
  else:
    for block in self.transformer_blocks:
      hidden_states, encoder_hidden_states = block(hidden_states, encoder_hidden_states, temb,
                                                   attention_mask, image_rotary_emb,
                                                   image_embed_seq_len)

  # 4. Single stream transformer blocks (Encoder)
  for block in self._motif_single_encoder_blocks:
    hidden_states, encoder_hidden_states = (self._gradient_checkpointing_func(
      block,
      hidden_states,
      encoder_hidden_states,
      temb,
      attention_mask,
      image_rotary_emb,
      image_embed_seq_len,
    ) if torch.is_grad_enabled() and self.gradient_checkpointing else block(
      hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb,
      image_embed_seq_len))

  # 5. Single stream transformer blocks (Decoder)
  if self.num_decoder_layers > 0:
    # NOTE: the decoder query stream is cloned right after patchify (full sequence); under
    # context parallelism chunk it down to this rank's local segment so it matches the
    # already-local encoder stream.
    if use_cp:
      decoder_hidden_states = torch.chunk(decoder_hidden_states, cp_config._world_size,
                                          dim=1)[cp_config._rank]

    encoder_hidden_states = hidden_states
    attention_mask = None

    for block in self._motif_single_decoder_blocks:
      decoder_hidden_states, encoder_hidden_states = (self._gradient_checkpointing_func(
        block, decoder_hidden_states, encoder_hidden_states, temb, attention_mask,
        image_rotary_emb) if torch.is_grad_enabled() and self.gradient_checkpointing else block(
          decoder_hidden_states, encoder_hidden_states, temb, attention_mask, image_rotary_emb))

    hidden_states = decoder_hidden_states

  # 6. Output projection
  hidden_states = self.norm_out(hidden_states, temb)
  hidden_states = self.proj_out(hidden_states)

  hidden_states = hidden_states.reshape(
    batch_size,
    post_patch_num_frames,
    post_patch_height,
    post_patch_width,
    -1,
    p_t,
    p,
    p,
  )
  hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
  hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

  if USE_PEFT_BACKEND:
    unscale_lora_layers(self, lora_scale)

  if not return_dict:
    return (hidden_states, )

  return Transformer2DModelOutput(sample=hidden_states, )
