import torch
from typing import Optional

from .functor_base import PatchFunctor
from ...logger import init_logger

logger = init_logger(__name__)

try:
  from diffusers.models.transformers.transformer_cosmos3 import (
    Cosmos3OmniTransformer,
    Cosmos3VLTextMoTDecoderLayer,
  )
except ImportError:
  raise ImportError("Cosmos3OmniPatchFunctor requires diffusers>=0.39.dev (Cosmos3). "
                    "Please install diffusers from source.")


class Cosmos3OmniPatchFunctor(PatchFunctor):
  """Adapt Cosmos3OmniTransformer (MoT: joint und/gen dual-stream) to cache-dit.

  Cosmos3's decoder layers take/return ``(und_seq, gen_seq)`` — the causal
  text (understanding) stream first and the generation (vision/sound/action)
  stream second. cache-dit's DBCache computes its cache decision on the FIRST
  positional arg (``hidden_states``). The und stream is (near-)static across
  denoising steps (text embeddings do not change with the timestep), so using
  it for the residual-diff decision would make every step a cache hit and
  break generation.

  This functor therefore rewires the block contract to
  ``(hidden_states=gen_seq, encoder_hidden_states=und_seq)`` —
  ForwardPattern.Pattern_0 — so the cache decision tracks the DENOISED
  stream, and patches the transformer forward's layer loop accordingly.
  """

  def _apply(
    self,
    transformer: Cosmos3OmniTransformer,
    **kwargs,
  ) -> Cosmos3OmniTransformer:

    for layer in transformer.layers:
      assert isinstance(layer, Cosmos3VLTextMoTDecoderLayer)
      layer.forward = __patch_layer_forward__.__get__(layer)

    transformer.forward = __patch_transformer_forward__.__get__(transformer)
    transformer._is_patched = True

    cls_name = transformer.__class__.__name__
    logger.warning(f"Applied {self.__class__.__name__} for {cls_name}.")

    return transformer


def __patch_layer_forward__(
  self: Cosmos3VLTextMoTDecoderLayer,
  hidden_states: torch.Tensor,  # gen stream (denoised tokens)
  encoder_hidden_states: torch.Tensor,  # und stream (text prefix)
  rotary_emb: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
  und_seq, gen_seq = encoder_hidden_states, hidden_states

  und_norm = self.input_layernorm(und_seq)
  gen_norm = self.input_layernorm_moe_gen(gen_seq)

  und_attn_out, gen_attn_out = self.self_attn(und_norm, gen_norm, rotary_emb)
  residual_und = und_seq + und_attn_out
  residual_gen = gen_seq + gen_attn_out

  mlp_out_und = self.mlp(self.post_attention_layernorm(residual_und))
  mlp_out_gen = self.mlp_moe_gen(self.post_attention_layernorm_moe_gen(residual_gen))

  # (hidden_states, encoder_hidden_states) == (gen, und) — Pattern_0
  return residual_gen + mlp_out_gen, residual_und + mlp_out_und


def __patch_transformer_forward__(
  self: Cosmos3OmniTransformer,
  input_ids: torch.Tensor,
  text_indexes: torch.Tensor,
  position_ids: torch.Tensor,
  und_len: int,
  sequence_length: int,
  vision_tokens: list[torch.Tensor],
  vision_token_shapes: list[tuple[int, int, int]],
  vision_sequence_indexes: torch.Tensor,
  vision_mse_loss_indexes: torch.Tensor,
  vision_timesteps: torch.Tensor,
  vision_noisy_frame_indexes: list[torch.Tensor],
  sound_tokens: Optional[list[torch.Tensor]] = None,
  sound_token_shapes: Optional[list[tuple[int, int, int]]] = None,
  sound_sequence_indexes: Optional[torch.Tensor] = None,
  sound_mse_loss_indexes: Optional[torch.Tensor] = None,
  sound_timesteps: Optional[torch.Tensor] = None,
  sound_noisy_frame_indexes: Optional[list[torch.Tensor]] = None,
  action_tokens: Optional[list[torch.Tensor]] = None,
  action_token_shapes: Optional[list[tuple[int, int, int]]] = None,
  action_sequence_indexes: Optional[torch.Tensor] = None,
  action_mse_loss_indexes: Optional[torch.Tensor] = None,
  action_timesteps: Optional[torch.Tensor] = None,
  action_noisy_frame_indexes: Optional[list[torch.Tensor]] = None,
  action_domain_ids: Optional[list[torch.Tensor]] = None,
) -> tuple[list[torch.Tensor], Optional[list[torch.Tensor]], Optional[list[torch.Tensor]]]:
  # Identical to upstream diffusers Cosmos3OmniTransformer.forward except the
  # decoder-layer loop passes (gen_seq, und_seq) — see the functor docstring.
  has_sound = sound_tokens is not None and sound_sequence_indexes is not None
  has_action = action_tokens is not None and action_sequence_indexes is not None

  packed_text_embedding = self.embed_tokens(input_ids)
  target_dtype = packed_text_embedding.dtype
  hidden_states = packed_text_embedding.new_zeros(size=(sequence_length, self.config.hidden_size))
  hidden_states[text_indexes] = packed_text_embedding

  packed_tokens_vision, original_latent_shapes = self._patchify_and_pack_latents(vision_tokens)
  packed_tokens_vision = self.proj_in(packed_tokens_vision)
  timesteps_vision = vision_timesteps * self.config.timestep_scale
  packed_timestep_embeds_vision = self.time_embedder(self.time_proj(timesteps_vision))
  packed_timestep_embeds_vision = packed_timestep_embeds_vision.to(target_dtype)
  packed_tokens_vision = self._apply_timestep_embeds_to_noisy_tokens(
    packed_tokens=packed_tokens_vision,
    packed_timestep_embeds=packed_timestep_embeds_vision,
    noisy_frame_indexes=vision_noisy_frame_indexes,
    token_shapes=vision_token_shapes,
  )
  hidden_states[vision_sequence_indexes] = packed_tokens_vision

  if has_sound:
    packed_tokens_sound = self._pack_sound_latents(sound_tokens,
                                                   sound_token_shapes).to(target_dtype)
    packed_tokens_sound = self.audio_proj_in(packed_tokens_sound) + self.audio_modality_embed
    timesteps_sound = sound_timesteps * self.config.timestep_scale
    packed_timestep_embeds_sound = self.time_embedder(self.time_proj(timesteps_sound))
    packed_timestep_embeds_sound = packed_timestep_embeds_sound.to(target_dtype)
    packed_tokens_sound = self._apply_timestep_embeds_to_noisy_tokens(
      packed_tokens=packed_tokens_sound,
      packed_timestep_embeds=packed_timestep_embeds_sound,
      noisy_frame_indexes=sound_noisy_frame_indexes,
      token_shapes=sound_token_shapes,
    )
    hidden_states[sound_sequence_indexes] = packed_tokens_sound

  if has_action:
    packed_tokens_action, per_token_domain_ids = self._pack_action_latents(
      action_tokens, action_token_shapes, action_domain_ids)
    packed_tokens_action = packed_tokens_action.to(target_dtype)
    per_token_domain_ids = per_token_domain_ids.to(device=packed_tokens_action.device)
    packed_tokens_action = self.action_proj_in(packed_tokens_action, per_token_domain_ids)
    packed_tokens_action = packed_tokens_action + self.action_modality_embed
    if action_mse_loss_indexes.numel() > 0:
      timesteps_action = action_timesteps * self.config.timestep_scale
      packed_timestep_embeds_action = self.time_embedder(self.time_proj(timesteps_action))
      packed_timestep_embeds_action = packed_timestep_embeds_action.to(target_dtype)
      packed_tokens_action = self._apply_timestep_embeds_to_noisy_tokens(
        packed_tokens=packed_tokens_action,
        packed_timestep_embeds=packed_timestep_embeds_action,
        noisy_frame_indexes=action_noisy_frame_indexes,
        token_shapes=action_token_shapes,
      )
    hidden_states[action_sequence_indexes] = packed_tokens_action

  cos, sin = self.rotary_emb(
    position_ids=position_ids.unsqueeze(0) if position_ids.ndim == 1 else position_ids.unsqueeze(1),
    device=hidden_states.device,
    dtype=hidden_states.dtype,
  )
  cos = cos.squeeze(0)
  sin = sin.squeeze(0)

  und_seq = hidden_states[:und_len]
  gen_seq = hidden_states[und_len:]
  rotary_emb = (cos[:und_len], sin[:und_len], cos[und_len:], sin[und_len:])
  for decoder_layer in self.layers:
    # NOTE(cache-dit): (gen, und) ordering — the gen stream must be the
    # first positional arg so DBCache's residual-diff decision tracks the
    # denoised tokens, not the static text prefix.
    if torch.is_grad_enabled() and self.gradient_checkpointing:
      gen_seq, und_seq = self._gradient_checkpointing_func(decoder_layer.__call__, gen_seq, und_seq,
                                                           rotary_emb)
    else:
      gen_seq, und_seq = decoder_layer(gen_seq, und_seq, rotary_emb)
  und_out = self.norm(und_seq)
  gen_out = self.norm_moe_gen(gen_seq)
  last_hidden_state = torch.cat([und_out, gen_out], dim=0)

  preds_vision_packed = self.proj_out(last_hidden_state[vision_mse_loss_indexes])
  preds_vision = self._unpatchify_and_unpack_latents(
    preds_vision_packed,
    token_shapes_vision=vision_token_shapes,
    noisy_frame_indexes_vision=vision_noisy_frame_indexes,
    original_latent_shapes=original_latent_shapes,
  )

  preds_sound: Optional[list[torch.Tensor]] = None
  if has_sound:
    preds_sound_packed = self.audio_proj_out(last_hidden_state[sound_mse_loss_indexes])
    preds_sound = self._unpack_sound_latents(preds_sound_packed, sound_token_shapes,
                                             sound_noisy_frame_indexes)

  preds_action: Optional[list[torch.Tensor]] = None
  if has_action:
    per_noisy_domain_ids = [
      domain_id.reshape(1).expand(len(noisy_idxs))
      for domain_id, noisy_idxs in zip(action_domain_ids, action_noisy_frame_indexes)
    ]
    per_noisy_domain_ids = torch.cat(per_noisy_domain_ids,
                                     dim=0).to(device=last_hidden_state.device)
    preds_action_packed = self.action_proj_out(last_hidden_state[action_mse_loss_indexes],
                                               per_noisy_domain_ids)
    preds_action = self._unpack_action_latents(preds_action_packed, action_token_shapes,
                                               action_noisy_frame_indexes)

  return preds_vision, preds_sound, preds_action
