import torch
from typing import Optional, Dict, Any

try:
  from diffusers.models.transformers.transformer_ltx2 import (
    LTX2VideoTransformer3DModel,
    AudioVisualModelOutput,
  )
except ImportError:
  raise ImportError("LTX2VideoTransformer3DModel is not available. "
                    "Please install the latest version of diffusers.")

from diffusers.utils import (
  USE_PEFT_BACKEND,
  scale_lora_layers,
  unscale_lora_layers,
)
from .functor_base import PatchFunctor
from ...logger import init_logger

logger = init_logger(__name__)


class LTX2PatchFunctor(PatchFunctor):

  def _apply(
    self,
    transformer: LTX2VideoTransformer3DModel,
    **kwargs,
  ) -> torch.nn.Module:

    assert isinstance(transformer, LTX2VideoTransformer3DModel)

    transformer.forward = __patch_transformer_forward__.__get__(transformer)
    if hasattr(transformer, "_old_forward"):
      # accelerate's offload hook dispatches through `_old_forward`, which was captured when the
      # hook was added -- i.e. before this patch. cache-dit's block wrapper also re-reads it, so
      # leaving it stale would call the upstream forward and fail on the keyword block call.
      transformer._old_forward = transformer.forward
    transformer._is_patched = True
    return transformer


def _coords_origin(video_coords: Optional[torch.Tensor]):
  """Return a hashable fingerprint of a token grid's origin.

  `LTX2DFRPipeline` rebases `video_coords` onto each tile's first base token, while a full canvas
  keeps one origin across denoising steps. Comparing the origin (plus the token count) therefore
  separates "next step of the same canvas" from "another tile of the same length".

  :param video_coords: RoPE coordinates, `(batch, 3, num_tokens, 2)`, or `None` when the forward
    builds them from `num_frames` / `height` / `width`.
  :returns: A tuple of the first token's coordinates, or `None`.
  """
  if video_coords is None:
    return None
  return tuple(video_coords[:, :, 0, :].detach().float().reshape(-1).tolist())


def __patch_transformer_forward__(
  self: LTX2VideoTransformer3DModel,
  hidden_states: torch.Tensor,
  audio_hidden_states: torch.Tensor,
  encoder_hidden_states: torch.Tensor,
  audio_encoder_hidden_states: torch.Tensor,
  timestep: torch.LongTensor,
  audio_timestep: Optional[torch.LongTensor] = None,
  sigma: Optional[torch.Tensor] = None,
  audio_sigma: Optional[torch.Tensor] = None,
  encoder_attention_mask: Optional[torch.Tensor] = None,
  audio_encoder_attention_mask: Optional[torch.Tensor] = None,
  num_frames: Optional[int] = None,
  height: Optional[int] = None,
  width: Optional[int] = None,
  fps: float = 24.0,
  audio_num_frames: Optional[int] = None,
  video_coords: Optional[torch.Tensor] = None,
  audio_coords: Optional[torch.Tensor] = None,
  isolate_modalities: bool = False,
  spatio_temporal_guidance_blocks: Optional[list] = None,
  perturbation_mask: Optional[torch.Tensor] = None,
  use_cross_timestep: bool = False,
  attention_kwargs: Optional[Dict[str, Any]] = None,
  video_self_attention_mask: Optional[torch.Tensor] = None,
  video_keyframes_mask: Optional[torch.Tensor] = None,
  return_dict: bool = True,
) -> torch.Tensor:
  """Forward pass for the LTX-2 audiovisual video transformer (LTX-2.0 ~ LTX-2.5).

  This mirrors `diffusers.models.transformers.transformer_ltx2.LTX2VideoTransformer3DModel.forward`
  (diffusers `0.42.0.dev0`, forward body at `transformer_ltx2.py:1366-1693`) line by line; the only
  difference is the transformer-block invocation, which is rewritten to pass the four required
  tensors positionally so that cache-dit's block wrapper
  (`CachedBlocks_Pattern_Base.forward(self, hidden_states, encoder_hidden_states, *args, **kwargs)`)
  can bind them. The audio stream is treated as the `encoder_hidden_states` stream of
  `ForwardPattern.Pattern_0`, which is why the adapter sets `check_forward_pattern=False`.

  Two notes on why the body is duplicated instead of reused:

  1. Upstream applies `@apply_lora_scale("attention_kwargs")`; replacing `forward` at instance
     level bypasses the decorator, so the LoRA scale/unscale calls are inlined here.
  2. Upstream now invokes blocks with keyword arguments only, which cannot satisfy the wrapper's
     positional contract.

  Gradient checkpointing is not mirrored: a cache-enabled transformer is inference-only, and the
  collapsed block wrapper cannot accept the positional call that `_gradient_checkpointing_func`
  requires. Calls that need it (STG, modality guidance, tile geometry changes) replay the original
  blocks in inference mode.

  :param hidden_states: Patchified video latents, `(batch, num_video_tokens, in_channels)`.
  :param audio_hidden_states: Patchified audio latents, `(batch, num_audio_tokens, audio_in_channels)`.
  :param encoder_hidden_states: Video text embeddings, `(batch, text_seq_len, caption_channels)`.
  :param audio_encoder_hidden_states: Audio text embeddings, `(batch, text_seq_len, caption_channels)`.
  :param timestep: Per-token video timestep, `(batch, num_video_tokens)`, already scaled by
    `config.timestep_scale_multiplier`.
  :param audio_timestep: Optional audio timestep, `(batch,)` or `(batch, num_audio_tokens)`.
  :param sigma: Optional scaled video timestep, `(batch,)`; used by cross-attention modulation of
    LTX-2.3+ when `use_prompt_adaln_single` / `use_cross_timestep` are enabled.
  :param audio_sigma: Optional scaled audio timestep, `(batch,)`; falls back to `sigma`.
  :param encoder_attention_mask: Optional multiplicative video text mask, `(batch, text_seq_len)`.
  :param audio_encoder_attention_mask: Optional multiplicative audio text mask.
  :param num_frames: Latent frame count, used when `video_coords` is not supplied.
  :param height: Latent height, used when `video_coords` is not supplied.
  :param width: Latent width, used when `video_coords` is not supplied.
  :param fps: Frame rate used when building video RoPE coordinates.
  :param audio_num_frames: Latent audio frame count, used when `audio_coords` is not supplied.
  :param video_coords: Optional video RoPE coordinates, `(batch, 3, num_video_tokens, 2)`.
  :param audio_coords: Optional audio RoPE coordinates, `(batch, 1, num_audio_tokens, 2)`.
  :param isolate_modalities: Disable audio-to-video / video-to-audio cross attention (modality
    guidance of LTX-2.3+). Such forwards bypass the cache, they are guidance-only passes.
  :param spatio_temporal_guidance_blocks: Block indices that apply STG. STG needs a per-block
    perturbation mask, which a single cache wrapper call cannot express, so these forwards bypass
    the cache and replay the original blocks.
  :param perturbation_mask: Per-sample STG mask, `(batch,)` or `(batch, 1, 1)`; 0 means perturbed.
  :param use_cross_timestep: Whether cross-modality sigmas drive cross-attention modulation
    (`True` for LTX-2.3+, `False` for LTX-2.0).
  :param attention_kwargs: Optional kwargs for the attention processors; `scale` is consumed here.
  :param video_self_attention_mask: Optional multiplicative video self-attention mask,
    `(batch, num_video_tokens, num_video_tokens)`; used by the IC-LoRA pipeline.
  :param video_keyframes_mask: Optional per-token keyframe marker, `(batch, num_video_tokens, 1)`.
    Tokens marked here receive `keyframes_abs_pos_embedding` (LTX-2.5).
  :param return_dict: Whether to return `AudioVisualModelOutput` or a `(video, audio)` tuple.

  :returns: `AudioVisualModelOutput` when `return_dict=True`, otherwise `(video, audio)`.
  """
  if attention_kwargs is not None:
    attention_kwargs = attention_kwargs.copy()
    lora_scale = attention_kwargs.pop("scale", 1.0)
  else:
    lora_scale = 1.0

  if USE_PEFT_BACKEND:
    # weight the lora layers by setting `lora_scale` for each PEFT layer
    scale_lora_layers(self, lora_scale)
  else:
    if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
      logger.warning(
        "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

  # Determine timestep for audio.
  audio_timestep = audio_timestep if audio_timestep is not None else timestep
  audio_sigma = audio_sigma if audio_sigma is not None else sigma

  # convert encoder_attention_mask to a bias the same way we do for attention_mask
  if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
    encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
    encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

  if audio_encoder_attention_mask is not None and audio_encoder_attention_mask.ndim == 2:
    audio_encoder_attention_mask = (
      1 - audio_encoder_attention_mask.to(audio_hidden_states.dtype)) * -10000.0
    audio_encoder_attention_mask = audio_encoder_attention_mask.unsqueeze(1)

  # Convert video_self_attention_mask from multiplicative mask ([0, 1]) to additive bias form
  # (0 / -10000) matching the encoder_attention_mask convention above. Shape: (B, T_v, T_v).
  if video_self_attention_mask is not None:
    video_self_attention_mask = (1 - video_self_attention_mask.to(hidden_states.dtype)) * -10000.0

  batch_size = hidden_states.size(0)

  # 1. Prepare RoPE positional embeddings
  if video_coords is None:
    video_coords = self.rope.prepare_video_coords(batch_size,
                                                  num_frames,
                                                  height,
                                                  width,
                                                  hidden_states.device,
                                                  fps=fps)
  if audio_coords is None:
    audio_coords = self.audio_rope.prepare_audio_coords(batch_size, audio_num_frames,
                                                        audio_hidden_states.device)

  video_rotary_emb = self.rope(video_coords, device=hidden_states.device)
  audio_rotary_emb = self.audio_rope(audio_coords, device=audio_hidden_states.device)

  video_cross_attn_rotary_emb = self.cross_attn_rope(video_coords[:, 0:1, :],
                                                     device=hidden_states.device)
  audio_cross_attn_rotary_emb = self.cross_attn_audio_rope(audio_coords[:, 0:1, :],
                                                           device=audio_hidden_states.device)

  # 2. Patchify input projections
  hidden_states = self.proj_in(hidden_states)
  audio_hidden_states = self.audio_proj_in(audio_hidden_states)

  # 2.1. Mark tokens whose latent encodes a single pixel frame (causal first frame, generated
  # keyframe slots). Ignored when the model was built without `use_keyframes_abs_pos_embedding`.
  if getattr(self.config, "use_keyframes_abs_pos_embedding",
             False) and video_keyframes_mask is not None:
    marker = (video_keyframes_mask > 0).to(dtype=hidden_states.dtype)
    hidden_states = hidden_states + marker * self.keyframes_abs_pos_embedding.to(
      dtype=hidden_states.dtype)

  # 3. Prepare timestep embeddings and modulation parameters
  timestep_cross_attn_gate_scale_factor = (self.config.cross_attn_timestep_scale_multiplier /
                                           self.config.timestep_scale_multiplier)

  # 3.1. Prepare global modality (video and audio) timestep embedding and modulation parameters
  # temb is used in the transformer blocks (as expected), while embedded_timestep is used for the
  # output layer modulation with scale_shift_table (and similarly for audio)
  temb, embedded_timestep = self.time_embed(
    timestep.flatten(),
    batch_size=batch_size,
    hidden_dtype=hidden_states.dtype,
  )
  temb = temb.view(batch_size, -1, temb.size(-1))
  embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

  temb_audio, audio_embedded_timestep = self.audio_time_embed(
    audio_timestep.flatten(),
    batch_size=batch_size,
    hidden_dtype=audio_hidden_states.dtype,
  )
  temb_audio = temb_audio.view(batch_size, -1, temb_audio.size(-1))
  audio_embedded_timestep = audio_embedded_timestep.view(batch_size, -1,
                                                         audio_embedded_timestep.size(-1))

  # 3.1.1. Prompt cross-attention modulation (LTX-2.3+). The module only exists when the model was
  # built with `prompt_modulation` and `use_prompt_adaln_single`; with `False` (LTX-2.5
  # "KV-cacheable cross-attention") the block modulates the text K/V with a static per-layer table.
  if self.prompt_modulation and getattr(self.config, "use_prompt_adaln_single", True):
    temb_prompt, _ = self.prompt_adaln(sigma.flatten(),
                                       batch_size=batch_size,
                                       hidden_dtype=hidden_states.dtype)
    temb_prompt_audio, _ = self.audio_prompt_adaln(audio_sigma.flatten(),
                                                   batch_size=batch_size,
                                                   hidden_dtype=audio_hidden_states.dtype)
    temb_prompt = temb_prompt.view(batch_size, -1, temb_prompt.size(-1))
    temb_prompt_audio = temb_prompt_audio.view(batch_size, -1, temb_prompt_audio.size(-1))
  else:
    temb_prompt = temb_prompt_audio = None

  # 3.2. Prepare global modality cross attention modulation parameters
  video_ca_timestep = audio_sigma.flatten() if use_cross_timestep else timestep.flatten()
  video_cross_attn_scale_shift, _ = self.av_cross_attn_video_scale_shift(
    video_ca_timestep,
    batch_size=batch_size,
    hidden_dtype=hidden_states.dtype,
  )
  video_cross_attn_a2v_gate, _ = self.av_cross_attn_video_a2v_gate(
    video_ca_timestep * timestep_cross_attn_gate_scale_factor,
    batch_size=batch_size,
    hidden_dtype=hidden_states.dtype,
  )
  video_cross_attn_scale_shift = video_cross_attn_scale_shift.view(
    batch_size, -1, video_cross_attn_scale_shift.shape[-1])
  video_cross_attn_a2v_gate = video_cross_attn_a2v_gate.view(batch_size, -1,
                                                             video_cross_attn_a2v_gate.shape[-1])

  audio_ca_timestep = sigma.flatten() if use_cross_timestep else audio_timestep.flatten()
  audio_cross_attn_scale_shift, _ = self.av_cross_attn_audio_scale_shift(
    audio_ca_timestep,
    batch_size=batch_size,
    hidden_dtype=audio_hidden_states.dtype,
  )
  audio_cross_attn_v2a_gate, _ = self.av_cross_attn_audio_v2a_gate(
    audio_ca_timestep * timestep_cross_attn_gate_scale_factor,
    batch_size=batch_size,
    hidden_dtype=audio_hidden_states.dtype,
  )
  audio_cross_attn_scale_shift = audio_cross_attn_scale_shift.view(
    batch_size, -1, audio_cross_attn_scale_shift.shape[-1])
  audio_cross_attn_v2a_gate = audio_cross_attn_v2a_gate.view(batch_size, -1,
                                                             audio_cross_attn_v2a_gate.shape[-1])

  # 4. Prepare prompt embeddings (LTX-2.0; LTX-2.3+ projects them inside the connectors)
  if getattr(self.config, "use_prompt_embeddings", True):
    encoder_hidden_states = self.caption_projection(encoder_hidden_states)
    encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))

    audio_encoder_hidden_states = self.audio_caption_projection(audio_encoder_hidden_states)
    audio_encoder_hidden_states = audio_encoder_hidden_states.view(batch_size, -1,
                                                                   audio_hidden_states.size(-1))

  # 5. Run transformer blocks
  block_kwargs = dict(
    temb=temb,
    temb_audio=temb_audio,
    temb_ca_scale_shift=video_cross_attn_scale_shift,
    temb_ca_audio_scale_shift=audio_cross_attn_scale_shift,
    temb_ca_gate=video_cross_attn_a2v_gate,
    temb_ca_audio_gate=audio_cross_attn_v2a_gate,
    temb_prompt=temb_prompt,
    temb_prompt_audio=temb_prompt_audio,
    video_rotary_emb=video_rotary_emb,
    audio_rotary_emb=audio_rotary_emb,
    ca_video_rotary_emb=video_cross_attn_rotary_emb,
    ca_audio_rotary_emb=audio_cross_attn_rotary_emb,
    encoder_attention_mask=encoder_attention_mask,
    audio_encoder_attention_mask=audio_encoder_attention_mask,
    self_attention_mask=video_self_attention_mask,
    audio_self_attention_mask=None,
    a2v_cross_attention_mask=None,
    v2a_cross_attention_mask=None,
    use_a2v_cross_attention=not isolate_modalities,
    use_v2a_cross_attention=not isolate_modalities,
  )

  spatio_temporal_guidance_blocks = spatio_temporal_guidance_blocks or []
  stg_blocks = set(spatio_temporal_guidance_blocks)
  if len(stg_blocks) > 0 and perturbation_mask is None:
    # If STG is being used and perturbation_mask is not supplied, perturb all batch elements.
    perturbation_mask = torch.zeros((batch_size, ), device=hidden_states.device)
  if perturbation_mask is not None and perturbation_mask.ndim == 1:
    perturbation_mask = perturbation_mask[:, None, None]
  all_perturbed = torch.all(perturbation_mask == 0) if perturbation_mask is not None else False

  # Off-cache forwards: STG needs a per-block perturbation mask and modality guidance changes the
  # block semantics, neither of which a single collapsed wrapper call can express. Replay the
  # original blocks directly so the result stays bit-exact w.r.t. the upstream forward, and leave
  # the cache step counter untouched.
  off_cache: bool = bool(stg_blocks) or isolate_modalities
  # Tile forwards (DFR epilogue) reuse the same transformer with a *subset* of tokens per call, and
  # `split_by_count` makes those tiles the same length as each other, so a token-count check alone
  # would let tile B reuse tile A's residual. Identify the forward by its coordinate origin instead:
  # tiles rebase `video_coords` to their first base token while a full canvas keeps one origin
  # across steps. Only cache-path forwards take part, otherwise the STG / modality passes of the
  # guided pipelines (different geometry) would keep resetting the reference.
  if not off_cache and not torch.compiler.is_compiling():
    marker = (int(hidden_states.shape[1]), _coords_origin(video_coords))
    last_marker = getattr(self, "_cache_forward_marker", None)
    if last_marker is not None and last_marker != marker:
      logger.warning_once("LTX-2: the video geometry changed between forwards (DFR tiles / "
                          "variable canvas); computing this forward without cache.")
      off_cache = True
    self._cache_forward_marker = marker

  blocks = self.transformer_blocks
  # After `enable_cache`, `transformer_blocks` holds a single wrapper that iterates the original
  # blocks internally; keep a handle on those originals for the off-cache replay below.
  wrapped: bool = len(blocks) == 1 and hasattr(blocks[0], "transformer_blocks")
  if off_cache:
    originals = blocks[0].transformer_blocks if wrapped else blocks
    for block_idx, block in enumerate(originals):
      block_kwargs["perturbation_mask"] = perturbation_mask if block_idx in stg_blocks else None
      block_kwargs["all_perturbed"] = all_perturbed if block_idx in stg_blocks else False
      hidden_states, audio_hidden_states = block(
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        audio_encoder_hidden_states,
        **block_kwargs,
      )
    block_kwargs.pop("perturbation_mask", None)
    block_kwargs.pop("all_perturbed", None)
  elif wrapped:
    hidden_states, audio_hidden_states = blocks[0](
      hidden_states,
      audio_hidden_states,
      encoder_hidden_states,
      audio_encoder_hidden_states,
      **block_kwargs,
    )
  else:
    for block in blocks:
      hidden_states, audio_hidden_states = block(
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        audio_encoder_hidden_states,
        **block_kwargs,
      )

  # 6. Output layers (including unpatchification)
  scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
  shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

  hidden_states = self.norm_out(hidden_states)
  hidden_states = hidden_states * (1 + scale) + shift
  output = self.proj_out(hidden_states)

  audio_scale_shift_values = (self.audio_scale_shift_table[None, None] +
                              audio_embedded_timestep[:, :, None])
  audio_shift, audio_scale = audio_scale_shift_values[:, :, 0], audio_scale_shift_values[:, :, 1]

  audio_hidden_states = self.audio_norm_out(audio_hidden_states)
  audio_hidden_states = audio_hidden_states * (1 + audio_scale) + audio_shift
  audio_output = self.audio_proj_out(audio_hidden_states)

  if USE_PEFT_BACKEND:
    # remove `lora_scale` from each PEFT layer
    unscale_lora_layers(self, lora_scale)

  if not return_dict:
    return (output, audio_output)
  return AudioVisualModelOutput(sample=output, audio_sample=audio_output)
