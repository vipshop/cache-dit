import torch
from typing import Optional, Dict, Any

try:
    from diffusers.models.transformers.transformer_ltx2 import (
        LTX2VideoTransformer3DModel,
        AudioVisualModelOutput,
    )
except ImportError:
    raise ImportError(
        "LTX2VideoTransformer3DModel is not available. "
        "Please install the latest version of diffusers."
    )

from diffusers.utils import (
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
)
from .functor_base import PatchFunctor
from cache_dit.logger import init_logger

logger = init_logger(__name__)


class LTX2PatchFunctor(PatchFunctor):

    def _apply(
        self,
        transformer: LTX2VideoTransformer3DModel,
        **kwargs,
    ) -> torch.nn.Module:

        if hasattr(transformer, "_is_patched"):
            return transformer

        assert isinstance(transformer, LTX2VideoTransformer3DModel)

        is_patched = False

        cls_name = transformer.__class__.__name__

        transformer.forward = __patch_transformer_forward__.__get__(transformer)

        is_patched = True

        if is_patched:
            logger.warning(f"Patched {cls_name} for cache-dit.")
            assert not getattr(transformer, "_is_parallelized", False), (
                "Please call `cache_dit.enable_cache` before Parallelize, "
                "the __patch_transformer_forward__ will overwrite the "
                "parallized forward and cause a downgrade of performance."
            )

        transformer._is_patched = is_patched  # True or False

        logger.info(f"Applied {self.__class__.__name__} for {cls_name}, " f"Patch: {is_patched}.")
        return transformer


def __patch_transformer_forward__(
    self: LTX2VideoTransformer3DModel,
    hidden_states: torch.Tensor,
    audio_hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    audio_encoder_hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    audio_timestep: Optional[torch.LongTensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    audio_encoder_attention_mask: Optional[torch.Tensor] = None,
    num_frames: Optional[int] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    fps: float = 24.0,
    audio_num_frames: Optional[int] = None,
    video_coords: Optional[torch.Tensor] = None,
    audio_coords: Optional[torch.Tensor] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
) -> torch.Tensor:
    """
    Forward pass for LTX-2.0 audiovisual video transformer.

    Args:
        hidden_states (`torch.Tensor`):
            Input patchified video latents of shape `(batch_size, num_video_tokens, in_channels)`.
        audio_hidden_states (`torch.Tensor`):
            Input patchified audio latents of shape `(batch_size, num_audio_tokens, audio_in_channels)`.
        encoder_hidden_states (`torch.Tensor`):
            Input video text embeddings of shape `(batch_size, text_seq_len, self.config.caption_channels)`.
        audio_encoder_hidden_states (`torch.Tensor`):
            Input audio text embeddings of shape `(batch_size, text_seq_len, self.config.caption_channels)`.
        timestep (`torch.Tensor`):
            Input timestep of shape `(batch_size, num_video_tokens)`. These should already be scaled by
            `self.config.timestep_scale_multiplier`.
        audio_timestep (`torch.Tensor`, *optional*):
            Input timestep of shape `(batch_size,)` or `(batch_size, num_audio_tokens)` for audio modulation
            params. This is only used by certain pipelines such as the I2V pipeline.
        encoder_attention_mask (`torch.Tensor`, *optional*):
            Optional multiplicative text attention mask of shape `(batch_size, text_seq_len)`.
        audio_encoder_attention_mask (`torch.Tensor`, *optional*):
            Optional multiplicative text attention mask of shape `(batch_size, text_seq_len)` for audio modeling.
        num_frames (`int`, *optional*):
            The number of latent video frames. Used if calculating the video coordinates for RoPE.
        height (`int`, *optional*):
            The latent video height. Used if calculating the video coordinates for RoPE.
        width (`int`, *optional*):
            The latent video width. Used if calculating the video coordinates for RoPE.
        fps: (`float`, *optional*, defaults to `24.0`):
            The desired frames per second of the generated video. Used if calculating the video coordinates for
            RoPE.
        audio_num_frames: (`int`, *optional*):
            The number of latent audio frames. Used if calculating the audio coordinates for RoPE.
        video_coords (`torch.Tensor`, *optional*):
            The video coordinates to be used when calculating the rotary positional embeddings (RoPE) of shape
            `(batch_size, 3, num_video_tokens, 2)`. If not supplied, this will be calculated inside `forward`.
        audio_coords (`torch.Tensor`, *optional*):
            The audio coordinates to be used when calculating the rotary positional embeddings (RoPE) of shape
            `(batch_size, 1, num_audio_tokens, 2)`. If not supplied, this will be calculated inside `forward`.
        attention_kwargs (`Dict[str, Any]`, *optional*):
            Optional dict of keyword args to be passed to the attention processor.
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether to return a dict-like structured output of type `AudioVisualModelOutput` or a tuple.

    Returns:
        `AudioVisualModelOutput` or `tuple`:
            If `return_dict` is `True`, returns a structured output of type `AudioVisualModelOutput`, otherwise a
            `tuple` is returned where the first element is the denoised video latent patch sequence and the second
            element is the denoised audio latent patch sequence.
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
                "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
            )

    # Determine timestep for audio.
    audio_timestep = audio_timestep if audio_timestep is not None else timestep

    # convert encoder_attention_mask to a bias the same way we do for attention_mask
    if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
        encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

    if audio_encoder_attention_mask is not None and audio_encoder_attention_mask.ndim == 2:
        audio_encoder_attention_mask = (
            1 - audio_encoder_attention_mask.to(audio_hidden_states.dtype)
        ) * -10000.0
        audio_encoder_attention_mask = audio_encoder_attention_mask.unsqueeze(1)

    batch_size = hidden_states.size(0)

    # 1. Prepare RoPE positional embeddings
    if video_coords is None:
        video_coords = self.rope.prepare_video_coords(
            batch_size, num_frames, height, width, hidden_states.device, fps=fps
        )
    if audio_coords is None:
        audio_coords = self.audio_rope.prepare_audio_coords(
            batch_size, audio_num_frames, audio_hidden_states.device
        )

    video_rotary_emb = self.rope(video_coords, device=hidden_states.device)
    audio_rotary_emb = self.audio_rope(audio_coords, device=audio_hidden_states.device)

    video_cross_attn_rotary_emb = self.cross_attn_rope(
        video_coords[:, 0:1, :], device=hidden_states.device
    )
    audio_cross_attn_rotary_emb = self.cross_attn_audio_rope(
        audio_coords[:, 0:1, :], device=audio_hidden_states.device
    )

    # 2. Patchify input projections
    hidden_states = self.proj_in(hidden_states)
    audio_hidden_states = self.audio_proj_in(audio_hidden_states)

    # 3. Prepare timestep embeddings and modulation parameters
    timestep_cross_attn_gate_scale_factor = (
        self.config.cross_attn_timestep_scale_multiplier / self.config.timestep_scale_multiplier
    )

    # 3.1. Prepare global modality (video and audio) timestep embedding and modulation parameters
    # temb is used in the transformer blocks (as expected), while embedded_timestep is used for the output layer
    # modulation with scale_shift_table (and similarly for audio)
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
    audio_embedded_timestep = audio_embedded_timestep.view(
        batch_size, -1, audio_embedded_timestep.size(-1)
    )

    # 3.2. Prepare global modality cross attention modulation parameters
    video_cross_attn_scale_shift, _ = self.av_cross_attn_video_scale_shift(
        timestep.flatten(),
        batch_size=batch_size,
        hidden_dtype=hidden_states.dtype,
    )
    video_cross_attn_a2v_gate, _ = self.av_cross_attn_video_a2v_gate(
        timestep.flatten() * timestep_cross_attn_gate_scale_factor,
        batch_size=batch_size,
        hidden_dtype=hidden_states.dtype,
    )
    video_cross_attn_scale_shift = video_cross_attn_scale_shift.view(
        batch_size, -1, video_cross_attn_scale_shift.shape[-1]
    )
    video_cross_attn_a2v_gate = video_cross_attn_a2v_gate.view(
        batch_size, -1, video_cross_attn_a2v_gate.shape[-1]
    )

    audio_cross_attn_scale_shift, _ = self.av_cross_attn_audio_scale_shift(
        audio_timestep.flatten(),
        batch_size=batch_size,
        hidden_dtype=audio_hidden_states.dtype,
    )
    audio_cross_attn_v2a_gate, _ = self.av_cross_attn_audio_v2a_gate(
        audio_timestep.flatten() * timestep_cross_attn_gate_scale_factor,
        batch_size=batch_size,
        hidden_dtype=audio_hidden_states.dtype,
    )
    audio_cross_attn_scale_shift = audio_cross_attn_scale_shift.view(
        batch_size, -1, audio_cross_attn_scale_shift.shape[-1]
    )
    audio_cross_attn_v2a_gate = audio_cross_attn_v2a_gate.view(
        batch_size, -1, audio_cross_attn_v2a_gate.shape[-1]
    )

    # 4. Prepare prompt embeddings
    encoder_hidden_states = self.caption_projection(encoder_hidden_states)
    encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))

    audio_encoder_hidden_states = self.audio_caption_projection(audio_encoder_hidden_states)
    audio_encoder_hidden_states = audio_encoder_hidden_states.view(
        batch_size, -1, audio_hidden_states.size(-1)
    )

    # 5. Run transformer blocks
    for block in self.transformer_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states, audio_hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                audio_hidden_states,
                encoder_hidden_states,
                audio_encoder_hidden_states,
                temb,
                temb_audio,
                video_cross_attn_scale_shift,
                audio_cross_attn_scale_shift,
                video_cross_attn_a2v_gate,
                audio_cross_attn_v2a_gate,
                video_rotary_emb,
                audio_rotary_emb,
                video_cross_attn_rotary_emb,
                audio_cross_attn_rotary_emb,
                encoder_attention_mask,
                audio_encoder_attention_mask,
            )
        else:
            hidden_states, audio_hidden_states = block(
                # Make block forward args consistent with original signature,
                # thus, also make it compatible with caching in cache-dit.
                # - Begin patching:
                # hidden_states=hidden_states,
                # audio_hidden_states=audio_hidden_states,
                # encoder_hidden_states=encoder_hidden_states,
                # audio_encoder_hidden_states=audio_encoder_hidden_states,
                # - After patching:
                hidden_states,
                audio_hidden_states,
                encoder_hidden_states,
                audio_encoder_hidden_states,
                temb=temb,
                temb_audio=temb_audio,
                temb_ca_scale_shift=video_cross_attn_scale_shift,
                temb_ca_audio_scale_shift=audio_cross_attn_scale_shift,
                temb_ca_gate=video_cross_attn_a2v_gate,
                temb_ca_audio_gate=audio_cross_attn_v2a_gate,
                video_rotary_emb=video_rotary_emb,
                audio_rotary_emb=audio_rotary_emb,
                ca_video_rotary_emb=video_cross_attn_rotary_emb,
                ca_audio_rotary_emb=audio_cross_attn_rotary_emb,
                encoder_attention_mask=encoder_attention_mask,
                audio_encoder_attention_mask=audio_encoder_attention_mask,
            )

    # 6. Output layers (including unpatchification)
    scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
    shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

    hidden_states = self.norm_out(hidden_states)
    hidden_states = hidden_states * (1 + scale) + shift
    output = self.proj_out(hidden_states)

    audio_scale_shift_values = (
        self.audio_scale_shift_table[None, None] + audio_embedded_timestep[:, :, None]
    )
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
