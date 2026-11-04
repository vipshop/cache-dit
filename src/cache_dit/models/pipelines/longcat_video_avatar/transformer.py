"""LongCat-Video-Avatar transformer (single-stream audio-driven video DiT).

Copied from `meituan-longcat/LongCat-Video` (main, 2026-11-03),
`longcat_video/modules/avatar/longcat_video_dit_avatar.py`.

Intentional differences from the upstream copy:

* class / attribute / `state_dict` names are unchanged, so the released
  `base_model/diffusion_pytorch_model-*.safetensors` loads with `strict=True`;
* the block loop is the only per-iteration call and every block keeps `hidden_states` as its first
  parameter, i.e. the `ForwardPattern.Pattern_3` contract of cache-dit;
* the autoregressive long-video path (`return_kv` / `kv_cache` / per-block cache bookkeeping) and the
  video-continuation reference-frame masking (`ref_img_index` / `mask_frame_range`) are not ported:
  cache-dit's caching and context parallelism both assume a fixed-length sequence per call;
* context parallelism keeps the frame axis as the split axis. The frequency table stays global
  (`RotaryPositionalEmbedding`), while the timestep embedding and the audio tokens are sliced to the
  local frame range before the block loop. `len(self.blocks)` is never used: the released checkpoint
  stores the depth in `config.depth`, and the block list holds a single element under caching;
* the modulation heads run in explicit float32 (`linear_fp32`) instead of `autocast(float32)`, so the
  module also runs on CPU;
* the multi-speaker branch (`ref_target_masks`) is rejected instead of silently collapsing to one
  speaker.
"""

from typing import Optional

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import (
  LongCatAvatarAudioCrossAttention,
  LongCatAvatarSelfAttention,
  LongCatAvatarTextCrossAttention,
)
from .blocks import (
  AudioProjModel,
  CaptionEmbedder,
  FeedForwardSwiGLU,
  FinalLayer_FP32,
  LayerNorm_FP32,
  PatchEmbed3D,
  TimestepEmbedder,
  modulate_fp32,
  silu_linear_fp32,
)


def cp_group(cp_config) -> tuple:
  """Resolve the context-parallel degree and this rank's offset inside it.

  `_ContextParallelConfig._rank` / `._world_size` are the *global* values handed to `setup()`, so a
  hybrid run (`--parallel ulysses_tp`, TP2 + CP2) reports 4 here and the frame axis would be cut
  into quarters that no rank owns. `_flattened_mesh` is the only CP-scoped view, and it is what the
  `blocks.0` split hook shards by.

  :param cp_config: cache-dit context-parallel config, or `None` when the sequence is not split.
  :returns: `(size, rank)`; `(1, 0)` when context parallelism is off.
  """

  mesh = getattr(cp_config, "_flattened_mesh", None) if cp_config is not None else None
  if mesh is None or mesh.size() <= 1:
    return 1, 0
  return int(mesh.size()), int(mesh.get_local_rank())


def frame_range(cp_config, global_frames: int) -> tuple:
  """Resolve the frame range owned by this rank.

  :param cp_config: cache-dit context-parallel config, or `None` when the sequence is not split.
  :param global_frames: number of latent frames of the full sequence.
  :returns: `(start, end)` frame range; `(0, global_frames)` when context parallelism is off.
  :raises ValueError: when the frame count is not divisible by the context-parallel size. Audio
    cross-attention reshapes the token axis by frame, so a rank must own whole frames.
  """

  world, rank = cp_group(cp_config)
  if world == 1:
    return 0, global_frames
  if global_frames % world != 0:
    raise ValueError(
      f"LongCat-Video-Avatar needs the latent frame count ({global_frames}) to be divisible by the "
      f"context-parallel size ({world}): the audio cross-attention reshapes the token axis by frame, "
      f"so a split inside a frame is not representable. Pick a clip whose latent frames are a "
      f"multiple of {world} (e.g. --num-frames 93 for cp=2, --num-frames 185 for cp=4) and rerun.")
  per_rank = global_frames // world
  start = rank * per_rank
  return start, start + per_rank


class LongCatAvatarSingleStreamBlock(nn.Module):
  """One audio-conditioned single-stream block.

  :param hidden_size: transformer width.
  :param num_heads: number of attention heads.
  :param mlp_ratio: FFN width multiplier.
  :param adaln_tembed_dim: width of the timestep embedding.
  :param output_dim: width of one audio token.
  :param audio_prenorm: whether to normalize the audio tokens before the cross-attention.
  :param index: zero-based position of the block inside `transformer.blocks`; the modulation and the
    audio gate are timestep driven, so nothing here indexes the stack, but keeping the index makes a
    future per-layer option expressible inside the block (the loop body is not replayed under
    caching).
  """

  def __init__(
    self,
    hidden_size: int,
    num_heads: int,
    mlp_ratio: int,
    adaln_tembed_dim: int,
    output_dim: int = 768,
    audio_prenorm: bool = True,
    index: int = 0,
  ):
    super().__init__()

    self.hidden_size = hidden_size
    self.index = index

    self.adaLN_modulation = nn.Sequential(
      nn.SiLU(),
      nn.Linear(adaln_tembed_dim, 6 * hidden_size, bias=True),
    )
    self.audio_adaLN_modulation = nn.Sequential(
      nn.SiLU(),
      nn.Linear(adaln_tembed_dim, 3 * hidden_size, bias=True),
    )

    self.mod_norm_attn = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=False)
    self.mod_norm_ffn = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=False)
    self.pre_crs_attn_norm = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=True)
    self.pre_video_crs_attn_norm = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=True)
    self.pre_audio_crs_attn_norm = (LayerNorm_FP32(output_dim, eps=1e-6, elementwise_affine=True)
                                    if audio_prenorm else nn.Identity())

    self.attn = LongCatAvatarSelfAttention(dim=hidden_size, num_heads=num_heads)
    self.cross_attn = LongCatAvatarTextCrossAttention(dim=hidden_size, num_heads=num_heads)
    self.audio_cross_attn = LongCatAvatarAudioCrossAttention(
      dim=hidden_size,
      encoder_hidden_states_dim=output_dim,
      num_heads=num_heads,
      qk_norm=True,
      qkv_bias=True,
    )
    self.ffn = FeedForwardSwiGLU(dim=hidden_size, hidden_dim=int(hidden_size * mlp_ratio))

  def forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep_embed: torch.Tensor,
    encoder_seqlen,
    latent_shape,
    num_cond_latents: int = 0,
    audio_hidden_states: Optional[torch.Tensor] = None,
    cp_config=None,
  ) -> torch.Tensor:
    """Run one block on the (local) token range.

    :param hidden_states: `[B, N, C]` residual stream of the local frame range.
    :param encoder_hidden_states: `[1, N_text, C]` packed text tokens (replicated).
    :param timestep_embed: `[B, T_local, C_t]` float32 timestep embedding of the local frames.
    :param encoder_seqlen: list of valid text lengths, one per sample.
    :param latent_shape: `(T_global, H, W)` of the full token grid.
    :param num_cond_latents: number of conditioning latent frames of the **full** sequence; the local
      count is clamped to the frames this rank owns.
    :param audio_hidden_states: `[B, T_local, M, C_audio]` audio tokens of the local frames.
    :param cp_config: cache-dit context-parallel config when the frame axis is split, else `None`.
    :returns: `[B, N, C]` updated residual stream.
    """

    x_dtype = hidden_states.dtype
    b, n, c = hidden_states.shape
    tokens_per_frame = latent_shape[1] * latent_shape[2]
    frames = n // tokens_per_frame
    frame_start, frame_end = frame_range(cp_config, latent_shape[0])
    cp_active = frame_end - frame_start != latent_shape[0]
    local_frame_range = (frame_start, frame_end) if cp_active else None

    num_cond_local = max(0, min(frames, int(num_cond_latents or 0) - frame_start))

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (silu_linear_fp32(
      self.adaLN_modulation, timestep_embed).unsqueeze(2).chunk(6, dim=-1))

    x_m = modulate_fp32(self.mod_norm_attn, hidden_states.view(b, frames, -1, c), shift_msa,
                        scale_msa).view(b, n, c)

    x_s = self.attn(
      x_m,
      shape=latent_shape,
      num_cond_latents=num_cond_local,
      local_frame_range=local_frame_range,
      cp_config=cp_config,
    )
    x = hidden_states + (gate_msa * x_s.view(b, frames, -1, c)).view(b, n, c)
    x = x.to(x_dtype)

    x = x + self.cross_attn(
      self.pre_crs_attn_norm(x),
      encoder_hidden_states,
      encoder_seqlen,
      num_cond_latents=num_cond_local,
      tokens_per_frame=tokens_per_frame,
    )

    audio_shift, audio_scale, audio_gate = (silu_linear_fp32(
      self.audio_adaLN_modulation, timestep_embed[:, num_cond_local:]).unsqueeze(2).chunk(3,
                                                                                          dim=-1))

    audio_output_cond, audio_output_noise = self.audio_cross_attn(
      self.pre_video_crs_attn_norm(x),
      self.pre_audio_crs_attn_norm(audio_hidden_states),
      frames,
      num_cond_latents=num_cond_local,
    )
    num_noise_frames = frames - num_cond_local
    audio_output_noise = modulate_fp32(
      self.mod_norm_attn,
      audio_output_noise.view(b, num_noise_frames, -1, c),
      audio_shift,
      audio_scale,
    ).view(b, -1, c)
    audio_add_x = (audio_gate * audio_output_noise.view(b, num_noise_frames, -1, c)).view(b, -1, c)
    if audio_output_cond is not None:
      audio_add_x = torch.cat([audio_output_cond, audio_add_x], dim=1).contiguous()
    x = (x + audio_add_x).to(x_dtype)

    x_m = modulate_fp32(self.mod_norm_ffn, x.view(b, frames, -1, c), shift_mlp,
                        scale_mlp).view(b, n, c)
    x_s = self.ffn(x_m)
    x = x + (gate_mlp * x_s.view(b, frames, -1, c)).view(b, n, c)
    return x.to(x_dtype)


class LongCatVideoAvatarTransformer3DModel(ModelMixin, ConfigMixin):
  """Audio-driven video transformer of LongCat-Video-Avatar.

  :param in_channels: latent channels of the VAE (16 for `AutoencoderKLWan`).
  :param out_channels: latent channels of the predicted velocity.
  :param hidden_size: transformer width.
  :param depth: number of blocks.
  :param num_heads: number of attention heads.
  :param caption_channels: text encoder hidden size.
  :param mlp_ratio: FFN width multiplier.
  :param adaln_tembed_dim: width of the timestep embedding.
  :param frequency_embedding_size: width of the sinusoidal timestep features.
  :param patch_size: `(t, h, w)` patch size; the temporal entry must stay 1.
  :param enable_flashattn3: accepted and ignored (cache-dit selects the attention backend).
  :param enable_flashattn2: accepted and ignored.
  :param enable_xformers: accepted and ignored.
  :param enable_bsa: must be False; block-sparse attention is not ported.
  :param bsa_params: accepted and ignored.
  :param cp_split_hw: accepted and ignored; kept so the released `config.json` keeps loading.
  :param text_tokens_zero_pad: whether padded text tokens are zeroed and the mask is dropped.
  :param audio_window: audio window length, in encoder frames, of one latent frame.
  :param audio_block: number of encoder feature groups per audio vector.
  :param audio_channel: width of the audio encoder (1280 for Whisper-Large).
  :param intermediate_dim: hidden width of the audio projection.
  :param output_dim: width of one audio token.
  :param context_tokens: audio tokens produced per latent frame.
  :param vae_scale: VAE temporal downsampling factor.
  :param audio_prenorm: whether the audio tokens are normalized before each cross-attention.
  :param class_range: multi-speaker position range of the upstream checkpoint; kept so the released
    `config.json` keeps loading, unused because the multi-speaker mode is rejected.
  :param class_interval: multi-speaker position interval of the upstream checkpoint; same as above.
  """

  _supports_gradient_checkpointing = False
  _repeated_blocks = ["LongCatAvatarSingleStreamBlock"]
  _cp_config = None

  @register_to_config
  def __init__(
    self,
    in_channels: int = 16,
    out_channels: int = 16,
    hidden_size: int = 4096,
    depth: int = 48,
    num_heads: int = 32,
    caption_channels: int = 4096,
    mlp_ratio: int = 4,
    adaln_tembed_dim: int = 512,
    frequency_embedding_size: int = 256,
    patch_size=(1, 2, 2),
    enable_flashattn3: bool = False,
    enable_flashattn2: bool = False,
    enable_xformers: bool = False,
    enable_bsa: bool = False,
    bsa_params=None,
    cp_split_hw=None,
    text_tokens_zero_pad: bool = False,
    audio_window: int = 5,
    audio_block: int = 12,
    audio_channel: int = 768,
    intermediate_dim: int = 512,
    output_dim: int = 768,
    context_tokens: int = 32,
    vae_scale: int = 4,
    audio_prenorm: bool = False,
    class_range: int = 24,
    class_interval: int = 4,
  ) -> None:
    super().__init__()

    if enable_bsa:
      raise NotImplementedError(
        "LongCat-Video-Avatar block-sparse attention (enable_bsa) is not supported by this port.")
    if tuple(patch_size)[0] != 1:
      raise NotImplementedError(
        "LongCat-Video-Avatar requires a temporal patch size of 1: the token grid keeps one token "
        "per latent frame, which the audio conditioning and the context-parallel frame split rely on."
      )

    self.patch_size = tuple(patch_size)
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.vae_scale = vae_scale
    self.audio_window = audio_window
    self.text_tokens_zero_pad = text_tokens_zero_pad

    self.x_embedder = PatchEmbed3D(self.patch_size, in_channels, hidden_size)
    self.t_embedder = TimestepEmbedder(
      t_embed_dim=adaln_tembed_dim,
      frequency_embedding_size=frequency_embedding_size,
    )
    self.y_embedder = CaptionEmbedder(in_channels=caption_channels, hidden_size=hidden_size)

    self.blocks = nn.ModuleList([
      LongCatAvatarSingleStreamBlock(
        hidden_size=hidden_size,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        adaln_tembed_dim=adaln_tembed_dim,
        output_dim=output_dim,
        audio_prenorm=audio_prenorm,
        index=index,
      ) for index in range(depth)
    ])

    self.audio_proj = AudioProjModel(
      seq_len=audio_window,
      seq_len_vf=audio_window + vae_scale - 1,
      blocks=audio_block,
      channels=audio_channel,
      intermediate_dim=intermediate_dim,
      output_dim=output_dim,
      context_tokens=context_tokens,
    )

    self.final_layer = FinalLayer_FP32(
      hidden_size,
      int(torch.tensor(self.patch_size).prod().item()),
      out_channels,
      adaln_tembed_dim,
    )

  def forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    num_cond_latents: int = 0,
    audio_embs: Optional[torch.Tensor] = None,
    num_ref_latents: Optional[int] = None,
    ref_img_index: Optional[int] = None,
    mask_frame_range: Optional[int] = None,
    ref_target_masks: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Predict the flow-matching velocity of one denoising step.

    :param hidden_states: `[B, in_channels, T, H, W]` noisy latents (conditioning frames included).
    :param timestep: `[B]` or `[B, T]` flow-matching timesteps.
    :param encoder_hidden_states: `[B, 1, N_text, caption_channels]` text embeddings.
    :param encoder_attention_mask: `[B, N_text]` text padding mask.
    :param num_cond_latents: number of leading latent frames that are conditioning frames.
    :param audio_embs: `[B, T_audio, audio_window, audio_block, audio_channel]` per-frame audio
      windows; `T_audio` covers the full clip at the VAE frame rate.
    :param num_ref_latents: reference-frame count of the long-video mode; unsupported here.
    :param ref_img_index: reference-frame index of the long-video mode; unsupported here.
    :param mask_frame_range: reference-frame attention mask radius; unsupported here.
    :param ref_target_masks: multi-speaker masks; unsupported here.
    :returns: `[B, out_channels, T, H, W]` float32 velocity.
    """

    if audio_embs is None:
      raise ValueError("LongCat-Video-Avatar requires `audio_embs`.")
    if num_ref_latents or ref_img_index is not None or mask_frame_range is not None:
      raise NotImplementedError(
        "LongCat-Video-Avatar long-video continuation (`num_ref_latents` / `ref_img_index` / "
        "`mask_frame_range`) is not supported by this port: the autoregressive kv-cache path is "
        "incompatible with caching and context parallelism.")
    if ref_target_masks is not None:
      raise NotImplementedError(
        "LongCat-Video-Avatar multi-speaker mode (`ref_target_masks`) is out of scope for this port "
        "(it needs caller-side speaker masks and a cross-rank gather of the reference attention map); "
        "run the single-speaker task instead.")
    if int(num_cond_latents or 0) > 1:
      raise NotImplementedError(
        "LongCat-Video-Avatar only supports a single conditioning latent frame here (image-to-video); "
        "the multi-frame conditioning branch belongs to the long-video mode.")

    batch, _, num_images, height, width = hidden_states.shape
    n_t = num_images // self.patch_size[0]
    n_h = height // self.patch_size[1]
    n_w = width // self.patch_size[2]

    if len(timestep.shape) == 1:
      timestep = timestep.unsqueeze(1).expand(-1, n_t)

    dtype = self.x_embedder.proj.weight.dtype
    hidden_states = self.x_embedder(hidden_states.to(dtype))
    timestep = timestep.to(dtype)
    encoder_hidden_states = encoder_hidden_states.to(dtype)

    t = self.t_embedder(timestep.float().flatten(), dtype=torch.float32).reshape(batch, n_t, -1)
    encoder_hidden_states = self.y_embedder(encoder_hidden_states)

    audio_hidden_states = self._audio_tokens(
      audio_embs.to(device=hidden_states.device, dtype=hidden_states.dtype))[:, -n_t:]

    if self.text_tokens_zero_pad and encoder_attention_mask is not None:
      encoder_hidden_states = encoder_hidden_states * encoder_attention_mask[:, None, :, None]
      encoder_attention_mask = (encoder_attention_mask * 0 + 1).to(encoder_attention_mask.dtype)

    encoder_hidden_states, y_seqlens = self._pack_text(encoder_hidden_states,
                                                       encoder_attention_mask)

    cp_config = getattr(self, "_cp_config", None)
    frame_start, frame_end = frame_range(cp_config, n_t)
    split = frame_end - frame_start != n_t
    if split and int(num_cond_latents or 0) > 0:
      # The blocking reason is not the collective *count* (that can be equalized) but the shapes:
      # `q` covers the noise frames of this rank while `k`/`v` of the conditioning frame exist only
      # on the rank that owns it, so the local sequence lengths differ per rank and Ulysses requires
      # them equal. Forcing one uniform attention over all local tokens keeps the shapes legal but
      # changes the semantics (measured 21.9 dB / SSIM 0.76 vs 38.6 dB / 0.974 for A2V).
      raise NotImplementedError(
        "The conditioning frame of the image-conditioned task cannot be combined with context "
        "parallelism: its K/V exist on the owning rank only, so the local `q` length (noise frames) "
        "and `k`/`v` length differ across ranks while Ulysses requires equal local sequence lengths. "
        "Use tensor parallelism for that task, or the audio+text task under context parallelism.")
    if split:
      # Preprocessing above runs on the full grid on every rank; only the block loop consumes the
      # local frame range, which is also the partition the `blocks.0` split hook applies to
      # `hidden_states`.
      t = t[:, frame_start:frame_end].contiguous()
      audio_hidden_states = audio_hidden_states[:, frame_start:frame_end].contiguous()
    local_shape = (frame_end - frame_start, n_h, n_w) if split else (n_t, n_h, n_w)

    for block in self.blocks:
      hidden_states = block(
        hidden_states,
        encoder_hidden_states,
        t,
        y_seqlens,
        (n_t, n_h, n_w),
        num_cond_latents,
        audio_hidden_states,
        cp_config,
      )

    hidden_states = self.final_layer(hidden_states, t, local_shape)
    hidden_states = self.unpatchify(hidden_states, n_t, n_h, n_w)
    return hidden_states.to(torch.float32)

  def _audio_tokens(self, audio_embs: torch.Tensor) -> torch.Tensor:
    """Turn the per-frame audio windows into per-frame audio tokens.

    :param audio_embs:`[B, T_audio, W, audio_block, audio_channel]` audio windows of the full clip.
    :returns:`[B, T_latent, context_tokens, output_dim]` audio tokens, one group per latent frame.
    """

    first_frame = audio_embs[:, :1, ...]
    latter_frames = audio_embs[:, 1:, ...]
    latter_frames = latter_frames.reshape(latter_frames.shape[0], -1, self.vae_scale,
                                          *latter_frames.shape[2:])
    if latter_frames.shape[2] < 3:
      raise ValueError(
        f"LongCat-Video-Avatar needs at least {4 * 3 + 1} pixel frames (4 latent frames): the audio "
        f"window of a latent frame is stitched from the middle frames of its VAE group, and a shorter "
        f"clip leaves that group empty (got {audio_embs.shape[1]} audio frames).")
    middle_index = self.audio_window // 2

    first = latter_frames[:, :, :1, :middle_index + 1, ...]
    first = first.reshape(first.shape[0], first.shape[1], -1, *first.shape[4:])
    last = latter_frames[:, :, -1:, middle_index:, ...]
    last = last.reshape(last.shape[0], last.shape[1], -1, *last.shape[4:])
    middle = latter_frames[:, :, 1:-1, middle_index:middle_index + 1, ...]
    middle = middle.reshape(middle.shape[0], middle.shape[1], -1, *middle.shape[4:])

    latter_windows = torch.concat([first, middle, last], dim=2)
    return self.audio_proj(first_frame, latter_windows)

  def _pack_text(self, encoder_hidden_states: torch.Tensor,
                 encoder_attention_mask: Optional[torch.Tensor]):
    """Compact the padded text tokens into one packed sequence.

    :param encoder_hidden_states: `[B, 1, N_text, C]` text embeddings.
    :param encoder_attention_mask: `[B, N_text]` padding mask, or `None`.
    :returns: `(packed [1, sum(valid), C], valid lengths per sample)`.
    """

    batch = encoder_hidden_states.shape[0]
    n_channels = encoder_hidden_states.shape[-1]
    if encoder_attention_mask is None:
      return encoder_hidden_states.squeeze(1).view(
        1, -1, n_channels), [encoder_hidden_states.shape[2]] * batch

    mask = encoder_attention_mask.squeeze(1).squeeze(1)
    tokens = encoder_hidden_states.squeeze(1).masked_select(mask.unsqueeze(-1) != 0)
    return tokens.view(1, -1, n_channels), mask.sum(dim=1).tolist()

  def unpatchify(self, x: torch.Tensor, n_t: int, n_h: int, n_w: int) -> torch.Tensor:
    """Fold the patch features back into a latent video.

    :param x:`[B, n_t * n_h * n_w, prod(patch_size) * out_channels]` patch features.
    :param n_t: number of latent frames.
    :param n_h: latent height.
    :param n_w: latent width.
    :returns:`[B, out_channels, n_t, n_h * patch_h, n_w * patch_w]` latents.
    """

    t_p, h_p, w_p = self.patch_size
    return x.view(
      x.shape[0],
      n_t,
      n_h,
      n_w,
      t_p,
      h_p,
      w_p,
      self.out_channels,
    ).permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(x.shape[0], self.out_channels, n_t * t_p, n_h * h_p,
                                              n_w * w_p)
