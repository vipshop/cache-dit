"""Causal Wan transformer used by LongLive 2.0 (`Efficient-Large-Model/LongLive-2.0-5B`).

Derived from `cache_dit/models/pipelines/causvid/transformer.py` (the in-tree causal Wan port), which
this file follows block for block - the chunk-causal attention wrapper, the CP `prepare_cp` hook, the
`_apply_rotary_emb` helper and the block forward are the CausVid versions with the LongLive 2.0
additions called out below. Keep the two in sync when either changes.

LongLive 2.0 is an autoregressive long-video generator built on the `Wan-AI/Wan2.2-TI2V-5B`
backbone: the latent clip is produced chunk by chunk (8 latent frames per chunk) and every chunk
attends to the KV cache of the chunks already generated. Attention is therefore **block-causal**:
bidirectional inside the current chunk, causal across chunks.

Reference implementation: `NVlabs/LongLive`, mirrored in
`sglang/python/sglang/multimodal_gen/runtime/{models/dits/longlive2.py,
pipelines_core/stages/model_specific_stages/longlive2.py}`. The released checkpoint is a native
Wan state dict (`model.*`), see `convert.py` for the key mapping.

Differences from the sglang implementation, on purpose:

* the KV cache is a plain global-coordinate buffer per layer (no ring buffer with eviction). Clips
  shorter than the attention window - which covers the released 61-frame / 16-latent-frame setting
  against `local_attn_size=32` - take the zero-copy prefix slice; a longer clip pays one
  `index_select` per layer per forward instead of sglang's in-place roll;
* cross-attention is the stock diffusers `WanAttnProcessor`: its text K/V depends only on the
  prompt, so the per-chunk cross-attention cache of the reference implementation is a pure speed
  optimisation and is not reproduced here;
* one forward processes exactly one chunk, and the caller decides the global token range through
  `current_start` / `current_end`;
* `force_full_compute=True` replays the raw block list, bypassing cache-dit's block-skip wrapper.
  The clean-latent pass of every chunk must write the real K/V of all layers, otherwise the next
  chunk reads stale keys.

Copied/adapted from `sglang` (Apache-2.0) and `diffusers` `models/transformers/transformer_wan.py`.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from diffusers.models.transformers.transformer_wan import (
  WanAttention,
  WanRotaryPosEmbed,
  WanTransformer3DModel,
  WanTransformerBlock,
  _get_qkv_projections,
)

from ....attention import _dispatch_attention_fn
from ....logger import init_logger

logger = init_logger(__name__)

# `num_frames_per_block` of the released checkpoint: latent frames produced by one chunk.
DEFAULT_CHUNK_LATENT_FRAMES = 8
# Attention window / sink of the released checkpoint, in latent frames.
DEFAULT_LOCAL_ATTN_FRAMES = 32
DEFAULT_SINK_FRAMES = 8


class LongLive2SelfAttentionKVCache(nn.Module):
  """Per-layer causal self-attention cache addressed by global token coordinates.

  :param batch_size: Batch size of the cached keys/values.
  :param seq_len: Total token count of the whole clip (all chunks), i.e. `frames * frame_seq_len`.
  :param heads: Head count held by this rank (already divided by the TP and CP sizes).
  :param head_dim: Per-head channel count.
  :param dtype: Cache dtype, normally the transformer dtype.
  :param device: Cache device.
  :param window_tokens: Recent-window length in tokens; `0` keeps every generated token visible.
  :param sink_tokens: Leading tokens that stay visible after the window slides.
  """

  def __init__(
    self,
    batch_size: int,
    seq_len: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    window_tokens: int = 0,
    sink_tokens: int = 0,
  ) -> None:
    super().__init__()
    self.seq_len = seq_len
    self.heads = heads
    self.head_dim = head_dim
    self.window_tokens = window_tokens
    self.sink_tokens = sink_tokens
    self.register_buffer(
      "k",
      torch.zeros(batch_size, seq_len, heads, head_dim, dtype=dtype, device=device),
      persistent=False,
    )
    self.register_buffer(
      "v",
      torch.zeros(batch_size, seq_len, heads, head_dim, dtype=dtype, device=device),
      persistent=False,
    )

  def write(self, start: int, end: int, key: torch.Tensor, value: torch.Tensor) -> None:
    """Store the K/V of one chunk into its global token slots.

    :param start: First global token index of the chunk.
    :param end: One past the last global token index of the chunk.
    :param key: Keys of the chunk, shaped `(B, end - start, heads, head_dim)`.
    :param value: Values of the chunk, same shape as `key`.
    """
    self.k[:, start:end] = key.to(self.k.dtype)
    self.v[:, start:end] = value.to(self.v.dtype)

  def visible(self, end: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the keys/values visible to the chunk ending at `end`.

    The visible set is `[0, end)` while the clip fits the attention window, and
    `[0, sink_tokens) U [tail, end)` afterwards, where the sink is packed *inside* the window
    budget: `tail = max(sink_tokens, end - (window_tokens - sink_tokens))`, so a 32-frame window with
    8 sink frames keeps 8 + 24 frames, not 8 + 32.

    :param end: One past the last visible global token index.
    :returns: A `(key, value)` pair of shape `(B, visible_len, heads, head_dim)`.
    """
    if self.window_tokens <= 0 or end <= self.window_tokens:
      return self.k[:, :end], self.v[:, :end]
    sink = min(self.sink_tokens, end)
    start = max(sink, end - max(0, self.window_tokens - sink))
    if start <= sink:
      return self.k[:, :end], self.v[:, :end]
    index = torch.cat([
      torch.arange(0, self.sink_tokens, device=self.k.device),
      torch.arange(start, end, device=self.k.device),
    ])
    return self.k.index_select(1, index), self.v.index_select(1, index)

  def reset(self) -> None:
    """Zero the cache before a new request."""
    self.k.zero_()
    self.v.zero_()


class LongLive2RotaryPosEmbed(WanRotaryPosEmbed):
  """Wan 3-D RoPE with a frame offset, for chunk-wise autoregressive decoding.

  The chunk placed at latent frame `start_frame` is rotated with the frame positions it would have
  in a full-clip forward, so a chunk never depends on how the clip was chunked.
  """

  def forward(self,
              hidden_states: torch.Tensor,
              start_frame: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the rotary tables for one chunk.

    :param hidden_states: Chunk latent shaped `(B, C, F, H, W)`.
    :param start_frame: Global index of the first latent frame of this chunk.
    :returns: A `(freqs_cos, freqs_sin)` pair, each shaped `(1, F_p * H_p * W_p, 1, head_dim)`.
    """
    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.patch_size
    ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

    split_sizes = [self.t_dim, self.h_dim, self.w_dim]

    freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
    freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

    frame_slice = slice(start_frame, start_frame + ppf)
    freqs_cos_f = freqs_cos[0][frame_slice].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
    freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
    freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

    freqs_sin_f = freqs_sin[0][frame_slice].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
    freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
    freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

    freqs_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w],
                          dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
    freqs_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w],
                          dim=-1).reshape(1, ppf * pph * ppw, 1, -1)

    return freqs_cos, freqs_sin


def _apply_rotary_emb(
  hidden_states: torch.Tensor,
  freqs_cos: torch.Tensor,
  freqs_sin: torch.Tensor,
) -> torch.Tensor:
  x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
  cos = freqs_cos[..., 0::2]
  sin = freqs_sin[..., 1::2]
  out = torch.empty_like(hidden_states)
  out[..., 0::2] = x1 * cos - x2 * sin
  out[..., 1::2] = x1 * sin + x2 * cos
  return out.type_as(hidden_states)


class LongLive2CausalAttnProcessor:
  """Block-causal self-attention with a KV cache written in global token coordinates.

  Execution order matters and is the same with and without context parallelism:

  1. project `q/k/v`, apply the per-head RMSNorm **before** the head unflatten (a distributed norm
     shards its weight along the channel axis and asserts that shape);
  2. apply RoPE with the chunk offset, on the local sequence shard;
  3. under Ulysses CP, all-to-all `q/k/v` from `(S_local, H_global)` to `(S_chunk, H_local)`;
  4. write the chunk K/V into `[current_start, current_end)`; every rank writes its own head slice;
  5. attend over the visible range - the cache already holds the current chunk, and the range is
     bidirectional, which is the upstream semantics (no causal mask inside a chunk);
  6. under CP, all-to-all the output back to `(S_local, H_global)` and project out.
  """

  _attention_backend = None
  _cp_config = None
  _parallel_config = None

  def __init__(self) -> None:
    self._cp_comm = None
    self._cp_comm_config = None

  def _cp_size(self) -> int:
    cp_config = getattr(self, "_cp_config", None)
    if cp_config is None:
      return 1
    return int(getattr(cp_config, "ulysses_degree", 1) or 1)

  def prepare_cp(self, cp_config) -> None:
    """Bind the context-parallel config and build the communicator ahead of the first forward.

    The pipeline calls this from `init_kv_cache`, i.e. outside any compiled region. Building the
    communicator lazily inside a compiled block traces `torch.distributed` object setup, which
    dynamo rejects once the block is quantized.

    :param cp_config: The context-parallel config cache-dit attached to this processor.
    """
    self._cp_config = cp_config
    if int(getattr(cp_config, "ulysses_degree", 1) or 1) <= 1:
      self._cp_comm = None
      self._cp_comm_config = None
      return
    from ....distributed.communicate.primitives import _All2AllComm

    if self._cp_comm is None or self._cp_comm_config is not cp_config:
      self._cp_comm = _All2AllComm(cp_config)
      self._cp_comm_config = cp_config

  def _comm(self):
    cp_config = self._cp_config
    if self._cp_comm is None or self._cp_comm_config is not cp_config:
      self.prepare_cp(cp_config)
    return self._cp_comm

  def __call__(
    self,
    attn: WanAttention,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    kv_cache: Optional[LongLive2SelfAttentionKVCache] = None,
    current_start: int = 0,
    current_end: int = 0,
  ) -> torch.Tensor:
    if attn.is_cross_attention:
      # Cross-attention keeps the stock diffusers processor: its text K/V is replicated and needs no
      # causal cache, and silently treating it as self-attention would attend the chunk to itself.
      raise ValueError("LongLive2CausalAttnProcessor is for self-attention only.")
    query, key, value = _get_qkv_projections(attn, hidden_states, None)

    # The norm runs on the flat `inner_dim` axis: a distributed RMSNorm shards its weight along the
    # channels and asserts that the incoming last dim matches, so it must stay un-unflattened.
    query = attn.norm_q(query)
    key = attn.norm_k(key)

    query = query.unflatten(2, (attn.heads, -1))
    key = key.unflatten(2, (attn.heads, -1))
    value = value.unflatten(2, (attn.heads, -1))

    if rotary_emb is not None:
      query = _apply_rotary_emb(query, *rotary_emb)
      key = _apply_rotary_emb(key, *rotary_emb)

    cp_size = self._cp_size()
    if cp_size > 1:
      comm = self._comm()
      comm.init_meta(query)
      query = comm.send_q(query).wait()
      key = comm.send_k(key).wait()
      value = comm.send_v(value).wait()

    # The projections are contiguous on the single-rank path but strided on the CP path; the kernel
    # a bf16 dispatch picks depends on that, and the two must not diverge by more than rounding. The
    # cached K/V slices stay views on purpose (zero copy over the cache).
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    if kv_cache is None:
      key_visible, value_visible = key, value
    else:
      if kv_cache.heads != key.shape[2]:
        raise ValueError(
          f"KV cache head count {kv_cache.heads} does not match the attention head count "
          f"{key.shape[2]}. Re-create the cache with `transformer.init_kv_cache(...)` after the "
          f"parallelism plan has been applied.")
      kv_cache.write(current_start, current_end, key, value)
      key_visible, value_visible = kv_cache.visible(current_end)

    hidden_states = _dispatch_attention_fn(
      query,
      key_visible,
      value_visible,
      attn_mask=attention_mask,
      dropout_p=0.0,
      is_causal=False,
      backend=self._attention_backend,
      # The all-to-all above already moved the sequence/heads layout; letting the dispatcher
      # communicate again would corrupt the layout.
      parallel_config=None,
      cp_config=None,
    )

    hidden_states = hidden_states.flatten(2, 3)
    hidden_states = hidden_states.type_as(query)

    if cp_size > 1:
      hidden_states = self._comm().send_o(hidden_states.unflatten(2, (query.shape[2], -1))).wait()
      hidden_states = hidden_states.flatten(2, 3)

    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)
    return hidden_states


class LongLive2TransformerBlock(WanTransformerBlock):
  """Causal Wan block: a diffusers `WanTransformerBlock` with a KV-cached self-attention.

  The parameter tree is identical to `WanTransformerBlock`, so a converted Wan state dict loads with
  `strict=True`. `index` identifies the layer inside the transformer; the KV cache container is
  passed as a whole and each block picks its own entry, because cache-dit replaces the block list
  with a single wrapper during a cached forward.
  """

  def __init__(self, *args, index: int = 0, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.index = index
    self.attn1.set_processor(LongLive2CausalAttnProcessor())

  def forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    rotary_emb: Tuple[torch.Tensor, torch.Tensor],
    kv_cache: Optional[list] = None,
    current_start: int = 0,
    current_end: int = 0,
  ) -> torch.Tensor:
    """Run one causal block.

    :param hidden_states: Tokens of the current chunk, shaped `(B, S, C)`.
    :param encoder_hidden_states: Projected text embeddings, shaped `(B, L, C)`.
    :param temb: Timestep modulation, either `(B, 6, C)` or `(B, S, 6, C)` (per-token).
    :param rotary_emb: A `(freqs_cos, freqs_sin)` pair for this chunk.
    :param kv_cache: Per-layer cache container; entry `self.index` belongs to this block.
    :param current_start: First global token index of the chunk.
    :param current_end: One past the last global token index of the chunk.
    :returns: The updated tokens, shaped `(B, S, C)`.
    """
    if temb.ndim == 4:
      shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
        self.scale_shift_table.unsqueeze(0) + temb.float()).chunk(6, dim=2)
      shift_msa, scale_msa, gate_msa = (shift_msa.squeeze(2), scale_msa.squeeze(2),
                                        gate_msa.squeeze(2))
      c_shift_msa, c_scale_msa, c_gate_msa = (
        c_shift_msa.squeeze(2),
        c_scale_msa.squeeze(2),
        c_gate_msa.squeeze(2),
      )
    else:
      shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
        self.scale_shift_table + temb.float()).chunk(6, dim=1)

    layer_cache = None if kv_cache is None else kv_cache[self.index]

    norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) +
                          shift_msa).type_as(hidden_states)
    attn_output = self.attn1(
      norm_hidden_states,
      None,
      None,
      rotary_emb,
      kv_cache=layer_cache,
      current_start=current_start,
      current_end=current_end,
    )
    hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

    norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
    attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
    hidden_states = hidden_states + attn_output

    norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) +
                          c_shift_msa).type_as(hidden_states)
    ff_output = self.ffn(norm_hidden_states)
    hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

    return hidden_states


class LongLive2Transformer3DModel(WanTransformer3DModel):
  """Wan 2.2 TI2V-5B transformer with LongLive 2.0's chunk-causal forward.

  Only the block class, the RoPE offset and the forward are replaced; everything else (config, fp32
  keep list, text/timestep embedding, output norm) is inherited, which also keeps the Wan tensor-
  parallel plan valid for this class.
  """

  _repeated_blocks = ["LongLive2TransformerBlock"]
  _no_split_modules = ["LongLive2TransformerBlock"]

  # The signature has to be explicit: `ConfigMixin.from_config` builds the instance from the keys of
  # `__init__`, so a `*args, **kwargs` subclass silently constructs the parent **defaults**
  # (`num_attention_heads=40`) while `config.json` still says 24 - and the checkpoint then fails to
  # load with a 5120-vs-3072 shape mismatch on `norm_k`.
  def __init__(
    self,
    patch_size: tuple[int, ...] = (1, 2, 2),
    num_attention_heads: int = 40,
    attention_head_dim: int = 128,
    in_channels: int = 16,
    out_channels: int = 16,
    text_dim: int = 4096,
    freq_dim: int = 256,
    ffn_dim: int = 13824,
    num_layers: int = 40,
    cross_attn_norm: bool = True,
    qk_norm: Optional[str] = "rms_norm_across_heads",
    eps: float = 1e-6,
    image_dim: Optional[int] = None,
    added_kv_proj_dim: Optional[int] = None,
    rope_max_seq_len: int = 1024,
    pos_embed_seq_len: Optional[int] = None,
  ) -> None:
    super().__init__(
      patch_size=patch_size,
      num_attention_heads=num_attention_heads,
      attention_head_dim=attention_head_dim,
      in_channels=in_channels,
      out_channels=out_channels,
      text_dim=text_dim,
      freq_dim=freq_dim,
      ffn_dim=ffn_dim,
      num_layers=num_layers,
      cross_attn_norm=cross_attn_norm,
      qk_norm=qk_norm,
      eps=eps,
      image_dim=image_dim,
      added_kv_proj_dim=added_kv_proj_dim,
      rope_max_seq_len=rope_max_seq_len,
      pos_embed_seq_len=pos_embed_seq_len,
    )
    inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
    self.rope = LongLive2RotaryPosEmbed(
      self.config.attention_head_dim,
      tuple(self.config.patch_size),
      self.config.rope_max_seq_len,
    )
    self.blocks = nn.ModuleList([
      LongLive2TransformerBlock(
        inner_dim,
        self.config.ffn_dim,
        self.config.num_attention_heads,
        self.config.qk_norm,
        self.config.cross_attn_norm,
        self.config.eps,
        self.config.added_kv_proj_dim,
        index=index,
      ) for index in range(self.config.num_layers)
    ])
    self._kv_cache = None

  def input_dtype(self) -> torch.dtype:
    """Return the dtype the latents must be in.

    `next(self.parameters()).dtype` is **not** usable here: `_keep_in_fp32_modules` puts
    `scale_shift_table` and the time/text embedders in fp32, so the first parameter is float32 while
    every kernel that touches the latent grid runs in the checkpoint dtype.

    :returns: The dtype of the patch embedding, i.e. the latent dtype.
    """
    return self.patch_embedding.weight.dtype

  def frame_seq_len(self, latent_shape: Tuple[int, ...]) -> int:
    """Token count of one latent frame for a chunk latent shape `(B, C, F, H, W)`."""
    _, _, _, height, width = latent_shape
    _, p_h, p_w = self.config.patch_size
    return (height // p_h) * (width // p_w)

  def init_kv_cache(
    self,
    batch_size: int,
    latent_shape: Tuple[int, int, int, int, int],
    num_frames: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    local_attn_frames: int = DEFAULT_LOCAL_ATTN_FRAMES,
    sink_frames: int = DEFAULT_SINK_FRAMES,
  ) -> list:
    """Allocate (or re-allocate) the per-layer KV caches for one clip.

    :param batch_size: Batch size of the request.
    :param latent_shape: Chunk latent shape `(B, C, F_chunk, H, W)`; `H`/`W` fix the frame token
      count, `F_chunk` is only used when `num_frames` is omitted.
    :param num_frames: Total latent frames of the whole clip (all chunks), which sets the cache size.
    :param device: Cache device; defaults to the transformer's device.
    :param dtype: Cache dtype; defaults to the transformer's dtype.
    :param local_attn_frames: Recent-window length in latent frames, `<= 0` disables the window.
    :param sink_frames: Leading frames that stay visible after the window slides.
    :returns: The new cache container, one entry per layer.
    """
    frame_seq_len = self.frame_seq_len(latent_shape)
    if num_frames is None:
      num_frames = latent_shape[2]
    num_layers = self.config.num_layers
    # `attn1.heads` is already divided by the TP size; the CP shard is taken inside the attention,
    # so the cache only ever holds the heads this rank owns. Read the live block list, not `blocks[0]`
    # which may currently be cache-dit's wrapper.
    blocks = self._raw_blocks()
    heads = blocks[0].attn1.heads
    cp_config = getattr(self, "_cp_config", None)
    cp_size = int(getattr(cp_config, "ulysses_degree", 1) or 1) if cp_config is not None else 1
    if heads % cp_size != 0:
      raise ValueError(
        f"LongLive2 needs the post-TP head count to divide by the Ulysses size, got {heads} over "
        f"{cp_size}. Re-run with an Ulysses size that divides the head count.")
    local_heads = heads // cp_size
    if cp_config is not None and cp_size > 1:
      # Built here, outside the compiled blocks, so a quantized graph never traces the communicator.
      for block in blocks:
        block.attn1.processor.prepare_cp(cp_config)
    param = next(self.parameters())
    device = param.device if device is None else device
    dtype = param.dtype if dtype is None else dtype
    total_len = frame_seq_len * num_frames
    window_tokens = max(local_attn_frames, 0) * frame_seq_len
    if 0 < window_tokens < latent_shape[2] * frame_seq_len:
      raise ValueError(
        f"The attention window ({local_attn_frames} latent frames) must be at least one chunk "
        f"({latent_shape[2]} latent frames) long, otherwise a chunk cannot attend to all of itself."
      )
    sink_tokens = max(sink_frames, 0) * frame_seq_len if window_tokens > 0 else 0
    if sink_tokens >= window_tokens > 0:
      raise ValueError(
        f"The sink ({sink_frames} latent frames) must be shorter than the attention window "
        f"({local_attn_frames} latent frames): the sink is part of the window budget.")
    self._kv_cache = [
      LongLive2SelfAttentionKVCache(batch_size,
                                    total_len,
                                    local_heads,
                                    self.config.attention_head_dim,
                                    dtype,
                                    device,
                                    window_tokens=window_tokens,
                                    sink_tokens=sink_tokens) for _ in range(num_layers)
    ]
    logger.debug(f"LongLive2 KV cache: {num_layers} layers, {total_len} tokens/layer, "
                 f"{local_heads} heads/rank, window {local_attn_frames} frames "
                 f"({dtype}, {device}).")
    return self._kv_cache

  def release_kv_cache(self) -> None:
    """Drop the per-layer KV caches, e.g. once the clip has been denoised."""
    self._kv_cache = None

  def _raw_blocks(self) -> nn.ModuleList:
    """Return the real block stack, unwrapping cache-dit's one-element block wrapper if present."""
    blocks = self.blocks
    if len(blocks) == 1 and hasattr(blocks[0], "transformer_blocks"):
      return blocks[0].transformer_blocks
    return blocks

  def _split_block_list(self):
    """Return `(visible_blocks, raw_blocks)`, with `raw_blocks is None` when no wrapper is
    installed.

    A cache-dit block wrapper is recognised by type, not by list length: it owns a
    `transformer_blocks` list. Both shapes are flattened - one wrapper for the whole stack, or one
    wrapper per block - because the clean-latent pass must reach the real blocks: replaying a wrapper
    would let its block skip fire and leave noisy K/V behind. A stack that mixes wrapped and raw
    entries is rejected instead of guessed at, since that is the only path to a silent dirty cache.
    """
    blocks = self.blocks
    wrapped = [block for block in blocks if hasattr(block, "transformer_blocks")]
    if not wrapped:
      return blocks, None
    if len(wrapped) != len(blocks):
      raise RuntimeError(
        f"Mixed block stack: {len(wrapped)} of {len(blocks)} entries look like cache-dit block "
        f"wrappers, so the clean-latent pass cannot tell which blocks are real. Please report this "
        f"with the cache-dit version.")
    return blocks, [inner for block in wrapped for inner in block.transformer_blocks]

  def forward(
    self,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    kv_cache: Optional[list] = None,
    current_start: int = 0,
    current_end: int = 0,
    force_full_compute: bool = False,
  ) -> torch.Tensor:
    """Denoise one chunk with the causal KV cache.

    :param hidden_states: Chunk latent shaped `(B, C, F_chunk, H, W)`.
    :param timestep: Diffusion timestep of the chunk, shaped `(B,)` or `(B, F_chunk)`. A per-frame
      timestep is expanded to one value per token, which is the Wan2.2 TI2V convention.
    :param encoder_hidden_states: Text embeddings `(B, L, text_dim)`.
    :param kv_cache: Container returned by :meth:`init_kv_cache`; `None` disables the cache.
    :param current_start: First global token index of the chunk.
    :param current_end: One past the last global token index of the chunk.
    :param force_full_compute: Replay the raw block stack instead of the cache wrapper. Required for
      the clean-latent pass, which is the only writer of the clean K/V of a chunk.
    :returns: The predicted chunk, shaped `(B, C, F_chunk, H, W)`.
    """
    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    frame_seq_len = post_patch_height * post_patch_width
    chunk_tokens = post_patch_num_frames * frame_seq_len
    cp_config = getattr(self, "_cp_config", None)
    cp_size = int(getattr(cp_config, "ulysses_degree", 1) or 1) if cp_config is not None else 1
    if cp_size > 1 and chunk_tokens % cp_size != 0:
      raise ValueError(
        f"Ulysses CP needs the chunk token count to divide by the CP size, got {chunk_tokens} "
        f"tokens over {cp_size} ranks. Change `--num-frames`/`--height`/`--width` or the CP size.")
    if cp_size > 1 and post_patch_num_frames % cp_size != 0:
      raise ValueError(
        f"Ulysses CP splits the per-frame timestep of this model, so the chunk's latent frame count "
        f"must divide by the CP size; got {post_patch_num_frames} frames over {cp_size} ranks. "
        f"Change `--num-frames` (chunk length) or the CP size.")
    start_frame = current_start // frame_seq_len
    rotary_emb = self.rope(hidden_states, start_frame)

    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2).contiguous()

    if timestep.ndim == 1:
      timestep = timestep.unsqueeze(1).expand(batch_size, post_patch_num_frames)
    # Under CP the hook on `""` has already sliced the per-frame timestep to this rank's frames, and
    # a frame-major token grid means that slice is exactly this rank's contiguous token range, so the
    # expansion below must use the *local* frame count rather than the chunk's.
    timestep = timestep.repeat_interleave(frame_seq_len, dim=1)
    local_tokens = timestep.shape[1]
    temb, timestep_proj, encoder_hidden_states, _ = self.condition_embedder(
      timestep.flatten(), encoder_hidden_states, None, timestep_seq_len=local_tokens)
    timestep_proj = timestep_proj.unflatten(2, (6, -1))

    blocks, raw_blocks = self._split_block_list()
    if force_full_compute and raw_blocks is None and len(blocks) == 1:
      # A one-element block list we cannot look through means some other wrapper is installed; the
      # clean-latent pass would then be block-skipped and the layer K/V of this chunk would stay at
      # the noisy step, silently corrupting every later chunk.
      raise RuntimeError(
        f"Cannot bypass the block-skip wrapper of type {type(blocks[0]).__name__}: the clean-latent "
        f"pass of a causal chunk must run every block. Please report this with the cache-dit version."
      )
    loop_blocks = raw_blocks if (force_full_compute and raw_blocks is not None) else blocks
    for block in loop_blocks:
      hidden_states = block(
        hidden_states,
        encoder_hidden_states,
        timestep_proj,
        rotary_emb,
        kv_cache=kv_cache,
        current_start=current_start,
        current_end=current_end,
      )

    # `temb` is always per token here `(B, S, C)`, the Wan2.2 TI2V modulation layout.
    shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(
      2, dim=2)
    shift = shift.squeeze(2).to(hidden_states.device)
    scale = scale.squeeze(2).to(hidden_states.device)
    hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) +
                     shift).type_as(hidden_states)
    # Under context parallelism the gather hook sits on `proj_out`, i.e. on 64-wide tokens instead of
    # the 1536-wide residual stream. `norm_out` itself is per token with a per-token modulation, so
    # running it on the local shard is exact.
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(batch_size, post_patch_num_frames, post_patch_height,
                                          post_patch_width, p_t, p_h, p_w, -1)
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

  @classmethod
  def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
    """Load the transformer from a converted diffusers directory, naming the fix-up on failure.

    :param pretrained_model_name_or_path: Directory holding `config.json` and the converted weights.
    :param kwargs: Forwarded to :meth:`WanTransformer3DModel.from_pretrained`.
    :returns: The loaded transformer.
    """
    try:
      return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
    except (OSError, KeyError, RuntimeError) as exc:
      raise RuntimeError(
        f"Could not load the LongLive 2.0 transformer from {pretrained_model_name_or_path}. The "
        f"released checkpoint is a native Wan state dict; convert it first with "
        f"`python -m cache_dit.models.pipelines.longlive2.convert`.") from exc
