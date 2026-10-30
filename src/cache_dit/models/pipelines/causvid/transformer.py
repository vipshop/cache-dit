"""Causal Wan transformer used by CausVid-Wan (`tianweiy/CausVid`).

CausVid distills `Wan-AI/Wan2.1-T2V-1.3B` into a 4-step autoregressive student: the latent clip is
generated chunk by chunk (3 latent frames per chunk) and every chunk attends to the KV cache of the
previous chunks. The attention is therefore **block-causal**: bidirectional inside the current chunk,
causal across chunks. Upstream implements it in `causvid/models/wan/causal_model.py`; this module is
a port onto the diffusers `WanTransformer3DModel` module tree so that the weights, the tensor-parallel
plan and the context-parallel hooks of the Wan family keep working.

Module naming: the block internals are the diffusers names (`norm1/attn1/norm2/attn3/ffn.net.*`), the
top level keeps the native Wan names (`patch_embedding`, `text_embedding`, `time_embedding`,
`time_projection`, `head`) because the reference checkpoint is a native Wan state dict. See
`convert.py` for the key mapping.

Differences from the upstream causal model, on purpose:

* the KV cache is a plain tensor pair per layer (no dict container) and is allocated lazily by
  :meth:`CausVidTransformer3DModel.init_kv_cache`, so the head count always matches the live TP/CP
  sharding;
* one forward processes exactly one chunk, and the caller decides the global token range through
  `current_start` / `current_end`;
* `force_full_compute=True` replays the raw block list, bypassing cache-dit's block-skip wrapper. The
  clean-latent pass of every chunk must write the real K/V of all layers, otherwise the next chunk
  reads stale keys.
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


class CausVidSelfAttentionKVCache(nn.Module):
  """Per-layer causal self-attention cache addressed by global token coordinates.

  :param batch_size: Batch size of the cached keys/values.
  :param seq_len: Total token count of the whole clip (all chunks), i.e. `frames * frame_seq_len`.
  :param heads: Head count held by this rank (already divided by the TP and CP sizes).
  :param head_dim: Per-head channel count.
  :param dtype: Cache dtype, normally the transformer dtype.
  :param device: Cache device.
  """

  def __init__(
    self,
    batch_size: int,
    seq_len: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
  ) -> None:
    super().__init__()
    self.seq_len = seq_len
    self.heads = heads
    self.head_dim = head_dim
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

    :param end: One past the last visible global token index.
    :returns: A `(key, value)` pair of shape `(B, end, heads, head_dim)`.
    """
    return self.k[:, :end], self.v[:, :end]

  def reset(self) -> None:
    """Zero the cache before a new request."""
    self.k.zero_()
    self.v.zero_()


class CausVidRotaryPosEmbed(WanRotaryPosEmbed):
  """Wan 3-D RoPE with a frame offset, for chunk-wise autoregressive decoding.

  The upstream causal model rotates the tokens of chunk `i` with the frame positions starting at
  `current_start // frame_seq_len`, so a chunk placed at the tail of the clip gets the same rotation
  it would have in a full-clip forward.
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


class CausVidCausalAttnProcessor:
  """Block-causal self-attention with a KV cache written in global token coordinates.

  Execution order matters and is the same with and without context parallelism:

  1. project `q/k/v`, apply the per-head RMSNorm **before** the head unflatten (a distributed norm
     shards its weight along the channel axis and asserts that shape);
  2. apply RoPE with the chunk offset, on the local sequence shard;
  3. under Ulysses CP, all-to-all `q/k/v` from `(S_local, H_global)` to `(S_chunk, H_local)`;
  4. write the chunk K/V into `[current_start, current_end)`; every rank writes its own head slice;
  5. attend over `[0, current_end)` — the cache already holds the current chunk, and the range is
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
    communicator lazily inside a compiled block traces `torch.distributed` object setup, which dynamo
    rejects once the block is quantized (`Polyfill handler ... does not have a traceable function`).

    :param cp_config: The context-parallel config cache-dit attached to this processor.
    """
    self._cp_config = cp_config
    if int(getattr(cp_config, "ulysses_degree", 1) or 1) <= 1:
      self._cp_comm = None
      self._cp_comm_config = None
      return
    from ....distributed.core._distributed_primitives import _All2AllComm

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
    kv_cache: Optional[CausVidSelfAttentionKVCache] = None,
    current_start: int = 0,
    current_end: int = 0,
  ) -> torch.Tensor:
    if attn.is_cross_attention:
      # Cross-attention keeps the stock diffusers processor: its text K/V is replicated and needs no
      # causal cache, and silently treating it as self-attention would attend the chunk to itself.
      raise ValueError("CausVidCausalAttnProcessor is for self-attention only.")
    query, key, value = _get_qkv_projections(attn, hidden_states, None)

    # The norm runs on the flat `inner_dim` axis: a distributed RMSNorm shards its weight along the
    # channels and asserts that the incoming last dim matches, so it must stay un-unflattened.
    query = attn.norm_q(query)
    key = attn.norm_k(key)

    query = query.unflatten(2, (attn.heads, -1))
    key = key.unflatten(2, (attn.heads, -1))
    value = value.unflatten(2, (attn.heads, -1))

    if rotary_emb is not None:
      # Applied on the local token slice, before any all-to-all: every token carries its own global
      # frame position, so no rank needs the tables of another rank.
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


class CausVidTransformerBlock(WanTransformerBlock):
  """Causal Wan block: a diffusers `WanTransformerBlock` with a KV-cached self-attention.

  The parameter tree is identical to `WanTransformerBlock`, so a converted Wan state dict loads with
  `strict=True`. `index` identifies the layer inside the transformer; the KV cache container is
  passed as a whole and each block picks its own entry, because cache-dit replaces the block list
  with a single wrapper during a cached forward.
  """

  def __init__(self, *args, index: int = 0, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.index = index
    self.attn1.set_processor(CausVidCausalAttnProcessor())

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
    :param temb: Timestep modulation, shaped `(B, 6, C)`.
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


class CausVidTransformer3DModel(WanTransformer3DModel):
  """Wan 2.1 transformer with CausVid's chunk-causal forward.

  Only the block class, the RoPE offset and the forward are replaced; everything else (config, fp32
  keep list, text/timestep embedding, output norm) is inherited, which also keeps the Wan tensor-
  parallel plan valid for this class.
  """

  _repeated_blocks = ["CausVidTransformerBlock"]
  _no_split_modules = ["CausVidTransformerBlock"]

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
    self.rope = CausVidRotaryPosEmbed(
      self.config.attention_head_dim,
      tuple(self.config.patch_size),
      self.config.rope_max_seq_len,
    )
    self.blocks = nn.ModuleList([
      CausVidTransformerBlock(
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
  ) -> list:
    """Allocate (or re-allocate) the per-layer KV caches for one clip.

    :param batch_size: Batch size of the request.
    :param latent_shape: Chunk latent shape `(B, C, F_chunk, H, W)`; `H`/`W` fix the frame token
      count, `F_chunk` is only used when `num_frames` is omitted.
    :param num_frames: Total latent frames of the whole clip (all chunks), which sets the cache size.
    :param device: Cache device; defaults to the transformer's device.
    :param dtype: Cache dtype; defaults to the transformer's dtype.
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
        f"CausVid-Wan needs the post-TP head count to divide by the Ulysses size, got {heads} over "
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
    self._kv_cache = [
      CausVidSelfAttentionKVCache(batch_size, total_len, local_heads,
                                  self.config.attention_head_dim, dtype, device)
      for _ in range(num_layers)
    ]
    logger.debug(f"CausVid KV cache: {num_layers} layers, {total_len} tokens/layer, "
                 f"{local_heads} heads/rank ({dtype}, {device}).")
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
    # During a cached forward cache-dit swaps `self.blocks` for a one-element wrapper that owns the
    # real stack; the clean-latent pass has to reach the real blocks, so both lists are returned.
    blocks = self.blocks
    if len(blocks) == 1 and hasattr(blocks[0], "transformer_blocks"):
      return blocks, blocks[0].transformer_blocks
    return blocks, None

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
    :param timestep: Diffusion timestep of the chunk, shaped `(B,)`.
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
    start_frame = current_start // frame_seq_len
    rotary_emb = self.rope(hidden_states, start_frame)

    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2).contiguous()

    temb, timestep_proj, encoder_hidden_states, _ = self.condition_embedder(
      timestep, encoder_hidden_states, None)
    timestep_proj = timestep_proj.unflatten(1, (6, -1))

    blocks, raw_blocks = self._split_block_list()
    if force_full_compute and raw_blocks is None and len(blocks) == 1:
      # A one-element block list we cannot look through means some other wrapper is installed; the
      # clean-latent pass would then be block-skipped and the layer K/V of this chunk would stay at
      # the noisy step, silently corrupting every later chunk.
      raise RuntimeError(
        f"Cannot bypass the block-skip wrapper of type {type(blocks[0]).__name__}: the clean-latent "
        f"pass of a causal chunk must run every block. Please report this with the cache-dit version."
      )
    if force_full_compute and raw_blocks is not None:
      for block in raw_blocks:
        hidden_states = block(
          hidden_states,
          encoder_hidden_states,
          timestep_proj,
          rotary_emb,
          kv_cache=kv_cache,
          current_start=current_start,
          current_end=current_end,
        )
    else:
      for block in blocks:
        hidden_states = block(
          hidden_states,
          encoder_hidden_states,
          timestep_proj,
          rotary_emb,
          kv_cache=kv_cache,
          current_start=current_start,
          current_end=current_end,
        )

    shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
    shift = shift.to(hidden_states.device)
    scale = scale.to(hidden_states.device)
    hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) +
                     shift).type_as(hidden_states)
    # Under context parallelism the gather hook sits on `proj_out`, i.e. on 64-wide tokens instead of
    # the 1536-wide residual stream. `norm_out` itself is per token with a per-batch modulation, so
    # running it on the local shard is exact.
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(batch_size, post_patch_num_frames, post_patch_height,
                                          post_patch_width, p_t, p_h, p_w, -1)
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    return hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)


def build_causvid_transformer_config(  # noqa: C901 - flat keyword mapping, no branching
  num_layers: int = 30,
  dim: int = 1536,
  num_attention_heads: int = 12,
  attention_head_dim: int = 128,
  ffn_dim: int = 8960,
  in_channels: int = 16,
  out_channels: int = 16,
  text_dim: int = 4096,
  freq_dim: int = 256,
  patch_size: Tuple[int, int, int] = (1, 2, 2),
  cross_attn_norm: bool = False,
  qk_norm: str = "rms_norm_across_heads",
  eps: float = 1e-6,
  rope_max_seq_len: int = 1024,
):
  """Return the `CausVidTransformer3DModel` arguments of the Wan2.1-T2V-1.3B student.

  The defaults are the released `tianweiy/CausVid` numbers; passing any of them explicitly makes the
  tiny-model tests independent of the checkpoint.

  :returns: A keyword dict accepted by :class:`CausVidTransformer3DModel`.
  """
  return dict(
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
    rope_max_seq_len=rope_max_seq_len,
  )
