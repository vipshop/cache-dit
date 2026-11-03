"""Attention modules of the LongCat-Video-Avatar transformer.

Copied from `meituan-longcat/LongCat-Video` (main, 2026-11-03),
`longcat_video/modules/avatar/attention.py` and `longcat_video/modules/attention.py`.

Intentional differences from the upstream copy:

* every attention call goes through `cache_dit.attention._dispatch_attention_fn` instead of
  flash-attn / flash-attn-3 / xformers / block-sparse kernels, so the backend is selected once by
  cache-dit and context parallelism can inject its all-to-all;
* the varlen text cross-attention (`flash_attn_varlen_func` with `cu_seqlens`) is reproduced with a
  per-sample loop over the padded batch: cache-dit's dispatcher takes dense tensors, and the batch is
  one or two samples (conditional / unconditional), so the loop costs nothing;
* the multi-speaker branch (per-head 1-D RoPE driven by `x_ref_attn_map`) is not ported — the
  pipeline raises instead of silently rendering a single speaker;
* `num_cond_latents` is expressed in **local** frames, because under context parallelism each rank
  only owns the conditional frames inside its own frame range.
"""

from typing import Optional

import torch
import torch.nn as nn

from ....attention import _dispatch_attention_fn
from .blocks import RMSNorm_FP32
from .rope_3d import RotaryPositionalEmbedding


def _attend(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  scale: float,
  cp_config=None,
) -> torch.Tensor:
  """Run scaled dot-product attention through cache-dit's dispatcher.

  :param q: `[B, heads, S, head_dim]` queries.
  :param k: `[B, heads, S_kv, head_dim]` keys.
  :param v: `[B, heads, S_kv, head_dim]` values.
  :param scale: softmax scale.
  :param cp_config: cache-dit context-parallel config, or `None` for a replicated attention.
  :returns: `[B, heads, S, head_dim]` attention output.
  """

  out = _dispatch_attention_fn(
    q.transpose(1, 2).contiguous(),
    k.transpose(1, 2).contiguous(),
    v.transpose(1, 2).contiguous(),
    dropout_p=0.0,
    is_causal=False,
    scale=scale,
    cp_config=cp_config,
  )
  return out.transpose(1, 2).contiguous()


class LongCatAvatarSelfAttention(nn.Module):
  """Self-attention with packed QKV, per-head Q/K RMSNorm and 3-D RoPE.

  :param dim: transformer width.
  :param num_heads: number of heads.
  :param cp_split_hw: ignored, kept for constructor parity with the upstream module.
  """

  def __init__(self, dim: int, num_heads: int, cp_split_hw=None) -> None:
    super().__init__()
    assert dim % num_heads == 0, "dim must be divisible by num_heads"
    self.dim = dim
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.scale = self.head_dim ** -0.5

    self.qkv = nn.Linear(dim, dim * 3, bias=True)
    self.q_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
    self.k_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
    self.proj = nn.Linear(dim, dim)

    self.rope_3d = RotaryPositionalEmbedding(self.head_dim)

  def forward(
    self,
    hidden_states: torch.Tensor,
    shape,
    num_cond_latents: int = 0,
    local_frame_range: Optional[tuple] = None,
    cp_config=None,
  ) -> torch.Tensor:
    """Attend over the (local) token sequence.

    :param hidden_states: `[B, N, dim]` activations of the local token range.
    :param shape: `(num_frames, height, width)` of the **global** token grid.
    :param num_cond_latents: number of **local** leading frames that are conditioning latents; they
      attend only among themselves. Context parallelism is rejected upstream when there is one,
      because only the rank owning that frame would take this branch and the ranks would then issue
      a different number of all-to-all collectives (deadlock).
    :param local_frame_range: `(start, end)` frames owned by this rank, or `None` when the whole
      grid is local.
    :param cp_config: cache-dit context-parallel config when the sequence is split, else `None`.
    :returns: `[B, N, dim]` attention output.
    """

    b, n, c = hidden_states.shape
    qkv = self.qkv(hidden_states).view(b, n, 3, self.num_heads,
                                       self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q = self.q_norm(q)
    k = self.k_norm(k)

    q, k = self.rope_3d(q, k, shape, local_frame_range)

    tokens_per_frame = shape[1] * shape[2]
    if num_cond_latents and num_cond_latents > 0:
      cond_tokens = num_cond_latents * tokens_per_frame
      q_cond = q[:, :, :cond_tokens].contiguous()
      k_cond = k[:, :, :cond_tokens].contiguous()
      v_cond = v[:, :, :cond_tokens].contiguous()
      x_cond = _attend(q_cond, k_cond, v_cond, self.scale, cp_config=None)
      q_noise = q[:, :, cond_tokens:].contiguous()
      x_noise = _attend(q_noise, k, v, self.scale, cp_config=cp_config)
      x = torch.cat([x_cond, x_noise], dim=2).contiguous()
    else:
      x = _attend(q, k, v, self.scale, cp_config=cp_config)

    x = x.transpose(1, 2).reshape(b, n, c)
    return self.proj(x)


class LongCatAvatarTextCrossAttention(nn.Module):
  """Text cross-attention with packed KV and per-head Q/K RMSNorm.

  :param dim: transformer width.
  :param num_heads: number of heads.
  """

  def __init__(self, dim, num_heads):
    super().__init__()
    assert dim % num_heads == 0, "dim must be divisible by num_heads"
    self.dim = dim
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.scale = self.head_dim ** -0.5

    self.q_linear = nn.Linear(dim, dim)
    self.kv_linear = nn.Linear(dim, dim * 2)
    self.proj = nn.Linear(dim, dim)

    self.q_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
    self.k_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)

  def _per_sample(self, x: torch.Tensor, cond: torch.Tensor, kv_seqlen) -> torch.Tensor:
    """Attend one sample against its own (unpadded) text tokens.

    :param x: `[N, dim]` queries of one sample.
    :param cond: `[M, dim]` valid text tokens of one sample.
    :param kv_seqlen: how many of `cond` belong to this sample.
    :returns: `[N, dim]` attention output.
    """

    q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
    kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
    k, v = kv.unbind(2)
    q = self.q_norm(q)
    k = self.k_norm(k)
    out = _dispatch_attention_fn(q,
                                 k,
                                 v,
                                 dropout_p=0.0,
                                 is_causal=False,
                                 scale=self.scale,
                                 cp_config=None)
    return out.reshape(-1, self.dim)

  def forward(
    self,
    hidden_states: torch.Tensor,
    cond: torch.Tensor,
    kv_seqlen,
    num_cond_latents: int = 0,
    tokens_per_frame: Optional[int] = None,
  ) -> torch.Tensor:
    """Attend the (local) video tokens against the packed text tokens.

    :param hidden_states:`[B, N, dim]` activations of the local token range.
    :param cond:`[1, sum(kv_seqlen), dim]` valid text tokens of the whole batch.
    :param kv_seqlen: list of valid text lengths, one per sample.
    :param num_cond_latents: number of **local** leading frames that are conditioning latents; their
        cross-attention output is zeroed instead of computed.
    :param tokens_per_frame: tokens of one latent frame, needed when `num_cond_latents > 0`.
    :returns:`[B, N, dim]` cross-attention output.
    """

    b, n, c = hidden_states.shape
    if num_cond_latents and num_cond_latents > 0:
      assert tokens_per_frame is not None, "tokens_per_frame is required with conditioning latents"
      cond_tokens = num_cond_latents * tokens_per_frame
      x_noise = hidden_states[:, cond_tokens:]
      out_noise = self._attend_batch(x_noise, cond, kv_seqlen)
      zeros = torch.zeros((b, cond_tokens, c), dtype=out_noise.dtype, device=out_noise.device)
      return torch.cat([zeros, out_noise], dim=1).contiguous()
    return self._attend_batch(hidden_states, cond, kv_seqlen)

  def _attend_batch(self, hidden_states: torch.Tensor, cond: torch.Tensor,
                    kv_seqlen) -> torch.Tensor:
    """Attend a batch whose samples may have different valid text lengths.

    :param hidden_states:`[B, N, dim]` queries.
    :param cond:`[1, sum(kv_seqlen), dim]` valid text tokens.
    :param kv_seqlen: list of valid text lengths, one per sample.
    :returns:`[B, N, dim]` attention output.
    """

    b, n, c = hidden_states.shape
    lengths = [int(v) for v in kv_seqlen]
    if len(set(lengths)) == 1:
      # Uniform text length: a plain batched attention, no varlen bookkeeping needed.
      q = self.q_norm(self.q_linear(hidden_states).view(b, n, self.num_heads, self.head_dim))
      kv = self.kv_linear(cond.squeeze(0)).view(1, -1, 2, self.num_heads, self.head_dim)
      k, v = kv.unbind(2)
      k = self.k_norm(k)
      k = k.expand(b, -1, -1, -1)
      v = v.expand(b, -1, -1, -1)
      out = _dispatch_attention_fn(q,
                                   k,
                                   v,
                                   dropout_p=0.0,
                                   is_causal=False,
                                   scale=self.scale,
                                   cp_config=None)
      return self.proj(out.reshape(b, n, c))

    outputs = []
    offset = 0
    for i, length in enumerate(lengths):
      sample_cond = cond[0, offset:offset + length]
      outputs.append(self._per_sample(hidden_states[i], sample_cond, length))
      offset += length
    return self.proj(torch.stack(outputs, dim=0))


class LongCatAvatarAudioCrossAttention(nn.Module):
  """Per-frame audio cross-attention.

  Each latent frame attends to its own `context_tokens` audio tokens, so the frame axis becomes the
  batch axis of the attention call: the frame count is the only sequence structure the module relies
  on, which keeps it usable under context parallelism (each rank attends its own frames).

  :param dim: transformer width.
  :param encoder_hidden_states_dim: width of one audio token.
  :param num_heads: number of heads.
  :param qkv_bias: whether the projections carry a bias.
  :param qk_norm: whether to RMSNorm Q and K per head.
  :param attn_drop: attention dropout (inference keeps it at 0).
  :param proj_drop: output dropout (inference keeps it at 0).
  :param eps: epsilon of the Q/K norms.
  """

  def __init__(
    self,
    dim: int,
    encoder_hidden_states_dim: int,
    num_heads: int,
    qkv_bias: bool,
    qk_norm: bool,
    attn_drop: float = 0.0,
    proj_drop: float = 0.0,
    eps: float = 1e-6,
  ) -> None:
    super().__init__()
    assert dim % num_heads == 0, "dim must be divisible by num_heads"
    self.dim = dim
    self.encoder_hidden_states_dim = encoder_hidden_states_dim
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.scale = self.head_dim ** -0.5

    self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
    self.q_norm = RMSNorm_FP32(self.head_dim, eps=eps) if qk_norm else nn.Identity()

    self.attn_drop = nn.Dropout(attn_drop)
    self.proj = nn.Linear(dim, dim)
    self.proj_drop = nn.Dropout(proj_drop)

    self.kv_linear = nn.Linear(encoder_hidden_states_dim, dim * 2, bias=qkv_bias)
    self.k_norm = RMSNorm_FP32(self.head_dim, eps=eps) if qk_norm else nn.Identity()

  def _process_cross_attn(self, x: torch.Tensor, cond: torch.Tensor, frames: int) -> torch.Tensor:
    """Attend `frames * S` video tokens against `frames` groups of audio tokens.

    :param x: `[B, frames * S, dim]` video tokens.
    :param cond: `[B, frames, M, dim_audio]` audio tokens.
    :param frames: number of latent frames in `x`.
    :returns: `[B, frames * S, dim]` attention output.
    """

    out_dtype = x.dtype
    b, n, c = x.shape
    tokens_per_frame = n // frames
    x = x.reshape(b * frames, tokens_per_frame, c)
    cond = cond.reshape(b * frames, cond.shape[2], cond.shape[3])

    q = self.q_linear(x).view(b * frames, tokens_per_frame, self.num_heads, self.head_dim)
    q = self.q_norm(q)

    # `_dispatch_attention_fn` takes `[B, S, H, D]`, so the KV pair is split on its own axis
    # instead of being permuted to the `[B, H, S, D]` layout the upstream flash-attn calls want.
    kv = self.kv_linear(cond).view(b * frames, -1, 2, self.num_heads, self.head_dim)
    k, v = kv.unbind(2)
    k = self.k_norm(k)

    out = _dispatch_attention_fn(q,
                                 k,
                                 v,
                                 dropout_p=0.0,
                                 is_causal=False,
                                 scale=self.scale,
                                 cp_config=None)
    out = out.reshape(b * frames, tokens_per_frame, c)
    out = self.proj(out)
    out = self.proj_drop(out)
    return out.reshape(b, n, c).type(out_dtype)

  def forward(
    self,
    hidden_states: torch.Tensor,
    cond: torch.Tensor,
    frames: int,
    num_cond_latents: int = 0,
  ):
    """Attend the video tokens against the per-frame audio tokens.

    :param hidden_states: `[B, frames * S, dim]` video tokens of the local frame range.
    :param cond: `[B, frames, M, dim_audio]` audio tokens of the same frames.
    :param frames: number of local latent frames.
    :param num_cond_latents: number of **local** leading frames that are conditioning latents; they
      receive no audio update (the pipeline returns zeros for them).
    :returns: `(cond_output, noise_output)`; `cond_output` is `None` when there is no conditioning
      frame on this rank, and the pair is `(None, output)` in that case.
    """

    b, n, c = hidden_states.shape
    if not num_cond_latents:
      return None, self._process_cross_attn(hidden_states, cond, frames)

    tokens_per_frame = n // frames
    cond_tokens = num_cond_latents * tokens_per_frame
    x_noise = hidden_states[:, cond_tokens:]
    cond_noise = cond[:, num_cond_latents:]
    output_noise = self._process_cross_attn(x_noise, cond_noise, frames - num_cond_latents)
    output_cond = torch.zeros((b, cond_tokens, c),
                              dtype=output_noise.dtype,
                              device=output_noise.device)
    return output_cond, output_noise
