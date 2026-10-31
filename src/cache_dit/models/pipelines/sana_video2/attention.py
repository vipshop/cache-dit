"""Attention layers of SANA-Video 2.0.

Copied from NVlabs/Sana (Apache-2.0, Copyright 2026 NVIDIA CORPORATION & AFFILIATES),
``diffusion/model/nets/sana_video2_blocks.py`` and ``sana_blocks.py``
(``MultiHeadCrossAttention``), reformatted to this repository's style. Parameter names match the
released checkpoint, which is loaded with ``strict=True``.

Intentional differences from the upstream copy:

* xFormers is not wired in. The official inference entry point sets ``DISABLE_XFORMERS=1``, so the
  vendored copy implements only that path (masked SDPA), which keeps the numerics identical to the
  reference run.
* Softmax and cross attention take the context-parallel path through
  :func:`cache_dit.attention._dispatch_attention_fn` instead of calling SDPA directly, and the
  gated linear attention all-reduces its per-head ``value @ key^T`` state across the CP ranks: the
  state is a sum over *all* tokens, so a sequence shard alone is not a valid partial result.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn

from .components import _RMSNorm, apply_rope_channel_first, apply_rope_token_first


def _cp_group(cp_config: Any) -> Optional[Any]:
  """Resolve the process group that owns the context-parallel collectives.

  The group must cover **every** rank that holds a token shard, which is the flattened
  ring x ulysses mesh, not one of the two sub-meshes: the token splitter shards over
  `_flattened_mesh`, so reducing over the Ulysses sub-mesh alone would leave a hybrid CP run with
  only half of the sequence state.

  :param cp_config: cache-dit context-parallel config, or `None`.
  :returns: The process group covering all CP ranks, or `None` when CP is inactive.
  """
  if cp_config is None:
    return None
  ulysses = int(getattr(cp_config, "ulysses_degree", 1) or 1)
  ring = int(getattr(cp_config, "ring_degree", 1) or 1)
  if ulysses <= 1 and ring <= 1:
    return None
  # `_flattened_mesh` is the ring x ulysses mesh and collapses to the single sub-mesh when one of
  # the two degrees is 1, so all three cases share one communicator.
  mesh = getattr(cp_config, "_flattened_mesh", None)
  return None if mesh is None else mesh.get_group()


def _sdpa(query: torch.Tensor,
          key: torch.Tensor,
          value: torch.Tensor,
          attn_mask: Optional[torch.Tensor] = None,
          cp_config: Any = None) -> torch.Tensor:
  """Run softmax attention, through the CP dispatcher when CP is active.

  :param query: `[B, S, heads, head_dim]`.
  :param key: `[B, S_kv, heads, head_dim]`.
  :param value: `[B, S_kv, heads, head_dim]`.
  :param attn_mask: Optional additive mask of shape `[B, 1, 1, S_kv]`.
  :param cp_config: cache-dit context-parallel config, or `None`.
  :returns: Attention output `[B, S, heads, head_dim]`.
  """
  from ....attention import _dispatch_attention_fn
  return _dispatch_attention_fn(
    query,
    key,
    value,
    attn_mask=attn_mask,
    dropout_p=0.0,
    is_causal=False,
    cp_config=cp_config,
  )


class GatedLinearAttention(nn.Module):
  """Bidirectional gated linear attention used by the non-anchor layers.

  ``out = o_norm((v @ (k * sigmoid(beta))^T) @ q)``, with no ``1 / sqrt(d)`` scaling, computed in
  fp32 when ``fp32_attention`` is set.

  :param dim: Hidden size.
  :param head_dim: Per-head channel count; ``heads = dim // head_dim``.
  :param qk_norm: Whether to RMS-normalize the fused q/k projections.
  :param norm_eps: Epsilon of those RMSNorm layers.
  :param fp32_attention: Accumulate the two matmuls in fp32.
  """

  def __init__(self,
               dim: int,
               head_dim: int,
               qk_norm: bool = True,
               norm_eps: float = 1e-5,
               fp32_attention: bool = True):
    super().__init__()
    if dim % head_dim:
      raise ValueError(f"dim={dim} must be divisible by head_dim={head_dim}.")
    self.heads = dim // head_dim
    self.dim = head_dim
    self.fp32_attention = fp32_attention
    self.qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.proj = nn.Linear(dim, dim, bias=True)
    if qk_norm:
      self.q_norm = _RMSNorm(dim, scale_factor=1.0, eps=norm_eps)
      self.k_norm = _RMSNorm(dim, scale_factor=1.0, eps=norm_eps)
    else:
      self.q_norm = nn.Identity()
      self.k_norm = nn.Identity()
    self.beta_proj = nn.Linear(dim, self.heads, bias=True)
    self.output_gate = nn.Linear(dim, dim, bias=True)
    self.o_norm = _RMSNorm(head_dim, scale_factor=1.0, eps=norm_eps, norm_dim=-2)

  def forward(self,
              x: torch.Tensor,
              rotary_emb: Optional[torch.Tensor] = None,
              cp_config: Any = None) -> torch.Tensor:
    """Attend over the (possibly sharded) token axis.

    :param x: Tokens `[B, S, C]`.
    :param rotary_emb: Complex RoPE table for the tokens at hand.
    :param cp_config: cache-dit context-parallel config, or `None`.
    :returns: Attention output `[B, S, C]`.
    """
    batch, tokens, channels = x.shape
    q, k, v = self.qkv(x).reshape(batch, tokens, 3, channels).unbind(2)
    output_dtype = q.dtype
    q = self.q_norm(q).transpose(-1, -2).reshape(batch, self.heads, self.dim, tokens)
    k = self.k_norm(k).transpose(-1, -2).reshape(batch, self.heads, self.dim, tokens)
    v = v.transpose(-1, -2).reshape(batch, self.heads, self.dim, tokens)
    if rotary_emb is not None:
      q = apply_rope_channel_first(q, rotary_emb)
      k = apply_rope_channel_first(k, rotary_emb)
    beta = torch.sigmoid(self.beta_proj(x)).transpose(1, 2).unsqueeze(2)
    k_gated = k * beta
    if self.fp32_attention:
      q, k_gated, v = q.float(), k_gated.float(), v.float()
    key_value = torch.matmul(v, k_gated.transpose(-1, -2))
    group = _cp_group(cp_config)
    if group is not None:
      # The state is a sum over all tokens of the full sequence, so every rank needs the sum, not
      # just its local shard. One collective per layer, on the already fp32 state.
      import torch.distributed as dist
      dist.all_reduce(key_value, op=dist.ReduceOp.SUM, group=group)
    out = torch.matmul(key_value, q).to(output_dtype)
    out = self.o_norm(out)
    out = out.reshape(batch, channels, tokens).permute(0, 2, 1)
    out = out * torch.sigmoid(self.output_gate(x))
    return self.proj(out)


class GatedSoftmaxAttention(nn.Module):
  """Dense anchor attention with RoPE and a learned output gate.

  :param dim: Hidden size.
  :param head_dim: Per-head channel count; ``heads = dim // head_dim``.
  :param qk_norm: Whether to RMS-normalize the fused q/k projections.
  :param norm_eps: Epsilon of those RMSNorm layers.
  :param fp32_attention: Run the SDPA call in fp32.
  """

  def __init__(self,
               dim: int,
               head_dim: int,
               qk_norm: bool = True,
               norm_eps: float = 1e-5,
               fp32_attention: bool = True):
    super().__init__()
    if dim % head_dim:
      raise ValueError(f"dim={dim} must be divisible by head_dim={head_dim}.")
    self.heads = dim // head_dim
    self.dim = head_dim
    self.fp32_attention = fp32_attention
    self.qkv = nn.Linear(dim, 3 * dim, bias=False)
    self.proj = nn.Linear(dim, dim, bias=True)
    if qk_norm:
      self.q_norm = _RMSNorm(dim, scale_factor=1.0, eps=norm_eps)
      self.k_norm = _RMSNorm(dim, scale_factor=1.0, eps=norm_eps)
    else:
      self.q_norm = nn.Identity()
      self.k_norm = nn.Identity()
    self.output_gate = nn.Linear(dim, dim, bias=True)

  def forward(self,
              x: torch.Tensor,
              rotary_emb: Optional[torch.Tensor] = None,
              cp_config: Any = None) -> torch.Tensor:
    """Dense attention over the (possibly sharded) token axis.

    :param x: Tokens `[B, S, C]`.
    :param rotary_emb: Complex RoPE table for the tokens at hand.
    :param cp_config: cache-dit context-parallel config, or `None`.
    :returns: Attention output `[B, S, C]`.
    """
    batch, tokens, channels = x.shape
    q, k, v = self.qkv(x).reshape(batch, tokens, 3, channels).unbind(2)
    output_dtype = q.dtype
    q = self.q_norm(q).reshape(batch, tokens, self.heads, self.dim)
    k = self.k_norm(k).reshape(batch, tokens, self.heads, self.dim)
    v = v.reshape(batch, tokens, self.heads, self.dim)
    if rotary_emb is not None:
      q = apply_rope_token_first(q, rotary_emb)
      k = apply_rope_token_first(k, rotary_emb)
    if self.fp32_attention:
      q, k, v = q.float(), k.float(), v.float()
    out = _sdpa(q, k, v, cp_config=cp_config)
    out = out.reshape(batch, tokens, channels).to(output_dtype)
    out = out * torch.sigmoid(self.output_gate(x))
    return self.proj(out)


class MultiHeadCrossAttention(nn.Module):
  """Text cross-attention; the key/value side is the replicated caption sequence.

  :param dim: Hidden size.
  :param num_heads: Number of attention heads (equals the hidden size / head dim).
  :param qk_norm: Whether to RMS-normalize the q and k projections.
  """

  def __init__(self, dim: int, num_heads: int, qk_norm: bool = False):
    super().__init__()
    if dim % num_heads:
      raise ValueError(f"d_model={dim} must be divisible by num_heads={num_heads}.")
    self.d_model = dim
    self.num_heads = num_heads
    self.head_dim = dim // num_heads
    self.q_linear = nn.Linear(dim, dim)
    self.kv_linear = nn.Linear(dim, 2 * dim)
    self.proj = nn.Linear(dim, dim)
    if qk_norm:
      self.q_norm = _RMSNorm(dim, scale_factor=1.0, eps=1e-6)
      self.k_norm = _RMSNorm(dim, scale_factor=1.0, eps=1e-6)
    else:
      self.q_norm = nn.Identity()
      self.k_norm = nn.Identity()

  def forward(self,
              x: torch.Tensor,
              cond: torch.Tensor,
              mask: Optional[torch.Tensor] = None,
              cp_config: Any = None) -> torch.Tensor:
    """Attend from latent tokens to caption tokens.

    :param x: Latent tokens `[B, S, C]`.
    :param cond: Caption features `[B, L, C]`, already projected to the hidden size.
    :param mask: Optional binary caption mask `[B, L]`; padding is masked out.
    :param cp_config: Ignored. The caption side is replicated on every rank, so cross-attention must
      stay local: routing it through the context-parallel dispatcher would all-to-all the key/value
      side and concatenate the caption sequence `cp_size` times.
    :returns: Cross-attention delta `[B, S, C]`.
    """
    batch, _, channels = x.shape
    q = self.q_linear(x)
    kv = self.kv_linear(cond).view(batch, -1, 2, channels)
    k, v = kv.unbind(2)
    q = self.q_norm(q).view(batch, -1, self.num_heads, self.head_dim)
    k = self.k_norm(k).view(batch, -1, self.num_heads, self.head_dim)
    v = v.view(batch, -1, self.num_heads, self.head_dim)
    attn_mask = None
    if mask is not None and mask.ndim == 2:
      attn_mask = ((1 - mask.to(q.dtype)) * -10000.0)[:, None, None, :]
    out = _sdpa(q, k, v, attn_mask=attn_mask, cp_config=None)
    return self.proj(out.reshape(batch, -1, channels))


__all__ = [
  "GatedLinearAttention",
  "GatedSoftmaxAttention",
  "MultiHeadCrossAttention",
]
