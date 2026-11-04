"""MAGI-2 packed attention: varlen metadata, learned sinks and the Ulysses exchange.

Adapted from ``vllm-omni @ 7f217a1`` ``vllm_omni/diffusion/models/magi2/{attention.py,modeling_magi2.py}``
class ``Magi2Attention`` (Apache-2.0, Copyright (c) 2026 SandAI). Intentional differences:

* the ``vllm_omni`` ``Attention`` layer plus ``AttentionMetadata`` side channel are gone: the packed
  kernel is called directly with the ``[tokens, heads, head_dim]`` layout it actually consumes;
* the Ulysses all-to-all stays model-owned (uneven ``split_sizes``, fused Q/K/V exchange), which is
  why the cache-dit attention dispatcher - a batched ``[B,S,H,D]`` API without ``cu_seqlens`` - is not
  used here; a batched mask would materialize an ``S x S`` matrix for every step;
* cache-dit style (2-space indent, Sphinx docstrings).
"""

import dataclasses

import torch
import torch.nn as nn

from .common_layers import ModalityDispatcher, make_grouped_linear
from .layers import MultiModalityRMSNorm, apply_rotary_emb, varlen_attention_with_sink
from .parallel import get_magi2_ulysses_group, scatter_heads_gather_seqlen, scatter_seqlen_gather_heads

__all__ = [
  "Magi2Attention",
  "VarlenHandler",
  "packed_attention_with_sink",
  "ulysses_packed_attention_with_sink",
]


@dataclasses.dataclass(frozen=True)
class VarlenHandler:
  """Packed-sequence metadata consumed by MAGI-2 attention.

  :param cu_seqlens_q: Cumulative query lengths per packed sample.
  :param cu_seqlens_k: Cumulative key lengths per packed sample.
  :param max_seqlen_q: Longest query segment.
  :param max_seqlen_k: Longest key segment.
  """

  cu_seqlens_q: torch.Tensor = None
  cu_seqlens_k: torch.Tensor = None
  max_seqlen_q: int = None
  max_seqlen_k: int = None

  def resolved(self, q_tokens: int, k_tokens: int):
    """Resolve the cumulative lengths and segment maxima.

    :param q_tokens: Total query tokens.
    :param k_tokens: Total key tokens.
    :returns: ``(cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)``.
    """

    if self.cu_seqlens_q is None:
      cu_q = torch.tensor([0, q_tokens], device="cpu", dtype=torch.int32)
    else:
      cu_q = self.cu_seqlens_q
    if self.cu_seqlens_k is None:
      cu_k = torch.tensor([0, k_tokens], device="cpu", dtype=torch.int32)
    else:
      cu_k = self.cu_seqlens_k
    max_q = self.max_seqlen_q
    max_k = self.max_seqlen_k
    if max_q is None:
      max_q = int(torch.diff(cu_q).max().item()) if cu_q.numel() > 1 else 0
    if max_k is None:
      max_k = int(torch.diff(cu_k).max().item()) if cu_k.numel() > 1 else 0
    return cu_q, cu_k, max_q, max_k


def packed_attention_with_sink(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  varlen: VarlenHandler,
  *,
  softcap: float = -1.0,
  sink: torch.Tensor = None,
) -> torch.Tensor:
  """Run packed attention over ``[tokens, heads, head_dim]`` tensors.

  :param q: Queries.
  :param k: Keys.
  :param v: Values.
  :param varlen: Packed-sequence metadata.
  :param softcap: Logit softcap; ``<= 0`` disables it.
  :param sink: Optional learned sink logits ``[sink_tokens, heads]``.
  :returns: Attention output.
  """

  bounds_q, bounds_k, max_q, max_k = varlen.resolved(q.shape[0], k.shape[0])
  return varlen_attention_with_sink(
    q,
    k,
    v,
    cu_seqlens_q=bounds_q,
    cu_seqlens_k=bounds_k,
    max_seqlen_q=max_q,
    max_seqlen_k=max_k,
    softcap=softcap,
    sink=sink,
  )


def ulysses_packed_attention_with_sink(
  q: torch.Tensor,
  k: torch.Tensor,
  v: torch.Tensor,
  varlen: VarlenHandler,
  split_sizes,
  *,
  softcap: float = -1.0,
  sink: torch.Tensor = None,
  group=None,
) -> torch.Tensor:
  """MAGI-2 attention with the overlapping Ulysses context/head exchange.

  :param q: ``[local_tokens, world*heads, head_dim]`` queries.
  :param k: Keys, same layout as ``q``.
  :param v: Values, same layout as ``q``.
  :param varlen: Packed-sequence metadata over the FULL sequence.
  :param split_sizes: Per-rank token counts.
  :param softcap: Logit softcap.
  :param sink: Learned sink logits ``[sink_tokens, world*heads]``.
  :param group: Ulysses group; defaults to the installed one.
  :returns: ``[local_tokens, world*heads, head_dim]`` attention output.
  """

  group = group or get_magi2_ulysses_group()
  if isinstance(split_sizes, torch.Tensor):
    split_sizes = [int(value) for value in split_sizes.detach().cpu().tolist()]
  if group.world_size > 1:
    q, k, v = scatter_heads_gather_seqlen((q, k, v), split_sizes, group)
    if sink is not None:
      if sink.shape[-1] % group.world_size:
        raise ValueError("attention sink heads must divide across Ulysses ranks")
      sink = torch.chunk(sink, group.world_size, dim=-1)[group.rank].contiguous()
  output = packed_attention_with_sink(q, k, v, varlen, softcap=softcap, sink=sink)
  if group.world_size > 1:
    output = scatter_seqlen_gather_heads(output.contiguous(), split_sizes, group)
  return output


class Magi2Attention(nn.Module):
  """MAGI-2 attention: grouped Q/KV projections, per-head gating and learned sinks.

  The block owns its own Ulysses communication (see :func:`ulysses_packed_attention_with_sink`), so
  cache-dit must not add a second collective on top of it.

  :param config: Preview architecture config.
  :param num_modality: Number of per-modality weight groups (3 on multimodal layers, else 1).
  """

  def __init__(self, config, *, num_modality: int) -> None:
    super().__init__()
    self.config = config
    self.num_modality = num_modality
    self.head_dim = config.head_dim
    global_q_size = config.num_heads_q * self.head_dim
    global_kv_size = config.num_heads_kv * self.head_dim

    self.pre_norm = MultiModalityRMSNorm(config.hidden_size, num_modality=num_modality)
    self.linear_g = make_grouped_linear(
      config.hidden_size,
      config.num_heads_q,
      num_experts=num_modality,
      bias=False,
      dtype=config.params_dtype,
      parallel_mode="column",
    )
    self.linear_qkv = make_grouped_linear(
      config.hidden_size,
      global_q_size + 2 * global_kv_size,
      num_experts=num_modality,
      bias=False,
      dtype=config.params_dtype,
      parallel_mode="column",
      qkv_splits=(global_q_size, global_kv_size, global_kv_size),
    )
    self.linear_proj = make_grouped_linear(
      global_q_size,
      config.hidden_size,
      num_experts=num_modality,
      bias=False,
      dtype=config.params_dtype,
      parallel_mode="row",
    )
    self.tp_group = self.linear_qkv.tp_group
    self.num_heads_q = config.num_heads_q // self.tp_group.world_size
    self.num_heads_kv = config.num_heads_kv // self.tp_group.world_size
    self.q_size = self.num_heads_q * self.head_dim
    self.kv_size = self.num_heads_kv * self.head_dim
    self.sinks = nn.Parameter(
      torch.empty(config.attention_sink_tokens, self.num_heads_q, dtype=torch.float32))
    if self.tp_group.world_size > 1:
      self.sinks.checkpoint_weight_transform = self._shard_sinks
    self.q_norm = MultiModalityRMSNorm(self.head_dim,
                                       num_modality=num_modality,
                                       out_dtype=torch.float32)
    self.k_norm = MultiModalityRMSNorm(self.head_dim,
                                       num_modality=num_modality,
                                       out_dtype=torch.float32)

  def shard_for_tp(self, tp_group) -> None:
    """Re-shard the projections, the sink and the head counts in place.

    :param tp_group: Parameter-parallel group.
    """

    if tp_group.world_size <= 1:
      return
    if self.tp_group.world_size > 1:
      raise RuntimeError("attention is already sharded; re-sharding needs a fresh module")

    self.tp_group = tp_group
    self.linear_g.shard_for_tp(tp_group)
    self.linear_qkv.shard_for_tp(tp_group)
    self.linear_proj.shard_for_tp(tp_group)
    self.num_heads_q = self.config.num_heads_q // tp_group.world_size
    self.num_heads_kv = self.config.num_heads_kv // tp_group.world_size
    self.q_size = self.num_heads_q * self.head_dim
    self.kv_size = self.num_heads_kv * self.head_dim
    start = tp_group.rank * self.num_heads_q
    sinks = self.sinks.data[:, start:start + self.num_heads_q].contiguous()
    self.sinks = nn.Parameter(sinks, requires_grad=self.sinks.requires_grad)
    self.sinks.checkpoint_weight_transform = self._shard_sinks

  def _shard_sinks(self, checkpoint_tensor: torch.Tensor) -> torch.Tensor:
    """Slice the sink heads owned by this rank.

    :param checkpoint_tensor: ``[sink_tokens, heads_q]`` checkpoint tensor.
    :returns: This rank's ``[sink_tokens, heads_q / tp]`` slice.
    """

    if tuple(checkpoint_tensor.shape) == tuple(self.sinks.shape):
      return checkpoint_tensor
    expected = (self.config.attention_sink_tokens, self.config.num_heads_q)
    if tuple(checkpoint_tensor.shape) != expected:
      raise ValueError(
        f"attention sinks have shape {tuple(checkpoint_tensor.shape)}, expected {expected}")
    start = self.tp_group.rank * self.num_heads_q
    return checkpoint_tensor[:, start:start + self.num_heads_q]

  def project(self, hidden_states: torch.Tensor, rope: torch.Tensor,
              modality_dispatcher: ModalityDispatcher):
    """Project hidden states into Q/K/V plus the per-head gate.

    :param hidden_states: ``[tokens, hidden]`` block input.
    :param rope: ``[tokens, head_dim]`` rotary table.
    :param modality_dispatcher: Modality grouping of the packed tokens.
    :returns: ``(q, k, v, gates)`` with ``[tokens, heads, head_dim]`` q/k/v.
    """

    normalized = self.pre_norm(hidden_states, modality_dispatcher)
    gates = self.linear_g(normalized, modality_dispatcher)
    qkv = self.linear_qkv(normalized, modality_dispatcher)
    q, k, v = torch.split(qkv, (self.q_size, self.kv_size, self.kv_size), dim=-1)
    q = q.view(-1, self.num_heads_q, self.head_dim)
    k = k.view(-1, self.num_heads_kv, self.head_dim)
    v = v.view(-1, self.num_heads_kv, self.head_dim)
    gates = gates.view(-1, self.num_heads_q, 1)

    q = self.q_norm(q, modality_dispatcher)
    k = self.k_norm(k, modality_dispatcher)
    q = modality_dispatcher.inverse_permute(q)
    k = modality_dispatcher.inverse_permute(k)
    v = modality_dispatcher.inverse_permute(v)
    sin, cos = rope.tensor_split(2, dim=-1)
    # The released implementation normalizes Q/K in FP32, applies RoPE and only then converts Q/K/V
    # to the checkpoint dtype at the attention boundary.
    q = apply_rotary_emb(q, cos, sin).to(self.config.params_dtype)
    k = apply_rotary_emb(k, cos, sin).to(self.config.params_dtype)
    v = v.to(self.config.params_dtype)
    return q, k, v, gates

  def attend(self, q, k, v, varlen_handler: VarlenHandler, cp_split_sizes) -> torch.Tensor:
    """Run packed attention with the model-owned Ulysses exchange.

    :param q: Queries.
    :param k: Keys.
    :param v: Values.
    :param varlen_handler: Packed-sequence metadata.
    :param cp_split_sizes: Per-rank token counts of the full sequence.
    :returns: Attention output.
    """

    return ulysses_packed_attention_with_sink(
      q,
      k,
      v,
      varlen_handler,
      cp_split_sizes,
      softcap=self.config.attention_softcap,
      sink=self.sinks,
    )

  def output(self, attention: torch.Tensor, gates: torch.Tensor,
             modality_dispatcher: ModalityDispatcher):
    """Apply the head gate and the output projection.

    :param attention: ``[tokens, heads, head_dim]`` attention output.
    :param gates: ``[tokens, heads, 1]`` per-head gate logits.
    :param modality_dispatcher: Modality grouping of the packed tokens.
    :returns: ``[tokens, hidden]`` projection output.
    """

    # Only the token count varies between calls; pinning the head count and head size lets a
    # dynamic-shape compile index the attention output with constants.
    torch._check(attention.shape[1:-1].numel() == self.num_heads_q)
    torch._check(attention.shape[-1] == self.head_dim)
    output = modality_dispatcher.permute(attention)
    output = output * torch.sigmoid(gates)
    output = output.reshape(-1, self.q_size).to(self.config.params_dtype)
    return self.linear_proj(output, modality_dispatcher)
