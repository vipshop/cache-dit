"""Sequence-sharding helpers of the vendored HY-Nomas DiT.

The model splits *video* tokens across context-parallel ranks while the text stream stays
replicated, and attention runs over the concatenation ``[history | video | text]``.  That is the
same head/token exchange Ulysses performs, but the keys and values carry a persistent history prefix
and the attention kernel is a mask built from that history, so the exchange is driven here and the
kernel is owned by :class:`~.kv_cache.NomasAttentionPlan` (the single place allowed to bypass
cache-dit's attention dispatcher).

Every collective takes an explicit ``group``: the context-parallel sub-mesh.  Using the global
process group would divide by ``tp_size * sp_size`` on a hybrid ``tp × cp`` run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from ....distributed.communicate.primitives import (
  _all_to_all_single_o_async,
  _all_to_all_single_qkv_async,
)


@dataclass(frozen=True)
class NomasParallelInfo:
  """Per-rank shard geometry of the DiT.

  :param tp_size: Tensor-parallel size (heads are already divided in the module metadata).
  :param sp_size: Context-parallel (Ulysses) size.
  :param sp_rank: This rank's position inside the context-parallel sub-mesh.
  :param group: The context-parallel process group; ``None`` when context parallelism is off.
  """

  tp_size: int = 1
  sp_size: int = 1
  sp_rank: int = 0
  group: Optional[dist.ProcessGroup] = None

  @property
  def head_shards(self) -> int:
    """Total number of head shards the per-rank projections carry."""
    return self.tp_size * self.sp_size

  def local_heads(self, num_attention_heads: int) -> int:
    """Heads each rank holds for the persistent KV cache.

    :param num_attention_heads: Global attention head count before any sharding.
    :returns: Number of heads per rank.
    """
    if num_attention_heads % self.head_shards:
      raise ValueError(f"{num_attention_heads} attention heads are not divisible by "
                       f"tp_size * sp_size = {self.head_shards}.")
    return num_attention_heads // self.head_shards


def local_token_range(num_tokens: int, info: NomasParallelInfo) -> Tuple[int, int]:
  """Contiguous video-token range owned by this rank.

  :param num_tokens: Video token count of the whole segment.
  :param info: Shard geometry.
  :returns: ``(start, stop)`` in absolute token coordinates.
  """
  if num_tokens % info.sp_size:
    raise ValueError(f"{num_tokens} video tokens are not divisible by the context-parallel "
                     f"degree {info.sp_size}; change the resolution or the segment length.")
  local = num_tokens // info.sp_size
  return info.sp_rank * local, (info.sp_rank + 1) * local


def ulysses_all_to_all(
  x: torch.Tensor,
  info: NomasParallelInfo,
  scatter_dim: int,
  gather_dim: int,
) -> torch.Tensor:
  """Head/token all-to-all around attention.

  ``(2, 1)`` turns ``(B, S_local, H_global, D)`` into ``(B, S_global, H_local, D)``; ``(1, 2)``
  turns the attention output back into ``(B, S_local, H_global, D)``.

  :param x: Query, key or value tensor.
  :param info: Shard geometry.
  :param scatter_dim: Dimension split across ranks.
  :param gather_dim: Dimension gathered from ranks.
  :returns: The exchanged tensor.
  """
  if info.sp_size == 1:
    return x
  if info.group is None:
    raise RuntimeError("Context parallelism is enabled but the CP process group is missing; the "
                       "planner must store it on the transformer.")
  if (scatter_dim, gather_dim) == (2, 1):
    if x.shape[2] % info.sp_size:
      # The shared primitive would pad the head axis instead of failing, and the reverse exchange
      # does not unpad, so ranks would end up with different head counts.
      raise ValueError(f"{x.shape[2]} heads are not divisible by the context-parallel degree "
                       f"{info.sp_size}; padding heads is not supported by this model")
    return _all_to_all_single_qkv_async(x, info.group)()
  if (scatter_dim, gather_dim) == (1, 2):
    return _all_to_all_single_o_async(x, info.group)()
  raise ValueError(f"Unsupported Ulysses exchange ({scatter_dim} -> {gather_dim}).")


def sp_all_gather(x: torch.Tensor, info: NomasParallelInfo, dim: int) -> torch.Tensor:
  """All-gather along ``dim`` inside the context-parallel sub-mesh.

  Used for the replicated text stream, whose per-rank head slices have to be recombined.

  :param x: Per-rank tensor.
  :param info: Shard geometry.
  :param dim: Dimension to concatenate.
  :returns: The gathered tensor.
  """
  if info.sp_size == 1:
    return x
  if info.group is None:
    raise RuntimeError("Context parallelism is enabled but the CP process group is missing.")
  x = x.contiguous()
  parts = [torch.empty_like(x) for _ in range(info.sp_size)]
  dist.all_gather(parts, x, group=info.group)
  return torch.cat(parts, dim=dim)
