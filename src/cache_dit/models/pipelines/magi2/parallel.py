"""Process-group surface and collectives used by the MAGI-2 Preview kernels.

Adapted from ``vllm-omni @ 7f217a1`` ``vllm_omni/diffusion/models/magi2/parallel.py``
(Apache-2.0, Copyright (c) 2025-2026 SandAI). Intentional differences:

* the groups are no longer discovered from vLLM's parallel state. cache-dit installs them with
  :func:`set_magi2_groups`, called by the CP/TP planners and by the pipeline;
* no group is created here (cache-dit owns the device meshes);
* ``Magi2ParallelGroup`` exposes ``group``/``world_size``/``rank`` exactly as upstream so the MoE
  and attention code keeps its contract.

MAGI-2 runs two parallel views over the packed token sequence: Ulysses context parallelism shards
tokens and gathers attention heads, while the multi-head MoE shards the MoE *head* axis (never the
expert axis). With ``--parallel tp`` the MoE head axis rides the TP group; otherwise it overlaps
with Ulysses, matching the released layout.
"""

import dataclasses

import torch
import torch.distributed as dist

__all__ = [
  "Magi2ParallelGroup",
  "Magi2SequenceDispatcher",
  "balanced_split_sizes",
  "ep_dispatch",
  "ep_undispatch",
  "gather_sequence",
  "get_magi2_ep_group",
  "get_magi2_tp_group",
  "get_magi2_ulysses_group",
  "reset_magi2_groups",
  "scatter_heads_gather_seqlen",
  "scatter_seqlen_gather_heads",
  "set_magi2_groups",
  "shard_sequence",
]


@torch.no_grad()
def _group_info(group):
  """Wrap a ``ProcessGroup`` in the small surface the kernels need.

  :param group: Process group, or ``None`` for a rank-local run.
  :returns: A :class:`Magi2ParallelGroup` describing the group.
  """

  if group is None or not dist.is_available() or not dist.is_initialized():
    return Magi2ParallelGroup(None, 1, 0)
  return Magi2ParallelGroup(
    group=group,
    world_size=dist.get_world_size(group),
    rank=dist.get_rank(group),
  )


@dataclasses.dataclass(frozen=True)
class Magi2ParallelGroup:
  """The process-group surface used by the MAGI-2 kernels.

  :param group: Underlying process group.
  :param world_size: Group size.
  :param rank: This process' rank inside the group.
  :param replicated_sequence: ``True`` when the group shards parameters and keeps the token sequence
    replicated (a parameter-parallel axis rather than a sequence-parallel one).
  """

  group: object = None
  world_size: int = 1
  rank: int = 0
  replicated_sequence: bool = False


_ULYSSES_GROUP = Magi2ParallelGroup(None, 1, 0)
_TP_GROUP = Magi2ParallelGroup(None, 1, 0, replicated_sequence=True)


def _as_group(value, *, replicated_sequence: bool = False):
  """Normalize a process group or an already-wrapped group.

  :param value: ``ProcessGroup``, :class:`Magi2ParallelGroup`, or ``None``.
  :param replicated_sequence: Value of the flag for a freshly wrapped group.
  :returns: A :class:`Magi2ParallelGroup`, or ``None``.
  """

  if value is None:
    return None
  if isinstance(value, Magi2ParallelGroup):
    return value
  info = _group_info(value)
  return Magi2ParallelGroup(
    group=info.group,
    world_size=info.world_size,
    rank=info.rank,
    replicated_sequence=replicated_sequence,
  )


def set_magi2_groups(ulysses_group=None, tp_group=None) -> None:
  """Install the process groups used by the MAGI-2 kernels.

  :param ulysses_group: Group for token sharding and the attention head/sequence exchange.
  :param tp_group: Group for parameter sharding (column/row grouped linear and MoE heads).
  """

  global _ULYSSES_GROUP, _TP_GROUP
  ulysses = _as_group(ulysses_group)
  if ulysses is not None:
    _ULYSSES_GROUP = ulysses
  tp = _as_group(tp_group, replicated_sequence=True)
  if tp is not None:
    _TP_GROUP = tp


def reset_magi2_groups() -> None:
  """Drop the installed groups (used by tests and by pipeline teardown)."""

  global _ULYSSES_GROUP, _TP_GROUP
  _ULYSSES_GROUP = Magi2ParallelGroup(None, 1, 0)
  _TP_GROUP = Magi2ParallelGroup(None, 1, 0, replicated_sequence=True)


def get_magi2_ulysses_group() -> Magi2ParallelGroup:
  """Return the Ulysses (token-sharding) group."""

  return _ULYSSES_GROUP


def get_magi2_tp_group() -> Magi2ParallelGroup:
  """Return the parameter-parallel group without a sequence-parallel fallback."""

  return _TP_GROUP


def get_magi2_ep_group() -> Magi2ParallelGroup:
  """Return the group sharding the MoE head axis.

  The TP group owns it when it has more than one rank; otherwise the head axis overlaps with
  Ulysses, exactly like the released SP-only layout.
  """

  if _TP_GROUP.world_size > 1:
    return _TP_GROUP
  return _ULYSSES_GROUP


def balanced_split_sizes(length: int, world_size: int) -> list:
  """Split ``length`` contiguously, putting one extra token on the early ranks.

  :param length: Sequence length to split.
  :param world_size: Number of shards.
  :returns: One shard length per rank.
  """

  if length < 0:
    raise ValueError(f"length must be non-negative, got {length}")
  if world_size < 1:
    raise ValueError(f"world_size must be positive, got {world_size}")
  base, remainder = divmod(length, world_size)
  return [base + int(rank < remainder) for rank in range(world_size)]


def shard_sequence(tensor: torch.Tensor, split_sizes=None, group=None):
  """Return this rank's contiguous sequence slice.

  Inputs are replicated at the pipeline boundary, so no collective is needed to dispatch.

  :param tensor: ``[tokens, ...]`` tensor to shard along dim 0.
  :param split_sizes: Pre-computed per-rank lengths; derived when omitted.
  :param group: Group to shard for; defaults to the Ulysses group.
  :returns: ``(local_tensor, split_sizes)``.
  """

  group = group or get_magi2_ulysses_group()
  split_sizes = split_sizes or balanced_split_sizes(tensor.shape[0], group.world_size)
  if len(split_sizes) != group.world_size or sum(split_sizes) != tensor.shape[0]:
    raise ValueError(f"split_sizes={split_sizes} do not partition sequence length "
                     f"{tensor.shape[0]} across {group.world_size} ranks")
  start = sum(split_sizes[:group.rank])
  return tensor.narrow(0, start, split_sizes[group.rank]).contiguous(), split_sizes


def gather_sequence(tensor: torch.Tensor, split_sizes, group=None) -> torch.Tensor:
  """Gather uneven contiguous sequence shards in rank order.

  :param tensor: This rank's ``[local_tokens, ...]`` shard.
  :param split_sizes: Per-rank lengths.
  :param group: Group to gather over; defaults to the Ulysses group.
  :returns: The full ``[sum(split_sizes), ...]`` tensor.
  """

  group = group or get_magi2_ulysses_group()
  if group.world_size == 1:
    return tensor
  if len(split_sizes) != group.world_size:
    raise ValueError("split_sizes length must equal the Ulysses world size")
  if tensor.shape[0] != split_sizes[group.rank]:
    raise ValueError(
      f"rank {group.rank} owns {tensor.shape[0]} tokens, expected {split_sizes[group.rank]}")
  output = torch.empty((sum(split_sizes), *tensor.shape[1:]),
                       dtype=tensor.dtype,
                       device=tensor.device)
  chunks = list(torch.split(output, list(split_sizes), dim=0))
  dist.all_gather(chunks, tensor.contiguous(), group=group.group)
  return output


def scatter_seqlen_gather_heads(tensor: torch.Tensor, split_sizes, group=None) -> torch.Tensor:
  """Ulysses ``[sum(S_r), H, D] -> [S_rank, world*H, D]`` exchange.

  :param tensor: Post-attention ``[sum(S_r), heads, head_dim]`` tensor.
  :param split_sizes: Per-rank lengths.
  :param group: Ulysses group.
  :returns: This rank's ``[S_rank, world*heads, head_dim]`` tensor.
  """

  group = group or get_magi2_ulysses_group()
  if group.world_size == 1:
    return tensor
  if tensor.ndim != 3 or not tensor.is_contiguous():
    raise ValueError("Ulysses attention input must be contiguous [S,H,D]")
  if len(split_sizes) != group.world_size or sum(split_sizes) != tensor.shape[0]:
    raise ValueError("split_sizes must partition the input sequence")

  local_tokens = split_sizes[group.rank]
  output = torch.empty(
    (group.world_size * local_tokens, tensor.shape[1], tensor.shape[2]),
    dtype=tensor.dtype,
    device=tensor.device,
  )
  dist.all_to_all_single(
    output,
    tensor,
    output_split_sizes=[local_tokens] * group.world_size,
    input_split_sizes=list(split_sizes),
    group=group.group,
  )
  return (output.view(group.world_size, local_tokens, tensor.shape[1],
                      tensor.shape[2]).permute(1, 0, 2,
                                               3).reshape(local_tokens,
                                                          group.world_size * tensor.shape[1],
                                                          tensor.shape[2]))


def scatter_heads_gather_seqlen(tensors, split_sizes, group=None) -> list:
  """Batched Ulysses exchange for Q/K/V.

  Each input is ``[S_rank, world*H_i, D]`` and each output is ``[sum(S_r), H_i, D]``. Fusing Q/K/V
  into one all-to-all keeps the released communication ordering and saves two collectives.

  :param tensors: Query, key and value tensors.
  :param split_sizes: Per-rank lengths.
  :param group: Ulysses group.
  :returns: The exchanged tensors, in input order.
  """

  group = group or get_magi2_ulysses_group()
  tensors = list(tensors)
  if group.world_size == 1 or not tensors:
    return tensors
  local_tokens = split_sizes[group.rank]
  if len(split_sizes) != group.world_size:
    raise ValueError("split_sizes length must equal the Ulysses world size")
  if any(t.ndim != 3 or t.shape[0] != local_tokens for t in tensors):
    raise ValueError("all Ulysses inputs must be [local_tokens, heads, dim]")

  reshaped = []
  local_head_counts = []
  head_dim = tensors[0].shape[-1]
  for tensor in tensors:
    if tensor.shape[-1] != head_dim or tensor.shape[1] % group.world_size:
      raise ValueError("attention heads must divide evenly across Ulysses ranks")
    local_heads = tensor.shape[1] // group.world_size
    local_head_counts.append(local_heads)
    reshaped.append(
      tensor.view(local_tokens, group.world_size, local_heads,
                  head_dim).permute(1, 0, 2, 3).reshape(group.world_size * local_tokens,
                                                        local_heads, head_dim))

  fused = torch.cat(reshaped, dim=1).contiguous()
  output = torch.empty((sum(split_sizes), fused.shape[1], head_dim),
                       dtype=fused.dtype,
                       device=fused.device)
  dist.all_to_all_single(
    output,
    fused,
    output_split_sizes=list(split_sizes),
    input_split_sizes=[local_tokens] * group.world_size,
    group=group.group,
  )
  return list(torch.split(output, local_head_counts, dim=1))


def ep_dispatch(tensor: torch.Tensor, group=None, sequence_split_sizes=None) -> torch.Tensor:
  """Dispatch ``[S, heads, D]`` so each rank evaluates a contiguous head shard.

  :param tensor: ``[tokens, heads, head_dim]`` MoE input.
  :param group: MoE-head group.
  :param sequence_split_sizes: Per-rank token counts when the sequence is already sharded.
  :returns: ``[sum(sequence_split_sizes), local_heads, head_dim]``.
  """

  group = group or get_magi2_ep_group()
  if group.world_size == 1:
    return tensor
  if tensor.ndim != 3 or tensor.shape[1] % group.world_size:
    raise ValueError(
      f"MoE head count {tensor.shape[1] if tensor.ndim >= 2 else '?'} must divide by "
      f"EP size {group.world_size}")
  sequence, heads, dim = tensor.shape
  local_heads = heads // group.world_size
  if sequence_split_sizes is None:
    local_size = torch.tensor([sequence], dtype=torch.int64, device=tensor.device)
    gathered_sizes = [torch.empty_like(local_size) for _ in range(group.world_size)]
    dist.all_gather(gathered_sizes, local_size, group=group.group)
    sequence_split_sizes = [int(size.item()) for size in gathered_sizes]
  if len(sequence_split_sizes) != group.world_size or sequence_split_sizes[group.rank] != sequence:
    raise ValueError("EP sequence split sizes do not describe the local tensor")
  send = tensor.contiguous().view(sequence, group.world_size, local_heads,
                                  dim).permute(1, 0, 2, 3).contiguous()
  output = torch.empty((sum(sequence_split_sizes), local_heads, dim),
                       dtype=tensor.dtype,
                       device=tensor.device)
  row_width = local_heads * dim
  dist.all_to_all_single(
    output.view(-1),
    send.view(-1),
    output_split_sizes=[size * row_width for size in sequence_split_sizes],
    input_split_sizes=[sequence * row_width] * group.world_size,
    group=group.group,
  )
  return output


def ep_undispatch(tensor: torch.Tensor, group=None, sequence_split_sizes=None) -> torch.Tensor:
  """Undo :func:`ep_dispatch` and restore the complete MoE head axis.

  :param tensor: ``[global_tokens, local_heads, head_dim]``.
  :param group: MoE-head group.
  :param sequence_split_sizes: Per-rank token counts used by the dispatch.
  :returns: ``[sum(sequence_split_sizes), heads, head_dim]``.
  """

  group = group or get_magi2_ep_group()
  if group.world_size == 1:
    return tensor
  if tensor.ndim != 3:
    raise ValueError("EP-dispatched tensor must be [global_tokens,local_heads,dim]")
  if sequence_split_sizes is None:
    if tensor.shape[0] % group.world_size:
      raise ValueError("uneven EP sequence requires explicit split sizes")
    sequence_split_sizes = [tensor.shape[0] // group.world_size] * group.world_size
  if len(sequence_split_sizes) != group.world_size or sum(sequence_split_sizes) != tensor.shape[0]:
    raise ValueError("EP sequence split sizes do not partition the global tensor")
  sequence = sequence_split_sizes[group.rank]
  local_heads, dim = tensor.shape[1:]
  send = tensor.contiguous()
  output = torch.empty((group.world_size, sequence, local_heads, dim),
                       dtype=tensor.dtype,
                       device=tensor.device)
  row_width = local_heads * dim
  dist.all_to_all_single(
    output.view(-1),
    send.view(-1),
    output_split_sizes=[sequence * row_width] * group.world_size,
    input_split_sizes=[size * row_width for size in sequence_split_sizes],
    group=group.group,
  )
  return output.permute(1, 0, 2, 3).contiguous().view(sequence, group.world_size * local_heads, dim)


class Magi2SequenceDispatcher:
  """Request-scoped dispatcher enforcing one consistent token split.

  :param group: Group to shard the packed sequence over; defaults to the Ulysses group.
  """

  def __init__(self, group=None) -> None:
    self.group = group or get_magi2_ulysses_group()
    self.split_sizes = None

  def dispatch(self, tensor: torch.Tensor) -> torch.Tensor:
    """Return this rank's contiguous slice and record the split.

    :param tensor: ``[tokens, ...]`` packed input.
    :returns: The local slice.
    """

    split_sizes = balanced_split_sizes(tensor.shape[0], self.group.world_size)
    if self.split_sizes is not None and split_sizes != self.split_sizes:
      raise ValueError(f"all packed inputs must share a sequence length; got {split_sizes}, "
                       f"expected {self.split_sizes}")
    self.split_sizes = split_sizes
    return shard_sequence(tensor, split_sizes, self.group)[0]

  def undispatch(self, tensor: torch.Tensor) -> torch.Tensor:
    """Gather the local slices back into the full packed sequence.

    :param tensor: ``[local_tokens, ...]`` output of the cached stack.
    :returns: The full-sequence tensor.
    """

    if self.split_sizes is None:
      raise RuntimeError("dispatch must be called before undispatch")
    output = gather_sequence(tensor, self.split_sizes, self.group)
    self.split_sizes = None
    return output
