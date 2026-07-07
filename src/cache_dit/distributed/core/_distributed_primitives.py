from typing import TYPE_CHECKING, Tuple, List, Callable, Optional

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as fc
import torch.nn.functional as F

from ...platforms import current_platform
from ...logger import init_logger

logger = init_logger(__name__)

if TYPE_CHECKING:
  from ._modeling_parallel import _ContextParallelConfig

try:
  from ...kernels import (
    fp8_comm_per_token_quant,
    fp8_comm_per_token_dequant,
    fp8_comm_qkv_permute_quant,
    fp8_comm_qkv_permute_dequant,
  )
except ImportError:

  def _fp8_comm_kernel_unavailable(*args, **kwargs):
    raise RuntimeError(
      "FP8 comm kernels could not be imported (e.g., Triton may not be available on this "
      "platform). FP8 async operations are not supported. Please install the required "
      "dependencies or disable FP8 mode.")

  fp8_comm_per_token_quant = _fp8_comm_kernel_unavailable
  fp8_comm_per_token_dequant = _fp8_comm_kernel_unavailable
  fp8_comm_qkv_permute_quant = _fp8_comm_kernel_unavailable
  fp8_comm_qkv_permute_dequant = _fp8_comm_kernel_unavailable

# Some helper distributed primitive functions for context parallel attention.
__all__ = [
  "_gather_size",
  "_RingP2PComm",
  "_All2AllComm",
  "_all_to_all_single_qkv_custom_heads",
  "_all_to_all_single_o_custom_heads",
]


# NOTE: We should always use the asynchronous all to all variants to keep the unified input/output shape
# for any_qkvo and non-any_qkvo cases, otherwise, the input/output shape will be different, which makes
# the unified function implementation complex and ugly.
def _wait_tensor(tensor) -> torch.Tensor:
  if isinstance(tensor, fc.AsyncCollectiveTensor):
    tensor = tensor.wait()

  return tensor


def _get_rank_world_size(group: dist.ProcessGroup, ) -> Tuple[int, int]:
  world_size = dist.get_world_size(group=group)
  rank = dist.get_rank(group=group)
  return rank, world_size


def _gather_size(size: int, group: dist.ProcessGroup) -> List[int]:
  """Gather the local size from all ranks.

  :param size: Local integer extent contributed by the current rank.
  :param group: Process group participating in the collective.
  :returns: Per-rank local sizes gathered across the process group.
  """
  # NOTE(Serving/CP Safety):
  # Do NOT cache this collective result.
  #
  # In "Ulysses Anything" mode, `size` (e.g. per-rank local seq_len / S_LOCAL)
  # may legitimately differ across ranks. If we cache based on the *local* `size`,
  # different ranks can have different cache hit/miss patterns across time.
  #
  # That can lead to a catastrophic distributed hang:
  # - some ranks hit cache and *skip* dist.all_gather()
  # - other ranks miss cache and *enter* dist.all_gather()
  # This mismatched collective participation will stall the process group and
  # eventually trigger NCCL watchdog timeouts (often surfacing later as ALLTOALL
  # timeouts in Ulysses attention).
  world_size = dist.get_world_size(group=group)
  # HACK: Use Gloo backend for all_gather to avoid H2D and D2H overhead
  comm_backends = str(dist.get_backend(group=group))
  # NOTE: e.g., dist.init_process_group(backend="cpu:gloo,cuda:nccl")
  gather_device = "cpu" if "cpu" in comm_backends else current_platform.default_device()
  gathered_sizes = [
    torch.empty((1, ), device=gather_device, dtype=torch.int64) for _ in range(world_size)
  ]
  dist.all_gather(
    gathered_sizes,
    torch.tensor([size], device=gather_device, dtype=torch.int64),
    group=group,
  )

  gathered_sizes = [s[0].item() for s in gathered_sizes]
  # NOTE: DON'T use tolist here due to graph break - Explanation:
  # Backend compiler `inductor` failed with aten._local_scalar_dense.default
  return gathered_sizes


def _split_head_sizes(
  H: int,
  group: dist.ProcessGroup,
) -> List[int]:
  """Split the head dimension size by world_size.

  :param H: Total number of attention heads before sharding.
  :param group: Process group used for tensor partitioning.
  :returns: Per-rank head counts after splitting `H` across the group.
  """
  assert H is not None, "Global head num H must be provided."
  rank, world_size = _get_rank_world_size(group)
  # e.g, H = 30, world_size = 4, output_split_sizes = [8, 8, 8, 6]
  output_split_sizes = []
  base_head_num = H // world_size
  remainder = H % world_size
  for i in range(world_size):
    if i < remainder:
      output_split_sizes.append(base_head_num + 1)
    else:
      output_split_sizes.append(base_head_num)
  return output_split_sizes


# Helper functions to pad/unpad head dimension for QKV and O projections
def _maybe_pad_qkv_head(
  x: torch.Tensor,
  H: int,
  group: dist.ProcessGroup,
) -> Tuple[torch.Tensor, int]:
  """Maybe pad the head dimension to be divisible by world_size.

  :param x: Input tensor for the operation.
  :param H: Original global head count before padding.
  :param group: Process group used for sharding.
  :returns: A tuple `(x_padded, H_PAD)` where `H_PAD` is the number of added heads.
  """
  _, world_size = _get_rank_world_size(group)
  H_PAD = 0
  if H % world_size != 0:
    H_PAD = world_size - (H % world_size)
    NEW_H_LOCAL = (H + H_PAD) // world_size
    # e.g., Allow: H=30, world_size=8 -> NEW_H_LOCAL=4, H_PAD=2.
    # NOT ALLOW: H=30, world_size=16 -> NEW_H_LOCAL=2, H_PAD=14.
    assert (H_PAD < NEW_H_LOCAL
            ), f"Padding head num {H_PAD} should be less than new local head num {NEW_H_LOCAL}"
    x = F.pad(x, (0, 0, 0, H_PAD)).contiguous()
  return x, H_PAD


def _maybe_unpad_qkv_head(
  x: torch.Tensor,
  H_PAD: int,
  group: dist.ProcessGroup,
) -> torch.Tensor:
  """Maybe unpad the head dimension.

  :param x: Input tensor for the operation.
  :param H_PAD: Number of padded head slots that may need to be removed.
  :param group: Process group used for sharding.
  :returns: The tensor with any trailing padded heads removed on the last rank.
  """
  rank, world_size = _get_rank_world_size(group)
  # Only the last rank may have padding
  if H_PAD > 0 and rank == world_size - 1:
    x = x[:, :, :-H_PAD, :]
  return x.contiguous()


def _maybe_pad_o_head(
  x: torch.Tensor,
  H: int,
  group: dist.ProcessGroup,
) -> Tuple[torch.Tensor, int]:
  """Maybe pad the head dimension to be divisible by world_size.

  :param x: Input tensor for the operation.
  :param H: Original global head count before padding.
  :param group: Process group used for sharding.
  :returns: A tuple `(x_padded, H_PAD)` where `H_PAD` is the number of padded output heads.
  """
  if H is None:
    return x, 0

  rank, world_size = _get_rank_world_size(group)
  H_PAD = 0
  # Only the last rank may need padding
  if H % world_size != 0:
    # We need to broadcast H_PAD to all ranks to keep consistency
    # in unpadding step later for all ranks.
    H_PAD = world_size - (H % world_size)
    NEW_H_LOCAL = (H + H_PAD) // world_size
    assert (H_PAD < NEW_H_LOCAL
            ), f"Padding head num {H_PAD} should be less than new local head num {NEW_H_LOCAL}"
    if rank == world_size - 1:
      x = F.pad(x, (0, 0, 0, H_PAD)).contiguous()
  return x, H_PAD


def _maybe_unpad_o_head(
  x: torch.Tensor,
  H_PAD: int,
  group: dist.ProcessGroup,
) -> torch.Tensor:
  """Maybe unpad the head dimension.

  :param x: Input tensor for the operation.
  :param H_PAD: Number of padded output-head slots to remove.
  :param group: Process group used for sharding.
  :returns: The tensor with padded output-head slots removed.
  """
  if H_PAD > 0:
    x = x[:, :, :-H_PAD, :]
  return x.contiguous()


# Helper functions to for all-to-all communication with Ulysses Attention
def _init_comm_metadata(
  query: torch.Tensor,
  **kwargs,
) -> dict:
  # query: (B, S_LOCAL, H_GLOBAL, D)
  assert (len(
    query.shape) == 4), "Query tensor must be 4-dimensional of shape (B, S_LOCAL, H_GLOBAL, D)"
  extra_kwargs = {}
  extra_kwargs["NUM_QO_HEAD"] = query.shape[2]
  extra_kwargs["Q_S_LOCAL"] = query.shape[1]
  # Add other kwargs if needed in future
  return extra_kwargs


class _All2AllAsyncHandle:
  """Wrap one async all-to-all wait callable with an explicit `wait()` API."""

  __slots__ = ("_wait_fn", )

  def __init__(self, wait_fn: Callable[[], torch.Tensor]):
    self._wait_fn = wait_fn

  def wait(self) -> torch.Tensor:
    """Wait for the async collective and return the communicated tensor."""
    return self._wait_fn()

  def __call__(self) -> torch.Tensor:
    return self.wait()


class _All2AllComm:
  """Resolve and launch Ulysses all-to-all communication for q/k/v/o/lse.

  The communication variant is chosen once from `_ContextParallelConfig`, so
  call sites no longer need to re-resolve q/k/v/o/lse functions repeatedly.
  Communication metadata is initialized once via `init_meta()` or lazily on the
  first `send_q()/send_k()/send_v()` call and then reused by subsequent
  `send_*()` calls.
  """

  __slots__ = (
    "_group",
    "_metadata",
    "_q_impl",
    "_k_impl",
    "_v_impl",
    "_o_impl",
    "_lse_impl",
  )

  def __init__(self, _cp_config: "_ContextParallelConfig"):
    if _cp_config is None or _cp_config._ulysses_mesh is None:
      raise ValueError(
        "_All2AllComm requires a context parallel config with an initialized Ulysses mesh.")

    self._group = _cp_config._ulysses_mesh.get_group()
    self._metadata: Optional[dict] = None
    self._q_impl = _select_all_to_all_qkv_async_impl(_cp_config)
    self._k_impl = _select_all_to_all_qkv_async_impl(_cp_config, fp8=False)
    self._v_impl = self._q_impl
    self._o_impl = _select_all_to_all_o_async_impl(_cp_config)
    self._lse_impl = _select_all_to_all_o_async_impl(_cp_config, fp8=False)

  @property
  def group(self) -> dist.ProcessGroup:
    """Return the Ulysses process group bound to this communicator."""
    return self._group

  def init_meta(self, query: torch.Tensor, **kwargs) -> "_All2AllComm":
    """Initialize reusable communication metadata and return `self`.

    :param query: Representative tensor shaped like `(B, S_LOCAL, H_GLOBAL, D)`.
    :param kwargs: Optional metadata overrides forwarded to `_init_comm_metadata`.
    :returns: The communicator itself for chained setup.
    """
    self._metadata = _init_comm_metadata(query, **kwargs)
    return self

  def init_metadata(self, query: torch.Tensor, **kwargs) -> dict:
    """Initialize metadata and return a copy for compatibility.

    :param query: Representative tensor shaped like `(B, S_LOCAL, H_GLOBAL, D)`.
    :param kwargs: Optional metadata overrides forwarded to `_init_comm_metadata`.
    :returns: The initialized metadata dictionary.
    """
    self.init_meta(query, **kwargs)
    assert self._metadata is not None
    return dict(self._metadata)

  def _resolve_metadata(
    self,
    x: Optional[torch.Tensor] = None,
    allow_implicit_init: bool = False,
    **metadata,
  ) -> dict:
    if metadata:
      self._metadata = dict(metadata)
      return self._metadata

    if self._metadata is None:
      if allow_implicit_init:
        if x is None:
          raise ValueError("Implicit metadata initialization requires an input tensor.")
        self._metadata = _init_comm_metadata(x)
        return self._metadata
      raise ValueError(
        "_All2AllComm metadata is not initialized. Call send_q()/send_k()/send_v() first, "
        "pass metadata explicitly, or initialize it with init_meta(query).")

    return self._metadata

  def _launch(
    self,
    impl: Callable[..., Callable[[], torch.Tensor]],
    x: torch.Tensor,
    allow_implicit_init: bool = False,
    **metadata,
  ) -> _All2AllAsyncHandle:
    resolved_metadata = self._resolve_metadata(
      x=x,
      allow_implicit_init=allow_implicit_init,
      **metadata,
    )
    return _All2AllAsyncHandle(impl(x, self._group, **resolved_metadata))

  def send_q(self, query: torch.Tensor, **metadata) -> _All2AllAsyncHandle:
    """Launch async all-to-all for the query tensor."""
    return self._launch(self._q_impl, query, allow_implicit_init=True, **metadata)

  def send_k(self, key: torch.Tensor, **metadata) -> _All2AllAsyncHandle:
    """Launch async all-to-all for the key tensor on the non-fp8 path."""
    return self._launch(self._k_impl, key, allow_implicit_init=True, **metadata)

  def send_v(self, value: torch.Tensor, **metadata) -> _All2AllAsyncHandle:
    """Launch async all-to-all for the value tensor."""
    return self._launch(self._v_impl, value, allow_implicit_init=True, **metadata)

  def send_o(self, out: torch.Tensor, **metadata) -> _All2AllAsyncHandle:
    """Launch async all-to-all for the attention output tensor."""
    return self._launch(self._o_impl, out, **metadata)

  def send_lse(self, lse: torch.Tensor, **metadata) -> _All2AllAsyncHandle:
    """Launch async all-to-all for LSE on the non-fp8 path."""
    return self._launch(self._lse_impl, lse, **metadata)

  def all_gather_tensor_dim(self, x: torch.Tensor, dim: int) -> torch.Tensor:
    """Gather a tensor across the Ulysses group along one dimension.

    :param x: Local tensor shard.
    :param dim: Dimension to concatenate gathered shards on.
    :returns: Tensor gathered across the Ulysses process group.
    """
    x = x.contiguous()
    shape = list(x.shape)
    dim_sizes = _gather_size(shape[dim], self._group)
    gathered = []
    for dim_size in dim_sizes:
      current_shape = list(shape)
      current_shape[dim] = dim_size
      gathered.append(torch.empty(current_shape, device=x.device, dtype=x.dtype))
    dist.all_gather(gathered, x, group=self._group)
    return torch.cat(gathered, dim=dim).contiguous()

  def gather_replicated_kv_for_local_q(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    num_q_heads: int,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather K/V sequence shards and select K/V heads for the local Q shard.

    :param query: Query tensor after standard Ulysses Q all-to-all.
    :param key: Local key tensor before sequence gather.
    :param value: Local value tensor before sequence gather.
    :param num_q_heads: Global query head count before Ulysses.
    :returns: K/V tensors matching the local Q head count.
    """
    key = self.all_gather_tensor_dim(key, dim=1)
    value = self.all_gather_tensor_dim(value, dim=1)
    num_kv_heads = key.shape[2]
    if num_kv_heads == query.shape[2]:
      return key, value
    if num_q_heads % num_kv_heads != 0:
      raise ValueError(f"GQA requires num_q_heads to be divisible by num_kv_heads, got "
                       f"{num_q_heads} and {num_kv_heads}.")

    rank, world_size = _get_rank_world_size(self._group)
    local_heads = (num_q_heads + world_size - 1) // world_size
    start = rank * local_heads
    end = min(start + local_heads, num_q_heads)
    q_head_indices = torch.arange(start, end, device=key.device, dtype=torch.long)
    if q_head_indices.numel() != query.shape[2]:
      raise ValueError(f"Local Q head count mismatch after Ulysses all-to-all: expected "
                       f"{q_head_indices.numel()}, got {query.shape[2]}.")

    repeat_factor = num_q_heads // num_kv_heads
    kv_head_indices = torch.div(q_head_indices, repeat_factor, rounding_mode="floor")
    key = key.index_select(2, kv_head_indices).contiguous()
    value = value.index_select(2, kv_head_indices).contiguous()
    return key, value

  def group_aligned_gqa_head_splits(
    self,
    num_q_heads: int,
    num_kv_heads: int,
  ) -> tuple[list[int], list[int]]:
    """Compute KV-group-aligned Q/KV head splits for GQA Ulysses.

    :param num_q_heads: Global query head count.
    :param num_kv_heads: Global key/value head count.
    :returns: A tuple `(q_split_sizes, kv_split_sizes)`.
    """
    rank, world_size = _get_rank_world_size(self._group)
    if num_q_heads % num_kv_heads != 0:
      raise ValueError(f"GQA requires num_q_heads to be divisible by num_kv_heads, got "
                       f"{num_q_heads} and {num_kv_heads}.")
    if world_size > num_kv_heads:
      raise ValueError(f"Group-aligned GQA requires world_size <= num_kv_heads, got "
                       f"{world_size} and {num_kv_heads}.")

    ratio = num_q_heads // num_kv_heads
    kv_split_sizes = []
    base_kv_heads = num_kv_heads // world_size
    remainder = num_kv_heads % world_size
    for i in range(world_size):
      kv_split_sizes.append(base_kv_heads + (1 if i < remainder else 0))
    q_split_sizes = [heads * ratio for heads in kv_split_sizes]
    if q_split_sizes[rank] == 0 or kv_split_sizes[rank] == 0:
      raise ValueError("Group-aligned GQA produced an empty local head shard.")
    return q_split_sizes, kv_split_sizes

  def send_group_aligned_gqa_qkv(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], list[int], int]:
    """Communicate Q/K/V for group-aligned GQA Ulysses.

    :param query: Local query tensor before Ulysses communication.
    :param key: Local key tensor before Ulysses communication.
    :param value: Local value tensor before Ulysses communication.
    :param attn_mask: Optional local attention mask shard.
    :returns: Communicated Q/K/V, gathered mask, Q split sizes, and local sequence length.
    """
    local_sequence_length = query.shape[1]
    q_split_sizes, kv_split_sizes = self.group_aligned_gqa_head_splits(query.shape[2], key.shape[2])
    query = _all_to_all_single_qkv_custom_heads(query, self._group, q_split_sizes)
    key = _all_to_all_single_qkv_custom_heads(key, self._group, kv_split_sizes)
    value = _all_to_all_single_qkv_custom_heads(value, self._group, kv_split_sizes)
    if attn_mask is not None:
      attn_mask = self.all_gather_tensor_dim(attn_mask, dim=1)
    return query, key, value, attn_mask, q_split_sizes, local_sequence_length

  def send_group_aligned_gqa_o(
    self,
    out: torch.Tensor,
    q_split_sizes: List[int],
    local_sequence_length: int,
  ) -> torch.Tensor:
    """Communicate output back to sequence-local, global-head layout.

    :param out: Attention output shaped `(B, S_GLOBAL, H_Q_LOCAL, D)`.
    :param q_split_sizes: Per-rank Q head split sizes used for the forward Q communication.
    :param local_sequence_length: Sequence length owned by this rank before Ulysses.
    :returns: Output shaped `(B, S_LOCAL, H_Q_GLOBAL, D)`.
    """
    return _all_to_all_single_o_custom_heads(out, self._group, q_split_sizes, local_sequence_length)


def _all_to_all_single_qkv_async(
  x: torch.Tensor,
  group: dist.ProcessGroup,
  **kwargs,
) -> torch.Tensor:
  """Launch async all-to-all for QKV tensors with evenly split heads.

  :param x: Input tensor for the operation.
  :param group: Process group used for communication.
  :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
  :returns: A wait callable that yields a tensor shaped `(B, S_GLOBAL, H_LOCAL, D)`.
  """
  _, world_size = _get_rank_world_size(group)
  B, S_LOCAL, H, D = x.shape
  x, H_PAD = _maybe_pad_qkv_head(x, H, group)
  H_LOCAL = (H + H_PAD) // world_size
  x = x.reshape(B, S_LOCAL, world_size, H_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
  _shape = x.shape  # (world_size, S_LOCAL, B, H_LOCAL, D)

  x = x.flatten()
  x = fc.all_to_all_single(x, None, None, group)

  def wait() -> torch.Tensor:
    nonlocal x, H_PAD
    x = _wait_tensor(x)
    # (world_size, S_LOCAL, B, H_LOCAL, D)
    # -> (S_GLOBAL, B, H_LOCAL, D)
    # -> (B, S_GLOBAL, H_LOCAL, D)
    x = x.reshape(_shape).flatten(0, 1).permute(1, 0, 2, 3).contiguous()
    x = _maybe_unpad_qkv_head(x, H_PAD, group)
    return x

  return wait


def _all_to_all_single_o_async(
  x: torch.Tensor,
  group: dist.ProcessGroup,
  **kwargs,
) -> torch.Tensor:
  """Launch async all-to-all for output tensors with evenly split heads.

  :param x: Input tensor for the operation.
  :param group: Process group used for communication.
  :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
  :returns: A wait callable that yields a tensor shaped `(B, S_LOCAL, H_GLOBAL, D)`.
  """
  # Assume H is provided in kwargs, since we can't infer H from x's shape.
  # The padding logic needs H to determine if padding is necessary.
  H = kwargs.get("NUM_QO_HEAD", None)
  _, world_size = _get_rank_world_size(group)
  x, H_PAD = _maybe_pad_o_head(x, H, group)
  B, S_GLOBAL, H_LOCAL, D = x.shape
  S_LOCAL = S_GLOBAL // world_size
  # (B, S_GLOBAL, H_LOCAL, D) -> (world_size, H_LOCAL, B, S_LOCAL, D)
  x = x.reshape(B, world_size, S_LOCAL, H_LOCAL, D).permute(1, 3, 0, 2, 4).contiguous()
  _shape = x.shape  # (world_size, H_LOCAL, B, S_LOCAL, D)

  x = x.flatten()
  x = fc.all_to_all_single(x, None, None, group)

  def wait() -> torch.Tensor:
    nonlocal x, H_PAD
    x = _wait_tensor(x)
    # (world_size, H_LOCAL, B, S_LOCAL, D)
    # -> (H_GLOBAL, B, S_LOCAL, D)
    # -> (B, S_LOCAL, H_GLOBAL, D)
    x = x.reshape(_shape).flatten(0, 1).permute(1, 2, 0, 3).contiguous()
    x = _maybe_unpad_o_head(x, H_PAD, group)
    return x

  return wait


def _all_to_all_single_qkv_uneven_heads_async(
  x: torch.Tensor,
  group: dist.ProcessGroup,
  **kwargs,
) -> torch.Tensor:
  """Another variant for uneven head splits without padding.

  :param x: Input tensor for the operation.
  :param group: Process group used for communication.
  :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
  :returns: A wait callable that yields a tensor shaped `(B, S_GLOBAL, H_LOCAL, D)`.
  """
  rank, world_size = _get_rank_world_size(group)
  B, S_LOCAL, H_GLOBAL, D = x.shape
  # NOTE: May use tensor_split here to ensure the same split policy
  # that we have used in the EquipartitionSharder sharding strategy. Please
  # note that the 'tensor_split' Splits a tensor into multiple sub-tensors,
  # all of which are views of input, thus may not introduce extra IO access.
  input_split_sizes = [i.size(2) for i in torch.tensor_split(x, world_size, dim=2)]
  H_LOCAL = input_split_sizes[rank]
  # [H_GLOBAL, B, S_LOCAL, D]
  x = x.permute(2, 0, 1, 3).contiguous()
  output_split_sizes = [H_LOCAL] * world_size
  # [H_GLOBAL, B, S_LOCAL, D]
  x = fc.all_to_all_single(x, output_split_sizes, input_split_sizes, group)

  def wait() -> torch.Tensor:
    nonlocal x
    x = _wait_tensor(x)
    # [world_size, H_LOCAL, B, S_LOCAL, D]
    x = x.reshape(world_size, H_LOCAL, B, S_LOCAL, D)
    # [B, world_size, S_LOCAL, H_LOCAL, D]
    x = x.permute(2, 0, 3, 1, 4).contiguous()
    # [B, S_GLOBAL, H_LOCAL, D]
    x = x.reshape(B, world_size * S_LOCAL, H_LOCAL, D)
    return x

  return wait


def _all_to_all_single_o_uneven_heads_async(
  x: torch.Tensor,
  group: dist.ProcessGroup,
  **kwargs,
) -> torch.Tensor:
  """Another variant for uneven head splits without padding.

  :param x: Input tensor for the operation.
  :param group: Process group used for communication.
  :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
  :returns: A wait callable that yields a tensor shaped `(B, S_LOCAL, H_GLOBAL, D)`.
  """
  # Assume H is provided in kwargs, since we can't infer H from x's shape.
  # The padding logic needs H to determine if padding is necessary.
  H = kwargs.get("NUM_QO_HEAD", None)
  B, S_GLOBAL, H_LOCAL, D = x.shape
  rank, world_size = _get_rank_world_size(group)
  # e.g, H = 30, world_size = 4, output_split_sizes = [8, 8, 8, 6]
  output_split_sizes = _split_head_sizes(H, group)

  H_GLOBAL = sum(output_split_sizes)
  S_LOCAL = S_GLOBAL // world_size
  # [B, world_size, S_LOCAL, H_LOCAL, D]
  x = x.reshape(B, world_size, S_LOCAL, H_LOCAL, D)
  # [world_size, H_LOCAL, B, S_LOCAL, D]
  x = x.permute(1, 3, 0, 2, 4).contiguous()
  # [world_size * H_LOCAL, B, S_LOCAL, D]
  x = x.flatten(0, 1)
  input_split_sizes = [H_LOCAL] * world_size
  # [world_size * H_LOCAL, B, S_LOCAL, D]
  x = fc.all_to_all_single(x, output_split_sizes, input_split_sizes, group)

  def wait() -> torch.Tensor:
    nonlocal x
    x = _wait_tensor(x)
    # [H_GLOBAL, B, S_LOCAL, D]
    x = x.reshape(H_GLOBAL, B, S_LOCAL, D)
    # [B, S_LOCAL, H_GLOBAL, D]
    x = x.permute(1, 2, 0, 3).contiguous()
    return x

  return wait


def _all_to_all_single_qkv_custom_heads(
  x: torch.Tensor,
  group: dist.ProcessGroup,
  head_split_sizes: List[int],
) -> torch.Tensor:
  """All-to-all QKV tensors with caller-provided head splits.

  :param x: Input tensor shaped `(B, S_LOCAL, H_GLOBAL, D)`.
  :param group: Process group used for communication.
  :param head_split_sizes: Per-rank output head counts.
  :returns: Tensor shaped `(B, S_GLOBAL, H_LOCAL, D)` on the current rank.
  """
  rank, world_size = _get_rank_world_size(group)
  if len(head_split_sizes) != world_size:
    raise ValueError(f"Expected {world_size} head split sizes, got {len(head_split_sizes)}.")

  B, S_LOCAL, H_GLOBAL, D = x.shape
  if sum(head_split_sizes) != H_GLOBAL:
    raise ValueError(f"Head split sizes sum to {sum(head_split_sizes)}, expected {H_GLOBAL}.")

  seq_split_sizes = _gather_size(S_LOCAL, group)
  input_split_sizes = [B * S_LOCAL * head_count * D for head_count in head_split_sizes]
  output_split_sizes = [B * seq_len * head_split_sizes[rank] * D for seq_len in seq_split_sizes]
  input_tensor = torch.cat([chunk.flatten() for chunk in torch.split(x, head_split_sizes, dim=2)])
  output_tensor = torch.empty(sum(output_split_sizes), device=x.device, dtype=x.dtype)

  dist.all_to_all_single(
    output_tensor,
    input_tensor.contiguous(),
    output_split_sizes=output_split_sizes,
    input_split_sizes=input_split_sizes,
    group=group,
  )

  chunks = []
  offset = 0
  H_LOCAL = head_split_sizes[rank]
  for seq_len, numel in zip(seq_split_sizes, output_split_sizes):
    chunks.append(output_tensor[offset:offset + numel].view(B, seq_len, H_LOCAL, D))
    offset += numel
  return torch.cat(chunks, dim=1).contiguous()


def _all_to_all_single_o_custom_heads(
  x: torch.Tensor,
  group: dist.ProcessGroup,
  head_split_sizes: List[int],
  local_sequence_length: int,
) -> torch.Tensor:
  """Inverse all-to-all for attention output with caller-provided head splits.

  :param x: Input tensor shaped `(B, S_GLOBAL, H_LOCAL, D)`.
  :param group: Process group used for communication.
  :param head_split_sizes: Per-rank input head counts from Q communication.
  :param local_sequence_length: Sequence length owned by the current rank before Ulysses.
  :returns: Tensor shaped `(B, S_LOCAL, H_GLOBAL, D)` on the current rank.
  """
  rank, world_size = _get_rank_world_size(group)
  if len(head_split_sizes) != world_size:
    raise ValueError(f"Expected {world_size} head split sizes, got {len(head_split_sizes)}.")

  B, S_GLOBAL, H_LOCAL, D = x.shape
  if H_LOCAL != head_split_sizes[rank]:
    raise ValueError(
      f"Local output heads mismatch: expected {head_split_sizes[rank]}, got {H_LOCAL}.")

  seq_split_sizes = _gather_size(local_sequence_length, group)
  if sum(seq_split_sizes) != S_GLOBAL:
    raise ValueError(f"Sequence split sizes sum to {sum(seq_split_sizes)}, expected {S_GLOBAL}.")

  sequence_chunks = torch.split(x, seq_split_sizes, dim=1)
  input_split_sizes = [B * seq_len * H_LOCAL * D for seq_len in seq_split_sizes]
  output_split_sizes = [
    B * local_sequence_length * head_count * D for head_count in head_split_sizes
  ]
  input_tensor = torch.cat([chunk.flatten() for chunk in sequence_chunks])
  output_tensor = torch.empty(sum(output_split_sizes), device=x.device, dtype=x.dtype)

  dist.all_to_all_single(
    output_tensor,
    input_tensor.contiguous(),
    output_split_sizes=output_split_sizes,
    input_split_sizes=input_split_sizes,
    group=group,
  )

  chunks = []
  offset = 0
  for head_count, numel in zip(head_split_sizes, output_split_sizes):
    chunks.append(output_tensor[offset:offset + numel].view(B, local_sequence_length, head_count,
                                                            D))
    offset += numel
  return torch.cat(chunks, dim=2).contiguous()


def _all_to_all_single_qkv_fp8_async(
  x: torch.Tensor,
  group: dist.ProcessGroup,
  **kwargs,
) -> Callable[..., torch.Tensor]:
  """Launch async FP8 all-to-all for QKV tensors with evenly split heads.

  :param x: Input tensor for the operation.
  :param group: Process group used for communication.
  :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
  :returns: A wait callable that yields a dequantized tensor shaped `(B, S_GLOBAL, H_LOCAL, D)`.
  """
  _, world_size = _get_rank_world_size(group)
  B, S_LOCAL, H, D = x.shape
  x, H_PAD = _maybe_pad_qkv_head(x, H, group)
  H_LOCAL = (H + H_PAD) // world_size
  x = x.reshape(B, S_LOCAL, world_size, H_LOCAL, D)
  x = fp8_comm_qkv_permute_quant(x)
  shape_with_scale = x.shape  # (world_size, S_LOCAL, B, H_LOCAL, D + itemsize)
  x = x.flatten()
  x = fc.all_to_all_single(x, None, None, group)

  def wait() -> torch.Tensor:
    nonlocal x, H_PAD
    x = _wait_tensor(x)
    x = x.reshape(shape_with_scale).flatten(0, 1)
    x = fp8_comm_qkv_permute_dequant(x)
    x = _maybe_unpad_qkv_head(x, H_PAD, group)
    return x

  return wait


def _all_to_all_single_o_fp8_async(
  x: torch.Tensor,
  group: dist.ProcessGroup,
  **kwargs,
) -> Callable[..., torch.Tensor]:
  """Launch async FP8 all-to-all for output tensors with evenly split heads.

  :param x: Input tensor for the operation.
  :param group: Process group used for communication.
  :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
  :returns: A wait callable that yields a dequantized tensor shaped `(B, S_LOCAL, H_GLOBAL, D)`.
  """
  # Assume H is provided in kwargs, since we can't infer H from x's shape.
  # The padding logic needs H to determine if padding is necessary.
  H = kwargs.get("NUM_QO_HEAD", None)
  _, world_size = _get_rank_world_size(group)
  x, H_PAD = _maybe_pad_o_head(x, H, group)
  B, S_GLOBAL, H_LOCAL, D = x.shape
  S_LOCAL = S_GLOBAL // world_size
  # (B, S_GLOBAL, H_LOCAL, D) -> (world_size, H_LOCAL, B, S_LOCAL, D)
  x = x.reshape(B, world_size, S_LOCAL, H_LOCAL, D).permute(1, 3, 0, 2, 4).contiguous()
  _shape = x.shape  # (world_size, H_LOCAL, B, S_LOCAL, D)

  x = fp8_comm_per_token_quant(x)
  shape_with_scale = x.shape  # (world_size, H_LOCAL, B, S_LOCAL, D + itemsize)
  x = x.flatten()
  x = fc.all_to_all_single(x, None, None, group)

  def wait() -> torch.Tensor:
    nonlocal x, H_PAD
    x = _wait_tensor(x)
    x = x.reshape(shape_with_scale)
    x = fp8_comm_per_token_dequant(x)
    # (world_size, H_LOCAL, B, S_LOCAL, D)
    # -> (H_GLOBAL, B, S_LOCAL, D)
    # -> (B, H_GLOBAL, S_LOCAL, D)
    x = x.reshape(_shape).flatten(0, 1).permute(1, 2, 0, 3).contiguous()
    x = _maybe_unpad_o_head(x, H_PAD, group)
    return x

  return wait


@torch.compiler.allow_in_graph
def _all_to_all_single_any_qkv_async(
  x: torch.Tensor,
  group: dist.ProcessGroup,
  **kwargs,
) -> Callable[..., torch.Tensor]:
  """Launch async all-to-all for QKV tensors with potentially uneven local sequence lengths.

  :param x: Input tensor for the operation.
  :param group: Process group used for communication.
  :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
  :returns: A wait callable that yields a tensor shaped `(B, S_GLOBAL, H_LOCAL, D)`.
  """
  _, world_size = _get_rank_world_size(group)
  B, S_LOCAL, H, D = x.shape
  x, H_PAD = _maybe_pad_qkv_head(x, H, group)
  H_LOCAL = (H + H_PAD) // world_size
  # (world_size, S_LOCAL, B, H_LOCAL, D)
  x = x.reshape(B, S_LOCAL, world_size, H_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()

  input_split_sizes = [S_LOCAL] * world_size
  # S_LOCAL maybe not equal for all ranks in dynamic shape case,
  # since we don't know the actual shape before this timing, thus,
  # we have to use all gather to collect the S_LOCAL first.
  output_split_sizes = _gather_size(S_LOCAL, group)
  # NOTE: The `if` branch will introduce graph break for torch.compile,
  # so, we choose to disable the even split optimization implementation
  # _all_to_all_single for now.
  x = x.flatten(0, 1)  # (world_size * S_LOCAL, B, H_LOCAL, D)
  x = fc.all_to_all_single(x, output_split_sizes, input_split_sizes, group)

  def wait() -> torch.Tensor:
    nonlocal x, H_PAD
    x = _wait_tensor(x)  # (S_GLOBAL, B, H_LOCAL, D)
    # (S_GLOBAL, B, H_LOCAL, D)
    # -> (B, S_GLOBAL, H_LOCAL, D)
    x = x.permute(1, 0, 2, 3).contiguous()
    x = _maybe_unpad_qkv_head(x, H_PAD, group)
    return x

  return wait


@torch.compiler.allow_in_graph
def _all_to_all_single_any_o_async(
  x: torch.Tensor,
  group: dist.ProcessGroup,
  **kwargs,
) -> Callable[..., torch.Tensor]:
  """Launch async all-to-all for output tensors with potentially uneven local sequence lengths.

  :param x: Input tensor for the operation.
  :param group: Process group used for communication.
  :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
  :returns: A wait callable that yields a tensor shaped `(B, S_LOCAL, H_GLOBAL, D)`.
  """
  # Assume H is provided in kwargs, since we can't infer H from x's shape.
  # The padding logic needs H to determine if padding is necessary.
  H = kwargs.get("NUM_QO_HEAD", None)
  rank, world_size = _get_rank_world_size(group)
  x, H_PAD = _maybe_pad_o_head(x, H, group)
  shape = x.shape  # (B, S_GLOBAL, H_LOCAL, D)
  (B, S_GLOBAL, H_LOCAL, D) = shape
  # input_split: e.g, S_GLOBAL=9 input splits across ranks [[5,4], [5,4],..]
  # output_split: e.g, S_GLOBAL=9 output splits across ranks [[5,5], [4,4],..]

  # WARN: In some cases, e.g, joint attn in Qwen-Image, the S_LOCAL can not infer
  # from tensor split due to: if c = torch.cat((a, b)), world_size=4, then,
  # c.tensor_split(4)[0].shape[1] may != to (a.tensor_split(4)[0].shape[1] +
  # b.tensor_split(4)[0].shape[1])

  # input_split_sizes = [o.size(1) for o in torch.tensor_split(x, world_size, dim=1)]
  # S_LOCAL = input_split_sizes[rank]

  S_LOCAL = kwargs.get("Q_S_LOCAL")
  input_split_sizes = _gather_size(S_LOCAL, group)

  x = x.permute(1, 0, 2, 3).contiguous()  # (S_GLOBAL, B, H_LOCAL, D)
  output_split_sizes = [S_LOCAL] * world_size
  x = fc.all_to_all_single(x, output_split_sizes, input_split_sizes, group)

  def wait() -> torch.Tensor:
    nonlocal x, H_PAD
    x = _wait_tensor(x)  # (S_GLOBAL, B, H_LOCAL, D)
    x = x.reshape(world_size, S_LOCAL, B, H_LOCAL, D)
    x = x.permute(2, 1, 0, 3, 4).contiguous()
    x = x.reshape(B, S_LOCAL, world_size * H_LOCAL, D)
    x = _maybe_unpad_o_head(x, H_PAD, group)
    return x

  return wait


@torch.compiler.allow_in_graph
def _all_to_all_single_any_qkv_fp8_async(
  x: torch.Tensor,
  group: dist.ProcessGroup,
  **kwargs,
) -> Callable[..., torch.Tensor]:
  """Launch async FP8 all-to-all for QKV tensors with uneven local sequence lengths.

  :param x: Input tensor for the operation.
  :param group: Process group used for communication.
  :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
  :returns: A wait callable that yields a dequantized tensor shaped `(B, S_GLOBAL, H_LOCAL, D)`.
  """
  _, world_size = _get_rank_world_size(group)
  B, S_LOCAL, H, D = x.shape
  x, H_PAD = _maybe_pad_qkv_head(x, H, group)
  H_LOCAL = (H + H_PAD) // world_size
  # (world_size, S_LOCAL, B, H_LOCAL, D)
  x = x.reshape(B, S_LOCAL, world_size, H_LOCAL, D)

  input_split_sizes = [S_LOCAL] * world_size
  # S_LOCAL maybe not equal for all ranks in dynamic shape case,
  # since we don't know the actual shape before this timing, thus,
  # we have to use all gather to collect the S_LOCAL first.
  output_split_sizes = _gather_size(S_LOCAL, group)
  # NOTE: The `if` branch will introduce graph break for torch.compile,
  # so, we choose to disable the even split optimization implementation
  # _all_to_all_single for now.
  x = fp8_comm_qkv_permute_quant(x)
  x = x.flatten(0, 1)
  x = fc.all_to_all_single(x, output_split_sizes, input_split_sizes, group)

  def wait() -> torch.Tensor:
    nonlocal x, H_PAD
    x = _wait_tensor(x)
    x = fp8_comm_qkv_permute_dequant(x)
    x = _maybe_unpad_qkv_head(x, H_PAD, group)
    return x

  return wait


@torch.compiler.allow_in_graph
def _all_to_all_single_any_o_fp8_async(
  x: torch.Tensor,
  group: dist.ProcessGroup,
  **kwargs,
) -> Callable[..., torch.Tensor]:
  """Launch async FP8 all-to-all for output tensors with uneven local sequence lengths.

  :param x: Input tensor for the operation.
  :param group: Process group used for communication.
  :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
  :returns: A wait callable that yields a dequantized tensor shaped `(B, S_LOCAL, H_GLOBAL, D)`.
  """
  # Assume H is provided in kwargs, since we can't infer H from x's shape.
  # The padding logic needs H to determine if padding is necessary.
  H = kwargs.get("NUM_QO_HEAD", None)
  rank, world_size = _get_rank_world_size(group)
  x, H_PAD = _maybe_pad_o_head(x, H, group)
  shape = x.shape  # (B, S_GLOBAL, H_LOCAL, D)
  x = fp8_comm_per_token_quant(x)
  (B, S_GLOBAL, H_LOCAL, D) = shape
  # input_split: e.g, S_GLOBAL=9 input splits across ranks [[5,4], [5,4],..]
  # output_split: e.g, S_GLOBAL=9 output splits across ranks [[5,5], [4,4],..]

  # WARN: In some cases, e.g, joint attn in Qwen-Image, the S_LOCAL can not infer
  # from tensor split due to: if c = torch.cat((a, b)), world_size=4, then,
  # c.tensor_split(4)[0].shape[1] may != to (a.tensor_split(4)[0].shape[1] +
  # b.tensor_split(4)[0].shape[1])

  # input_split_sizes = [o.size(1) for o in torch.tensor_split(x, world_size, dim=1)]
  # S_LOCAL = input_split_sizes[rank]

  S_LOCAL = kwargs.get("Q_S_LOCAL")
  input_split_sizes = _gather_size(S_LOCAL, group)

  x = x.permute(1, 0, 2, 3).contiguous()  # (S_GLOBAL, B, H_LOCAL, D)
  output_split_sizes = [S_LOCAL] * world_size
  x = fc.all_to_all_single(x, output_split_sizes, input_split_sizes, group)

  def wait() -> torch.Tensor:
    nonlocal x, H_PAD
    x = _wait_tensor(x)  # (S_GLOBAL, B, H_LOCAL, D)
    x = fp8_comm_per_token_dequant(x)
    x = x.reshape(world_size, S_LOCAL, B, H_LOCAL, D)
    x = x.permute(2, 1, 0, 3, 4).contiguous()
    x = x.reshape(B, S_LOCAL, world_size * H_LOCAL, D)
    x = _maybe_unpad_o_head(x, H_PAD, group)
    return x

  return wait


# Unified functions to select proper all to all implementations according to
# Ulysses Float8 or other settings. Mainly used in Async Ulysses Attention.


def _select_all_to_all_qkv_async_impl(
  _cp_config: Optional["_ContextParallelConfig"] = None,
  fp8: Optional[bool] = None,
) -> Callable[..., torch.Tensor]:
  _force_disable_float8 = (fp8 is not None) and (not fp8)
  ulysses_anything = bool(_cp_config is not None and _cp_config.ulysses_anything)
  ulysses_float8 = bool(_cp_config is not None and _cp_config.ulysses_float8)

  if ulysses_anything:
    if ulysses_float8 and not _force_disable_float8:
      return _all_to_all_single_any_qkv_fp8_async
    return _all_to_all_single_any_qkv_async

  if ulysses_float8 and not _force_disable_float8:
    return _all_to_all_single_qkv_fp8_async
  return _all_to_all_single_qkv_async


def _select_all_to_all_o_async_impl(
  _cp_config: Optional["_ContextParallelConfig"] = None,
  fp8: Optional[bool] = None,
) -> Callable[..., torch.Tensor]:
  _force_disable_float8 = (fp8 is not None) and (not fp8)
  ulysses_anything = bool(_cp_config is not None and _cp_config.ulysses_anything)
  ulysses_float8 = bool(_cp_config is not None and _cp_config.ulysses_float8)

  if ulysses_anything:
    if ulysses_float8 and not _force_disable_float8:
      return _all_to_all_single_any_o_fp8_async
    return _all_to_all_single_any_o_async

  if ulysses_float8 and not _force_disable_float8:
    return _all_to_all_single_o_fp8_async
  return _all_to_all_single_o_async


def _all_to_all_qkv_async_fn(
  _cp_config: Optional["_ContextParallelConfig"] = None,
  fp8: Optional[bool] = None,
) -> Callable[..., torch.Tensor]:
  return _select_all_to_all_qkv_async_impl(_cp_config, fp8)


def _all_to_all_o_async_fn(
  _cp_config: Optional["_ContextParallelConfig"] = None,
  fp8: Optional[bool] = None,
) -> Callable[..., torch.Tensor]:
  return _select_all_to_all_o_async_impl(_cp_config, fp8)


# Ring-based point-to-point communication primitives for key/value rotation in ring attention.
# Adapted from: https://github.com/zhuzilin/ring-flash-attention/blob/main/ring_flash_attn/utils.py#L98
class _RingP2PComm:
  """Pairwise ring communicator used by ring attention key/value rotation.

  :param _cp_config: Context parallel config carrying an initialized ring mesh.
  """

  def __init__(self, _cp_config: "_ContextParallelConfig"):
    if _cp_config is None or _cp_config._ring_mesh is None:
      raise ValueError(
        "_RingP2PComm requires a context parallel config with an initialized ring mesh.")

    self._process_group = _cp_config._ring_mesh.get_group()
    self._ops = []
    self.rank = dist.get_rank(self._process_group)
    self.world_size = dist.get_world_size(self._process_group)
    self._reqs = None

    self.send_rank = (self.rank + 1) % self.world_size
    self.recv_rank = (self.rank - 1) % self.world_size

    if self._process_group is not None:
      self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
      self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

  def send_recv(
    self,
    to_send: torch.Tensor,
    recv_tensor: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Launch one async point-to-point send/recv pair.

    :param to_send: Tensor to send to the next rank in the ring.
    :param recv_tensor: Optional receive buffer to reuse.
    :returns: Receive buffer for the incoming tensor.
    """

    to_send = to_send.contiguous()
    if recv_tensor is None:
      res = torch.empty_like(to_send).contiguous()
    else:
      res = recv_tensor

    send_op = dist.P2POp(dist.isend, to_send, self.send_rank, group=self._process_group)
    recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
    self._ops.append(send_op)
    self._ops.append(recv_op)
    return res

  def commit(self):
    """Submit the queued point-to-point operations."""

    if self._reqs is not None:
      raise RuntimeError("commit called twice")
    self._reqs = dist.batch_isend_irecv(self._ops)

  def wait(self):
    """Wait for the queued point-to-point operations to complete."""

    if self._reqs is None:
      raise RuntimeError("wait called before commit")
    for req in self._reqs:
      req.wait()
    self._reqs = None
    self._ops = []

  def send_recv_kv(
    self,
    k: torch.Tensor,
    v: torch.Tensor,
    k_buffer: Optional[torch.Tensor] = None,
    v_buffer: Optional[torch.Tensor] = None,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exchange key/value tensors independently around the ring."""

    next_k, next_v = self.send_recv(k, k_buffer), self.send_recv(v, v_buffer)
    self.commit()
    return next_k, next_v

  def batch_send_recv_kv(
    self,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_buffer: Optional[torch.Tensor] = None,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Exchange key/value tensors via one batched send/recv buffer."""

    kv_concat = torch.cat([k, v], dim=0)
    kv_recv = self.send_recv(kv_concat, kv_buffer)
    self.commit()

    seq_local = k.size(0)
    # Just views of kv_recv, no copy. DON'T use contiguous here.
    # contiguous() will create a copy of empty tensor, which
    # causes wrong results.
    next_k, next_v = torch.split(kv_recv, [seq_local, seq_local], dim=0)
    return next_k, next_v
