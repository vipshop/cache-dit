"""DTensor-specific runtime and serialization helpers for SVDQ.

This module centralizes the tensor-parallel support that would otherwise be
scattered across the generic SVDQ runtime, the quantizer, and the PTQ save
path. The design keeps a single `SVDQW4A4ShardLinear` runtime module for both
colwise and rowwise TP layouts, and expresses their differences through a
small shard spec/helper layer instead of splitting the runtime into separate
module classes.

The structure is intentionally inspired by torchao's float8 TP integration in
`torchao.float8.float8_tensor_parallel`: that code keeps TP-specific behavior
behind a dedicated boundary layer instead of leaking it into the main module
implementation. The important difference is scope. TorchAO extends
`ColwiseParallel` and `RowwiseParallel` at the TP style layer because its
special handling happens while preparing DTensor inputs/outputs. SVDQ's TP
special handling is different: the hard problems are the quantized runtime
module itself and full-checkpoint serialization from local shards. Because of
that, SVDQ keeps the core TP styles unchanged and places the TP-specific logic
here in `SVDQW4A4ShardLinear` plus save helpers, rather than introducing
`SVDQColwiseParallel` or `SVDQRowwiseParallel` wrappers.

Serialization contract:
- TP save artifacts are always full quantized checkpoints.
- All ranks participate in gathering local shard tensors.
- Only TP rank 0 writes checkpoint files to disk.
- `load_svdq()` continues to load full quantized weights and does not need to
  understand sharded checkpoint formats.
"""

from __future__ import annotations

import dataclasses
import typing as tp
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from .linear import SVDQW4A4Linear

if TYPE_CHECKING:
  from torch.distributed import ProcessGroup
  from torch.distributed._tensor import DeviceMesh
  from torch.distributed._tensor import Placement


@dataclasses.dataclass
class SVDQShardSpec:
  """Runtime and save/load metadata for a TP-sharded SVDQ linear.

  `placement` tracks the sharding style inherited from the original TP float
  linear. The same spec is used both by the runtime forward path and by the
  full-checkpoint save helpers.
  """

  mesh: DeviceMesh
  placement: Placement
  tp_size: int
  tp_rank: int
  output_replicate: bool = False

  @property
  def group(self) -> ProcessGroup:
    return self.mesh.get_group()

  @property
  def is_colwise(self) -> bool:
    return bool(self.placement.is_shard(0))

  @property
  def is_rowwise(self) -> bool:
    return bool(self.placement.is_shard(1))

  def with_output_replicate(self, enabled: bool = True) -> "SVDQShardSpec":
    return dataclasses.replace(self, output_replicate=enabled)

  def state_dict_rule(self, param_name: str) -> tuple[str, int | None]:
    if self.is_colwise:
      shard_rules = {
        "qweight": ("cat", 0),
        "wscales": ("cat", 1),
        "proj_up": ("cat", 0),
        "bias": ("cat", 0),
        "wcscales": ("cat", 0),
        "proj_down": ("stack", 0),
      }
    elif self.is_rowwise:
      shard_rules = {
        "qweight": ("cat", 1),
        "wscales": ("cat", 0),
        "proj_down": ("cat", 0),
        "smooth_factor": ("cat", 0),
        "smooth_factor_orig": ("cat", 0),
        "proj_up": ("stack", 0),
      }
    else:
      raise NotImplementedError(
        "SVDQ DTensor support only handles linear TP placements Shard(0) and Shard(1), "
        f"got {self.placement}.")
    return shard_rules.get(param_name, ("replicate", None))


def _all_gather_tensor_objects(local_tensor: torch.Tensor,
                               shard_spec: SVDQShardSpec) -> list[torch.Tensor]:
  gathered: list[torch.Tensor | None] = [None] * shard_spec.tp_size
  dist.all_gather_object(gathered, local_tensor.detach().cpu().contiguous(), group=shard_spec.group)
  return [tp.cast(torch.Tensor, tensor) for tensor in gathered]


class SVDQW4A4ShardLinear(SVDQW4A4Linear):
  """SVDQ runtime module for TP local shards.

  The packed parameters already correspond to the local shard owned by the current TP rank. Forward
  therefore runs the normal local packed GEMM and only applies the TP collectives that reconstruct
  the expected output layout.
  """

  def __init__(self, *args, tp_spec: SVDQShardSpec, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self._tp_spec = tp_spec
    self._tp_output_replicate = tp_spec.output_replicate
    self._tp_mesh = tp_spec.mesh
    self._tp_placement = tp_spec.placement

  @staticmethod
  def resolve_local(
    tensor: torch.Tensor | None, ) -> tuple[torch.Tensor | None, SVDQShardSpec | None]:
    from torch.distributed._tensor import DTensor

    if tensor is None:
      return None, None
    if not isinstance(tensor, DTensor):
      return tensor, None

    placement = tensor.placements[0]
    local_tensor = tensor.to_local() if hasattr(tensor, "to_local") else tensor._local_tensor
    return local_tensor, SVDQShardSpec(
      mesh=tensor.device_mesh,
      placement=placement,
      tp_size=tensor.device_mesh.size(),
      tp_rank=tensor.device_mesh.get_local_rank(),
    )

  @staticmethod
  def slice_input(vector: torch.Tensor, shard_spec: SVDQShardSpec | None) -> torch.Tensor:
    if shard_spec is None or not shard_spec.is_rowwise:
      return vector
    shards = vector.chunk(shard_spec.tp_size, dim=0)
    return shards[shard_spec.tp_rank].contiguous()

  @staticmethod
  def shard_for_load(
    state_dict: dict[str, torch.Tensor],
    shard_spec: SVDQShardSpec | None,
  ) -> dict[str, torch.Tensor]:
    if shard_spec is None:
      return state_dict

    local_state_dict: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
      shard_kind, shard_dim = shard_spec.state_dict_rule(key)
      if shard_kind == "replicate":
        local_state_dict[key] = tensor
        continue
      if shard_kind == "stack":
        local_state_dict[key] = tensor[shard_spec.tp_rank].contiguous()
        continue
      shards = tensor.chunk(shard_spec.tp_size, dim=tp.cast(int, shard_dim))
      local_state_dict[key] = shards[shard_spec.tp_rank].contiguous()
    return local_state_dict

  def mark_replicate(self) -> None:
    self._tp_spec = self._tp_spec.with_output_replicate(True)
    self._tp_output_replicate = True

  def gather_for_save(self) -> tuple[dict[str, torch.Tensor], bool]:
    """Collect a full CPU state_dict for save and report whether this rank writes."""

    shard_spec = self._tp_spec
    state_dict: dict[str, torch.Tensor] = {}
    for key, tensor in self.state_dict().items():
      shard_kind, shard_dim = shard_spec.state_dict_rule(key)
      local_tensor = tensor.detach().cpu().contiguous()
      if shard_spec.tp_size == 1 or not dist.is_initialized():
        state_dict[key] = local_tensor
        continue

      gathered = _all_gather_tensor_objects(local_tensor, shard_spec)
      if shard_spec.tp_rank != 0:
        continue
      if shard_kind == "replicate":
        state_dict[key] = gathered[0]
      elif shard_kind == "stack":
        state_dict[key] = torch.stack(gathered, dim=0).contiguous()
      else:
        state_dict[key] = torch.cat(gathered, dim=tp.cast(int, shard_dim)).contiguous()

    return state_dict, shard_spec.tp_rank == 0

  def forward(self, x: torch.Tensor, output: torch.Tensor | None = None) -> torch.Tensor:
    from torch.distributed._tensor import DTensor

    local_x = x._local_tensor if isinstance(x, DTensor) else x
    local_output = output._local_tensor if isinstance(output, DTensor) else output
    result = self._forward_plain(local_x, local_output)

    if self._tp_spec.is_colwise:
      if self._tp_output_replicate:
        gather_list = [torch.empty_like(result) for _ in range(self._tp_spec.tp_size)]
        dist.all_gather(gather_list, result, group=self._tp_spec.group)
        result = torch.cat(gather_list, dim=-1)
      return result

    if self._tp_spec.is_rowwise:
      dist.all_reduce(result, op=dist.ReduceOp.SUM, group=self._tp_spec.group)
      return result

    raise NotImplementedError(
      "SVDQ DTensor runtime only supports linear TP placements Shard(0) and Shard(1), "
      f"got {self._tp_spec.placement}.")


__all__ = [
  "SVDQShardSpec",
  "SVDQW4A4ShardLinear",
]
