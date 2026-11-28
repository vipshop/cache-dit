"""Initialize TP linear weights from identical CPU parameters on every rank."""

from collections.abc import Iterable
from hashlib import sha256

import torch.distributed as dist
from torch import Tensor, nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel


def validate_local_weight_shards(
  component: nn.Module,
  projections: Iterable[tuple[str, bool]],
  replica_mesh: DeviceMesh,
) -> None:
  """Check CPU eligibility for local projections and metadata for every planned projection.

  Yield (path, initialize_locally) pairs from projections. Run planner-specific checks during
  iteration so any validation error is reported across the replica.
  """
  try:
    metadata = list(_projection_metadata(component, projections))
  except (AttributeError, RuntimeError, ValueError) as exc:
    report = (str(exc), None)
  else:
    report = (None, sha256(repr(metadata).encode()).digest())

  # Every rank must reach the collective even when its local weights fail validation.
  reports = [None] * replica_mesh.size()
  dist.all_gather_object(reports, report, group=replica_mesh.get_group())
  component_name = type(component).__name__
  for rank, (error, _) in enumerate(reports):
    if error is not None:
      raise ValueError(f"{component_name} local weight sharding: replica rank {rank}: {error}")
  if len({signature for _, signature in reports}) != 1:
    raise ValueError(f"{component_name} local weight sharding requires matching weight metadata "
                     "across the replica.")


def _projection_metadata(
  component: nn.Module,
  projections: Iterable[tuple[str, bool]],
) -> Iterable[tuple]:
  for path, initialize_locally in projections:
    linear = component.get_submodule(path)
    if initialize_locally and (type(linear) is not nn.Linear or linear.weight is None):
      raise ValueError(f"{path} must be a Linear with a weight parameter.")
    for name, param in linear.named_parameters(recurse=False):
      metadata = (path, name, param.shape, param.dtype)
      if initialize_locally:
        if type(param) is not nn.Parameter or not param.is_cpu or not param.is_contiguous():
          raise ValueError(f"{path}.{name} must be a contiguous CPU parameter.")
        metadata += (param.stride(), param.requires_grad)
      yield metadata


def local_shard(tensor: Tensor, mesh: DeviceMesh, dim: int) -> Tensor:
  """Select one chunk using DTensor's chunk sizes, including trailing empty shards."""
  size = tensor.size(dim)
  chunk = (size + mesh.size() - 1) // mesh.size()
  start = min(chunk * mesh.get_local_rank(), size)
  return tensor.narrow(dim, start, min(chunk, size - start))


def local_parameter(
  param: nn.Parameter,
  local: Tensor,
  mesh: DeviceMesh,
  placement: Shard | Replicate,
) -> nn.Parameter:
  """Create a DTensor parameter on the mesh with the original global shape and stride."""
  dtensor = DTensor.from_local(
    local,
    mesh,
    (placement, ),
    run_check=False,
    shape=param.shape,
    stride=param.stride(),
  )
  return nn.Parameter(dtensor, requires_grad=param.requires_grad)


def shard_linear_weights(
  linear: nn.Linear,
  mesh: DeviceMesh,
  style: ColwiseParallel | RowwiseParallel,
) -> None:
  """Materialize the placements expected by the native linear TP style."""
  colwise = isinstance(style, ColwiseParallel)
  for name, param in linear.named_parameters(recurse=False):
    local = param.detach()
    if not colwise and name == "bias":
      placement = Replicate()
    else:
      placement = Shard(0 if colwise else 1)
      local = local_shard(local, mesh, placement.dim)
    local = local.contiguous()
    linear.register_parameter(name, local_parameter(param, local, mesh, placement))
