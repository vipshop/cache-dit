"""Adapted from diffusers' modeling_parallel config primitives for cache-dit CP."""

import dataclasses
from typing import Any, Literal

import torch


@dataclasses.dataclass
class _ContextParallelConfig:
  """Describe one internal context-parallel region.

  :param ring_degree: Number of ranks used by Ring attention.
  :param ulysses_degree: Number of ranks used by Ulysses attention.
  :param convert_to_fp32: Whether Ring attention outputs should be accumulated in fp32.
  :param rotate_method: KV rotation strategy for Ring attention. Only "p2p" is supported now.
  :param mesh: Optional user-provided device mesh.
  :param ulysses_anything: Whether uneven sequence/head Ulysses mode is enabled.
  :param ulysses_float8: Whether Ulysses communication should use fp8 kernels when available.
  :param ulysses_async: Whether model-specific async Ulysses attention patches are enabled.
  :param extra_kwargs: Reserved extension storage for cache-dit specific runtime hints.
  """

  ring_degree: int | None = None
  ulysses_degree: int | None = None
  convert_to_fp32: bool = True
  rotate_method: Literal["allgather", "alltoall", "p2p"] = "p2p"
  mesh: torch.distributed.device_mesh.DeviceMesh | None = None
  ulysses_anything: bool = False
  ulysses_float8: bool = False
  ulysses_async: bool = False
  extra_kwargs: dict[str, Any] | None = None

  _rank: int | None = None
  _world_size: int | None = None
  _device: torch.device | None = None
  _mesh: torch.distributed.device_mesh.DeviceMesh | None = None
  _flattened_mesh: torch.distributed.device_mesh.DeviceMesh | None = None
  _ring_mesh: torch.distributed.device_mesh.DeviceMesh | None = None
  _ulysses_mesh: torch.distributed.device_mesh.DeviceMesh | None = None
  _ring_local_rank: int | None = None
  _ulysses_local_rank: int | None = None

  def __post_init__(self) -> None:
    if self.ring_degree is None:
      self.ring_degree = 1
    if self.ulysses_degree is None:
      self.ulysses_degree = 1

    if self.ring_degree == 1 and self.ulysses_degree == 1:
      raise ValueError(
        "Either ring_degree or ulysses_degree must be greater than 1 in order to use context parallel inference"
      )
    if self.ring_degree < 1 or self.ulysses_degree < 1:
      raise ValueError("ring_degree and ulysses_degree must be greater than or equal to 1.")
    if self.rotate_method not in {"p2p"}:
      raise NotImplementedError(
        f"Only rotate_method='p2p' is supported now, but got {self.rotate_method}.")
    if (self.ulysses_anything or self.ulysses_float8
        or self.ulysses_async) and self.ulysses_degree == 1:
      raise ValueError(
        "ulysses_degree must be greater than 1 when ulysses_anything, ulysses_float8, or ulysses_async is enabled."
      )
    if self.ulysses_anything:
      if self.ring_degree > 1:
        raise ValueError("ulysses_anything cannot be enabled when ring_degree > 1.")

  @property
  def mesh_shape(self) -> tuple[int, int]:
    """Return the logical CP mesh shape.

    :returns: Tuple of `(ring_degree, ulysses_degree)`.
    """

    return (self.ring_degree, self.ulysses_degree)

  @property
  def mesh_dim_names(self) -> tuple[str, str]:
    """Return the CP mesh dimension names.

    :returns: Mesh dimension names consumed by `init_device_mesh`.
    """

    return ("ring", "ulysses")

  @property
  def context_parallel_config(self) -> "_ContextParallelConfig":
    """Return `self` for diffusers-style compatibility.

    Some diffusers utilities expect a wrapper object exposing
    `config.context_parallel_config`. Cache-dit now passes the
    `_ContextParallelConfig` directly, so this property keeps those
    call sites working while avoiding an extra wrapper class.
    """

    return self

  def setup(
    self,
    rank: int,
    world_size: int,
    device: torch.device,
    mesh: torch.distributed.device_mesh.DeviceMesh,
  ) -> None:
    """Attach runtime mesh metadata to the config.

    :param rank: Current distributed rank.
    :param world_size: Total process count.
    :param device: Current device for this rank.
    :param mesh: Device mesh backing the CP region.
    """

    self._rank = rank
    self._world_size = world_size
    self._device = device
    self._mesh = mesh

    if self.ulysses_degree * self.ring_degree > world_size:
      raise ValueError(
        f"The product of ring_degree ({self.ring_degree}) and ulysses_degree ({self.ulysses_degree}) must not exceed world size ({world_size})."
      )

    self._flattened_mesh = self._mesh["ring", "ulysses"]._flatten()
    self._ring_mesh = self._mesh["ring"]
    self._ulysses_mesh = self._mesh["ulysses"]
    self._ring_local_rank = self._ring_mesh.get_local_rank()
    self._ulysses_local_rank = self._ulysses_mesh.get_local_rank()


@dataclasses.dataclass(frozen=True)
class _ContextParallelInput:
  """Describe how one forward input should be sharded.

  :param split_dim: Dimension to split.
  :param expected_dims: Optional dimensionality check.
  :param split_output: Whether the tensor should be split after forward.
  """

  split_dim: int
  expected_dims: int | None = None
  split_output: bool = False


@dataclasses.dataclass(frozen=True)
class _ContextParallelOutput:
  """Describe how one forward output should be gathered.

  :param gather_dim: Dimension to gather.
  :param expected_dims: Optional dimensionality check.
  """

  gather_dim: int
  expected_dims: int | None = None


_ContextParallelInputType = dict[str | int, _ContextParallelInput | list[_ContextParallelInput]
                                 | tuple[_ContextParallelInput, ...]]
_ContextParallelOutputType = _ContextParallelOutput | list[_ContextParallelOutput] | tuple[
  _ContextParallelOutput, ...]
_ContextParallelModelPlan = dict[str, _ContextParallelInputType | _ContextParallelOutputType]
