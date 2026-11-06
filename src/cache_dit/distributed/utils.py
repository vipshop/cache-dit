from __future__ import annotations

from typing import Any, Optional

import torch
import torch.distributed


def _divisors(n: int) -> list[int]:
  n = int(n)
  if n <= 0:
    return []
  small: list[int] = []
  large: list[int] = []
  d = 1
  while d * d <= n:
    if n % d == 0:
      small.append(d)
      if d * d != n:
        large.append(n // d)
    d += 1
  return small + list(reversed(large))


def shard_div_attr(
  obj: Any,
  attr: str,
  tp_size: int,
  *,
  what: Optional[str] = None,
  context: Optional[str] = None,
) -> int:
  """Shard (divide) an integer attribute by tp_size, with a fail-fast divisibility check.

  This helper is primarily used for attributes such as attention head counts. It validates that the
  requested tensor-parallel size divides the original value exactly before mutating the attribute in
  place.

  :param obj: Object carrying the integer attribute to shard.
  :param attr: Attribute name to divide by `tp_size`.
  :param tp_size: Requested tensor-parallel size.
  :param what: Optional human-readable object label used in error messages.
  :param context: Optional context prefix used in error messages.
  :returns: The updated attribute value after sharding.
  """
  tp_size = int(tp_size)
  if tp_size <= 0:
    raise ValueError(f"[TP] Invalid tp_size={tp_size}.")

  if not hasattr(obj, attr):
    raise AttributeError(f"[TP] Object {type(obj).__name__} has no attribute '{attr}'.")

  raw = getattr(obj, attr)
  try:
    value = int(raw)
  except Exception as e:
    raise TypeError(
      f"[TP] Attribute '{attr}' on {type(obj).__name__} must be int-like, got {raw!r}.") from e

  if value <= 0:
    raise ValueError(f"[TP] Attribute '{attr}' must be > 0, got {value}.")

  if value % tp_size != 0:
    divs = [d for d in _divisors(value) if d > 1]
    divs_str = ", ".join(map(str, divs)) if divs else "(none)"
    obj_name = what or type(obj).__name__
    prefix = f"{context}: " if context else f"{obj.__class__.__name__}.{attr}: "
    raise ValueError(f"[TP] {prefix}Unsupported tp_size={tp_size} for {obj_name}.{attr}={value}. "
                     f"{attr} must be divisible by tp_size. Valid tp_size (>1): {divs_str}.")

  new_value = value // tp_size
  setattr(obj, attr, new_value)
  return new_value


class _DistributedRMSNorm(torch.nn.Module):
  """TP-aware RMSNorm: the global mean square is all-reduced before normalizing the local shard.

  A QK norm normalizes the full projection width, so after a column-wise shard each rank only sees
  part of the vector. Averaging the partial mean squares (all shards have the same size) restores the
  exact global statistic; the weight itself is sharded along the same axis.

  :param tp_mesh: Tensor parallel mesh.
  :param dim: Normalized shape of the original norm.
  :param eps: Epsilon of the original norm.
  :param weight: Sharded weight (a slice of the original parameter), or None.
  """

  def __init__(
    self,
    tp_mesh: torch.distributed.device_mesh.DeviceMesh,
    dim: torch.Size,
    eps: float,
    weight: Optional[torch.nn.Parameter],
  ):
    super().__init__()
    self.tp_mesh = tp_mesh
    self.dim = dim
    self.eps = eps
    self.weight = weight

  @classmethod
  def from_rmsnorm(cls, tp_mesh: torch.distributed.device_mesh.DeviceMesh,
                   rmsnorm: torch.nn.Module) -> "_DistributedRMSNorm":
    """Build the distributed norm from a diffusers *or* ``torch.nn`` RMSNorm.

    ``diffusers.models.normalization.RMSNorm`` exposes ``dim``, while ``torch.nn.RMSNorm`` exposes
    ``normalized_shape``; both are carried through unchanged.

    :param tp_mesh: Tensor parallel mesh.
    :param rmsnorm: Original norm module.
    :returns: The TP-aware replacement.
    """

    tp_size = tp_mesh.get_group().size()
    tp_rank = tp_mesh.get_group().rank()
    assert getattr(rmsnorm, "bias",
                   None) is None, ("The distributed RMSNorm only implements the bias-free variant.")
    dim = getattr(rmsnorm, "dim", None)
    if dim is None:
      dim = rmsnorm.normalized_shape
    weight = rmsnorm.weight
    if weight is not None:
      weight = torch.nn.Parameter(weight.detach().chunk(tp_size, dim=0)[tp_rank].clone())
    return cls(tp_mesh=tp_mesh, dim=dim, eps=rmsnorm.eps, weight=weight)

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Normalize with the globally reduced mean square.

    :param hidden_states: Sharded activation.
    :returns: Normalized activation of the same shape.
    """

    input_dtype = hidden_states.dtype
    if self.weight is not None:
      assert hidden_states.shape[-1] == self.weight.shape[0], (
        f"Expected the local feature width to match the sharded weight "
        f"({hidden_states.shape[-1]} vs {self.weight.shape[0]}).")
    variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
    torch.distributed.all_reduce(
      variance,
      op=torch.distributed.ReduceOp.AVG,
      group=self.tp_mesh.get_group(),
    )
    hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
    if self.weight is not None:
      hidden_states = self.weight.to(hidden_states.dtype) * hidden_states
    return hidden_states.to(input_dtype)
