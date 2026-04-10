from __future__ import annotations

from typing import Any, Optional


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
