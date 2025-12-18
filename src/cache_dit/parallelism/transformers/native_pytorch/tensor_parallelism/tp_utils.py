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


def shard_divisible_attr(
    obj: Any,
    attr: str,
    tp_size: int,
    *,
    what: Optional[str] = None,
    context: Optional[str] = None,
) -> int:
    """
    Shard (divide) an integer attribute by tp_size, with a fail-fast divisibility check.

    This is primarily used for sharding attention `heads` / `num_heads` in tensor parallelism
    planners. If the value is not divisible by tp_size, we raise a clear ValueError during
    model initialization (before serving / inference).
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
            f"[TP] Attribute '{attr}' on {type(obj).__name__} must be int-like, got {raw!r}."
        ) from e

    if value <= 0:
        raise ValueError(f"[TP] Attribute '{attr}' must be > 0, got {value}.")

    if value % tp_size != 0:
        divs = [d for d in _divisors(value) if d > 1]
        divs_str = ", ".join(map(str, divs)) if divs else "(none)"
        obj_name = what or type(obj).__name__
        prefix = f"{context}: " if context else ""
        raise ValueError(
            f"[TP] {prefix}Unsupported tp_size={tp_size} for {obj_name}.{attr}={value}. "
            f"{attr} must be divisible by tp_size. Valid tp_size (>1): {divs_str}."
        )

    new_value = value // tp_size
    setattr(obj, attr, new_value)
    return new_value
