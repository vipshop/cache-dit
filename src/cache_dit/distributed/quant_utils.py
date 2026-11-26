"""Configuration helpers for Ulysses low-precision communication."""

from typing import Any, Dict


def _as_bool(value: Any) -> bool:
  if isinstance(value, bool):
    return value
  text = str(value).strip().lower()
  if text in ("1", "true", "yes", "on"):
    return True
  if text in ("0", "false", "no", "off"):
    return False
  raise ValueError(f"quantize_k expects a boolean, got {value!r}.")


def normalize_quant_communicate_kwargs(kwargs: Dict[str, Any] | None) -> Dict[str, Any]:
  """Validate and populate defaults for low-precision Ulysses communication.

  :param kwargs: User options with `bits`, `quant_type`, and `quantize_k`; empty disables it.
  :returns: Normalized options with all keys populated and K quantization enabled by default.
  """
  if not kwargs:
    return {}
  unknown = set(kwargs) - {"bits", "quant_type", "quantize_k"}
  if unknown:
    raise ValueError(f"Unknown quant_communicate_kwargs keys: {sorted(unknown)}.")
  bits = int(kwargs.get("bits", 8))
  if bits not in (4, 6, 8):
    raise ValueError(f"quant_communicate_kwargs bits must be 4, 6, or 8, got {bits}.")
  default_quant_type = "per_group_16"
  quant_type = str(kwargs.get("quant_type", default_quant_type)).replace("-", "_")
  if quant_type not in ("per_token", "per_group_16", "per_group_32"):
    raise ValueError(f"Unsupported quant_communicate_kwargs quant_type: {quant_type}.")
  if bits == 4 and quant_type != "per_group_16":
    raise ValueError("FP4 communication supports only quant_type='per_group_16'.")
  return {
    "bits": bits,
    "quant_type": quant_type,
    "quantize_k": _as_bool(kwargs.get("quantize_k", True)),
  }


def quant_communicate_group_size(kwargs: Dict[str, Any]) -> int:
  """Return the kernel group size from normalized communication options.

  :param kwargs: Normalized communication options.
  :returns: Zero for per-token, otherwise 16 or 32.
  """
  quant_type = kwargs.get("quant_type", "per_token")
  return {
    "per_token": 0,
    "per_group_16": 16,
    "per_group_32": 32,
  }[quant_type]
