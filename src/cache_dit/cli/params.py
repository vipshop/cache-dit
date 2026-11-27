"""Shared `k=v,k=v` parsing for CLI params flags.

Syntax is validated while argparse runs; keys are resolved once the flag that scopes them
(`--quantize-params`, `--svdq-params`, ...) is known, mirroring `--calibrator-params`.
"""

import argparse
from typing import Any, Callable, Dict, Optional

from ..offload import _parse_byte_size_arg
from ..quantization.config import TEXT_ENCODER_QUANT_TYPES, TRANSFORMER_QUANT_TYPES

Caster = Callable[[str], Any]

_TRUE_WORDS = ("1", "true", "yes", "on")
_FALSE_WORDS = ("0", "false", "no", "off")
_NULL_WORDS = ("auto", "none")


def parse_param_pairs(
  value: str,
  flag: str,
  example: str,
  allow_empty: bool = True,
) -> Optional[Dict[str, str]]:
  """Parse a ``k=v,k=v`` list into a raw key -> value mapping.

  Only the syntax is validated here; keys are resolved later against the spec of the flag that
  scopes them, because argparse does not guarantee the scoping flag is parsed first.

  :param value: Raw argument value.
  :param flag: Flag name used in error messages.
  :param example: Example value quoted in the error message.
  :param allow_empty: Whether an empty value is accepted; turn it off for a flag that is itself the
    switch, so an empty shell variable cannot silently disable the feature.
  :returns: The raw mapping, or None for an empty value.
  """

  if value is None or value.strip() == "":
    if allow_empty:
      return None
    raise argparse.ArgumentTypeError(f"{flag} needs at least one key=value pair, e.g. {example}.")
  params: Dict[str, str] = {}
  for item in value.split(","):
    item = item.strip()
    if not item:
      continue
    key, sep, raw = item.partition("=")
    if sep == "" or key.strip() == "":
      raise argparse.ArgumentTypeError(
        f"Invalid {flag} entry '{item}'. Expected 'key=value' pairs, e.g. {example}.")
    params[key.strip().lower()] = raw.strip()
  return params


def pairs_parser(
  flag: str,
  example: str,
  allow_empty: bool = True,
) -> Callable[[str], Optional[Dict[str, str]]]:
  """Build the argparse ``type`` of one params flag.

  :param flag: Flag name used in error messages.
  :param example: Example value quoted in the error message.
  :param allow_empty: Whether an empty value is accepted, see `parse_param_pairs`.
  :returns: A single-argument callable for ``add_argument(type=...)``.
  """

  return lambda value: parse_param_pairs(value, flag, example, allow_empty)


def _as_bool(text: str) -> bool:
  low = text.strip().lower()
  if low in _TRUE_WORDS:
    return True
  if low in _FALSE_WORDS:
    return False
  raise ValueError(f"expects a boolean, got '{text}'.")


def _one_of(choices, normalize=lambda text: text) -> Caster:

  def _cast(text: str) -> Any:
    value = normalize(text.strip())
    if value not in choices:
      raise ValueError(f"expects one of {list(choices)}, got '{text}'.")
    return value

  return _cast


def _optional(caster: Caster) -> Caster:

  def _cast(text: str) -> Any:
    low = text.strip().lower()
    return None if low in _NULL_WORDS else caster(text)

  return _cast


def _plus_list(text: str) -> list:
  return [item.strip() for item in text.split("+") if item.strip()]


def _snake(text: str) -> str:
  return text.strip().lower().replace("-", "_")


_quant_type = _one_of(TRANSFORMER_QUANT_TYPES, _snake)
_text_encoder_type = _one_of(TEXT_ENCODER_QUANT_TYPES, _snake)


def _text_encoder(text: str) -> Any:
  low = text.strip().lower()
  if low in _TRUE_WORDS:
    return True
  return _text_encoder_type(text)


# `--quantize-params` keys. All but `text_encoder` are `QuantizeConfig` fields and share their name;
# `text_encoder` selects a component on the CLI and dispatches to the text encoder quantization
# helpers.
QUANTIZE_PARAM_SPECS: Dict[str, Caster] = {
  "quant_type": _quant_type,
  "backend": _one_of(("torchao", "cache_dit")),
  "regional_quantize": _as_bool,
  "per_tensor_fallback": _as_bool,
  "exclude_layers": _plus_list,
  "text_encoder": _text_encoder,
  "verbose": _as_bool,
  "fp8_fused_mlp": _as_bool,
  "fp4_fused_mlp": _as_bool,
  "mxfp4_quantize_device": _one_of(("auto", "cpu", "cuda")),
}

# `--svdq-params` keys: exactly the `svdq_kwargs` accepted by `QuantizeConfig`, same order.
SVDQ_PARAM_SPECS: Dict[str, Caster] = {
  "streaming": _as_bool,
  "calibrate_precision": str,
  "runtime_kernel": _optional(str),
  "lora_layout": _optional(str),
  "quantize_device": str,
  "offload_quantized_layers_to_cpu": _as_bool,
  "layerwise_offload": _as_bool,
  "async_transfer": _as_bool,
  "transfer_buckets": int,
  "prefetch_limit": _as_bool,
  "max_copy_streams": _optional(int),
  "max_inflight_prefetch_bytes": _optional(_parse_byte_size_arg),
  "persistent_buckets": int,
  "persistent_bins": int,
  "defer_move_to_execution_device": _as_bool,
  "activation_buffer_flush_sample_count": _optional(int),
  "activation_buffer_flush_cpu_bytes": _optional(_parse_byte_size_arg),
  "smooth_strategy": str,
  "few_shot_steps": _optional(int),
  "few_shot_relax_factor": float,
  "few_shot_relax_top_ratio": float,
  "few_shot_relax_strategy": str,
  "few_shot_auto_compile": _as_bool,
  "fused_mlp": _as_bool,
  "svd_refine_iters": int,
  "packed_a4_stages": int,
  "packed_a4_inner_group": int,
}

# `--attn-params` keys per attention backend. Every key maps onto one process-wide setter of its
# backend: `set_sol_attn_config(**fields)` for sol_attn, `set_ffpa_<key>(value)` for ffpa*.
_FFPA_HYBRID_KEYS = {
  "hybrid": _as_bool,
  "hybrid_n_early": int,
}
ATTN_PARAM_SPECS: Dict[str, Dict[str, Caster]] = {
  "sol_attn": {
    "tau": float,
    "thresh_type": _one_of(("diag", "exact")),
    "strict": _as_bool,
    "sink_tokens": int,
  },
  "ffpa": dict(_FFPA_HYBRID_KEYS),
  "ffpa_fp8": {
    **_FFPA_HYBRID_KEYS,
    "fp8_hadamard": _as_bool,
    "fp8_smooth_v": _as_bool,
  },
  "ffpa_fp4": {
    **_FFPA_HYBRID_KEYS,
    "fp4_hadamard": _as_bool,
    "fp4_pv_mm_type": _one_of(("fp4", "fp8")),
    "fp4_smooth_v": _as_bool,
  },
}
ATTN_PARAM_SPECS["ffpa_fp8_per_block"] = ATTN_PARAM_SPECS["ffpa_fp8"]


def resolve_param_pairs(
  raw: Optional[Dict[str, str]],
  specs: Dict[str, Caster],
  flag: str,
) -> Dict[str, Any]:
  """Validate raw pairs against a spec table and coerce every value to its typed form.

  :param raw: Raw mapping from :func:`parse_param_pairs`.
  :param specs: Allowed keys and their casters.
  :param flag: Flag name used in error messages.
  :returns: The coerced mapping, empty when nothing was passed.
  """

  if not raw:
    return {}
  unknown = sorted(set(raw) - set(specs))
  if unknown:
    raise ValueError(f"Unknown {flag} keys {unknown}. Valid keys: {list(specs)}.")
  resolved: Dict[str, Any] = {}
  for key, text in raw.items():
    try:
      resolved[key] = specs[key](text)
    except (ValueError, argparse.ArgumentTypeError) as e:
      raise ValueError(f"Invalid {flag} '{key}={text}': {e}") from e
  return resolved


def resolve_attn_params(backend: Optional[str], raw: Optional[Dict[str, str]]) -> Dict[str, Any]:
  """Resolve `--attn-params` against the keys of the selected attention backend.

  :param backend: Value of `--attn`, or None when it was not passed.
  :param raw: Raw mapping from :func:`parse_param_pairs`.
  :returns: The coerced mapping, empty when nothing was passed.
  """

  if not raw:
    return {}
  if backend is None:
    raise ValueError("--attn-params requires --attn.")
  specs = ATTN_PARAM_SPECS.get(backend)
  if specs is None:
    raise ValueError(f"The '{backend}' attention backend takes no --attn-params; backends with "
                     f"options: {list(ATTN_PARAM_SPECS)}.")
  return resolve_param_pairs(raw, specs, "--attn-params")
