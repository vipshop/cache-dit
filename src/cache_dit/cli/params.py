"""Shared `k=v,k=v` parsing for CLI params flags.

Syntax is validated while argparse runs; keys are resolved once the flag that scopes them
(`--quantize-params`, `--svdq-params`, ...) is known, mirroring `--calibrator-params`.
"""

import argparse
from typing import Any, Callable, Dict, Optional, Tuple

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

# Attention backends selectable with `--attn-params type=<backend>`. Those with options are the
# keys of `ATTN_PARAM_SPECS`.
ATTN_BACKENDS = (
  "flash",
  "flash_varlen",
  "_flash_3",  # FlashAttention-3
  # Based on this fix: https://github.com/huggingface/diffusers/pull/12563
  "native",  # native pytorch attention: sdpa
  "_native_cudnn",
  # '_sdpa_cudnn' is only in cache-dit to support context parallelism
  # with attn masks, e.g., ZImage. It is not in diffusers yet.
  "_sdpa_cudnn",
  "sage",  # Need install sageattention: https://github.com/thu-ml/SageAttention
  "sage3",  # SageAttention-3 Blackwell FP4 backend
  "sol_attn",  # NVIDIA Sol-Attn sparse attention (BF16, head_dim=128)
  # ffpa-attn CUDA backends (fp16/fp8: sm_120 + sm_89; fp4: sm_120 only;
  # forward inference only; attn_mask supported without CP):
  # Need install ffpa-attn: https://github.com/vipshop/ffpa-attn
  "ffpa",  # CUTE_TMA fp16/bf16
  "ffpa_fp8",  # CUTE_TMA_FP8
  "ffpa_fp4",  # CUTE_TMA_FP4 NVFP4 (sm_120 only)
  "ffpa_fp8_per_block",  # CUTE_TMA_FP8 per-block quant (fastest, lowest precision)
  "_native_npu",  # native npu attention
  "_npu_fia",  # npu fused infer attention
  "_mindiesd_laser",  # MindIE-SD laser attention
)


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


def parse_attn_params(value: str) -> Dict[str, str]:
  """Parse the CLI `--attn-params "type=<backend>,k=v,k=v"`.

  The `type` key selects the attention backend, so it is required and validated here; the other
  keys are resolved against that backend later in :func:`resolve_attn_params`.

  :param value: Raw argument value.
  :returns: The raw string mapping, including `type`.
  """

  example = "'type=ffpa_fp8,fp8_hadamard=true'"
  raw = parse_param_pairs(value, "--attn-params", example, allow_empty=False)
  if "type" not in raw:
    raise argparse.ArgumentTypeError(f"--attn-params needs a 'type' key, e.g. {example}.")
  if raw["type"].lower() not in ATTN_BACKENDS:
    raise argparse.ArgumentTypeError(f"Invalid --attn-params 'type={raw['type']}'. Expected one of "
                                     f"{list(ATTN_BACKENDS)}.")
  return raw


def resolve_attn_params(raw: Optional[Dict[str, str]]) -> Tuple[Optional[str], Dict[str, Any]]:
  """Split `--attn-params` into the attention backend and the options of that backend.

  :param raw: Raw mapping from the CLI, including the `type` selector; None when the flag was
    not passed.
  :returns: `(backend or None, coerced options)`.
  """

  if not raw:
    return None, {}
  backend = raw["type"].lower()
  options = {key: text for key, text in raw.items() if key != "type"}
  if not options:
    return backend, {}
  specs = ATTN_PARAM_SPECS.get(backend)
  if specs is None:
    raise ValueError(f"The '{backend}' attention backend takes no --attn-params options; backends "
                     f"with options: {list(ATTN_PARAM_SPECS)}.")
  return backend, resolve_param_pairs(options, specs, "--attn-params")


def _degree(text: str) -> int:

  def _cast(value: str) -> int:
    return int(value)

  value = _cast(text.strip())
  if value < 1:
    raise ValueError(f"expects an integer >= 1, got '{text}'.")
  return value


# `--parallel-params` keys. `ulysses` / `ring` / `tp` are the parallelism degrees; their product
# must equal the world size. The other keys select parallel components and Ulysses / Ring options.
PARALLEL_PARAM_SPECS: Dict[str, Caster] = {
  "ulysses": _degree,
  "ring": _degree,
  "tp": _degree,
  "vae": _as_bool,
  "text": _as_bool,
  "controlnet": _as_bool,
  "ulysses_anything": _as_bool,
  "ulysses_async": _as_bool,
  "ring_rotate_method": _one_of(("allgather", "p2p")),
  "ring_convert_to_fp32": _as_bool,
}
PARALLEL_PARAM_ALIASES = {
  "uaa": "ulysses_anything",
  "uaqkv": "ulysses_async",
  "rotate": "ring_rotate_method",
}
PARALLEL_AXES = ("ulysses", "ring", "tp")
# Name of the hybrid parallelism types, by the axes they combine.
_PARALLEL_TYPE_NAMES = {
  ("ulysses", ): "ulysses",
  ("ring", ): "ring",
  ("tp", ): "tp",
  ("ulysses", "ring"): "usp",
  ("ulysses", "tp"): "ulysses_tp",
  ("ring", "tp"): "ring_tp",
  ("ulysses", "ring", "tp"): "usp_tp",
}


def resolve_parallel_params(raw: Optional[Dict[str, str]]) -> Tuple[str, Tuple[int, ...], Dict]:
  """Resolve `--parallel-params` into the parallelism type, its degrees and its options.

  :param raw: Raw mapping from the CLI; None when the flag was not passed.
  :returns:`(parallelism type or None, active axes, coerced fields)`. An axis is active when its
    degree is greater than one, or when it is the only degree given (a single GPU still runs e.g.
    `ulysses=1` through the Ulysses path). Degrees default to one.
  """

  if not raw:
    return None, (), {}
  canonical: Dict[str, str] = {}
  for key, text in raw.items():
    key = PARALLEL_PARAM_ALIASES.get(key, key)
    if key in canonical:
      raise ValueError(f"--parallel-params key '{key}' is given twice.")
    canonical[key] = text
  fields = resolve_param_pairs(canonical, PARALLEL_PARAM_SPECS, "--parallel-params")
  active = tuple(axis for axis in PARALLEL_AXES if fields.get(axis, 1) > 1)
  if not active:
    given = tuple(axis for axis in PARALLEL_AXES if axis in fields)
    active = given if len(given) == 1 else ()
  return _PARALLEL_TYPE_NAMES.get(active), active, fields
