"""`--cache-params` / `--calibrator-params` parsing and resolution."""

import argparse
from typing import Any, Dict, Optional, Tuple

from ..caching import (
  CalibratorConfig,
  DMDCalibratorConfig,
  FoCaCalibratorConfig,
  SVDCalibratorConfig,
  HiCacheCalibratorConfig,
  BRACECalibratorConfig,
  BeziCastCalibratorConfig,
  ResidualMotionCalibratorConfig,
  TaylorSeerCalibratorConfig,
)
from ..caching.cache_contexts.calibrators import normalize_calibrator_type
from ..logger import init_logger
from .params import parse_param_pairs, resolve_cache_params
from .utils import DEFAULT_RESIDUAL_DIFF_THRESHOLD

logger = init_logger(__name__)


def _optional_float(text: str) -> Optional[float]:
  """Parse an optional float, accepting `auto` or `none` as None."""

  if text.strip().lower() in ("auto", "none"):
    return None
  return float(text)


# `--calibrator-params` short keys per calibrator: key -> (config field, caster).
_CALIBRATOR_PARAM_SPECS: Dict[str, Dict[str, Tuple[str, Any]]] = {
  "taylorseer": {
    "order": ("taylorseer_order", int),
  },
  "dmd": {
    "history": ("dmd_history", int),
    "rank": ("dmd_rank", int),
    "ridge": ("dmd_ridge", float),
    "svd_precision": ("dmd_svd_precision", str),
  },
  "foca": {},
  "svd": {
    "tau": ("svd_tau", float),
    "beta": ("svd_beta", float),
    "max_rank": ("svd_max_rank", int),
    "predictor": ("svd_predictor", str),
    "order": ("svd_order", int),
  },
  "hicache": {
    "order": ("hicache_order", int),
    "sigma": ("hicache_sigma", float),
  },
  "brace": {
    "capacity": ("brace_capacity", int),
    "gamma": ("brace_gamma", float),
  },
  "bezicast": {
    "history": ("bezicast_history", int),
    "control_points": ("bezicast_control_points", int),
    "ridge": ("bezicast_ridge", float),
    "guardrail": ("bezicast_guardrail", bool),
    "u_max": ("bezicast_u_max", float),
    "beta_max": ("bezicast_beta_max", float),
    "rho": ("bezicast_rho", _optional_float),
    "gamma": ("bezicast_gamma", float),
  },
  "residual_motion": {
    "beta": ("residual_motion_beta", float),
  },
}

# Keys every calibrator accepts; they live on `CalibratorConfig` itself.
_COMMON_CALIBRATOR_PARAM_SPECS: Dict[str, Tuple[str, Any]] = {
  "calibrator_cache_type": ("calibrator_cache_type", str),
  "cache_type": ("calibrator_cache_type", str),  # deprecated alias
  "encoder": ("enable_encoder_calibrator", bool),
  "enable_encoder_calibrator": ("enable_encoder_calibrator", bool),
}

_CALIBRATOR_PARAM_CHOICES = {
  "calibrator_cache_type": ("residual", "hidden_states"),
  "dmd_svd_precision": ("low", "medium", "high"),
  "svd_predictor": ("ema", "taylor"),
  "brace_capacity": (1, 2, 3),
  "bezicast_history": (2, 3, 4, 5, 6),
}

_CALIBRATOR_CONFIG_CLASSES = {
  "taylorseer": TaylorSeerCalibratorConfig,
  "dmd": DMDCalibratorConfig,
  "foca": FoCaCalibratorConfig,
  "svd": SVDCalibratorConfig,
  "hicache": HiCacheCalibratorConfig,
  "brace": BRACECalibratorConfig,
  "bezicast": BeziCastCalibratorConfig,
  "residual_motion": ResidualMotionCalibratorConfig,
}


def _parse_calibrator_arg(value: str) -> str:
  """Normalize a calibrator name (the `type` key of `--calibrator-params`) into its short name.

  :param value: Raw argument value, e.g. `svd`, `svd-calibrator`, or `svd-cache`.
  :returns: The canonical calibrator name.
  """

  name = normalize_calibrator_type(value)
  if name not in _CALIBRATOR_CONFIG_CLASSES:
    raise argparse.ArgumentTypeError(
      f"Unknown calibrator '{value}'. Expected one of "
      f"{sorted(_CALIBRATOR_CONFIG_CLASSES)}; long forms such as 'svd-calibrator' are accepted.")
  return name


def parse_calibrator_params(value: str) -> Optional[Dict[str, str]]:
  """Parse a `--calibrator-params "k=v,k=v"` list into a raw key -> value mapping.

  Only the syntax is validated here. The generate CLI goes through
  :func:`parse_cli_calibrator_params`, which also requires the `type` key.

  :param value: Raw argument value.
  :returns: The raw string mapping, or None for an empty value.
  """

  return parse_param_pairs(value, "--calibrator-params", "'type=svd,tau=0.9,beta=0.9'")


def parse_cli_calibrator_params(value: str) -> Dict[str, str]:
  """Parse the generate CLI `--calibrator-params "type=taylorseer,k=v,k=v"`.

  The `type` key selects the calibrator, so it is required and validated here. It is not a
  calibrator option: it is dropped in `maybe_postprocess_args` before the other keys are resolved
  in :func:`resolve_calibrator_params`.

  :param value: Raw argument value.
  :returns: The raw string mapping, including `type`.
  """

  example = "'type=svd,tau=0.9,beta=0.9'"
  raw = parse_param_pairs(value, "--calibrator-params", example, allow_empty=False)
  if "type" not in raw:
    raise argparse.ArgumentTypeError(f"--calibrator-params needs a 'type' key, e.g. {example}.")
  # Fail here rather than after the model has been loaded.
  _parse_calibrator_arg(raw["type"])
  return raw


def _coerce_calibrator_param(field: str, caster: Any, text: str) -> Any:
  """Coerce one `--calibrator-params` value into the type its config field expects.

  :param field: Target config field name (used for enum validation).
  :param caster: `bool`, `int`, `float`, or `str`.
  :param text: Raw value text.
  :returns: The coerced value.
  """

  if caster is bool:
    low = text.strip().lower()
    if low in ("1", "true", "yes", "on"):
      value = True
    elif low in ("0", "false", "no", "off"):
      value = False
    else:
      raise ValueError(f"'{field}' expects a boolean, got '{text}'.")
    return value
  value = caster(text)
  choices = _CALIBRATOR_PARAM_CHOICES.get(field)
  if choices is not None and value not in choices:
    raise ValueError(f"'{field}' expects one of {list(choices)}, got '{value}'.")
  return value


def resolve_calibrator_params(calibrator_type: str, raw: Optional[Dict[str,
                                                                       str]]) -> Dict[str, Any]:
  """Map raw `--calibrator-params` keys onto config fields of the selected calibrator.

  Every key must be a known option of that calibrator, so callers drop the selector `type` key
  first (see `maybe_postprocess_args`).

  :param calibrator_type: Canonical calibrator name.
  :param raw: Raw key -> value mapping from the command line.
  :returns: Config field -> coerced value mapping.
  """

  if not raw:
    return {}
  spec = dict(_CALIBRATOR_PARAM_SPECS.get(calibrator_type, {}))
  spec.update(_COMMON_CALIBRATOR_PARAM_SPECS)
  valid_keys = sorted(key for key in spec if key != "cache_type")
  resolved: Dict[str, Any] = {}
  for key, text in raw.items():
    if key not in spec:
      raise ValueError(f"Unknown --calibrator-params key '{key}' for calibrator "
                       f"'{calibrator_type}'. Valid keys: {valid_keys}.")
    if key == "cache_type":
      logger.warning("--calibrator-params 'cache_type' is deprecated; use "
                     "'calibrator_cache_type' instead.")
    field, caster = spec[key]
    resolved[field] = _coerce_calibrator_param(field, caster, text)
  return resolved


def build_calibrator_config(args) -> Optional[CalibratorConfig]:
  """Build the selected calibrator config from resolved --calibrator-params fields.

  Unspecified options use the selected config dataclass's defaults.

  :param args: Parsed command line arguments.
  :returns: The calibrator config, or None when no calibrator was requested.
  """

  calibrator_type = getattr(args, "calibrator", None)
  if calibrator_type is None:
    return None
  fields = getattr(args, "calibrator_fields", None) or {}
  return _CALIBRATOR_CONFIG_CLASSES[calibrator_type](**fields)


def _resolve_cli_cache_params(args) -> None:
  """Resolve `--cache-params` into the plain attributes the cache setup reads.

  Passing the flag enables the cache, so `type=DBCache` alone is the former `--cache`. The `type`
  key seeds the SeaCache / SoftCache switches and any explicit key overrides them. The DBCache
  defaults are always written back: the per-model presets treat `residual_diff_threshold` holding
  its default as "the user did not touch it".

  :param args: Parsed CLI namespace, updated in place.
  """

  cache_type, fields = resolve_cache_params(args.cache_params)
  args.cache_type = cache_type
  args.cache = args.cache_params is not None
  args.Fn_compute_blocks = fields.get("Fn_compute_blocks", 1)
  args.Bn_compute_blocks = fields.get("Bn_compute_blocks", 0)
  args.residual_diff_threshold = fields.get("residual_diff_threshold",
                                            DEFAULT_RESIDUAL_DIFF_THRESHOLD)
  args.max_warmup_steps = fields.get("max_warmup_steps", 8)
  args.warmup_interval = fields.get("warmup_interval", 1)
  args.max_cached_steps = fields.get("max_cached_steps", -1)
  args.max_continuous_cached_steps = fields.get("max_continuous_cached_steps", 3)
  mask_policy = fields.get("mask_policy")
  args.mask_policy = {
    "s": "slow",
    "m": "medium",
    "f": "fast",
    "u": "ultra",
  }.get(mask_policy, mask_policy)
  # A mask policy used to imply the mask switch, and still does.
  args.steps_mask = bool(fields.get("steps_mask", False) or args.mask_policy is not None)
  args.seacache = fields.get("seacache", cache_type in ("seacache", "softcache"))
  args.seacache_mode = fields.get("seacache_mode", "flow")
  args.seacache_feature_shape = fields.get("seacache_feature_shape")
  args.seacache_power_exp = fields.get("seacache_power_exp", 2.0)
  args.softcache = fields.get("softcache", cache_type == "softcache")
  args.softcache_tau = fields.get("softcache_tau", 0.5)
  args.softcache_jf_max = fields.get("softcache_jf_max", 8.0)

  if args.seacache and args.config_path is not None:
    raise ValueError("--cache-params SeaCache options cannot be combined with --config-path; "
                     "set SeaCache in YAML.")
  if args.seacache_feature_shape is not None:
    if not args.seacache:
      raise ValueError(
        "--cache-params 'seacache_feature_shape' requires a SeaCache or SoftCache type.")
    if len(args.seacache_feature_shape) not in (2, 3) or any(
        dim <= 0 for dim in args.seacache_feature_shape):
      raise ValueError(
        "--cache-params 'seacache_feature_shape' requires positive H W or F H W dimensions.")
  if args.softcache and not args.seacache:
    raise ValueError("--cache-params 'softcache=true' requires SeaCache.")
  if args.softcache and args.seacache_mode != "flow":
    raise ValueError("--cache-params 'softcache=true' requires 'seacache_mode=flow'.")
  if args.softcache and args.softcache_tau <= 0.0:
    raise ValueError("--cache-params 'softcache_tau' must be positive.")
