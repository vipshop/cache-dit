"""Shared CLI helpers; the feature modules live in `entrypoint`, `cache`, `parallel`, `quantize`,
`attention`, `compile`, `offload` and `optimize`.

The public names are re-exported lazily so `from cache_dit.cli.utils import ...` keeps working
without an import cycle between the helper module and the feature modules.
"""

import ast
import argparse
from typing import Any, Dict, Optional

from ..logger import init_logger
from ..platforms import current_platform
from ..summary import strify as summary_strify

logger = init_logger(__name__)

# Default of the `rdt` key in `--cache-params`. Model-specific cache presets treat "the flag still holds
# this value" as "the user did not touch it", so the literal has to live in one place.
DEFAULT_RESIDUAL_DIFF_THRESHOLD = 0.12


class MemoryTracker:
  """Track peak GPU memory usage during execution."""

  def __init__(self, device=None):
    self.device = device if device is not None else current_platform.current_device()
    self.enabled = current_platform.is_accelerator_available()
    self.peak_memory = 0

  def __enter__(self):
    if self.enabled:
      current_platform.reset_peak_memory_stats(self.device)
      current_platform.synchronize(self.device)
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    if self.enabled:
      current_platform.synchronize(self.device)
      self.peak_memory = current_platform.max_memory_allocated(self.device)

  def get_peak_memory_gb(self):
    """Get peak memory in GB.

    :returns: The resolved peak memory gb.
    """
    return self.peak_memory / (1024 ** 3)

  def report(self):
    """Print memory usage report."""
    if self.enabled:
      peak_gb = self.get_peak_memory_gb()
      logger.info(f"Peak GPU memory usage: {peak_gb:.2f} GB")
      return peak_gb
    return 0


def GiB():
  try:
    if not current_platform.is_accelerator_available():
      return 0
    total_memory_bytes = current_platform.get_device_properties(
      current_platform.current_device(), ).total_memory
    total_memory_gib = total_memory_bytes / (1024 ** 3)
    return int(total_memory_gib)
  except Exception:
    return 0


def parse_extra_input_kwargs(s: str) -> Optional[Dict[str, Any]]:
  """Parse a Python dict literal string into a dict.

  Supports int, float, bool, None, list, dict, tuple, str values.
  Uses :func:`ast.literal_eval` for safe evaluation.

  :param s: A Python dict literal string, e.g. ``"{'key': 123, 'arr': [1,2]}"``.
  :returns: The parsed dict, or None if the input is empty.
  :raises argparse.ArgumentTypeError: If the string is not a valid Python dict literal.
  """
  if s is None or s.strip() == "":
    return None
  try:
    result = ast.literal_eval(s)
  except (ValueError, SyntaxError) as e:
    raise argparse.ArgumentTypeError(
      f"Invalid Python dict literal for --extra-input-kwargs: {e}\n"
      f"Expected format: \"{{'key1': value1, 'key2': value2, ...}}\"")
  if not isinstance(result, dict):
    raise argparse.ArgumentTypeError(
      f"--extra-input-kwargs expects a dict literal, got {type(result).__name__}. "
      f"Expected format: \"{{'key1': value1, 'key2': value2, ...}}\"")
  return result


def strify(args, pipe_or_stats):
  base_str = ""
  if args.height is not None and args.width is not None:
    base_str += f"{args.height}x{args.width}_"
  quantize_type = args.quantize_type if args.quantize else ""
  base_str = summary_strify(pipe_or_stats)
  if quantize_type not in base_str:
    base_str = f"C{int(args.compile)}_{quantize_type}_{base_str}"
  else:
    base_str = f"C{int(args.compile)}_{base_str}"
  if args.parallel_type == "ring" or args.parallel_type == "usp":
    if args.ring_rotate_method != "p2p":
      base_str += f"_rotated_{args.ring_rotate_method}"
    if args.ring_no_convert_to_fp32:
      base_str += "_no_fp32"
  if args.parallel_text_encoder:
    if "_TEP" not in base_str:
      base_str += "_TEP"  # Text Encoder Parallelism
  if args.parallel_vae:
    if "_VAEP" not in base_str:
      base_str += "_VAEP"  # VAE Parallelism
  if args.parallel_controlnet:
    if "_CNP" not in base_str:
      base_str += "_CNP"  # ControlNet Parallelism
  if getattr(args, "adaln_host_residency", False):
    base_str += "_ADALN"  # AdaLN tables streamed, transformer resident
  if args.attn is not None:
    base_str += f"_{args.attn.strip('_')}"
  # Fixed order so the filename does not depend on the order of the --attn-params keys.
  attn_fields = getattr(args, "attn_fields", None) or {}
  if attn_fields.get("hybrid_n_early") is not None:
    base_str += f"_hybrid_n_early_{attn_fields['hybrid_n_early']}"
  for key in ("hybrid", "fp4_hadamard", "fp8_hadamard"):
    if attn_fields.get(key):
      base_str += f"_{key}"
  if attn_fields.get("fp4_pv_mm_type") is not None:
    base_str += f"_fp4_pv_mm_type_{attn_fields['fp4_pv_mm_type']}"
  for key in ("fp4_smooth_v", "fp8_smooth_v"):
    if attn_fields.get(key):
      base_str += f"_{key}"
  if args.cuda_graph:
    base_str += "_cuda_graph"
  # __ -> _, ___ -> _, etc.
  base_str = base_str.strip("_").replace("__", "_").replace("___", "_")
  return base_str


def create_profiler_from_args(args, profile_name=None):
  from ..profile import ProfilerContext

  return ProfilerContext(
    enabled=args.profile,
    activities=getattr(args, "profile_activities", ["CPU", "GPU"]),
    output_dir=getattr(args, "profile_dir", None),
    profile_name=profile_name or getattr(args, "profile_name", None),
    with_stack=getattr(args, "profile_with_stack", True),
    record_shapes=getattr(args, "profile_record_shapes", True),
  )


# Backwards-compatible view of the feature modules: `from cache_dit.cli.utils import get_args`
# still works, and the modules themselves import these helpers without a cycle.
_LAZY_EXPORTS = {
  "get_args": "entrypoint",
  "get_base_args": "entrypoint",
  "maybe_postprocess_args": "entrypoint",
  "parse_calibrator_params": "cache",
  "parse_cli_calibrator_params": "cache",
  "resolve_calibrator_params": "cache",
  "build_calibrator_config": "cache",
  "_resolve_cli_cache_params": "cache",
  "_resolve_cli_parallelism": "parallel",
  "get_text_encoder_from_pipe": "parallel",
  "prepare_extra_parallel_modules": "parallel",
  "get_rank_device": "parallel",
  "maybe_init_distributed": "parallel",
  "maybe_destroy_distributed": "parallel",
  "parse_quant_communicate_params": "quantize",
  "_base_svdq_kwargs": "quantize",
  "_derive_cli_svdq_kwargs": "quantize",
  "_resolve_cli_quantization": "quantize",
  "_should_defer_few_shot_pipeline_move": "quantize",
  "_should_skip_initial_few_shot_pipeline_move": "quantize",
  "_register_deferred_few_shot_pipeline_move_callbacks": "quantize",
  "maybe_finalize_deferred_svdq_pipe_move": "quantize",
  "_resolve_cli_quantize_exclude_layers": "quantize",
  "_quantize_fields": "quantize",
  "_resolve_cli_quantize_backend": "quantize",
  "build_quantize_config": "quantize",
  "maybe_quantize_transformer": "quantize",
  "maybe_quantize_text_encoder": "quantize",
  "pipe_quant_bnb_4bit_config": "quantize",
  "_apply_attn_params": "attention",
  "_compile_mode": "compile",
  "_compile_options": "compile",
  "_force_compile_dynamic": "compile",
  "_compile_transformer_module": "compile",
  "_register_deferred_few_shot_compile_callback": "compile",
  "_should_defer_few_shot_compile": "compile",
  "maybe_compile_transformer": "compile",
  "maybe_compile_text_encoder": "compile",
  "maybe_compile_controlnet": "compile",
  "maybe_compile_vae": "compile",
  "maybe_vae_tiling_or_slicing": "offload",
  "maybe_cpu_offload": "offload",
  "_append_generic_offload_target": "offload",
  "_resolve_generic_offload_targets": "offload",
  "_is_text_encoder_offload_target": "offload",
  "_keep_vision_tower_resident": "offload",
  "_stream_filter_excluding": "offload",
  "_needs_vision_inputs": "offload",
  "_layerwise_kwargs_for": "offload",
  "_resolve_layerwise_target_arg": "offload",
  "_resolve_layerwise_target_kwargs": "offload",
  "_apply_layerwise_cpu_offload": "offload",
  "_declared_offload_owner": "offload",
  "_apply_declared_residency": "offload",
  "_adaln_offload_target_names": "offload",
  "_move_to_device_without_adaln": "offload",
  "maybe_adaln_host_residency": "offload",
  "maybe_generic_module_offload": "offload",
  "maybe_apply_optimization": "optimize",
}


def __getattr__(name: str):
  """Import a feature function on first access to keep `utils` import-cycle free."""

  module_name = _LAZY_EXPORTS.get(name)
  if module_name is None:
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
  from importlib import import_module

  value = getattr(import_module(f".{module_name}", __package__), name)
  globals()[name] = value
  return value


def __dir__(self):
  return sorted(set(globals()) | set(_LAZY_EXPORTS))
