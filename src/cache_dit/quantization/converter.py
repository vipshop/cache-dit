"""Convert a diffusion model to SVDQ int4 quantized format.

Usage::

    python3 -m cache_dit.quantization.converter \
        --model-path /path/to/FLUX.1-dev \
        --save-dir ./FLUX.1-dev-svdq \
        --quant-type svdq-int4-r128-dq

Only SVDQ dynamic quantization (``_dq`` quant types) is currently supported.
"""

from __future__ import annotations

import argparse
import os
import sys
import torch

from ..logger import init_logger

logger = init_logger(__name__)


def _get_args(argv: list[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Convert a diffusion model to SVDQ int4 quantized format.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument(
    "--model-path",
    type=str,
    required=True,
    help="Path to the pretrained diffusion model directory (e.g. /path/to/FLUX.1-dev).",
  )
  parser.add_argument(
    "--save-dir",
    type=str,
    required=True,
    help="Directory where the quantized checkpoint and config will be saved.",
  )
  parser.add_argument(
    "--quant-type",
    type=str,
    required=True,
    help="SVDQ quant type, e.g. 'svdq-int4-r128-dq' or 'svdq_int4_r128_dq'. "
    "Currently only DQ types (ending with '_dq') are supported.",
  )
  parser.add_argument(
    "--torch-dtype",
    type=str,
    default="bfloat16",
    choices=("float16", "bfloat16", "float32"),
    help="Torch dtype used when loading the float pipeline.",
  )
  parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="Device to run quantization on.",
  )
  parser.add_argument(
    "--svdq-smooth-strategy",
    type=str,
    default=None,
    choices=("identity", "weight", "weight_inv", "few_shot"),
    help="SVDQ DQ smooth strategy. Defaults to 'identity' when not set.",
  )
  parser.add_argument(
    "--svdq-calibrate-precision",
    type=str,
    default="low",
    choices=("low", "medium", "high"),
    help="Precision plan for SVDQ calibration math and low-rank decomposition.",
  )
  parser.add_argument(
    "--svdq-runtime-kernel",
    type=str,
    default="v1",
    choices=("v1", "v2"),
    help="Packed runtime GEMM kernel used by SVDQW4A4Linear.",
  )
  parser.add_argument(
    "--verbose",
    action="store_true",
    default=False,
    help="Print detailed quantization information.",
  )
  return parser.parse_args(argv)


def _resolve_torch_dtype(torch_dtype: str):

  return {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
  }[torch_dtype]


def _module_memory_gib(module) -> float:
  """Return the total memory of all parameters in *module*, in GiB."""

  total_bytes = 0
  for param in module.parameters():
    total_bytes += param.numel() * param.element_size()
  return total_bytes / (1024 * 1024 * 1024)


def entrypoint(argv: list[str] | None = None) -> None:
  args = _get_args(argv)

  quant_type = args.quant_type.replace("-", "_").lower()
  if not quant_type.endswith("_dq"):
    logger.error(
      "Currently only SVDQ dynamic quantization (quant types ending with '_dq') is supported. "
      "Got %r.",
      args.quant_type,
    )
    sys.exit(1)

  from diffusers import DiffusionPipeline

  from cache_dit.quantization import QuantizeConfig
  from cache_dit.quantization import quantize

  torch_dtype = _resolve_torch_dtype(args.torch_dtype)
  device = torch.device(args.device)

  # --- Load the pipeline ---
  logger.info("Loading pipeline from %s ...", args.model_path)
  pipe = DiffusionPipeline.from_pretrained(
    args.model_path,
    torch_dtype=torch_dtype,
  )
  pipe.to(device)
  logger.info("Pipeline loaded: %s", type(pipe).__name__)

  # --- Build quantization config ---
  svdq_kwargs = {
    "calibrate_precision": args.svdq_calibrate_precision,
    "runtime_kernel": args.svdq_runtime_kernel,
  }
  if args.svdq_smooth_strategy is not None:
    svdq_kwargs["smooth_strategy"] = args.svdq_smooth_strategy

  save_dir = os.path.abspath(args.save_dir)
  config = QuantizeConfig(
    quant_type=quant_type,
    serialize_to=save_dir,
    svdq_kwargs=svdq_kwargs if svdq_kwargs else None,
    verbose=args.verbose,
  )

  # --- Quantize and serialize ---
  memory_before_gib = _module_memory_gib(pipe.transformer)
  logger.info("Transformer memory before quantize: %.2f GiB", memory_before_gib)
  logger.info("Quantizing transformer (%s) and saving to %s ...", quant_type, save_dir)
  pipe.transformer = quantize(pipe.transformer, config)
  memory_after_gib = _module_memory_gib(pipe.transformer)
  logger.info("Transformer memory after quantize: %.2f GiB (%.1fx reduction)", memory_after_gib,
              memory_before_gib / memory_after_gib)

  expected_safetensors = os.path.join(save_dir, f"{quant_type}.safetensors")
  expected_config = os.path.join(save_dir, "quant_config.json")
  logger.info("Quantized checkpoint saved to %s", expected_safetensors)
  logger.info("Quant config saved to %s", expected_config)

  # --- Verify the artifacts exist ---
  if not os.path.isfile(expected_safetensors):
    logger.error("Expected safetensors file not found: %s", expected_safetensors)
    sys.exit(1)
  if not os.path.isfile(expected_config):
    logger.error("Expected quant_config.json not found: %s", expected_config)
    sys.exit(1)

  logger.info("Conversion complete. Load the quantized model with:")
  logger.info("  cache_dit.load(transformer, %r)", save_dir)
  logger.info("  cache_dit.load(transformer, %r)", expected_safetensors)


def main() -> None:
  entrypoint()


if __name__ == "__main__":
  main()
