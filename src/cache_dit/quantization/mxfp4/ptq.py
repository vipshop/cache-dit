# coding=utf-8
# Copyright (c) Cache-DiT Contributors. All rights reserved.
"""MXFP4 post-training quantization entry (native Cache-DiT backend)."""

from typing import Dict, List, Optional

import torch

from ..config import QuantizeConfig
from .linear import MXFP4Linear
from .tensor import resolve_mxfp4_quantize_device
from ...logger import init_logger

logger = init_logger(__name__)


def _is_excluded(name: str, exclude_layers: List[str]) -> bool:
  for exclude_name in exclude_layers:
    if exclude_name in name:
      return True
  return False


def _replace_linears(
  container: torch.nn.Module,
  quantize_config: QuantizeConfig,
  stats: Dict[str, int],
  skipped_names: List[str],
) -> None:
  block_size = int(getattr(quantize_config, "mxfp4_block_size", 32) or 32)
  weight_only = quantize_config.quant_type.endswith("weight_only")
  exclude_layers = list(quantize_config.exclude_layers or [])
  requested_device = getattr(quantize_config, "mxfp4_quantize_device", "auto")

  for name, submod in list(container.named_modules()):
    if not isinstance(submod, torch.nn.Linear):
      continue
    if _is_excluded(name, exclude_layers):
      stats["skipped"] += 1
      skipped_names.append(name)
      continue
    if submod.in_features % (2 * block_size) != 0:
      # Cannot pack along K with the requested block size: keep in high precision.
      stats["skipped"] += 1
      skipped_names.append(f"{name} (in_features={submod.in_features} %% {2 * block_size} != 0)")
      continue
    parent_path, _, attr = name.rpartition(".")
    parent = container.get_submodule(parent_path) if parent_path else container
    # Pack on the accelerator when one is available and keep the packed buffers
    # on the weight's original device, so quantization never moves the module.
    quantize_device = resolve_mxfp4_quantize_device(submod.weight, requested_device)
    stats["device"] = stats["device"] or str(quantize_device)
    quantized = MXFP4Linear(submod,
                            block_size=block_size,
                            weight_only=weight_only,
                            quantize_device=quantize_device)
    setattr(parent, attr, quantized)
    stats["quantized"] += 1


def quantize_mxfp4(
  module: torch.nn.Module,
  quantize_config: QuantizeConfig,
  **kwargs,
) -> torch.nn.Module:
  """Replace ``torch.nn.Linear`` layers inside ``module`` with :class:`MXFP4Linear`.

  Honors ``regional_quantize``/``repeated_blocks`` and ``exclude_layers`` the same
  way as the torchao backend does, so the precision behaviour stays consistent
  across backends.
  """
  if getattr(module, "_is_quantized", False):
    logger.warning("Module %s is already quantized (type: %s), skip MXFP4 quantization.",
                   module.__class__.__name__, getattr(module, "_quantize_type", None))
    return module

  stats = {"quantized": 0, "skipped": 0, "device": ""}
  skipped_names: List[str] = []

  repeated_blocks: Optional[List[str]] = (list(
    getattr(quantize_config, "repeated_blocks", None) or getattr(module, "_repeated_blocks", None)
    or []))
  if quantize_config.regional_quantize and repeated_blocks:
    containers = [m for m in module.modules() if m.__class__.__name__ in repeated_blocks]
    if not containers:
      raise ValueError(f"Regional MXFP4 quantization failed because {repeated_blocks} "
                       "classes are not found in the module.")
  else:
    containers = [module]

  for container in containers:
    _replace_linears(container, quantize_config, stats, skipped_names)

  logger.info(
    "MXFP4 quantization done for %s: %d Linear layers quantized (block_size=%d), "
    "%d skipped, packing on %s.", module.__class__.__name__, stats["quantized"],
    int(getattr(quantize_config, "mxfp4_block_size", 32) or 32), stats["skipped"], stats["device"]
    or "no layer")
  if quantize_config.verbose and skipped_names:
    logger.info("MXFP4 skipped layers: %s", skipped_names[:32])

  module._is_quantized = True
  module._quantize_config = quantize_config
  module._quantize_type = quantize_config.quant_type
  return module
