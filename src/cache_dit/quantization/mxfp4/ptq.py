# coding=utf-8
# Copyright (c) Cache-DiT Contributors. All rights reserved.
"""MXFP4 post-training quantization entry (native Cache-DiT backend).

Layer selection mirrors the native FP8 backend: ``exclude_layers`` wins over
``filter_fn``, regional scoping follows ``repeated_blocks``, optional accuracy
observers are captured before replacement and attached afterwards, and a run
that quantizes nothing raises instead of returning an unquantized module.
"""

from typing import Dict, List, Optional, Set

import torch

from ..config import QuantizeConfig
from ..observers import QuantizeObserver
from .linear import MXFP4Linear
from .tensor import MXFP4_BLOCK_SIZE, resolve_mxfp4_quantize_device
from ...logger import init_logger

logger = init_logger(__name__)


def _replace_module(root: torch.nn.Module, local_name: str, replacement: torch.nn.Module) -> None:
  parent_path, _, attr = local_name.rpartition(".")
  parent = root.get_submodule(parent_path) if parent_path else root
  setattr(parent, attr, replacement)


def quantize_mxfp4(
  module: torch.nn.Module,
  quantize_config: QuantizeConfig,
  **kwargs,
) -> torch.nn.Module:
  """Replace ``torch.nn.Linear`` layers inside ``module`` with :class:`MXFP4Linear`.

  :param module: Model whose Linear layers are packed to MXFP4 in place.
  :param quantize_config: Resolved quantization config; ``regional_quantize``,
    ``repeated_blocks``, ``exclude_layers``, ``filter_fn``, ``observer``,
    ``mxfp4_block_size`` and ``mxfp4_quantize_device`` are honored.
  :returns: The same module, with eligible Linear layers replaced.
  :raises ValueError: If no Linear layer can be packed.
  """
  from ...utils import check_quantized

  if not quantize_config.is_mxfp4():
    raise ValueError(f"cache-dit MXFP4 backend does not support quant_type "
                     f"{quantize_config.quant_type!r}; use 'mxfp4_weight_only'.")
  if check_quantized(module):
    logger.warning("Module %s is already quantized (type: %s), skip MXFP4 quantization.",
                   module.__class__.__name__, getattr(module, "_quantize_type", None))
    return module

  observer = QuantizeObserver(getattr(quantize_config, "observer", None))
  observer_refs = observer.capture_references(module)

  block_size = int(
    getattr(quantize_config, "mxfp4_block_size", MXFP4_BLOCK_SIZE) or MXFP4_BLOCK_SIZE)
  requested_device = getattr(quantize_config, "mxfp4_quantize_device", "auto")
  excluded = list(quantize_config.exclude_layers or [])
  filter_fn = quantize_config.filter_fn
  # Same precedence as the FP8 backend: the model's declaration wins.
  repeated_blocks: Optional[List[str]] = (list(
    getattr(module, "_repeated_blocks", None) or getattr(quantize_config, "repeated_blocks", None)
    or []))

  quantized_layers: List[str] = []
  skipped_layers: Dict[str, str] = {}
  packing_devices: Set[str] = set()
  seen_linears: Set[int] = set()
  num_linear = 0

  def quantize_region(region: torch.nn.Module, prefix: str) -> None:
    nonlocal num_linear
    for local_name, child in list(region.named_modules()):
      if not isinstance(child, torch.nn.Linear):
        continue
      if id(child) in seen_linears:
        continue
      seen_linears.add(id(child))
      num_linear += 1
      full_name = f"{prefix}.{local_name}" if prefix and local_name else (local_name or prefix)
      if any(pattern in full_name for pattern in excluded):
        skipped_layers[full_name] = "exclude_layers"
        continue
      if filter_fn is not None and not filter_fn(child, full_name):
        skipped_layers[full_name] = "filter_fn"
        continue
      weight = child.weight
      if weight.dim() != 2:
        skipped_layers[full_name] = "weight is not 2D"
        continue
      if type(weight).__name__ == "DTensor":
        skipped_layers[full_name] = "DTensor/TP weights are not supported by cache-dit MXFP4"
        continue
      # The E8M0 scale group is block_size elements wide, so K must be a multiple
      # of it, and two codes share a byte so K must also be even.
      if child.in_features % block_size != 0 or child.in_features % 2 != 0:
        skipped_layers[full_name] = (
          f"in_features={child.in_features} is not a multiple of block_size={block_size} / even")
        continue
      # Pack on the accelerator when one is available and keep the packed buffers
      # on the weight's original device, so quantization never moves the module.
      quantize_device = resolve_mxfp4_quantize_device(child.weight, requested_device)
      packing_devices.add(str(quantize_device))
      replacement = MXFP4Linear(child, block_size=block_size, quantize_device=quantize_device)
      _replace_module(region, local_name, replacement)
      quantized_layers.append(full_name)

  if quantize_config.regional_quantize and repeated_blocks:
    containers = [(name, submodule) for name, submodule in module.named_modules()
                  if submodule.__class__.__name__ in repeated_blocks]
    if not containers:
      raise ValueError(f"Regional MXFP4 quantization failed because {repeated_blocks} "
                       "classes are not found in the module.")
    for name, block in containers:
      quantize_region(block, name)
  else:
    if quantize_config.regional_quantize and not repeated_blocks:
      logger.warning(
        "%s declares no repeated blocks; MXFP4 quantization will cover the "
        "whole module.", module.__class__.__name__)
    quantize_region(module, "")

  if not quantized_layers:
    reasons = "; ".join(f"{name}: {reason}" for name, reason in list(skipped_layers.items())[:5])
    raise ValueError("cache-dit MXFP4 did not quantize any linear layer. Check repeated_blocks/"
                     f"exclude_layers/filter_fn/mxfp4_block_size. First skip reasons: "
                     f"{reasons or 'no linear found'}.")

  logger.info(
    "MXFP4 quantization done for %s: %d/%d Linear layers quantized (block_size=%d), "
    "%d skipped, packing on %s.", module.__class__.__name__, len(quantized_layers), num_linear,
    block_size, len(skipped_layers), ", ".join(sorted(packing_devices)))
  if quantize_config.verbose and skipped_layers:
    logger.info("MXFP4 skipped layers: %s", list(skipped_layers.items())[:32])

  module._is_quantized = True
  module._quantize_config = quantize_config
  module._quantize_type = quantize_config.quant_type
  module._exclude_layers = excluded
  module._mxfp4_quantized_layers = tuple(quantized_layers)
  module._mxfp4_skipped_layers = dict(skipped_layers)
  if observer_refs:
    observer.attach(module, observer_refs)
  return module
