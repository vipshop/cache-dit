"""Online FP8 quantization workflow for cache-dit's native backend.

This module owns *where* quantization is applied; :mod:`fp8.linear` owns the
runtime W8A8 math. Supported public types map to :class:`QW8A8Linear` modes as
``float8_per_row -> per_row``, ``float8_per_tensor -> per_tensor`` and
``float8_per_block -> per_block``. Weights are quantized once during module
replacement; activations are quantized dynamically on every forward.

The flow is: validate config/device -> capture optional observer references ->
select repeated-block regions -> resolve each layer's precision mode -> replace
eligible ``nn.Linear`` modules -> attach observers and metadata. Selection honors
``exclude_layers``, ``filter_fn`` and the longest matching ``precision_plan``
pattern. Non-BF16 and DTensor/TP weights are skipped with a reason. Shape
misalignment is not a skip reason: ``QW8A8Linear`` pads before quantization and
depads after GEMM. This is an online-only backend, so calibration, serialized FP8
checkpoints and loading quantized weights are intentionally rejected.
"""

import torch
from torch import nn

from ...logger import init_logger
from ..config import QuantizeConfig
from ..observers import QuantizeObserver
from .linear import QW8A8Linear

logger = init_logger(__name__)

_FP8_QUANT_TYPES = frozenset({"float8_per_row", "float8_per_tensor", "float8_per_block"})


def _mode_supported_reason(linear: nn.Linear, mode: str) -> str | None:
  """Return a hard incompatibility reason, or None when replacement is allowed.

  This check intentionally does not reject K/N misalignment or select a fallback
  mode. ``QW8A8Linear.from_linear`` zero-pads before quantization, and
  ``fp8_gemm`` depads output, so all ordinary Linear shapes remain on the
  requested recipe. Only structural/semantic incompatibilities skip a layer:
  non-2D weights, DTensor/TP weights and non-BF16 weights. ``mode`` is retained
  in this helper's interface for backend symmetry; it does not alter these hard
  checks.
  """

  weight = getattr(linear, "weight", None)
  if weight is None or weight.dim() != 2:
    return "weight is not 2D"
  if type(weight).__name__ == "DTensor":
    return "DTensor/TP weights are not supported by cache-dit FP8"
  if weight.dtype != torch.bfloat16:
    return f"weight dtype {weight.dtype} != bfloat16"
  return None


def _resolve_layer_mode(full_name: str, base_mode: str,
                        precision_plan: dict[str, str] | None) -> str:
  """Resolve the effective granularity for one layer.

  :param full_name: Fully qualified module name.
  :param base_mode: Config-level granularity.
  :param precision_plan: Optional per-layer overrides.
  :returns: Effective granularity.
  """

  if not precision_plan:
    return base_mode
  matches = [(pattern, mode) for pattern, mode in precision_plan.items() if pattern in full_name]
  if not matches:
    return base_mode
  return max(matches, key=lambda item: len(item[0]))[1]


def _replace_module(root: nn.Module, local_name: str, replacement: nn.Module) -> None:
  """Replace a descendant module in place.

  :param root: Region root.
  :param local_name: Dot-separated descendant name.
  :param replacement: New module.
  """

  parent_name, _, child_name = local_name.rpartition(".")
  parent = root.get_submodule(parent_name) if parent_name else root
  setattr(parent, child_name, replacement)


def quantize_fp8(module: nn.Module, quantize_config: QuantizeConfig) -> nn.Module:
  """Replace eligible Linear layers with online W8A8 modules, in place.

  With ``regional_quantize=True`` and repeated-block metadata, only blocks whose
  class name matches ``repeated_blocks`` are traversed. Otherwise the whole
  module is traversed. Within each region, exclusion/filter rules run first,
  then the longest matching precision-plan pattern chooses the FP8 granularity.
  A shared Linear encountered through multiple names is replaced only once.

  Each replacement pads and quantizes the BF16 weight immediately. Activation
  quantization remains dynamic: no activation statistics are collected here.
  Shape alignment does not cause a skip because ``QW8A8Linear`` handles it.
  DTensor/TP weights and non-BF16 weights are skipped with diagnostics; if every
  candidate is excluded or unsupported, this function raises instead of
  returning a deceptively unquantized module.

  :param module: Transformer or other module to modify. The function is not a
    functional/pure transformation: descendants are replaced in place.
  :param quantize_config: Validated config selecting one native FP8 type.
    ``calibrate_fn`` and ``serialize_to`` must be absent; this backend is
    online-only. The current device must satisfy the backend architecture gate.
  :returns: The same ``module`` object with ``QW8A8Linear`` descendants and
    quantization metadata. Optional observers are attached after replacement,
    using references captured before mutation.
  """

  if not isinstance(module, nn.Module):
    raise TypeError(f"quantize_fp8 expects nn.Module, got {type(module)}.")
  if quantize_config.quant_type not in _FP8_QUANT_TYPES:
    raise ValueError(f"cache-dit FP8 backend does not support quant_type "
                     f"{quantize_config.quant_type!r}; expected one of {sorted(_FP8_QUANT_TYPES)}.")
  if quantize_config.calibrate_fn is not None or quantize_config.serialize_to is not None:
    raise ValueError("cache-dit FP8 is online-only; calibrate_fn and serialize_to are not "
                     "supported.")
  if not torch.cuda.is_available():
    raise RuntimeError("cache-dit FP8 quantization requires CUDA.")
  capability = torch.cuda.get_device_capability()
  if capability < (10, 0):
    raise RuntimeError(f"cache-dit FP8 kernels require sm100 or newer, got sm{capability[0]}"
                       f"{capability[1]}. Use --quantize-backend torchao on this device.")

  from ...utils import check_quantized

  if check_quantized(module):
    logger.warning("Module %s is already quantized, skipping cache-dit FP8.",
                   module.__class__.__name__)
    return module

  observer = QuantizeObserver(getattr(quantize_config, "observer", None))
  observer_refs = observer.capture_references(module)

  repeated_blocks = getattr(module, "_repeated_blocks", None) or quantize_config.repeated_blocks
  plan = dict(quantize_config.precision_plan or {})
  for quant_type in plan.values():
    if quant_type not in _FP8_QUANT_TYPES:
      raise ValueError(f"precision_plan contains {quant_type!r}, but the cache-dit FP8 backend "
                       f"only supports {sorted(_FP8_QUANT_TYPES)}.")
  excluded = list(quantize_config.exclude_layers or [])
  filter_fn = quantize_config.filter_fn

  quantized_layers: list[str] = []
  skipped_layers: dict[str, str] = {}
  seen_linears: set[int] = set()
  num_linear = 0

  def quantize_region(region: nn.Module, prefix: str) -> None:
    nonlocal num_linear
    candidates = list(region.named_modules())
    for local_name, child in candidates:
      if not isinstance(child, nn.Linear):
        continue
      if id(child) in seen_linears:
        continue
      seen_linears.add(id(child))
      num_linear += 1
      full_name = f"{prefix}.{local_name}" if prefix and local_name else (local_name or prefix)
      if any(pattern in full_name for pattern in excluded):
        continue
      if filter_fn is not None and not filter_fn(child, full_name):
        continue
      quant_type = _resolve_layer_mode(full_name, quantize_config.quant_type, plan)
      mode = quant_type.removeprefix("float8_")
      reason = _mode_supported_reason(child, mode)
      if reason is not None:
        skipped_layers[full_name] = reason
        logger.warning("[FP8] skipping %s: %s", full_name, reason)
        continue
      replacement = QW8A8Linear.from_linear(child, mode)
      _replace_module(region, local_name, replacement)
      quantized_layers.append(full_name)

  if quantize_config.regional_quantize:
    if not repeated_blocks:
      logger.warning(
        "%s declares no repeated blocks; cache-dit FP8 will quantize the whole "
        "module. Set repeated_blocks explicitly to keep regional scoping.",
        module.__class__.__name__)
      quantize_region(module, "")
    else:
      blocks = [(name, submodule) for name, submodule in module.named_modules()
                if submodule.__class__.__name__ in repeated_blocks]
      if not blocks:
        raise ValueError(f"Regional cache-dit FP8 found no blocks matching {repeated_blocks}.")
      for name, block in blocks:
        quantize_region(block, name)
  else:
    quantize_region(module, "")

  logger.info("[FP8] Quantized %d/%d linear layers with %s (cache_dit backend); "
              "%d skipped.", len(quantized_layers), num_linear, quantize_config.quant_type,
              len(skipped_layers))
  if not quantized_layers:
    reasons = "; ".join(f"{name}: {reason}" for name, reason in list(skipped_layers.items())[:5])
    raise ValueError(
      "cache-dit FP8 did not quantize any linear layer. Check repeated_blocks/"
      f"exclude_layers/precision_plan. First skip reasons: {reasons or 'no linear found'}.")

  module._is_quantized = True
  module._quantize_config = quantize_config
  module._quantize_type = quantize_config.quant_type
  module._exclude_layers = excluded
  module._fp8_quantized_layers = tuple(quantized_layers)
  module._fp8_skipped_layers = dict(skipped_layers)
  if observer_refs:
    observer.attach(module, observer_refs)
  return module


__all__ = ["QW8A8Linear", "quantize_fp8"]
