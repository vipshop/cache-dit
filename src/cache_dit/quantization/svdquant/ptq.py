from __future__ import annotations

import dataclasses
import gc
import inspect
import json
import os
import time
from typing import Any

import torch
from torch import nn

from ...offload import LayerwiseOffloadHandle
from ...offload import _apply_layerwise_offload
from ...offload import _find_offload_related_hf_hook
from ...offload import get_layerwise_offload_handles
from ...logger import init_logger
from ...utils import check_quantized
from ..config import QuantizeConfig
from ..config import _parse_svdq_quant_type
from ..config import _resolve_svdq_kwargs
from .linear import SVDQW4A4Linear
from .quantizer import _ActivationSpanAccumulator
from .quantizer import _apply_few_shot_relaxation
from .quantizer import _normalize_dtype
from .quantizer import _quantize_linear_svdq_w4a4_from_activation_span
from .quantizer import _quantize_linear_svdq_w4a4_from_smooth_scale
from .quantizer import _resolve_math_dtype
from .quantizer import validate_svdq_linear_geometry

logger = init_logger(__name__)

_SVDQ_METADATA_KEY = "cache_dit_svdq_ptq"
_SVDQ_FORMAT_VERSION = 2
_SVDQ_QUANT_CONFIG_FORMAT = "cache_dit_svdq_quant_config"
_SVDQ_QUANT_CONFIG_VERSION = 2
_SVDQ_QUANT_CONFIG_FILENAME = "quant_config.json"
_ROOT_LAYER_NAME = "__root__"
_RUNTIME_LAYERWISE_OFFLOAD_HANDLE_ATTR = "_svdq_runtime_layerwise_offload_handle"


@dataclasses.dataclass(frozen=True)
class _ResolvedSVDQLoadSource:
  checkpoint_path: str
  quant_config_snapshot: dict[str, Any] | None = None


def _import_safetensors():
  try:
    from safetensors import safe_open
    from safetensors.torch import save_file
  except ImportError as exc:
    raise ImportError(
      "SVDQ PTQ serialization/loading requires `safetensors`. "
      "Install with `pip install cache-dit[quantization]` or `pip install safetensors`.") from exc
  return safe_open, save_file


def _serialize_layer_name(layer_name: str) -> str:
  return _ROOT_LAYER_NAME if layer_name == "" else layer_name


def _deserialize_layer_name(layer_name: str) -> str:
  return "" if layer_name == _ROOT_LAYER_NAME else layer_name


def _get_named_submodule(module: nn.Module, module_name: str) -> nn.Module:
  return module if module_name == "" else module.get_submodule(module_name)


def _get_parent_and_child(module: nn.Module, module_name: str) -> tuple[nn.Module, str]:
  if not module_name:
    raise ValueError("The root module does not have a parent/child path.")
  if "." not in module_name:
    return module, module_name
  parent_name, child_name = module_name.rsplit(".", 1)
  return module.get_submodule(parent_name), child_name


def _call_with_supported_kwargs(callback: Any, **kwargs) -> Any:
  try:
    signature = inspect.signature(callback)
  except (TypeError, ValueError):
    return callback(**kwargs)

  if any(parameter.kind == inspect.Parameter.VAR_KEYWORD
         for parameter in signature.parameters.values()):
    return callback(**kwargs)

  accepted_kwargs = {name: value for name, value in kwargs.items() if name in signature.parameters}
  return callback(**accepted_kwargs)


def _infer_module_execution_device(module: nn.Module) -> torch.device:
  for submodule in module.modules():
    hook = getattr(submodule, "_hf_hook", None)
    execution_device = getattr(hook, "execution_device", None)
    if execution_device is not None:
      return torch.device(execution_device)

  try:
    parameter = next(module.parameters())
    if parameter.device.type != "meta":
      return parameter.device
  except StopIteration:
    pass

  try:
    buffer = next(module.buffers())
    if buffer.device.type != "meta":
      return buffer.device
  except StopIteration:
    pass

  return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_svdq_quantize_device(module: nn.Module, svdq_kwargs: dict[str, Any]) -> torch.device:
  requested = str(svdq_kwargs.get("quantize_device", "auto")).lower()
  inferred = _infer_module_execution_device(module)

  if requested == "auto":
    return inferred

  if requested == "cpu":
    return torch.device("cpu")

  if requested != "cuda":
    raise ValueError(f"Unsupported SVDQ quantize_device: {requested!r}.")

  if not torch.cuda.is_available():
    raise RuntimeError(
      "svdq_kwargs['quantize_device'] = 'cuda' was requested, but CUDA is unavailable. "
      "Use 'auto' or 'cpu' instead.")

  if inferred.type == "cuda":
    return inferred
  return torch.device("cuda", torch.cuda.current_device())


def _resolve_quantized_layer_storage_device(
  svdq_kwargs: dict[str, Any],
  quantize_device: torch.device,
) -> torch.device:
  if bool(svdq_kwargs.get("offload_quantized_layers_to_cpu", False)):
    return torch.device("cpu")
  return quantize_device


def _maybe_collect_svdq_garbage(device: torch.device | None = None) -> None:
  gc.collect()
  if device is None:
    if torch.cuda.is_available():
      torch.cuda.empty_cache()
    return
  if device.type == "cuda" and torch.cuda.is_available():
    torch.cuda.empty_cache()


def _maybe_enable_layerwise_collection_offload(
  module: nn.Module,
  *,
  onload_device: torch.device,
  async_transfer: bool = False,
  transfer_buckets: int = 1,
  prefetch_limit: bool = False,
  max_copy_streams: int | None = None,
  max_inflight_prefetch_bytes: int | None = None,
  persistent_buckets: int = 0,
  persistent_bins: int = 1,
) -> LayerwiseOffloadHandle | None:
  """Enable temporary collection-time layerwise offload on the root module.

  Collection-time offload intentionally applies to all parameterized leaf submodules in the root
  module, not only to the candidate quantized linear layers. Otherwise non-quantized parameterized
  layers, such as norms or embeddings, would stay on CPU during calibration / few-shot collection
  and slow the observed forward path.

  :param module: Root module whose collection-time forward path should run through layerwise
    onload/offload.
  :param onload_device: Execution device used for the temporary collection-time offload handle.
  :param async_transfer: Whether collection-time offload should use async transfer.
  :param transfer_buckets: Prefetch budget used by async collection-time offload.
  :param prefetch_limit: Whether collection-time offload should enable the conservative future-
    prefetch target-count limit.
  :param max_copy_streams: Maximum number of async copy streams used by collection-time offload.
  :param max_inflight_prefetch_bytes: Maximum total CUDA residency budget, in bytes, used by future-
    target collection-time prefetch.
  :param persistent_buckets: How many selected targets should remain resident on the execution
    device during collection-time offload.
  :param persistent_bins: How many evenly distributed bins should be used when placing the
    persistent collection-time targets.
  :returns: The temporary collection-time layerwise offload handle, if enabled.
  """

  if onload_device.type != "cuda":
    return None
  existing_layerwise_handles = get_layerwise_offload_handles(module)
  if existing_layerwise_handles:
    logger.warning(
      "Skipping cache-dit layerwise CPU offload for SVDQ activation collection on %s because "
      "%d cache-dit layerwise offload handle(s) are already registered on the same module.",
      module.__class__.__name__,
      len(existing_layerwise_handles),
    )
    return None
  existing_offload_hook = _find_offload_related_hf_hook(module)
  if existing_offload_hook is not None:
    hook_module_name, hook = existing_offload_hook
    logger.warning(
      "Skipping cache-dit layerwise CPU offload for SVDQ activation collection on %s because "
      "an existing accelerate offload hook %s is already registered on %s.",
      module.__class__.__name__,
      type(hook).__name__,
      hook_module_name or _ROOT_LAYER_NAME,
    )
    return None
  if _infer_module_execution_device(module).type == "cuda":
    return None
  return _apply_layerwise_offload(
    module,
    onload_device=onload_device,
    offload_device=torch.device("cpu"),
    async_transfer=async_transfer,
    transfer_buckets=transfer_buckets,
    prefetch_limit=prefetch_limit,
    max_copy_streams=max_copy_streams,
    max_inflight_prefetch_bytes=max_inflight_prefetch_bytes,
    persistent_buckets=persistent_buckets,
    persistent_bins=persistent_bins,
  )


def _log_boxed_summary(lines: list[str]) -> None:
  if not lines:
    return
  max_length = max(len(line) for line in lines) + 2
  logger.info("-" * max_length)
  logger.info("\n".join(line.ljust(max_length) + "|" for line in lines))
  logger.info("-" * max_length)


def _log_skipped_reasons(skipped_reasons: list[str], *, verbose: bool) -> None:
  if not verbose or not skipped_reasons:
    return

  skipped_reasons_counter: dict[str, int] = {}
  for reason in skipped_reasons:
    skipped_reasons_counter[reason] = skipped_reasons_counter.get(reason, 0) + 1

  rows: list[tuple[str, str, int]] = []
  name_width = 0
  reason_width = 0
  for entry, count in skipped_reasons_counter.items():
    layer_name, reason = entry.split(" -> ", 1)
    rows.append((layer_name, reason, count))
    name_width = max(name_width, len(layer_name))
    reason_width = max(reason_width, len(reason))

  detail_lines = [
    f"{layer_name.ljust(name_width)}: {reason.ljust(reason_width)}: {count:<4} layers"
    for layer_name, reason, count in rows
  ]
  _log_boxed_summary(detail_lines)


def _get_checkpoint_dir(checkpoint_path: str) -> str:
  return os.path.dirname(os.path.abspath(checkpoint_path))


def _get_quant_config_path(checkpoint_path: str) -> str:
  return os.path.join(_get_checkpoint_dir(checkpoint_path), _SVDQ_QUANT_CONFIG_FILENAME)


def _attach_quantization_metadata(
  module: nn.Module,
  *,
  quant_type: str,
  quantized_layer_names: list[str],
  exclude_layers: list[str],
  svdq_kwargs: dict[str, Any],
  checkpoint_path: str | None,
  quantize_config: QuantizeConfig | None = None,
) -> None:
  module._is_quantized = True
  module._quantize_type = quant_type
  module._exclude_layers = list(exclude_layers)
  module._svdq_quantized_layers = tuple(quantized_layer_names)
  module._svdq_kwargs = dict(svdq_kwargs)
  module._svdq_checkpoint_path = checkpoint_path
  if quantize_config is not None:
    module._quantize_config = quantize_config


def _register_post_quantize_callback(
  module: nn.Module,
  callback: Any,
  *,
  callback_name: str | None = None,
) -> None:
  """Register a root-module callback for work that must run after few-shot materialization.

  Callbacks are attached to the owner/root module being quantized, not to the individual linear
  submodules that get replaced. Follow-up work such as compilation is generally a root-module
  concern, so this registry intentionally lives at that level.
  """

  callbacks = getattr(module, "_svdq_post_quantize_callbacks", None)
  if callbacks is None:
    callbacks = []
    module._svdq_post_quantize_callbacks = callbacks

  if callback_name is not None:
    callback_names = getattr(module, "_svdq_post_quantize_callback_names", None)
    if callback_names is None:
      callback_names = set()
      module._svdq_post_quantize_callback_names = callback_names
    if callback_name in callback_names:
      return
    callback_names.add(callback_name)

  callbacks.append(callback)


def _pop_post_quantize_callbacks(module: nn.Module) -> list[Any]:
  callbacks = list(getattr(module, "_svdq_post_quantize_callbacks", []))
  if hasattr(module, "_svdq_post_quantize_callbacks"):
    del module._svdq_post_quantize_callbacks
  if hasattr(module, "_svdq_post_quantize_callback_names"):
    del module._svdq_post_quantize_callback_names
  return callbacks


def _run_post_quantize_callbacks(
  module: nn.Module,
  *,
  few_shot_auto_compile: bool,
) -> None:
  """Run root-level post-quantize callbacks or use a generic in-place compile fallback."""

  callbacks = _pop_post_quantize_callbacks(module)
  if not callbacks and few_shot_auto_compile:
    try:
      if hasattr(module, "compile_repeated_blocks") and callable(module.compile_repeated_blocks):
        logger.info("Few-shot runtime quantization completed; compiling repeated blocks in-place.")
        module.compile_repeated_blocks()
      elif hasattr(module, "compile") and callable(module.compile):
        logger.info("Few-shot runtime quantization completed; compiling root module in-place via "
                    "nn.Module.compile().")
        module.compile()
      else:
        logger.warning(
          "few_shot_auto_compile was requested, but no owner callback was registered and the "
          "module does not support compile_repeated_blocks() or nn.Module.compile(); skipping "
          "automatic compile.")
    except Exception as exc:
      logger.warning(
        "few_shot_auto_compile fallback failed with %s: %s",
        type(exc).__name__,
        exc,
      )
    return

  for callback in callbacks:
    try:
      callback(module)
    except Exception as exc:
      logger.warning("Deferred post-quantize callback failed with %s: %s", type(exc).__name__, exc)


def _clear_pending_quantization_state(module: nn.Module) -> None:
  for attr_name in (
      "_svdq_pending_quantization",
      "_svdq_few_shot_controller",
      "_svdq_cleanup_pending_quantization",
  ):
    if hasattr(module, attr_name):
      delattr(module, attr_name)


def _remove_runtime_layerwise_offload_handle(module: nn.Module) -> bool:
  handle = getattr(module, _RUNTIME_LAYERWISE_OFFLOAD_HANDLE_ATTR, None)
  if handle is None:
    return False
  if isinstance(handle, LayerwiseOffloadHandle):
    handle.remove()
  delattr(module, _RUNTIME_LAYERWISE_OFFLOAD_HANDLE_ATTR)
  return True


def _maybe_enable_layerwise_runtime_offload(
  module: nn.Module,
  *,
  quantized_layer_names: list[str],
  onload_device: torch.device,
  svdq_kwargs: dict[str, Any],
) -> LayerwiseOffloadHandle | None:
  """Enable temporary low-memory runtime offload after few-shot quantization materializes.

  The temporary handle is intentionally broader than the quantized layer list: it applies
  layerwise offload to all parameterized leaf submodules in the root module so non-quantized
  parameterized layers do not stay on CPU for the remainder of the current pipeline/module call.

  :param module: Root module whose current call should finish on the low-memory path.
  :param quantized_layer_names: Names of the SVDQ layers that were just materialized.
  :param onload_device: Execution device used by the temporary runtime offload handle.
  :param svdq_kwargs: Resolved SVDQ kwargs for the current quantization flow.
  :returns: The temporary runtime layerwise offload handle, if enabled.
  """

  if onload_device.type != "cuda":
    return None
  if not bool(svdq_kwargs.get("layerwise_offload", False)):
    return None
  if not bool(svdq_kwargs.get("offload_quantized_layers_to_cpu", False)):
    return None

  _remove_runtime_layerwise_offload_handle(module)
  existing_layerwise_handles = get_layerwise_offload_handles(module)
  if existing_layerwise_handles:
    logger.warning(
      "Skipping temporary runtime layerwise offload on %s because %d cache-dit layerwise "
      "offload handle(s) are already registered on the same module.",
      module.__class__.__name__,
      len(existing_layerwise_handles),
    )
    return None
  existing_offload_hook = _find_offload_related_hf_hook(module)
  if existing_offload_hook is not None:
    hook_module_name, hook = existing_offload_hook
    logger.warning(
      "Skipping temporary runtime layerwise offload on %s because an existing accelerate "
      "offload hook %s is already registered on %s.",
      module.__class__.__name__,
      type(hook).__name__,
      hook_module_name or _ROOT_LAYER_NAME,
    )
    return None

  handle = _apply_layerwise_offload(
    module,
    onload_device=onload_device,
    offload_device=torch.device("cpu"),
    async_transfer=bool(svdq_kwargs.get("async_transfer", False)),
    transfer_buckets=int(svdq_kwargs.get("transfer_buckets", 1)),
    prefetch_limit=bool(svdq_kwargs.get("prefetch_limit", False)),
    max_copy_streams=svdq_kwargs.get("max_copy_streams"),
    max_inflight_prefetch_bytes=svdq_kwargs.get("max_inflight_prefetch_bytes"),
    persistent_buckets=int(svdq_kwargs.get("persistent_buckets", 0)),
    persistent_bins=int(svdq_kwargs.get("persistent_bins", 1)),
  )
  setattr(module, _RUNTIME_LAYERWISE_OFFLOAD_HANDLE_ATTR, handle)
  logger.info(
    "Enabled temporary layerwise offload for %d parameterized leaf submodules on %s after "
    "\nquantizing %d SVDQ layers so the current pipeline/module call can finish before the final "
    "move to %s.",
    len(handle.targets),
    module.__class__.__name__,
    len(quantized_layer_names),
    onload_device,
  )
  return handle


@dataclasses.dataclass
class SVDQPTQContext:
  """Resolved plan for a module-level SVDQ PTQ run.

  The context snapshots the quantization scope, discovered candidate `nn.Linear` layers, skip
  reasons, and resolved SVDQ kwargs derived from the `QuantizeConfig`. It is shared between
  calibration, replacement, serialization, and summary logging.
  """

  root_module: nn.Module
  quantize_config: QuantizeConfig
  rank: int
  precision: str = "int4"
  regional_quantize: bool = True
  repeated_blocks: list[str] | None = None
  exclude_layers: list[str] = dataclasses.field(default_factory=list)
  filter_fn: Any = None
  svdq_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
  verbose: bool = False
  linear_layer_names: list[str] = dataclasses.field(default_factory=list)
  candidate_layer_names: list[str] = dataclasses.field(default_factory=list)
  skipped_reasons: list[str] = dataclasses.field(default_factory=list)

  @property
  def quantized_region(self) -> str:
    if self.regional_quantize and self.repeated_blocks is not None:
      return str(self.repeated_blocks)
    return self.root_module.__class__.__name__

  @property
  def workflow_name(self) -> str:
    return "SVDQ DQ" if self.quantize_config.is_svdq_dq() else "SVDQ PTQ"

  @classmethod
  def from_config(cls, root_module: nn.Module, quantize_config: QuantizeConfig) -> "SVDQPTQContext":
    """Resolve a PTQ execution context from a root module and config.

    This gathers the repeated-block scope, exclusion rules, resolved SVDQ kwargs, and the final list
    of candidate linear layers that satisfy the migrated W4A4 geometry constraints.

    :param root_module: Root module that will be searched for quantizable linear layers.
    :param quantize_config: Structured quantization configuration.
    :returns: A resolved PTQ context describing scope, candidates, and SVDQ settings.
    """

    repeated_blocks = getattr(
      root_module,
      "_repeated_blocks",
      quantize_config.repeated_blocks if quantize_config.repeated_blocks else None,
    )
    regional_quantize = quantize_config.regional_quantize and repeated_blocks is not None
    context = cls(
      root_module=root_module,
      quantize_config=quantize_config,
      rank=quantize_config.get_svdq_rank(),
      regional_quantize=regional_quantize,
      repeated_blocks=list(repeated_blocks) if repeated_blocks is not None else None,
      exclude_layers=list(quantize_config.exclude_layers or []),
      filter_fn=quantize_config.filter_fn,
      svdq_kwargs=quantize_config.get_svdq_kwargs(),
      verbose=quantize_config.verbose,
    )
    context.candidate_layer_names = context._collect_candidate_layer_names()
    return context

  def _record_skip(self, layer_name: str, reason: str) -> None:
    self.skipped_reasons.append(f"{layer_name or _ROOT_LAYER_NAME} -> {reason}")

  def _is_excluded(self, layer_name: str) -> bool:
    for exclude_name in self.exclude_layers:
      if exclude_name in layer_name:
        return True
    return False

  def _iter_modules_in_scope(self):
    if not self.regional_quantize or not self.repeated_blocks:
      yield from self.root_module.named_modules()
      return

    region_found = False
    seen_names: set[str] = set()
    for block_name, block in self.root_module.named_modules():
      if not block_name or block.__class__.__name__ not in self.repeated_blocks:
        continue
      region_found = True
      for local_name, submodule in block.named_modules():
        full_name = block_name if local_name == "" else f"{block_name}.{local_name}"
        if full_name in seen_names:
          continue
        seen_names.add(full_name)
        yield full_name, submodule

    if not region_found:
      raise ValueError(
        f"Regional SVDQ PTQ failed because repeated block classes {self.repeated_blocks} "
        "were not found in the target module.")

  def _collect_candidate_layer_names(self) -> list[str]:
    candidate_layer_names: list[str] = []
    seen_names: set[str] = set()
    for layer_name, submodule in self._iter_modules_in_scope():
      if layer_name in seen_names or not isinstance(submodule, nn.Linear):
        continue
      seen_names.add(layer_name)
      self.linear_layer_names.append(layer_name)

      if self.filter_fn is not None:
        if not bool(
            _call_with_supported_kwargs(
              self.filter_fn,
              m=submodule,
              module=submodule,
              name=layer_name,
            )):
          self._record_skip(layer_name, "filtered by filter_fn")
          continue
      elif self._is_excluded(layer_name):
        self._record_skip(layer_name, "excluded by exclude_layers")
        continue

      try:
        validate_svdq_linear_geometry(
          submodule.in_features,
          submodule.out_features,
          rank=self.rank,
          precision=self.precision,
        )
      except (NotImplementedError, ValueError) as exc:
        self._record_skip(layer_name, str(exc))
        continue

      candidate_layer_names.append(layer_name)

    if not candidate_layer_names:
      raise ValueError("SVDQ PTQ found no quantizable Linear layers in the target module. "
                       f"Skipped layers: {self.skipped_reasons}.")
    return candidate_layer_names

  def quantize_summary_lines(
    self,
    *,
    observed_layer_names: list[str] | None,
    quantized_layer_names: list[str],
    serialize_to: str | None,
  ) -> list[str]:
    skipped_total = max(len(self.linear_layer_names) - len(quantized_layer_names), 0)
    lines = [
      f"SVDQuant    Region: {self.quantized_region}",
      f"SVDQuant      Type: {self.quantize_config.quant_type}",  # dq or no-dq
      f"SVDQuant      Rank: {self.rank}",
      f"Quantized   Layers: {len(quantized_layer_names)} / {len(self.candidate_layer_names)}",
      f"Skipped     Layers: {skipped_total}",
      f"Linear      Layers: {len(self.linear_layer_names)}",
    ]
    if observed_layer_names is not None:
      lines.insert(
        3,
        f"Observed    Layers: {len(observed_layer_names)} / {len(self.candidate_layer_names)}",
      )
    if self.verbose:
      lines.append(f"SVDQuant    Kwargs: {self.svdq_kwargs}")
      lines.append(f"Skipped   Patterns: {self.exclude_layers}")
      if serialize_to is not None:
        lines.append(f"Checkpoint      Path: {serialize_to}")
    return lines


class SVDQPTQCalibrator:
  """Collect per-layer activation spans for module-level SVDQ PTQ.

  The calibrator registers forward-pre-hooks on candidate float `nn.Linear`
  layers, reduces each observed activation batch to a per-channel span vector,
  and exposes the finalized spans to `quantize_svdq_ptq`.
  """

  def __init__(self, context: SVDQPTQContext) -> None:
    self.context = context
    self.activation_spans: dict[str, torch.Tensor] = {}
    self._handles: list[Any] = []
    self._accumulators: dict[str, _ActivationSpanAccumulator] = {}
    self._offload_handle: LayerwiseOffloadHandle | None = None

  def _make_hook(self, layer_name: str):
    accumulator = self._accumulators[layer_name]

    def hook(module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
      if not args or not isinstance(args[0], torch.Tensor):
        raise TypeError(
          f"SVDQ PTQ expected tensor inputs for layer {layer_name or _ROOT_LAYER_NAME}.")
      if not isinstance(module, nn.Linear):
        raise TypeError(f"Expected nn.Linear during calibration, got {type(module)}.")

      activation = args[0]
      if activation.shape[-1] != module.in_features:
        raise ValueError(
          f"Expected activation last dim {module.in_features} for layer {layer_name}, "
          f"got {activation.shape[-1]}.")

      accumulator.add_tensor(activation.detach().reshape(-1, module.in_features))

    return hook

  def register(self, *, observation_device: torch.device | None = None) -> None:
    """Register calibration hooks on all candidate float linear layers."""

    calibrate_precision = self.context.svdq_kwargs["calibrate_precision"]
    flush_sample_count = self.context.svdq_kwargs["activation_buffer_flush_sample_count"]
    flush_cpu_bytes = self.context.svdq_kwargs["activation_buffer_flush_cpu_bytes"]

    if (observation_device is not None
        and bool(self.context.svdq_kwargs.get("layerwise_offload", False))):
      self._offload_handle = _maybe_enable_layerwise_collection_offload(
        self.context.root_module,
        onload_device=observation_device,
        async_transfer=bool(self.context.svdq_kwargs.get("async_transfer", False)),
        transfer_buckets=int(self.context.svdq_kwargs.get("transfer_buckets", 1)),
        prefetch_limit=bool(self.context.svdq_kwargs.get("prefetch_limit", False)),
        max_copy_streams=self.context.svdq_kwargs.get("max_copy_streams"),
        max_inflight_prefetch_bytes=self.context.svdq_kwargs.get("max_inflight_prefetch_bytes"),
        persistent_buckets=int(self.context.svdq_kwargs.get("persistent_buckets", 0)),
        persistent_bins=int(self.context.svdq_kwargs.get("persistent_bins", 1)),
      )

    for layer_name in self.context.candidate_layer_names:
      submodule = _get_named_submodule(self.context.root_module, layer_name)
      if not isinstance(submodule, nn.Linear):
        raise TypeError(
          f"Expected nn.Linear during calibration registration, got {type(submodule)}.")
      accumulator_device = observation_device or submodule.weight.device
      torch_dtype = _normalize_dtype(None, accumulator_device)
      math_dtype = _resolve_math_dtype(torch_dtype, calibrate_precision)
      self._accumulators[layer_name] = _ActivationSpanAccumulator(
        device=accumulator_device,
        torch_dtype=torch_dtype,
        math_dtype=math_dtype,
        flush_sample_count=flush_sample_count,
        flush_cpu_bytes=flush_cpu_bytes,
      )
      self._handles.append(submodule.register_forward_pre_hook(self._make_hook(layer_name)))

  def remove(self) -> None:
    """Remove calibration hooks and finalize any buffered activation spans."""

    for handle in self._handles:
      handle.remove()
    self._handles.clear()
    self.finalize()
    for accumulator in self._accumulators.values():
      accumulator.release()
    self._accumulators.clear()
    if self._offload_handle is not None:
      self._offload_handle.remove()
      self._offload_handle = None
    _maybe_collect_svdq_garbage()

  def finalize(self) -> None:
    """Materialize finalized activation spans for all observed layers."""

    for layer_name, accumulator in self._accumulators.items():
      if accumulator.has_observations:
        self.activation_spans[layer_name] = accumulator.finalize()


@dataclasses.dataclass
class SVDQFewShotRuntimeController:
  """Defer SVDQ DQ materialization until enough runtime activations are observed.

  `few_shot_steps` counts root-module forward passes on the armed transformer/module, not
  top-level pipeline calls. The counter is cumulative for the lifetime of this controller, so
  reusing the same pipeline/module instance across multiple runs keeps accumulating toward the
  threshold instead of resetting to zero per pipeline invocation.
  """

  context: SVDQPTQContext
  quantize_device: torch.device
  completed_forwards: int = 0
  collect_current_forward: bool = False
  activation_spans: dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
  _handles: list[Any] = dataclasses.field(default_factory=list)
  _accumulators: dict[str, _ActivationSpanAccumulator] = dataclasses.field(default_factory=dict)
  _finalized: bool = False
  _offload_handle: LayerwiseOffloadHandle | None = None

  def _make_layer_hook(self, layer_name: str):
    accumulator = self._accumulators[layer_name]

    def hook(module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
      if not self.collect_current_forward or self._finalized:
        return
      if not args or not isinstance(args[0], torch.Tensor):
        raise TypeError(
          f"SVDQuant few-shot expected tensor inputs for layer {layer_name or _ROOT_LAYER_NAME}.")
      if not isinstance(module, nn.Linear):
        raise TypeError(f"Expected nn.Linear during few-shot collection, got {type(module)}.")

      activation = args[0]
      if activation.shape[-1] != module.in_features:
        raise ValueError(
          f"Expected activation last dim {module.in_features} for layer {layer_name}, "
          f"got {activation.shape[-1]}.")
      accumulator.add_tensor(activation.detach().reshape(-1, module.in_features))

    return hook

  def _on_root_forward_pre(self, _module: nn.Module, _args: tuple[Any, ...]) -> None:
    # few_shot_steps is defined in terms of root-module forwards. If the same transformer/module
    # instance is reused across multiple pipeline runs, this counter keeps accumulating until the
    # threshold is reached and runtime quantization is materialized.
    self.collect_current_forward = self.completed_forwards < self.context.svdq_kwargs[
      "few_shot_steps"]

  def _on_root_forward_post(
    self,
    module: nn.Module,
    _args: tuple[Any, ...],
    output: Any,
  ) -> Any:
    if not self.collect_current_forward or self._finalized:
      return output

    self.completed_forwards += 1
    if self.completed_forwards >= self.context.svdq_kwargs["few_shot_steps"]:
      self._finalize_runtime_quantization(module)
    return output

  def arm(self) -> nn.Module:
    if "" in self.context.candidate_layer_names:
      raise NotImplementedError(
        "SVDQuant few-shot DQ does not support quantizing a root nn.Linear directly; wrap it in a "
        "parent module or use immediate SVDQ DQ instead.")

    calibrate_precision = self.context.svdq_kwargs["calibrate_precision"]
    flush_sample_count = self.context.svdq_kwargs["activation_buffer_flush_sample_count"]
    flush_cpu_bytes = self.context.svdq_kwargs["activation_buffer_flush_cpu_bytes"]

    if bool(self.context.svdq_kwargs.get("layerwise_offload", False)):
      self._offload_handle = _maybe_enable_layerwise_collection_offload(
        self.context.root_module,
        onload_device=self.quantize_device,
        async_transfer=bool(self.context.svdq_kwargs.get("async_transfer", False)),
        transfer_buckets=int(self.context.svdq_kwargs.get("transfer_buckets", 1)),
        prefetch_limit=bool(self.context.svdq_kwargs.get("prefetch_limit", False)),
        max_copy_streams=self.context.svdq_kwargs.get("max_copy_streams"),
        max_inflight_prefetch_bytes=self.context.svdq_kwargs.get("max_inflight_prefetch_bytes"),
        persistent_buckets=int(self.context.svdq_kwargs.get("persistent_buckets", 0)),
        persistent_bins=int(self.context.svdq_kwargs.get("persistent_bins", 1)),
      )

    for layer_name in self.context.candidate_layer_names:
      submodule = _get_named_submodule(self.context.root_module, layer_name)
      if not isinstance(submodule, nn.Linear):
        raise TypeError(f"Expected nn.Linear during few-shot registration, got {type(submodule)}.")
      torch_dtype = _normalize_dtype(None, self.quantize_device)
      math_dtype = _resolve_math_dtype(torch_dtype, calibrate_precision)
      self._accumulators[layer_name] = _ActivationSpanAccumulator(
        device=self.quantize_device,
        torch_dtype=torch_dtype,
        math_dtype=math_dtype,
        flush_sample_count=flush_sample_count,
        flush_cpu_bytes=flush_cpu_bytes,
      )
      self._handles.append(submodule.register_forward_pre_hook(self._make_layer_hook(layer_name)))

    self._handles.append(
      self.context.root_module.register_forward_pre_hook(self._on_root_forward_pre))
    self._handles.append(self.context.root_module.register_forward_hook(self._on_root_forward_post))

    self.context.root_module._svdq_pending_quantization = True
    self.context.root_module._svdq_few_shot_controller = self
    self.context.root_module._svdq_cleanup_pending_quantization = self.cleanup
    logger.info(
      "SVDQuant few-shot runtime quantization for %s. \nObserving %d cumulative root forwards "
      "before materializing quantized linear layers; \nRerunning the same pipeline/module keeps "
      "advancing the same counter.",
      self.context.root_module.__class__.__name__,
      self.context.svdq_kwargs["few_shot_steps"],
    )
    return self.context.root_module

  def cleanup(self) -> None:
    for handle in self._handles:
      handle.remove()
    self._handles.clear()
    if self._offload_handle is not None:
      self._offload_handle.remove()
      self._offload_handle = None
    for accumulator in self._accumulators.values():
      accumulator.release()
    self._accumulators.clear()
    self.activation_spans.clear()
    _clear_pending_quantization_state(self.context.root_module)
    _maybe_collect_svdq_garbage(self.quantize_device)

  def _finalize_runtime_quantization(self, module: nn.Module) -> None:
    if self._finalized:
      return

    self._finalized = True
    for layer_name, accumulator in self._accumulators.items():
      if accumulator.has_observations:
        self.activation_spans[layer_name] = accumulator.finalize()

    quantize_start_time = time.perf_counter()
    resolved_smooth_scales = _resolve_few_shot_smooth_scales(
      module,
      self.context,
      self.activation_spans,
      quantize_device=self.quantize_device,
    )
    try:
      module, quantized_layer_names = _quantize_context_layers(
        module,
        self.context,
        quantize_device=self.quantize_device,
        activation_spans=self.activation_spans,
        resolved_smooth_scales=resolved_smooth_scales,
      )
    except Exception:
      resolved_smooth_scales.clear()
      self.cleanup()
      raise
    observed_layer_names = sorted(self.activation_spans)
    resolved_smooth_scales.clear()
    _maybe_collect_svdq_garbage(self.quantize_device)
    runtime_quantize_time_s = time.perf_counter() - quantize_start_time

    if not quantized_layer_names:
      self.cleanup()
      raise RuntimeError(
        "SVDQuant few-shot dynamic quantization completed activation collection but no layers were "
        f"quantized. Skipped layers: {self.context.skipped_reasons}.")

    self.cleanup()
    _attach_quantization_metadata(
      module,
      quant_type=self.context.quantize_config.quant_type,
      quantized_layer_names=quantized_layer_names,
      exclude_layers=self.context.exclude_layers,
      svdq_kwargs=self.context.svdq_kwargs,
      checkpoint_path=None,
      quantize_config=self.context.quantize_config,
    )
    _maybe_enable_layerwise_runtime_offload(
      module,
      quantized_layer_names=quantized_layer_names,
      onload_device=self.quantize_device,
      svdq_kwargs=self.context.svdq_kwargs,
    )
    module._svdq_runtime_quantize_time_s = runtime_quantize_time_s
    module._svdq_runtime_quantized_after_forwards = self.completed_forwards
    _log_quantize_summary(
      self.context,
      quantized_layer_names=quantized_layer_names,
      serialize_to=None,
      observed_layer_names=observed_layer_names,
    )
    _run_post_quantize_callbacks(
      module,
      few_shot_auto_compile=bool(self.context.svdq_kwargs.get("few_shot_auto_compile", False)),
    )


def _build_serialized_metadata(
  *,
  quant_type: str,
  rank: int,
  quantized_layer_names: list[str],
  svdq_kwargs: dict[str, Any],
) -> dict[str, str]:
  payload = {
    "format": "cache_dit_svdq_ptq",
    "version": _SVDQ_FORMAT_VERSION,
    "quant_type": quant_type,
    "rank": rank,
    "precision": "int4",
    "quantized_layer_names": [_serialize_layer_name(name) for name in quantized_layer_names],
    "svdq_kwargs": svdq_kwargs,
  }
  return {_SVDQ_METADATA_KEY: json.dumps(payload, sort_keys=True)}


def _build_quant_config_snapshot(
  *,
  checkpoint_path: str,
  quant_type: str,
  rank: int,
  exclude_layers: list[str],
  regional_quantize: bool,
  repeated_blocks: list[str] | None,
  verbose: bool,
  svdq_kwargs: dict[str, Any],
  backend: Any,
) -> dict[str, Any]:
  checkpoint_dir = _get_checkpoint_dir(checkpoint_path)
  return {
    "format": _SVDQ_QUANT_CONFIG_FORMAT,
    "version": _SVDQ_QUANT_CONFIG_VERSION,
    "backend": backend.value if hasattr(backend, "value") else str(backend),
    "quant_type": quant_type,
    "rank": rank,
    "checkpoint_path": os.path.relpath(checkpoint_path, checkpoint_dir),
    "exclude_layers": list(exclude_layers),
    "regional_quantize": regional_quantize,
    "repeated_blocks": list(repeated_blocks) if repeated_blocks is not None else None,
    "verbose": verbose,
    "svdq_kwargs": dict(svdq_kwargs),
  }


def _save_quant_config_snapshot(snapshot: dict[str, Any], *, checkpoint_path: str) -> str:
  quant_config_path = _get_quant_config_path(checkpoint_path)
  with open(quant_config_path, "w", encoding="utf-8") as file:
    json.dump(snapshot, file, indent=2, sort_keys=True)
    file.write("\n")
  return quant_config_path


def _save_quantized_module(
  module: nn.Module,
  *,
  serialize_to: str,
  quant_type: str,
  rank: int,
  quantized_layer_names: list[str],
  svdq_kwargs: dict[str, Any],
) -> None:
  _, save_file = _import_safetensors()
  tensors: dict[str, torch.Tensor] = {}
  for layer_name in quantized_layer_names:
    submodule = _get_named_submodule(module, layer_name)
    if not isinstance(submodule, SVDQW4A4Linear):
      raise TypeError(
        f"Expected SVDQW4A4Linear at {layer_name or _ROOT_LAYER_NAME}, got {type(submodule)}.")
    serialized_name = _serialize_layer_name(layer_name)
    for key, tensor in submodule.state_dict().items():
      tensors[f"{serialized_name}.{key}"] = tensor.detach().cpu().contiguous()

  save_file(
    tensors,
    serialize_to,
    metadata=_build_serialized_metadata(
      quant_type=quant_type,
      rank=rank,
      quantized_layer_names=quantized_layer_names,
      svdq_kwargs=svdq_kwargs,
    ),
  )


def _validate_quant_config_snapshot(path: str, payload: dict[str, Any]) -> dict[str, Any]:
  if payload.get("format") != _SVDQ_QUANT_CONFIG_FORMAT:
    raise ValueError(f"Invalid SVDQ quant_config format in {path}: {payload.get('format')!r}.")
  if payload.get("version") != _SVDQ_QUANT_CONFIG_VERSION:
    raise ValueError(f"Unsupported SVDQ quant_config version {payload.get('version')} in {path}.")

  quant_type = payload.get("quant_type")
  if not isinstance(quant_type, str):
    raise ValueError(f"SVDQ quant_config at {path} is missing a valid quant_type.")
  _parse_svdq_quant_type(quant_type)

  rank = payload.get("rank")
  if not isinstance(rank, int):
    raise ValueError(f"SVDQ quant_config at {path} is missing a valid integer rank.")

  checkpoint_path = payload.get("checkpoint_path")
  if not isinstance(checkpoint_path, str) or not checkpoint_path:
    raise ValueError(f"SVDQ quant_config at {path} is missing a valid checkpoint_path.")

  exclude_layers = payload.get("exclude_layers", [])
  if not isinstance(exclude_layers, list) or any(not isinstance(layer_name, str)
                                                 for layer_name in exclude_layers):
    raise ValueError(f"SVDQ quant_config at {path} has an invalid exclude_layers field.")

  repeated_blocks = payload.get("repeated_blocks")
  if repeated_blocks is not None and (not isinstance(repeated_blocks, list)
                                      or any(not isinstance(block_name, str)
                                             for block_name in repeated_blocks)):
    raise ValueError(f"SVDQ quant_config at {path} has an invalid repeated_blocks field.")

  regional_quantize = payload.get("regional_quantize", True)
  if not isinstance(regional_quantize, bool):
    raise ValueError(f"SVDQ quant_config at {path} has an invalid regional_quantize field.")

  verbose = payload.get("verbose", False)
  if not isinstance(verbose, bool):
    raise ValueError(f"SVDQ quant_config at {path} has an invalid verbose field.")

  svdq_kwargs = payload.get("svdq_kwargs", {})
  payload["svdq_kwargs"] = _resolve_svdq_kwargs(svdq_kwargs)
  payload["exclude_layers"] = exclude_layers
  payload["regional_quantize"] = regional_quantize
  payload["repeated_blocks"] = repeated_blocks
  payload["verbose"] = verbose
  return payload


def _load_quant_config_snapshot(path: str) -> dict[str, Any]:
  if not os.path.isfile(path):
    raise FileNotFoundError(f"SVDQ quant_config.json not found: {path}")

  with open(path, "r", encoding="utf-8") as file:
    try:
      payload = json.load(file)
    except json.JSONDecodeError as exc:
      raise ValueError(f"Invalid SVDQ quant_config JSON in {path}: {exc.msg}.") from exc

  if not isinstance(payload, dict):
    raise ValueError(f"SVDQ quant_config at {path} must contain a JSON object.")
  return _validate_quant_config_snapshot(path, payload)


def _maybe_load_quant_config_snapshot_for_checkpoint(checkpoint_path: str) -> dict[str, Any] | None:
  quant_config_path = _get_quant_config_path(checkpoint_path)
  if not os.path.isfile(quant_config_path):
    return None
  return _load_quant_config_snapshot(quant_config_path)


def _resolve_directory_load_source(directory_path: str) -> _ResolvedSVDQLoadSource:
  normalized_directory_path = os.path.abspath(directory_path)
  if not os.path.isdir(normalized_directory_path):
    raise FileNotFoundError(f"SVDQ PTQ checkpoint directory not found: {directory_path}")

  quant_config_path = os.path.join(normalized_directory_path, _SVDQ_QUANT_CONFIG_FILENAME)
  snapshot = _load_quant_config_snapshot(quant_config_path)
  checkpoint_path = snapshot["checkpoint_path"]
  resolved_checkpoint_path = (checkpoint_path
                              if os.path.isabs(checkpoint_path) else os.path.normpath(
                                os.path.join(normalized_directory_path, checkpoint_path)))
  if not os.path.isfile(resolved_checkpoint_path):
    raise FileNotFoundError("SVDQ PTQ checkpoint referenced by quant_config.json not found: "
                            f"{resolved_checkpoint_path}")
  return _ResolvedSVDQLoadSource(
    checkpoint_path=resolved_checkpoint_path,
    quant_config_snapshot=snapshot,
  )


def _resolve_load_path(quantize_config_or_path: QuantizeConfig | str, ) -> _ResolvedSVDQLoadSource:
  if isinstance(quantize_config_or_path, QuantizeConfig):
    if quantize_config_or_path.serialize_to is None:
      raise ValueError("serialize_to must be set when loading SVDQ PTQ from QuantizeConfig.")
    return _ResolvedSVDQLoadSource(
      checkpoint_path=os.path.abspath(quantize_config_or_path.serialize_to))
  if isinstance(quantize_config_or_path, str):
    if os.path.isdir(quantize_config_or_path):
      return _resolve_directory_load_source(quantize_config_or_path)
    checkpoint_path = os.path.abspath(quantize_config_or_path)
    return _ResolvedSVDQLoadSource(
      checkpoint_path=checkpoint_path,
      quant_config_snapshot=_maybe_load_quant_config_snapshot_for_checkpoint(checkpoint_path),
    )
  raise TypeError("quantize_config_or_path must be a QuantizeConfig or a safetensors file path, "
                  f"got {type(quantize_config_or_path)}.")


def _validate_serialized_payload(path: str, payload: dict[str, Any]) -> dict[str, Any]:
  if payload.get("format") != "cache_dit_svdq_ptq":
    raise ValueError(f"Invalid SVDQ PTQ checkpoint format in {path}: {payload.get('format')!r}.")
  if payload.get("version") != _SVDQ_FORMAT_VERSION:
    raise ValueError(f"Unsupported SVDQ PTQ checkpoint version {payload.get('version')} in {path}.")

  quant_type = payload.get("quant_type")
  if not isinstance(quant_type, str):
    raise ValueError(f"SVDQ PTQ checkpoint at {path} is missing a valid quant_type.")
  _parse_svdq_quant_type(quant_type)

  rank = payload.get("rank")
  if not isinstance(rank, int):
    raise ValueError(f"SVDQ PTQ checkpoint at {path} is missing a valid integer rank.")

  quantized_layer_names = payload.get("quantized_layer_names")
  if not isinstance(quantized_layer_names, list) or not quantized_layer_names:
    raise ValueError(
      f"SVDQ PTQ checkpoint at {path} must contain a non-empty quantized_layer_names list.")

  svdq_kwargs = payload.get("svdq_kwargs", {})
  payload["svdq_kwargs"] = _resolve_svdq_kwargs(svdq_kwargs)
  return payload


def _load_serialized_checkpoint(path: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
  if not os.path.isfile(path):
    raise FileNotFoundError(f"SVDQ PTQ checkpoint not found: {path}")

  safe_open, _ = _import_safetensors()
  with safe_open(path, framework="pt", device="cpu") as handle:
    metadata = handle.metadata() or {}
    payload_raw = metadata.get(_SVDQ_METADATA_KEY)
    if payload_raw is None:
      raise ValueError(f"SVDQ PTQ metadata key {_SVDQ_METADATA_KEY!r} not found in {path}.")
    try:
      payload = json.loads(payload_raw)
    except json.JSONDecodeError as exc:
      raise ValueError(f"Invalid SVDQ PTQ metadata JSON in {path}: {exc.msg}.") from exc
    tensors = {key: handle.get_tensor(key) for key in handle.keys()}

  return tensors, _validate_serialized_payload(path, payload)


def _log_quantize_summary(
  context: SVDQPTQContext,
  *,
  quantized_layer_names: list[str],
  serialize_to: str | None,
  observed_layer_names: list[str] | None,
) -> None:
  _log_boxed_summary(
    context.quantize_summary_lines(
      observed_layer_names=observed_layer_names,
      quantized_layer_names=quantized_layer_names,
      serialize_to=serialize_to,
    ))
  _log_skipped_reasons(context.skipped_reasons, verbose=context.verbose)


def _log_load_summary(
  *,
  module: nn.Module,
  quant_type: str,
  rank: int,
  loaded_layer_names: list[str],
  checkpoint_path: str,
  svdq_kwargs: dict[str, Any],
  verbose: bool,
) -> None:
  lines = [
    f"SVDQ Load       Region: {module.__class__.__name__}",
    f"SVDQ Load   Quant Type: {quant_type}",
    f"SVDQ Load         Rank: {rank}",
    f"Loaded   Linear Layers: {len(loaded_layer_names)}",
  ]
  if verbose:
    lines.append(f"SVDQ            Kwargs: {svdq_kwargs}")
    lines.append(f"Loaded          Layers: {loaded_layer_names}")
    lines.append(f"Checkpoint        Path: {checkpoint_path}")
  _log_boxed_summary(lines)


def _resolve_few_shot_smooth_scales(
  module: nn.Module,
  context: SVDQPTQContext,
  activation_spans: dict[str, torch.Tensor],
  *,
  quantize_device: torch.device,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
  resolved: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
  calibrate_precision = context.svdq_kwargs["calibrate_precision"]
  relax_factor = float(context.svdq_kwargs["few_shot_relax_factor"])
  relax_top_ratio = float(context.svdq_kwargs["few_shot_relax_top_ratio"])
  relax_strategy = str(context.svdq_kwargs["few_shot_relax_strategy"])
  alpha = 0.5

  for layer_name, activation_span in activation_spans.items():
    if layer_name not in context.candidate_layer_names:
      continue
    float_module = _get_named_submodule(module, layer_name)
    if not isinstance(float_module, nn.Linear):
      raise TypeError(
        f"Expected nn.Linear at {layer_name or _ROOT_LAYER_NAME}, got {type(float_module)}.")
    torch_dtype = _normalize_dtype(None, quantize_device)
    math_dtype = _resolve_math_dtype(torch_dtype, calibrate_precision)
    weight = float_module.weight.detach().to(device=quantize_device, dtype=torch_dtype)
    weight_span = weight.to(dtype=math_dtype).abs().amax(dim=0)
    smooth_scale, smooth_scale_orig = _apply_few_shot_relaxation(
      activation_span,
      weight_span,
      alpha=alpha,
      relax_factor=relax_factor,
      relax_top_ratio=relax_top_ratio,
      relax_strategy=relax_strategy,
      math_dtype=math_dtype,
      output_dtype=torch_dtype,
    )
    resolved[layer_name] = (smooth_scale, smooth_scale_orig)
    del weight, weight_span
  return resolved


def _quantize_context_layers(
  module: nn.Module,
  context: SVDQPTQContext,
  *,
  quantize_device: torch.device,
  activation_spans: dict[str, torch.Tensor] | None = None,
  resolved_smooth_scales: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> tuple[nn.Module, list[str]]:
  quantized_layer_names: list[str] = []
  offload_quantized_layers_to_cpu = bool(
    context.svdq_kwargs.get("offload_quantized_layers_to_cpu", False))
  quantized_layer_device = _resolve_quantized_layer_storage_device(
    context.svdq_kwargs,
    quantize_device,
  )
  for layer_name in context.candidate_layer_names:
    activation_span = None if activation_spans is None else activation_spans.get(layer_name)
    smooth_scales = None if resolved_smooth_scales is None else resolved_smooth_scales.get(
      layer_name)
    if activation_spans is not None and activation_span is None and smooth_scales is None:
      context._record_skip(layer_name, "not observed during calibration")
      continue

    float_module = _get_named_submodule(module, layer_name)
    if not isinstance(float_module, nn.Linear):
      raise TypeError(
        f"Expected nn.Linear at {layer_name or _ROOT_LAYER_NAME}, got {type(float_module)}.")
    float_module_device = float_module.weight.device
    quantized_torch_dtype = _normalize_dtype(None, quantize_device)

    if smooth_scales is not None:
      smooth_scale, smooth_scale_orig = smooth_scales
      quantized_module = _quantize_linear_svdq_w4a4_from_smooth_scale(
        float_module,
        smooth_scale,
        smooth_scale_orig=smooth_scale_orig,
        quant_type=context.quantize_config.quant_type,
        rank=context.rank,
        precision=context.precision,
        torch_dtype=quantized_torch_dtype,
        device=quantize_device,
        calibrate_precision=context.svdq_kwargs["calibrate_precision"],
        runtime_kernel=context.svdq_kwargs["runtime_kernel"],
      )
    else:
      quantized_module = _quantize_linear_svdq_w4a4_from_activation_span(
        float_module,
        activation_span,
        quant_type=context.quantize_config.quant_type,
        rank=context.rank,
        smooth_strategy=context.svdq_kwargs["smooth_strategy"],
        precision=context.precision,
        torch_dtype=quantized_torch_dtype,
        device=quantize_device,
        calibrate_precision=context.svdq_kwargs["calibrate_precision"],
        runtime_kernel=context.svdq_kwargs["runtime_kernel"],
      )

    if quantized_layer_device != quantize_device:
      quantized_module = quantized_module.to(device=quantized_layer_device)

    if layer_name == "":
      module = quantized_module
    else:
      parent, child_name = _get_parent_and_child(module, layer_name)
      setattr(parent, child_name, quantized_module)
    quantized_layer_names.append(layer_name)

    del float_module, quantized_module
    if quantize_device.type == "cuda" and (offload_quantized_layers_to_cpu
                                           or float_module_device.type != "cuda"):
      _maybe_collect_svdq_garbage(quantize_device)

  return module, quantized_layer_names


def quantize_svdq_ptq(module: nn.Module, quantize_config: QuantizeConfig) -> nn.Module:
  """Run post-training SVDQ quantization over a module in place.

  :param module: Root module containing the float `nn.Linear` layers to quantize.
  :param quantize_config: SVDQ `QuantizeConfig` describing calibration, scope
  filters, serialization target, and SVDQ-specific kwargs.

  :returns: The input module with eligible `nn.Linear` submodules replaced by
  `SVDQW4A4Linear` instances. The return value may be a new module only when
  the root module itself is quantized.
  """

  if not isinstance(module, nn.Module):
    raise TypeError(f"Expected nn.Module, got {type(module)}.")
  if not quantize_config.is_svdq():
    raise ValueError(
      f"quantize_svdq_ptq only supports SVDQ quant types, got {quantize_config.quant_type}.")
  if check_quantized(module):
    logger.warning("Module %s is already quantized, skipping SVDQ PTQ.", module.__class__.__name__)
    return module

  context = SVDQPTQContext.from_config(module, quantize_config)
  calibrator = SVDQPTQCalibrator(context)
  quantize_device = _resolve_svdq_quantize_device(module, context.svdq_kwargs)
  calibrator.register(observation_device=quantize_device)

  was_training = module.training
  module.eval()
  try:
    with torch.inference_mode():
      _call_with_supported_kwargs(
        quantize_config.calibrate_fn,
        model=module,
        quantize_config=quantize_config,
        calibrator=calibrator,
        component_name=getattr(module, "_actual_module_name", module.__class__.__name__),
        svdq_kwargs=context.svdq_kwargs,
      )
  finally:
    calibrator.remove()
    module.train(was_training)

  module, quantized_layer_names = _quantize_context_layers(
    module,
    context,
    quantize_device=quantize_device,
    activation_spans=calibrator.activation_spans,
  )
  observed_layer_names = sorted(calibrator.activation_spans)

  if not quantized_layer_names:
    raise RuntimeError("SVDQ PTQ completed calibration but no layers were quantized. "
                       f"Skipped layers: {context.skipped_reasons}.")

  _attach_quantization_metadata(
    module,
    quant_type=quantize_config.quant_type,
    quantized_layer_names=quantized_layer_names,
    exclude_layers=context.exclude_layers,
    svdq_kwargs=context.svdq_kwargs,
    checkpoint_path=quantize_config.serialize_to,
    quantize_config=quantize_config,
  )
  _save_quantized_module(
    module,
    serialize_to=quantize_config.serialize_to,
    quant_type=quantize_config.quant_type,
    rank=context.rank,
    quantized_layer_names=quantized_layer_names,
    svdq_kwargs=context.svdq_kwargs,
  )
  _save_quant_config_snapshot(
    _build_quant_config_snapshot(
      checkpoint_path=quantize_config.serialize_to,
      quant_type=quantize_config.quant_type,
      rank=context.rank,
      exclude_layers=context.exclude_layers,
      regional_quantize=context.regional_quantize,
      repeated_blocks=context.repeated_blocks,
      verbose=context.verbose,
      svdq_kwargs=context.svdq_kwargs,
      backend=quantize_config.backend,
    ),
    checkpoint_path=quantize_config.serialize_to,
  )
  module.train(was_training)
  _log_quantize_summary(
    context,
    quantized_layer_names=quantized_layer_names,
    serialize_to=quantize_config.serialize_to,
    observed_layer_names=observed_layer_names,
  )
  calibrator.activation_spans.clear()
  _maybe_collect_svdq_garbage(quantize_device)
  return module


def quantize_svdq_dq(module: nn.Module, quantize_config: QuantizeConfig) -> nn.Module:
  """Run in-memory SVDQ dynamic quantization over a module in place.

  :param module: Root module containing the float `nn.Linear` layers to quantize.
  :param quantize_config: SVDQ `QuantizeConfig` describing the dynamic quantization
    scope and rank.

  :returns: The input module with eligible `nn.Linear` submodules replaced by
    `SVDQW4A4Linear` instances. The return value may be a new module only when
    the root module itself is quantized.
  """

  if not isinstance(module, nn.Module):
    raise TypeError(f"Expected nn.Module, got {type(module)}.")
  if not quantize_config.is_svdq_dq():
    raise ValueError("quantize_svdq_dq only supports SVDQ dynamic quant types ending with '_dq', "
                     f"got {quantize_config.quant_type}.")
  if check_quantized(module):
    logger.warning(
      "Module %s is already quantized, skipping SVDQ dynamic quantization.",
      module.__class__.__name__,
    )
    return module
  if getattr(module, "_svdq_pending_quantization", False):
    logger.warning(
      "Module %s already has pending SVDQuant few-shot quantization, skipping re-arming.",
      module.__class__.__name__,
    )
    return module

  context = SVDQPTQContext.from_config(module, quantize_config)
  quantize_device = _resolve_svdq_quantize_device(module, context.svdq_kwargs)

  if quantize_config.is_svdq_dq_few_shot():
    controller = SVDQFewShotRuntimeController(context=context, quantize_device=quantize_device)
    return controller.arm()

  was_training = module.training
  module.eval()
  try:
    module, quantized_layer_names = _quantize_context_layers(
      module,
      context,
      quantize_device=quantize_device,
      activation_spans=None,
    )
  finally:
    module.train(was_training)

  if not quantized_layer_names:
    raise RuntimeError("SVDQ dynamic quantization found no layers to quantize. "
                       f"Skipped layers: {context.skipped_reasons}.")

  _attach_quantization_metadata(
    module,
    quant_type=quantize_config.quant_type,
    quantized_layer_names=quantized_layer_names,
    exclude_layers=context.exclude_layers,
    svdq_kwargs=context.svdq_kwargs,
    checkpoint_path=None,
    quantize_config=quantize_config,
  )
  _log_quantize_summary(
    context,
    quantized_layer_names=quantized_layer_names,
    serialize_to=None,
    observed_layer_names=None,
  )
  _maybe_collect_svdq_garbage(quantize_device)
  return module


def load_svdq(
  module: nn.Module,
  quantize_config_or_path: QuantizeConfig | str,
) -> nn.Module:
  """Load a serialized SVDQ PTQ checkpoint into a float module in place.

  :param module: Root module containing float `nn.Linear` layers that match the
  serialized checkpoint layout.
  :param quantize_config_or_path: Either a `QuantizeConfig` whose `serialize_to`
  points at an SVDQ checkpoint or a direct checkpoint path.

  :returns: The module with serialized layers replaced by `SVDQW4A4Linear` instances.
  The return value may be a new module if the root layer itself was
  serialized and loaded.
  """

  if not isinstance(module, nn.Module):
    raise TypeError(f"Expected nn.Module, got {type(module)}.")

  resolved_load_source = _resolve_load_path(quantize_config_or_path)
  checkpoint_path = resolved_load_source.checkpoint_path
  tensors, payload = _load_serialized_checkpoint(checkpoint_path)
  quant_type = payload["quant_type"]
  rank = int(payload["rank"])
  quantized_layer_names = [
    _deserialize_layer_name(name) for name in payload.get("quantized_layer_names", [])
  ]
  svdq_kwargs = dict(payload.get("svdq_kwargs", {}))
  quant_config_snapshot = resolved_load_source.quant_config_snapshot

  if quant_config_snapshot is not None and quant_config_snapshot["quant_type"] != quant_type:
    raise ValueError("The SVDQ checkpoint quant_type does not match quant_config.json: "
                     f"{quant_type} vs {quant_config_snapshot['quant_type']}.")

  if isinstance(quantize_config_or_path, QuantizeConfig):
    if not quantize_config_or_path.is_svdq():
      raise ValueError(
        "load_svdq only supports QuantizeConfig entries whose quant_type starts with 'svdq'.")
    if quantize_config_or_path.quant_type != quant_type:
      raise ValueError("The SVDQ checkpoint quant_type does not match QuantizeConfig: "
                       f"{quant_type} vs {quantize_config_or_path.quant_type}.")

  load_device = _infer_module_execution_device(module)
  was_training = module.training
  loaded_layer_names: list[str] = []
  for layer_name in quantized_layer_names:
    serialized_name = _serialize_layer_name(layer_name)
    layer_state_dict = {
      key.removeprefix(f"{serialized_name}."): tensor
      for key, tensor in tensors.items() if key.startswith(f"{serialized_name}.")
    }
    if not layer_state_dict:
      raise ValueError(f"No serialized tensors found for SVDQ layer {layer_name!r}.")

    float_module = _get_named_submodule(module, layer_name)
    if not isinstance(float_module, nn.Linear):
      raise TypeError(
        f"Expected a float nn.Linear at {layer_name or _ROOT_LAYER_NAME} when loading SVDQ PTQ, "
        f"got {type(float_module)}.")

    torch_dtype = layer_state_dict["smooth_factor"].dtype
    quantized_module = SVDQW4A4Linear.from_linear(
      float_module,
      rank=rank,
      precision="int4",
      runtime_kernel=svdq_kwargs["runtime_kernel"],
      torch_dtype=torch_dtype,
      device=load_device,
    )
    incompatible = quantized_module.load_state_dict(layer_state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
      raise RuntimeError(
        f"Unexpected SVDQ PTQ state_dict mismatch for {layer_name!r}: "
        f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}.")

    if layer_name == "":
      module = quantized_module
    else:
      parent, child_name = _get_parent_and_child(module, layer_name)
      setattr(parent, child_name, quantized_module)
    loaded_layer_names.append(layer_name)

  _attach_quantization_metadata(
    module,
    quant_type=quant_type,
    quantized_layer_names=loaded_layer_names,
    exclude_layers=(list(quantize_config_or_path.exclude_layers or []) if isinstance(
      quantize_config_or_path, QuantizeConfig) else list((quant_config_snapshot or {}).get(
        "exclude_layers", getattr(module, "_exclude_layers", [])))),
    svdq_kwargs=svdq_kwargs,
    checkpoint_path=checkpoint_path,
    quantize_config=(quantize_config_or_path
                     if isinstance(quantize_config_or_path, QuantizeConfig) else None),
  )
  module.train(was_training)
  _log_load_summary(
    module=module,
    quant_type=quant_type,
    rank=rank,
    loaded_layer_names=loaded_layer_names,
    checkpoint_path=checkpoint_path,
    svdq_kwargs=svdq_kwargs,
    verbose=isinstance(quantize_config_or_path, QuantizeConfig) and quantize_config_or_path.verbose,
  )
  return module


__all__ = ["load_svdq", "quantize_svdq_dq", "quantize_svdq_ptq", "SVDQPTQCalibrator"]
