from __future__ import annotations

import argparse
import dataclasses
import inspect
import re
from decimal import Decimal
from typing import Any, Callable, Iterable

import torch
from torch import nn

from ..logger import init_logger

try:
  from diffusers import DiffusionPipeline
except ImportError:
  DiffusionPipeline = None

try:
  from ..caching import BlockAdapter
except ImportError:
  BlockAdapter = None

try:
  from accelerate.hooks import AlignDevicesHook as _AccelerateAlignDevicesHook
  from accelerate.hooks import CpuOffload as _AccelerateCpuOffload
except ImportError:
  _AccelerateAlignDevicesHook = None
  _AccelerateCpuOffload = None

logger = init_logger(__name__)

_LAYERWISE_OFFLOAD_HANDLES_ATTR = "_cache_dit_layerwise_offload_handles"
_ROOT_TARGET_NAME = "<root>"
_MAX_EFFECTIVE_ASYNC_TRANSFER_BUCKETS = 4
_MAX_FUTURE_PREFETCH_WINDOW = 8
_BYTE_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?i?b)?\s*$", re.IGNORECASE)
_BYTE_SIZE_UNITS = {
  None: 1,
  "b": 1,
  "kib": 1024,
  "mib": 1024 ** 2,
  "gib": 1024 ** 3,
  "tib": 1024 ** 4,
  "kb": 1024,
  "mb": 1024 ** 2,
  "gb": 1024 ** 3,
  "tb": 1024 ** 4,
}


def _parse_byte_size_arg(value: str) -> int:
  """Parse a positive byte-size string.

  Accepts raw byte integers such as ``4096`` and binary-size suffixes such as ``512MiB`` or
  ``4GiB``. Decimal values are allowed when a suffix is present, for example ``0.5GiB``.

  :param value: Raw byte-size string.
  :returns: Parsed positive size in bytes.
  """

  match = _BYTE_SIZE_RE.match(value)
  if match is None:
    raise argparse.ArgumentTypeError("Expected a positive byte value like 4096, 512MiB, or 4GiB.")

  number_text, unit_text = match.groups()
  try:
    number = Decimal(number_text)
  except Exception as exc:
    raise argparse.ArgumentTypeError(
      f"Invalid byte-size value {value!r}; expected a positive number.") from exc

  if number <= 0:
    raise argparse.ArgumentTypeError(f"Byte-size value must be > 0, got {value!r}.")

  normalized_unit = unit_text.lower() if unit_text is not None else None
  multiplier = _BYTE_SIZE_UNITS.get(normalized_unit)
  if multiplier is None:
    raise argparse.ArgumentTypeError(
      f"Unsupported byte-size suffix in {value!r}; use bytes or KiB/MiB/GiB/TiB.")

  byte_value = number * multiplier
  if byte_value != byte_value.to_integral_value():
    raise argparse.ArgumentTypeError(
      f"Byte-size value {value!r} does not resolve to a whole number of bytes.")
  return int(byte_value)


def _normalize_max_inflight_prefetch_bytes_arg(value: int | str | None) -> int | None:
  if value is None:
    return None
  if isinstance(value, bool):
    raise TypeError(f"max_inflight_prefetch_bytes must be an int, str, or None, got {type(value)}.")
  if isinstance(value, int):
    if value < 1:
      raise ValueError(f"max_inflight_prefetch_bytes must be >= 1, got {value}.")
    return value
  if isinstance(value, str):
    try:
      return _parse_byte_size_arg(value)
    except argparse.ArgumentTypeError as exc:
      raise ValueError(str(exc)) from exc
  raise TypeError(f"max_inflight_prefetch_bytes must be an int, str, or None, got {type(value)}.")


def _resolve_persistent_target_spans(
  *,
  target_count: int,
  persistent_buckets: int,
  persistent_bins: int,
) -> list[tuple[int, int]]:
  effective_persistent_buckets = min(persistent_buckets, target_count)
  if target_count <= 0 or effective_persistent_buckets <= 0:
    return []

  effective_persistent_bins = min(max(persistent_bins, 1), effective_persistent_buckets,
                                  target_count)
  boundaries = [(index * target_count) // effective_persistent_bins
                for index in range(effective_persistent_bins + 1)]
  capacities = [
    boundaries[index + 1] - boundaries[index] for index in range(effective_persistent_bins)
  ]
  allocations = [0 for _ in range(effective_persistent_bins)]

  remaining = effective_persistent_buckets
  while remaining > 0:
    made_progress = False
    for index, capacity in enumerate(capacities):
      if allocations[index] >= capacity:
        continue
      allocations[index] += 1
      remaining -= 1
      made_progress = True
      if remaining == 0:
        break
    if not made_progress:
      break

  return [(boundaries[index], boundaries[index] + allocated)
          for index, allocated in enumerate(allocations) if allocated > 0]


def _collect_persistent_target_spans(targets: list["_LayerwiseTarget"]) -> list[tuple[int, int]]:
  spans: list[tuple[int, int]] = []
  span_start: int | None = None
  for index, target in enumerate(targets):
    if target.persistent:
      if span_start is None:
        span_start = index
      continue
    if span_start is not None:
      spans.append((span_start, index))
      span_start = None
  if span_start is not None:
    spans.append((span_start, len(targets)))
  return spans


def _target_prefetch_residency_bytes(target: "_LayerwiseTarget") -> int:
  return sum(tensor_state.cpu_tensor.numel() * tensor_state.cpu_tensor.element_size()
             for tensor_state in target.tensor_states)


_FORWARD_HOOK_SUPPORTS_ALWAYS_CALL = "always_call" in inspect.signature(
  nn.Module.register_forward_hook).parameters
_FORWARD_HOOK_SUPPORTS_PREPEND = "prepend" in inspect.signature(
  nn.Module.register_forward_hook).parameters
_FORWARD_PRE_HOOK_SUPPORTS_PREPEND = "prepend" in inspect.signature(
  nn.Module.register_forward_pre_hook).parameters


def _map_tensors(value: Any, *, transform: Callable[[torch.Tensor], torch.Tensor]) -> Any:
  if isinstance(value, torch.Tensor):
    return transform(value)
  if dataclasses.is_dataclass(value) and not isinstance(value, type):
    init_field_values: dict[str, Any] = {}
    non_init_field_values: dict[str, Any] = {}
    for field in dataclasses.fields(value):
      mapped_value = _map_tensors(getattr(value, field.name), transform=transform)
      if field.init:
        init_field_values[field.name] = mapped_value
      else:
        non_init_field_values[field.name] = mapped_value

    rebuilt_value = dataclasses.replace(value, **init_field_values)
    for field_name, mapped_value in non_init_field_values.items():
      object.__setattr__(rebuilt_value, field_name, mapped_value)
    return rebuilt_value
  if isinstance(value, tuple):
    return tuple(_map_tensors(item, transform=transform) for item in value)
  if isinstance(value, list):
    return [_map_tensors(item, transform=transform) for item in value]
  if isinstance(value, dict):
    return {key: _map_tensors(item, transform=transform) for key, item in value.items()}
  return value


def _find_first_tensor_device(value: Any) -> torch.device | None:
  if isinstance(value, torch.Tensor):
    return value.device
  if isinstance(value, tuple) or isinstance(value, list):
    for item in value:
      device = _find_first_tensor_device(item)
      if device is not None:
        return device
    return None
  if isinstance(value, dict):
    for item in value.values():
      device = _find_first_tensor_device(item)
      if device is not None:
        return device
  return None


def _move_tree_to_device(
  value: Any,
  device: torch.device,
  *,
  non_blocking: bool,
) -> Any:
  return _map_tensors(
    value,
    transform=lambda tensor: tensor.to(device=device, non_blocking=non_blocking),
  )


def _module_has_direct_state(module: nn.Module) -> bool:
  try:
    next(module.parameters(recurse=False))
    return True
  except StopIteration:
    pass

  try:
    next(module.buffers(recurse=False))
    return True
  except StopIteration:
    pass
  return False


def _module_uses_meta_tensors(module: nn.Module) -> bool:
  for tensor in list(module.parameters()) + list(module.buffers()):
    if tensor.device.type == "meta":
      return True
  return False


def _move_module_state(module: nn.Module, device: torch.device, *, non_blocking: bool) -> None:
  if _module_uses_meta_tensors(module):
    raise ValueError(
      "Layerwise offload does not support modules with meta tensors. Remove external meta-based "
      "offload hooks or materialize the module before applying cache_dit.offload.")
  module.to(device=device, non_blocking=non_blocking)


def _iter_registered_hf_hooks(hook: Any):
  if hook is None:
    return
  yield hook
  nested_hooks = getattr(hook, "hooks", None)
  if isinstance(nested_hooks, (list, tuple)):
    for nested_hook in nested_hooks:
      yield from _iter_registered_hf_hooks(nested_hook)


def _is_offload_related_hf_hook(hook: Any) -> bool:
  if hook is None:
    return False
  if _AccelerateCpuOffload is not None and isinstance(hook, _AccelerateCpuOffload):
    return True
  if _AccelerateAlignDevicesHook is not None and isinstance(hook, _AccelerateAlignDevicesHook):
    return bool(getattr(hook, "offload", False))

  hook_cls_name = type(hook).__name__
  if hook_cls_name == "CpuOffload":
    return True
  if hook_cls_name == "AlignDevicesHook":
    return bool(getattr(hook, "offload", False))
  return False


def _find_offload_related_hf_hook(module: nn.Module) -> tuple[str, Any] | None:
  for submodule_name, submodule in module.named_modules():
    registered_hook = getattr(submodule, "_hf_hook", None)
    if registered_hook is None:
      continue
    for hook in _iter_registered_hf_hooks(registered_hook):
      if _is_offload_related_hf_hook(hook):
        return submodule_name, hook
  return None


def _call_module_filter(
  module_filter: Callable[..., bool],
  *,
  name: str,
  module: nn.Module,
) -> bool:
  try:
    signature = inspect.signature(module_filter)
  except (TypeError, ValueError):
    return bool(module_filter(name, module))

  if any(parameter.kind == inspect.Parameter.VAR_KEYWORD
         for parameter in signature.parameters.values()):
    return bool(module_filter(name=name, module=module))

  accepted_kwargs = {}
  if "name" in signature.parameters:
    accepted_kwargs["name"] = name
  if "module" in signature.parameters:
    accepted_kwargs["module"] = module
  if len(signature.parameters) == 1 and not accepted_kwargs:
    return bool(module_filter(module))
  if not accepted_kwargs:
    return bool(module_filter(name, module))
  return bool(module_filter(**accepted_kwargs))


def _default_module_filter(name: str, module: nn.Module) -> bool:
  if name == "":
    return False
  if any(True for _ in module.children()):
    return False
  return _module_has_direct_state(module)


def _resolve_target_modules(
  root_module: nn.Module,
  *,
  module_names: Iterable[str] | None,
  module_filter: Callable[..., bool] | None,
) -> list[tuple[str, nn.Module]]:
  resolved: list[tuple[str, nn.Module]] = []
  seen_module_ids: set[int] = set()

  if module_names is not None:
    for module_name in module_names:
      submodule = root_module if module_name == "" else root_module.get_submodule(module_name)
      if id(submodule) in seen_module_ids:
        continue
      seen_module_ids.add(id(submodule))
      resolved.append((module_name, submodule))
    return resolved

  effective_filter = module_filter or _default_module_filter
  for module_name, submodule in root_module.named_modules():
    if id(submodule) in seen_module_ids:
      continue
    if not _call_module_filter(effective_filter, name=module_name, module=submodule):
      continue
    seen_module_ids.add(id(submodule))
    resolved.append((module_name, submodule))

  return resolved


def _targets_cover_all_parameterized_leaf_modules(
  root_module: nn.Module,
  *,
  resolved_targets: list[tuple[str, nn.Module]],
) -> bool:
  expected_targets = _resolve_target_modules(
    root_module,
    module_names=None,
    module_filter=None,
  )
  return {id(module)
          for _name, module in resolved_targets
          } == {id(module)
                for _name, module in expected_targets}


def _iter_direct_state_entries(module: nn.Module):
  for name, parameter in module.named_parameters(recurse=False):
    yield "parameter", name, parameter, parameter.requires_grad
  for name, buffer in module.named_buffers(recurse=False):
    yield "buffer", name, buffer, False


def _get_direct_state_tensor(module: nn.Module, *, kind: str, name: str) -> torch.Tensor:
  if kind == "parameter":
    parameter = module._parameters.get(name)
    if parameter is None:
      raise KeyError(f"Missing parameter {name!r} on module {module.__class__.__name__}.")
    return parameter.data
  tensor = module._buffers.get(name)
  if tensor is None:
    raise KeyError(f"Missing buffer {name!r} on module {module.__class__.__name__}.")
  return tensor


def _assign_direct_state_tensor(
  module: nn.Module,
  *,
  kind: str,
  name: str,
  tensor: torch.Tensor,
  requires_grad: bool,
) -> None:
  if kind == "parameter":
    parameter = module._parameters.get(name)
    if parameter is None:
      raise KeyError(f"Missing parameter {name!r} on module {module.__class__.__name__}.")
    parameter.data = tensor
    if parameter.requires_grad != requires_grad:
      parameter.requires_grad_(requires_grad)
    return
  if name not in module._buffers:
    raise KeyError(f"Missing buffer {name!r} on module {module.__class__.__name__}.")
  module._buffers[name] = tensor


def _infer_module_state_device(module: nn.Module) -> torch.device:
  for _kind, _name, tensor, _requires_grad in _iter_direct_state_entries(module):
    return tensor.device
  return torch.device("cpu")


@dataclasses.dataclass
class _LayerwiseTensorState:
  kind: str
  name: str
  requires_grad: bool
  cpu_tensor: torch.Tensor


@dataclasses.dataclass
class _LayerwiseTarget:
  name: str
  module: nn.Module
  index: int
  persistent: bool = False
  return_devices: list[torch.device] = dataclasses.field(default_factory=list)
  tensor_states: list[_LayerwiseTensorState] = dataclasses.field(default_factory=list)
  resident_device: torch.device = dataclasses.field(default_factory=lambda: torch.device("cpu"))
  pending_onload_event: torch.cuda.Event | None = None
  pending_offload_event: torch.cuda.Event | None = None
  pending_onload_stream_index: int | None = None
  pending_offload_stream_index: int | None = None
  prefetch_residency_bytes: int = 0


class LayerwiseOffloadHandleGroup:
  """Aggregate multiple per-root layerwise offload handles.

  This is returned when the public layerwise offload API is applied to a pipeline-like object that
  resolves to multiple root modules. Each underlying handle still owns its own leaf-module hooks;
  the group only provides a single lifecycle object for callers.
  """

  def __init__(
    self,
    *,
    source: Any,
    handles: Iterable["LayerwiseOffloadHandle"],
    root_names: Iterable[str],
  ) -> None:
    self.source = source
    self.handles = tuple(handles)
    self.root_names = tuple(root_names)

  @property
  def module_names(self) -> list[str]:
    aggregated_names: list[str] = []
    for root_name, handle in zip(self.root_names, self.handles):
      prefix = f"{root_name}." if root_name else ""
      for module_name in handle.module_names:
        if module_name:
          aggregated_names.append(f"{prefix}{module_name}")
        else:
          aggregated_names.append(root_name)
    return aggregated_names

  def __len__(self) -> int:
    return len(self.handles)

  def __iter__(self):
    return iter(self.handles)

  def remove(self, *, offload: bool = True) -> None:
    for handle in self.handles:
      handle.remove(offload=offload)


def _is_diffusion_pipeline_instance(value: Any) -> bool:
  return DiffusionPipeline is not None and isinstance(value, DiffusionPipeline)


def _is_block_adapter_instance(value: Any) -> bool:
  return BlockAdapter is not None and isinstance(value, BlockAdapter)


def _append_public_offload_root(
  roots: list[tuple[str, nn.Module]],
  seen_module_ids: set[int],
  name: str,
  module: Any,
) -> None:
  if not isinstance(module, nn.Module):
    return
  if id(module) in seen_module_ids:
    return
  seen_module_ids.add(id(module))
  roots.append((name, module))


def _resolve_named_public_offload_root(root_or_adapter: Any, module_name: str) -> nn.Module:
  if isinstance(root_or_adapter, nn.Module):
    return root_or_adapter if module_name == "" else root_or_adapter.get_submodule(module_name)

  path_parts = module_name.split(".") if module_name else []
  current: Any = root_or_adapter
  for index, path_part in enumerate(path_parts):
    if isinstance(current, nn.Module):
      remainder = ".".join(path_parts[index:])
      current = current if remainder == "" else current.get_submodule(remainder)
      break
    if isinstance(current, dict):
      if path_part not in current:
        raise AttributeError(f"{type(current).__name__} has no key {path_part!r}.")
      current = current[path_part]
      continue
    if isinstance(current, (list, tuple)):
      try:
        current = current[int(path_part)]
      except (ValueError, IndexError) as exc:
        raise AttributeError(
          f"{type(current).__name__} cannot resolve index {path_part!r} in {module_name!r}."
        ) from exc
      continue
    if not hasattr(current, path_part):
      raise AttributeError(f"{type(current).__name__} has no attribute {path_part!r}.")
    current = getattr(current, path_part)

  if not isinstance(current, nn.Module):
    raise TypeError(
      f"Resolved object for module name {module_name!r} is not an nn.Module: {type(current)}.")
  return current


def _collect_public_offload_roots(root_or_adapter: Any) -> list[tuple[str, nn.Module]]:
  roots: list[tuple[str, nn.Module]] = []
  seen_module_ids: set[int] = set()

  if isinstance(root_or_adapter, nn.Module):
    return [("", root_or_adapter)]

  if _is_block_adapter_instance(root_or_adapter):
    transformer = getattr(root_or_adapter, "transformer", None)
    if isinstance(transformer, list):
      flattened_transformers = (BlockAdapter.flatten(transformer)
                                if BlockAdapter is not None else transformer)
      for index, module in enumerate(flattened_transformers):
        _append_public_offload_root(roots, seen_module_ids, f"transformer[{index}]", module)
    else:
      _append_public_offload_root(roots, seen_module_ids, "transformer", transformer)
    if roots:
      return roots
    root_or_adapter = getattr(root_or_adapter, "pipe", root_or_adapter)

  if _is_diffusion_pipeline_instance(root_or_adapter):
    components = getattr(root_or_adapter, "components", None)
    if isinstance(components, dict):
      for name, module in components.items():
        _append_public_offload_root(roots, seen_module_ids, str(name), module)
      if roots:
        return roots

  if hasattr(root_or_adapter, "__dict__"):
    for name, module in vars(root_or_adapter).items():
      _append_public_offload_root(roots, seen_module_ids, str(name), module)

  return roots


def _resolve_public_offload_root_specs(
  root_or_adapter: Any,
  *,
  module_names: Iterable[str] | None,
  module_filter: Callable[..., bool] | None,
) -> list[tuple[str, nn.Module, Iterable[str] | None, Callable[..., bool] | None]]:
  if isinstance(root_or_adapter, nn.Module):
    return [("", root_or_adapter, module_names, module_filter)]

  if module_names is not None:
    resolved_roots: list[tuple[str, nn.Module, None, None]] = []
    seen_module_ids: set[int] = set()
    for module_name in module_names:
      resolved_module = _resolve_named_public_offload_root(root_or_adapter, module_name)
      if id(resolved_module) in seen_module_ids:
        continue
      seen_module_ids.add(id(resolved_module))
      resolved_roots.append((module_name, resolved_module, None, None))
    return resolved_roots

  resolved_roots = _collect_public_offload_roots(root_or_adapter)
  if module_filter is None:
    return [(name, module, None, None) for name, module in resolved_roots]
  return [(name, module, None, None) for name, module in resolved_roots
          if _call_module_filter(module_filter, name=name, module=module)]


class LayerwiseOffloadHandle:
  """Registered layerwise offload hooks for a root module.

  The handle owns the forward pre/post hooks that move selected submodules to the onload device
  just in time, preserve the caller-visible tensor device for outputs, and offload the submodule
  state back after execution. When ``async_transfer=True``, the handle owns separate CUDA onload
  and offload copy stream pools sized by ``transfer_buckets`` so offload waits on the current
  compute stream do not serialize later onload prefetch behind the same copy lane.

  :param root_module: Root module that owns the registered hooks.
  :param targets: Selected submodules that participate in layerwise offload.
  :param handles: Torch hook handles registered on each target module.
  :param onload_device: Device used during the target module forward.
  :param offload_device: Residency device used after the target module forward.
  :param output_device: Optional fixed output device. When omitted, outputs are returned to the
    first tensor device observed in the input tree for each call.
  :param non_blocking: Whether visible tensor transfers should request non-blocking copies.
  :param async_transfer: Whether module state onload/offload should use dedicated CUDA onload and
    offload copy stream pools.
  :param keep_activations_onload_device: Whether the root module forward should keep intermediate
    activations on the onload device and only move the final root output back to the caller-
    visible device. This is the runtime behavior toggle, not a parameter-residency toggle:
    parameters are still offloaded layer-by-layer after each leaf forward. It is enabled only
    when the selected targets cover every parameterized leaf submodule in the root module.
  :param transfer_buckets: Base future-prefetch depth when async transfer is enabled. Runtime
    uses this to size the async copy-stream pools and as the base depth for optional future-
    prefetch target limiting.
  :param prefetch_limit: Whether async transfer should enable the conservative future-
    prefetch target-count limit. When enabled, runtime caps pending/ready future targets to
    ``min(4 * transfer_buckets, 8)``. When disabled, target-count limiting is off and only other
    explicit constraints such as ``max_inflight_prefetch_bytes`` can stop additional future
    prefetch.
  :param max_copy_streams: Maximum size of the async onload/offload copy stream pools. This caps
    copy-lane concurrency without reducing the logical prefetch lookahead depth.
  :param max_inflight_prefetch_bytes: Maximum total CUDA residency budget, in bytes, that async
    future-target prefetch may consume at once across both pending transfers and ready-but-not-
    yet-consumed prefetched targets. When omitted, runtime leaves byte-budget limiting disabled.
  :param persistent_buckets: How many selected targets should stay resident on the onload device
    for the full handle lifetime instead of participating in per-forward onload/offload.
  :param persistent_bins: How many evenly distributed bins should be used when selecting the
    persistent targets. A value of 1 keeps the original prefix behavior, while larger values split
    the persistent budget across multiple uniformly spaced target ranges.

  The created handle is also attached to ``root_module`` so callers can manage the lifecycle via
  ``get_layerwise_offload_handles(root_module)`` or ``remove_layerwise_offload(root_module)``
  without keeping a separate owner object.
  """

  def __init__(
    self,
    *,
    root_module: nn.Module,
    targets: list[_LayerwiseTarget],
    handles: list[Any],
    onload_device: torch.device,
    offload_device: torch.device,
    output_device: torch.device | None,
    non_blocking: bool,
    async_transfer: bool,
    keep_activations_onload_device: bool,
    transfer_buckets: int,
    prefetch_limit: bool,
    max_copy_streams: int | None,
    max_inflight_prefetch_bytes: int | None,
    persistent_buckets: int,
    persistent_bins: int,
  ) -> None:
    self.root_module = root_module
    self.targets = targets
    self.module_names = [target.name for target in targets]
    self._handles = handles
    self.onload_device = onload_device
    self.offload_device = offload_device
    self.output_device = output_device
    self.non_blocking = non_blocking
    self.async_transfer = async_transfer
    self.keep_activations_onload_device = keep_activations_onload_device
    self.transfer_buckets = transfer_buckets
    self.prefetch_limit = prefetch_limit
    self.max_copy_streams = max_copy_streams
    self.max_inflight_prefetch_bytes = max_inflight_prefetch_bytes
    self.persistent_buckets = persistent_buckets
    self.persistent_bins = persistent_bins
    self.effective_persistent_buckets = sum(1 for target in targets if target.persistent)
    self.persistent_target_spans = _collect_persistent_target_spans(targets)
    self.effective_persistent_bins = len(self.persistent_target_spans)
    self.persistent_module_names = [target.name for target in targets if target.persistent]
    resolved_max_copy_streams = (_MAX_EFFECTIVE_ASYNC_TRANSFER_BUCKETS
                                 if max_copy_streams is None else max_copy_streams)
    self.effective_max_copy_streams = (min(resolved_max_copy_streams,
                                           _MAX_EFFECTIVE_ASYNC_TRANSFER_BUCKETS, transfer_buckets)
                                       if async_transfer else transfer_buckets)
    if (async_transfer and max_copy_streams is not None
        and self.effective_max_copy_streams < resolved_max_copy_streams):
      logger.warning(
        "Clamping layerwise async copy streams from %d to %d to avoid excessive copy-lane "
        "concurrency.",
        resolved_max_copy_streams,
        self.effective_max_copy_streams,
      )
    self.effective_transfer_buckets = (min(4 * transfer_buckets, _MAX_FUTURE_PREFETCH_WINDOW)
                                       if async_transfer and prefetch_limit else None)
    self.effective_max_inflight_prefetch_bytes = max_inflight_prefetch_bytes
    self._onload_copy_streams = (
      [torch.cuda.Stream(device=onload_device)
       for _ in range(self.effective_max_copy_streams)] if async_transfer else [])
    self._offload_copy_streams = (
      [torch.cuda.Stream(device=onload_device)
       for _ in range(self.effective_max_copy_streams)] if async_transfer else [])
    self._onload_copy_stream_loads = [0 for _ in self._onload_copy_streams]
    self._offload_copy_stream_loads = [0 for _ in self._offload_copy_streams]
    self._onload_copy_stream_rr_cursor = 0
    self._offload_copy_stream_rr_cursor = 0
    self._pending_onload_targets: set[int] = set()
    self._ready_onload_targets: set[int] = set()
    self._pending_offload_targets: set[int] = set()
    self._pending_onload_residency_bytes = 0
    self._ready_onload_residency_bytes = 0
    self._root_return_devices: list[torch.device] = []
    self._removed = False
    _register_layerwise_offload_handle(root_module, self)

  def _bind_cpu_state(self, target: _LayerwiseTarget) -> None:
    for tensor_state in target.tensor_states:
      _assign_direct_state_tensor(
        target.module,
        kind=tensor_state.kind,
        name=tensor_state.name,
        tensor=tensor_state.cpu_tensor,
        requires_grad=tensor_state.requires_grad,
      )
    if target.index in self._ready_onload_targets:
      self._ready_onload_targets.discard(target.index)
      self._ready_onload_residency_bytes = max(
        0,
        self._ready_onload_residency_bytes - target.prefetch_residency_bytes,
      )
    target.resident_device = self.offload_device

  def _tracked_prefetch_target_count(self) -> int:
    return len(self._pending_onload_targets) + len(self._ready_onload_targets)

  def _tracked_prefetch_residency_bytes(self) -> int:
    return self._pending_onload_residency_bytes + self._ready_onload_residency_bytes

  def _mark_prefetched_target_ready(self, target: _LayerwiseTarget) -> None:
    if target.index in self._ready_onload_targets:
      return
    self._ready_onload_targets.add(target.index)
    self._ready_onload_residency_bytes += target.prefetch_residency_bytes

  def _consume_prefetched_target(self, target: _LayerwiseTarget) -> None:
    if target.index in self._pending_onload_targets:
      self._pending_onload_targets.discard(target.index)
      self._pending_onload_residency_bytes = max(
        0,
        self._pending_onload_residency_bytes - target.prefetch_residency_bytes,
      )
    if target.index in self._ready_onload_targets:
      self._ready_onload_targets.discard(target.index)
      self._ready_onload_residency_bytes = max(
        0,
        self._ready_onload_residency_bytes - target.prefetch_residency_bytes,
      )

  def _select_copy_stream(
    self,
    *,
    transfer_kind: str,
    copy_streams: list[torch.cuda.Stream],
    copy_stream_loads: list[int],
    rr_cursor_attr: str,
  ) -> tuple[int, torch.cuda.Stream]:
    if not copy_streams:
      raise RuntimeError(f"Async layerwise {transfer_kind} requires a CUDA copy stream pool.")

    start_index = getattr(self, rr_cursor_attr)
    selected_index = start_index
    selected_load: int | None = None
    for offset in range(len(copy_streams)):
      stream_index = (start_index + offset) % len(copy_streams)
      stream_load = copy_stream_loads[stream_index]
      if selected_load is None or stream_load < selected_load:
        selected_index = stream_index
        selected_load = stream_load
        if stream_load == 0:
          break

    setattr(self, rr_cursor_attr, (selected_index + 1) % len(copy_streams))
    copy_stream_loads[selected_index] += 1
    return selected_index, copy_streams[selected_index]

  def _select_onload_copy_stream(self) -> tuple[int, torch.cuda.Stream]:
    return self._select_copy_stream(
      transfer_kind="onload",
      copy_streams=self._onload_copy_streams,
      copy_stream_loads=self._onload_copy_stream_loads,
      rr_cursor_attr="_onload_copy_stream_rr_cursor",
    )

  def _select_offload_copy_stream(self) -> tuple[int, torch.cuda.Stream]:
    return self._select_copy_stream(
      transfer_kind="offload",
      copy_streams=self._offload_copy_streams,
      copy_stream_loads=self._offload_copy_stream_loads,
      rr_cursor_attr="_offload_copy_stream_rr_cursor",
    )

  def _release_copy_stream(
    self,
    stream_index: int | None,
    *,
    copy_stream_loads: list[int],
  ) -> None:
    if stream_index is None:
      return
    if 0 <= stream_index < len(copy_stream_loads) and copy_stream_loads[stream_index] > 0:
      copy_stream_loads[stream_index] -= 1

  def _release_onload_copy_stream(self, stream_index: int | None) -> None:
    self._release_copy_stream(
      stream_index,
      copy_stream_loads=self._onload_copy_stream_loads,
    )

  def _release_offload_copy_stream(self, stream_index: int | None) -> None:
    self._release_copy_stream(
      stream_index,
      copy_stream_loads=self._offload_copy_stream_loads,
    )

  def _clear_pending_onload(self, target: _LayerwiseTarget) -> None:
    target.pending_onload_event = None
    was_tracked_pending = target.index in self._pending_onload_targets
    if was_tracked_pending:
      self._pending_onload_targets.discard(target.index)
      self._pending_onload_residency_bytes = max(
        0,
        self._pending_onload_residency_bytes - target.prefetch_residency_bytes,
      )
      if target.resident_device == self.onload_device and not target.return_devices:
        self._mark_prefetched_target_ready(target)
    self._release_onload_copy_stream(target.pending_onload_stream_index)
    target.pending_onload_stream_index = None

  def _clear_pending_offload(self, target: _LayerwiseTarget) -> None:
    target.pending_offload_event = None
    self._pending_offload_targets.discard(target.index)
    self._release_offload_copy_stream(target.pending_offload_stream_index)
    target.pending_offload_stream_index = None

  def _drain_pending_onload(self, target: _LayerwiseTarget) -> None:
    if target.pending_onload_event is None:
      return
    target.pending_onload_event.synchronize()
    self._clear_pending_onload(target)

  def _drain_pending_offload(self, target: _LayerwiseTarget) -> None:
    if target.pending_offload_event is None:
      return
    target.pending_offload_event.synchronize()
    self._clear_pending_offload(target)

  def _reap_completed_onload(self, target: _LayerwiseTarget) -> None:
    if target.pending_onload_event is None:
      return
    if not target.pending_onload_event.query():
      return
    self._clear_pending_onload(target)

  def _reap_completed_offload(self, target: _LayerwiseTarget) -> None:
    if target.pending_offload_event is None:
      return
    if not target.pending_offload_event.query():
      return
    self._clear_pending_offload(target)

  def _reap_completed_transfers(self) -> None:
    for target in self.targets:
      self._reap_completed_onload(target)
      self._reap_completed_offload(target)

  def _drain_target_transfers(self, target: _LayerwiseTarget) -> None:
    self._drain_pending_onload(target)
    self._drain_pending_offload(target)

  def _tensor_state_requires_sync_back(
    self,
    target: _LayerwiseTarget,
    tensor_state: _LayerwiseTensorState,
  ) -> bool:
    # Cache-DiT only serves inference/eval workloads, so parameter tensors are treated as
    # immutable after forward in eval mode and can keep using the pinned CPU mirror that was
    # prepared ahead of time. Buffers still sync back because custom inference modules may update
    # them even when training is disabled.
    if tensor_state.kind != "parameter":
      return True
    return target.module.training

  def _materialize_onload_sync(self, target: _LayerwiseTarget) -> None:
    self._drain_target_transfers(target)
    for tensor_state in target.tensor_states:
      onload_tensor = tensor_state.cpu_tensor.to(device=self.onload_device, non_blocking=False)
      _assign_direct_state_tensor(
        target.module,
        kind=tensor_state.kind,
        name=tensor_state.name,
        tensor=onload_tensor,
        requires_grad=tensor_state.requires_grad,
      )
    target.resident_device = self.onload_device

  def _materialize_offload_sync(self, target: _LayerwiseTarget) -> None:
    self._drain_target_transfers(target)
    for tensor_state in target.tensor_states:
      current_tensor = _get_direct_state_tensor(
        target.module,
        kind=tensor_state.kind,
        name=tensor_state.name,
      )
      if (current_tensor.device.type != "cpu"
          and self._tensor_state_requires_sync_back(target, tensor_state)):
        tensor_state.cpu_tensor.copy_(current_tensor, non_blocking=False)
    self._bind_cpu_state(target)

  def _target_requires_sync_back(self, target: _LayerwiseTarget) -> bool:
    return any(
      self._tensor_state_requires_sync_back(target, tensor_state)
      for tensor_state in target.tensor_states)

  def _schedule_target_onload(self, target: _LayerwiseTarget, *, allow_wait: bool) -> None:
    if not self.async_transfer:
      return
    self._reap_completed_transfers()
    if target.pending_onload_event is not None or target.resident_device == self.onload_device:
      return
    if target.pending_offload_event is not None:
      if not allow_wait:
        return
      self._drain_pending_offload(target)

    stream_index, copy_stream = self._select_onload_copy_stream()

    with torch.cuda.stream(copy_stream):
      # Onload prefetch does not wait on the current compute stream. The consumer stream
      # establishes the real readiness dependency in _await_target_onload() via wait_event(),
      # so issuing the H2D copy immediately gives future targets a wider overlap window.
      for tensor_state in target.tensor_states:
        onload_tensor = tensor_state.cpu_tensor.to(device=self.onload_device, non_blocking=True)
        _assign_direct_state_tensor(
          target.module,
          kind=tensor_state.kind,
          name=tensor_state.name,
          tensor=onload_tensor,
          requires_grad=tensor_state.requires_grad,
        )
      event = torch.cuda.Event()
      event.record(copy_stream)

    logger.debug(
      "Layerwise async onload scheduled for %s on onload copy stream[%d].",
      target.name or _ROOT_TARGET_NAME,
      stream_index,
    )
    target.pending_onload_event = event
    target.pending_onload_stream_index = stream_index
    target.resident_device = self.onload_device
    self._pending_onload_targets.add(target.index)
    self._pending_onload_residency_bytes += target.prefetch_residency_bytes

  def _await_target_onload(self, target: _LayerwiseTarget) -> None:
    self._reap_completed_onload(target)
    if target.pending_onload_event is None:
      return
    event = target.pending_onload_event
    current_stream = torch.cuda.current_stream(device=self.onload_device)
    current_stream.wait_event(event)
    for tensor_state in target.tensor_states:
      onload_tensor = _get_direct_state_tensor(
        target.module,
        kind=tensor_state.kind,
        name=tensor_state.name,
      )
      onload_tensor.record_stream(current_stream)
    self._clear_pending_onload(target)

  def _schedule_target_offload(self, target: _LayerwiseTarget) -> None:
    if not self.async_transfer:
      return
    if target.persistent:
      return
    self._reap_completed_transfers()
    if target.pending_offload_event is not None or target.resident_device == self.offload_device:
      return
    if target.pending_onload_event is not None:
      self._await_target_onload(target)

    target_requires_sync_back = self._target_requires_sync_back(target)
    inflight_gpu_tensors = [
      _get_direct_state_tensor(
        target.module,
        kind=tensor_state.kind,
        name=tensor_state.name,
      ) for tensor_state in target.tensor_states
    ]

    if not target_requires_sync_back:
      current_stream = torch.cuda.current_stream(device=self.onload_device)
      for current_tensor in inflight_gpu_tensors:
        # The module state is rebound to the CPU mirror right after this branch, so keep the
        # current GPU storage alive on the execution stream until all queued kernels finish.
        current_tensor.record_stream(current_stream)
      event = torch.cuda.Event()
      event.record(current_stream)
      logger.debug(
        "Layerwise async offload skipped D2H sync-back for %s; tracking lifetime on current "
        "stream.",
        target.name or _ROOT_TARGET_NAME,
      )
      self._bind_cpu_state(target)
      target.pending_offload_event = event
      target.pending_offload_stream_index = None
      self._pending_offload_targets.add(target.index)
      return

    stream_index, copy_stream = self._select_offload_copy_stream()

    current_stream = torch.cuda.current_stream(device=self.onload_device)
    copy_stream.wait_stream(current_stream)
    with torch.cuda.stream(copy_stream):
      for tensor_state, current_tensor in zip(target.tensor_states, inflight_gpu_tensors):
        current_tensor.record_stream(copy_stream)
        tensor_state.cpu_tensor.copy_(current_tensor, non_blocking=True)
      event = torch.cuda.Event()
      event.record(copy_stream)

    logger.debug(
      "Layerwise async offload scheduled for %s on offload copy stream[%d].",
      target.name or _ROOT_TARGET_NAME,
      stream_index,
    )
    self._bind_cpu_state(target)
    target.pending_offload_event = event
    target.pending_offload_stream_index = stream_index
    self._pending_offload_targets.add(target.index)

  def _prefetch_bucket_targets(self, target: _LayerwiseTarget) -> None:
    if not self.async_transfer:
      return
    self._reap_completed_transfers()
    available_target_budget = None
    if self.effective_transfer_buckets is not None:
      available_target_budget = (self.effective_transfer_buckets -
                                 self._tracked_prefetch_target_count())
      if available_target_budget <= 0:
        return
    scheduled = 0
    start_index = target.index + 1
    for next_target in self.targets[start_index:]:
      if available_target_budget is not None and scheduled >= available_target_budget:
        break
      if next_target.persistent:
        continue
      if next_target.return_devices:
        continue
      if next_target.pending_onload_event is not None:
        continue
      if next_target.index in self._ready_onload_targets:
        continue
      if next_target.pending_offload_event is not None:
        continue
      if next_target.resident_device == self.onload_device:
        continue
      if (self.effective_max_inflight_prefetch_bytes is not None
          and self._tracked_prefetch_residency_bytes() + next_target.prefetch_residency_bytes
          > self.effective_max_inflight_prefetch_bytes):
        continue
      self._schedule_target_onload(next_target, allow_wait=False)
      if next_target.pending_onload_event is not None:
        scheduled += 1

  def remove(self, *, offload: bool = True) -> None:
    """Remove all hooks and optionally offload registered targets immediately.

    :param offload: Whether to move registered targets back to the offload device after the hooks
        are removed.
    """

    if self._removed:
      return

    for handle in self._handles:
      handle.remove()
    self._handles.clear()

    if offload:
      for target in self.targets:
        if self.async_transfer or target.tensor_states:
          self._materialize_offload_sync(target)
        else:
          _move_module_state(
            target.module,
            self.offload_device,
            non_blocking=self.non_blocking,
          )
          target.resident_device = self.offload_device
        target.return_devices.clear()
    elif self.async_transfer:
      for target in self.targets:
        self._drain_target_transfers(target)

    self._pending_onload_targets.clear()
    self._ready_onload_targets.clear()
    self._pending_offload_targets.clear()
    self._pending_onload_residency_bytes = 0
    self._ready_onload_residency_bytes = 0

    self._removed = True
    _detach_layerwise_offload_handle(self.root_module, self)


def _prepare_async_target(module: nn.Module, *, index: int, name: str) -> _LayerwiseTarget:
  if _module_uses_meta_tensors(module):
    raise ValueError(
      "Layerwise offload does not support modules with meta tensors. Remove external meta-based "
      "offload hooks or materialize the module before applying cache_dit.offload.")

  tensor_states: list[_LayerwiseTensorState] = []
  for kind, tensor_name, tensor, requires_grad in _iter_direct_state_entries(module):
    if tensor.device.type == "cpu":
      cpu_tensor = tensor.detach()
      if not cpu_tensor.is_pinned():
        cpu_tensor = cpu_tensor.pin_memory()
    else:
      cpu_tensor = torch.empty_like(tensor, device="cpu", pin_memory=True)
      cpu_tensor.copy_(tensor, non_blocking=False)
    tensor_states.append(
      _LayerwiseTensorState(
        kind=kind,
        name=tensor_name,
        requires_grad=requires_grad,
        cpu_tensor=cpu_tensor,
      ))
    if tensor.device.type == "cpu" and cpu_tensor.data_ptr() != tensor.data_ptr():
      _assign_direct_state_tensor(
        module,
        kind=kind,
        name=tensor_name,
        tensor=cpu_tensor,
        requires_grad=requires_grad,
      )

  target = _LayerwiseTarget(
    name=name,
    module=module,
    index=index,
    tensor_states=tensor_states,
    resident_device=_infer_module_state_device(module),
  )
  target.prefetch_residency_bytes = _target_prefetch_residency_bytes(target)
  return target


def _get_registered_layerwise_offload_handles(
  root_module: nn.Module, ) -> list[LayerwiseOffloadHandle]:
  handles = getattr(root_module, _LAYERWISE_OFFLOAD_HANDLES_ATTR, None)
  if handles is None:
    return []
  if isinstance(handles, list):
    return [handle for handle in handles if isinstance(handle, LayerwiseOffloadHandle)]
  if isinstance(handles, LayerwiseOffloadHandle):
    return [handles]
  return []


def _register_layerwise_offload_handle(
  root_module: nn.Module,
  handle: LayerwiseOffloadHandle,
) -> None:
  handles = _get_registered_layerwise_offload_handles(root_module)
  if any(existing_handle is handle for existing_handle in handles):
    return
  handles.append(handle)
  setattr(root_module, _LAYERWISE_OFFLOAD_HANDLES_ATTR, handles)


def _detach_layerwise_offload_handle(
  root_module: nn.Module,
  handle: LayerwiseOffloadHandle,
) -> None:
  handles = [
    existing_handle for existing_handle in _get_registered_layerwise_offload_handles(root_module)
    if existing_handle is not handle and not existing_handle._removed
  ]
  if handles:
    setattr(root_module, _LAYERWISE_OFFLOAD_HANDLES_ATTR, handles)
  elif hasattr(root_module, _LAYERWISE_OFFLOAD_HANDLES_ATTR):
    delattr(root_module, _LAYERWISE_OFFLOAD_HANDLES_ATTR)


def get_layerwise_offload_handles(root_module: Any, ) -> tuple[LayerwiseOffloadHandle, ...]:
  """Return cache-dit layerwise offload handles attached to a module or pipeline-like object.

  :param root_module: Root module, DiffusionPipeline, or BlockAdapter that may own layerwise offload
    hooks.
  :returns: Attached layerwise offload handles in registration order.
  """

  resolved_root_specs = _resolve_public_offload_root_specs(
    root_module,
    module_names=None,
    module_filter=None,
  )
  handles: list[LayerwiseOffloadHandle] = []
  seen_handle_ids: set[int] = set()
  for _name, resolved_root_module, _module_names, _module_filter in resolved_root_specs:
    for handle in _get_registered_layerwise_offload_handles(resolved_root_module):
      if id(handle) in seen_handle_ids:
        continue
      seen_handle_ids.add(id(handle))
      handles.append(handle)
  return tuple(handles)


def remove_layerwise_offload(
  root_module: Any,
  *,
  offload: bool = True,
) -> int:
  """Remove all cache-dit layerwise offload hooks attached to a module or pipeline-like object.

  :param root_module: Root module, DiffusionPipeline, or BlockAdapter that owns the layerwise
    offload hooks.
  :param offload: Whether to move registered targets back to the offload device.
  :returns: Number of attached handles that were removed.
  """

  handles = list(get_layerwise_offload_handles(root_module))
  for handle in handles:
    handle.remove(offload=offload)
  return len(handles)


def _apply_public_layerwise_offload(
  root_module: Any,
  *,
  module_names: Iterable[str] | None = None,
  module_filter: Callable[..., bool] | None = None,
  onload_device: torch.device | str,
  offload_device: torch.device | str = "cpu",
  output_device: torch.device | str | None = None,
  non_blocking: bool = False,
  eager_offload: bool = True,
  async_transfer: bool = False,
  transfer_buckets: int = 1,
  prefetch_limit: bool = False,
  max_copy_streams: int | None = None,
  max_inflight_prefetch_bytes: int | str | None = None,
  persistent_buckets: int = 0,
  persistent_bins: int = 1,
) -> LayerwiseOffloadHandle | LayerwiseOffloadHandleGroup:
  resolved_root_specs = _resolve_public_offload_root_specs(
    root_module,
    module_names=module_names,
    module_filter=module_filter,
  )
  if not resolved_root_specs:
    raise ValueError("Layerwise offload did not match any root modules.")

  handles: list[LayerwiseOffloadHandle] = []
  root_names: list[str] = []
  for root_name, resolved_root_module, resolved_module_names, resolved_module_filter in resolved_root_specs:
    handles.append(
      _apply_layerwise_offload(
        resolved_root_module,
        module_names=resolved_module_names,
        module_filter=resolved_module_filter,
        onload_device=onload_device,
        offload_device=offload_device,
        output_device=output_device,
        non_blocking=non_blocking,
        eager_offload=eager_offload,
        async_transfer=async_transfer,
        transfer_buckets=transfer_buckets,
        prefetch_limit=prefetch_limit,
        max_copy_streams=max_copy_streams,
        max_inflight_prefetch_bytes=max_inflight_prefetch_bytes,
        persistent_buckets=persistent_buckets,
        persistent_bins=persistent_bins,
      ))
    root_names.append(root_name)

  if len(handles) == 1:
    return handles[0]
  return LayerwiseOffloadHandleGroup(
    source=root_module,
    handles=handles,
    root_names=root_names,
  )


def _apply_layerwise_offload(
  root_module: nn.Module,
  *,
  module_names: Iterable[str] | None = None,
  module_filter: Callable[..., bool] | None = None,
  onload_device: torch.device | str,
  offload_device: torch.device | str = "cpu",
  output_device: torch.device | str | None = None,
  non_blocking: bool = False,
  eager_offload: bool = True,
  async_transfer: bool = False,
  transfer_buckets: int = 1,
  prefetch_limit: bool = False,
  max_copy_streams: int | None = None,
  max_inflight_prefetch_bytes: int | str | None = None,
  persistent_buckets: int = 0,
  persistent_bins: int = 1,
) -> LayerwiseOffloadHandle:
  """Attach layerwise onload/offload hooks to selected submodules.

  :param root_module: Root module that owns the selected submodules.
  :param module_names: Optional explicit submodule names. When omitted, ``module_filter`` or the
    default leaf-module selection is used.
  :param module_filter: Optional predicate used to select submodules when ``module_names`` is not
    provided.
  :param onload_device: Device used during the selected submodule forward.
  :param offload_device: Residency device after the selected submodule forward.
  :param output_device: Optional fixed output device. When omitted, each submodule returns outputs
    to the first tensor device seen in that call's input tree.
  :param non_blocking: Whether visible tensor transfers should request non-blocking copies.
  :param eager_offload: Whether to move selected submodules to the offload device immediately.
  :param async_transfer: Whether module state onload/offload should use dedicated CUDA onload and
    offload copy stream pools. The async path currently supports CUDA onload plus CPU offload
    only.
  :param transfer_buckets: Base future-prefetch depth when ``async_transfer`` is enabled.
    Runtime uses this to size the async copy-stream pools and as the base depth for optional
    future-prefetch target limiting.
  :param prefetch_limit: Whether async transfer should enable the conservative future-
    prefetch target-count limit. When enabled, runtime caps pending/ready future targets to
    ``min(4 * transfer_buckets, 8)``. When disabled, target-count limiting is off and only other
    explicit constraints such as ``max_inflight_prefetch_bytes`` can stop additional future
    prefetch.
  :param max_copy_streams: Maximum number of async onload/offload copy streams. When omitted,
    runtime uses the requested transfer_buckets but still applies a hard safety cap.
  :param max_inflight_prefetch_bytes: Maximum total future-target CUDA residency allowed for
    async prefetch at once across both pending transfers and ready-but-not-yet-consumed
    prefetched targets. Accepts either an integer byte count or a string such as ``512MiB`` or
    ``4GiB``. When omitted, runtime does not apply an implicit byte-budget cap.
  :param persistent_buckets: How many selected targets should stay resident on the onload device
    for the full handle lifetime instead of participating in per-forward onload/offload.
  :param persistent_bins: How many evenly distributed bins should be used when selecting the
    persistent targets. A value of 1 keeps the original prefix behavior, while larger values split
    the persistent budget across multiple uniformly spaced target ranges.
  :returns: A handle that can remove the registered hooks. The same handle is also attached to
    ``root_module`` and can be removed later with ``remove_layerwise_offload(root_module)``.
  """

  resolved_onload_device = torch.device(onload_device)
  resolved_offload_device = torch.device(offload_device)
  resolved_output_device = None if output_device is None else torch.device(output_device)
  if isinstance(transfer_buckets, bool) or not isinstance(transfer_buckets, int):
    raise TypeError(f"transfer_buckets must be an int, got {type(transfer_buckets)}.")
  if transfer_buckets < 1:
    raise ValueError(f"transfer_buckets must be >= 1, got {transfer_buckets}.")
  if not isinstance(prefetch_limit, bool):
    raise TypeError(f"prefetch_limit must be a bool, got {type(prefetch_limit)}.")
  if max_copy_streams is not None:
    if isinstance(max_copy_streams, bool) or not isinstance(max_copy_streams, int):
      raise TypeError(f"max_copy_streams must be an int or None, got {type(max_copy_streams)}.")
    if max_copy_streams < 1:
      raise ValueError(f"max_copy_streams must be >= 1, got {max_copy_streams}.")
  max_inflight_prefetch_bytes = _normalize_max_inflight_prefetch_bytes_arg(
    max_inflight_prefetch_bytes)
  if isinstance(persistent_buckets, bool) or not isinstance(persistent_buckets, int):
    raise TypeError(f"persistent_buckets must be an int, got {type(persistent_buckets)}.")
  if persistent_buckets < 0:
    raise ValueError(f"persistent_buckets must be >= 0, got {persistent_buckets}.")
  if isinstance(persistent_bins, bool) or not isinstance(persistent_bins, int):
    raise TypeError(f"persistent_bins must be an int, got {type(persistent_bins)}.")
  if persistent_bins < 1:
    raise ValueError(f"persistent_bins must be >= 1, got {persistent_bins}.")
  if async_transfer:
    if resolved_onload_device.type != "cuda":
      raise ValueError("async_transfer requires a CUDA onload device.")
    if resolved_offload_device.type != "cpu":
      raise ValueError("async_transfer currently supports CPU offload only.")
    if not torch.cuda.is_available():
      raise RuntimeError("async_transfer requires CUDA, but CUDA is unavailable.")

  resolved_targets = _resolve_target_modules(
    root_module,
    module_names=module_names,
    module_filter=module_filter,
  )
  if (not resolved_targets and module_names is None and module_filter is None
      and _module_has_direct_state(root_module)):
    resolved_targets = [("", root_module)]
  if not resolved_targets:
    raise ValueError("Layerwise offload did not match any target submodules.")
  has_full_parameterized_leaf_coverage = _targets_cover_all_parameterized_leaf_modules(
    root_module,
    resolved_targets=resolved_targets,
  )
  persistent_target_spans = _resolve_persistent_target_spans(
    target_count=len(resolved_targets),
    persistent_buckets=persistent_buckets,
    persistent_bins=persistent_bins,
  )
  persistent_target_indices = {
    target_index
    for span_start, span_end in persistent_target_spans
    for target_index in range(span_start, span_end)
  }
  use_cpu_tensor_state_mirror = resolved_offload_device.type == "cpu"

  targets = []
  for index, (name, submodule) in enumerate(resolved_targets):
    is_persistent = index in persistent_target_indices
    if use_cpu_tensor_state_mirror:
      target = _prepare_async_target(submodule, index=index, name=name)
      target.persistent = is_persistent
    else:
      target = _LayerwiseTarget(
        name=name,
        module=submodule,
        index=index,
        persistent=is_persistent,
        resident_device=_infer_module_state_device(submodule),
      )
    targets.append(target)

  handles: list[Any] = []
  handle = LayerwiseOffloadHandle(
    root_module=root_module,
    targets=targets,
    handles=handles,
    onload_device=resolved_onload_device,
    offload_device=resolved_offload_device,
    output_device=resolved_output_device,
    non_blocking=non_blocking,
    async_transfer=async_transfer,
    keep_activations_onload_device=has_full_parameterized_leaf_coverage,
    transfer_buckets=transfer_buckets,
    prefetch_limit=prefetch_limit,
    max_copy_streams=max_copy_streams,
    max_inflight_prefetch_bytes=max_inflight_prefetch_bytes,
    persistent_buckets=persistent_buckets,
    persistent_bins=persistent_bins,
  )

  # Full parameterized-leaf coverage is the precondition; the actual enabled behavior is keeping
  # intermediate activations on the onload device across the whole root forward.
  if has_full_parameterized_leaf_coverage:

    def root_pre_hook(
      module: nn.Module,
      args: tuple[Any, ...],
      kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
      return_device = resolved_output_device
      if return_device is None:
        return_device = (_find_first_tensor_device(args) or _find_first_tensor_device(kwargs)
                         or resolved_offload_device)
      handle._root_return_devices.append(return_device)
      args = _move_tree_to_device(args, resolved_onload_device, non_blocking=non_blocking)
      kwargs = _move_tree_to_device(kwargs, resolved_onload_device, non_blocking=non_blocking)
      return args, kwargs

    def root_post_hook(
      module: nn.Module,
      args: tuple[Any, ...],
      kwargs: dict[str, Any],
      output: Any,
    ) -> Any:
      return_device = handle._root_return_devices.pop(
      ) if handle._root_return_devices else resolved_offload_device
      return _move_tree_to_device(output, return_device, non_blocking=non_blocking)

    root_pre_kwargs: dict[str, Any] = {"with_kwargs": True}
    if _FORWARD_PRE_HOOK_SUPPORTS_PREPEND:
      root_pre_kwargs["prepend"] = True
    handles.append(root_module.register_forward_pre_hook(root_pre_hook, **root_pre_kwargs))

    root_post_kwargs: dict[str, Any] = {"with_kwargs": True}
    if _FORWARD_HOOK_SUPPORTS_ALWAYS_CALL:
      root_post_kwargs["always_call"] = True
    if _FORWARD_HOOK_SUPPORTS_PREPEND:
      root_post_kwargs["prepend"] = False
    handles.append(root_module.register_forward_hook(root_post_hook, **root_post_kwargs))

  for target in targets:

    def pre_hook(
      module: nn.Module,
      args: tuple[Any, ...],
      kwargs: dict[str, Any],
      *,
      target: _LayerwiseTarget = target,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
      if async_transfer:
        if not target.return_devices:
          is_prefetched_target = (target.pending_onload_event is not None
                                  or target.index in handle._ready_onload_targets)
          if is_prefetched_target:
            # The current target was already counted inside the future-prefetch window. Consume
            # that window slot before trying to refill lookahead so the widened prefetch window
            # keeps counting ready-but-not-yet-used targets until execution actually reaches them.
            handle._consume_prefetched_target(target)
            handle._prefetch_bucket_targets(target)
            if target.pending_onload_event is not None:
              # The current target was prefetched earlier on a copy stream, so keep the original
              # overlap path and only wait for readiness right before execution.
              handle._await_target_onload(target)
          else:
            if target.resident_device != resolved_onload_device:
              # The current target was not prefetched. Materialize it synchronously on the active
              # execution path so allocator reuse stays local to the compute stream; async copy
              # streams are reserved for future-target prefetch where overlap is actually
              # possible.
              handle._materialize_onload_sync(target)
            handle._prefetch_bucket_targets(target)
      elif not target.return_devices and target.resident_device != resolved_onload_device:
        if target.tensor_states:
          handle._materialize_onload_sync(target)
        else:
          _move_module_state(
            module,
            resolved_onload_device,
            non_blocking=non_blocking,
          )
          target.resident_device = resolved_onload_device

      if handle.keep_activations_onload_device:
        return_device = resolved_onload_device
      else:
        return_device = resolved_output_device
      if return_device is None:
        return_device = (_find_first_tensor_device(args) or _find_first_tensor_device(kwargs)
                         or resolved_offload_device)
      target.return_devices.append(return_device)
      args = _move_tree_to_device(args, resolved_onload_device, non_blocking=non_blocking)
      kwargs = _move_tree_to_device(kwargs, resolved_onload_device, non_blocking=non_blocking)
      return args, kwargs

    def post_hook(
      module: nn.Module,
      args: tuple[Any, ...],
      kwargs: dict[str, Any],
      output: Any,
      *,
      target: _LayerwiseTarget = target,
    ) -> Any:
      return_device = target.return_devices.pop(
      ) if target.return_devices else resolved_offload_device
      output = _move_tree_to_device(output, return_device, non_blocking=non_blocking)
      if not target.return_devices and not target.persistent:
        if async_transfer:
          handle._schedule_target_offload(target)
        else:
          if target.tensor_states:
            handle._materialize_offload_sync(target)
          else:
            _move_module_state(
              module,
              resolved_offload_device,
              non_blocking=non_blocking,
            )
            target.resident_device = resolved_offload_device
      return output

    pre_kwargs: dict[str, Any] = {"with_kwargs": True}
    if _FORWARD_PRE_HOOK_SUPPORTS_PREPEND:
      pre_kwargs["prepend"] = True
    handles.append(target.module.register_forward_pre_hook(pre_hook, **pre_kwargs))

    post_kwargs: dict[str, Any] = {"with_kwargs": True}
    if _FORWARD_HOOK_SUPPORTS_ALWAYS_CALL:
      post_kwargs["always_call"] = True
    if _FORWARD_HOOK_SUPPORTS_PREPEND:
      post_kwargs["prepend"] = False
    handles.append(target.module.register_forward_hook(post_hook, **post_kwargs))

  # Materialize persistent targets immediately so the first inference already benefits from
  # their CUDA residency instead of paying the first-use onload cost inside the forward path.
  for target in targets:
    if not target.persistent:
      continue
    if async_transfer:
      handle._materialize_onload_sync(target)
    elif target.resident_device != resolved_onload_device:
      if target.tensor_states:
        handle._materialize_onload_sync(target)
      else:
        _move_module_state(
          target.module,
          resolved_onload_device,
          non_blocking=non_blocking,
        )
        target.resident_device = resolved_onload_device

  if eager_offload:
    for target in targets:
      if target.persistent:
        continue
      if async_transfer:
        handle._materialize_offload_sync(target)
      else:
        if target.tensor_states:
          handle._materialize_offload_sync(target)
        else:
          _move_module_state(
            target.module,
            resolved_offload_device,
            non_blocking=non_blocking,
          )
          target.resident_device = resolved_offload_device

  logger.info(
    "Enabled layerwise offload for %d submodules from %s to %s%s%s%s.",
    len(targets),
    resolved_offload_device,
    resolved_onload_device,
    " with async transfer" if async_transfer else "",
    (f" with {handle.effective_persistent_buckets} persistent targets"
     if handle.effective_persistent_buckets else ""),
    (f" across {handle.effective_persistent_bins} bins"
     if handle.effective_persistent_bins > 1 else ""),
  )
  return handle


def layerwise_offload(
  root_module: Any,
  *,
  module_names: Iterable[str] | None = None,
  module_filter: Callable[..., bool] | None = None,
  onload_device: torch.device | str = "cuda",
  offload_device: torch.device | str = "cpu",
  output_device: torch.device | str | None = None,
  non_blocking: bool = False,
  eager_offload: bool = True,
  async_transfer: bool = False,
  transfer_buckets: int = 1,
  prefetch_limit: bool = False,
  max_copy_streams: int | None = None,
  max_inflight_prefetch_bytes: int | str | None = None,
  persistent_buckets: int = 0,
  persistent_bins: int = 1,
) -> LayerwiseOffloadHandle | LayerwiseOffloadHandleGroup:
  """Public wrapper for generic submodule-level onload/offload hooks.

  :param root_module: Root module, DiffusionPipeline, or BlockAdapter that owns the selected
    submodules.
  :param module_names: Optional explicit submodule names. For module inputs, these names are
    resolved inside the module as before. For pipeline-like inputs, they are resolved as explicit
    component/module paths and each resolved module becomes an offload root.
  :param module_filter: Optional predicate used to select submodules when ``module_names`` is not
    provided. For pipeline-like inputs, this filter is applied to discovered root components.
  :param onload_device: Device used during the selected submodule forward.
  :param offload_device: Residency device after the selected submodule forward.
  :param output_device: Optional fixed output device for all selected submodules.
  :param non_blocking: Whether visible tensor transfers should request non-blocking copies.
  :param eager_offload: Whether to move selected submodules to the offload device immediately.
  :param async_transfer: Whether module state onload/offload should use dedicated CUDA onload and
    offload copy stream pools. The async path currently supports CUDA onload plus CPU offload
    only.
  :param transfer_buckets: How many future targets should be prefetched when ``async_transfer``
    is enabled.
  :param prefetch_limit: Whether to enable the conservative future-prefetch target-count
    limit of ``min(4 * transfer_buckets, 8)``.
  :param max_copy_streams: Maximum number of async onload/offload copy streams.
  :param max_inflight_prefetch_bytes: Maximum total future-target CUDA residency allowed for
    async prefetch at once. Accepts either an integer byte count or a string such as ``512MiB``
    or ``4GiB``.
  :param persistent_buckets: How many selected targets should stay resident on the onload device
    for the full handle lifetime.
  :param persistent_bins: How many evenly distributed bins should be used when selecting the
    persistent targets. A value of 1 keeps the original prefix behavior.
  :returns: A single handle for module inputs or a grouped handle for pipeline-like inputs. The
    underlying per-root handles are also attached to their resolved root modules and can be
    removed later with ``remove_layerwise_offload(root_module)``.
  """

  return _apply_public_layerwise_offload(
    root_module,
    module_names=module_names,
    module_filter=module_filter,
    onload_device=onload_device,
    offload_device=offload_device,
    output_device=output_device,
    non_blocking=non_blocking,
    eager_offload=eager_offload,
    async_transfer=async_transfer,
    transfer_buckets=transfer_buckets,
    prefetch_limit=prefetch_limit,
    max_copy_streams=max_copy_streams,
    max_inflight_prefetch_bytes=max_inflight_prefetch_bytes,
    persistent_buckets=persistent_buckets,
    persistent_bins=persistent_bins,
  )


def layerwise_cpu_offload(
  root_module: Any,
  *,
  module_names: Iterable[str] | None = None,
  module_filter: Callable[..., bool] | None = None,
  onload_device: torch.device | str = "cuda",
  output_device: torch.device | str | None = None,
  non_blocking: bool = False,
  eager_offload: bool = True,
  async_transfer: bool = False,
  transfer_buckets: int = 1,
  prefetch_limit: bool = False,
  max_copy_streams: int | None = None,
  max_inflight_prefetch_bytes: int | str | None = None,
  persistent_buckets: int = 0,
  persistent_bins: int = 1,
) -> LayerwiseOffloadHandle | LayerwiseOffloadHandleGroup:
  """Convenience wrapper for cache-dit layerwise CPU offload.

  :param root_module: Root module, DiffusionPipeline, or BlockAdapter that owns the selected
    submodules.
  :param module_names: Optional explicit submodule names. For pipeline-like inputs, these names
    are resolved as explicit component/module paths and each resolved module becomes an offload
    root.
  :param module_filter: Optional predicate used to select submodules when ``module_names`` is not
    provided. For pipeline-like inputs, this filter is applied to discovered root components.
  :param onload_device: Device used during the selected submodule forward.
  :param output_device: Optional fixed output device for all selected submodules.
  :param non_blocking: Whether visible tensor transfers should request non-blocking copies.
  :param eager_offload: Whether to move selected submodules to CPU immediately.
  :param async_transfer: Whether module state onload/offload should use dedicated CUDA onload and
    offload copy stream pools.
  :param transfer_buckets: How many future targets should be prefetched when ``async_transfer``
    is enabled.
  :param prefetch_limit: Whether to enable the conservative future-prefetch target-count
    limit of ``min(4 * transfer_buckets, 8)``.
  :param max_copy_streams: Maximum number of async onload/offload copy streams.
  :param max_inflight_prefetch_bytes: Maximum total future-target CUDA residency allowed for
    async prefetch at once. Accepts either an integer byte count or a string such as ``512MiB``
    or ``4GiB``.
  :param persistent_buckets: How many selected targets should stay resident on the onload device
    for the full handle lifetime.
  :param persistent_bins: How many evenly distributed bins should be used when selecting the
    persistent targets. A value of 1 keeps the original prefix behavior.
  :returns: A single handle for module inputs or a grouped handle for pipeline-like inputs. The
    underlying per-root handles are also attached to their resolved root modules and can be
    removed later with ``remove_layerwise_offload(root_module)``.
  """

  return _apply_public_layerwise_offload(
    root_module,
    module_names=module_names,
    module_filter=module_filter,
    onload_device=onload_device,
    offload_device=torch.device("cpu"),
    output_device=output_device,
    non_blocking=non_blocking,
    eager_offload=eager_offload,
    async_transfer=async_transfer,
    transfer_buckets=transfer_buckets,
    prefetch_limit=prefetch_limit,
    max_copy_streams=max_copy_streams,
    max_inflight_prefetch_bytes=max_inflight_prefetch_bytes,
    persistent_buckets=persistent_buckets,
    persistent_bins=persistent_bins,
  )


__all__ = [
  "LayerwiseOffloadHandle",
  "LayerwiseOffloadHandleGroup",
  "_parse_byte_size_arg",
  "_apply_layerwise_offload",
  "_find_offload_related_hf_hook",
  "get_layerwise_offload_handles",
  "layerwise_offload",
  "layerwise_cpu_offload",
  "remove_layerwise_offload",
]
