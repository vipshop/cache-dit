from __future__ import annotations

import dataclasses
import inspect
from typing import Any, Callable, Iterable

import torch
from torch import nn

from ..logger import init_logger

try:
  from accelerate.hooks import AlignDevicesHook as _AccelerateAlignDevicesHook
  from accelerate.hooks import CpuOffload as _AccelerateCpuOffload
except ImportError:
  _AccelerateAlignDevicesHook = None
  _AccelerateCpuOffload = None

logger = init_logger(__name__)

_LAYERWISE_OFFLOAD_HANDLES_ATTR = "_cache_dit_layerwise_offload_handles"

_FORWARD_HOOK_SUPPORTS_ALWAYS_CALL = "always_call" in inspect.signature(
  nn.Module.register_forward_hook).parameters
_FORWARD_HOOK_SUPPORTS_PREPEND = "prepend" in inspect.signature(
  nn.Module.register_forward_hook).parameters
_FORWARD_PRE_HOOK_SUPPORTS_PREPEND = "prepend" in inspect.signature(
  nn.Module.register_forward_pre_hook).parameters


def _map_tensors(value: Any, *, transform: Callable[[torch.Tensor], torch.Tensor]) -> Any:
  if isinstance(value, torch.Tensor):
    return transform(value)
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


@dataclasses.dataclass
class _LayerwiseTarget:
  name: str
  module: nn.Module
  return_devices: list[torch.device] = dataclasses.field(default_factory=list)


class LayerwiseOffloadHandle:
  """Registered layerwise offload hooks for a root module.

  The handle owns the forward pre/post hooks that move selected submodules to the onload device
  just in time, preserve the caller-visible tensor device for outputs, and offload the submodule
  state back after execution. This utility targets inference and calibration workloads rather than
  training-time autograd.

  :param root_module: Root module that owns the registered hooks.
  :param targets: Selected submodules that participate in layerwise offload.
  :param handles: Torch hook handles registered on each target module.
  :param onload_device: Device used during the target module forward.
  :param offload_device: Residency device used after the target module forward.
  :param output_device: Optional fixed output device. When omitted, outputs are returned to the
    first tensor device observed in the input tree for each call.
  :param non_blocking: Whether tensor transfers should request non-blocking copies.

  The created handle is also attached to `root_module` so callers can manage the lifecycle via
  `get_layerwise_offload_handles(root_module)` or `remove_layerwise_offload(root_module)` without
  keeping a separate owner object.
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
  ) -> None:
    self.root_module = root_module
    self.targets = targets
    self.module_names = [target.name for target in targets]
    self._handles = handles
    self.onload_device = onload_device
    self.offload_device = offload_device
    self.output_device = output_device
    self.non_blocking = non_blocking
    self._removed = False
    _register_layerwise_offload_handle(root_module, self)

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
        _move_module_state(
          target.module,
          self.offload_device,
          non_blocking=self.non_blocking,
        )
        target.return_devices.clear()

    self._removed = True
    _detach_layerwise_offload_handle(self.root_module, self)


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


def get_layerwise_offload_handles(root_module: nn.Module, ) -> tuple[LayerwiseOffloadHandle, ...]:
  """Return cache-dit layerwise offload handles attached to a root module.

  :param root_module: Root module that may own layerwise offload hooks.
  :returns: Attached layerwise offload handles in registration order.
  """

  return tuple(_get_registered_layerwise_offload_handles(root_module))


def remove_layerwise_offload(
  root_module: nn.Module,
  *,
  offload: bool = True,
) -> int:
  """Remove all cache-dit layerwise offload hooks attached to a root module.

  :param root_module: Root module that owns the layerwise offload hooks.
  :param offload: Whether to move registered targets back to the offload device.
  :returns: Number of attached handles that were removed.
  """

  handles = list(_get_registered_layerwise_offload_handles(root_module))
  for handle in handles:
    handle.remove(offload=offload)
  return len(handles)


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
) -> LayerwiseOffloadHandle:
  """Attach layerwise onload/offload hooks to selected submodules.

  :param root_module: Root module that owns the selected submodules.
  :param module_names: Optional explicit submodule names. When omitted, `module_filter` or the
    default leaf-module selection is used.
  :param module_filter: Optional predicate used to select submodules when `module_names` is not
    provided.
  :param onload_device: Device used during the selected submodule forward.
  :param offload_device: Residency device after the selected submodule forward.
  :param output_device: Optional fixed output device. When omitted, each submodule returns outputs
    to the first tensor device seen in that call's input tree.
  :param non_blocking: Whether transfers should request non-blocking copies.
  :param eager_offload: Whether to move selected submodules to the offload device immediately.
  :returns: A handle that can remove the registered hooks. The same handle is also attached to
    `root_module` and can be removed later with `remove_layerwise_offload(root_module)`.
  """

  resolved_onload_device = torch.device(onload_device)
  resolved_offload_device = torch.device(offload_device)
  resolved_output_device = None if output_device is None else torch.device(output_device)
  targets = [
    _LayerwiseTarget(name=name, module=submodule) for name, submodule in _resolve_target_modules(
      root_module,
      module_names=module_names,
      module_filter=module_filter,
    )
  ]

  if not targets:
    raise ValueError("Layerwise offload did not match any target submodules.")

  handles: list[Any] = []
  for target in targets:

    def pre_hook(
      module: nn.Module,
      args: tuple[Any, ...],
      kwargs: dict[str, Any],
      *,
      target: _LayerwiseTarget = target,
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
      if not target.return_devices:
        _move_module_state(
          module,
          resolved_onload_device,
          non_blocking=non_blocking,
        )

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
      if not target.return_devices:
        _move_module_state(
          module,
          resolved_offload_device,
          non_blocking=non_blocking,
        )
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

  handle = LayerwiseOffloadHandle(
    root_module=root_module,
    targets=targets,
    handles=handles,
    onload_device=resolved_onload_device,
    offload_device=resolved_offload_device,
    output_device=resolved_output_device,
    non_blocking=non_blocking,
  )

  if eager_offload:
    for target in targets:
      _move_module_state(
        target.module,
        resolved_offload_device,
        non_blocking=non_blocking,
      )

  logger.info(
    "Enabled layerwise offload for %d submodules from %s to %s.",
    len(targets),
    resolved_offload_device,
    resolved_onload_device,
  )
  return handle


def layerwise_offload(
  root_module: nn.Module,
  *,
  module_names: Iterable[str] | None = None,
  module_filter: Callable[..., bool] | None = None,
  onload_device: torch.device | str = "cuda",
  offload_device: torch.device | str = "cpu",
  output_device: torch.device | str | None = None,
  non_blocking: bool = False,
  eager_offload: bool = True,
) -> LayerwiseOffloadHandle:
  """Public wrapper for generic submodule-level onload/offload hooks.

  :param root_module: Root module that owns the selected submodules.
  :param module_names: Optional explicit submodule names.
  :param module_filter: Optional predicate used to select submodules when `module_names` is not
    provided.
  :param onload_device: Device used during the selected submodule forward.
  :param offload_device: Residency device after the selected submodule forward.
  :param output_device: Optional fixed output device for all selected submodules.
  :param non_blocking: Whether transfers should request non-blocking copies.
  :param eager_offload: Whether to move selected submodules to the offload device immediately.
  :returns: A handle that can remove the registered hooks. The same handle is also attached to
    `root_module` and can be removed later with `remove_layerwise_offload(root_module)`.
  """

  return _apply_layerwise_offload(
    root_module,
    module_names=module_names,
    module_filter=module_filter,
    onload_device=onload_device,
    offload_device=offload_device,
    output_device=output_device,
    non_blocking=non_blocking,
    eager_offload=eager_offload,
  )


def layerwise_cpu_offload(
  root_module: nn.Module,
  *,
  module_names: Iterable[str] | None = None,
  module_filter: Callable[..., bool] | None = None,
  onload_device: torch.device | str = "cuda",
  output_device: torch.device | str | None = None,
  non_blocking: bool = False,
  eager_offload: bool = True,
) -> LayerwiseOffloadHandle:
  """Convenience wrapper for cache-dit layerwise CPU offload.

  :param root_module: Root module that owns the selected submodules.
  :param module_names: Optional explicit submodule names.
  :param module_filter: Optional predicate used to select submodules when `module_names` is not
    provided.
  :param onload_device: Device used during the selected submodule forward.
  :param output_device: Optional fixed output device for all selected submodules.
  :param non_blocking: Whether transfers should request non-blocking copies.
  :param eager_offload: Whether to move selected submodules to CPU immediately.
  :returns: A handle that can remove the registered hooks. The same handle is also attached to
    `root_module` and can be removed later with `remove_layerwise_offload(root_module)`.
  """

  return _apply_layerwise_offload(
    root_module,
    module_names=module_names,
    module_filter=module_filter,
    onload_device=onload_device,
    offload_device=torch.device("cpu"),
    output_device=output_device,
    non_blocking=non_blocking,
    eager_offload=eager_offload,
  )


__all__ = [
  "LayerwiseOffloadHandle",
  "_apply_layerwise_offload",
  "_find_offload_related_hf_hook",
  "get_layerwise_offload_handles",
  "layerwise_offload",
  "layerwise_cpu_offload",
  "remove_layerwise_offload",
]
