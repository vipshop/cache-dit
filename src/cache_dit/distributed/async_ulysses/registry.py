import importlib
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import torch

from ...logger import init_logger

logger = init_logger(__name__)

MethodWrapperFactory = Callable[[Callable[..., Any]], Callable[..., Any]]
MethodPatchProvider = Callable[[], Sequence["MethodPatchSpec"]]
PlannerClass = type["AsyncUlyssesPlanner"]


@dataclass(frozen=True)
class MethodPatchSpec:
  target_cls: type
  method_name: str
  wrapper_factory: MethodWrapperFactory


class AsyncUlyssesPlanner:
  """Base class for model-specific async Ulysses patch planners."""

  @classmethod
  def get_method_patches(cls) -> Sequence[MethodPatchSpec]:
    raise NotImplementedError("get_method_patches must be implemented by subclasses")

  @classmethod
  def apply(cls, module: Optional[torch.nn.Module] = None) -> bool:
    del module
    for patch_spec in cls.get_method_patches():
      AsyncUlyssesRegistry._apply_method_patch(patch_spec)
    return True


class AsyncUlyssesRegistry:
  """Register and apply async Ulysses wrappers for diffusers models only."""

  _planner_registry: dict[str, PlannerClass] = {}
  _patched_methods: dict[tuple[type, str], Callable[..., Any]] = {}
  _original_methods: dict[tuple[type, str], Callable[..., Any]] = {}
  _activated: bool = False

  @classmethod
  def register(
    cls,
    owner_names: str | Sequence[str],
  ) -> Callable[[type[AsyncUlyssesPlanner] | MethodPatchProvider], type[AsyncUlyssesPlanner]
                | MethodPatchProvider]:

    def decorator(
      planner_or_provider: type[AsyncUlyssesPlanner] | MethodPatchProvider,
    ) -> type[AsyncUlyssesPlanner] | MethodPatchProvider:
      names = (owner_names, ) if isinstance(owner_names, str) else tuple(owner_names)
      planner_cls = cls._normalize_planner(planner_or_provider)
      cls._validate_planner_targets(planner_cls)
      for name in names:
        assert name not in cls._planner_registry, (
          f"AsyncUlyssesPlanner with name {name} is already registered.")
        cls._planner_registry[name] = planner_cls
      return planner_or_provider

    return decorator

  @classmethod
  def get_planner(cls, module_or_name: Any) -> PlannerClass:
    cls._ensure_activated()
    if isinstance(module_or_name, str):
      name = module_or_name
    elif module_or_name is None:
      raise ValueError("No async Ulysses planner registered under name: None")
    else:
      if not cls._is_diffusers_cls(module_or_name.__class__):
        raise ValueError("Async Ulysses currently only supports models from diffusers. "
                         f"Got {module_or_name.__class__.__name__} from "
                         f"{module_or_name.__class__.__module__}.")
      name = module_or_name.__class__.__name__

    planner_cls = cls._planner_registry.get(name)
    if planner_cls is not None:
      return planner_cls

    planner_cls = None
    matched_prefix_len = -1
    for planner_name, registered_planner in cls._planner_registry.items():
      if name.startswith(planner_name) and len(planner_name) > matched_prefix_len:
        planner_cls = registered_planner
        matched_prefix_len = len(planner_name)
    if planner_cls is None:
      raise ValueError(f"No async Ulysses planner registered under name: {name}")
    return planner_cls

  @classmethod
  def apply(cls, module: Optional[torch.nn.Module]) -> bool:
    if module is None:
      return False

    try:
      planner_cls = cls.get_planner(module)
    except ValueError:
      return False
    planner_cls.apply(module)
    return True

  @classmethod
  def get_original_method(
    cls,
    target_cls: type,
    method_name: str,
  ) -> Optional[Callable[..., Any]]:
    return cls._original_methods.get((target_cls, method_name))

  @classmethod
  def _apply_method_patch(cls, patch_spec: MethodPatchSpec) -> None:
    patch_key = (patch_spec.target_cls, patch_spec.method_name)
    if patch_key in cls._patched_methods:
      return

    original = getattr(patch_spec.target_cls, patch_spec.method_name)
    wrapper = patch_spec.wrapper_factory(original)
    setattr(patch_spec.target_cls, patch_spec.method_name, wrapper)
    cls._original_methods[patch_key] = original
    cls._patched_methods[patch_key] = wrapper

  @classmethod
  def _ensure_activated(cls) -> None:
    if cls._activated:
      return

    cls._safe_import(".transformers")
    cls._safe_import(".controlnets")
    cls._activated = True

  @staticmethod
  def _is_diffusers_cls(target_cls: type) -> bool:
    module_name = getattr(target_cls, "__module__", "")
    return module_name == "diffusers" or module_name.startswith("diffusers.")

  @classmethod
  def _validate_planner_targets(cls, planner_cls: PlannerClass) -> None:
    patch_specs = tuple(planner_cls.get_method_patches())
    if not patch_specs:
      raise ValueError("AsyncUlyssesPlanner must define at least one patch for a diffusers module.")

    unsupported_targets = []
    for patch_spec in patch_specs:
      if not cls._is_diffusers_cls(patch_spec.target_cls):
        target_module = getattr(patch_spec.target_cls, "__module__", "")
        unsupported_targets.append(f"{patch_spec.target_cls.__qualname__} ({target_module})")

    if unsupported_targets:
      raise TypeError("Async Ulysses currently only supports diffusers models and processors. "
                      f"Unsupported targets: {', '.join(unsupported_targets)}")

  @classmethod
  def _normalize_planner(
    cls,
    planner_or_provider: type[AsyncUlyssesPlanner] | MethodPatchProvider,
  ) -> PlannerClass:
    if isinstance(planner_or_provider, type) and issubclass(planner_or_provider,
                                                            AsyncUlyssesPlanner):
      return planner_or_provider

    provider = planner_or_provider

    class _ProviderAsyncUlyssesPlanner(AsyncUlyssesPlanner):

      @classmethod
      def get_method_patches(cls) -> Sequence[MethodPatchSpec]:
        return tuple(provider())

    _ProviderAsyncUlyssesPlanner.__name__ = provider.__name__
    _ProviderAsyncUlyssesPlanner.__qualname__ = provider.__qualname__
    _ProviderAsyncUlyssesPlanner.__module__ = provider.__module__
    return _ProviderAsyncUlyssesPlanner

  @classmethod
  def _safe_import(cls, module_name: str) -> None:
    package = __package__ if __package__ is not None else ""
    try:
      importlib.import_module(module_name, package=package)
    except ImportError as exc:
      logger.debug(f"Failed to activate async Ulysses module {module_name}: {exc}")
