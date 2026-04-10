"""Adapted from diffusers' forward hook management primitives for cache-dit CP."""
import functools
from typing import Any

import torch


def unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
  """Return the innermost wrapped module when wrappers expose `.module`.

  :param module: Candidate wrapped module.
  :returns: The innermost module.
  """

  current = module
  visited: set[int] = set()
  while hasattr(current, "module") and id(current) not in visited:
    visited.add(id(current))
    next_module = getattr(current, "module")
    if not isinstance(next_module, torch.nn.Module):
      break
    current = next_module
  return current


class ModelHook:
  """Base hook with pre/post-forward callbacks."""

  _is_stateful = False

  def __init__(self) -> None:
    self.fn_ref: HookFunctionReference | None = None

  def initialize_hook(self, module: torch.nn.Module) -> torch.nn.Module:
    return module

  def deinitalize_hook(self, module: torch.nn.Module) -> torch.nn.Module:
    return module

  def pre_forward(self, module: torch.nn.Module, *args,
                  **kwargs) -> tuple[tuple[Any], dict[str, Any]]:
    return args, kwargs

  def post_forward(self, module: torch.nn.Module, output: Any) -> Any:
    return output

  def detach_hook(self, module: torch.nn.Module) -> torch.nn.Module:
    return module


class HookFunctionReference:

  def __init__(self) -> None:
    self.pre_forward = None
    self.post_forward = None
    self.forward = None
    self.original_forward = None


class HookRegistry:
  """Manage a stack of forward hooks for one module."""

  _registry_attr = "_cache_dit_hook"

  def __init__(self, module_ref: torch.nn.Module) -> None:
    self.hooks: dict[str, ModelHook] = {}
    self._module_ref = module_ref
    self._hook_order: list[str] = []
    self._fn_refs: list[HookFunctionReference] = []

  def register_hook(self, hook: ModelHook, name: str) -> None:
    if name in self.hooks:
      raise ValueError(f"Hook with name {name} already exists in the registry.")

    self._module_ref = hook.initialize_hook(self._module_ref)

    def create_new_forward(function_reference: HookFunctionReference):

      def new_forward(module, *args, **kwargs):
        args, kwargs = function_reference.pre_forward(module, *args, **kwargs)
        output = function_reference.forward(*args, **kwargs)
        return function_reference.post_forward(module, output)

      return new_forward

    forward = self._module_ref.forward

    fn_ref = HookFunctionReference()
    fn_ref.pre_forward = hook.pre_forward
    fn_ref.post_forward = hook.post_forward
    fn_ref.forward = forward

    if hasattr(hook, "new_forward"):
      fn_ref.original_forward = forward
      fn_ref.forward = functools.update_wrapper(
        functools.partial(hook.new_forward, self._module_ref), hook.new_forward)

    rewritten_forward = create_new_forward(fn_ref)
    self._module_ref.forward = functools.update_wrapper(
      functools.partial(rewritten_forward, self._module_ref), rewritten_forward)

    hook.fn_ref = fn_ref
    self.hooks[name] = hook
    self._hook_order.append(name)
    self._fn_refs.append(fn_ref)

  def get_hook(self, name: str) -> ModelHook | None:
    return self.hooks.get(name)

  def remove_hook(self, name: str, recurse: bool = True) -> None:
    if name in self.hooks:
      num_hooks = len(self._hook_order)
      hook = self.hooks[name]
      index = self._hook_order.index(name)
      fn_ref = self._fn_refs[index]

      old_forward = fn_ref.original_forward if fn_ref.original_forward is not None else fn_ref.forward
      if index == num_hooks - 1:
        self._module_ref.forward = old_forward
      else:
        self._fn_refs[index + 1].forward = old_forward

      self._module_ref = hook.deinitalize_hook(self._module_ref)
      del self.hooks[name]
      self._hook_order.pop(index)
      self._fn_refs.pop(index)

    if recurse:
      for module_name, module in unwrap_module(self._module_ref).named_modules():
        if module_name == "":
          continue
        module = unwrap_module(module)
        if hasattr(module, self._registry_attr):
          getattr(module, self._registry_attr).remove_hook(name, recurse=False)

  @classmethod
  def check_if_exists_or_initialize(cls, module: torch.nn.Module) -> "HookRegistry":
    if not hasattr(module, cls._registry_attr):
      setattr(module, cls._registry_attr, cls(module))
    return getattr(module, cls._registry_attr)

  def __repr__(self) -> str:
    registry_repr = ""
    for index, hook_name in enumerate(self._hook_order):
      hook = self.hooks[hook_name]
      hook_repr = hook.__repr__(
      ) if hook.__class__.__repr__ is not object.__repr__ else hook.__class__.__name__
      registry_repr += f"  ({index}) {hook_name} - {hook_repr}"
      if index < len(self._hook_order) - 1:
        registry_repr += "\n"
    return f"HookRegistry(\n{registry_repr}\n)"
