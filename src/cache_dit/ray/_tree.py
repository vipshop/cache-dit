from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import torch


def tree_map_tensors(value: Any, fn: Callable[[torch.Tensor], torch.Tensor]) -> Any:
  """Apply a function to every tensor in a nested Python object.

  :param value: Nested object that may contain tensors.
  :param fn: Function applied to each tensor leaf.
  :returns: Object with the same container structure and transformed tensor leaves.
  """

  if isinstance(value, torch.Tensor):
    return fn(value)
  if dataclasses.is_dataclass(value) and not isinstance(value, type):
    updates = {
      field.name: tree_map_tensors(getattr(value, field.name), fn)
      for field in dataclasses.fields(value)
    }
    return dataclasses.replace(value, **updates)
  if isinstance(value, Mapping):
    return type(value)((key, tree_map_tensors(item, fn)) for key, item in value.items())
  if isinstance(value, tuple) and hasattr(value, "_fields"):
    return type(value)(*(tree_map_tensors(item, fn) for item in value))
  if isinstance(value, tuple):
    return tuple(tree_map_tensors(item, fn) for item in value)
  if isinstance(value, list):
    return [tree_map_tensors(item, fn) for item in value]
  if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
    return type(value)(tree_map_tensors(item, fn) for item in value)
  return value


def cpu_tensor_tree(value: Any) -> Any:
  """Detach and move tensor leaves to CPU for Ray object-store transport.

  :param value: Nested object that may contain tensors.
  :returns: A nested object whose tensor leaves live on CPU.
  """

  return tree_map_tensors(value, lambda tensor: tensor.detach().cpu())


def device_tensor_tree(value: Any, device: torch.device) -> Any:
  """Move tensor leaves in a nested object to a target device.

  :param value: Nested object that may contain tensors.
  :param device: Destination torch device.
  :returns: A nested object whose tensor leaves live on ``device``.
  """

  return tree_map_tensors(value, lambda tensor: tensor.to(device=device))


def first_tensor_device(value: Any) -> torch.device | None:
  """Return the device of the first tensor leaf in a nested object.

  :param value: Nested object that may contain tensors.
  :returns: Device of the first tensor leaf, or ``None`` if no tensor exists.
  """

  if isinstance(value, torch.Tensor):
    return value.device
  if dataclasses.is_dataclass(value) and not isinstance(value, type):
    for field in dataclasses.fields(value):
      device = first_tensor_device(getattr(value, field.name))
      if device is not None:
        return device
  if isinstance(value, Mapping):
    for item in value.values():
      device = first_tensor_device(item)
      if device is not None:
        return device
  if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
    for item in value:
      device = first_tensor_device(item)
      if device is not None:
        return device
  return None
