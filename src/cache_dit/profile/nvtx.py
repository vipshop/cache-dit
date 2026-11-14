"""Optional stage and module NVTX ranges, independent of Torch collection windows."""

import logging
import sys
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from typing import Mapping

import torch
from torch.utils._pytree import tree_leaves

from ..platforms import current_platform

logger = logging.getLogger(__name__)
_NVTX_ENABLED = ContextVar("cache_dit_nvtx", default=False)


@contextmanager
def nvtx_scope(enabled: bool = True):
  """Enable stage annotations in this context and restore the previous setting.

  :param enabled: Whether stage ranges should emit NVTX; disabled scopes work on any platform.
  """
  if enabled and current_platform.device_type != "cuda":
    raise RuntimeError(
      "NVTX annotations require an available CUDA device and a CUDA PyTorch build.")
  token = _NVTX_ENABLED.set(enabled)
  try:
    yield
  finally:
    _NVTX_ENABLED.reset(token)


def _pop_range(exc_type, *_):
  try:
    torch.cuda.nvtx.range_pop()
  except BaseException as error:
    if exc_type is None:
      raise
    logger.warning("NVTX cleanup failed while preserving the original exception: %s", error)


@contextmanager
def stage_range(name: str):
  """Annotate an actual execution stage when its enclosing NVTX scope is enabled.

  :param name: Stage name supplied at the actual execution boundary.
  """
  if not _NVTX_ENABLED.get():
    yield
    return
  torch.cuda.nvtx.range_push(name)
  with ExitStack() as stack:
    stack.push(_pop_range)
    yield


class LayerwiseNvtxMarker:
  """Own temporary forward hooks for explicitly named module roots and enable stage ranges."""

  def __init__(self, roots: Mapping[str, torch.nn.Module]):
    """Configure annotations without inspecting the module trees.

    :param roots: Named execution roots; shared modules use their first declared path.
    """
    self.roots = roots

  def __enter__(self):
    self._stack = ExitStack()
    self._calls = []
    labels = {}

    # Keep both hooks outside Dynamo: NVTX and call bookkeeping are runtime side effects.
    @torch.compiler.disable
    def pre_hook(module, args, kwargs):
      self._calls.append((module, False))
      shapes = [
        tuple(value.shape) for value in tree_leaves((args, kwargs))
        if isinstance(value, torch.Tensor)
      ]
      torch.cuda.nvtx.range_push(f"{labels[id(module)]} inputs={shapes}")
      self._calls[-1] = (module, True)

    @torch.compiler.disable
    def post_hook(module, args, kwargs, output):
      if self._calls and self._calls[-1][0] is module:
        _, pushed = self._calls.pop()
        if pushed:
          torch.cuda.nvtx.range_pop()

    try:
      self._stack.enter_context(nvtx_scope())
      for prefix, root in self.roots.items():
        if not prefix:
          raise ValueError("NVTX roots require explicit nonempty names.")
        compiled_prefix = None
        for name, module in root.named_modules(prefix=prefix):
          if compiled_prefix and name.startswith(compiled_prefix):
            continue
          # Keep compiled roots visible without changing their descendants' TP hook guards.
          if module._compiled_call_impl is not None:
            compiled_prefix = f"{name}."
          # Set the boundary before deduplication so aliases still skip compiled descendants.
          if id(module) in labels:
            continue
          params = {
            key: tuple(value.shape)
            for key, value in module.named_parameters(recurse=False)
          }
          labels[id(module)] = f"{name} ({type(module).__name__}) params={params}"
          self._stack.callback(
            module.register_forward_pre_hook(pre_hook, with_kwargs=True, prepend=True).remove)
          self._stack.callback(
            module.register_forward_hook(post_hook, with_kwargs=True, always_call=True).remove)
    except BaseException:
      self.__exit__(*sys.exc_info())
      raise
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    # PyTorch's always_call handles Exception; drain interrupted BaseException calls on exit.
    for _, pushed in self._calls:
      if pushed:
        self._stack.push(_pop_range)
    self._calls.clear()
    try:
      self._stack.__exit__(exc_type, exc_val, exc_tb)
    except BaseException as error:
      if exc_type is None:
        raise
      logger.warning("NVTX cleanup failed while preserving the original exception: %s", error)
