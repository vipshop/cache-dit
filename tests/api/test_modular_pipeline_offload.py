"""Regression tests for layerwise offload on diffusers modular pipelines.

A `ModularPipeline` owns the same components (`transformer`, `text_encoder`, `vae`) as a
`DiffusionPipeline`, but it is neither a `DiffusionPipeline` nor an `nn.Module`, and it has no
`enable_(sequential_)cpu_offload` method. These tests pin the two behaviours that make the offload
path work for it:

1. component discovery reads the modular `components` mapping instead of walking `__dict__`;
2. `--cpu-offload` / `--sequential-cpu-offload` fall back to cache-dit's own layerwise CPU offload
   instead of calling the missing diffusers pipeline method.
"""
import argparse

import pytest
import torch

import cache_dit._utils.utils as cli_utils
from cache_dit.offload import (
  get_layerwise_offload_handles,
  layerwise_offload,
  remove_layerwise_offload,
)
from cache_dit.offload.layerwise import _collect_public_offload_roots


def _fake_modular_pipeline(components):
  """Build a bare `ModularPipeline` (no __init__) whose `components` mapping is `components`."""
  from diffusers import ModularPipeline

  class _FakeModularPipeline(ModularPipeline):

    @property
    def components(self):
      return self._fake_components

  pipe = object.__new__(_FakeModularPipeline)
  pipe._fake_components = components
  for name, module in components.items():
    if isinstance(module, torch.nn.Module):
      setattr(pipe, name, module)
  return pipe


def _offload_args(**overrides) -> argparse.Namespace:
  args = argparse.Namespace(
    cpu_offload=False,
    sequential_cpu_offload=False,
    module_layerwise_cpu_offload=False,
    layerwise_async_transfer=False,
    layerwise_transfer_buckets=1,
    layerwise_prefetch_limit=False,
    layerwise_max_copy_streams=None,
    layerwise_max_inflight_prefetch_bytes=None,
    layerwise_persistent_buckets=0,
    layerwise_persistent_bins=1,
    layerwise_text_async_transfer=None,
    layerwise_text_transfer_buckets=None,
    layerwise_text_prefetch_limit=None,
    layerwise_text_max_copy_streams=None,
    layerwise_text_max_inflight_prefetch_bytes=None,
    layerwise_text_persistent_buckets=None,
    layerwise_text_persistent_bins=None,
  )
  for key, value in overrides.items():
    setattr(args, key, value)
  return args


def test_collect_offload_roots_reads_modular_components():
  transformer = torch.nn.Linear(4, 4, bias=False)
  text_encoder = torch.nn.Linear(4, 4, bias=False)
  pipe = _fake_modular_pipeline({
    "transformer": transformer,
    "text_encoder": text_encoder,
    "scheduler": object(),
  })
  # an nn.Module that is NOT a declared component must not become an offload root, which is what
  # separates the `components` path from the old `vars(pipe)` fallback
  pipe._internal_helper = torch.nn.Linear(4, 4, bias=False)

  roots = _collect_public_offload_roots(pipe)

  assert roots == [("transformer", transformer), ("text_encoder", text_encoder)]


def test_layerwise_offload_accepts_modular_pipeline():
  transformer = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
  pipe = _fake_modular_pipeline({"transformer": transformer})

  layerwise_offload(pipe, module_names=["transformer"], onload_device="cpu")

  assert len(get_layerwise_offload_handles(pipe)) == 1
  assert remove_layerwise_offload(pipe) == 1


def test_cpu_offload_switch_falls_back_to_layerwise_for_modular(monkeypatch):
  transformer = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
  pipe = _fake_modular_pipeline({"transformer": transformer})

  applied = []
  monkeypatch.setattr(
    cli_utils,
    "layerwise_cpu_offload",
    lambda module, **kwargs: applied.append((module, kwargs)) or "handle",
  )
  monkeypatch.setattr(cli_utils, "remove_layerwise_offload", lambda module, **kwargs: 0)
  monkeypatch.setattr(cli_utils, "_find_offload_related_hf_hook", lambda module: None)

  for flag in ("cpu_offload", "sequential_cpu_offload"):
    applied.clear()
    used_offload = cli_utils.maybe_cpu_offload(_offload_args(**{flag: True}), pipe)
    # the diffusers pipeline API must not be touched, and the module is streamed layer by layer
    assert used_offload is True
    assert [module for module, _kwargs in applied] == [transformer]


def test_layerwise_offload_moves_mirror_less_targets_whole():
  """Targets without a pinned mirror must still be offloaded (sync and async paths)."""
  from cache_dit.offload import layerwise as offload_module

  module = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False), torch.nn.Linear(4, 4, bias=False))
  original_supports = offload_module._supports_tensor_state_mirror
  offload_module._supports_tensor_state_mirror = lambda target: False
  try:
    for async_transfer in (False, True):
      if async_transfer and not torch.cuda.is_available():
        continue
      # `torch.float64` keeps the check independent of the CUDA availability of bf16 kernels
      if not async_transfer:
        handle = layerwise_offload(
          module,
          module_names=["0"],
          onload_device="cpu",
          offload_device="cpu",
          async_transfer=False,
        )
      else:
        handle = layerwise_offload(
          module,
          module_names=["0"],
          onload_device="cuda",
          offload_device="cpu",
          async_transfer=True,
        )
        # a whole-module move must have really pushed the weights to CPU at apply time
        assert module[0].weight.device.type == "cpu"
        assert module[0](torch.randn(2, 4, device="cuda")).device.type == "cuda"
        assert module[0].weight.device.type == "cpu"
      handle.remove()
  finally:
    offload_module._supports_tensor_state_mirror = original_supports


@pytest.mark.skipif(not torch.cuda.is_available(), reason="torchao fp8 quantization requires CUDA")
def test_layerwise_offload_skips_mirror_for_quantized_state():
  """A torchao `Float8Tensor` weight must not be routed through the pinned mirror."""
  from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, quantize_

  from cache_dit.offload.layerwise import _supports_tensor_state_mirror

  module = torch.nn.Linear(64, 64, bias=False).cuda()
  assert _supports_tensor_state_mirror(module) is True

  quantize_(module, Float8DynamicActivationFloat8WeightConfig())

  assert type(module.weight.data).__module__.startswith("torchao")
  assert _supports_tensor_state_mirror(module) is False
