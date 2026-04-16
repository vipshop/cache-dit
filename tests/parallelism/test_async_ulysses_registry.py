import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cache_dit.distributed.async_ulysses import AsyncUlyssesPlanner, AsyncUlyssesRegistry, MethodPatchSpec
import cache_dit.distributed.async_ulysses.flux as async_flux
import cache_dit.distributed.async_ulysses.flux2 as async_flux2
import cache_dit.distributed.async_ulysses.qwen_image as async_qwen_image
import cache_dit.distributed.async_ulysses.zimage as async_zimage
import cache_dit.distributed.async_ulysses.ovis_image as async_ovis_image
import cache_dit.distributed.async_ulysses.longcat_image as async_longcat_image
import cache_dit.distributed.async_ulysses.zimage_controlnet as async_zimage_controlnet
from diffusers.models.controlnets.controlnet_z_image import ZSingleStreamAttnProcessor as ZImageControlNetProcessor
from diffusers.models.transformers.transformer_flux import FluxAttnProcessor


class _DummyOwner(torch.nn.Module):
  pass


_DummyOwner.__module__ = "diffusers.models.testing"


class _DummyProcessor:

  def __call__(self):
    return "original"


_DummyProcessor.__module__ = "diffusers.models.testing"


def test_async_ulysses_registry_applies_patch_once(monkeypatch):
  monkeypatch.setattr(AsyncUlyssesRegistry, "_planner_registry", {})
  monkeypatch.setattr(AsyncUlyssesRegistry, "_patched_methods", {})
  monkeypatch.setattr(AsyncUlyssesRegistry, "_original_methods", {})
  monkeypatch.setattr(AsyncUlyssesRegistry, "_activated", True)

  build_calls = []

  def _build_wrapper(original):
    build_calls.append(original)

    def _wrapped(self):
      return "wrapped"

    return _wrapped

  @AsyncUlyssesRegistry.register("_DummyOwner")
  class _DummyAsyncUlyssesPlanner(AsyncUlyssesPlanner):

    @classmethod
    def get_method_patches(cls):
      return [MethodPatchSpec(_DummyProcessor, "__call__", _build_wrapper)]

  owner = _DummyOwner()

  assert AsyncUlyssesRegistry.get_planner(owner) is _DummyAsyncUlyssesPlanner
  assert AsyncUlyssesRegistry.apply(owner) is True
  first_wrapper = _DummyProcessor.__call__
  assert _DummyProcessor()() == "wrapped"
  assert len(build_calls) == 1

  assert AsyncUlyssesRegistry.apply(owner) is True
  assert _DummyProcessor.__call__ is first_wrapper
  assert len(build_calls) == 1


def test_async_ulysses_registry_returns_false_for_unknown_module(monkeypatch):
  monkeypatch.setattr(AsyncUlyssesRegistry, "_planner_registry", {})
  monkeypatch.setattr(AsyncUlyssesRegistry, "_patched_methods", {})
  monkeypatch.setattr(AsyncUlyssesRegistry, "_original_methods", {})
  monkeypatch.setattr(AsyncUlyssesRegistry, "_activated", True)

  class _UnknownDiffusersModule(torch.nn.Module):
    pass

  _UnknownDiffusersModule.__module__ = "diffusers.models.testing"
  assert AsyncUlyssesRegistry.apply(_UnknownDiffusersModule()) is False


def test_async_ulysses_registry_rejects_non_diffusers_targets(monkeypatch):
  monkeypatch.setattr(AsyncUlyssesRegistry, "_planner_registry", {})
  monkeypatch.setattr(AsyncUlyssesRegistry, "_patched_methods", {})
  monkeypatch.setattr(AsyncUlyssesRegistry, "_original_methods", {})
  monkeypatch.setattr(AsyncUlyssesRegistry, "_activated", True)

  class _NonDiffusersProcessor:

    def __call__(self):
      return "original"

  def _build_wrapper(original):
    return original

  with pytest.raises(TypeError, match="diffusers"):

    @AsyncUlyssesRegistry.register("CustomModel")
    class _UnsupportedAsyncUlyssesPlanner(AsyncUlyssesPlanner):

      @classmethod
      def get_method_patches(cls):
        return [MethodPatchSpec(_NonDiffusersProcessor, "__call__", _build_wrapper)]


def test_async_flux_ulysses_attn_sends_each_projection_immediately(monkeypatch):
  events = []

  class _FakeHandle:

    def __init__(self, value):
      self._value = value

    def wait(self):
      return self._value

  class _FakeComm:

    def __init__(self, _cp_config):
      pass

    def send_q(self, query):
      events.append("send_q")
      return _FakeHandle(query)

    def send_k(self, key):
      events.append("send_k")
      return _FakeHandle(key)

    def send_v(self, value):
      events.append("send_v")
      return _FakeHandle(value)

    def send_o(self, out):
      return _FakeHandle(out)

  class _FakeAttn:
    heads = 2
    added_kv_proj_dim = None
    to_out = nn.ModuleList([nn.Identity(), nn.Identity()])

    @staticmethod
    def to_v(hidden_states):
      events.append("to_v")
      return hidden_states

    @staticmethod
    def to_q(hidden_states):
      events.append("to_q")
      return hidden_states

    @staticmethod
    def to_k(hidden_states):
      events.append("to_k")
      return hidden_states

    @staticmethod
    def norm_q(query):
      return query

    @staticmethod
    def norm_k(key):
      return key

  monkeypatch.setattr(async_flux, "_All2AllComm", _FakeComm)
  monkeypatch.setattr(async_flux, "_dispatch_attention_fn",
                      lambda query, key, value, **kwargs: query)

  result = async_flux.FluxAsyncUlyssesPlanner._async_ulysses_attn_flux(
    SimpleNamespace(_cp_config=object(), _attention_backend="native"),
    _FakeAttn(),
    torch.randn(1, 2, 4),
  ).wait()

  assert result.shape == (1, 2, 2, 2)
  assert events == ["to_v", "send_v", "to_q", "send_q", "to_k", "send_k"]


def test_async_ulysses_registry_smoke_applies_real_flux_and_controlnet_patches(monkeypatch):
  monkeypatch.setattr(AsyncUlyssesRegistry, "_planner_registry", {})
  monkeypatch.setattr(AsyncUlyssesRegistry, "_patched_methods", {})
  monkeypatch.setattr(AsyncUlyssesRegistry, "_original_methods", {})
  monkeypatch.setattr(AsyncUlyssesRegistry, "_activated", True)

  importlib.reload(async_flux)
  importlib.reload(async_zimage_controlnet)

  class FluxTransformer2DModel(nn.Module):
    pass

  class ZImageControlNetModel(nn.Module):
    pass

  FluxTransformer2DModel.__module__ = "diffusers.models.testing"
  ZImageControlNetModel.__module__ = "diffusers.models.testing"

  flux_before = FluxAttnProcessor.__call__
  controlnet_before = ZImageControlNetProcessor.__call__

  assert AsyncUlyssesRegistry.apply(FluxTransformer2DModel()) is True
  assert FluxAttnProcessor.__call__ is not flux_before

  assert AsyncUlyssesRegistry.apply(ZImageControlNetModel()) is True
  assert ZImageControlNetProcessor.__call__ is not controlnet_before

  monkeypatch.setattr(FluxAttnProcessor, "__call__", flux_before)
  monkeypatch.setattr(ZImageControlNetProcessor, "__call__", controlnet_before)


def test_async_ulysses_registry_get_planner_returns_class_based_planners(monkeypatch):
  monkeypatch.setattr(AsyncUlyssesRegistry, "_planner_registry", {})
  monkeypatch.setattr(AsyncUlyssesRegistry, "_patched_methods", {})
  monkeypatch.setattr(AsyncUlyssesRegistry, "_original_methods", {})
  monkeypatch.setattr(AsyncUlyssesRegistry, "_activated", True)

  importlib.reload(async_flux2)
  importlib.reload(async_qwen_image)
  importlib.reload(async_zimage)
  importlib.reload(async_zimage_controlnet)
  if async_ovis_image.OvisImageAsyncUlyssesPlanner is not None:
    importlib.reload(async_ovis_image)
  if async_longcat_image.LongCatImageAsyncUlyssesPlanner is not None:
    importlib.reload(async_longcat_image)

  assert AsyncUlyssesRegistry.get_planner(
    "Flux2Transformer2DModel") is async_flux2.Flux2AsyncUlyssesPlanner
  assert AsyncUlyssesRegistry.get_planner(
    "QwenImageTransformer2DModel") is async_qwen_image.QwenImageAsyncUlyssesPlanner
  assert AsyncUlyssesRegistry.get_planner(
    "ZImageTransformer2DModel") is async_zimage.ZImageAsyncUlyssesPlanner
  assert AsyncUlyssesRegistry.get_planner(
    "ZImageControlNetModel") is async_zimage_controlnet.ZImageControlNetAsyncUlyssesPlanner
  if async_ovis_image.OvisImageAsyncUlyssesPlanner is not None:
    assert AsyncUlyssesRegistry.get_planner(
      "OvisImageTransformer2DModel") is async_ovis_image.OvisImageAsyncUlyssesPlanner
  if async_longcat_image.LongCatImageAsyncUlyssesPlanner is not None:
    assert AsyncUlyssesRegistry.get_planner(
      "LongCatImageTransformer2DModel") is async_longcat_image.LongCatImageAsyncUlyssesPlanner
