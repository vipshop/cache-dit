from types import SimpleNamespace

import pytest
import torch
from diffusers.models.modeling_utils import ModelMixin

from cache_dit.distributed.config import ParallelismConfig
import cache_dit.distributed.controlnets as controlnet_cp
import cache_dit.distributed.controlnets.dispatch as controlnet_dispatch
import cache_dit.distributed.transformers.flux as cp_plan_flux
import cache_dit.distributed.transformers as transformer_cp
import cache_dit.distributed.transformers.dispatch as transformer_dispatch


class _DummyControlNet(ModelMixin):

  def __init__(self):
    super().__init__()


def test_maybe_enable_context_parallelism_uses_custom_cp_plan(monkeypatch):
  transformer = torch.nn.Linear(4, 4)
  cp_plan = {"": {"hidden_states": object()}}
  parallelism_config = ParallelismConfig(ulysses_size=2, cp_plan=cp_plan)
  calls = []

  def _raise_if_called(_transformer):
    raise AssertionError("planner registry should not be consulted when custom cp_plan is provided")

  def _record_enable(_transformer, *, config, cp_plan):
    calls.append(("enable", config, cp_plan))

  def _record_patch(_transformer, **kwargs):
    calls.append(("patch", _transformer))
    return _transformer

  monkeypatch.setattr(transformer_cp.ContextParallelismPlannerRegister, "get_planner",
                      _raise_if_called)
  monkeypatch.setattr(transformer_dispatch, "_enable_context_parallelism", _record_enable)
  monkeypatch.setattr(transformer_dispatch, "_maybe_patch_native_parallel_config", _record_patch)

  result = transformer_cp.maybe_enable_context_parallelism(transformer, parallelism_config)

  assert result is transformer
  assert len(calls) == 2
  assert calls[0][0] == "enable"
  assert calls[0][2] is cp_plan
  assert calls[1][0] == "patch"


def test_maybe_enable_controlnet_context_parallelism_uses_custom_cp_plan(monkeypatch):
  controlnet = _DummyControlNet()
  cp_plan = {"": {"hidden_states": object()}}
  parallelism_config = ParallelismConfig(ulysses_size=2, cp_plan=cp_plan)
  calls = []

  def _raise_if_called(_controlnet):
    raise AssertionError("planner registry should not be consulted when custom cp_plan is provided")

  def _record_enable(_controlnet, *, config, cp_plan):
    calls.append(("enable", config, cp_plan))

  monkeypatch.setattr(controlnet_dispatch.ControlNetContextParallelismPlannerRegister,
                      "get_planner", _raise_if_called)
  monkeypatch.setattr(controlnet_dispatch, "_enable_context_parallelism", _record_enable)

  result = controlnet_cp.maybe_enable_context_parallelism(controlnet, parallelism_config)

  assert result is controlnet
  assert len(calls) == 1
  assert calls[0][0] == "enable"
  assert calls[0][2] is cp_plan


def test_patch_flux_attn_processor_falls_back_without_cp_config(monkeypatch):
  hidden_states = torch.randn(1, 2, 3)
  calls = []

  def _original(self, attn, hidden_states, **kwargs):
    calls.append((self, attn, hidden_states, kwargs))
    return "fallback"

  monkeypatch.setattr(cp_plan_flux, "flux_attn_processor__call__", _original)

  result = cp_plan_flux.__patch_flux_attn_processor__(
    SimpleNamespace(),
    SimpleNamespace(),
    hidden_states,
  )

  assert result == "fallback"
  assert len(calls) == 1


def test_async_ulysses_flux_requires_cp_config():
  with pytest.raises(RuntimeError, match="_cp_config"):
    cp_plan_flux._async_ulysses_attn_flux(
      SimpleNamespace(),
      SimpleNamespace(),
      torch.randn(1, 2, 3),
    )
