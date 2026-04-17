from types import SimpleNamespace

import pytest
import torch
from diffusers.models.modeling_utils import ModelMixin

from cache_dit.distributed.backend import ParallelismBackend
from cache_dit.distributed.config import ParallelismConfig
import cache_dit.distributed.controlnets as controlnet_entrypoints
import cache_dit.distributed.dispatch as distributed_dispatch
import cache_dit.distributed.transformers as transformer_entrypoints
import cache_dit.distributed.controlnets.dispatch as controlnet_dispatch
import cache_dit.distributed.async_ulysses.flux as async_flux
import cache_dit.distributed.transformers.flux as cp_plan_flux
import cache_dit.distributed.transformers.flux2 as cp_plan_flux2
import cache_dit.distributed.transformers.dispatch as transformer_dispatch
import cache_dit.distributed.controlnets.zimage_controlnet as cp_plan_zimage_controlnet


class _DummyControlNet(ModelMixin):

  def __init__(self):
    super().__init__()


class _DummyBackendAwareModule(ModelMixin):

  def __init__(self):
    super().__init__()
    self.backends: list[str] = []

  def set_attention_backend(self, backend: str) -> None:
    self.backends.append(backend)


def test_parallelize_transformer_context_uses_custom_cp_plan(monkeypatch):
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

  monkeypatch.setattr(transformer_dispatch.ContextParallelismPlannerRegister, "get_planner",
                      _raise_if_called)
  monkeypatch.setattr(transformer_dispatch, "_enable_context_parallelism", _record_enable)
  monkeypatch.setattr(transformer_dispatch, "_maybe_patch_parallel_config", _record_patch)

  result = transformer_dispatch._parallelize_transformer_context(transformer, parallelism_config)

  assert result is transformer
  assert len(calls) == 2
  assert calls[0][0] == "enable"
  assert calls[0][2] is cp_plan
  assert calls[1][0] == "patch"


def test_parallelize_controlnet_cp_uses_custom_cp_plan(monkeypatch):
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

  result = controlnet_dispatch._parallelize_controlnet_cp(controlnet, parallelism_config)

  assert result is controlnet
  assert len(calls) == 1
  assert calls[0][0] == "enable"
  assert calls[0][2] is cp_plan


@pytest.mark.parametrize(
  ("backend", "expected_backends"),
  [
    (ParallelismBackend.NATIVE_DIFFUSER, ["native"]),
    (ParallelismBackend.NATIVE_HYBRID, ["native"]),
    (ParallelismBackend.NATIVE_PYTORCH, []),
  ],
)
def test_enable_parallelism_defaults_native_backend_only_for_diffusers_cp_or_hybrid(
  monkeypatch,
  backend,
  expected_backends,
):
  monkeypatch.setattr("cache_dit.attention._maybe_register_custom_attn_backends", lambda: None)
  monkeypatch.setattr(transformer_entrypoints, "parallelize_transformer",
                      lambda transformer, parallelism_config: transformer)
  monkeypatch.setattr(distributed_dispatch, "maybe_empty_cache", lambda: None)

  module = _DummyBackendAwareModule()
  parallelism_config = SimpleNamespace(
    backend=backend,
    attention_backend=None,
    extra_parallel_modules=None,
  )

  result = distributed_dispatch.enable_parallelism(module, parallelism_config)

  assert result is module
  assert module.backends == expected_backends


def test_enable_parallelism_applies_attention_backend_to_parallelized_controlnet(monkeypatch):
  monkeypatch.setattr("cache_dit.attention._maybe_register_custom_attn_backends", lambda: None)
  monkeypatch.setattr(transformer_entrypoints, "parallelize_transformer",
                      lambda transformer, parallelism_config: transformer)
  monkeypatch.setattr(controlnet_entrypoints, "parallelize_controlnet",
                      lambda controlnet, parallelism_config: controlnet)
  monkeypatch.setattr(distributed_dispatch, "check_text_encoder", lambda _module: False)
  monkeypatch.setattr(distributed_dispatch, "check_controlnet", lambda _module: True)
  monkeypatch.setattr(distributed_dispatch, "check_auto_encoder", lambda _module: False)
  monkeypatch.setattr(distributed_dispatch, "check_parallelized", lambda _module: False)
  monkeypatch.setattr(distributed_dispatch, "maybe_empty_cache", lambda: None)

  transformer = _DummyBackendAwareModule()
  controlnet = _DummyBackendAwareModule()
  parallelism_config = SimpleNamespace(
    backend=ParallelismBackend.NATIVE_DIFFUSER,
    attention_backend=None,
    extra_parallel_modules=[controlnet],
  )

  result = distributed_dispatch.enable_parallelism(transformer, parallelism_config)

  assert result is transformer
  assert transformer.backends == ["native"]
  assert controlnet.backends == ["native"]


def test_patch_flux_attn_processor_falls_back_without_cp_config(monkeypatch):
  hidden_states = torch.randn(1, 2, 3)
  calls = []

  def _original(self, attn, hidden_states, **kwargs):
    calls.append((self, attn, hidden_states, kwargs))
    return "fallback"

  wrapper = async_flux.FluxAsyncUlyssesPlanner._build_flux_attn_patch(_original)

  result = wrapper(
    SimpleNamespace(),
    SimpleNamespace(),
    hidden_states,
  )

  assert result == "fallback"
  assert len(calls) == 1


def test_async_ulysses_flux_requires_cp_config():
  with pytest.raises(RuntimeError, match="_cp_config"):
    async_flux.FluxAsyncUlyssesPlanner._async_ulysses_attn_flux(
      SimpleNamespace(),
      SimpleNamespace(),
      torch.randn(1, 2, 3),
    )


def test_flux_planner_uses_async_ulysses_registry(monkeypatch):
  planner = cp_plan_flux.FluxContextParallelismPlanner()
  planner._cp_planner_preferred_native_diffusers = False
  calls = []

  monkeypatch.setattr(cp_plan_flux.AsyncUlyssesRegistry, "apply",
                      lambda transformer: calls.append(transformer) or True)

  transformer = object()
  cp_plan = planner._apply(
    transformer=transformer,
    parallelism_config=SimpleNamespace(ulysses_async=True),
  )

  assert transformer in calls
  assert "proj_out" in cp_plan


def test_controlnet_planner_uses_async_ulysses_registry(monkeypatch):
  planner = cp_plan_zimage_controlnet.ZImageControlNetContextParallelismPlanner()
  planner._cp_planner_preferred_native_diffusers = False
  calls = []

  monkeypatch.setattr(cp_plan_zimage_controlnet.AsyncUlyssesRegistry, "apply",
                      lambda controlnet: calls.append(controlnet) or True)

  controlnet = SimpleNamespace(
    control_layers=[object()],
    control_noise_refiner=[object()],
  )
  cp_plan = planner._apply(
    controlnet=controlnet,
    parallelism_config=SimpleNamespace(ulysses_async=True),
  )

  assert controlnet in calls
  assert "control_layers.0" in cp_plan


def test_flux2_klein_kv_planner_registers_fine_grained_attention_subplans(monkeypatch):
  planner = cp_plan_flux2.Flux2ContextParallelismPlanner()
  planner._cp_planner_preferred_native_diffusers = False
  transformer = SimpleNamespace(
    transformer_blocks=[
      SimpleNamespace(attn=SimpleNamespace(added_kv_proj_dim=16)),
      SimpleNamespace(attn=SimpleNamespace(added_kv_proj_dim=16)),
    ],
    single_transformer_blocks=[object(), object()],
  )

  monkeypatch.setattr(cp_plan_flux2, "_is_klein_kv", lambda _transformer: True)

  cp_plan = planner._apply(
    transformer=transformer,
    parallelism_config=SimpleNamespace(ulysses_async=False),
  )

  assert "transformer_blocks.0" not in cp_plan
  assert "transformer_blocks.*" not in cp_plan
  assert cp_plan["transformer_blocks.*.attn.to_q"]["input"].split_dim == 1
  assert cp_plan["transformer_blocks.*.attn.to_k"]["input"].split_dim == 1
  assert isinstance(cp_plan["transformer_blocks.*.attn.to_v"], list)
  assert cp_plan["transformer_blocks.*.attn.to_v"][0]["input"].split_dim == 1
  assert cp_plan["transformer_blocks.*.attn.to_v"][1].gather_dim == 1
  assert cp_plan["transformer_blocks.*.attn.add_q_proj"]["input"].split_dim == 1
  assert cp_plan["transformer_blocks.*.attn.add_k_proj"]["input"].split_dim == 1
  assert "transformer_blocks.*.attn.norm_added_q" not in cp_plan
  assert isinstance(cp_plan["transformer_blocks.*.attn.add_v_proj"], list)
  assert cp_plan["transformer_blocks.*.attn.add_v_proj"][0]["input"].split_dim == 1
  assert cp_plan["transformer_blocks.*.attn.add_v_proj"][1].gather_dim == 1
  assert "transformer_blocks.*.attn.norm_k" not in cp_plan
  assert "transformer_blocks.*.attn.norm_added_k" not in cp_plan
  assert cp_plan["transformer_blocks.*.attn.to_add_out"].gather_dim == 1
  assert cp_plan["transformer_blocks.*.attn.to_out.0"].gather_dim == 1
  assert "single_transformer_blocks.0" not in cp_plan
  assert "single_transformer_blocks.*" not in cp_plan
  assert "single_transformer_blocks.*.attn.to_qkv_mlp_proj" in cp_plan
  assert isinstance(cp_plan["single_transformer_blocks.*.attn.to_qkv_mlp_proj"], list)
  assert cp_plan["single_transformer_blocks.*.attn.to_qkv_mlp_proj"][0]["input"].split_dim == 1
  assert cp_plan["single_transformer_blocks.*.attn.to_qkv_mlp_proj"][1].gather_dim == 1
  assert cp_plan["single_transformer_blocks.*.attn.norm_q"]["x"].split_dim == 1
  assert cp_plan["single_transformer_blocks.*.attn.norm_q"]["x"].expected_dims == 4
  assert cp_plan["single_transformer_blocks.*.attn.norm_k"]["x"].split_dim == 1
  assert cp_plan["single_transformer_blocks.*.attn.norm_k"]["x"].expected_dims == 4
  assert cp_plan["single_transformer_blocks.*.attn.mlp_act_fn"]["x"].split_dim == 1
  assert cp_plan["single_transformer_blocks.*.attn.to_out"].gather_dim == 1


def test_flux2_klein_kv_planner_skips_added_kv_hooks_without_added_kv_proj(monkeypatch):
  planner = cp_plan_flux2.Flux2ContextParallelismPlanner()
  planner._cp_planner_preferred_native_diffusers = False
  transformer = SimpleNamespace(
    transformer_blocks=[
      SimpleNamespace(attn=SimpleNamespace(added_kv_proj_dim=None)),
      SimpleNamespace(attn=SimpleNamespace(added_kv_proj_dim=None)),
    ],
    single_transformer_blocks=[object()],
  )

  monkeypatch.setattr(cp_plan_flux2, "_is_klein_kv", lambda _transformer: True)

  cp_plan = planner._apply(
    transformer=transformer,
    parallelism_config=SimpleNamespace(ulysses_async=False),
  )

  assert "transformer_blocks.*.attn.add_q_proj" not in cp_plan
  assert "transformer_blocks.*.attn.add_k_proj" not in cp_plan
  assert "transformer_blocks.*.attn.add_v_proj" not in cp_plan
  assert "transformer_blocks.*.attn.norm_added_q" not in cp_plan
  assert "transformer_blocks.*.attn.norm_added_k" not in cp_plan
  assert "transformer_blocks.*.attn.norm_k" not in cp_plan
  assert "transformer_blocks.*.attn.to_add_out" not in cp_plan


def test_flux2_kv_attention_local_query_global_kv_uses_separate_query_and_key_counts(monkeypatch):
  dispatch_calls = []

  def _record_dispatch(query, key, value, attn_mask=None, backend=None, cp_config=None, **kwargs):
    del value, attn_mask, backend, cp_config, kwargs
    dispatch_calls.append((query.shape[1], key.shape[1]))
    return torch.zeros_like(query)

  monkeypatch.setattr(cp_plan_flux2, "_dispatch_attention_fn", _record_dispatch)

  output = cp_plan_flux2._flux2_kv_attention_local_query_global_kv(
    query=torch.randn(1, 2, 1, 2),
    key=torch.randn(1, 5, 1, 2),
    value=torch.randn(1, 5, 1, 2),
    query_num_txt_tokens=1,
    query_num_ref_tokens=1,
    key_num_txt_tokens=2,
    key_num_ref_tokens=1,
    backend="native",
  )

  assert output.shape[1] == 2
  assert dispatch_calls == [(1, 5), (1, 1)]
