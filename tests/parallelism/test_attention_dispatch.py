from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from diffusers.models.attention_dispatch import AttentionBackendName, _AttentionBackendRegistry

from cache_dit.envs import ENV
import cache_dit.attention as attention
from cache_dit.attention._attention_dispatch import (
  _AttnBackend,
  _AttnBackendRegistry,
  _ContextParallelConfig,
  _default_active_backend,
  _dispatch_attention_fn,
  _native_attention,
)
from cache_dit.attention._diffusers_bridge import (
  _register_cache_dit_attn_backends_to_diffusers, )


def test_dispatch_attention_fn_uses_cache_dit_native_backend():
  query = torch.randn(2, 5, 3, 4)
  key = torch.randn(2, 5, 3, 4)
  value = torch.randn(2, 5, 3, 4)

  out = _dispatch_attention_fn(query, key, value, backend=_AttnBackend.NATIVE)

  expected = F.scaled_dot_product_attention(
    query.permute(0, 2, 1, 3),
    key.permute(0, 2, 1, 3),
    value.permute(0, 2, 1, 3),
  ).permute(0, 2, 1, 3)

  assert out.shape == expected.shape
  assert torch.allclose(out, expected)
  assert _AttnBackendRegistry.get_backend(_AttnBackend.NATIVE) is _native_attention


def test_diffusers_bridge_registers_cache_dit_native_backend():
  registered = _register_cache_dit_attn_backends_to_diffusers()

  assert _AttnBackend.NATIVE.value in registered
  assert callable(_AttentionBackendRegistry._backends[AttentionBackendName.NATIVE])
  assert _AttentionBackendRegistry._is_context_parallel_available(AttentionBackendName.NATIVE)


def test_maybe_register_custom_attn_backends_is_not_env_gated(monkeypatch):
  monkeypatch.setattr(ENV, "CACHE_DIT_CUSTOM_ATTN_ALREADY_DISPATCH", False)
  monkeypatch.setattr(
    attention,
    "_register_cache_dit_attn_backends_to_diffusers",
    lambda: [_AttnBackend.NATIVE.value],
  )

  attention._maybe_register_custom_attn_backends()

  assert ENV.CACHE_DIT_CUSTOM_ATTN_ALREADY_DISPATCH is True


def test_default_active_backend_uses_cache_dit_env(monkeypatch):
  monkeypatch.setenv("CACHE_DIT_ATTN_BACKEND", _AttnBackend.SAGE.value)
  monkeypatch.setenv("DIFFUSERS_ATTN_BACKEND", _AttnBackend.NATIVE.value)

  assert _default_active_backend() is _AttnBackend.SAGE


def test_dispatch_attention_fn_passes_cp_config_to_backend(monkeypatch):
  query = torch.randn(2, 5, 3, 4)
  key = torch.randn(2, 5, 3, 4)
  value = torch.randn(2, 5, 3, 4)
  captured = {}
  cp_config = _ContextParallelConfig(ring_degree=2, ulysses_degree=1)

  def _capture_backend(query, key, value, _cp_config=None):
    captured["cp_config"] = _cp_config
    return query

  monkeypatch.setitem(_AttnBackendRegistry._backends, _AttnBackend.NATIVE, _capture_backend)
  monkeypatch.setitem(
    _AttnBackendRegistry._supported_arg_names,
    _AttnBackend.NATIVE,
    {"query", "key", "value", "_cp_config"},
  )

  out = _dispatch_attention_fn(
    query,
    key,
    value,
    backend=_AttnBackend.NATIVE,
    cp_config=cp_config,
  )

  assert out is query
  assert captured["cp_config"] is cp_config


def test_dispatch_attention_fn_legacy_parallel_config_alias_is_normalized(monkeypatch):
  query = torch.randn(2, 5, 3, 4)
  key = torch.randn(2, 5, 3, 4)
  value = torch.randn(2, 5, 3, 4)
  captured = {}
  legacy_parallel_config = SimpleNamespace(
    ring_degree=2,
    ulysses_degree=1,
    convert_to_fp32=True,
    rotate_method="p2p",
    mesh=None,
    ulysses_anything=False,
    ulysses_float8=False,
    ulysses_async=False,
    extra_kwargs=None,
  )

  def _capture_backend(query, key, value, _cp_config=None):
    captured["cp_config"] = _cp_config
    return query

  monkeypatch.setitem(_AttnBackendRegistry._backends, _AttnBackend.NATIVE, _capture_backend)
  monkeypatch.setitem(
    _AttnBackendRegistry._supported_arg_names,
    _AttnBackend.NATIVE,
    {"query", "key", "value", "_cp_config"},
  )

  out = _dispatch_attention_fn(
    query,
    key,
    value,
    backend=_AttnBackend.NATIVE,
    parallel_config=legacy_parallel_config,
  )

  assert out is query
  assert isinstance(captured["cp_config"], _ContextParallelConfig)
  assert captured["cp_config"].ring_degree == 2
  assert captured["cp_config"].ulysses_degree == 1


def test_diffusers_bridge_translates_parallel_config_to_cp_config(monkeypatch):
  query = torch.randn(2, 5, 3, 4)
  key = torch.randn(2, 5, 3, 4)
  value = torch.randn(2, 5, 3, 4)
  captured = {}
  legacy_parallel_config = SimpleNamespace(
    ring_degree=2,
    ulysses_degree=1,
    convert_to_fp32=True,
    rotate_method="p2p",
    mesh=None,
    ulysses_anything=False,
    ulysses_float8=False,
    ulysses_async=False,
    extra_kwargs=None,
  )

  def _capture_backend(query, key, value, _cp_config=None, **kwargs):
    captured["cp_config"] = _cp_config
    return query

  monkeypatch.setitem(_AttnBackendRegistry._backends, _AttnBackend.NATIVE, _capture_backend)
  monkeypatch.setitem(_AttnBackendRegistry._constraints, _AttnBackend.NATIVE, [])
  monkeypatch.setitem(
    _AttnBackendRegistry._supported_arg_names,
    _AttnBackend.NATIVE,
    {"query", "key", "value", "_cp_config"},
  )

  _register_cache_dit_attn_backends_to_diffusers()
  backend_fn = _AttentionBackendRegistry._backends[AttentionBackendName.NATIVE]

  out = backend_fn(
    query,
    key,
    value,
    _parallel_config=legacy_parallel_config,
  )

  assert out is query
  assert isinstance(captured["cp_config"], _ContextParallelConfig)
  assert captured["cp_config"].ring_degree == 2


def test_flash_backend_is_registered_when_available():
  backend = _AttnBackendRegistry.get_backend(_AttnBackend.FLASH)

  if backend is None:
    pytest.skip("flash-attn is not available in this environment")

  assert callable(backend)
  assert _AttnBackendRegistry.is_context_parallel_available(_AttnBackend.FLASH)


def test_diffusers_bridge_registers_cache_dit_flash_backend(monkeypatch):
  query = torch.randn(2, 5, 3, 4)
  key = torch.randn(2, 5, 3, 4)
  value = torch.randn(2, 5, 3, 4)
  captured = {}
  legacy_parallel_config = SimpleNamespace(
    ring_degree=2,
    ulysses_degree=1,
    convert_to_fp32=True,
    rotate_method="p2p",
    mesh=None,
    ulysses_anything=False,
    ulysses_float8=False,
    ulysses_async=False,
    extra_kwargs=None,
  )

  def _capture_backend(query, key, value, _cp_config=None, **kwargs):
    captured["cp_config"] = _cp_config
    return query

  monkeypatch.setattr(
    _AttnBackendRegistry,
    "_backends",
    {
      **_AttnBackendRegistry._backends,
      _AttnBackend.FLASH: _capture_backend,
    },
  )
  monkeypatch.setattr(
    _AttnBackendRegistry,
    "_constraints",
    {
      **_AttnBackendRegistry._constraints,
      _AttnBackend.FLASH: [],
    },
  )
  monkeypatch.setattr(
    _AttnBackendRegistry,
    "_supported_arg_names",
    {
      **_AttnBackendRegistry._supported_arg_names,
      _AttnBackend.FLASH: {"query", "key", "value", "_cp_config"},
    },
  )
  monkeypatch.setattr(
    _AttnBackendRegistry,
    "_bridge_to_diffusers",
    set(_AttnBackendRegistry._bridge_to_diffusers) | {_AttnBackend.FLASH.value},
  )
  monkeypatch.setattr(
    _AttnBackendRegistry,
    "_supports_context_parallel",
    set(_AttnBackendRegistry._supports_context_parallel) | {_AttnBackend.FLASH.value},
  )

  registered = _register_cache_dit_attn_backends_to_diffusers()
  backend_fn = _AttentionBackendRegistry._backends[AttentionBackendName.FLASH]

  out = backend_fn(
    query,
    key,
    value,
    _parallel_config=legacy_parallel_config,
  )

  assert _AttnBackend.FLASH.value in registered
  assert out is query
  assert isinstance(captured["cp_config"], _ContextParallelConfig)
  assert captured["cp_config"].ring_degree == 2
  assert _AttentionBackendRegistry._is_context_parallel_available(AttentionBackendName.FLASH)
