import torch

import cache_dit.caching as caching
from diffusers.models.modeling_utils import ModelMixin

from cache_dit.attention import set_attn_backend
from cache_dit.attention._attention_dispatch import _AttnBackend


class _DummyDiffusersModule(ModelMixin):

  def __init__(self):
    super().__init__()
    self.backends: list[str] = []

  def set_attention_backend(self, backend: str) -> None:
    self.backends.append(backend)


class _DummyProcessor(torch.nn.Module):

  def __init__(self):
    super().__init__()


class _DummyAttentionModule(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.processor = _DummyProcessor()
    self._attention_backend = None


class _DummyNonDiffusersTransformer(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.attn = _DummyAttentionModule()


def test_set_attn_backend_is_not_reexported_from_caching_module():
  assert not hasattr(caching, "set_attn_backend")


def test_set_attn_backend_uses_diffusers_api_when_available():
  module = _DummyDiffusersModule()

  set_attn_backend(module, _AttnBackend.NATIVE.value)

  assert module.backends == [_AttnBackend.NATIVE.value]


def test_set_attn_backend_keeps_diffusers_only_backend_strings():
  module = _DummyDiffusersModule()

  set_attn_backend(module, "flash")

  assert module.backends == ["flash"]


def test_set_attn_backend_falls_back_to_local_attention_backend_attr():
  module = _DummyNonDiffusersTransformer()

  set_attn_backend(module, _AttnBackend.NATIVE.value)

  assert module.attn._attention_backend == _AttnBackend.NATIVE.value
  assert module.attn.processor._attention_backend == _AttnBackend.NATIVE.value


def test_set_attn_backend_accepts_cache_dit_flash_backend_for_local_modules():
  module = _DummyNonDiffusersTransformer()

  set_attn_backend(module, _AttnBackend.FLASH.value)

  assert module.attn._attention_backend == _AttnBackend.FLASH.value
  assert module.attn.processor._attention_backend == _AttnBackend.FLASH.value
