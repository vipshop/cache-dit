"""Tests for SageAttention-3 backend registration and dispatch."""
import torch

from diffusers.models.modeling_utils import ModelMixin
from cache_dit.attention import set_attn_backend
from cache_dit.attention.backends.register import _AttnBackend, _AttnBackendRegistry


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


def test_sage3_backend_is_registered():
  """_AttnBackend.SAGE3 must exist and normalize to 'sage3'."""
  assert _AttnBackend.SAGE3.value == "sage3"
  normalized = _AttnBackendRegistry.normalize_backend("sage3")
  assert normalized == _AttnBackend.SAGE3


def test_sage3_backend_is_registered_in_registry():
  """The SAGE3 backend fn must be present in the registry."""
  backend_fn = _AttnBackendRegistry.get_backend(_AttnBackend.SAGE3)
  assert backend_fn is not None, "SAGE3 backend not registered"
  assert callable(backend_fn)


def test_sage3_supports_context_parallel():
  """SAGE3 must advertise context parallel support."""
  assert _AttnBackendRegistry.is_context_parallel_available(_AttnBackend.SAGE3)


def test_set_attn_backend_sage3_diffusers():
  """set_attn_backend with 'sage3' must dispatch correctly on diffusers modules."""
  module = _DummyDiffusersModule()
  set_attn_backend(module, _AttnBackend.SAGE3.value)
  assert module.backends == [_AttnBackend.SAGE3.value]


def test_set_attn_backend_sage3_non_diffusers():
  """set_attn_backend with 'sage3' must set _attention_backend on local modules."""
  module = _DummyNonDiffusersTransformer()
  set_attn_backend(module, _AttnBackend.SAGE3.value)
  assert module.attn._attention_backend == _AttnBackend.SAGE3.value
  assert module.attn.processor._attention_backend == _AttnBackend.SAGE3.value
