import torch

from cache_dit.distributed import _ContextParallelConfig
from cache_dit.distributed.core import (
  _get_submodule_by_name,
  _normalize_parallel_config,
  validate_context_parallel_attention_backend,
)


class _DiffusersLikeContextParallelConfig:

  def __init__(self):
    self.ring_degree = 2
    self.ulysses_degree = 1
    self.convert_to_fp32 = False
    self.rotate_method = "p2p"
    self.mesh = None
    self.ulysses_anything = False
    self._rank = None
    self._world_size = None
    self._device = None
    self._mesh = None
    self._flattened_mesh = None
    self._ring_mesh = None
    self._ulysses_mesh = None
    self._ring_local_rank = None
    self._ulysses_local_rank = None


class _DiffusersLikeParallelConfig:

  def __init__(self):
    self.context_parallel_config = _DiffusersLikeContextParallelConfig()


class _ModuleDictModel(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.blocks = torch.nn.ModuleDict({
      "proj_in": torch.nn.Linear(4, 4),
      "proj_out": torch.nn.Linear(4, 4),
    })


class _ProcessorlessAttention(torch.nn.Module):

  def forward(self, hidden_states):
    return hidden_states


class _ProcessorlessAttentionModel(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.attn = _ProcessorlessAttention()


def test_normalize_parallel_config_wraps_diffusers_like_config():
  config = _normalize_parallel_config(_DiffusersLikeParallelConfig())

  assert isinstance(config, _ContextParallelConfig)
  assert config.context_parallel_config is config
  assert config.ring_degree == 2
  assert config.ulysses_degree == 1
  assert config.convert_to_fp32 is False


def test_get_submodule_by_name_supports_moduledict_values():
  model = _ModuleDictModel()

  submodules = _get_submodule_by_name(model, "blocks")

  assert isinstance(submodules, list)
  assert [module.__class__.__name__ for module in submodules] == ["Linear", "Linear"]


def test_validate_context_parallel_attention_backend_skips_processorless_attention():
  model = _ProcessorlessAttentionModel()
  config = _ContextParallelConfig(ring_degree=2, ulysses_degree=1)

  validate_context_parallel_attention_backend(
    model,
    config,
    attn_classes_extra=(_ProcessorlessAttention, ),
  )
