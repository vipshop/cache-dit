import pytest

from cache_dit.caching.block_adapters.adapters import boogu_image_adapter
from cache_dit.caching.forward_pattern import ForwardPattern
from cache_dit.models.pipelines.boogu_image.transformer_boogu import (
  BooguImageTransformer2DModel, )


class _FakePipe:

  def __init__(self, transformer) -> None:
    self.transformer = transformer

  def __call__(self, **kwargs):  # pragma: no cover - never invoked here
    raise AssertionError("adapter tests must not run the pipeline")


def _tiny_transformer() -> BooguImageTransformer2DModel:
  return BooguImageTransformer2DModel(
    patch_size=2,
    in_channels=4,
    hidden_size=96,
    num_layers=3,
    num_double_stream_layers=1,
    num_refiner_layers=1,
    num_attention_heads=4,
    num_kv_heads=2,
    multiple_of=32,
    axes_dim_rope=(8, 8, 8),
    axes_lens=(64, 64, 64),
    instruction_feature_configs=dict(instruction_feat_dim=32,
                                     reduce_type="mean",
                                     num_instruction_feat_layers=1),
  )


def test_boogu_adapter_wires_both_block_stacks() -> None:
  transformer = _tiny_transformer()
  adapter = boogu_image_adapter(_FakePipe(transformer))

  assert adapter.transformer is transformer
  assert user_blocks(adapter) == [
    transformer.double_stream_layers,
    transformer.single_stream_layers,
  ]


def user_blocks(adapter):
  blocks = adapter.blocks
  return list(blocks) if isinstance(blocks, (list, tuple)) else [blocks]


def test_boogu_adapter_patterns_and_cfg_flags() -> None:
  adapter = boogu_image_adapter(_FakePipe(_tiny_transformer()))

  patterns = adapter.forward_pattern
  assert list(patterns) == [ForwardPattern.Pattern_0, ForwardPattern.Pattern_3]
  # Two forward passes per step (conditional + unconditional).
  assert adapter.has_separate_cfg is True
  # The double-stream block takes `(img_hidden_states, instruct_hidden_states)`,
  # so the pattern signature check stays off on purpose.
  assert adapter.check_forward_pattern is False


def test_boogu_adapter_rejects_foreign_transformer() -> None:

  class _Other:

    double_stream_layers = [object()]
    single_stream_layers = [object()]

  with pytest.raises(TypeError, match="vendored BooguImageTransformer2DModel"):
    boogu_image_adapter(_FakePipe(_Other()))


def test_boogu_repeated_blocks_only_lists_single_stream() -> None:
  assert BooguImageTransformer2DModel._repeated_blocks == ["BooguImageSingleStreamTransformerBlock"]
