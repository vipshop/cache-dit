from cache_dit.caching import ForwardPattern
from cache_dit.caching.block_adapters.block_registers import BlockAdapterRegister
from cache_dit.models.pipelines.ming_image.transformer import MingImageTransformer2DModel


def _transformer():
  return MingImageTransformer2DModel(
    in_channels=4,
    dim=64,
    n_layers=2,
    n_refiner_layers=1,
    n_heads=2,
    n_kv_heads=2,
    cap_feat_dim=8,
    axes_dims=(8, 12, 12),
    axes_lens=(256, 64, 64),
  )


def test_ming_adapter_locks_main_stack_and_joint_cfg():
  transformer = _transformer()
  adapter = BlockAdapterRegister.get_adapter(transformer)

  assert adapter is not None
  assert adapter.transformer is transformer
  assert adapter.blocks is transformer.layers
  assert adapter.blocks_name == "layers"
  assert adapter.forward_pattern == ForwardPattern.Pattern_3
  assert adapter.check_forward_pattern is False
  assert adapter.has_separate_cfg is False
  assert transformer._repeated_blocks == ["ZImageTransformerBlock"]


def test_ming_adapter_is_registered_before_zimage_fallback():
  _, names = BlockAdapterRegister.supported_pipelines()
  assert "MingImage" in names
  assert names.index("MingImage") < names.index("ZImage")
