"""Regression tests for Qwen-Image VAE-P tile geometry selection."""
from types import SimpleNamespace

from cache_dit.distributed.autoencoders.autoencoder_kl_qwen_image import _set_tile_geometry

_DEFAULT = (256, 224, 192, 160)


def _vae():
  return SimpleNamespace(
    tile_sample_min_height=_DEFAULT[0],
    tile_sample_min_width=_DEFAULT[1],
    tile_sample_stride_height=_DEFAULT[2],
    tile_sample_stride_width=_DEFAULT[3],
  )


def _geometry(vae):
  return (
    vae.tile_sample_min_height,
    vae.tile_sample_min_width,
    vae.tile_sample_stride_height,
    vae.tile_sample_stride_width,
  )


def test_large_image_default_geometry_is_unchanged():
  vae = _vae()
  _set_tile_geometry(vae, 1024, 1024, 512, 384, _DEFAULT)
  assert _geometry(vae) == (512, 512, 384, 384)


def test_qwen21_optimized_geometry_is_limited_to_1024_square():
  optimized = (1024, 1024, 640, 576)
  vae = _vae()
  _set_tile_geometry(vae, 1024, 1024, 512, 384, _DEFAULT, optimized)
  assert _geometry(vae) == (640, 640, 576, 576)

  _set_tile_geometry(vae, 1536, 1536, 512, 384, _DEFAULT, optimized)
  assert _geometry(vae) == (512, 512, 384, 384)


def test_small_image_restores_original_geometry_after_large_image():
  vae = _vae()
  _set_tile_geometry(vae, 1024, 1024, 512, 384, _DEFAULT, (1024, 1024, 640, 576))
  assert _geometry(vae) == (640, 640, 576, 576)

  _set_tile_geometry(vae, 768, 768, 512, 384, _DEFAULT, (1024, 1024, 640, 576))
  assert _geometry(vae) == _DEFAULT
