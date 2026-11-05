"""Tests for the vendored PixelUMM decoder and its checkpoint loaders.

Two things here are easy to get wrong and expensive to notice late:

* the fp8 checkpoint variant stores its big projections as ``*.fp8_weight`` (float8_e4m3fn) plus
  ``*.fp8_scale`` block scales, so a plain ``load_state_dict`` sees missing keys, and a wrong
  dequant silently shifts every decoded frame;
* the decode windows overlap, and the cosine blend has to be a partition of unity or the seams
  flicker.
"""
import pytest
import torch
from safetensors.torch import save_file

from cache_dit.models.pipelines.hy_nomas.components import (
  NomasPixelUMMDecoder,
  PixelUMMArch,
  _dequant_blockwise,
  load_pixelumm_state_dict,
  merge_windows,
  patchify,
  unpatchify,
  window_starts,
)

TINY = PixelUMMArch(
  hidden_size=64,
  intermediate_size=128,
  num_layers=1,
  num_heads=4,
  num_kv_heads=2,
  head_dim=8,
  local_attention_tile=0,
)
BLOCK = 128


def test_patchify_roundtrip():
  video = torch.randn(1, 3, 8, 32, 32)
  tokens = patchify(video)
  assert tokens.shape == (1, 2 * 2 * 2, 3072)
  restored = unpatchify(tokens, (2, 2, 2))
  assert torch.equal(restored, video)


def test_dequant_blockwise_matches_explicit_loop():
  torch.manual_seed(0)
  weight = torch.randn(BLOCK * 2, BLOCK * 3).to(torch.float8_e4m3fn)
  scale = torch.rand(2, 3, dtype=torch.float32) + 0.5
  got = _dequant_blockwise(weight, scale)
  expected = torch.empty(BLOCK * 2, BLOCK * 3, dtype=torch.float32)
  for bi in range(2):
    for bj in range(3):
      rows = slice(bi * BLOCK, (bi + 1) * BLOCK)
      cols = slice(bj * BLOCK, (bj + 1) * BLOCK)
      expected[rows, cols] = weight[rows, cols].to(torch.float32) * scale[bi, bj]
  assert torch.equal(got, expected)


def test_dequant_rejects_untileable_scale():
  weight = torch.zeros(BLOCK, BLOCK, dtype=torch.float32).to(torch.float8_e4m3fn)
  with pytest.raises(ValueError, match="does not tile"):
    _dequant_blockwise(weight, torch.ones(3, 3))


def _reference_state(arch: PixelUMMArch, fp8: bool):
  """Checkpoint-shaped state (``decoder.`` prefix) plus the values a correct load must produce."""
  torch.manual_seed(1)
  module = NomasPixelUMMDecoder(arch).decoder
  state, expected = {}, {}
  for name, value in module.state_dict().items():
    if fp8 and name.endswith("weight") and value.ndim == 2 and value.shape[0] % BLOCK == 0 \
        and value.shape[1] % BLOCK == 0:
      blocks = (value.shape[0] // BLOCK, value.shape[1] // BLOCK)
      scale = torch.rand(blocks, dtype=torch.float32) * 0.1 + 0.01
      tiled = scale.repeat_interleave(BLOCK, dim=1).repeat_interleave(BLOCK, dim=0)
      quant = (value.float() / tiled).to(torch.float8_e4m3fn)
      state[f"decoder.{name[:-len('weight')]}fp8_weight"] = quant
      state[f"decoder.{name[:-len('weight')]}fp8_scale"] = scale
      expected[name] = quant.to(torch.float32) * tiled
      continue
    state[f"decoder.{name}"] = value.to(torch.float32) if value.is_floating_point() else value
    expected[name] = state[f"decoder.{name}"]
  return state, expected


@pytest.mark.parametrize("fp8", [False, True])
def test_loader_roundtrips_either_variant(tmp_path, fp8):
  state, expected = _reference_state(TINY, fp8)
  path = tmp_path / ("fp8.safetensors" if fp8 else "fp32.safetensors")
  save_file(state, str(path))
  loaded = NomasPixelUMMDecoder.from_checkpoint(str(path), dtype=torch.float32, arch=TINY)
  for name, value in loaded.decoder.state_dict().items():
    assert torch.equal(value, expected[name]), name


def test_loader_refuses_unconverted_module(tmp_path):
  """`load_state_dict` copies into the existing dtype, so the conversion must be enforced."""
  from cache_dit.models.pipelines.hy_nomas.components import _load_pixelumm_weights

  state, _ = _reference_state(TINY, False)
  path = tmp_path / "fp32.safetensors"
  save_file(state, str(path))
  module = NomasPixelUMMDecoder(TINY)  # left in float32 on purpose
  with pytest.raises(RuntimeError, match="convert the decoder to torch.bfloat16"):
    _load_pixelumm_weights(module, str(path), torch.bfloat16)
  # The matching dtype still loads, so the guard rejects only the silent case.
  _load_pixelumm_weights(module, str(path), torch.float32)


def test_decode_contract_and_windows():
  decoder = NomasPixelUMMDecoder(TINY).eval().requires_grad_(False)
  latent = torch.randn(1, 32, 2, 2, 2)
  out = decoder.decode(latent)
  assert out.sample.shape == (1, 3, 8, 64, 64)
  assert torch.isfinite(out.sample).all()
  # Frame-indexed noise makes the decode reproducible without any RNG plumbing.
  assert torch.equal(out.sample, decoder.decode(latent).sample)
  assert window_starts(10, 10, 5) == [0]
  assert window_starts(24, 10, 5) == [0, 5, 10, 14]
  with pytest.raises(ValueError):
    window_starts(10, 10, 11)


def test_merge_windows_is_partition_of_unity():
  # Two 8-frame windows starting at 0 and 6 overlap by 2 frames: 6 + 2 + 6 = 14.
  parts = [torch.ones(1, 1, 8, 1, 1), torch.ones(1, 1, 8, 1, 1)]
  merged = merge_windows(parts, [0, 6])
  assert merged.shape == (1, 1, 14, 1, 1)
  assert torch.allclose(merged, torch.ones_like(merged))
