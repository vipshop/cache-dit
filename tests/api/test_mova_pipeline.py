"""CPU tests for the MOVA pipeline's cache guards and dtypes.

The guards are the only pipeline code that has to be exercised before the 73 GB of weights exist:
a cached run must see exactly one conditional and one unconditional forward per step, and the
loader must reject `device_map` instead of silently leaving the pipeline on CPU.
"""

import types

import pytest
import torch

from cache_dit.models.pipelines.mova.pipeline import MOVAPipeline, _resolve_dtypes


def _stub_pipe(*, cached: bool) -> MOVAPipeline:
  """Build a pipeline shell whose transformer reports the given caching state.

  :param cached: Value of ``transformer._is_cached``.
  :returns: A `MOVAPipeline` instance without components.
  """

  pipe = MOVAPipeline.__new__(MOVAPipeline)
  pipe.transformer = types.SimpleNamespace(_is_cached=cached)
  return pipe


def test_cached_run_accepts_guidance_above_one():
  """The shipped recipe (guidance 5.0, one forward per branch) is the supported mode."""

  _stub_pipe(cached=True)._check_cache_compatible(5.0, False)


def test_cached_run_rejects_guidance_one():
  """Guidance 1.0 runs a single forward per step, which halves the cache statistics."""

  with pytest.raises(ValueError, match="guidance scale above 1"):
    _stub_pipe(cached=True)._check_cache_compatible(1.0, False)


def test_cached_run_rejects_merged_cfg():
  """A batched cond/uncond forward breaks the same bookkeeping from the other side."""

  with pytest.raises(ValueError, match="cfg_merge"):
    _stub_pipe(cached=True)._check_cache_compatible(5.0, True)


def test_uncached_run_accepts_both_modes():
  """Without caching the pipeline may merge the branches or run guidance-free."""

  pipe = _stub_pipe(cached=False)
  pipe._check_cache_compatible(1.0, True)


def test_resolve_dtypes_keeps_the_audio_vae_in_float32():
  """The DAC decoder is fp32; every other component follows the requested dtype."""

  resolved = _resolve_dtypes(torch.bfloat16)
  assert resolved["default"] is torch.bfloat16
  assert resolved["audio_vae"] is torch.float32

  overridden = _resolve_dtypes({"default": torch.bfloat16, "video_vae": torch.float32})
  assert overridden["video_vae"] is torch.float32
  assert overridden["audio_vae"] is torch.float32


def test_from_pretrained_rejects_device_map(tmp_path):
  """`--device-map-balance` would leave the pipeline on CPU, so the loader fails fast."""

  with pytest.raises(NotImplementedError, match="device-map-balance"):
    MOVAPipeline.from_pretrained(str(tmp_path), torch_dtype=torch.bfloat16, device_map="balanced")
