"""CPU checks for the trimmed Boogu-Image pipeline."""

import pytest
import torch

from cache_dit.models.pipelines.boogu_image.pipeline_boogu import BooguImagePipeline
from cache_dit.models.pipelines.boogu_image.pipeline_boogu import _resolve_dtypes


def _bare_pipe() -> BooguImagePipeline:
  # No components: the guards under test run before anything touches them.
  return BooguImagePipeline(transformer=None, vae=None, scheduler=None, mllm=None, processor=None)


def test_resolve_dtypes_per_component() -> None:
  dtypes = _resolve_dtypes({"default": torch.bfloat16, "vae": torch.float32})
  assert dtypes == {
    "transformer": torch.bfloat16,
    "vae": torch.float32,
    "mllm": torch.bfloat16,
  }
  assert _resolve_dtypes(torch.bfloat16)["mllm"] is torch.bfloat16
  assert _resolve_dtypes("auto")["vae"] is None


def test_from_pretrained_rejects_device_map(tmp_path) -> None:
  with pytest.raises(NotImplementedError, match="device_map"):
    BooguImagePipeline.from_pretrained(str(tmp_path), device_map="balanced")


def test_call_rejects_instruction_rewriter() -> None:
  pipe = _bare_pipe()
  with pytest.raises(NotImplementedError, match="instruction rewriter"):
    pipe(instruction="a cat", use_rewrite_text_instruction=True)


def test_call_rejects_remote_rewriter() -> None:
  pipe = _bare_pipe()
  with pytest.raises(NotImplementedError, match="instruction rewriter"):
    pipe(instruction="a cat", use_dashscope_remote_rewriting=True)


def test_call_rejects_boosted_orthogonal_guidance() -> None:
  pipe = _bare_pipe()
  with pytest.raises(NotImplementedError, match="Boosted orthogonal guidance"):
    pipe(instruction="a cat", use_boosted_orthogonal_guidance=True)


def test_cached_call_requires_two_forwards_per_step() -> None:
  pipe = _bare_pipe()

  class _CachedTransformer:

    _is_cached = True

  pipe.transformer = _CachedTransformer()

  # `guidance_scale == 1` would run a single forward per step.
  with pytest.raises(ValueError, match="exactly two transformer forwards"):
    pipe(instruction="a cat", text_guidance_scale=1.0)

  # Double guidance (text + image) would run three.
  from PIL import Image

  ref = Image.new("RGB", (8, 8), color=(128, 128, 128))
  with pytest.raises(ValueError, match="exactly two transformer forwards"):
    pipe(instruction="a cat",
         input_images=[[ref]],
         text_guidance_scale=4.0,
         image_guidance_scale=2.0)


def test_example_imports_the_vendored_pipeline() -> None:
  import inspect

  from cache_dit._utils.examples import boogu_image_edit_example

  source = inspect.getsource(boogu_image_edit_example)
  assert "from ..models.pipelines.boogu_image import BooguImagePipeline" in source
  # The third-party package must not come back through the example.
  assert "from boogu." not in source
