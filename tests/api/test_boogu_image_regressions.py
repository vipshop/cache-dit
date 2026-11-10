"""Regression tests for review findings on the vendored Boogu-Image stack.

Each case pins a bug that a review caught and a fix closed, so the same defect
cannot come back silently:

* fused-op predicates must reject CPU tensors and non-RMSNorm modules;
* the vendored `RMSNorm` must accept `elementwise_affine` (the
  `norm_type="rms_norm"` branch of `LuminaLayerNormContinuous` used to raise);
* a custom list schedule must reach the scheduler, not die in `torch.from_numpy`;
* image preprocessing must tolerate either size limit being unset;
* VLM preprocessing must crop the single PIL image instead of iterating it.
"""

import types

import numpy as np
import pytest
import torch
from PIL import Image

from cache_dit.models.pipelines.boogu_image.block_lumina2 import (
  LuminaLayerNormContinuous, )
from cache_dit.models.pipelines.boogu_image.image_processor import (
  BooguImageProcessor, )
from cache_dit.models.pipelines.boogu_image.pipeline_boogu import BooguImagePipeline
from cache_dit.models.pipelines.boogu_image.rms_norm import RMSNorm
from cache_dit.models.pipelines.boogu_image.scheduler import (
  FlowMatchEulerDiscreteScheduler, )


def test_rms_norm_convention_detects_only_rms_norms() -> None:
  from diffusers.models.normalization import RMSNorm as DiffusersRMSNorm

  from cache_dit.models.pipelines.boogu_image import fused_ops

  assert fused_ops._rms_norm_convention(RMSNorm(8)) is False
  assert fused_ops._rms_norm_convention(RMSNorm(8, zero_centered_weight=True)) is True
  # `Attention(qk_norm="rms_norm")` builds diffusers' class, not ours.
  assert fused_ops._rms_norm_convention(DiffusersRMSNorm(8, eps=1e-5)) is False
  # A LayerNorm must never be fused: the kernel does no mean subtraction.
  assert fused_ops._rms_norm_convention(torch.nn.LayerNorm(8)) is None


def test_fused_scale_rejects_cpu_tensor() -> None:
  from cache_dit.models.pipelines.boogu_image import fused_ops

  cpu_scale = torch.ones(1, 8, dtype=torch.bfloat16)
  assert fused_ops._fusable_scale(cpu_scale, 8) is False
  # Same shape/dtype/contiguity as the accepted case; only the device differs.
  assert fused_ops._fusable_scale(cpu_scale.view(1, 8), 8) is False


def test_gated_rms_norm_falls_back_for_cpu_residual() -> None:
  from cache_dit.models.pipelines.boogu_image import fused_ops

  x = torch.randn(1, 4, 8, dtype=torch.bfloat16)
  residual = torch.randn(1, 4, 8, dtype=torch.bfloat16)
  gate = torch.randn(1, 8, dtype=torch.bfloat16)
  norm = RMSNorm(8)

  out = fused_ops.apply_gated_rms_norm(x, norm, gate, residual)
  expected = residual + gate.unsqueeze(1).tanh() * norm(x)
  assert torch.equal(out, expected)


def test_rms_norm_supports_elementwise_affine_false() -> None:
  norm = RMSNorm(8, elementwise_affine=False)
  assert norm.weight is None
  x = torch.randn(2, 3, 8)
  assert torch.allclose(norm(x), torch.nn.functional.rms_norm(x, (8, ), None, 1e-5))


def test_layer_norm_continuous_rms_branch_constructs() -> None:
  layer = LuminaLayerNormContinuous(
    embedding_dim=8,
    conditioning_embedding_dim=8,
    norm_type="rms_norm",
  )
  assert isinstance(layer.norm, RMSNorm)
  out = layer(torch.randn(2, 3, 8), torch.randn(2, 8))
  assert out.shape == (2, 3, 8)


def test_scheduler_accepts_a_list_schedule() -> None:
  scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000,
                                              do_shift=False,
                                              dynamic_time_shift=False)
  scheduler.set_timesteps(timesteps=[1.0, 0.5, 0.0], device="cpu")
  assert torch.equal(scheduler.timesteps, torch.tensor([1.0, 0.5, 0.0]))


def test_image_processor_tolerates_unset_limits() -> None:
  processor = BooguImageProcessor(vae_scale_factor=8)
  # Both limits unset: no ratio math at all, and no upscaling.
  assert processor.get_new_height_width(Image.new("RGB", (96, 64)),
                                        max_pixels=None,
                                        max_side_length=None) == (64, 96)
  # Only the side limit set: the longest side is clamped to 48, aspect kept.
  assert processor.get_new_height_width(Image.new("RGB", (96, 64)),
                                        max_pixels=None,
                                        max_side_length=48) == (32, 48)


def test_preprocess_vlm_input_crops_single_image() -> None:
  processor = BooguImageProcessor(vae_scale_factor=8)
  pipe = types.SimpleNamespace(image_processor=processor)
  image = Image.new("RGB", (64, 64), color=(10, 20, 30))

  out = BooguImagePipeline.preprocess_vlm_input_pil_images(pipe, [image],
                                                           crops_coords=(8, 8, 40, 40))
  assert len(out) == 1
  assert out[0].size == (32, 32)


def test_crop_path_matches_upstream_signature() -> None:
  """`crops_coords` is a single 4-tuple, not a per-image list."""
  processor = BooguImageProcessor(vae_scale_factor=8)
  pipe = types.SimpleNamespace(image_processor=processor)
  out = BooguImagePipeline.preprocess_vlm_input_pil_images(pipe, [Image.new("RGB", (32, 32))],
                                                           crops_coords=(0, 0, 16, 16))
  assert np.asarray(out[0]).shape[0] == out[0].size[1]


@pytest.mark.parametrize("limit", [None, 4096])
def test_preprocess_does_not_raise_for_optional_limits(limit) -> None:
  processor = BooguImageProcessor(vae_scale_factor=8, max_pixels=limit)
  processor.get_new_height_width(Image.new("RGB", (128, 96)))
