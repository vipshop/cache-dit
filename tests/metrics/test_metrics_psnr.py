"""Regression tests for the PSNR computation of 8-bit image pairs.

`calculate_psnr` used to subtract the uint8 arrays coming out of `cv2.imread`
directly, so negative residuals wrapped modulo 256 and PSNR was inflated by
5-21 dB (the more so the larger the difference). These tests pin the value to
skimage's formula, which is the reference the CLI's `--psnr-type skimage`
branch already produces.
"""

import cv2
import numpy as np
import pytest
from skimage.metrics import peak_signal_noise_ratio

from cache_dit.metrics.metrics import calculate_psnr, compute_psnr_file, set_psnr_type


def _noisy_pair(seed: int = 0, size: int = 32) -> tuple[np.ndarray, np.ndarray]:
  rng = np.random.default_rng(seed)
  true = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
  offset = rng.integers(-48, 49, true.shape)
  test = np.clip(true.astype(np.int16) + offset, 0, 255).astype(np.uint8)
  return true, test


def test_uint8_pair_matches_skimage():
  true, test = _noisy_pair()
  custom = calculate_psnr(true, test)
  reference = peak_signal_noise_ratio(true, test)
  assert custom == pytest.approx(reference, rel=1e-6)


def test_all_negative_residuals_are_not_wrapped():
  # uint8 would wrap `0 - 255` to 1 and report 48.13 dB instead of 0 dB.
  true = np.zeros((8, 8, 3), dtype=np.uint8)
  test = np.full((8, 8, 3), 255, dtype=np.uint8)
  assert calculate_psnr(true, test) == pytest.approx(0.0, abs=1e-6)


def test_identical_images_are_infinite():
  true, _ = _noisy_pair()
  assert calculate_psnr(true, true.copy()) == float("inf")


def test_float_input_keeps_the_same_value():
  true, test = _noisy_pair()
  assert calculate_psnr(true, test) == pytest.approx(calculate_psnr(true.astype(np.float64),
                                                                    test.astype(np.float64)),
                                                     rel=1e-9)


def test_compute_psnr_file_both_types_agree(tmp_path):
  true, test = _noisy_pair()
  true_path = tmp_path / "true.png"
  test_path = tmp_path / "test.png"
  cv2.imwrite(str(true_path), true)
  cv2.imwrite(str(test_path), test)
  # PNG is lossless, so the file-based path must reproduce the in-memory value.
  custom = compute_psnr_file(str(true_path), str(test_path))
  set_psnr_type("skimage")
  try:
    reference = compute_psnr_file(str(true_path), str(test_path))
  finally:
    set_psnr_type("custom")
  assert custom == pytest.approx(reference, rel=1e-6)
