"""CPU tests for the audio metrics (SI-SDR / SNR / mel PSNR / mel SSIM).

The values are checked against their closed forms on analytic signals, so the tests pin the
definition (mean removal, least-squares rescaling, dynamic-range normalization) rather than just "it
returns a number".
"""

import sys

import numpy as np
import pytest

from cache_dit.metrics import (
  compute_mel_psnr,
  compute_mel_ssim,
  compute_si_sdr,
  compute_snr,
)

_RATE = 24000
_PERIODS = 100
_SAMPLES = 1024


def _reference():
  time = np.arange(_SAMPLES, dtype=np.float32) / _SAMPLES
  return np.cos(2 * np.pi * _PERIODS * time).astype(np.float32)


def _orthogonal_noise():
  time = np.arange(_SAMPLES, dtype=np.float32) / _SAMPLES
  return np.sin(2 * np.pi * _PERIODS * time).astype(np.float32)


def test_si_sdr_and_snr_are_infinite_for_identical_waveforms():
  reference = _reference()
  assert compute_si_sdr(reference, reference.copy())[0] == float("inf")
  assert compute_snr(reference, reference.copy())[0] == float("inf")


def test_si_sdr_is_scale_invariant_but_snr_is_not():
  reference = _reference()
  # SI-SDR rescales the estimate first, so a global gain is free.
  assert compute_si_sdr(reference, 0.25 * reference)[0] == float("inf")
  # SNR keeps the absolute scale: `2x` leaves a residual of the same size as the signal (0 dB).
  assert compute_snr(reference, 2.0 * reference)[0] == pytest.approx(0.0, abs=1e-6)


def test_si_sdr_matches_the_closed_form_for_orthogonal_noise():
  reference = _reference()
  noise = _orthogonal_noise()
  # The noise is orthogonal to the reference over whole periods, so `alpha == 1` and the ratio is
  # set by the noise amplitude alone: 20 * log10(1 / 0.1) = 20 dB.
  si_sdr, n = compute_si_sdr(reference, reference + 0.1 * noise)
  assert n == 1
  assert si_sdr == pytest.approx(20.0, abs=0.05)

  snr, _ = compute_snr(reference, reference + 0.1 * noise)
  assert snr == pytest.approx(20.0, abs=0.05)


def test_length_mismatch_is_truncated_to_the_common_prefix():
  reference = _reference()
  shorter = np.concatenate([reference, np.zeros(64, dtype=np.float32)])
  assert compute_si_sdr(reference, shorter)[0] == float("inf")


def test_sample_rate_mismatch_is_rejected(tmp_path):
  sf = pytest.importorskip("soundfile")
  reference = _reference()
  first = tmp_path / "a.wav"
  second = tmp_path / "b.wav"
  sf.write(first, reference, _RATE)
  sf.write(second, reference, _RATE // 2)
  with pytest.raises(ValueError, match="sample rates"):
    compute_si_sdr(first.as_posix(), second.as_posix())


def test_mel_metrics_are_exact_for_identical_audio():
  pytest.importorskip("torchaudio")
  reference = _reference()
  assert compute_mel_psnr(reference, reference.copy())[0] == float("inf")
  assert compute_mel_ssim(reference, reference.copy())[0] == pytest.approx(1.0, abs=1e-6)


def test_mel_metrics_react_to_added_noise():
  pytest.importorskip("torchaudio")
  reference = _reference()
  noisy = reference + 0.05 * np.random.default_rng(0).normal(size=_SAMPLES).astype(np.float32)
  mel_psnr, _ = compute_mel_psnr(reference, noisy)
  mel_ssim, _ = compute_mel_ssim(reference, noisy)
  assert np.isfinite(mel_psnr) and 0.0 < mel_psnr < 20.0
  # A pure tone has an extremely peaky spectrogram, so the local correlation — and therefore SSIM —
  # can even go negative; only "clearly below the identical value" is asserted here.
  assert mel_ssim < 0.99


def test_audio_directory_pairing_keeps_identical_pairs(tmp_path):
  sf = pytest.importorskip("soundfile")
  reference, rate = _reference(), _RATE
  true_dir = tmp_path / "ref"
  test_dir = tmp_path / "test"
  true_dir.mkdir()
  test_dir.mkdir()
  # `subtype="FLOAT"` keeps the pair bit-exact: PCM_16 quantization would turn a scaled copy into a
  # finite SI-SDR instead of `inf`.
  sf.write(true_dir / "a.wav", reference, rate, subtype="FLOAT")
  sf.write(true_dir / "b.wav", reference, rate, subtype="FLOAT")
  sf.write(test_dir / "a.wav", reference, rate, subtype="FLOAT")
  sf.write(test_dir / "b.wav", 0.5 * reference, rate, subtype="FLOAT")

  value, count = compute_si_sdr(true_dir.as_posix(), test_dir.as_posix())
  # `a.wav` is bit-identical (inf) and is reported through the count instead of being averaged in.
  assert count == 2
  assert value == float("inf")

  # `b.wav` is half the reference, and SNR keeps the scale: 20 * log10(1 / 0.5) = 6.02 dB.
  value, count = compute_snr(true_dir.as_posix(), test_dir.as_posix())
  assert count == 2
  assert value == pytest.approx(20 * np.log10(2.0), abs=1e-3)


def test_cli_runs_audio_metrics_on_a_file_pair(tmp_path, monkeypatch):
  sf = pytest.importorskip("soundfile")
  pytest.importorskip("torchaudio")
  reference, rate = _reference(), _RATE
  first = tmp_path / "ref.wav"
  second = tmp_path / "test.wav"
  sf.write(first, reference, rate)
  sf.write(second, 0.5 * reference, rate)

  from cache_dit.metrics import entrypoint

  monkeypatch.setattr(
    sys,
    "argv",
    ["cache-dit-metrics", "si_sdr", "mel_ssim", "-a1",
     first.as_posix(), "-a2",
     second.as_posix()],
  )
  entrypoint()
