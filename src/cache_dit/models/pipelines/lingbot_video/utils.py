"""Sigma-schedule helpers for LingBot-Video.

Vendored from the official `lingbot_video/utils.py` (https://github.com/Robbyant/lingbot-video,
Apache-2.0). Only the flow-matching sigma builder survives: the batch-CFG prompt padding helper is
dropped together with the batched-CFG branch, and the video/caption IO helpers are pipeline-agnostic
and not needed here.
"""

from typing import Optional, Sequence

import numpy as np

LOW_NOISE_TAIL_V1_DEFAULT_STEPS = 2


def validate_refiner_sigmas(
  sigmas: Sequence[float] | np.ndarray,
  t_thresh: Optional[float] = None,
) -> np.ndarray:
  arr = np.asarray(list(sigmas), dtype=np.float64)
  if arr.ndim != 1 or arr.size == 0:
    raise ValueError("refiner sigma schedule must be a non-empty 1D list")
  if not np.all(np.isfinite(arr)):
    raise ValueError("refiner sigma schedule contains non-finite values")
  if np.any(arr < 0.0) or np.any(arr > 1.0):
    raise ValueError(f"refiner sigma schedule values must be in [0, 1], got {arr.tolist()}")
  if arr.size > 1 and not np.all(np.diff(arr) < 0.0):
    raise ValueError(f"refiner sigma schedule must be strictly descending, got {arr.tolist()}")
  if t_thresh is not None and abs(float(arr[0]) - float(t_thresh)) > 1e-6:
    raise ValueError(
      f"refiner sigma schedule must start at t_thresh={float(t_thresh)}, got {float(arr[0])}")
  return arr


def compute_refiner_sigmas(
  *,
  sigma_max: float,
  sigma_min: float,
  num_inference_steps: int,
  shift: float,
  t_thresh: Optional[float],
  tail_steps: int = 0,
) -> Optional[np.ndarray]:
  """Build the low-noise refiner sigma schedule, or `None` for a plain text-to-X run.

  :param sigma_max: Scheduler sigma at the first timestep.
  :param sigma_min: Scheduler sigma at the last timestep.
  :param num_inference_steps: Requested steps before the schedule is trimmed at `t_thresh`.
  :param shift: Flow-matching shift applied to the linear sigma ramp.
  :param t_thresh: Start sigma of the refiner pass; `None` disables the refiner schedule.
  :param tail_steps: Extra evenly spaced steps appended below the last trimmed sigma.
  :returns: Strictly descending sigmas in `[0, 1]`, or `None` when `t_thresh` is `None`.
  """
  if t_thresh is None:
    return None
  t_value = float(t_thresh)
  if not (0.0 < t_value <= 1.0):
    raise ValueError(f"refiner t_thresh must lie in (0, 1], got {t_value}")
  steps = int(num_inference_steps)
  if steps < 1:
    raise ValueError(f"num_inference_steps must be >= 1, got {steps}")
  tail = int(tail_steps or 0)
  if tail < 0:
    raise ValueError(f"refiner_sigma_tail_steps must be >= 0, got {tail}")

  base = np.linspace(float(sigma_max), float(sigma_min), steps + 1).copy()[:-1]
  shift_value = float(shift)
  shifted = shift_value * base / (1.0 + (shift_value - 1.0) * base)
  eps = 1e-6
  sigmas = shifted[shifted <= t_value + eps]
  if sigmas.size == 0 or abs(float(sigmas[0]) - t_value) > eps:
    sigmas = np.concatenate([[t_value], sigmas])
  if tail > 0:
    start = float(sigmas[-1])
    stop = min(float(sigma_min), start)
    extra = np.linspace(start, stop, tail + 2, dtype=np.float64)[1:-1]
    sigmas = np.concatenate([sigmas, extra])
  return validate_refiner_sigmas(sigmas, t_value).astype(np.float32)
