"""Flow solvers used by SANA-Video 2.0.

The text-to-video branch reproduces the official ``DPMS`` run: ``skip_type="time_uniform_flow"``,
``order=2``, ``method="multistep"``, ``model_type="flow"``, with first-order endpoint steps, which
is the recipe in ``diffusion/model/dpm_solver.py`` of NVlabs/Sana. The text-image-to-video branch
follows the official ``LTXFlowEuler``: a ``FlowMatchEulerDiscreteScheduler`` driven with per-frame
timesteps so the conditioning frame stays clean.
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
from diffusers import FlowMatchEulerDiscreteScheduler


def sample_flow_dpm(
    predict_noise: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    latents: torch.Tensor,
    steps: int,
    shift: float,
    callback: Optional[Callable[[int, torch.Tensor, torch.Tensor], None]] = None) -> torch.Tensor:
  """Second-order multistep DPM-Solver++ with first-order endpoint steps.

  :param predict_noise: ``f(latents, sigma) -> eps``; called once per step.
  :param latents: Initial noise `[B, C, F, H, W]` (fp32 upstream).
  :param steps: Number of solver steps; at least 2.
  :param shift: Flow shift of the sigma schedule (12.0 for the 720p checkpoint).
  :param callback: Optional ``f(index, sigma_next, latents)`` hook used for progress bars.
  :returns: Denoised latents.
  """
  if steps < 2:
    raise ValueError("Flow DPM-Solver requires at least two inference steps")
  betas = torch.linspace(1.0, 0.001, steps + 1, device="cpu").to(latents.device)
  sigmas = 1.0 - betas
  times = (shift * sigmas / (1 + (shift - 1) * sigmas)).flip(0)
  previous_time = previous_prediction = None
  x = latents
  for index in range(steps):
    s = times[index].reshape((1, ) * x.ndim)
    t = times[index + 1].reshape((1, ) * x.ndim)
    noise = predict_noise(x, times[index])
    prediction = (x - s * noise) / (1 - s)
    lambda_s = torch.log(1 - s) - torch.log(s)
    lambda_t = torch.log(1 - t) - torch.log(t)
    h = lambda_t - lambda_s
    alpha_t = torch.exp(torch.log(1 - t))
    phi = torch.expm1(-h)
    if index == 0 or index == steps - 1:
      x = t / s * x - alpha_t * phi * prediction
    else:
      lambda_previous = torch.log(1 - previous_time) - torch.log(previous_time)
      ratio = (lambda_s - lambda_previous) / h
      derivative = (1.0 / ratio) * (prediction - previous_prediction)
      x = t / s * x - alpha_t * phi * prediction - 0.5 * (alpha_t * phi) * derivative
    previous_time, previous_prediction = s, prediction
    if callback is not None:
      callback(index, times[index + 1], x)
  return x


def sample_ltx_euler(
    predict_flow: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    latents: torch.Tensor,
    steps: int,
    shift: float,
    condition_frames: int = 1,
    callback: Optional[Callable[[int, torch.Tensor, torch.Tensor], None]] = None) -> torch.Tensor:
  """Flow Euler with the conditioning latent frame(s) held clean.

  :param predict_flow: ``f(latents, per_frame_timestep) -> velocity``.
  :param latents: Initial noise with the conditioning frame already written in.
  :param steps: Number of Euler steps.
  :param shift: Flow shift of the schedule.
  :param condition_frames: Number of leading latent frames that stay untouched.
  :param callback: Optional ``f(index, sigma, latents)`` hook used for progress bars.
  :returns: Denoised latents.
  """
  with torch.device("cpu"):
    scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)
    scheduler.set_timesteps(steps, device=latents.device)
  condition_mask = torch.zeros_like(latents, dtype=torch.float32)
  condition_mask[:, :, :condition_frames] = 1
  for index, time in enumerate(scheduler.timesteps):
    timestep = torch.minimum(time.expand(latents.shape).float(), (1 - condition_mask) * 1000.0)
    prediction = predict_flow(latents, timestep[:, :1, :, :1, :1])
    batch, channels = latents.shape[:2]
    updated = (scheduler.step(
      -prediction.reshape(batch, channels, -1).transpose(1, 2),
      time,
      latents.reshape(batch, channels, -1).transpose(1, 2),
      per_token_timesteps=timestep.reshape(batch, channels, -1)[:, 0],
      return_dict=False,
    )[0].transpose(1, 2).reshape(latents.shape))
    denoise_mask = time / 1000 - 1e-6 < (1.0 - condition_mask)
    latents = torch.where(denoise_mask, updated, latents).to(latents.dtype)
    if callback is not None:
      callback(index, time, latents)
  return latents


__all__ = ["sample_flow_dpm", "sample_ltx_euler"]
