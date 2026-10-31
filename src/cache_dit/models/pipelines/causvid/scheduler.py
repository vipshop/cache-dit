"""Flow-matching sigma grid used by CausVid-Wan.

CausVid does not run a scheduler loop: it looks up the sigma of a fixed DMD timestep table
(`1000 / 757 / 522`), converts the model output to `x0` and re-noises with the next sigma. The lookup
grid is a shifted linear sigma ramp over the 1000 training timesteps, which is *not* what
`diffusers.FlowMatchEulerDiscreteScheduler` produces, so the grid is reproduced here.

Ported from `causvid/models/wan/flow_match.py` (itself from DiffSynth-Studio), Apache-2.0.
"""

from typing import Optional

import torch


class CausVidFlowMatchScheduler:
  """Shifted linear sigma ramp plus the `x0`/`add_noise` helpers used by the sampler.

  :param num_train_timesteps: Size of the training grid the DMD timesteps index into.
  :param shift: Flow-matching shift; the released checkpoint uses 8.0.
  :param sigma_max: Largest sigma of the ramp.
  :param sigma_min: Smallest sigma of the ramp; CausVid trains with 0.0.
  :param extra_one_step: Append one extra ramp point before dropping the tail, matching upstream.
  """

  def __init__(
    self,
    num_train_timesteps: int = 1000,
    shift: float = 8.0,
    sigma_max: float = 1.0,
    sigma_min: float = 0.0,
    extra_one_step: bool = True,
  ) -> None:
    self.num_train_timesteps = num_train_timesteps
    self.shift = shift
    self.sigma_max = sigma_max
    self.sigma_min = sigma_min
    self.extra_one_step = extra_one_step
    self.set_timesteps(num_train_timesteps)

  def set_timesteps(self, num_inference_steps: int = 1000) -> None:
    """Rebuild the sigma grid.

    :param num_inference_steps: Number of ramp points, i.e. the training timestep count.
    """
    sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min)
    if self.extra_one_step:
      sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
    else:
      sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
    self.sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
    self.timesteps = self.sigmas * self.num_train_timesteps

  def sigma(self, timestep: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """Look up the sigma of each timestep by nearest grid point.

    :param timestep: Diffusion timesteps shaped `(B,)`.
    :param device: Device of the returned sigma; defaults to the timestep device.
    :returns: Sigma shaped `(B,)`; callers broadcast it over their own latent rank.
    """
    device = timestep.device if device is None else device
    sigmas = self.sigmas.to(device)
    timesteps = self.timesteps.to(device)
    timestep = timestep.to(device)
    timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
    return sigmas[timestep_id]

  def add_noise(
    self,
    original_samples: torch.Tensor,
    noise: torch.Tensor,
    timestep: torch.Tensor,
  ) -> torch.Tensor:
    """Corrupt a clean latent with the sigma of `timestep`.

    :param original_samples: Clean latent, any rank with the batch on dim 0.
    :param noise: Gaussian noise with the latent shape.
    :param timestep: Target timesteps shaped `(B,)`.
    :returns: The corrupted latent, in the dtype of `noise`.
    """
    sigma = self.sigma(timestep, device=noise.device).view(-1, *([1] * (noise.ndim - 1)))
    return ((1 - sigma) * original_samples + sigma * noise).type_as(noise)
