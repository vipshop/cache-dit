"""MiniMax-H3 scheduler with the re-noising update the six-step Pika adapters were trained for.

Subclasses diffusers' `MiniMaxH3Scheduler` instead of copying it: the rectified-flow arithmetic and
the shift are unchanged, and four things differ from the installed release, all copied from diffusers
commit `d0b9b1b5a540e5f9df2f8ec236347bf9c5aa7d1f`
(`src/diffusers/schedulers/scheduling_minimax_h3.py`, Apache-2.0):

* `__init__` accepts `stochastic_sampling` and `shift_provided_sigmas`, and registers both to the
  config. The installed release only takes `shift`, so `ConfigMixin.extract_init_dict` would **drop**
  these keys without a word and the ladder would silently sample deterministically;
* `set_timesteps` pushes explicit `sigmas` through the exponential shift when
  `shift_provided_sigmas` is set (the six-step adapters ship a base grid and are trained on the
  shifted one);
* `step` takes the request `generator` and, with `stochastic_sampling`, replaces the Euler blend with
  the re-noising update `x_next = (1 - sigma') * x0 + sigma' * noise`;
* `scale_noise` keeps a scalar `timestep` as a Python float instead of casting it to the sample dtype.
  The installed release casts, and `1 - 0.999` evaluated in bfloat16 is ~2% short of `1e-3`, which the
  bf16 forward amplifies; the only caller (`keyframe_noise_aug = 0.999`) is exactly that case.

Pika H3 Flex holds two instances per request — `shift = 12.0` for video rows and `3.0` for audio rows
— and both are stepped inside one transformer call.
"""

import torch
from diffusers import MiniMaxH3Scheduler
from diffusers.configuration_utils import register_to_config
from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3SchedulerOutput
from diffusers.utils import deprecate
from diffusers.utils.torch_utils import randn_tensor


class PikaH3FlexScheduler(MiniMaxH3Scheduler):
  r"""Rectified-flow scheduler whose optional update is the few-step re-noising one.

  :param shift: Exponential sigma shift; the released checkpoints use `12.0` for video and `3.0` for
    audio.
  :param stochastic_sampling: Predict the clean sample, then re-noise it onto the next sigma with
    fresh noise instead of blending `x_t` and `x0`. The six-step adapters were trained for this.
  :param shift_provided_sigmas: Whether explicit `sigmas` passed to `set_timesteps` are a base grid the
    scheduler shifts, or a finished schedule taken verbatim. Unset keeps the installed release's
    verbatim default with a deprecation warning.
  """

  @register_to_config
  def __init__(
    self,
    shift: float = 12.0,
    stochastic_sampling: bool = False,
    shift_provided_sigmas: bool | None = None,
  ):
    super().__init__(shift=shift)
    if shift <= 0:
      raise ValueError(f"`shift` must be positive, got {shift}.")
    self.num_inference_steps: int | None = None
    self.sigmas: torch.Tensor | None = None
    self.timesteps: torch.Tensor | None = None
    self._shift = float(shift)
    self._step_index: int | None = None
    self._begin_index: int | None = None

  def set_timesteps(
    self,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    sigmas: list[float] | torch.Tensor | None = None,
  ) -> None:
    r"""Build the sigma / timestep schedule.

    :param num_inference_steps: Number of sigma grid points, terminal `0` included. Ignored when
      `sigmas` is given.
    :param device: Device the schedule tensors are moved to. The grid itself is always built on CPU in
      float32 so the schedule does not depend on the accelerator.
    :param sigmas: Explicit grid, strictly decreasing and terminating at `0.0`. With
      `shift_provided_sigmas` it is a base grid in `[0, 1]` that goes through the shift.
    """
    if sigmas is None:
      super().set_timesteps(num_inference_steps=num_inference_steps, device=device)
      return

    sigmas = torch.as_tensor(sigmas, dtype=torch.float64).flatten().cpu()
    if self.config.shift_provided_sigmas is None:
      deprecate(
        "sigmas",
        "0.44.0",
        "`PikaH3FlexScheduler.set_timesteps` takes explicit `sigmas` verbatim unless "
        "`shift_provided_sigmas=True` is set in the scheduler config.",
        standard_warn=False,
      )
    if self.config.shift_provided_sigmas:
      # Shifted in float64 before the float32 cast, which is how the few-step reference builds it.
      sigmas = self._shift * sigmas / (1 + (self._shift - 1) * sigmas)
    sigmas = sigmas.to(torch.float32)
    if sigmas.numel() < 2 or not bool((sigmas[1:] < sigmas[:-1]).all()) or sigmas[-1].item() != 0.0:
      raise ValueError("`sigmas` must hold at least two strictly decreasing values ending at 0.0.")

    self.sigmas = sigmas.to(device=device)
    # t = 1 - sigma, and t = 1 is clean. The terminal sigma has no model evaluation.
    self.timesteps = (1.0 - sigmas[:-1]).to(device=device)
    self.num_inference_steps = int(self.timesteps.numel())
    self._step_index = None
    self._begin_index = None

  def scale_noise(
    self,
    sample: torch.FloatTensor,
    timestep: float | torch.FloatTensor,
    noise: torch.FloatTensor,
  ) -> torch.FloatTensor:
    r"""Rectified-flow forward process, in MiniMax-H3's `t` convention: `x_t = t*x_0 + (1 - t)*noise`.

    A scalar `timestep` is a `noise_aug` level, not a schedule entry, so it is taken at face value and
    kept as a Python float: `1 - 0.999` then evaluates in double precision. Casting it to the sample
    dtype first — what the installed release does — makes `1 - t` come out ~2% short in bfloat16, and
    the bf16 forward turns that into a visibly different conditioning anchor.

    :param sample: The clean sample `x_0`.
    :param timestep: Target time in `[0, 1]`; `1` returns `sample` unchanged.
    :param noise: The noise to mix in.
    :returns: The noised sample.
    """
    if not isinstance(timestep, torch.Tensor):
      return timestep * sample + (1.0 - timestep) * noise
    timestep = timestep.to(device=sample.device, dtype=sample.dtype)
    while timestep.ndim < sample.ndim:
      timestep = timestep.unsqueeze(-1)
    return timestep * sample + (1.0 - timestep) * noise

  def step(
    self,
    model_output: torch.FloatTensor,
    timestep: float | torch.FloatTensor,
    sample: torch.FloatTensor,
    generator: torch.Generator | None = None,
    return_dict: bool = True,
  ):
    r"""Take one Euler step, or one re-noising step with `stochastic_sampling`.

    The model output is a data-ward velocity, so the denoised estimate is `x0 = x_t + (1 - t) * v`.
    With `stochastic_sampling` the update is `x_next = (1 - sigma') * x0 + sigma' * noise` with fresh
    noise from `generator`, everything in float32, so the last step (`sigma' = 0`) returns `x0`.

    :param model_output: The transformer's velocity prediction at `timestep`.
    :param timestep: The current timestep, one of `self.timesteps` (so `timestep == 1 - sigma`).
    :param sample: The current sample `x_t`.
    :param generator: Generator the fresh noise is drawn from with `stochastic_sampling`.
    :param return_dict: Whether to return a `MiniMaxH3SchedulerOutput` instead of a plain tuple.
    :returns: The sample for the next step.
    """
    if isinstance(timestep, int) or (isinstance(timestep, torch.Tensor)
                                     and not timestep.is_floating_point()):
      raise ValueError(
        "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
        " `PikaH3FlexScheduler.step()` is not supported. Make sure to pass one of the"
        " `scheduler.timesteps` values.")

    if self._step_index is None:
      self._step_index = (self.index_for_timestep(timestep)
                          if self._begin_index is None else self._begin_index)

    if self.config.stochastic_sampling:
      # Few-step re-noising: x0 from the grid sigma, then re-noised to the next sigma with fresh noise.
      sigma = self.sigmas[self._step_index].to(device=sample.device, dtype=torch.float32)
      sigma_next = self.sigmas[self._step_index + 1].to(device=sample.device, dtype=torch.float32)
      denoised = sample.to(torch.float32) + sigma * model_output.to(torch.float32)
      noise = randn_tensor(
        sample.shape,
        generator=generator,
        device=sample.device,
        dtype=torch.float32,
      )
      prev_sample = ((1.0 - sigma_next) * denoised + sigma_next * noise).to(dtype=sample.dtype)
      self._step_index += 1
      if not return_dict:
        return (prev_sample, )
      return MiniMaxH3SchedulerOutput(prev_sample=prev_sample)

    return super().step(
      model_output,
      timestep,
      sample,
      return_dict=return_dict,
    )
