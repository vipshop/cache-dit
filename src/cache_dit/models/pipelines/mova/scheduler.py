"""Paired flow-matching scheduler of MOVA, vendored from OpenMOSS/MOVA.

Copied from ``mova/diffusion/schedulers/flow_match.py`` and ``flow_match_pair.py`` (Apache-2.0).
Only the framework registration (``mova.registry``) is removed; the numerics — including the
``train_timesteps`` / ``train_sigmas`` nearest-neighbour lookup inside ``timestep_to_sigma`` — are
kept verbatim, because the two towers are advanced independently and any change here moves the
sampling trajectory.

``FlowMatchScheduler`` is *not* a diffusers ``SchedulerMixin`` upstream either; the pair variant adds
it so ``SchedulerMixin`` helpers exist, and ``register_to_config`` is what fills the config.
"""

import math
from typing import Optional

import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin

__all__ = ["FlowMatchScheduler", "FlowMatchPairScheduler"]


class FlowMatchScheduler:
  """Flow-matching sigma schedule with the DiffSynth-style shift.

  :param num_inference_steps: Default inference step count.
  :param num_train_timesteps: Grid size used to express timesteps in ``[0, 1000]``.
  :param shift: Sigma shift applied after the linear ramp.
  :param sigma_max: Largest sigma.
  :param sigma_min: Smallest sigma.
  :param inverse_timesteps: Whether to flip the sigma order.
  :param extra_one_step: Whether the ramp holds ``num_inference_steps + 1`` points and drops the last.
  :param reverse_sigmas: Whether to complement the sigmas.
  :param exponential_shift: Whether to use the exponential (SD3-style) shift.
  :param exponential_shift_mu: Mu of the exponential shift.
  :param shift_terminal: Optional terminal sigma the schedule is rescaled to.
  """

  def __init__(
    self,
    num_inference_steps: int = 100,
    num_train_timesteps: int = 1000,
    shift: float = 3.0,
    sigma_max: float = 1.0,
    sigma_min: float = 0.003 / 1.002,
    inverse_timesteps: bool = False,
    extra_one_step: bool = False,
    reverse_sigmas: bool = False,
    exponential_shift: bool = False,
    exponential_shift_mu: Optional[float] = None,
    shift_terminal: Optional[float] = None,
  ):
    self.num_train_timesteps = num_train_timesteps
    self.shift = shift
    self.sigma_max = sigma_max
    self.sigma_min = sigma_min
    self.inverse_timesteps = inverse_timesteps
    self.extra_one_step = extra_one_step
    self.reverse_sigmas = reverse_sigmas
    self.exponential_shift = exponential_shift
    self.exponential_shift_mu = exponential_shift_mu
    self.shift_terminal = shift_terminal
    self.train_timesteps = None
    self.train_sigmas = None
    self.set_timesteps(num_train_timesteps)
    self.set_timesteps(num_inference_steps)

  def set_timesteps(self,
                    num_inference_steps: int = 100,
                    denoising_strength: float = 1.0,
                    training: bool = False,
                    shift: Optional[float] = None,
                    dynamic_shift_len: Optional[int] = None,
                    device: Optional[torch.device] = None):
    """Rebuild ``sigmas`` / ``timesteps``.

    :param num_inference_steps: Step count of the new schedule.
    :param denoising_strength: Fraction of the sigma range to use.
    :param training: Whether to also build the training weights.
    :param shift: Overrides the configured shift.
    :param dynamic_shift_len: Sequence length for the dynamic shift.
    :param device: Kept for signature compatibility; the tables stay on CPU.
    """

    del device
    if shift is not None:
      self.shift = shift
    sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
    if self.extra_one_step:
      self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
    else:
      self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
    if self.inverse_timesteps:
      self.sigmas = torch.flip(self.sigmas, dims=[0])
    if self.exponential_shift:
      mu = self.calculate_shift(
        dynamic_shift_len) if dynamic_shift_len is not None else self.exponential_shift_mu
      self.sigmas = math.exp(mu) / (math.exp(mu) + (1 / self.sigmas - 1))
    else:
      self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
    if self.shift_terminal is not None:
      one_minus_z = 1 - self.sigmas
      scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
      self.sigmas = 1 - (one_minus_z / scale_factor)
    if self.reverse_sigmas:
      self.sigmas = 1 - self.sigmas
    self.timesteps = self.sigmas * self.num_train_timesteps
    if self.train_timesteps is None:
      self.train_timesteps = self.timesteps
      self.train_sigmas = self.sigmas
    if training:
      x = self.timesteps
      y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
      y_shifted = y - y.min()
      self.linear_timesteps_weights = y_shifted * (num_inference_steps / y_shifted.sum())
      self.training = True
    else:
      self.training = False

  def step(self,
           model_output: torch.Tensor,
           timestep: torch.Tensor,
           sample: torch.Tensor,
           to_final: bool = False,
           **kwargs):
    """Advance one step along the configured schedule.

    :param model_output: Predicted velocity.
    :param timestep: Current timestep.
    :param sample: Current sample.
    :param to_final: Whether this is the last step.
    :returns: The updated sample.
    """

    del kwargs
    if isinstance(timestep, torch.Tensor):
      timestep = timestep.cpu()
    timestep_id = torch.argmin((self.timesteps - timestep).abs())
    sigma = self.sigmas[timestep_id]
    if to_final or timestep_id + 1 >= len(self.timesteps):
      sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
    else:
      sigma_ = self.sigmas[timestep_id + 1]
    return sample + model_output * (sigma_ - sigma)

  def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor,
                timestep: torch.Tensor) -> torch.Tensor:
    """Add noise at ``timestep`` (training helper).

    :param original_samples: Clean samples.
    :param noise: Noise to add.
    :param timestep: Timestep of the mix.
    :returns: The noisy samples.
    """

    if isinstance(timestep, torch.Tensor):
      timestep = timestep.cpu()
    timestep_id = torch.argmin((self.timesteps - timestep).abs())
    sigma = self.sigmas[timestep_id]
    return (1 - sigma) * original_samples + sigma * noise

  def calculate_shift(self,
                      image_seq_len: int,
                      base_seq_len: int = 256,
                      max_seq_len: int = 8192,
                      base_shift: float = 0.5,
                      max_shift: float = 0.9) -> float:
    """Return the exponential-shift mu for a sequence length.

    :param image_seq_len: Token count.
    :param base_seq_len: Sequence length the base shift belongs to.
    :param max_seq_len: Sequence length the max shift belongs to.
    :param base_shift: Shift at ``base_seq_len``.
    :param max_shift: Shift at ``max_seq_len``.
    :returns: The interpolated mu.
    """

    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


class FlowMatchPairScheduler(FlowMatchScheduler, SchedulerMixin, ConfigMixin):
  """Flow-matching schedule exposed as ``[steps, 2]`` pairs, one column per tower.

  :param num_inference_steps: Default inference step count.
  :param num_train_timesteps: Grid size used to express timesteps in ``[0, 1000]``.
  :param shift: Sigma shift applied after the linear ramp.
  :param sigma_max: Largest sigma.
  :param sigma_min: Smallest sigma.
  :param inverse_timesteps: Whether to flip the sigma order.
  :param extra_one_step: Whether the ramp holds ``num_inference_steps + 1`` points and drops the last.
  :param reverse_sigmas: Whether to complement the sigmas.
  :param exponential_shift: Whether to use the exponential (SD3-style) shift.
  :param exponential_shift_mu: Mu of the exponential shift.
  :param shift_terminal: Optional terminal sigma the schedule is rescaled to.
  """

  @register_to_config
  def __init__(
    self,
    num_inference_steps: int = 100,
    num_train_timesteps: int = 1000,
    shift: float = 3.0,
    sigma_max: float = 1.0,
    sigma_min: float = 0.003 / 1.002,
    inverse_timesteps: bool = False,
    extra_one_step: bool = False,
    reverse_sigmas: bool = False,
    exponential_shift: bool = False,
    exponential_shift_mu: Optional[float] = None,
    shift_terminal: Optional[float] = None,
  ):
    self._pair_postprocess_fn = None
    self._pair_postprocess_requires_source = False
    self.pair_timesteps: Optional[torch.Tensor] = None
    self.pair_sigmas: Optional[torch.Tensor] = None
    self.timesteps: Optional[torch.Tensor] = None
    self.sigmas: Optional[torch.Tensor] = None
    super().__init__(
      num_inference_steps=num_inference_steps,
      num_train_timesteps=num_train_timesteps,
      shift=shift,
      sigma_max=sigma_max,
      sigma_min=sigma_min,
      inverse_timesteps=inverse_timesteps,
      extra_one_step=extra_one_step,
      reverse_sigmas=reverse_sigmas,
      exponential_shift=exponential_shift,
      exponential_shift_mu=exponential_shift_mu,
      shift_terminal=shift_terminal,
    )

  def set_pair_postprocess(self, fn):
    """Install a post-processing hook over the pair table.

    :param fn: Callable taking ``pairs`` (and ``source`` when it declares ``_requires_source``).
    """

    if fn is not None and not callable(fn):
      raise TypeError("pair_postprocess must be a callable object or None")
    self._pair_postprocess_fn = fn
    self._pair_postprocess_requires_source = False if fn is None else bool(
      getattr(fn, "_requires_source", False))
    if self.timesteps is None or self.sigmas is None:
      raise RuntimeError("Scheduler not initialized, please call set_timesteps() first")
    self._refresh_pair_cache()

  def set_pair_postprocess_by_name(self, name: Optional[str], **kwargs):
    """Install a named pair post-processing hook.

    Supports ``None`` / ``"none"`` / ``"off"`` / ``"false"`` / ``"no"`` and ``"dual_sigma_shift"``.

    :param name: Hook name.
    :param kwargs: ``visual_shift`` / ``audio_shift`` / ``*_denoising_strength`` / ``*_exponential_shift_mu``.
    """

    if name is None or str(name).lower() in ("none", "off", "false", "no"):
      self.set_pair_postprocess(None)
      return
    if name != "dual_sigma_shift":
      raise ValueError(f"Unsupported pair postprocessing name: {name}")

    visual_shift = float(kwargs.get("visual_shift", self.shift))
    audio_shift = float(kwargs.get("audio_shift", self.shift))
    visual_denoising_strength = float(kwargs.get("visual_denoising_strength", 1.0))
    audio_denoising_strength = float(kwargs.get("audio_denoising_strength", 1.0))
    visual_mu = kwargs.get("visual_exponential_shift_mu", self.exponential_shift_mu)
    audio_mu = kwargs.get("audio_exponential_shift_mu", self.exponential_shift_mu)

    def _dual_sigma_shift(pairs: torch.Tensor, *, source: str):
      if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must be a torch.Tensor with shape [N, 2]")
      if source not in ("timesteps", "sigmas"):
        raise ValueError("source only supports 'timesteps' or 'sigmas'")

      num_steps = pairs.shape[0]
      device = pairs.device
      dtype = pairs.dtype

      def _build_column(shift_value: float, denoising_strength: float, mu_override):
        if shift_value <= 0 or denoising_strength <= 0:
          raise ValueError("shift and denoising_strength must be positive")
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
          base = torch.linspace(sigma_start,
                                self.sigma_min,
                                num_steps + 1,
                                device=device,
                                dtype=dtype)[:-1]
        else:
          base = torch.linspace(sigma_start, self.sigma_min, num_steps, device=device, dtype=dtype)
        if self.inverse_timesteps:
          base = torch.flip(base, dims=[0])
        if self.exponential_shift:
          if mu_override is None:
            raise RuntimeError(
              "exponential_shift is enabled but exponential_shift_mu is not provided")
          exp_mu = math.exp(float(mu_override))
          base = exp_mu / (exp_mu + (1 / base - 1))
        else:
          base = shift_value * base / (1 + (shift_value - 1) * base)
        if self.shift_terminal is not None:
          one_minus_z = 1 - base
          scale_factor = one_minus_z[-1] / (1 - self.shift_terminal)
          base = 1 - (one_minus_z / scale_factor)
        if self.reverse_sigmas:
          base = 1 - base
        return base * self.num_train_timesteps if source == "timesteps" else base

      visual_column = _build_column(visual_shift, visual_denoising_strength, visual_mu)
      audio_column = _build_column(audio_shift, audio_denoising_strength, audio_mu)
      return torch.stack([visual_column, audio_column], dim=1)

    _dual_sigma_shift._requires_source = True
    self.set_pair_postprocess(_dual_sigma_shift)

  def get_pairs(self, source: str = "timesteps") -> torch.Tensor:
    """Return the ``[steps, 2]`` pair table.

    :param source: ``"timesteps"`` or ``"sigmas"``.
    :returns: The pair table.
    """

    if source == "timesteps":
      pairs = self.pair_timesteps
    elif source == "sigmas":
      pairs = self.pair_sigmas
    else:
      raise ValueError("source only supports 'timesteps' or 'sigmas'")
    if pairs is None:
      raise RuntimeError("Scheduler not initialized, please call set_timesteps() first")
    return pairs

  @property
  def visual_timesteps(self) -> torch.Tensor:
    """First column of the pair table."""

    if self.pair_timesteps is None:
      raise RuntimeError("Scheduler not initialized, please call set_timesteps() first")
    return self.pair_timesteps[:, 0]

  @property
  def audio_timesteps(self) -> torch.Tensor:
    """Second column of the pair table."""

    if self.pair_timesteps is None:
      raise RuntimeError("Scheduler not initialized, please call set_timesteps() first")
    return self.pair_timesteps[:, 1]

  def set_timesteps(self, *args, **kwargs):
    """Rebuild the schedule and the cached pair table."""

    super().set_timesteps(*args, **kwargs)
    self._refresh_pair_cache()

  def timestep_to_sigma(self, timestep) -> torch.Tensor:
    """Map a timestep to its sigma with a nearest-neighbour lookup on the training grid.

    :param timestep: Scalar timestep.
    :returns: The matching sigma.
    """

    t_cpu = torch.tensor(float(timestep))
    idx = torch.argmin((self.train_timesteps - t_cpu).abs())
    return self.train_sigmas[idx]

  def step_from_to(self, model_output: torch.Tensor, timestep_from: torch.Tensor,
                   timestep_to: Optional[torch.Tensor], sample: torch.Tensor) -> torch.Tensor:
    """Advance one step with an explicit ``(from, to)`` timestep pair.

    :param model_output: Predicted velocity.
    :param timestep_from: Current timestep.
    :param timestep_to: Next timestep, or None on the last step.
    :param sample: Current sample.
    :returns: ``sample + model_output * (sigma(to) - sigma(from))``.
    """

    sigma_from = self.timestep_to_sigma(timestep_from)
    if timestep_to is None:
      sigma_to = torch.tensor(
        1.0 if (self.inverse_timesteps or self.reverse_sigmas) else 0.0,
        device=sigma_from.device,
        dtype=sigma_from.dtype,
      )
    else:
      sigma_to = self.timestep_to_sigma(timestep_to)
    return sample + model_output * (sigma_to - sigma_from)

  def _make_pairs_from_vector(self, vec: torch.Tensor) -> torch.Tensor:
    """Duplicate a schedule vector into a two-column table."""

    if vec.numel() == 0:
      raise ValueError("input vector length must be greater than 0")
    return torch.stack([vec, vec], dim=1)

  def _refresh_pair_cache(self) -> None:
    """Rebuild ``pair_timesteps`` / ``pair_sigmas`` from the current schedule."""

    if self.timesteps is None or self.sigmas is None:
      raise RuntimeError("Scheduler not initialized, please call set_timesteps() first")

    def _apply_postprocess(pairs: torch.Tensor, source: str) -> torch.Tensor:
      if self._pair_postprocess_fn is None:
        return pairs
      if self._pair_postprocess_requires_source:
        modified = self._pair_postprocess_fn(pairs, source=source)
      else:
        modified = self._pair_postprocess_fn(pairs)
      if not isinstance(modified, torch.Tensor) or modified.shape != pairs.shape:
        raise ValueError("pair_postprocess must return a tensor of the same shape")
      return modified

    self.pair_timesteps = _apply_postprocess(self._make_pairs_from_vector(self.timesteps),
                                             "timesteps")
    self.pair_sigmas = _apply_postprocess(self._make_pairs_from_vector(self.sigmas), "sigmas")
