"""Flow-matching UniPC scheduler used by HiDream-O1-Image.

Ported line by line from the HiDream-O1-Image release sampler via vllm-omni
(`vllm_omni/diffusion/models/schedulers/scheduling_flow_unipc_multistep.py`, Apache-2.0), which is
itself adapted from diffusers' `UniPCMultistepScheduler` v0.31.0. The numbers must stay identical to
the reference: every coefficient (`h_phi_*`, `B_h`, `rhos_p`, `rhos_c`) feeds the 50-step denoising
loop, and a wrong coefficient still renders a plausible image, so it cannot be caught by comparing
our own runs against each other.

Differences from the reference are limited to dropped vllm plumbing: no `BaseScheduler` base class,
no `solver_p` sub-solver, no `deprecate()` calls, and `safe_linalg_solve` inlined as
`torch.linalg.solve` (the numpy fallback only covers a ROCm CPU-LAPACK gap).
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin, SchedulerOutput


class HiDreamO1ImageScheduler(SchedulerMixin, ConfigMixin):
  """UniPC (predictor/corrector) sampler for flow-matching models.

  :param num_train_timesteps: Number of diffusion steps used in training.
  :param solver_order: UniPC order; `2` is recommended for guided sampling.
  :param prediction_type: Only `"flow_prediction"` is supported.
  :param shift: Schedule shift parameter (HiDream-O1-Image uses `3.0`).
  :param use_dynamic_shifting: Whether the shift depends on the resolution.
  :param thresholding: Whether to apply dynamic thresholding on `x0`.
  :param dynamic_thresholding_ratio: Quantile used by dynamic thresholding.
  :param sample_max_value: Clip value used by dynamic thresholding.
  :param predict_x0: Whether to update the predicted `x0` instead of the velocity.
  :param solver_type: `bh1` or `bh2`.
  :param lower_order_final: Whether to lower the order on the last steps.
  :param disable_corrector: Step indices that skip the corrector.
  :param timestep_spacing: Timestep spacing of the noise schedule.
  :param steps_offset: Offset added to the timesteps.
  :param final_sigmas_type: `"zero"` or `"sigma_min"`.
  """

  _compatibles = [e.name for e in KarrasDiffusionSchedulers]
  order = 1

  @register_to_config
  def __init__(
    self,
    num_train_timesteps: int = 1000,
    solver_order: int = 2,
    prediction_type: str = "flow_prediction",
    shift: float = 1.0,
    use_dynamic_shifting: bool = False,
    thresholding: bool = False,
    dynamic_thresholding_ratio: float = 0.995,
    sample_max_value: float = 1.0,
    predict_x0: bool = True,
    solver_type: str = "bh2",
    lower_order_final: bool = True,
    disable_corrector: Tuple[int, ...] = (),
    timestep_spacing: str = "linspace",
    steps_offset: int = 0,
    final_sigmas_type: Optional[str] = "zero",
    **kwargs,
  ):
    if solver_type not in ("bh1", "bh2"):
      if solver_type in ("midpoint", "heun", "logrho"):
        self.register_to_config(solver_type="bh2")
      else:
        raise NotImplementedError(f"{solver_type} is not implemented for {self.__class__}")

    self.predict_x0 = predict_x0
    self.num_inference_steps: Optional[int] = None

    alphas = np.linspace(1, 1 / num_train_timesteps, num_train_timesteps)[::-1].copy()
    sigmas = 1.0 - alphas
    sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32)

    if not use_dynamic_shifting:
      sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

    self.sigmas = sigmas
    self.timesteps = sigmas * num_train_timesteps
    self.num_train_timesteps = num_train_timesteps

    self.model_outputs: List[Optional[torch.Tensor]] = [None] * solver_order
    self.timestep_list: List[Optional[torch.Tensor]] = [None] * solver_order
    self.lower_order_nums = 0
    self.disable_corrector = list(disable_corrector)
    self.last_sample: Optional[torch.Tensor] = None
    self._step_index: Optional[int] = None
    self._begin_index: Optional[int] = None
    self.this_order: int = 1

    self.sigmas = self.sigmas.to("cpu")
    self.sigma_min = self.sigmas[-1].item()
    self.sigma_max = self.sigmas[0].item()

  @property
  def step_index(self) -> Optional[int]:
    """Index counter of the current timestep, advanced after every `step()`."""
    return self._step_index

  @property
  def begin_index(self) -> Optional[int]:
    """Index of the first timestep, set by `set_begin_index()`."""
    return self._begin_index

  def set_shift(self, shift: float) -> None:
    """Set the schedule shift parameter.

    :param shift: New shift value.
    """
    self.config.shift = shift

  def set_begin_index(self, begin_index: int = 0) -> None:
    """Set the begin index of the sampler.

    :param begin_index: Index the sampler starts from.
    """
    self._begin_index = begin_index

  def set_timesteps(
    self,
    num_inference_steps: int = None,
    device: Optional[Union[str, torch.device]] = None,
    sigmas: Optional[List[float]] = None,
    mu: Optional[float] = None,
    shift: Optional[float] = None,
  ) -> None:
    """Build the discrete timesteps used by the denoising loop.

    :param num_inference_steps: Total number of timesteps.
    :param device: Device the timesteps are placed on.
    :param sigmas: Optional custom sigma schedule.
    :param mu: Dynamic-shifting parameter; required when `use_dynamic_shifting` is on.
    :param shift: Overrides the configured shift.
    """
    if self.config.use_dynamic_shifting and mu is None:
      raise ValueError("Must pass a value for `mu` when `use_dynamic_shifting` is True")

    if sigmas is None:
      sigmas = np.linspace(self.sigma_max, self.sigma_min, num_inference_steps + 1).copy()[:-1]

    if self.config.use_dynamic_shifting:
      sigmas = self.time_shift(mu, 1.0, sigmas)
    else:
      if shift is None:
        shift = self.config.shift
      sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

    if self.config.final_sigmas_type == "sigma_min":
      sigma_last = self.sigma_min
    elif self.config.final_sigmas_type == "zero":
      sigma_last = 0
    else:
      raise ValueError(
        f"`final_sigmas_type` must be 'zero' or 'sigma_min', got {self.config.final_sigmas_type}")

    timesteps = sigmas * self.config.num_train_timesteps
    sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

    self.sigmas = torch.from_numpy(sigmas)
    self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)

    self.num_inference_steps = len(timesteps)
    self.clear_history()
    self._step_index = None
    self._begin_index = None
    self.sigmas = self.sigmas.to("cpu")

  def clear_history(self) -> None:
    """Drop the multistep history; call before starting another run."""
    self.model_outputs = [None] * self.config.solver_order
    self.timestep_list = [None] * self.config.solver_order
    self.lower_order_nums = 0
    self.last_sample = None

  def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
    """Apply dynamic thresholding to the predicted `x0`.

    :param sample: Predicted `x0`.
    :returns: Thresholded sample.
    """
    dtype = sample.dtype
    batch_size, channels, *remaining_dims = sample.shape

    if dtype not in (torch.float32, torch.float64):
      sample = sample.float()

    sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))
    abs_sample = sample.abs()

    s = torch.quantile(abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
    s = torch.clamp(s, min=1, max=self.config.sample_max_value)
    s = s.unsqueeze(1)
    sample = torch.clamp(sample, -s, s) / s

    sample = sample.reshape(batch_size, channels, *remaining_dims)
    return sample.to(dtype)

  def _sigma_to_t(self, sigma: torch.Tensor) -> torch.Tensor:
    """Convert sigma to a timestep.

    :param sigma: Sigma value.
    :returns: The equivalent timestep.
    """
    return sigma * self.config.num_train_timesteps

  def _sigma_to_alpha_sigma_t(self, sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert sigma to the flow-matching `(alpha_t, sigma_t)` pair.

    :param sigma: Sigma value.
    :returns: `(1 - sigma, sigma)`.
    """
    return 1 - sigma, sigma

  def time_shift(self, mu: float, sigma: float, t: np.ndarray) -> np.ndarray:
    """Apply the dynamic-shift transform to a sigma schedule.

    :param mu: Shift strength.
    :param sigma: Shift exponent.
    :param t: Sigma schedule.
    :returns: The shifted schedule.
    """
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)

  def convert_model_output(
    self,
    model_output: torch.Tensor,
    *args,
    sample: Optional[torch.Tensor] = None,
    **kwargs,
  ) -> torch.Tensor:
    """Convert the raw model output to the quantity UniPC integrates.

    :param model_output: Raw velocity predicted by the model.
    :param sample: Current sample.
    :returns: Predicted `x0` when `predict_x0`, otherwise the velocity.
    """
    if sample is None:
      if len(args) > 1:
        sample = args[1]
      else:
        raise ValueError("missing `sample` as a required keyword argument")

    sigma = self.sigmas[self.step_index].to(sample.device)
    alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)

    if self.predict_x0:
      if self.config.prediction_type != "flow_prediction":
        raise ValueError(f"prediction_type given as {self.config.prediction_type} must be "
                         "`flow_prediction` for HiDreamO1ImageScheduler.")
      sigma_t = sigma.to(sample.device)
      x0_pred = sample - sigma_t * model_output
      if self.config.thresholding:
        x0_pred = self._threshold_sample(x0_pred)
      return x0_pred

    if self.config.prediction_type != "flow_prediction":
      raise ValueError(f"prediction_type given as {self.config.prediction_type} must be "
                       "`flow_prediction` for HiDreamO1ImageScheduler.")
    sigma_t = sigma.to(sample.device)
    epsilon = sample - (1 - sigma_t) * model_output
    if self.config.thresholding:
      sigma_t = sigma.to(sample.device)
      x0_pred = sample - sigma_t * model_output
      x0_pred = self._threshold_sample(x0_pred)
      epsilon = model_output + x0_pred
    return epsilon

  def multistep_uni_p_bh_update(
    self,
    model_output: torch.Tensor,
    *args,
    sample: Optional[torch.Tensor] = None,
    order: Optional[int] = None,
    **kwargs,
  ) -> torch.Tensor:
    """One predictor step of UniP (B(h) variant).

    :param model_output: Raw model output at the current step.
    :param sample: Current sample.
    :param order: UniP order used at this step.
    :returns: The sample at the previous timestep.
    """
    if sample is None:
      if len(args) > 1:
        sample = args[1]
      else:
        raise ValueError("missing `sample` as a required keyword argument")
    if order is None:
      if len(args) > 2:
        order = args[2]
      else:
        raise ValueError("missing `order` as a required keyword argument")

    model_output_list = self.model_outputs
    m0 = model_output_list[-1]
    x = sample
    device = sample.device

    sigma_t, sigma_s0 = (
      self.sigmas[self.step_index + 1].to(device),
      self.sigmas[self.step_index].to(device),
    )
    alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
    alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
    h = lambda_t - lambda_s0

    rks = []
    d1s = []
    for i in range(1, order):
      si = self.step_index - i
      mi = model_output_list[-(i + 1)]
      alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si].to(device))
      lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
      rk = (lambda_si - lambda_s0) / h
      rks.append(rk)
      assert mi is not None
      d1s.append((mi - m0) / rk)

    rks.append(1.0)
    rks = torch.tensor(rks, device=device)

    r = []
    b = []

    hh = -h if self.predict_x0 else h
    h_phi_1 = torch.expm1(hh)
    h_phi_k = h_phi_1 / hh - 1

    factorial_i = 1

    if self.config.solver_type == "bh1":
      b_h = hh
    elif self.config.solver_type == "bh2":
      b_h = torch.expm1(hh)
    else:
      raise NotImplementedError()

    for i in range(1, order + 1):
      r.append(torch.pow(rks, i - 1))
      b.append(h_phi_k * factorial_i / b_h)
      factorial_i *= i + 1
      h_phi_k = h_phi_k / hh - 1 / factorial_i

    r = torch.stack(r)
    b = torch.tensor(b, device=device)

    if len(d1s) > 0:
      d1s = torch.stack(d1s, dim=1)
      if order == 2:
        rhos_p = torch.tensor([0.5], dtype=x.dtype, device=device)
      else:
        rhos_p = torch.linalg.solve(r[:-1, :-1], b[:-1]).to(device).to(x.dtype)
    else:
      d1s = None

    if self.predict_x0:
      x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
      pred_res = torch.einsum("k,bkc...->bc...", rhos_p, d1s) if d1s is not None else 0
      x_t = x_t_ - alpha_t * b_h * pred_res
    else:
      x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
      pred_res = torch.einsum("k,bkc...->bc...", rhos_p, d1s) if d1s is not None else 0
      x_t = x_t_ - sigma_t * b_h * pred_res

    return x_t.to(x.dtype)

  def multistep_uni_c_bh_update(
    self,
    this_model_output: torch.Tensor,
    *args,
    last_sample: Optional[torch.Tensor] = None,
    this_sample: Optional[torch.Tensor] = None,
    order: Optional[int] = None,
    **kwargs,
  ) -> torch.Tensor:
    """One corrector step of UniC (B(h) variant).

    :param this_model_output: Model output at `x_t`.
    :param last_sample: Sample before the last predictor step.
    :param this_sample: Sample after the last predictor step.
    :param order: UniC order; the effective accuracy is `order + 1`.
    :returns: The corrected sample.
    """
    if last_sample is None:
      if len(args) > 1:
        last_sample = args[1]
      else:
        raise ValueError("missing `last_sample` as a required keyword argument")
    if this_sample is None:
      if len(args) > 2:
        this_sample = args[2]
      else:
        raise ValueError("missing `this_sample` as a required keyword argument")
    if order is None:
      if len(args) > 3:
        order = args[3]
      else:
        raise ValueError("missing `order` as a required keyword argument")

    model_output_list = self.model_outputs
    m0 = model_output_list[-1]
    x = last_sample
    model_t = this_model_output
    device = this_sample.device

    sigma_t, sigma_s0 = (
      self.sigmas[self.step_index].to(device),
      self.sigmas[self.step_index - 1].to(device),
    )
    alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
    alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
    h = lambda_t - lambda_s0

    rks = []
    d1s = []
    for i in range(1, order):
      si = self.step_index - (i + 1)
      mi = model_output_list[-(i + 1)]
      alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si].to(device))
      lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
      rk = (lambda_si - lambda_s0) / h
      rks.append(rk)
      assert mi is not None
      d1s.append((mi - m0) / rk)

    rks.append(1.0)
    rks = torch.tensor(rks, device=device)

    r = []
    b = []

    hh = -h if self.predict_x0 else h
    h_phi_1 = torch.expm1(hh)
    h_phi_k = h_phi_1 / hh - 1

    factorial_i = 1

    if self.config.solver_type == "bh1":
      b_h = hh
    elif self.config.solver_type == "bh2":
      b_h = torch.expm1(hh)
    else:
      raise NotImplementedError()

    for i in range(1, order + 1):
      r.append(torch.pow(rks, i - 1))
      b.append(h_phi_k * factorial_i / b_h)
      factorial_i *= i + 1
      h_phi_k = h_phi_k / hh - 1 / factorial_i

    r = torch.stack(r)
    b = torch.tensor(b, device=device)

    d1s = torch.stack(d1s, dim=1) if len(d1s) > 0 else None

    if order == 1:
      rhos_c = torch.tensor([0.5], dtype=x.dtype, device=device)
    else:
      rhos_c = torch.linalg.solve(r, b).to(device).to(x.dtype)

    if self.predict_x0:
      x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
      corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], d1s) if d1s is not None else 0
      d1_t = model_t - m0
      x_t = x_t_ - alpha_t * b_h * (corr_res + rhos_c[-1] * d1_t)
    else:
      x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
      corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], d1s) if d1s is not None else 0
      d1_t = model_t - m0
      x_t = x_t_ - sigma_t * b_h * (corr_res + rhos_c[-1] * d1_t)

    return x_t.to(x.dtype)

  def index_for_timestep(
    self,
    timestep: torch.Tensor,
    schedule_timesteps: Optional[torch.Tensor] = None,
  ) -> int:
    """Index of `timestep` in the schedule.

    :param timestep: Timestep to look up.
    :param schedule_timesteps: Optional schedule override.
    :returns: The index of the timestep.
    """
    if schedule_timesteps is None:
      schedule_timesteps = self.timesteps
    indices = (schedule_timesteps == timestep).nonzero()
    pos = 1 if len(indices) > 1 else 0
    return int(indices[pos].item())

  def _init_step_index(self, timestep: torch.Tensor) -> None:
    """Initialize the step counter from `timestep`.

    :param timestep: The first timestep of the run.
    """
    if self.begin_index is None:
      if isinstance(timestep, torch.Tensor):
        timestep = timestep.to(self.timesteps.device)
      self._step_index = self.index_for_timestep(timestep)
    else:
      self._step_index = self._begin_index

  def step(
    self,
    model_output: torch.Tensor,
    timestep: Union[int, torch.Tensor],
    sample: torch.Tensor,
    return_dict: bool = True,
    generator: Optional[torch.Generator] = None,
  ) -> Union[SchedulerOutput, Tuple]:
    """Advance the multistep UniPC sampler by one step.

    :param model_output: Raw model output at the current step.
    :param timestep: Current discrete timestep.
    :param sample: Current sample.
    :param return_dict: Whether to return a `SchedulerOutput`.
    :param generator: Unused, accepted for API compatibility.
    :returns: The sample at the previous timestep.
    """
    if self.num_inference_steps is None:
      raise ValueError("Number of inference steps is 'None', you need to run 'set_timesteps' "
                       "after creating the scheduler")

    if self.step_index is None:
      self._init_step_index(timestep)

    use_corrector = (self.step_index > 0 and self.step_index - 1 not in self.disable_corrector
                     and self.last_sample is not None)

    model_output_convert = self.convert_model_output(model_output, sample=sample)

    if use_corrector:
      sample = self.multistep_uni_c_bh_update(
        this_model_output=model_output_convert,
        last_sample=self.last_sample,
        this_sample=sample,
        order=self.this_order,
      )

    for i in range(self.config.solver_order - 1):
      self.model_outputs[i] = self.model_outputs[i + 1]
      self.timestep_list[i] = self.timestep_list[i + 1]

    self.model_outputs[-1] = model_output_convert
    self.timestep_list[-1] = timestep

    if self.config.lower_order_final:
      this_order = min(self.config.solver_order, len(self.timesteps) - self.step_index)
    else:
      this_order = self.config.solver_order

    self.this_order = min(this_order, self.lower_order_nums + 1)
    assert self.this_order > 0

    self.last_sample = sample
    prev_sample = self.multistep_uni_p_bh_update(
      model_output=model_output,
      sample=sample,
      order=self.this_order,
    )

    if self.lower_order_nums < self.config.solver_order:
      self.lower_order_nums += 1

    assert self._step_index is not None
    self._step_index += 1

    if not return_dict:
      return (prev_sample, )
    return SchedulerOutput(prev_sample=prev_sample)

  def scale_model_input(self, sample: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    """Return the sample unchanged (flow matching needs no input scaling).

    :param sample: Current sample.
    :returns: The same sample.
    """
    return sample

  def add_noise(
    self,
    original_samples: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.IntTensor,
  ) -> torch.Tensor:
    """Add noise to the samples at the given timesteps.

    :param original_samples: Clean samples.
    :param noise: Noise to add.
    :param timesteps: Timesteps used for the noise level.
    :returns: The noisy samples.
    """
    sigmas = self.sigmas.to(device=original_samples.device, dtype=original_samples.dtype)
    schedule_timesteps = self.timesteps.to(original_samples.device)
    timesteps = timesteps.to(original_samples.device)

    if self.begin_index is None:
      step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
    elif self.step_index is not None:
      step_indices = [self.step_index] * timesteps.shape[0]
    else:
      step_indices = [self.begin_index] * timesteps.shape[0]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < len(original_samples.shape):
      sigma = sigma.unsqueeze(-1)

    alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
    return alpha_t * original_samples + sigma_t * noise

  def __len__(self) -> int:
    return self.config.num_train_timesteps
