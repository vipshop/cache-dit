"""Numerical parity between the vendored FlowUniPC scheduler and diffusers' UniPC.

The shipped scheduler (`cache_dit.models.pipelines.hidream_o1_image.scheduler`) is a line-by-line
port of the release sampler (via vllm-omni), which is itself diffusers' `UniPCMultistepScheduler`
converted to flow matching. A wrong UniPC coefficient still renders a plausible image, and comparing
our own runs against each other cannot catch it, so this test drives both implementations with the
same schedule and the same fake velocity field and compares every `prev_sample`.

The two schedules differ by one timestep integer: the reference multiplies sigmas by
`num_train_timesteps` and truncates (`to(torch.int64)`), diffusers rounds. That is a property of the
reference, not of the solver, so the test copies our schedule onto the diffusers instance and only
compares the UniPC update itself.

Usage:
  python -m pytest tests/api/test_hidream_o1_image_scheduler.py -q
"""

import pytest
import torch

from cache_dit.models.pipelines.hidream_o1_image import HiDreamO1ImageScheduler

diffusers = pytest.importorskip("diffusers")
UniPCMultistepScheduler = diffusers.UniPCMultistepScheduler


def _pair(num_steps: int = 4, shift: float = 3.0):
  ours = HiDreamO1ImageScheduler(
    num_train_timesteps=1000,
    shift=shift,
    prediction_type="flow_prediction",
    use_dynamic_shifting=False,
  )
  reference = UniPCMultistepScheduler(
    num_train_timesteps=1000,
    use_flow_sigmas=True,
    flow_shift=shift,
    prediction_type="flow_prediction",
    use_dynamic_shifting=False,
    solver_order=2,
    lower_order_final=True,
  )
  ours.set_timesteps(num_steps, device="cpu")
  reference.set_timesteps(num_steps, device="cpu")
  # Align the schedule: the reference truncates, diffusers rounds.
  reference.timesteps = ours.timesteps.clone()
  reference.sigmas = ours.sigmas.clone()
  reference._step_index = None
  reference._begin_index = None
  return ours, reference


def _velocity(step_index: int, sample: torch.Tensor) -> torch.Tensor:
  """A deterministic, step-dependent stand-in for the DiT output."""
  scale = 0.5 + 0.1 * step_index
  return torch.tanh(scale * sample) * 0.75 + 0.05


@pytest.mark.parametrize("num_steps", [2, 3, 4, 6])
def test_matches_diffusers_unipc_step_by_step(num_steps):
  ours, reference = _pair(num_steps)
  generator = torch.Generator().manual_seed(0)
  sample = torch.randn(1, 2, 3, 4, generator=generator, dtype=torch.float64)
  ours_sample = sample.clone()
  reference_sample = sample.clone()

  for step_index, timestep in enumerate(ours.timesteps):
    velocity = _velocity(step_index, ours_sample)
    ours_sample = ours.step(velocity, timestep, ours_sample, return_dict=False)[0]
    reference_sample = reference.step(velocity, timestep, reference_sample, return_dict=False)[0]
    torch.testing.assert_close(
      ours_sample,
      reference_sample,
      atol=1e-12,
      rtol=0,
      msg=f"step {step_index} diverged",
    )


def test_schedule_matches_the_reference_truncation():
  ours = HiDreamO1ImageScheduler(num_train_timesteps=1000,
                                 shift=3.0,
                                 prediction_type="flow_prediction")
  ours.set_timesteps(4, device="cpu")
  # Truncation of `sigma * 1000`, exactly like the release sampler.
  expected = torch.tensor([999, 899, 749, 499])
  assert torch.equal(ours.timesteps, expected), ours.timesteps.tolist()
  assert ours.sigmas.shape == (5, ), "sigmas must carry the final `zero` entry"
  assert float(ours.sigmas[-1]) == 0.0


def test_custom_timesteps_and_sigma_final():
  scheduler = HiDreamO1ImageScheduler(num_train_timesteps=1000,
                                      shift=3.0,
                                      prediction_type="flow_prediction",
                                      final_sigmas_type="zero")
  scheduler.set_timesteps(3, device="cpu")
  assert len(scheduler.timesteps) == 3
  assert scheduler.num_inference_steps == 3
  # A second `set_timesteps` must clear the multistep history of the previous run.
  sample = torch.zeros(1, 2)
  scheduler.step(torch.ones(1, 2), scheduler.timesteps[0], sample, return_dict=False)
  assert scheduler.last_sample is not None
  scheduler.set_timesteps(3, device="cpu")
  assert scheduler.last_sample is None
  assert scheduler.model_outputs == [None, None]


def test_flow_prediction_is_the_only_supported_type():
  with pytest.raises(ValueError, match="flow_prediction"):
    scheduler = HiDreamO1ImageScheduler(num_train_timesteps=1000,
                                        shift=3.0,
                                        prediction_type="epsilon")
    scheduler.set_timesteps(2, device="cpu")
    sample = torch.zeros(1, 2)
    scheduler.step(torch.ones(1, 2), scheduler.timesteps[0], sample, return_dict=False)
