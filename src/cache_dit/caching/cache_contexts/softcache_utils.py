"""SoftCache: the official SeaCache refresh rule plus headroom-adaptive sampler jumps, for DBCache.

Every step SoftCache measures how much the SEA-filtered, modulated input of the first block changed
since the previous step and accumulates it. Once the sum reaches `tau` the step is computed in full
(SeaCache's own rule); otherwise the cache is reused and, if the jump gate allows, the current sampler
step is enlarged by a factor that grows with the unspent budget (headroom = 1 - accumulated / tau).
The remaining schedule is re-spaced in place on the live `FlowMatchEulerDiscreteScheduler`, at fixed
length: the nodes the jump removes become trailing `sigma = 0` entries, whose steps have `dt = 0` and
are skipped without any compute.
"""

from typing import List, Optional, Tuple

import torch


def regrid_tail(sigmas: List[float], i: int, target: float) -> List[float]:
  """After jumping from node i to sigma=`target`, re-space the rest of the schedule linearly
  from target to 0 with only as many nodes as the old schedule had below target. The nodes
  this removes are the compute a jump saves.

  Verbatim port of the reference implementation (`softcache.method.regrid_tail`).
  """
  n_tail = len(sigmas) - (i + 1)
  n_below = sum(1 for s in sigmas if 0.0 < s < target) + 1  # +1 for the terminal 0
  n_new = min(n_tail - 1, n_below)
  if n_new <= 1:  # no node left below target: still take the final step target -> 0
    return sigmas[:i + 1] + ([target, 0.0] if target > 0 else [0.0])
  tail = [target * (1.0 - k / (n_new - 1)) for k in range(n_new)]
  tail[0], tail[-1] = target, 0.0
  return sigmas[:i + 1] + tail


def softcache_rel_l1(current: torch.Tensor, previous: torch.Tensor, eps: float = 1e-16) -> float:
  """Relative L1 change of the probe, exactly as the official SeaCache `rel_l1`."""

  return float(((current - previous).abs().mean() / (previous.abs().mean() + eps)).detach().cpu())


def softcache_decide(
  index: int,
  sigma: float,
  raw: float,
  accumulated: float,
  *,
  tau: float,
  jf_max: float,
  sigma_hi: float,
  raw_rel_l1_max: float,
) -> Tuple[str, Optional[float]]:
  """Return `("fresh", None)`, `("cache", None)` or `("jump", jf)` for one sampler step.

  Verbatim port of the reference rule (`softcache.method.decide`): refresh at the first step and
  whenever the accumulated SeaCache change since the last refresh reaches `tau` (the official
  SeaCache rule). Otherwise reuse the cache and, if the gate allows, jump with a stride that grows
  with the unspent budget: `jf = 1 + (jf_max - 1) * (1 - accumulated / tau)`.

  :param index: Index of the sampler step.
  :param sigma: Its flow-matching sigma.
  :param raw: Probe change since the previous step (`softcache_rel_l1`).
  :param accumulated: Sum of `raw` since the last refresh, this step included.
  :param tau: Staleness budget; values above 1 are valid.
  :param jf_max: Cap on the jump factor.
  :param sigma_hi: No jump in the volatile early sigmas above this value.
  :param raw_rel_l1_max: No jump when the probe changed by more than this since the last step.
  """

  if index == 0 or accumulated >= tau:
    return "fresh", None
  if sigma <= sigma_hi and raw <= raw_rel_l1_max:
    headroom = max(0.0, 1.0 - accumulated / max(1e-9, tau))
    jf = 1.0 + (jf_max - 1.0) * headroom
    if jf > 1.01:  # a stride under 1.01 skips no scheduled node, so it is not a jump
      return "jump", jf
  return "cache", None


def validate_softcache_scheduler(scheduler) -> None:
  """Reject schedulers whose in-place regrid semantics SoftCache has not verified."""

  from diffusers import FlowMatchEulerDiscreteScheduler

  if not isinstance(scheduler, FlowMatchEulerDiscreteScheduler):
    raise ValueError("SoftCache requires a FlowMatchEulerDiscreteScheduler, got "
                     f"{type(scheduler).__name__}.")
  if getattr(scheduler.config, "stochastic_sampling", False):
    raise ValueError("SoftCache does not support stochastic_sampling.")


def softcache_step_index(scheduler) -> int:
  """The scheduler's index of the step about to be taken (diffusers sets it lazily)."""

  index = scheduler.step_index
  if index is None:
    index = scheduler.begin_index if scheduler.begin_index is not None else 0
  return int(index)


@torch.compiler.disable
def softcache_rewrite_tail(
  scheduler,
  i: int,
  jf: float,
  real: Optional[List[float]] = None,
) -> Tuple[int, List[float]]:
  """Enlarge step i by `jf` and re-space the tail of `scheduler.sigmas`/`timesteps` in place.

  :param scheduler: Live `FlowMatchEulerDiscreteScheduler` of the running pipeline.
  :param i: Index of the step being taken.
  :param jf: Jump factor from `softcache_jump_factor`.
  :param real: The float64 schedule returned by the previous jump of this request, or `None` on
      the first jump. Regridding from it rather than from the float32 `scheduler.sigmas` keeps the
      node counts of successive jumps identical to the reference implementation.
  :returns: The number of scheduled steps the jump removed (now `dt = 0` padding), and the new
      unpadded float64 schedule to pass to the next jump.
  """

  sigmas, timesteps = scheduler.sigmas, scheduler.timesteps
  scale = float(scheduler.config.num_train_timesteps)
  if len(timesteps) != len(sigmas) - 1:
    raise ValueError("SoftCache expects len(timesteps) == len(sigmas) - 1.")
  length = len(sigmas)
  if real is None:
    values = [float(s) for s in sigmas]
    # The schedule ends at its first terminal 0; anything after it is padding.
    real = values[:next(k for k, s in enumerate(values) if s <= 0.0) + 1]
  end = len(real) - 1
  if i + 1 >= end:  # the final step already lands on 0: nothing left to skip
    return 0, real
  expected = sigmas[:end].to(timesteps.device, torch.float32) * scale
  if not torch.allclose(timesteps[:end].float(), expected, rtol=1e-4, atol=1e-3):
    raise ValueError("SoftCache requires scheduler.timesteps == scheduler.sigmas[:-1] * "
                     "num_train_timesteps.")
  sigma = real[i]
  target = max(0.0, sigma + jf * (real[i + 1] - sigma))
  new = regrid_tail(real, i, target)
  removed = len(real) - len(new)
  padded = torch.tensor(new + [0.0] * (length - len(new)), dtype=torch.float64)
  sigmas[i + 1:] = padded[i + 1:].to(sigmas.device, sigmas.dtype)
  timesteps[i + 1:] = (padded[i + 1:-1] * scale).to(timesteps.device, timesteps.dtype)
  return removed, new
