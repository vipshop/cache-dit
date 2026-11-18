# HiCache (arXiv 2508.16984): keep TaylorSeer's finite-difference ladder and
# replace only the extrapolation basis with a damped Hermite one:
#
#     TaylorSeer : F_hat = F_0 + sum_k (1/k!) * d^k F * elapsed^k
#     HiCache    : F_hat = F_0 + sum_k (1/k!) * d^k F * H~_k(elapsed)
#     H~_k(x)    = sigma^k * H_k(sigma * x)      (H = physicist Hermite)
#
# The ladder d^{k+1}F = (d^k F_new - d^k F_prev) / window is IDENTICAL to
# TaylorSeer's, so HiCacheState subclasses TaylorSeerState and overrides only
# `_basis_term`. Both `derivative()` and the accumulation loop stay single-copy
# in `taylorseer.py`; a missing override would silently degrade to TaylorSeer,
# which is why the tests assert the concrete state type and a non-equal output.
#
# Physicist-Hermite recurrence H_{k+1}(y) = 2y H_k(y) - 2k H_{k-1}(y) with
# H_0 = 1, H_1 = 2y pulls the sigma factors inside:
#
#     H~_0 = 1,  H~_1 = 2 * sigma^2 * x
#     H~_{k+1}(x) = 2 * sigma^2 * x * H~_k(x) - 2k * sigma^2 * H~_{k-1}(x)
#
# so every higher order is damped by sigma^2 (its first-order gain is 2*sigma^2,
# which is what a sigma sweep actually moves). sigma = 1 does NOT recover
# TaylorSeer: the basis is the plain physicist Hermite polynomial, whose
# ablation in the paper is *worse* than Taylor, so sigma = 1 is kept only as a
# control point. sigma = 1/sqrt(2) makes the order-1 term match Taylor's.
#
# NOT replicated from the paper: the fixed refresh interval N_interval and the
# per-layer feature caching. Scheduling stays with DBCache's adaptive
# `residual_diff_threshold` and its block granularity, therefore `elapsed` (and
# the spacing `window` of the ladder) varies per hit and `1/k! * d^k F * H~_k`
# is a basis substitution, not a Hermite interpolation of uniformly spaced
# samples. The paper's sigma (calibrated at a fixed h ~ 7/9 on FLUX) is coupled
# to that spacing and must be re-swept per model and cache setting.
from .taylorseer import TaylorSeerCalibrator, TaylorSeerState

from ....logger import init_logger

logger = init_logger(__name__)

# The paper caps the expansion order at N_order <= 4.
MAX_HICACHE_ORDER = 4


def validate_hicache_params(sigma: float, order: int):
  """Validate the HiCache hyper-parameters, raising `ValueError` on the first offender.

  Called both by the config dataclass (so YAML/CLI mistakes fail at parse time) and by the
  calibrator itself (so a hand-built instance, or one produced by `CalibratorConfig.empty()`,
  cannot slip through).

  :param sigma: Damping factor of the Hermite basis, in (0, 1]. The paper uses 0.4-0.5.
  :param order: Highest finite-difference order kept, in [0, `MAX_HICACHE_ORDER`].
  """

  if not 0.0 < sigma <= 1.0:
    raise ValueError(f"sigma must be in (0, 1], got {sigma}.")
  if not 0 <= order <= MAX_HICACHE_ORDER:
    raise ValueError(f"order must be in [0, {MAX_HICACHE_ORDER}], got {order}.")


class HiCacheState(TaylorSeerState):
  """Per-stream HiCache state: the TaylorSeer ladder with a damped-Hermite basis.

  Shares `derivative()`, the step counters and the shape-change reset of `TaylorSeerState`; only
  the extrapolation weight (`_basis_term`) differs.
  """

  def __init__(
    self,
    n_derivatives: int = 1,
    max_warmup_steps: int = 1,
    skip_interval_steps: int = 1,
    sigma: float = 0.5,
  ):
    """Initialize the ladder and the Hermite damping factor.

    :param n_derivatives: Number of Hermite orders kept (the paper's N_order).
    :param max_warmup_steps: Number of initial full-compute warmup steps.
    :param skip_interval_steps: Interval between later full-compute refresh steps.
    :param sigma: Damping factor of the Hermite basis, in (0, 1].
    """

    validate_hicache_params(sigma, n_derivatives)
    super().__init__(
      n_derivatives=n_derivatives,
      max_warmup_steps=max_warmup_steps,
      skip_interval_steps=skip_interval_steps,
    )
    self.sigma = sigma

  def hermite_basis(self, i: int, elapsed: float) -> float:
    """Evaluate `H~_i(elapsed) = sigma^i * H_i(sigma * elapsed)`.

    :param i: Basis order, starting at 0 for the constant term.
    :param elapsed: Steps between the current step and the last full-compute anchor.
    :returns: The damped-Hermite weight of the i-th finite difference.
    """

    if i <= 0:
      return 1.0
    sigma_sq = self.sigma * self.sigma
    prev, cur = 1.0, 2.0 * sigma_sq * elapsed
    for k in range(1, i):
      prev, cur = cur, 2.0 * sigma_sq * elapsed * cur - 2.0 * k * sigma_sq * prev
    return cur

  def _basis_term(self, i: int, elapsed: int) -> float:
    """Return the HiCache weight of the i-th finite difference.

    :param i: Basis order, starting at 0 for the constant term.
    :param elapsed: Steps between the current step and the last full-compute anchor.
    :returns: The damped-Hermite weight `H~_i(elapsed)`.
    """

    return self.hermite_basis(i, elapsed)


class HiCacheCalibrator(TaylorSeerCalibrator):
  """Calibrator that forecasts tensors with the HiCache damped-Hermite series."""

  def __init__(
    self,
    n_derivatives: int = 1,
    max_warmup_steps: int = 1,
    skip_interval_steps: int = 1,
    sigma: float = 0.5,
    **kwargs,
  ):
    """Create a calibrator whose states are keyed by logical tensor names.

    :param n_derivatives: Number of Hermite orders kept per stream (the paper's N_order).
    :param max_warmup_steps: Number of initial full-compute warmup steps.
    :param skip_interval_steps: Interval between later full-compute refresh steps.
    :param sigma: Damping factor of the Hermite basis, in (0, 1].
    :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
    """

    # Validate before the base class builds any state: the per-stream HiCacheState is created
    # lazily on the first full-compute step, so an assert there would only fire mid-run.
    validate_hicache_params(sigma, n_derivatives)
    super().__init__(
      n_derivatives=n_derivatives,
      max_warmup_steps=max_warmup_steps,
      skip_interval_steps=skip_interval_steps,
      **kwargs,
    )
    self.sigma = sigma

  def maybe_init_state(
    self,
    name: str = "default",
  ):
    """Lazily create one HiCache state for a named tensor stream.

    :param name: Logical tensor-stream name.
    """

    if name not in self.states:
      self.states[name] = HiCacheState(
        n_derivatives=self.n_derivatives,
        max_warmup_steps=self.max_warmup_steps,
        skip_interval_steps=self.skip_interval_steps,
        sigma=self.sigma,
      )

  def __repr__(self):
    return f"HiCacheCalibrator_O({self.n_derivatives}, sigma={self.sigma})"
