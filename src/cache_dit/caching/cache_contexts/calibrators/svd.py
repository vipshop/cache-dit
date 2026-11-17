# SVD-Cache (arXiv 2601.07396): split one cached block tensor in the token x
# channel plane into a low-rank principal part and a high-frequency residual,
# then forecast the principal part and add the frozen residual back:
#
#     P(Y)   = ((Y V) / sigma) * sigma V^T          (principal part, rank k)
#     R      = Y - P(Y)                             (high-frequency tail)
#     F_hat  = <predictor>(P(Y)) + R
#
# Two predictors are available (``predictor=``):
#
#   "ema"    the paper's per-step smoothing: every full-compute step blends the
#            new principal part in, and every cached step advances the same
#            blend one more tick towards the anchor. Bounded by construction
#            (beta^j -> 0) but BACKWARD-looking: it is a convex blend of two
#            real anchors, so it can only relax reuse, never extrapolate.
#            MEASURED: does not beat plain reuse, and gets worse as beta grows
#            (FLUX.1-dev, 20.62 dB vs 23.11 dB at 1.94x, 15.04 dB vs 15.95 dB
#            at 4.94x). beta = 0 reproduces DBCache's plain reuse exactly.
#
#   "taylor" the paper's Fig. 5(b) combination (SVD + Taylor): a divided-
#            difference Taylor ladder is kept on the LOW-RANK COEFFICIENTS
#            c = Y V (shape [M, k], k ~ 4 on FLUX) and extrapolated forward by
#            the elapsed step count, so the forecast is a genuine extrapolation
#            whose derivative estimate is low-pass filtered by the subspace
#            truncation. One extra [M,k] @ [k,D] GEMM per cached step.
#
# The basis (V, sigma) is fitted ONCE per stream from the first full-compute
# feature seen in the current inference, with the rank chosen by the energy
# criterion sum(sigma^2[:k]) >= tau * sum(sigma^2). There is no per-model
# offline artifact: the subspace is prompt-invariant enough to be estimated at
# run time, which keeps cache-dit's "zero calibration files" property. The basis
# is frozen for the whole inference on purpose: the Taylor ladder lives in that
# basis, so refitting it would silently invalidate every stored coefficient.
#
# Relationship with DBCache: a drop-in forecast layer, not an algorithm change.
# DBCache fuses hidden_states = Bn_residual(t) + Fn_output(t+1); the calibrator
# only replaces the LEFT operand, and Fn_output(t+1) is always recomputed by the
# Fn blocks, so nothing in the decision path (thresholds, counters, skip
# granularity) is touched.
#
# Difference from DMDCalibrator: DMD fits a dynamical model along the TIME axis
# (snapshot matrix [d, n] with n <= 6 steps, eigendecomposition of the
# propagator); SVD-Cache splits ONE step's feature matrix along the token and
# channel axes. Both carry "SVD" in the name and share no numerical machinery.
import math

import torch
from typing import Dict, List, Optional, Tuple

from .base import CalibratorBase

from ....logger import init_logger

logger = init_logger(__name__)

# Singular values below this fraction of the leading one are treated as
# numerically zero, so they never enter the retained rank (a rank-deficient or
# near-singular basis would blow up the 1 / sigma normalisation).
_SIGMA_REL_FLOOR = 1e-8

SVD_PREDICTORS = ("ema", "taylor")


def validate_svd_params(
  tau: float,
  beta: float,
  max_rank: int,
  predictor: str,
  n_derivatives: int,
):
  """Validate the SVD-Cache hyper-parameters, raising `ValueError` on the first offender.

  Called both by the config dataclass (so YAML/CLI mistakes fail at parse time) and by the
  calibrator itself (so a hand-built instance cannot slip through).

  :param tau: Cumulative squared-singular-value energy to retain, in (0, 1].
  :param beta: EMA coefficient of the principal stream, in [0, 1).
  :param max_rank: Hard cap on the retained rank, >= 0.
  :param predictor: One of ``SVD_PREDICTORS``.
  :param n_derivatives: Taylor orders kept, >= 0.
  """

  if not 0.0 < tau <= 1.0:
    raise ValueError(f"tau must be in (0, 1], got {tau}.")
  if not 0.0 <= beta < 1.0:
    raise ValueError(f"beta must be in [0, 1), got {beta}.")
  if max_rank < 0:
    raise ValueError(f"max_rank must be >= 0, got {max_rank}.")
  if predictor not in SVD_PREDICTORS:
    raise ValueError(f"predictor must be one of {SVD_PREDICTORS}, got {predictor!r}.")
  if n_derivatives < 0:
    raise ValueError(f"n_derivatives must be >= 0, got {n_derivatives}.")


class SVDState:
  """Per-stream SVD-Cache state for one named calibration stream."""

  def __init__(
    self,
    tau: float = 0.9,
    beta: float = 0.9,
    max_rank: int = 0,
    predictor: str = "ema",
    n_derivatives: int = 1,
  ):
    """Initialize an empty state; the basis is fitted lazily on the first update.

    :param tau: Cumulative squared-singular-value energy to retain, in (0, 1].
    :param beta: EMA coefficient of the principal stream, in [0, 1); unused by the "taylor"
        predictor.
    :param max_rank: Hard cap on the retained rank; 0 keeps the energy choice.
    :param predictor: "ema" for the paper's principal EMA, "taylor" for the SVD+Taylor combination
        (Taylor ladder on the low-rank coefficients).
    :param n_derivatives: Taylor orders kept by the "taylor" predictor.
    """

    assert 0.0 < tau <= 1.0, f"tau must be in (0, 1], got {tau}."
    assert 0.0 <= beta < 1.0, f"beta must be in [0, 1), got {beta}."
    assert max_rank >= 0, f"max_rank must be >= 0, got {max_rank}."
    assert predictor in SVD_PREDICTORS, (f"predictor must be one of {SVD_PREDICTORS}, "
                                         f"got {predictor!r}.")
    assert n_derivatives >= 0, f"n_derivatives must be >= 0, got {n_derivatives}."
    self.tau = tau
    self.beta = beta
    self.max_rank = max_rank
    self.predictor = predictor
    self.n_derivatives = n_derivatives
    self.order = n_derivatives + 1
    self.reset()

  def reset(self):
    """Drop the basis, the forecast buffers, and every step counter."""

    # `current_step` restarts at -1 (the first `mark_step_begin` yields 0), which
    # the cache context also relies on when a fresh context is built per call.
    self.current_step = -1
    self.last_non_approximated_step = -1
    self.basis: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    self.ref_shape: Optional[torch.Size] = None
    self.ref_dtype: Optional[torch.dtype] = None
    self.principal: Optional[torch.Tensor] = None
    self.residual: Optional[torch.Tensor] = None
    self.ema: Optional[torch.Tensor] = None
    self.coeff: Optional[torch.Tensor] = None
    # Divided-difference Taylor ladder on the low-rank coefficients ([M, k]).
    self._dc_prev: List[Optional[torch.Tensor]] = [None] * self.order
    self._dc_cur: List[Optional[torch.Tensor]] = [None] * self.order
    self._ema_step: Optional[int] = None
    self._warned = False

  def mark_step_begin(self):
    """Advance the logical step counter by one."""

    self.current_step += 1

  def _fit_basis(self, Y: torch.Tensor):
    """Fit the low-rank basis from the first full-compute tensor of this stream.

    The feature is flattened to ``[M, D]`` with the channel axis last, so a
    ``[B, S, D]`` block tensor shares one basis across the batch and a packed
    ``[S, D]`` tensor fits directly. Higher-rank inputs are rejected instead of
    being silently projected along the wrong axis.

    :param Y: Full-compute block tensor, 2D ``[S, D]`` or 3D ``[B, S, D]``.
    :returns: ``(V, sigma)`` — ``V`` is ``[D, k]``, ``sigma`` is ``[k]``.
    """

    assert Y.dim() in (
      2,
      3,
    ), f"SVDCache expects a 2D or 3D block tensor, got shape {tuple(Y.shape)}."
    X = Y.reshape(-1, Y.shape[-1]).to(torch.float32)
    _, S, Vh = torch.linalg.svd(X, full_matrices=False)

    # Rank by retained energy, then bound it by the numerically non-zero spectrum
    # so the 1 / sigma normalisation stays finite.
    energy = S.pow(2)
    total = energy.sum()
    if total <= 0:
      rank = 1
    else:
      cum = energy.cumsum(0) / total
      rank = int((cum < self.tau).sum().item()) + 1
    valid = int((S > S[0] * _SIGMA_REL_FLOOR).sum().item()) if S.numel() else 1
    rank = max(1, min(rank, max(valid, 1), S.numel()))
    if self.max_rank > 0:
      rank = min(rank, self.max_rank)

    V = Vh[:rank].mT.contiguous()
    sigma = S[:rank].clone()
    retained = float(energy[:rank].sum() / total) if total > 0 else 1.0
    logger.info(f"SVD-Cache basis fitted: rank={rank}/{S.numel()} "
                f"(tau={self.tau}, max_rank={self.max_rank}), "
                f"retained_energy={retained:.4f}")
    return V, sigma

  def _split(self, Y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a block tensor into its principal part, residual, and coefficients.

    :param Y: Full-compute block tensor in the reference shape.
    :returns: ``(principal, residual, coeff)`` — the first two in the reference shape and
        dtype, ``coeff`` is the fp32 ``[M, k]`` projection ``Y V`` that the Taylor ladder
        extrapolates.
    """

    assert self.basis is not None, "Basis must be fitted before splitting."
    V, sigma = self.basis
    # A rank-deficient feature (or an all-zero one) leaves a zero singular value;
    # clamp it so the 1 / sigma normalisation stays finite instead of producing nan.
    sigma = sigma.clamp_min(torch.finfo(sigma.dtype).tiny)
    X = Y.reshape(-1, Y.shape[-1]).to(torch.float32)
    coeff = X @ V  # [M, k]
    principal = ((coeff / sigma) * sigma) @ V.mT
    residual = X - principal
    return (principal.to(Y.dtype).reshape(Y.shape), residual.to(Y.dtype).reshape(Y.shape), coeff)

  def _update_taylor(self, coeff: torch.Tensor):
    """Maintain the divided-difference Taylor ladder on the low-rank coefficients.

    Same recurrence as ``TaylorSeerState.derivative`` but on ``[M, k]`` instead of the full
    ``[M, D]`` tensor, which is what makes the SVD+Taylor combination cheap.

    :param coeff: Current full-compute coefficients ``Y V``, shape ``[M, k]``.
    """

    # Shift the ladder first (same ordering as TaylorSeerState.update): the new derivatives
    # are divided differences against the ladder of the previous anchor, so shifting after
    # the loop would compare against an extra-step-old ladder and skip one order.
    self._dc_prev = self._dc_cur
    dc_cur: List[Optional[torch.Tensor]] = [None] * self.order
    dc_cur[0] = coeff
    window = self.current_step - self.last_non_approximated_step
    for i in range(self.n_derivatives):
      # Unlike TaylorSeerState there is no `current_step > 1` guard: with a frozen basis and
      # an explicit anchor step the divided difference is well defined from the second
      # anchor on, which keeps the first skip run after warmup from silently degrading to
      # plain reuse.
      if self._dc_prev[i] is not None and window > 0:
        dc_cur[i + 1] = (dc_cur[i] - self._dc_prev[i]) / window
      else:
        break
    self._dc_cur = dc_cur

  def _approximate_taylor(self) -> Optional[torch.Tensor]:
    """Extrapolate the coefficients by the elapsed step count and project back.

    :returns: The forecast principal part in the reference shape (fp32), or None when the
        extrapolated coefficients blow up.
    """

    assert self.basis is not None
    elapsed = self.current_step - self.last_non_approximated_step
    acc: Optional[torch.Tensor] = None
    for i, derivative in enumerate(self._dc_cur):
      if derivative is None:
        break
      term = derivative * (elapsed ** i) / math.factorial(i)
      acc = term if acc is None else acc + term
    assert acc is not None, "Taylor ladder is empty; update() must run before approximate()."
    # The coefficients are [M, k] (k ~ 4 here), so this divergence guard costs nothing; unlike
    # a full-tensor isfinite it does not add a [B, S, D] reduction to every cached step.
    if not torch.isfinite(acc).all():
      return None
    return (acc @ self.basis[0].mT).reshape(self.ref_shape)

  def update(self, Y: torch.Tensor):
    """Commit a full-compute block tensor: fit/refresh the basis, EMA, and ladder.

    :param Y: Full-compute block tensor in the reference shape.
    """

    if self.ref_shape is not None and Y.shape != self.ref_shape:
      self.reset()
    if self.basis is None:
      self.ref_shape = Y.shape
      self.ref_dtype = Y.dtype
      self.basis = self._fit_basis(Y)

    principal, residual, coeff = self._split(Y)
    if self.ema is None:
      self.ema = principal.clone()
    else:
      # EMA across full-compute steps: a lag filter on the principal stream, so a
      # single noisy anchor cannot swing the forecast.
      self.ema = (self.beta * self.ema.float() + (1.0 - self.beta) * principal.float()).to(Y.dtype)
    self.principal = principal
    self.residual = residual
    self.coeff = coeff
    self._update_taylor(coeff)
    self.last_non_approximated_step = self.current_step
    # A new anchor restarts the per-cached-step EMA advance.
    self._ema_step = None

  def _fallback(self) -> torch.Tensor:
    """Return the anchor value (principal + residual) when the forecast degenerates."""

    anchor = (self.principal.float() + self.residual.float()).to(self.ref_dtype)
    if not self._warned:
      logger.warning("SVD-Cache forecast is not finite; falling back to the anchor value.")
      self._warned = True
    return anchor.reshape(self.ref_shape)

  def _approximate_ema(self) -> torch.Tensor:
    """Advance the principal EMA one tick towards the anchor's principal part.

    The advance is idempotent within a step, so a second query for the same step returns the same
    tensor instead of drifting.

    :returns: The forecast principal part in the reference shape, fp32.
    """

    if self._ema_step != self.current_step:
      self.ema = (self.beta * self.ema.float() + (1.0 - self.beta) * self.principal.float()).to(
        self.ref_dtype)
      self._ema_step = self.current_step
    return self.ema.float()

  def approximate(self) -> torch.Tensor:
    """Forecast the block tensor for the current (cached) step.

    The principal part comes from the configured predictor ("ema" advances the paper's smoothing one
    tick, "taylor" extrapolates the low-rank coefficient ladder forward), and the residual frozen at
    the last full-compute step is added back.

    :returns: The forecast tensor in the reference shape and dtype.
    """

    assert self.residual is not None, (
      "SVD-Cache approximate() called before any full-compute update; the cache "
      "context must warm up at least one step before reusing a block group.")
    if self.predictor == "taylor":
      forecast = self._approximate_taylor()
      if forecast is None:
        return self._fallback()
    else:
      # The EMA is a convex combination of two finite anchors, so it cannot blow up; the
      # per-step finiteness check is only needed on the extrapolating path.
      forecast = self._approximate_ema()
    return (forecast + self.residual.float()).to(self.ref_dtype).reshape(self.ref_shape)


class SVDCalibrator(CalibratorBase):
  """Calibrator that forecasts cached block tensors with an SVD low-rank split."""

  def __init__(
    self,
    tau: float = 0.9,
    beta: float = 0.9,
    max_rank: int = 0,
    predictor: str = "ema",
    n_derivatives: int = 1,
    **kwargs,
  ):
    """Create a calibrator whose states are keyed by logical tensor names.

    :param tau: Cumulative squared-singular-value energy to retain, in (0, 1].
    :param beta: EMA coefficient of the principal stream, in [0, 1).
    :param max_rank: Hard cap on the retained rank; 0 keeps the energy choice.
    :param predictor: "ema" for the paper's principal EMA, "taylor" for the SVD+Taylor combination
        (Taylor ladder on the low-rank coefficients).
    :param n_derivatives: Taylor orders kept by the "taylor" predictor.
    :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
    """

    # Validate eagerly and with ValueError: the per-stream SVDState is created lazily on the
    # first full-compute step, so an assert there would only fire in the middle of a
    # diffusion run (and would be stripped entirely under `python -O`).
    validate_svd_params(tau, beta, max_rank, predictor, n_derivatives)
    self.tau = tau
    self.beta = beta
    self.max_rank = max_rank
    self.predictor = predictor
    self.n_derivatives = n_derivatives
    self.states: Dict[str, SVDState] = {}
    self.reset_cache()

  def reset_cache(self):
    """Reset every tracked `SVDState` without dropping the key mapping."""

    if self.states:
      for state in self.states.values():
        state.reset()

  def maybe_init_state(self, name: str = "default"):
    """Lazily create one SVD state for a named tensor stream.

    :param name: Logical tensor-stream name.
    """

    if name not in self.states:
      self.states[name] = SVDState(
        tau=self.tau,
        beta=self.beta,
        max_rank=self.max_rank,
        predictor=self.predictor,
        n_derivatives=self.n_derivatives,
      )

  def mark_step_begin(self, *args, **kwargs):
    """Advance every tracked state's step counter.

    :param args: Additional positional arguments forwarded to the underlying implementation.
    :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
    """

    if self.states:
      for state in self.states.values():
        state.mark_step_begin()

  def approximate(self, name: str = "default") -> torch.Tensor:
    """Forecast the next tensor for one named stream.

    :param name: Logical tensor-stream name.
    :returns: The forecast tensor for the named stream.
    """

    assert name in self.states, f"State '{name}' not found."
    return self.states[name].approximate()

  def update(self, Y: torch.Tensor, name: str = "default"):
    """Feed a fully computed tensor into one named SVD state.

    :param Y: Fully computed tensor for the named stream.
    :param name: Logical tensor-stream name.
    """

    self.maybe_init_state(name)
    self.states[name].update(Y)

  def step(self, Y: torch.Tensor, name: str = "default"):
    """Advance one named stream and return either computed or forecast output.

    :param Y: Fully computed tensor for the named stream when computation is required.
    :param name: Logical tensor-stream name.
    :returns: The exact tensor or the SVD-Cache forecast for the named stream.
    """

    self.maybe_init_state(name)
    state = self.states[name]
    state.mark_step_begin()
    if state.residual is None:
      state.update(Y)
      return Y
    return state.approximate()

  def __repr__(self):
    return (f"SVDCalibrator_T({self.tau}, {self.predictor}, "
            f"B{self.beta}, R{self.max_rank})")
