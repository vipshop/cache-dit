from .base import CalibratorBase
from .taylorseer import TaylorSeerCalibrator
from .foca import FoCaCalibrator
from .dmd import DMDCalibrator
from .svd import SVDCalibrator, validate_svd_params
from .hicache import HiCacheCalibrator, validate_hicache_params
from .brace import BRACECalibrator, validate_brace_params
from .bezicast import BeziCastCalibrator, validate_bezicast_params
from .residual_motion import ResidualMotionCalibrator, validate_residual_motion_params

import dataclasses
from typing import Any, Dict, Optional

from cache_dit.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class CalibratorConfig:
  """Base config for optional calibrators used by cache contexts.

  Calibrators refine hidden-state reuse by forecasting either residuals or full hidden states for
  selected branches. This base config stores the generic enable flags, the selected calibrator type,
  and backend-specific kwargs.
  """

  # enable_calibrator (`bool`, *required*,  defaults to False):
  #     Whether to enable calibrator, if True. means that user want to use DBCache
  #     with specific calibrator for hidden_states (or hidden_states residual),
  #     such as taylorseer, foca, and so on.
  enable_calibrator: bool = False
  # enable_encoder_calibrator (`bool`, *required*,  defaults to False):
  #     Whether to enable calibrator, if True. means that user want to use DBCache
  #     with specific calibrator for encoder_hidden_states (or encoder_hidden_states
  #     residual), such as taylorseer, foca, and so on.
  enable_encoder_calibrator: bool = False
  # calibrator_type (`str`, *required*,  defaults to 'taylorseer'):
  #    The specific type for calibrator, taylorseer or foca, etc.
  calibrator_type: str = "taylorseer"
  # calibrator_cache_type (`str`, *required*,  defaults to 'residual'):
  #    The specific cache type for calibrator, residual or hidden_states.
  calibrator_cache_type: str = "residual"
  # calibrator_kwargs (`dict`, *optional*, defaults to {}):
  #    Init kwargs for specific calibrator, taylorseer or foca, etc.
  calibrator_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

  def strify(self, **kwargs) -> str:
    """Return a short human-readable tag for logging and summaries.

    :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
    :returns: A short summary string describing the calibrator selection.
    """

    return "CalibratorBase"

  def to_kwargs(self) -> Dict:
    """Return implementation kwargs used by the calibrator factory.

    :returns: A shallow copy of the calibrator-specific initialization kwargs; `None` is treated as
        empty, which is what `empty()`/`reset()` leave behind for `refresh_context`.
    """

    return dict(self.calibrator_kwargs or {})

  def as_dict(self) -> dict:
    return dataclasses.asdict(self)

  def update(self, **kwargs) -> "CalibratorConfig":
    """Update known fields in place while ignoring unknown keys.

    :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
    :returns: `self` after applying the known non-`None` fields.
    """

    for key, value in kwargs.items():
      if hasattr(self, key):
        if value is not None:
          setattr(self, key, value)
    return self

  def empty(self, **kwargs) -> "CalibratorConfig":
    """Clear non-constant fields so callers can rebuild a config from scratch.

    :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
    :returns: `self` after clearing mutable fields and applying optional overrides.
    """

    # Set all fields to None
    skip_constants = {"calibrator_type"}
    for field in dataclasses.fields(self):
      if hasattr(self, field.name):
        if field.name not in skip_constants:
          setattr(self, field.name, None)
    if kwargs:
      self.update(**kwargs)
    return self

  def reset(self, **kwargs) -> "CalibratorConfig":
    """Reset this config to an empty state, then apply optional overrides.

    :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
    :returns: `self` after resetting the config.
    """

    return self.empty(**kwargs)


@dataclasses.dataclass
class TaylorSeerCalibratorConfig(CalibratorConfig):
  """Config for the TaylorSeer forecasting calibrator."""

  # TaylorSeers: From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers
  # link: https://arxiv.org/pdf/2503.06923

  # enable_calibrator (`bool`, *required*,  defaults to True):
  #     Whether to enable calibrator, if True. means that user want to use DBCache
  #     with specific calibrator for hidden_states (or hidden_states residual),
  #     such as taylorseer, foca, and so on.
  enable_calibrator: bool = True
  # enable_encoder_calibrator (`bool`, *required*,  defaults to True):
  #     Whether to enable calibrator, if True. means that user want to use DBCache
  #     with specific calibrator for encoder_hidden_states (or encoder_hidden_states
  #     residual), such as taylorseer, foca, and so on.
  enable_encoder_calibrator: bool = True
  # calibrator_type (`str`, *required*,  defaults to 'taylorseer'):
  #    The specific type for calibrator, taylorseer or foca, etc.
  calibrator_type: str = "taylorseer"
  # taylorseer_order (`int`, *required*, defaults to 1):
  #    The order of taylorseer, higher values of n_derivatives will lead to longer computation time,
  #    the recommended value is 1 or 2. Please check [TaylorSeers: From Reusing to Forecasting:
  #    Accelerating Diffusion Models with TaylorSeers](https://arxiv.org/pdf/2503.06923) for
  #    more details.
  taylorseer_order: int = 1

  def strify(self, **kwargs) -> str:
    """Return a compact tag that includes the selected Taylor order.

    :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
    :returns: A compact TaylorSeer tag for logs, summaries, or filenames.
    """

    if kwargs.get("details", False):
      if self.taylorseer_order:
        return f"TaylorSeer_O({self.taylorseer_order})"
      return "NONE"

    if self.taylorseer_order:
      return f"T1O{self.taylorseer_order}"
    return "NONE"

  def to_kwargs(self) -> Dict:
    """Translate config fields into `TaylorSeerCalibrator` init kwargs.

    :returns: Keyword arguments expected by `TaylorSeerCalibrator`.
    """

    kwargs = dict(self.calibrator_kwargs or {})
    kwargs["n_derivatives"] = self.taylorseer_order
    return kwargs


@dataclasses.dataclass
class DMDCalibratorConfig(CalibratorConfig):
  """Config for the Dynamic Mode Decomposition (Prony) forecasting calibrator.

  An EXPONENTIAL-basis alternative to TaylorSeer's polynomial forecast: DMD (Schmid 2010; the SVD-
  regularised generalisation of Prony's method) identifies the linear propagator of the cached
  feature stream from recent compute-step snapshots and extrapolates by eigenvalue powers — exact on
  the (locally) exponential trajectories diffusion features follow, where a polynomial diverges with
  the cache interval. NOT Distribution Matching Distillation.
  """

  # enable_calibrator (`bool`, *required*,  defaults to True):
  #     Whether to enable calibrator, if True. means that user want to use DBCache
  #     with specific calibrator for hidden_states (or hidden_states residual),
  #     such as taylorseer, foca, dmd, and so on.
  enable_calibrator: bool = True
  # enable_encoder_calibrator (`bool`, *required*,  defaults to True):
  #     Whether to enable calibrator, if True. means that user want to use DBCache
  #     with specific calibrator for encoder_hidden_states (or encoder_hidden_states
  #     residual), such as taylorseer, foca, dmd, and so on.
  enable_encoder_calibrator: bool = True
  # calibrator_type (`str`, *required*,  defaults to 'dmd'):
  #    The specific type for calibrator, taylorseer, foca or dmd, etc.
  calibrator_type: str = "dmd"
  # dmd_history (`int`, *required*, defaults to 6):
  #    Number of recent compute-step snapshots retained per stream. >= 4 uniformly
  #    spaced snapshots are needed before the exponential fit engages (one complex
  #    pole costs two real degrees of freedom); below the floor the calibrator
  #    falls back to the Taylor expansion automatically. 5-6 is the sweet spot —
  #    the feature dynamics drift across timesteps, so longer windows hurt.
  dmd_history: int = 6
  # dmd_rank (`int`, *optional*, defaults to 0):
  #    SVD truncation rank of the snapshot matrix; 0 selects it from the spectrum
  #    (drop modes below 1e-4 of the leading singular value). The truncation is
  #    what rejects the noise subspace.
  dmd_rank: int = 0
  # dmd_ridge (`float`, *optional*, defaults to 1e-8):
  #    Tikhonov term added to the inverted singular values.
  dmd_ridge: float = 1e-8
  # dmd_svd_precision (`str`, *optional*, defaults to 'medium'):
  #    SVD precision mode for the DMD snapshot matrix. ``"medium"`` uses default
  #    ``torch.linalg.svd`` (gesdd, balanced), ``"low"`` uses randomised
  #    ``torch.svd_lowrank`` (niter=1, deterministic seed), ``"high"`` uses
  #    ``driver="gesvd"``.
  dmd_svd_precision: str = "medium"

  def strify(self, **kwargs) -> str:
    """Return a compact tag that includes the snapshot-history length and SVD precision.

    :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
    :returns: A compact DMD tag for logs, summaries, or filenames.
    """

    prec = self.dmd_svd_precision[0]
    if kwargs.get("details", False):
      return f"DMD_H({self.dmd_history}, {self.dmd_svd_precision})"
    return f"DMDH{self.dmd_history}{prec}" if prec != "m" else f"DMDH{self.dmd_history}"

  def to_kwargs(self) -> Dict:
    """Translate config fields into `DMDCalibrator` init kwargs.

    :returns: Keyword arguments expected by `DMDCalibrator`.
    """

    kwargs = dict(self.calibrator_kwargs or {})
    kwargs["history"] = self.dmd_history
    kwargs["rank"] = self.dmd_rank
    kwargs["ridge"] = self.dmd_ridge
    kwargs["svd_precision"] = self.dmd_svd_precision
    return kwargs


@dataclasses.dataclass
class FoCaCalibratorConfig(CalibratorConfig):
  """Config placeholder for the future FoCa calibrator backend."""

  # FoCa: Forecast then Calibrate: Feature Caching as ODE for Efficient Diffusion Transformers
  # link: https://arxiv.org/pdf/2508.16211

  # enable_calibrator (`bool`, *required*,  defaults to True):
  #     Whether to enable calibrator, if True. means that user want to use DBCache
  #     with specific calibrator for hidden_states (or hidden_states residual),
  #     such as taylorseer, foca, and so on.
  enable_calibrator: bool = True
  # enable_encoder_calibrator (`bool`, *required*,  defaults to True):
  #     Whether to enable calibrator, if True. means that user want to use DBCache
  #     with specific calibrator for encoder_hidden_states (or encoder_hidden_states
  #     residual), such as taylorseer, foca, and so on.
  enable_encoder_calibrator: bool = True
  # calibrator_type (`str`, *required*,  defaults to 'taylorseer'):
  #    The specific type for calibrator, taylorseer or foca, etc.
  calibrator_type: str = "foca"

  def strify(self, **kwargs) -> str:
    return "FoCa"

  def to_kwargs(self) -> Dict:
    """Translate config fields into `FoCaCalibrator` init kwargs.

    FoCA has no hyper-parameters, so this returns an empty dict.

    :returns: Empty keyword arguments dict for `FoCaCalibrator`.
    """

    return {}


def _tag_number(value: float) -> str:
  """Render a hyper-parameter compactly for summary tags and artifact filenames.

  :param value: Numeric hyper-parameter value.
  :returns: `0.9` as `9`, `0.85` as `85`, `1.0` as `1`.
  """

  text = f"{value:g}"
  return text[2:] if text.startswith("0.") else text


@dataclasses.dataclass
class SVDCalibratorConfig(CalibratorConfig):
  """Config for the SVD-Cache low-rank split calibrator.

  Splits the cached block tensor in the token x channel plane into a rank-k principal part and a
  high-frequency residual, EMA-smooths the principal stream, freezes the residual, and reuses their
  sum for the cached steps. NOTE: this is SVD-Cache (arXiv 2601.07396), not SVDQuant — the latter is
  a weight-quantization backend selected with `--quantize-params quant_type=svdq_*`.
  """

  # enable_calibrator (`bool`, *required*,  defaults to True):
  #     Whether to enable calibrator, if True. means that user want to use DBCache
  #     with specific calibrator for hidden_states (or hidden_states residual),
  #     such as taylorseer, foca, dmd, svd, and so on.
  enable_calibrator: bool = True
  # enable_encoder_calibrator (`bool`, *required*,  defaults to True):
  #     Whether to enable calibrator, if True. means that user want to use DBCache
  #     with specific calibrator for encoder_hidden_states (or encoder_hidden_states
  #     residual), such as taylorseer, foca, dmd, svd, and so on.
  enable_encoder_calibrator: bool = True
  # calibrator_type (`str`, *required*,  defaults to 'svd'):
  #    The specific type for calibrator, taylorseer, foca, dmd or svd, etc.
  calibrator_type: str = "svd"
  # svd_tau (`float`, *optional*, defaults to 0.9):
  #    Cumulative squared-singular-value energy retained by the principal subspace,
  #    in (0, 1]. 0.85-0.9 keeps a handful of directions on image DiTs; 1.0 keeps the
  #    full rank and therefore reproduces plain reuse (unit tests only).
  svd_tau: float = 0.9
  # svd_beta (`float`, *optional*, defaults to 0.9):
  #    EMA coefficient of the principal stream, in [0, 1). 0 makes the forecast
  #    identical to DBCache's plain reuse; larger values smooth harder across steps.
  svd_beta: float = 0.9
  # svd_max_rank (`int`, *optional*, defaults to 0):
  #    Hard cap on the retained rank; 0 keeps the energy-based choice.
  svd_max_rank: int = 0
  # svd_predictor (`str`, *optional*, defaults to 'ema'):
  #    How the principal part is forecast: 'ema' is the paper's per-step smoothing of the
  #    principal stream, 'taylor' is the SVD + Taylor combination (paper Fig. 5(b)) which
  #    keeps a Taylor ladder on the low-rank coefficients and extrapolates it forward.
  svd_predictor: str = "ema"
  # svd_order (`int`, *optional*, defaults to 1):
  #    Taylor orders kept by the 'taylor' predictor; ignored by 'ema'.
  svd_order: int = 1

  def __post_init__(self):
    """Validate the SVD-Cache hyper-parameters at construction time.

    :raises ValueError: When any field is outside its domain, so a bad YAML/CLI value fails while
        the config is built instead of in the middle of the first diffusion run.
    """

    validate_svd_params(self.svd_tau, self.svd_beta, self.svd_max_rank, self.svd_predictor,
                        self.svd_order)

  def strify(self, **kwargs) -> str:
    """Return a compact tag that includes the energy target and the predictor.

    :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
    :returns: A compact SVD-Cache tag for logs, summaries, or filenames.
    """

    tau = _tag_number(self.svd_tau)
    rank = f"R{self.svd_max_rank}" if self.svd_max_rank else ""
    if kwargs.get("details", False):
      predictor = (f"taylor{self.svd_order}"
                   if self.svd_predictor == "taylor" else f"ema, B={self.svd_beta}")
      return f"SVD_T({self.svd_tau}, {predictor}, R={self.svd_max_rank})"
    if self.svd_predictor == "taylor":
      return f"SVDT{tau}T{self.svd_order}{rank}"
    return f"SVDT{tau}B{_tag_number(self.svd_beta)}{rank}"

  def to_kwargs(self) -> Dict:
    """Translate config fields into `SVDCalibrator` init kwargs.

    :returns: Keyword arguments expected by `SVDCalibrator`.
    """

    kwargs = dict(self.calibrator_kwargs or {})
    kwargs["tau"] = self.svd_tau
    kwargs["beta"] = self.svd_beta
    kwargs["max_rank"] = self.svd_max_rank
    kwargs["predictor"] = self.svd_predictor
    kwargs["n_derivatives"] = self.svd_order
    return kwargs


@dataclasses.dataclass
class HiCacheCalibratorConfig(CalibratorConfig):
  """Config for the HiCache damped-Hermite forecasting calibrator.

  HiCache (arXiv 2508.16984) keeps TaylorSeer's finite-difference ladder and replaces only the
  extrapolation basis: ``elapsed**k -> sigma**k * H_k(sigma * elapsed)``, with ``H`` the
  physicist Hermite polynomial. The paper's fixed refresh interval and per-layer caching are NOT
  replicated: DBCache's adaptive threshold owns the schedule, so ``sigma`` is coupled to the cache
  aggressiveness and must be re-swept per model instead of copied from the paper.
  """

  # enable_calibrator (`bool`, *required*,  defaults to True):
  #     Whether to enable calibrator, if True. means that user want to use DBCache
  #     with specific calibrator for hidden_states (or hidden_states residual),
  #     such as taylorseer, foca, dmd, svd, hicache, and so on.
  enable_calibrator: bool = True
  # enable_encoder_calibrator (`bool`, *required*,  defaults to True):
  #     Whether to enable calibrator, if True. means that user want to use DBCache
  #     with specific calibrator for encoder_hidden_states (or encoder_hidden_states
  #     residual), such as taylorseer, foca, dmd, svd, hicache, and so on.
  enable_encoder_calibrator: bool = True
  # calibrator_type (`str`, *required*,  defaults to 'hicache'):
  #    The specific type for calibrator, taylorseer, foca, dmd, svd or hicache, etc.
  calibrator_type: str = "hicache"
  # hicache_order (`int`, *optional*, defaults to 2):
  #    Highest finite-difference order kept (the paper's N_order, capped at 4). 0 reproduces
  #    plain anchor reuse, 1 keeps a first-order extrapolation, and 2 is what the paper's FLUX
  #    numbers use.
  hicache_order: int = 2
  # hicache_sigma (`float`, *optional*, defaults to 0.5):
  #    Damping factor of the Hermite basis, in (0, 1]. The paper uses 0.4-0.5. sigma=1 keeps the
  #    undamped physicist-Hermite basis, whose ablation in the paper is worse than Taylor, so it
  #    is a control point rather than a default.
  hicache_sigma: float = 0.5

  def __post_init__(self):
    """Validate the HiCache hyper-parameters while the config is built.

    :raises ValueError: When `hicache_order` or `hicache_sigma` is outside its domain, so a bad
        YAML/CLI value fails at parse time instead of in the middle of the first diffusion run.
    """

    validate_hicache_params(self.hicache_sigma, self.hicache_order)

  def strify(self, **kwargs) -> str:
    """Return a compact tag that includes the order and the damping factor.

    :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
    :returns: A compact HiCache tag for logs, summaries, or filenames.
    """

    if kwargs.get("details", False):
      return f"HiCache_O({self.hicache_order}, sigma={self.hicache_sigma:g})"
    return f"HC{self.hicache_order}S{self.hicache_sigma:g}"

  def to_kwargs(self) -> Dict:
    """Translate config fields into `HiCacheCalibrator` init kwargs.

    :returns: Keyword arguments expected by `HiCacheCalibrator`.
    """

    kwargs = dict(self.calibrator_kwargs or {})
    kwargs["n_derivatives"] = self.hicache_order
    kwargs["sigma"] = self.hicache_sigma
    return kwargs


@dataclasses.dataclass
class BRACECalibratorConfig(CalibratorConfig):
  """Config for BRACE barycentric rational forecasting with adapted Chebyshev weights.

  BRACE (arXiv 2608.07572) aggregates a FIFO of real full-compute anchors using its second
  barycentric form. Only the predictor is enabled here; DBCache retains adaptive scheduling and
  block granularity, and logical step indices provide the canonical coordinates.
  """

  enable_calibrator: bool = True
  enable_encoder_calibrator: bool = True
  calibrator_type: str = "brace"
  brace_capacity: int = 3
  brace_gamma: float = 0.7

  def __post_init__(self):
    """Validate BRACE parameters while the config is built.

    :raises ValueError: When capacity or gamma is outside its supported domain.
    """

    validate_brace_params(self.brace_capacity, self.brace_gamma)

  def strify(self, **kwargs) -> str:
    """Return a filename-safe tag containing the window capacity and gamma.

    :param kwargs: Set `details=True` for a human-readable summary tag.
    :returns: A compact or detailed BRACE configuration tag.
    """

    if kwargs.get("details", False):
      return f"BRACE_C({self.brace_capacity}, gamma={self.brace_gamma:g})"
    return f"BRACEC{self.brace_capacity}G{_tag_number(self.brace_gamma)}"

  def to_kwargs(self) -> Dict:
    """Translate config fields into `BRACECalibrator` constructor kwargs.

    :returns: Keyword arguments expected by `BRACECalibrator`.
    """

    kwargs = dict(self.calibrator_kwargs or {})
    kwargs["capacity"] = self.brace_capacity
    kwargs["gamma"] = self.brace_gamma
    return kwargs


@dataclasses.dataclass
class BeziCastCalibratorConfig(CalibratorConfig):
  """Config for BeziCast Bernstein/Bezier forecasting and its optional guardrail.

  BeziCast (arXiv 2610.16448) fits historical anchors with Tikhonov-stabilized Bernstein equivalent
  weights. DBCache retains scheduling and block granularity. The default pure predictor uses the
  paper's history=6, control_points=4, and ridge=0.5; guardrail thresholds are opt-in because the
  paper does not provide a universal default setting.
  """

  enable_calibrator: bool = True
  enable_encoder_calibrator: bool = True
  calibrator_type: str = "bezicast"
  bezicast_history: int = 6
  bezicast_control_points: int = 4
  bezicast_ridge: float = 0.5
  bezicast_guardrail: bool = False
  bezicast_u_max: float = 1.5
  bezicast_beta_max: float = 1.0
  bezicast_rho: Optional[float] = None
  bezicast_gamma: float = 1.0

  def __post_init__(self):
    """Validate BeziCast fitting and guardrail parameters while building the config.

    :raises ValueError: When any parameter is outside its supported domain.
    """

    validate_bezicast_params(
      self.bezicast_history,
      self.bezicast_control_points,
      self.bezicast_ridge,
      self.bezicast_guardrail,
      self.bezicast_u_max,
      self.bezicast_beta_max,
      self.bezicast_rho,
      self.bezicast_gamma,
    )

  def strify(self, **kwargs) -> str:
    """Return a filename-safe tag with fitting and guardrail settings.

    :param kwargs: Set `details=True` for a human-readable summary tag.
    :returns: A compact or detailed BeziCast configuration tag.
    """

    if kwargs.get("details", False):
      rho = "auto" if self.bezicast_rho is None else f"{self.bezicast_rho:g}"
      return (f"BeziCast_K({self.bezicast_history}, n={self.bezicast_control_points}, "
              f"ridge={self.bezicast_ridge:g}, guardrail={self.bezicast_guardrail}, "
              f"u_max={self.bezicast_u_max:g}, beta_max={self.bezicast_beta_max:g}, "
              f"rho={rho}, gamma={self.bezicast_gamma:g})")
    tag = (f"BZCK{self.bezicast_history}N{self.bezicast_control_points}"
           f"R{_tag_number(self.bezicast_ridge)}U{_tag_number(self.bezicast_u_max)}"
           f"G{int(self.bezicast_guardrail)}")
    if self.bezicast_guardrail:
      rho = "auto" if self.bezicast_rho is None else f"{self.bezicast_rho:g}"
      tag += f"RHO{rho}B{self.bezicast_beta_max:g}GM{self.bezicast_gamma:g}"
    return tag

  def to_kwargs(self) -> Dict:
    """Translate config fields into `BeziCastCalibrator` constructor kwargs.

    :returns: Keyword arguments expected by `BeziCastCalibrator`.
    """

    kwargs = dict(self.calibrator_kwargs or {})
    kwargs.update(
      history=self.bezicast_history,
      control_points=self.bezicast_control_points,
      ridge=self.bezicast_ridge,
      guardrail=self.bezicast_guardrail,
      u_max=self.bezicast_u_max,
      beta_max=self.bezicast_beta_max,
      rho=self.bezicast_rho,
      gamma=self.bezicast_gamma,
    )
    return kwargs


@dataclasses.dataclass
class ResidualMotionCalibratorConfig(CalibratorConfig):
  """Config for SoftCache's Residual Motion predictor (arXiv 2610.22369).

  Moves the cached residual along the secant of the last two real anchors, measured in the
  flow-matching sigma rather than the step count, so it stays right on non-uniform sigma grids.
  Needs a scheduler with flow-matching `sigmas`: the pipeline's, or `residual_motion_scheduler`
  passed to `enable_cache` for transformer-only caching.
  """

  enable_calibrator: bool = True
  enable_encoder_calibrator: bool = True
  calibrator_type: str = "residual_motion"
  residual_motion_beta: float = 1.0

  def __post_init__(self):
    """Validate the secant step size while building the config.

    :raises ValueError: When beta is outside [0, 1].
    """

    validate_residual_motion_params(self.residual_motion_beta)

  def strify(self, **kwargs) -> str:
    """Return a filename-safe tag containing beta.

    :param kwargs: Set `details=True` for a human-readable summary tag.
    :returns: A compact or detailed Residual Motion configuration tag.
    """

    if kwargs.get("details", False):
      return f"ResidualMotion(beta={self.residual_motion_beta:g})"
    return f"RMB{_tag_number(self.residual_motion_beta)}"

  def to_kwargs(self) -> Dict:
    """Translate config fields into `ResidualMotionCalibrator` constructor kwargs.

    :returns: Keyword arguments expected by `ResidualMotionCalibrator`.
    """

    kwargs = dict(self.calibrator_kwargs or {})
    kwargs["beta"] = self.residual_motion_beta
    return kwargs


_CALIBRATOR_ALIASES = {
  "svd-calibrator": "svd",
  "svd-cache": "svd",
  "svdcache": "svd",
  "hicache-calibrator": "hicache",
  "hi-cache": "hicache",
  "brace-calibrator": "brace",
  "bezicast-calibrator": "bezicast",
  "bezier-cast": "bezicast",
  "residual-motion": "residual_motion",
  "rm": "residual_motion",
  "taylorseer-calibrator": "taylorseer",
  "dmd-calibrator": "dmd",
  "foca-calibrator": "foca",
}


def normalize_calibrator_type(calibrator_type: str) -> str:
  """Normalize a calibrator name so long forms resolve to the canonical short name.

  :param calibrator_type: Canonical name or a long/alias form, e.g. `svd-calibrator`.
  :returns: The canonical calibrator name, lower-cased.
  """

  key = str(calibrator_type).strip().lower()
  return _CALIBRATOR_ALIASES.get(key, key)


class Calibrator:
  """Factory that instantiates supported calibrator implementations."""

  _supported_calibrators = [
    "taylorseer",
    "dmd",
    "foca",
    "svd",
    "hicache",
    "brace",
    "bezicast",
    "residual_motion",
  ]

  def __new__(
    cls,
    calibrator_config: CalibratorConfig,
  ) -> CalibratorBase:
    """Construct the calibrator implementation described by `calibrator_config`.

    :param calibrator_config: Structured calibrator configuration to instantiate.
    :returns: The concrete calibrator implementation.
    """

    calibrator_type = normalize_calibrator_type(calibrator_config.calibrator_type)
    assert (calibrator_type in cls._supported_calibrators
            ), f"Calibrator {calibrator_config.calibrator_type} is not supported now!"

    if calibrator_type == "taylorseer":
      return TaylorSeerCalibrator(**calibrator_config.to_kwargs())
    elif calibrator_type == "dmd":
      return DMDCalibrator(**calibrator_config.to_kwargs())
    elif calibrator_type == "foca":
      return FoCaCalibrator(**calibrator_config.to_kwargs())
    elif calibrator_type == "svd":
      return SVDCalibrator(**calibrator_config.to_kwargs())
    elif calibrator_type == "hicache":
      return HiCacheCalibrator(**calibrator_config.to_kwargs())
    elif calibrator_type == "brace":
      return BRACECalibrator(**calibrator_config.to_kwargs())
    elif calibrator_type == "bezicast":
      return BeziCastCalibrator(**calibrator_config.to_kwargs())
    elif calibrator_type == "residual_motion":
      return ResidualMotionCalibrator(**calibrator_config.to_kwargs())
    else:
      raise ValueError(f"Calibrator {calibrator_config.calibrator_type} is not supported now!")
