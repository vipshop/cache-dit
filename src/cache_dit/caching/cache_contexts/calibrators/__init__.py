from .base import CalibratorBase
from .taylorseer import TaylorSeerCalibrator
from .foca import FoCaCalibrator

import dataclasses
from typing import Any, Dict


from cache_dit.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class CalibratorConfig:
    """Base config for optional calibrators used by cache contexts.

    Calibrators refine hidden-state reuse by forecasting either residuals or
    full hidden states for selected branches. This base config stores the
    generic enable flags, the selected calibrator type, and backend-specific
    kwargs.
    """

    # enable_calibrator (`bool`, *required*,  defaults to False):
    #     Whether to enable calibrator, if True. means that user want to use DBCache
    #     with specific calibrator for hidden_states (or hidden_states redisual),
    #     such as taylorseer, foca, and so on.
    enable_calibrator: bool = False
    # enable_encoder_calibrator (`bool`, *required*,  defaults to False):
    #     Whether to enable calibrator, if True. means that user want to use DBCache
    #     with specific calibrator for encoder_hidden_states (or encoder_hidden_states
    #     redisual), such as taylorseer, foca, and so on.
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
        """Return a short human-readable tag for logging and summaries."""

        return "CalibratorBase"

    def to_kwargs(self) -> Dict:
        """Return implementation kwargs used by the calibrator factory."""

        return self.calibrator_kwargs.copy()

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    def update(self, **kwargs) -> "CalibratorConfig":
        """Update known fields in place while ignoring unknown keys."""

        for key, value in kwargs.items():
            if hasattr(self, key):
                if value is not None:
                    setattr(self, key, value)
        return self

    def empty(self, **kwargs) -> "CalibratorConfig":
        """Clear non-constant fields so callers can rebuild a config from scratch."""

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
        """Reset this config to an empty state, then apply optional overrides."""

        return self.empty(**kwargs)


@dataclasses.dataclass
class TaylorSeerCalibratorConfig(CalibratorConfig):
    """Config for the TaylorSeer forecasting calibrator."""

    # TaylorSeers: From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers
    # link: https://arxiv.org/pdf/2503.06923

    # enable_calibrator (`bool`, *required*,  defaults to True):
    #     Whether to enable calibrator, if True. means that user want to use DBCache
    #     with specific calibrator for hidden_states (or hidden_states redisual),
    #     such as taylorseer, foca, and so on.
    enable_calibrator: bool = True
    # enable_encoder_calibrator (`bool`, *required*,  defaults to True):
    #     Whether to enable calibrator, if True. means that user want to use DBCache
    #     with specific calibrator for encoder_hidden_states (or encoder_hidden_states
    #     redisual), such as taylorseer, foca, and so on.
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
        """Return a compact tag that includes the selected Taylor order."""

        if kwargs.get("details", False):
            if self.taylorseer_order:
                return f"TaylorSeer_O({self.taylorseer_order})"
            return "NONE"

        if self.taylorseer_order:
            return f"T1O{self.taylorseer_order}"
        return "NONE"

    def to_kwargs(self) -> Dict:
        """Translate config fields into `TaylorSeerCalibrator` init kwargs."""

        kwargs = self.calibrator_kwargs.copy()
        kwargs["n_derivatives"] = self.taylorseer_order
        return kwargs


@dataclasses.dataclass
class FoCaCalibratorConfig(CalibratorConfig):
    """Config placeholder for the future FoCa calibrator backend."""

    # FoCa: Forecast then Calibrate: Feature Caching as ODE for Efficient Diffusion Transformers
    # link: https://arxiv.org/pdf/2508.16211

    # enable_calibrator (`bool`, *required*,  defaults to True):
    #     Whether to enable calibrator, if True. means that user want to use DBCache
    #     with specific calibrator for hidden_states (or hidden_states redisual),
    #     such as taylorseer, foca, and so on.
    enable_calibrator: bool = True
    # enable_encoder_calibrator (`bool`, *required*,  defaults to True):
    #     Whether to enable calibrator, if True. means that user want to use DBCache
    #     with specific calibrator for encoder_hidden_states (or encoder_hidden_states
    #     redisual), such as taylorseer, foca, and so on.
    enable_encoder_calibrator: bool = True
    # calibrator_type (`str`, *required*,  defaults to 'taylorseer'):
    #    The specific type for calibrator, taylorseer or foca, etc.
    calibrator_type: str = "foca"

    def strify(self, **kwargs) -> str:
        return "FoCa"


class Calibrator:
    """Factory that instantiates supported calibrator implementations."""

    _supported_calibrators = [
        "taylorseer",
        # TODO: FoCa
    ]

    def __new__(
        cls,
        calibrator_config: CalibratorConfig,
    ) -> CalibratorBase:
        """Construct the calibrator implementation described by `calibrator_config`."""

        assert (
            calibrator_config.calibrator_type in cls._supported_calibrators
        ), f"Calibrator {calibrator_config.calibrator_type} is not supported now!"

        if calibrator_config.calibrator_type.lower() == "taylorseer":
            return TaylorSeerCalibrator(**calibrator_config.to_kwargs())
        else:
            raise ValueError(
                f"Calibrator {calibrator_config.calibrator_type} is not supported now!"
            )
