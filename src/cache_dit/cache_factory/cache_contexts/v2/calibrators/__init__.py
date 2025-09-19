from cache_dit.cache_factory.cache_contexts.v2.calibrators.base import (
    CalibratorBase,
)
from cache_dit.cache_factory.cache_contexts.v2.calibrators.taylorseer import (
    TaylorSeerCalibrator,
)
from cache_dit.cache_factory.cache_contexts.v2.calibrators.foca import (
    FoCaCalibrator,
)

import dataclasses
from typing import Any, Dict


from cache_dit.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class CalibratorConfig:  # no V1
    enable_calibrator: bool = False
    enable_encoder_calibrator: bool = False
    calibrator_type: str = "taylorseer"  # taylorseer or foca, etc.
    calibrator_cache_type: str = "residual"  # residual or hidden_states
    calibrator_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def strify(self) -> str:
        return "CalibratorBase"

    def to_kwargs(self) -> Dict:
        return self.calibrator_kwargs.copy()


@dataclasses.dataclass
class TaylorSeerCalibratorConfig(CalibratorConfig):
    enable_calibrator: bool = True
    enable_encoder_calibrator: bool = True
    calibrator_type: str = "taylorseer"
    taylorseer_order: int = 1

    def strify(self) -> str:
        if self.taylorseer_order:
            return f"TaylorSeer_O({self.taylorseer_order})"
        return "TaylorSeer_O(0)"

    def to_kwargs(self) -> Dict:
        kwargs = self.calibrator_kwargs.copy()
        kwargs["n_derivatives"] = self.taylorseer_order
        return kwargs


@dataclasses.dataclass
class FoCaCalibratorConfig(CalibratorConfig):
    enable_calibrator: bool = True
    enable_encoder_calibrator: bool = True
    calibrator_type: str = "foca"

    def strify(self) -> str:
        return "FoCa"


class Calibrator:
    _supported_calibrators = [
        "taylorseer",
    ]

    def __new__(
        cls,
        calibrator_config: CalibratorConfig,
    ) -> CalibratorBase:
        assert (
            calibrator_config.calibrator_type in cls._supported_calibrators
        ), f"Calibrator {calibrator_config.calibrator_type} is not supported now!"

        if calibrator_config.calibrator_type.lower() == "taylorseer":
            return TaylorSeerCalibrator(**calibrator_config.to_kwargs())
        else:
            raise ValueError(
                f"Calibrator {calibrator_config.calibrator_type} is not supported now!"
            )
