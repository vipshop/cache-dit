from cache_dit.cache_factory.cache_contexts.v2.calibrators.base import (
    CalibratorBase,
)
from cache_dit.cache_factory.cache_contexts.v2.calibrators.taylorseer import (
    TaylorSeerCalibrator,
)
from cache_dit.cache_factory.cache_contexts.v2.calibrators.foca import (
    FoCaCalibrator,
)

from cache_dit.logger import init_logger

logger = init_logger(__name__)


class Calibrator:
    _supported_calibrators = [
        "taylorseer",
    ]

    def __new__(
        cls,
        calibrator_type: str = "taylorseer",
        **kwargs,
    ) -> CalibratorBase:
        assert (
            calibrator_type in cls._supported_calibrators
        ), f"Calibrator {calibrator_type} is not supported now!"

        if calibrator_type.lower() == "taylorseer":
            return TaylorSeerCalibrator(**kwargs)
        else:
            raise ValueError(
                f"Calibrator {calibrator_type} is not supported now!"
            )
