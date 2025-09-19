from cache_dit.cache_factory.cache_contexts.calibrators.base import (
    CalibratorBase,
)
from cache_dit.cache_factory.cache_contexts.calibrators.taylorseer import (
    TaylorSeer,
)
from cache_dit.cache_factory.cache_contexts.calibrators.foca import FoCa

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
            return TaylorSeer(**kwargs)
        else:
            raise ValueError(
                f"Calibrator {calibrator_type} is not supported now!"
            )
