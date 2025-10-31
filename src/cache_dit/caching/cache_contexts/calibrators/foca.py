from cache_dit.caching.cache_contexts.calibrators.base import (
    CalibratorBase,
)


class FoCaCalibrator(CalibratorBase):
    # TODO: Support FoCa, Forecast then Calibrate: Feature Caching as ODE for
    # Efficient Diffusion Transformers, https://arxiv.org/pdf/2508.16211

    def __init__(self, *args, **kwargs):
        super().__init__()

    def reset_cache(self, *args, **kwargs):
        raise NotImplementedError("reset_cache method is not implemented.")

    def approximate(self, *args, **kwargs):
        raise NotImplementedError("approximate method is not implemented.")

    def mark_step_begin(self, *args, **kwargs):
        raise NotImplementedError("mark_step_begin method is not implemented.")

    def update(self, *args, **kwargs):
        raise NotImplementedError("update method is not implemented.")

    def __repr__(self):
        return "FoCaCalibrator"
