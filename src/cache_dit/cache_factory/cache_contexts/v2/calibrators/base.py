from abc import abstractmethod

from cache_dit.logger import init_logger

logger = init_logger(__name__)


class CalibratorBase:

    @abstractmethod
    def reset_cache(self, *args, **kwargs):
        raise NotImplementedError("reset_cache method is not implemented.")

    @abstractmethod
    def approximate(self, *args, **kwargs):
        raise NotImplementedError("approximate method is not implemented.")

    @abstractmethod
    def mark_step_begin(self, *args, **kwargs):
        raise NotImplementedError("mark_step_begin method is not implemented.")

    @abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError("update method is not implemented.")

    def __repr__(self):
        return "CalibratorBase"
