from abc import abstractmethod

from ....logger import init_logger

logger = init_logger(__name__)


class CalibratorBase:
  """Abstract interface implemented by all cache calibrator backends."""

  @abstractmethod
  def reset_cache(self, *args, **kwargs):
    """Clear cached state between inference sessions or forced refreshes.

    :param args: Additional positional arguments forwarded to the underlying implementation.
    :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
    """

    raise NotImplementedError("reset_cache method is not implemented.")

  @abstractmethod
  def approximate(self, *args, **kwargs):
    """Return an approximated tensor for the current step without full compute.

    :param args: Additional positional arguments forwarded to the underlying implementation.
    :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
    """

    raise NotImplementedError("approximate method is not implemented.")

  @abstractmethod
  def mark_step_begin(self, *args, **kwargs):
    """Advance internal step counters before one transformer execution begins.

    :param args: Additional positional arguments forwarded to the underlying implementation.
    :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
    """

    raise NotImplementedError("mark_step_begin method is not implemented.")

  @abstractmethod
  def update(self, *args, **kwargs):
    """Ingest a fully computed tensor so future approximations stay accurate.

    :param args: Additional positional arguments forwarded to the underlying implementation.
    :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
    """

    raise NotImplementedError("update method is not implemented.")

  def __repr__(self):
    return "CalibratorBase"
