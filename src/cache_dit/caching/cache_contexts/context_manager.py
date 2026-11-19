from ..cache_types import CacheType
from .cache_manager import CachedContextManager
from ...logger import init_logger

logger = init_logger(__name__)


class ContextManager:
  _supported_managers = (CachedContextManager, )

  def __new__(
    cls,
    cache_type: CacheType,
    name: str = "default",
    persistent_context: bool = False,
  ) -> CachedContextManager:
    if cache_type == CacheType.DBCache:
      return CachedContextManager(
        name=name,
        persistent_context=persistent_context,
      )
    else:
      raise ValueError(f"Unsupported cache_type: {cache_type}.")
