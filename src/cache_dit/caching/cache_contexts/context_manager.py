from ..cache_types import CacheType
from .cache_manager import CachedContextManager
from .prune_manager import PrunedContextManager
from cache_dit.logger import init_logger

logger = init_logger(__name__)


class ContextManager:
    _supported_managers = (
        CachedContextManager,
        PrunedContextManager,
    )

    def __new__(
        cls,
        cache_type: CacheType,
        name: str = "default",
        persistent_context: bool = False,
    ) -> CachedContextManager | PrunedContextManager:
        if cache_type == CacheType.DBCache:
            return CachedContextManager(
                name=name,
                persistent_context=persistent_context,
            )
        elif cache_type == CacheType.DBPrune:
            return PrunedContextManager(
                name=name,
                persistent_context=persistent_context,
            )
        else:
            raise ValueError(f"Unsupported cache_type: {cache_type}.")
