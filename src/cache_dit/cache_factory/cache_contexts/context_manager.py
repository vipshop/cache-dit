from cache_dit.cache_factory.cache_types import CacheType
from cache_dit.cache_factory.cache_contexts.cache_manager import (
    CachedContextManager,
)
from cache_dit.cache_factory.cache_contexts.prune_manager import (
    PrunedContextManager,
)
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
    ) -> CachedContextManager | PrunedContextManager:
        if cache_type == CacheType.DBCache:
            return CachedContextManager(name)
        elif cache_type == CacheType.DBPrune:
            return PrunedContextManager(name)
        else:
            raise ValueError(f"Unsupported cache_type: {cache_type}.")
