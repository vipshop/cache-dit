from enum import Enum
from cache_dit.logger import init_logger

logger = init_logger(__name__)


class CacheType(Enum):
    NONE = "NONE"
    DBCache = "Dual_Block_Cache"

    @staticmethod
    def type(type_hint: "CacheType | str") -> "CacheType":
        if isinstance(type_hint, CacheType):
            return type_hint
        return cache_type(type_hint)


def cache_type(type_hint: "CacheType | str") -> "CacheType":
    if type_hint is None:
        return CacheType.NONE

    if isinstance(type_hint, CacheType):
        return type_hint

    elif type_hint.lower() in (
        "dual_block_cache",
        "db_cache",
        "dbcache",
        "db",
    ):
        return CacheType.DBCache
    return CacheType.NONE


def block_range(start: int, end: int, step: int = 1) -> list[int]:
    if start > end or end <= 0 or step <= 1:
        return []
    # Always compute 0 and end - 1 blocks for DB Cache
    return list(sorted(set([0] + list(range(start, end, step)) + [end - 1])))


def default_options(cache_type: "CacheType | str") -> dict:
    _NONE_OPTIONS = {
        "cache_type": CacheType.NONE,
    }

    _DBCACHE_OPTIONS = {
        "cache_type": CacheType.DBCache,
        "residual_diff_threshold": 0.12,
        "warmup_steps": 8,
        "max_cached_steps": -1,  # -1 means no limit
        "Fn_compute_blocks": 8,
        "Bn_compute_blocks": 0,
        "max_Fn_compute_blocks": 16,
        "max_Bn_compute_blocks": 16,
        "Fn_compute_blocks_ids": [],  # 0, 1, 2, ..., 7, etc.
        "Bn_compute_blocks_ids": [],  # 0, 1, 2, ..., 7, etc.
    }

    if cache_type == CacheType.DBCache:
        return _DBCACHE_OPTIONS
    elif cache_type == CacheType.NONE:
        return _NONE_OPTIONS
    else:
        raise ValueError(f"Unknown cache type: {cache_type}")
