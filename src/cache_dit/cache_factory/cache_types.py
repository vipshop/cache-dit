from enum import Enum
from cache_dit.logger import init_logger

logger = init_logger(__name__)


class CacheType(Enum):
    NONE = "NONE"
    DBCache = "Dual_Block_Cache"

    @staticmethod
    def type(cache_type: "CacheType | str") -> "CacheType":
        if isinstance(cache_type, CacheType):
            return cache_type
        return CacheType.cache_type(cache_type)

    @staticmethod
    def cache_type(cache_type: "CacheType | str") -> "CacheType":
        if cache_type is None:
            return CacheType.NONE

        if isinstance(cache_type, CacheType):
            return cache_type

        elif cache_type.lower() in (
            "dual_block_cache",
            "db_cache",
            "dbcache",
            "db",
        ):
            return CacheType.DBCache
        elif cache_type.lower() in (
            "none_cache",
            "nonecache",
            "no_cache",
            "nocache",
            "none",
            "no",
        ):
            return CacheType.NONE
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")

    @staticmethod
    def block_range(start: int, end: int, step: int = 1) -> list[int]:
        if start > end or end <= 0 or step <= 1:
            return []
        # Always compute 0 and end - 1 blocks for DB Cache
        return list(
            sorted(set([0] + list(range(start, end, step)) + [end - 1]))
        )

    @staticmethod
    def default_options(cache_type: "CacheType | str") -> dict:
        _no_options = {
            "cache_type": CacheType.NONE,
        }

        _Fn_compute_blocks = 8
        _Bn_compute_blocks = 0

        _db_options = {
            "cache_type": CacheType.DBCache,
            "residual_diff_threshold": 0.12,
            "warmup_steps": 8,
            "max_cached_steps": -1,  # -1 means no limit
            "Fn_compute_blocks": _Fn_compute_blocks,
            "Bn_compute_blocks": _Bn_compute_blocks,
            "max_Fn_compute_blocks": 16,
            "max_Bn_compute_blocks": 16,
            "Fn_compute_blocks_ids": [],  # 0, 1, 2, ..., 7, etc.
            "Bn_compute_blocks_ids": [],  # 0, 1, 2, ..., 7, etc.
        }

        if cache_type == CacheType.DBCache:
            return _db_options
        elif cache_type == CacheType.NONE:
            return _no_options
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")
