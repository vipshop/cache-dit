from enum import Enum

from diffusers import DiffusionPipeline

from cache_dit.cache_factory.dual_block_cache.diffusers_adapters import (
    apply_db_cache_on_pipe,
)
from cache_dit.cache_factory.first_block_cache.diffusers_adapters import (
    apply_fb_cache_on_pipe,
)
from cache_dit.cache_factory.dynamic_block_prune.diffusers_adapters import (
    apply_db_prune_on_pipe,
)
from cache_dit.logger import init_logger


logger = init_logger(__name__)


class CacheType(Enum):
    NONE = "NONE"
    FBCache = "First_Block_Cache"
    DBCache = "Dual_Block_Cache"
    DBPrune = "Dynamic_Block_Prune"

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
        if cache_type.lower() in (
            "first_block_cache",
            "fb_cache",
            "fbcache",
            "fb",
        ):
            return CacheType.FBCache
        elif cache_type.lower() in (
            "dual_block_cache",
            "db_cache",
            "dbcache",
            "db",
        ):
            return CacheType.DBCache
        elif cache_type.lower() in (
            "dynamic_block_prune",
            "db_prune",
            "dbprune",
            "dbp",
        ):
            return CacheType.DBPrune
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
    def range(start: int, end: int, step: int = 1) -> list[int]:
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

        _fb_options = {
            "cache_type": CacheType.FBCache,
            "residual_diff_threshold": 0.08,
            "warmup_steps": 8,
            "max_cached_steps": 8,
        }

        _Fn_compute_blocks = 8
        _Bn_compute_blocks = 8

        _db_options = {
            "cache_type": CacheType.DBCache,
            "residual_diff_threshold": 0.12,
            "warmup_steps": 8,
            "max_cached_steps": -1,  # -1 means no limit
            # Fn=1, Bn=0, means FB Cache, otherwise, Dual Block Cache
            "Fn_compute_blocks": _Fn_compute_blocks,
            "Bn_compute_blocks": _Bn_compute_blocks,
            "max_Fn_compute_blocks": 16,
            "max_Bn_compute_blocks": 16,
            "Fn_compute_blocks_ids": [],  # 0, 1, 2, ..., 7, etc.
            "Bn_compute_blocks_ids": [],  # 0, 1, 2, ..., 7, etc.
        }

        _dbp_options = {
            "cache_type": CacheType.DBPrune,
            "residual_diff_threshold": 0.08,
            "Fn_compute_blocks": _Fn_compute_blocks,
            "Bn_compute_blocks": _Bn_compute_blocks,
            "warmup_steps": 8,
            "max_pruned_steps": -1,  # -1 means no limit
        }

        if cache_type == CacheType.FBCache:
            return _fb_options
        elif cache_type == CacheType.DBCache:
            return _db_options
        elif cache_type == CacheType.DBPrune:
            return _dbp_options
        elif cache_type == CacheType.NONE:
            return _no_options
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")


def apply_cache_on_pipe(pipe: DiffusionPipeline, *args, **kwargs):
    assert isinstance(pipe, DiffusionPipeline)

    if hasattr(pipe, "_is_cached") and pipe._is_cached:
        return pipe

    if hasattr(pipe, "_is_pruned") and pipe._is_pruned:
        return pipe

    cache_type = kwargs.pop("cache_type", None)
    if cache_type is None:
        logger.warning(
            "No cache type specified, we will use DBCache by default. "
            "Please specify the cache_type explicitly if you want to "
            "use a different cache type."
        )
        # Force to use DBCache with default cache options
        return apply_db_cache_on_pipe(
            pipe,
            **CacheType.default_options(CacheType.DBCache),
        )

    cache_type = CacheType.type(cache_type)

    if cache_type == CacheType.FBCache:
        return apply_fb_cache_on_pipe(pipe, *args, **kwargs)
    elif cache_type == CacheType.DBCache:
        return apply_db_cache_on_pipe(pipe, *args, **kwargs)
    elif cache_type == CacheType.DBPrune:
        return apply_db_prune_on_pipe(pipe, *args, **kwargs)
    elif cache_type == CacheType.NONE:
        logger.warning(
            f"Cache type is {cache_type}, no caching will be applied."
        )
        return pipe
    else:
        raise ValueError(f"Unknown cache type: {cache_type}")
