from enum import Enum
from cache_dit.logger import init_logger

logger = init_logger(__name__)


class CacheType(Enum):
    NONE = "NONE"
    DBCache = "DBCache"  # "Dual_Block_Cache"
    DBPrune = "DBPrune"  # "Dynamic_Block_Prune"

    @staticmethod
    def type(type_hint: "CacheType | str") -> "CacheType":
        if isinstance(type_hint, CacheType):
            return type_hint
        return cache_type(type_hint)

    def __str__(self) -> str:
        return self.value


def cache_type(type_hint: "CacheType | str") -> "CacheType":
    if type_hint is None:
        return CacheType.NONE

    if isinstance(type_hint, CacheType):
        return type_hint
    elif type_hint.upper() in (
        "DUAL_BLOCK_CACHE",
        "DB_CACHE",
        "DBCACHE",
        "DB",
    ):
        return CacheType.DBCache
    elif type_hint.upper() in (
        "DYNAMIC_BLOCK_PRUNE",
        "DB_PRUNE",
        "DBPRUNE",
        "DBP",
    ):
        return CacheType.DBPrune
    elif type_hint.upper() in (
        "NONE",
        "NO_CACHE",
        "NOCACHE",
        "NC",
    ):
        return CacheType.NONE
    return CacheType.NONE


def block_range(start: int, end: int, step: int = 1) -> list[int]:
    if start > end or end <= 0 or step <= 1:
        return []
    # Always compute 0 and end - 1 blocks for DB Cache
    return list(sorted(set([0] + list(range(start, end, step)) + [end - 1])))
