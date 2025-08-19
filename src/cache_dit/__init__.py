try:
    from ._version import version as __version__, version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")

from typing import Dict, List
from cache_dit.cache_factory import load_options
from cache_dit.cache_factory import enable_cache
from cache_dit.cache_factory import supported_patterns
from cache_dit.cache_factory import match_pattern
from cache_dit.cache_factory import CacheType
from cache_dit.compile import set_compile_configs
from cache_dit.logger import init_logger

NONE = CacheType.NONE
DBCache = CacheType.DBCache


def cache_type(type_hint: "CacheType | str") -> CacheType:
    return CacheType.type(cache_type=type_hint)


def default_options(cache_type: CacheType) -> Dict:
    return CacheType.default_options(cache_type)


def block_range(start: int, end: int, step: int = 1) -> List[int]:
    return CacheType.range(start, end, step)
