try:
    from ._version import version as __version__, version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")


from cache_dit.utils import disable_print
from cache_dit.logger import init_logger
from cache_dit.caching import load_options
from cache_dit.caching import enable_cache
from cache_dit.caching import disable_cache
from cache_dit.caching import cache_type
from cache_dit.caching import block_range
from cache_dit.caching import CacheType
from cache_dit.caching import BlockAdapter
from cache_dit.caching import ParamsModifier
from cache_dit.caching import ForwardPattern
from cache_dit.caching import PatchFunctor
from cache_dit.caching import BasicCacheConfig
from cache_dit.caching import DBCacheConfig
from cache_dit.caching import DBPruneConfig
from cache_dit.caching import CalibratorConfig
from cache_dit.caching import TaylorSeerCalibratorConfig
from cache_dit.caching import FoCaCalibratorConfig
from cache_dit.caching import supported_pipelines
from cache_dit.caching import get_adapter
from cache_dit.compile import set_compile_configs
from cache_dit.parallelism import ParallelismBackend
from cache_dit.parallelism import ParallelismConfig
from cache_dit.summary import supported_matrix
from cache_dit.summary import summary
from cache_dit.summary import strify

try:
    from cache_dit.quantize import quantize
except ImportError as e:  # noqa: F841
    err_msg = str(e)

    def quantize(*args, **kwargs):
        raise ImportError(
            "Quantization requires additional dependencies. "
            "Please install cache-dit[quantization] or cache-dit[all] "
            f"to use this feature. Error message: {err_msg}"
        )


NONE = CacheType.NONE
DBCache = CacheType.DBCache
DBPrune = CacheType.DBPrune

Pattern_0 = ForwardPattern.Pattern_0
Pattern_1 = ForwardPattern.Pattern_1
Pattern_2 = ForwardPattern.Pattern_2
Pattern_3 = ForwardPattern.Pattern_3
Pattern_4 = ForwardPattern.Pattern_4
Pattern_5 = ForwardPattern.Pattern_5
