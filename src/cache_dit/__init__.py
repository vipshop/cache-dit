try:
    from ._version import version as __version__, version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")


from .utils import disable_print
from .logger import init_logger
from .caching import load_options  # deprecated
from .caching import load_cache_config
from .caching import load_parallelism_config
from .caching import load_configs
from .caching import enable_cache
from .caching import refresh_context
from .caching import steps_mask
from .caching import disable_cache
from .caching import set_attn_backend
from .caching import cache_type
from .caching import block_range
from .caching import CacheType
from .caching import BlockAdapter
from .caching import ParamsModifier
from .caching import ForwardPattern
from .caching import PatchFunctor
from .caching import BasicCacheConfig
from .caching import DBCacheConfig
from .caching import DBPruneConfig
from .caching import CalibratorConfig
from .caching import TaylorSeerCalibratorConfig
from .caching import FoCaCalibratorConfig
from .caching import supported_pipelines
from .caching import get_adapter
from .parallelism import ParallelismBackend
from .parallelism import ParallelismConfig
from .compile import set_compile_configs
from .summary import supported_matrix
from .summary import summary
from .summary import strify
from .profiler import ProfilerContext
from .profiler import profile_function
from .profiler import create_profiler_context
from .profiler import get_profiler_output_dir
from .profiler import set_profiler_output_dir
from .quantize import quantize

NONE = CacheType.NONE
DBCache = CacheType.DBCache
DBPrune = CacheType.DBPrune

Pattern_0 = ForwardPattern.Pattern_0
Pattern_1 = ForwardPattern.Pattern_1
Pattern_2 = ForwardPattern.Pattern_2
Pattern_3 = ForwardPattern.Pattern_3
Pattern_4 = ForwardPattern.Pattern_4
Pattern_5 = ForwardPattern.Pattern_5
