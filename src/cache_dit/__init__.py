try:
    from ._version import version as __version__, version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")

from cache_dit.utils import summary
from cache_dit.utils import strify
from cache_dit.utils import disable_print
from cache_dit.logger import init_logger
from cache_dit.cache_factory import load_options
from cache_dit.cache_factory import enable_cache
from cache_dit.cache_factory import disable_cache
from cache_dit.cache_factory import cache_type
from cache_dit.cache_factory import block_range
from cache_dit.cache_factory import CacheType
from cache_dit.cache_factory import BlockAdapter
from cache_dit.cache_factory import ParamsModifier
from cache_dit.cache_factory import ForwardPattern
from cache_dit.cache_factory import PatchFunctor
from cache_dit.cache_factory import BasicCacheConfig
from cache_dit.cache_factory import CalibratorConfig
from cache_dit.cache_factory import TaylorSeerCalibratorConfig
from cache_dit.cache_factory import FoCaCalibratorConfig
from cache_dit.cache_factory import supported_pipelines
from cache_dit.cache_factory import get_adapter
from cache_dit.compile import set_compile_configs
from cache_dit.quantize import quantize


NONE = CacheType.NONE
DBCache = CacheType.DBCache

Pattern_0 = ForwardPattern.Pattern_0
Pattern_1 = ForwardPattern.Pattern_1
Pattern_2 = ForwardPattern.Pattern_2
Pattern_3 = ForwardPattern.Pattern_3
Pattern_4 = ForwardPattern.Pattern_4
Pattern_5 = ForwardPattern.Pattern_5
