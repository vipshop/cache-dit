try:
    from ._version import version as __version__, version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")

from cache_dit.cache_factory import load_options
from cache_dit.cache_factory import enable_cache
from cache_dit.cache_factory import cache_type
from cache_dit.cache_factory import default_options
from cache_dit.cache_factory import block_range
from cache_dit.cache_factory import CacheType
from cache_dit.cache_factory import ForwardPattern
from cache_dit.cache_factory import BlockAdapterParams
from cache_dit.compile import set_compile_configs
from cache_dit.utils import summary
from cache_dit.logger import init_logger

NONE = CacheType.NONE
DBCache = CacheType.DBCache

BlockAdapter = BlockAdapterParams

Forward_Pattern_0 = ForwardPattern.Pattern_0
Forward_Pattern_1 = ForwardPattern.Pattern_1
Forward_Pattern_2 = ForwardPattern.Pattern_2
Forward_Pattern_3 = ForwardPattern.Pattern_3
