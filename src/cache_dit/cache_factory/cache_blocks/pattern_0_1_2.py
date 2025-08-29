from cache_dit.cache_factory import ForwardPattern
from cache_dit.cache_factory.cache_blocks.pattern_base import (
    DBCachedBlocks_Pattern_Base,
)
from cache_dit.logger import init_logger

logger = init_logger(__name__)


class DBCachedBlocks_Pattern_0_1_2(DBCachedBlocks_Pattern_Base):
    _supported_patterns = [
        ForwardPattern.Pattern_0,
        ForwardPattern.Pattern_1,
        ForwardPattern.Pattern_2,
    ]
    ...
