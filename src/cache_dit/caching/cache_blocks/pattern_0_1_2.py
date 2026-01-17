from ..forward_pattern import ForwardPattern
from .pattern_base import (
    CachedBlocks_Pattern_Base,
    PrunedBlocks_Pattern_Base,
)
from cache_dit.logger import init_logger

logger = init_logger(__name__)


class CachedBlocks_Pattern_0_1_2(CachedBlocks_Pattern_Base):
    _supported_patterns = [
        ForwardPattern.Pattern_0,
        ForwardPattern.Pattern_1,
        ForwardPattern.Pattern_2,
    ]
    ...


class PrunedBlocks_Pattern_0_1_2(PrunedBlocks_Pattern_Base):
    _supported_patterns = [
        ForwardPattern.Pattern_0,
        ForwardPattern.Pattern_1,
        ForwardPattern.Pattern_2,
    ]
    ...
