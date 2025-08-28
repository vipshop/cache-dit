from cache_dit.cache_factory.cache_blocks.pattern_0_1_2 import (
    DBCachedTransformerBlocks_Pattern_0_1_2,
)
from cache_dit.cache_factory.cache_blocks.pattern_3 import (
    DBCachedTransformerBlocks_Pattern_3,
)


class DBCachedTransformerBlocks:
    def __new__(cls, *args, **kwargs):
        forward_pattern = kwargs.get("forward_pattern", None)
        assert forward_pattern is not None, "forward_pattern can't be None."
        if (
            forward_pattern
            in DBCachedTransformerBlocks_Pattern_0_1_2._supported_patterns
        ):
            return DBCachedTransformerBlocks_Pattern_0_1_2(*args, **kwargs)
        elif (
            forward_pattern
            in DBCachedTransformerBlocks_Pattern_3._supported_patterns
        ):
            return DBCachedTransformerBlocks_Pattern_3(*args, **kwargs)
        else:
            raise ValueError(f"Pattern {forward_pattern} is not supported now!")
