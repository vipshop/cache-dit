import torch

from cache_dit.cache_factory import ForwardPattern
from cache_dit.cache_factory.cache_contexts.cache_context import CachedContext
from cache_dit.cache_factory.cache_contexts.cache_manager import (
    CachedContextManager,
)

from cache_dit.cache_factory.cache_blocks.pattern_0_1_2 import (
    CachedBlocks_Pattern_0_1_2,
)
from cache_dit.cache_factory.cache_blocks.pattern_3_4_5 import (
    CachedBlocks_Pattern_3_4_5,
)
from cache_dit.cache_factory.cache_blocks.pattern_utils import (
    patch_cached_stats,
    remove_cached_stats,
)

from cache_dit.logger import init_logger

logger = init_logger(__name__)


class CachedBlocks:
    def __new__(
        cls,
        # 0. Transformer blocks configuration
        transformer_blocks: torch.nn.ModuleList,
        transformer: torch.nn.Module = None,
        forward_pattern: ForwardPattern = None,
        check_forward_pattern: bool = True,
        check_num_outputs: bool = True,
        # 1. Cache context configuration
        # 'transformer_blocks', 'blocks', 'single_transformer_blocks',
        # 'layers', 'single_stream_blocks', 'double_stream_blocks'
        cache_prefix: str = None,  # cache_prefix maybe un-need.
        # Usually, blocks_name, etc.
        cache_context: CachedContext | str = None,
        cache_manager: CachedContextManager = None,
        **kwargs,
    ):
        assert transformer is not None, "transformer can't be None."
        assert forward_pattern is not None, "forward_pattern can't be None."
        assert cache_context is not None, "cache_context can't be None."
        assert cache_manager is not None, "cache_manager can't be None."
        if forward_pattern in CachedBlocks_Pattern_0_1_2._supported_patterns:
            return CachedBlocks_Pattern_0_1_2(
                # 0. Transformer blocks configuration
                transformer_blocks,
                transformer=transformer,
                forward_pattern=forward_pattern,
                check_forward_pattern=check_forward_pattern,
                check_num_outputs=check_num_outputs,
                # 1. Cache context configuration
                cache_prefix=cache_prefix,
                cache_context=cache_context,
                cache_manager=cache_manager,
                **kwargs,
            )
        elif forward_pattern in CachedBlocks_Pattern_3_4_5._supported_patterns:
            return CachedBlocks_Pattern_3_4_5(
                # 0. Transformer blocks configuration
                transformer_blocks,
                transformer=transformer,
                forward_pattern=forward_pattern,
                check_forward_pattern=check_forward_pattern,
                check_num_outputs=check_num_outputs,
                # 1. Cache context configuration
                cache_prefix=cache_prefix,
                cache_context=cache_context,
                cache_manager=cache_manager,
                **kwargs,
            )
        else:
            raise ValueError(f"Pattern {forward_pattern} is not supported now!")
