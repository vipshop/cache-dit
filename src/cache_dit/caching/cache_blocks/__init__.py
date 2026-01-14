import torch

from ..forward_pattern import ForwardPattern
from ..cache_types import CacheType
from ..cache_contexts.cache_context import CachedContext
from ..cache_contexts.prune_context import PrunedContext
from ..cache_contexts.cache_manager import (
    CachedContextManager,
)
from ..cache_contexts.prune_manager import (
    PrunedContextManager,
)

from .pattern_0_1_2 import (
    CachedBlocks_Pattern_0_1_2,
    PrunedBlocks_Pattern_0_1_2,
)
from .pattern_3_4_5 import (
    CachedBlocks_Pattern_3_4_5,
    PrunedBlocks_Pattern_3_4_5,
)
from .pattern_utils import apply_stats
from .pattern_utils import remove_stats

from cache_dit.logger import init_logger

logger = init_logger(__name__)


class UnifiedBlocks:
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
        cache_context: CachedContext | PrunedContext | str = None,
        context_manager: CachedContextManager | PrunedContextManager = None,
        cache_type: CacheType = CacheType.DBCache,
        **kwargs,
    ):
        if cache_type == CacheType.DBCache:
            return CachedBlocks(
                # 0. Transformer blocks configuration
                transformer_blocks,
                transformer=transformer,
                forward_pattern=forward_pattern,
                check_forward_pattern=check_forward_pattern,
                check_num_outputs=check_num_outputs,
                # 1. Cache context configuration
                cache_prefix=cache_prefix,
                cache_context=cache_context,
                context_manager=context_manager,
                cache_type=cache_type,
                **kwargs,
            )
        elif cache_type == CacheType.DBPrune:
            return PrunedBlocks(
                # 0. Transformer blocks configuration
                transformer_blocks,
                transformer=transformer,
                forward_pattern=forward_pattern,
                check_forward_pattern=check_forward_pattern,
                check_num_outputs=check_num_outputs,
                # 1. Cache context configuration
                cache_prefix=cache_prefix,
                cache_context=cache_context,
                context_manager=context_manager,
                cache_type=cache_type,
                **kwargs,
            )
        else:
            raise ValueError(f"Cache type {cache_type} is not supported now!")


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
        cache_context: CachedContext | PrunedContext | str = None,
        context_manager: CachedContextManager | PrunedContextManager = None,
        cache_type: CacheType = CacheType.DBCache,
        **kwargs,
    ):
        assert transformer is not None, "transformer can't be None."
        assert forward_pattern is not None, "forward_pattern can't be None."
        assert cache_context is not None, "cache_context can't be None."
        assert context_manager is not None, "context_manager can't be None."
        if forward_pattern in CachedBlocks_Pattern_0_1_2._supported_patterns:
            if cache_type == CacheType.DBCache:
                assert isinstance(
                    context_manager, CachedContextManager
                ), "context_manager must be CachedContextManager for DBCache."
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
                    context_manager=context_manager,
                    cache_type=cache_type,
                    **kwargs,
                )
            else:
                raise ValueError(f"Cache type {cache_type} is not supported now!")
        elif forward_pattern in CachedBlocks_Pattern_3_4_5._supported_patterns:
            if cache_type == CacheType.DBCache:
                assert isinstance(
                    context_manager, CachedContextManager
                ), "context_manager must be CachedContextManager for DBCache."
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
                    context_manager=context_manager,
                    cache_type=cache_type,
                    **kwargs,
                )
            else:
                raise ValueError(f"Cache type {cache_type} is not supported now!")
        else:
            raise ValueError(f"Pattern {forward_pattern} is not supported now!")


class PrunedBlocks:
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
        cache_context: CachedContext | PrunedContext | str = None,
        context_manager: CachedContextManager | PrunedContextManager = None,
        cache_type: CacheType = CacheType.DBCache,
        **kwargs,
    ):
        assert transformer is not None, "transformer can't be None."
        assert forward_pattern is not None, "forward_pattern can't be None."
        assert cache_context is not None, "cache_context can't be None."
        assert context_manager is not None, "context_manager can't be None."
        if forward_pattern in PrunedBlocks_Pattern_0_1_2._supported_patterns:
            if cache_type == CacheType.DBPrune:
                assert isinstance(
                    context_manager, PrunedContextManager
                ), "context_manager must be PrunedContextManager for DBPrune."
                return PrunedBlocks_Pattern_0_1_2(
                    # 0. Transformer blocks configuration
                    transformer_blocks,
                    transformer=transformer,
                    forward_pattern=forward_pattern,
                    check_forward_pattern=check_forward_pattern,
                    check_num_outputs=check_num_outputs,
                    # 1. Cache context configuration
                    cache_prefix=cache_prefix,
                    cache_context=cache_context,
                    context_manager=context_manager,
                    cache_type=cache_type,
                    **kwargs,
                )
            else:
                raise ValueError(f"Cache type {cache_type} is not supported now!")
        elif forward_pattern in PrunedBlocks_Pattern_3_4_5._supported_patterns:
            if cache_type == CacheType.DBPrune:
                assert isinstance(
                    context_manager, PrunedContextManager
                ), "context_manager must be PrunedContextManager for DBPrune."
                return PrunedBlocks_Pattern_3_4_5(
                    # 0. Transformer blocks configuration
                    transformer_blocks,
                    transformer=transformer,
                    forward_pattern=forward_pattern,
                    check_forward_pattern=check_forward_pattern,
                    check_num_outputs=check_num_outputs,
                    # 1. Cache context configuration
                    cache_prefix=cache_prefix,
                    cache_context=cache_context,
                    context_manager=context_manager,
                    cache_type=cache_type,
                    **kwargs,
                )
            else:
                raise ValueError(f"Cache type {cache_type} is not supported now!")
        else:
            raise ValueError(f"Pattern {forward_pattern} is not supported now!")
