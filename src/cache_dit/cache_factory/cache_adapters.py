import torch

import unittest
import functools

from typing import Dict
from contextlib import ExitStack
from diffusers import DiffusionPipeline
from cache_dit.cache_factory import CacheType
from cache_dit.cache_factory import CachedContext
from cache_dit.cache_factory import ForwardPattern
from cache_dit.cache_factory import BlockAdapter
from cache_dit.cache_factory import BlockAdapterRegistry
from cache_dit.cache_factory import CachedBlocks

from cache_dit.logger import init_logger

logger = init_logger(__name__)


# Unified Cached Adapter
class CachedAdapter:

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    @classmethod
    def apply(
        cls,
        pipe: DiffusionPipeline = None,
        block_adapter: BlockAdapter = None,
        # forward_pattern: ForwardPattern = ForwardPattern.Pattern_0,
        **cache_context_kwargs,
    ) -> DiffusionPipeline:
        assert (
            pipe is not None or block_adapter is not None
        ), "pipe or block_adapter can not both None!"

        if pipe is not None:
            if BlockAdapterRegistry.is_supported(pipe):
                logger.info(
                    f"{pipe.__class__.__name__} is officially supported by cache-dit. "
                    "Use it's pre-defined BlockAdapter directly!"
                )
                block_adapter = BlockAdapterRegistry.get_adapter(pipe)
                return cls.cachify(
                    block_adapter,
                    **cache_context_kwargs,
                )
            else:
                raise ValueError(
                    f"{pipe.__class__.__name__} is not officially supported "
                    "by cache-dit, please set BlockAdapter instead!"
                )
        else:
            logger.info(
                "Adapting cache acceleration using custom BlockAdapter!"
            )
            return cls.cachify(
                block_adapter,
                **cache_context_kwargs,
            )

    @classmethod
    def cachify(
        cls,
        block_adapter: BlockAdapter,
        **cache_context_kwargs,
    ) -> DiffusionPipeline:

        if block_adapter.auto:
            block_adapter = BlockAdapter.auto_block_adapter(
                block_adapter,
            )

        if BlockAdapter.check_block_adapter(block_adapter):
            block_adapter = BlockAdapter.normalize(block_adapter)
            # 0. Apply cache on pipeline: wrap cache context
            cls.create_context(
                block_adapter,
                **cache_context_kwargs,
            )
            # 1. Apply cache on transformer: mock cached transformer blocks
            cls.mock_blocks(
                block_adapter,
            )
            cls.patch_params(
                block_adapter,
                **cache_context_kwargs,
            )
        return block_adapter.pipe

    @classmethod
    def patch_params(
        cls,
        block_adapter: BlockAdapter,
        **cache_context_kwargs,
    ):
        block_adapter.transformer._forward_pattern = (
            block_adapter.forward_pattern
        )
        block_adapter.transformer._has_separate_cfg = (
            block_adapter.has_separate_cfg
        )
        block_adapter.transformer._cache_context_kwargs = cache_context_kwargs
        block_adapter.pipe.__class__._cache_context_kwargs = (
            cache_context_kwargs
        )
        for blocks, forward_pattern in zip(
            block_adapter.blocks, block_adapter.forward_pattern
        ):
            blocks._forward_pattern = forward_pattern
            blocks._cache_context_kwargs = cache_context_kwargs

    @classmethod
    def check_context_kwargs(cls, pipe, **cache_context_kwargs):
        # Check cache_context_kwargs
        if not cache_context_kwargs["do_separate_cfg"]:
            # Check cfg for some specific case if users don't set it as True
            cache_context_kwargs["do_separate_cfg"] = (
                BlockAdapterRegistry.has_separate_cfg(pipe)
            )
            logger.info(
                f"Use default 'do_separate_cfg': {cache_context_kwargs['do_separate_cfg']}, "
                f"Pipeline: {pipe.__class__.__name__}."
            )

        if cache_type := cache_context_kwargs.pop("cache_type", None):
            assert (
                cache_type == CacheType.DBCache
            ), "Custom cache setting only support for DBCache now!"

        return cache_context_kwargs

    @classmethod
    def create_context(
        cls,
        block_adapter: BlockAdapter,
        **cache_context_kwargs,
    ) -> DiffusionPipeline:
        if getattr(block_adapter.pipe, "_is_cached", False):
            return block_adapter.pipe

        # Check cache_context_kwargs
        cache_context_kwargs = cls.check_context_kwargs(
            block_adapter.pipe,
            **cache_context_kwargs,
        )
        # Apply cache on pipeline: wrap cache context
        cache_kwargs, _ = CachedContext.collect_cache_kwargs(
            default_attrs={},
            **cache_context_kwargs,
        )
        original_call = block_adapter.pipe.__class__.__call__

        @functools.wraps(original_call)
        def new_call(self, *args, **kwargs):
            with ExitStack() as stack:
                # cache context will reset for each pipe inference
                for blocks_name in block_adapter.blocks_name:
                    stack.enter_context(
                        CachedContext.cache_context(
                            CachedContext.reset_cache_context(
                                blocks_name,
                                **cache_kwargs,
                            ),
                        )
                    )
                outputs = original_call(self, *args, **kwargs)
                cls.patch_stats(block_adapter)
                return outputs

        block_adapter.pipe.__class__.__call__ = new_call
        block_adapter.pipe.__class__._is_cached = True
        return block_adapter.pipe

    @classmethod
    def patch_stats(cls, block_adapter: BlockAdapter):
        from cache_dit.cache_factory.cache_blocks.utils import (
            patch_cached_stats,
        )

        patch_cached_stats(block_adapter.transformer)
        for blocks, blocks_name in zip(
            block_adapter.blocks, block_adapter.blocks_name
        ):
            patch_cached_stats(blocks, blocks_name)

    @classmethod
    def mock_blocks(
        cls,
        block_adapter: BlockAdapter,
    ) -> torch.nn.Module:

        if getattr(block_adapter.transformer, "_is_cached", False):
            return block_adapter.transformer

        # Check block forward pattern matching
        block_adapter = BlockAdapter.normalize(block_adapter)
        for forward_pattern, blocks in zip(
            block_adapter.forward_pattern, block_adapter.blocks
        ):
            assert BlockAdapter.match_blocks_pattern(
                blocks,
                forward_pattern=forward_pattern,
                check_num_outputs=block_adapter.check_num_outputs,
            ), (
                "No block forward pattern matched, "
                f"supported lists: {ForwardPattern.supported_patterns()}"
            )

        # Apply cache on transformer: mock cached transformer blocks
        # TODO: Use blocks_name to spearate cached context for different
        # blocks list. For example, single_transformer_blocks and
        # transformer_blocks should have different cached context and
        # forward pattern.
        cached_blocks = cls.collect_cached_blocks(
            block_adapter=block_adapter,
        )
        dummy_blocks = torch.nn.ModuleList()

        original_forward = block_adapter.transformer.forward

        assert isinstance(block_adapter.dummy_blocks_names, list)

        @functools.wraps(original_forward)
        def new_forward(self, *args, **kwargs):
            with ExitStack() as stack:
                for blocks_name in block_adapter.blocks_name:
                    stack.enter_context(
                        unittest.mock.patch.object(
                            self,
                            blocks_name,
                            cached_blocks[blocks_name],
                        )
                    )
                for dummy_name in block_adapter.dummy_blocks_names:
                    stack.enter_context(
                        unittest.mock.patch.object(
                            self,
                            dummy_name,
                            dummy_blocks,
                        )
                    )
                return original_forward(*args, **kwargs)

        block_adapter.transformer.forward = new_forward.__get__(
            block_adapter.transformer
        )
        block_adapter.transformer._is_cached = True

        return block_adapter.transformer

    @classmethod
    def collect_cached_blocks(
        cls,
        block_adapter: BlockAdapter,
    ) -> Dict[str, torch.nn.ModuleList]:
        block_adapter = BlockAdapter.normalize(block_adapter)

        cached_blocks_bind_context = {}

        for i in range(len(block_adapter.blocks)):
            cached_blocks_bind_context[block_adapter.blocks_name[i]] = (
                torch.nn.ModuleList(
                    [
                        CachedBlocks(
                            block_adapter.blocks[i],
                            block_adapter.blocks_name[i],
                            block_adapter.blocks_name[i],  # context name
                            transformer=block_adapter.transformer,
                            forward_pattern=block_adapter.forward_pattern[i],
                            check_num_outputs=block_adapter.check_num_outputs,
                        )
                    ]
                )
            )

        return cached_blocks_bind_context
