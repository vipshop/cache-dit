import torch

import unittest
import functools

from contextlib import ExitStack
from typing import Dict, List, Tuple, Any

from diffusers import DiffusionPipeline

from cache_dit.cache_factory import CacheType
from cache_dit.cache_factory import BlockAdapter
from cache_dit.cache_factory import ParamsModifier
from cache_dit.cache_factory import BlockAdapterRegistry
from cache_dit.cache_factory import CachedContextManager
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
            # 0. Must normalize block_adapter before apply cache
            block_adapter = BlockAdapter.normalize(block_adapter)
            # 1. Apply cache on pipeline: wrap cache context, must
            # call create_context before mock_blocks.
            cls.create_context(
                block_adapter,
                **cache_context_kwargs,
            )
            # 2. Apply cache on transformer: mock cached blocks
            cls.mock_blocks(
                block_adapter,
            )
        return block_adapter.pipe

    @classmethod
    def patch_params(
        cls,
        block_adapter: BlockAdapter,
        contexts_kwargs: List[Dict],
    ):
        block_adapter.pipe._cache_context_kwargs = contexts_kwargs[0]

        params_shift = 0
        for i in range(len(block_adapter.transformer)):

            block_adapter.transformer[i]._forward_pattern = (
                block_adapter.forward_pattern
            )
            block_adapter.transformer[i]._has_separate_cfg = (
                block_adapter.has_separate_cfg
            )
            block_adapter.transformer[i]._cache_context_kwargs = (
                contexts_kwargs[params_shift]
            )

            blocks = block_adapter.blocks[i]
            for j in range(len(blocks)):
                blocks[j]._forward_pattern = block_adapter.forward_pattern[i][j]
                blocks[j]._cache_context_kwargs = contexts_kwargs[
                    params_shift + j
                ]

            params_shift += len(blocks)

    @classmethod
    def check_context_kwargs(cls, pipe, **cache_context_kwargs):
        # Check cache_context_kwargs
        if not cache_context_kwargs["enable_spearate_cfg"]:
            # Check cfg for some specific case if users don't set it as True
            cache_context_kwargs["enable_spearate_cfg"] = (
                BlockAdapterRegistry.has_separate_cfg(pipe)
            )
            logger.info(
                f"Use default 'enable_spearate_cfg': "
                f"{cache_context_kwargs['enable_spearate_cfg']}, "
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

        BlockAdapter.assert_normalized(block_adapter)

        if BlockAdapter.is_cached(block_adapter):
            return block_adapter.pipe

        # Check cache_context_kwargs
        cache_context_kwargs = cls.check_context_kwargs(
            block_adapter.pipe,
            **cache_context_kwargs,
        )
        # Apply cache on pipeline: wrap cache context
        pipe_cls_name = block_adapter.pipe.__class__.__name__

        # Each Pipeline should have it's own context manager instance.
        # Different transformers (Wan2.2, etc) should shared the same
        # cache manager but with different cache context (according
        # to their unique instance id).
        cache_manager = CachedContextManager(
            name=f"{pipe_cls_name}_{hash(id(block_adapter.pipe))}",
        )
        block_adapter.pipe._cache_manager = cache_manager  # instance level

        flatten_contexts, contexts_kwargs = cls.modify_context_params(
            block_adapter, cache_manager, **cache_context_kwargs
        )

        original_call = block_adapter.pipe.__class__.__call__

        @functools.wraps(original_call)
        def new_call(self, *args, **kwargs):
            with ExitStack() as stack:
                # cache context will be reset for each pipe inference
                for context_name, context_kwargs in zip(
                    flatten_contexts, contexts_kwargs
                ):
                    stack.enter_context(
                        cache_manager.enter_context(
                            cache_manager.reset_context(
                                context_name,
                                **context_kwargs,
                            ),
                        )
                    )
                outputs = original_call(self, *args, **kwargs)
                cls.patch_stats(block_adapter)
                return outputs

        block_adapter.pipe.__class__.__call__ = new_call
        block_adapter.pipe.__class__._original_call = original_call
        block_adapter.pipe.__class__._is_cached = True

        cls.patch_params(block_adapter, contexts_kwargs)

        return block_adapter.pipe

    @classmethod
    def modify_context_params(
        cls,
        block_adapter: BlockAdapter,
        cache_manager: CachedContextManager,
        **cache_context_kwargs,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:

        flatten_contexts = BlockAdapter.flatten(
            block_adapter.unique_blocks_name
        )
        contexts_kwargs = [
            cache_context_kwargs.copy()
            for _ in range(
                len(flatten_contexts),
            )
        ]

        for i in range(len(contexts_kwargs)):
            contexts_kwargs[i]["name"] = flatten_contexts[i]

        if block_adapter.params_modifiers is None:
            return flatten_contexts, contexts_kwargs

        flatten_modifiers: List[ParamsModifier] = BlockAdapter.flatten(
            block_adapter.params_modifiers,
        )

        for i in range(
            min(len(contexts_kwargs), len(flatten_modifiers)),
        ):
            contexts_kwargs[i].update(
                flatten_modifiers[i]._context_kwargs,
            )
            contexts_kwargs[i], _ = cache_manager.collect_cache_kwargs(
                default_attrs={}, **contexts_kwargs[i]
            )

        return flatten_contexts, contexts_kwargs

    @classmethod
    def patch_stats(
        cls,
        block_adapter: BlockAdapter,
    ):
        from cache_dit.cache_factory.cache_blocks.utils import (
            patch_cached_stats,
        )

        cache_manager = block_adapter.pipe._cache_manager

        for i in range(len(block_adapter.transformer)):
            patch_cached_stats(
                block_adapter.transformer[i],
                cache_context=block_adapter.unique_blocks_name[i][-1],
                cache_manager=cache_manager,
            )
            for blocks, unique_name in zip(
                block_adapter.blocks[i],
                block_adapter.unique_blocks_name[i],
            ):
                patch_cached_stats(
                    blocks,
                    cache_context=unique_name,
                    cache_manager=cache_manager,
                )

    @classmethod
    def mock_blocks(
        cls,
        block_adapter: BlockAdapter,
    ) -> List[torch.nn.Module]:

        BlockAdapter.assert_normalized(block_adapter)

        if BlockAdapter.is_cached(block_adapter):
            return block_adapter.transformer

        # Apply cache on transformer: mock cached transformer blocks
        for (
            cached_blocks,
            transformer,
            blocks_name,
            unique_blocks_name,
            dummy_blocks_names,
        ) in zip(
            cls.collect_cached_blocks(block_adapter),
            block_adapter.transformer,
            block_adapter.blocks_name,
            block_adapter.unique_blocks_name,
            block_adapter.dummy_blocks_names,
        ):
            cls.mock_transformer(
                cached_blocks,
                transformer,
                blocks_name,
                unique_blocks_name,
                dummy_blocks_names,
            )

        return block_adapter.transformer

    @classmethod
    def mock_transformer(
        cls,
        cached_blocks: Dict[str, torch.nn.ModuleList],
        transformer: torch.nn.Module,
        blocks_name: List[str],
        unique_blocks_name: List[str],
        dummy_blocks_names: List[str],
    ) -> torch.nn.Module:
        dummy_blocks = torch.nn.ModuleList()

        original_forward = transformer.forward

        assert isinstance(dummy_blocks_names, list)

        @functools.wraps(original_forward)
        def new_forward(self, *args, **kwargs):
            with ExitStack() as stack:
                for name, context_name in zip(
                    blocks_name,
                    unique_blocks_name,
                ):
                    stack.enter_context(
                        unittest.mock.patch.object(
                            self, name, cached_blocks[context_name]
                        )
                    )
                for dummy_name in dummy_blocks_names:
                    stack.enter_context(
                        unittest.mock.patch.object(
                            self, dummy_name, dummy_blocks
                        )
                    )
                return original_forward(*args, **kwargs)

        transformer.forward = new_forward.__get__(transformer)
        transformer._original_forward = original_forward
        transformer._is_cached = True

        return transformer

    @classmethod
    def collect_cached_blocks(
        cls,
        block_adapter: BlockAdapter,
    ) -> List[Dict[str, torch.nn.ModuleList]]:

        BlockAdapter.assert_normalized(block_adapter)

        total_cached_blocks: List[Dict[str, torch.nn.ModuleList]] = []
        assert hasattr(block_adapter.pipe, "_cache_manager")
        assert isinstance(
            block_adapter.pipe._cache_manager, CachedContextManager
        )

        for i in range(len(block_adapter.transformer)):

            cached_blocks_bind_context = {}
            for j in range(len(block_adapter.blocks[i])):
                cached_blocks_bind_context[
                    block_adapter.unique_blocks_name[i][j]
                ] = torch.nn.ModuleList(
                    [
                        CachedBlocks(
                            # 0. Transformer blocks configuration
                            block_adapter.blocks[i][j],
                            transformer=block_adapter.transformer[i],
                            forward_pattern=block_adapter.forward_pattern[i][j],
                            check_num_outputs=block_adapter.check_num_outputs,
                            # 1. Cache context configuration
                            cache_prefix=block_adapter.blocks_name[i][j],
                            cache_context=block_adapter.unique_blocks_name[i][
                                j
                            ],
                            cache_manager=block_adapter.pipe._cache_manager,
                        )
                    ]
                )

            total_cached_blocks.append(cached_blocks_bind_context)

        return total_cached_blocks
