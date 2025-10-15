import copy
import torch
import unittest
import functools
from contextlib import ExitStack
from typing import Dict, List, Tuple, Any, Union, Callable, Optional

from diffusers import DiffusionPipeline

from cache_dit.cache_factory.cache_types import CacheType
from cache_dit.cache_factory.block_adapters import BlockAdapter
from cache_dit.cache_factory.block_adapters import ParamsModifier
from cache_dit.cache_factory.block_adapters import BlockAdapterRegistry
from cache_dit.cache_factory.cache_contexts import ContextManager
from cache_dit.cache_factory.cache_contexts import BasicCacheConfig
from cache_dit.cache_factory.cache_contexts import CalibratorConfig
from cache_dit.cache_factory.cache_blocks import UnifiedBlocks
from cache_dit.logger import init_logger

logger = init_logger(__name__)


# Unified Cached Adapter
class CachedAdapter:

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    @classmethod
    def apply(
        cls,
        pipe_or_adapter: Union[
            DiffusionPipeline,
            BlockAdapter,
        ],
        **context_kwargs,
    ) -> Union[
        DiffusionPipeline,
        BlockAdapter,
    ]:
        assert (
            pipe_or_adapter is not None
        ), "pipe or block_adapter can not both None!"

        if isinstance(pipe_or_adapter, DiffusionPipeline):
            if BlockAdapterRegistry.is_supported(pipe_or_adapter):
                logger.info(
                    f"{pipe_or_adapter.__class__.__name__} is officially "
                    "supported by cache-dit. Use it's pre-defined BlockAdapter "
                    "directly!"
                )
                block_adapter = BlockAdapterRegistry.get_adapter(
                    pipe_or_adapter
                )
                if params_modifiers := context_kwargs.pop(
                    "params_modifiers",
                    None,
                ):
                    block_adapter.params_modifiers = params_modifiers

                return cls.cachify(
                    block_adapter,
                    **context_kwargs,
                ).pipe
            else:
                raise ValueError(
                    f"{pipe_or_adapter.__class__.__name__} is not officially supported "
                    "by cache-dit, please set BlockAdapter instead!"
                )
        else:
            assert isinstance(pipe_or_adapter, BlockAdapter)
            logger.info(
                "Adapting Cache Acceleration using custom BlockAdapter!"
            )
            if pipe_or_adapter.params_modifiers is None:
                if params_modifiers := context_kwargs.pop(
                    "params_modifiers", None
                ):
                    pipe_or_adapter.params_modifiers = params_modifiers

            return cls.cachify(
                pipe_or_adapter,
                **context_kwargs,
            )

    @classmethod
    def cachify(
        cls,
        block_adapter: BlockAdapter,
        **context_kwargs,
    ) -> BlockAdapter:

        if block_adapter.auto:
            block_adapter = BlockAdapter.auto_block_adapter(
                block_adapter,
            )

        if BlockAdapter.check_block_adapter(block_adapter):

            # 0. Must normalize block_adapter before apply cache
            block_adapter = BlockAdapter.normalize(block_adapter)
            if BlockAdapter.is_cached(block_adapter):
                return block_adapter

            # 1. Apply cache on pipeline: wrap cache context, must
            # call create_context before mock_blocks.
            _, contexts_kwargs = cls.create_context(
                block_adapter,
                **context_kwargs,
            )

            # 2. Apply cache on transformer: mock cached blocks
            cls.mock_blocks(
                block_adapter,
                contexts_kwargs,
            )

        return block_adapter

    @classmethod
    def check_context_kwargs(
        cls,
        block_adapter: BlockAdapter,
        **context_kwargs,
    ):
        # Check context_kwargs
        cache_config: BasicCacheConfig = context_kwargs["cache_config"]  # ref
        assert cache_config is not None, "cache_config can not be None."
        if cache_config.enable_separate_cfg is None:
            # Check cfg for some specific case if users don't set it as True
            if BlockAdapterRegistry.has_separate_cfg(block_adapter):
                cache_config.enable_separate_cfg = True
                logger.info(
                    f"Use custom 'enable_separate_cfg' from BlockAdapter: True. "
                    f"Pipeline: {block_adapter.pipe.__class__.__name__}."
                )
            else:
                cache_config.enable_separate_cfg = (
                    BlockAdapterRegistry.has_separate_cfg(block_adapter.pipe)
                )
                logger.info(
                    f"Use default 'enable_separate_cfg' from block adapter "
                    f"register: {cache_config.enable_separate_cfg}, "
                    f"Pipeline: {block_adapter.pipe.__class__.__name__}."
                )
        else:
            logger.info(
                f"Use custom 'enable_separate_cfg' from cache context "
                f"kwargs: {cache_config.enable_separate_cfg}. "
                f"Pipeline: {block_adapter.pipe.__class__.__name__}."
            )

        cache_type = context_kwargs.pop("cache_type", None)
        if cache_type is not None:
            assert isinstance(
                cache_type, CacheType
            ), f"cache_type must be CacheType Enum, but got {type(cache_type)}."
            assert cache_type == cache_config.cache_type, (
                f"cache_type from context_kwargs ({cache_type}) must be the same "
                f"as that from cache_config ({cache_config.cache_type})."
            )

        return context_kwargs

    @classmethod
    def create_context(
        cls,
        block_adapter: BlockAdapter,
        **context_kwargs,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:

        BlockAdapter.assert_normalized(block_adapter)

        if BlockAdapter.is_cached(block_adapter.pipe):
            return block_adapter.pipe

        # Check context_kwargs
        context_kwargs = cls.check_context_kwargs(
            block_adapter, **context_kwargs
        )
        # Apply cache on pipeline: wrap cache context
        pipe_cls_name = block_adapter.pipe.__class__.__name__

        # Each Pipeline should have it's own context manager instance.
        # Different transformers (Wan2.2, etc) should shared the same
        # cache manager but with different cache context (according
        # to their unique instance id).
        cache_config: BasicCacheConfig = context_kwargs.get(
            "cache_config", None
        )
        assert cache_config is not None, "cache_config can not be None."
        context_manager = ContextManager(
            name=f"{pipe_cls_name}_{hash(id(block_adapter.pipe))}",
            cache_type=cache_config.cache_type,
        )
        block_adapter.pipe._context_manager = context_manager  # instance level

        flatten_contexts, contexts_kwargs = cls.modify_context_params(
            block_adapter, **context_kwargs
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
                        context_manager.enter_context(
                            context_manager.reset_context(
                                context_name,
                                **context_kwargs,
                            ),
                        )
                    )
                outputs = original_call(self, *args, **kwargs)
                cls.apply_stats_hooks(block_adapter)
                return outputs

        block_adapter.pipe.__class__.__call__ = new_call
        block_adapter.pipe.__class__._original_call = original_call
        block_adapter.pipe.__class__._is_cached = True

        cls.apply_params_hooks(block_adapter, contexts_kwargs)

        return flatten_contexts, contexts_kwargs

    @classmethod
    def modify_context_params(
        cls,
        block_adapter: BlockAdapter,
        **context_kwargs,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:

        flatten_contexts = BlockAdapter.flatten(
            block_adapter.unique_blocks_name
        )
        contexts_kwargs = [
            copy.deepcopy(context_kwargs)  # must deep copy
            for _ in range(
                len(flatten_contexts),
            )
        ]

        for i in range(len(contexts_kwargs)):
            contexts_kwargs[i]["name"] = flatten_contexts[i]

        if block_adapter.params_modifiers is None:
            for i in range(len(contexts_kwargs)):
                cls._config_messages(**contexts_kwargs[i])
            return flatten_contexts, contexts_kwargs

        flatten_modifiers: List[ParamsModifier] = BlockAdapter.flatten(
            block_adapter.params_modifiers,
        )

        for i in range(
            min(len(contexts_kwargs), len(flatten_modifiers)),
        ):
            if "cache_config" in flatten_modifiers[i]._context_kwargs:
                modifier_cache_config = flatten_modifiers[
                    i
                ]._context_kwargs.get("cache_config", None)
                modifier_calibrator_config = flatten_modifiers[
                    i
                ]._context_kwargs.get("calibrator_config", None)
                if modifier_cache_config is not None:
                    assert isinstance(
                        modifier_cache_config, BasicCacheConfig
                    ), (
                        f"cache_config must be BasicCacheConfig, but got "
                        f"{type(modifier_cache_config)}."
                    )
                    contexts_kwargs[i]["cache_config"].update(
                        **modifier_cache_config.as_dict()
                    )
                if modifier_calibrator_config is not None:
                    assert isinstance(
                        modifier_calibrator_config, CalibratorConfig
                    ), (
                        f"calibrator_config must be CalibratorConfig, but got "
                        f"{type(modifier_calibrator_config)}."
                    )
                    if (
                        contexts_kwargs[i].get("calibrator_config", None)
                        is None
                    ):
                        contexts_kwargs[i][
                            "calibrator_config"
                        ] = modifier_calibrator_config
                    else:
                        contexts_kwargs[i]["calibrator_config"].update(
                            **modifier_calibrator_config.as_dict()
                        )
            cls._config_messages(**contexts_kwargs[i])

        return flatten_contexts, contexts_kwargs

    @classmethod
    def _config_messages(cls, **contexts_kwargs):
        cache_config: BasicCacheConfig = contexts_kwargs.get(
            "cache_config", None
        )
        calibrator_config: CalibratorConfig = contexts_kwargs.get(
            "calibrator_config", None
        )
        if cache_config is not None:
            message = f"Collected Context Config: {cache_config.strify()}"
            if calibrator_config is not None:
                message += f", Calibrator Config: {calibrator_config.strify(details=True)}"
            else:
                message += ", Calibrator Config: None"
            logger.info(message)

    @classmethod
    def mock_blocks(
        cls,
        block_adapter: BlockAdapter,
        contexts_kwargs: List[Dict],
    ) -> List[torch.nn.Module]:

        BlockAdapter.assert_normalized(block_adapter)

        if BlockAdapter.is_cached(block_adapter.transformer):
            return block_adapter.transformer

        # Apply cache on transformer: mock cached transformer blocks
        for (
            unified_blocks,
            transformer,
            blocks_name,
            unique_blocks_name,
            dummy_blocks_names,
        ) in zip(
            cls.collect_unified_blocks(
                block_adapter,
                contexts_kwargs,
            ),
            block_adapter.transformer,
            block_adapter.blocks_name,
            block_adapter.unique_blocks_name,
            block_adapter.dummy_blocks_names,
        ):
            cls.mock_transformer(
                unified_blocks,
                transformer,
                blocks_name,
                unique_blocks_name,
                dummy_blocks_names,
            )

        return block_adapter.transformer

    @classmethod
    def mock_transformer(
        cls,
        unified_blocks: Dict[str, torch.nn.ModuleList],
        transformer: torch.nn.Module,
        blocks_name: List[str],
        unique_blocks_name: List[str],
        dummy_blocks_names: List[str],
    ) -> torch.nn.Module:
        dummy_blocks = torch.nn.ModuleList()

        original_forward = transformer.forward

        assert isinstance(dummy_blocks_names, list)

        from accelerate import hooks

        _hf_hook: Optional[hooks.ModelHook] = None

        if getattr(transformer, "_hf_hook", None) is not None:
            _hf_hook = transformer._hf_hook  # hooks from accelerate.hooks
            if hasattr(transformer, "_old_forward"):
                logger.warning(
                    "_hf_hook is not None, so, we have to re-direct transformer's "
                    f"original_forward({id(original_forward)}) to transformer's "
                    f"_old_forward({id(transformer._old_forward)})"
                )
                original_forward = transformer._old_forward

        # TODO: remove group offload hooks the re-apply after cache applied.
        # hooks = _diffusers_hook.hooks.copy(); _diffusers_hook.hooks.clear()
        # re-apply hooks to transformer after cache applied.
        # from diffusers.hooks.hooks import HookFunctionReference, HookRegistry
        # from diffusers.hooks.group_offloading import apply_group_offloading

        def new_forward(self, *args, **kwargs):
            with ExitStack() as stack:
                for name, context_name in zip(
                    blocks_name,
                    unique_blocks_name,
                ):
                    stack.enter_context(
                        unittest.mock.patch.object(
                            self, name, unified_blocks[context_name]
                        )
                    )
                for dummy_name in dummy_blocks_names:
                    stack.enter_context(
                        unittest.mock.patch.object(
                            self, dummy_name, dummy_blocks
                        )
                    )
                outputs = original_forward(*args, **kwargs)
            return outputs

        def new_forward_with_hf_hook(self, *args, **kwargs):
            # Compatible with model cpu offload
            if _hf_hook is not None and hasattr(_hf_hook, "pre_forward"):
                args, kwargs = _hf_hook.pre_forward(self, *args, **kwargs)

            outputs = new_forward(self, *args, **kwargs)

            if _hf_hook is not None and hasattr(_hf_hook, "post_forward"):
                outputs = _hf_hook.post_forward(self, outputs)

            return outputs

        # NOTE: Still can't fully compatible with group offloading
        transformer.forward = functools.update_wrapper(
            functools.partial(new_forward_with_hf_hook, transformer),
            new_forward_with_hf_hook,
        )

        transformer._original_forward = original_forward
        transformer._is_cached = True

        return transformer

    @classmethod
    def collect_unified_blocks(
        cls,
        block_adapter: BlockAdapter,
        contexts_kwargs: List[Dict],
    ) -> List[Dict[str, torch.nn.ModuleList]]:

        BlockAdapter.assert_normalized(block_adapter)

        total_cached_blocks: List[Dict[str, torch.nn.ModuleList]] = []
        assert hasattr(block_adapter.pipe, "_context_manager")
        assert isinstance(
            block_adapter.pipe._context_manager,
            ContextManager._supported_managers,
        )

        for i in range(len(block_adapter.transformer)):

            unified_blocks_bind_context = {}
            for j in range(len(block_adapter.blocks[i])):
                cache_config: BasicCacheConfig = contexts_kwargs[
                    i * len(block_adapter.blocks[i]) + j
                ]["cache_config"]
                unified_blocks_bind_context[
                    block_adapter.unique_blocks_name[i][j]
                ] = torch.nn.ModuleList(
                    [
                        UnifiedBlocks(
                            # 0. Transformer blocks configuration
                            block_adapter.blocks[i][j],
                            transformer=block_adapter.transformer[i],
                            forward_pattern=block_adapter.forward_pattern[i][j],
                            check_forward_pattern=block_adapter.check_forward_pattern,
                            check_num_outputs=block_adapter.check_num_outputs,
                            # 1. Cache/Prune context configuration
                            cache_prefix=block_adapter.blocks_name[i][j],
                            cache_context=block_adapter.unique_blocks_name[i][
                                j
                            ],
                            context_manager=block_adapter.pipe._context_manager,
                            cache_type=cache_config.cache_type,
                        )
                    ]
                )

            total_cached_blocks.append(unified_blocks_bind_context)

        return total_cached_blocks

    @classmethod
    def apply_params_hooks(
        cls,
        block_adapter: BlockAdapter,
        contexts_kwargs: List[Dict],
    ):
        block_adapter.pipe._context_kwargs = contexts_kwargs[0]

        params_shift = 0
        for i in range(len(block_adapter.transformer)):

            block_adapter.transformer[i]._forward_pattern = (
                block_adapter.forward_pattern
            )
            block_adapter.transformer[i]._has_separate_cfg = (
                block_adapter.has_separate_cfg
            )
            block_adapter.transformer[i]._context_kwargs = contexts_kwargs[
                params_shift
            ]

            blocks = block_adapter.blocks[i]
            for j in range(len(blocks)):
                blocks[j]._forward_pattern = block_adapter.forward_pattern[i][j]
                blocks[j]._context_kwargs = contexts_kwargs[params_shift + j]

            params_shift += len(blocks)

    @classmethod
    def apply_stats_hooks(
        cls,
        block_adapter: BlockAdapter,
    ):
        from cache_dit.cache_factory.cache_blocks import (
            apply_stats,
        )

        context_manager = block_adapter.pipe._context_manager

        for i in range(len(block_adapter.transformer)):
            apply_stats(
                block_adapter.transformer[i],
                cache_context=block_adapter.unique_blocks_name[i][-1],
                context_manager=context_manager,
            )
            for blocks, unique_name in zip(
                block_adapter.blocks[i],
                block_adapter.unique_blocks_name[i],
            ):
                apply_stats(
                    blocks,
                    cache_context=unique_name,
                    context_manager=context_manager,
                )

    @classmethod
    def maybe_release_hooks(
        cls,
        pipe_or_adapter: Union[
            DiffusionPipeline,
            BlockAdapter,
        ],
    ):
        # release model hooks
        def _release_blocks_hooks(blocks):
            return

        def _release_transformer_hooks(transformer):
            if hasattr(transformer, "_original_forward"):
                original_forward = transformer._original_forward
                transformer.forward = original_forward.__get__(transformer)
                del transformer._original_forward
            if hasattr(transformer, "_is_cached"):
                del transformer._is_cached

        def _release_pipeline_hooks(pipe):
            if hasattr(pipe, "_original_call"):
                original_call = pipe.__class__._original_call
                pipe.__class__.__call__ = original_call
                del pipe.__class__._original_call
            if hasattr(pipe, "_context_manager"):
                context_manager = pipe._context_manager
                if isinstance(
                    context_manager, ContextManager._supported_managers
                ):
                    context_manager.clear_contexts()
                del pipe._context_manager
            if hasattr(pipe, "_is_cached"):
                del pipe.__class__._is_cached

        cls.release_hooks(
            pipe_or_adapter,
            _release_blocks_hooks,
            _release_transformer_hooks,
            _release_pipeline_hooks,
        )

        # release params hooks
        def _release_blocks_params(blocks):
            if hasattr(blocks, "_forward_pattern"):
                del blocks._forward_pattern
            if hasattr(blocks, "_context_kwargs"):
                del blocks._context_kwargs

        def _release_transformer_params(transformer):
            if hasattr(transformer, "_forward_pattern"):
                del transformer._forward_pattern
            if hasattr(transformer, "_has_separate_cfg"):
                del transformer._has_separate_cfg
            if hasattr(transformer, "_context_kwargs"):
                del transformer._context_kwargs
            for blocks in BlockAdapter.find_blocks(transformer):
                _release_blocks_params(blocks)

        def _release_pipeline_params(pipe):
            if hasattr(pipe, "_context_kwargs"):
                del pipe._context_kwargs

        cls.release_hooks(
            pipe_or_adapter,
            _release_blocks_params,
            _release_transformer_params,
            _release_pipeline_params,
        )

        # release stats hooks
        from cache_dit.cache_factory.cache_blocks import (
            remove_stats,
        )

        cls.release_hooks(
            pipe_or_adapter, remove_stats, remove_stats, remove_stats
        )

    @classmethod
    def release_hooks(
        cls,
        pipe_or_adapter: Union[
            DiffusionPipeline,
            BlockAdapter,
        ],
        _release_blocks: Callable,
        _release_transformer: Callable,
        _release_pipeline: Callable,
    ):
        if isinstance(pipe_or_adapter, DiffusionPipeline):
            pipe = pipe_or_adapter
            _release_pipeline(pipe)
            if hasattr(pipe, "transformer"):
                _release_transformer(pipe.transformer)
            if hasattr(pipe, "transformer_2"):  # Wan 2.2
                _release_transformer(pipe.transformer_2)
        elif isinstance(pipe_or_adapter, BlockAdapter):
            adapter = pipe_or_adapter
            BlockAdapter.assert_normalized(adapter)
            _release_pipeline(adapter.pipe)
            for transformer in BlockAdapter.flatten(adapter.transformer):
                _release_transformer(transformer)
            for blocks in BlockAdapter.flatten(adapter.blocks):
                _release_blocks(blocks)
