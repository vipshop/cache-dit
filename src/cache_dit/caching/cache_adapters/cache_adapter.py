import copy
import torch
import unittest
import functools
from contextlib import ExitStack
from typing import Dict, List, Tuple, Any, Union, Callable, Optional

from diffusers import DiffusionPipeline, ModelMixin

from ..cache_types import CacheType
from ..block_adapters import BlockAdapter
from ..block_adapters import FakeDiffusionPipeline
from ..block_adapters import ParamsModifier
from ..block_adapters import BlockAdapterRegister
from ..cache_contexts import ContextManager
from ..cache_contexts import BasicCacheConfig
from ..cache_contexts import CalibratorConfig
from ..cache_blocks import UnifiedBlocks
from cache_dit.logger import init_logger

try:
    from accelerate import hooks

    _accelerate_is_availble = True
except ImportError:
    _accelerate_is_availble = False


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
            # Transformer-only
            torch.nn.Module,
            ModelMixin,
        ],
        **context_kwargs,
    ) -> Union[
        DiffusionPipeline,
        BlockAdapter,
    ]:
        assert pipe_or_adapter is not None, "pipe or block_adapter can not both None!"

        if isinstance(pipe_or_adapter, (DiffusionPipeline, torch.nn.Module, ModelMixin)):
            if BlockAdapterRegister.is_supported(pipe_or_adapter):
                logger.info(
                    f"{pipe_or_adapter.__class__.__name__} is officially "
                    "supported by cache-dit. Use it's pre-defined BlockAdapter "
                    "directly!"
                )
                block_adapter = BlockAdapterRegister.get_adapter(pipe_or_adapter)
                assert block_adapter is not None, (
                    f"BlockAdapter for {pipe_or_adapter.__class__.__name__} " "should not be None!"
                )
                if params_modifiers := context_kwargs.pop(
                    "params_modifiers",
                    None,
                ):
                    block_adapter.params_modifiers = params_modifiers

                block_adapter = cls.cachify(block_adapter, **context_kwargs)
                if isinstance(pipe_or_adapter, DiffusionPipeline):
                    return block_adapter.pipe

                return block_adapter.transformer

            else:
                raise ValueError(
                    f"{pipe_or_adapter.__class__.__name__} is not officially supported "
                    "by cache-dit, please set BlockAdapter instead!"
                )
        else:
            assert isinstance(pipe_or_adapter, BlockAdapter)
            logger.info("Adapting Cache Acceleration using custom BlockAdapter!")
            if pipe_or_adapter.params_modifiers is None:
                if params_modifiers := context_kwargs.pop("params_modifiers", None):
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
            if BlockAdapterRegister.has_separate_cfg(block_adapter):
                cache_config.enable_separate_cfg = True
                logger.info(
                    f"Use custom 'enable_separate_cfg' from BlockAdapter: True. "
                    f"Pipeline: {block_adapter.pipe.__class__.__name__}."
                )
            else:
                cache_config.enable_separate_cfg = BlockAdapterRegister.has_separate_cfg(
                    block_adapter.pipe
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
            logger.warning("Pipeline has been already cached, skip creating cache context again.")
            return None, block_adapter.pipe

        # Check context_kwargs
        context_kwargs = cls.check_context_kwargs(block_adapter, **context_kwargs)

        # Each Pipeline should have it's own context manager instance.
        # Different transformers (Wan2.2, etc) should shared the same
        # cache manager but with different cache context (according
        # to their unique instance id).
        cache_config: BasicCacheConfig = context_kwargs.get("cache_config", None)
        assert cache_config is not None, "cache_config can not be None."
        # Apply cache on pipeline: wrap cache context
        pipe_cls_name = block_adapter.pipe.__class__.__name__
        context_manager = ContextManager(
            name=f"{pipe_cls_name}_{hash(id(block_adapter.pipe))}",
            cache_type=cache_config.cache_type,
            # Force use persistent_context for FakeDiffusionPipeline
            persistent_context=isinstance(block_adapter.pipe, FakeDiffusionPipeline),
        )
        flatten_contexts, contexts_kwargs = cls.modify_context_params(
            block_adapter, **context_kwargs
        )

        block_adapter.pipe._context_manager = context_manager  # instance level

        if not context_manager.persistent_context:

            original_call = block_adapter.pipe.__class__.__call__

            @functools.wraps(original_call)
            def new_call(self, *args, **kwargs):
                with ExitStack() as stack:
                    # cache context will be reset for each pipe inference
                    for context_name, context_kwargs in zip(flatten_contexts, contexts_kwargs):
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

        else:
            # Init persistent cache context for transformer
            for context_name, context_kwargs in zip(flatten_contexts, contexts_kwargs):
                context_manager.reset_context(
                    context_name,
                    **context_kwargs,
                )

        block_adapter.pipe.__class__._is_cached = True

        cls.apply_params_hooks(block_adapter, contexts_kwargs)

        return flatten_contexts, contexts_kwargs

    @classmethod
    def modify_context_params(
        cls,
        block_adapter: BlockAdapter,
        **context_kwargs,
    ) -> Tuple[List[str], List[Dict[str, Any]]]:

        flatten_contexts = BlockAdapter.flatten(block_adapter.unique_blocks_name)
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
            contexts_kwargs[i] = cls._modify_context_params(
                flatten_modifiers[i]._context_kwargs,
                contexts_kwargs[i],
            )
            cls._config_messages(**contexts_kwargs[i])

        return flatten_contexts, contexts_kwargs

    @classmethod
    def _modify_context_params(
        cls,
        new_context_kwargs: Dict[str, Any],
        old_context_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        modified_context_kwargs = copy.deepcopy(old_context_kwargs)
        if "cache_config" in new_context_kwargs:
            new_cache_config = new_context_kwargs.get("cache_config", None)
            new_calibrator_config = new_context_kwargs.get("calibrator_config", None)
            # Modify cache_config
            if new_cache_config is not None:
                assert isinstance(new_cache_config, BasicCacheConfig), (
                    f"cache_config must be BasicCacheConfig, but got " f"{type(new_cache_config)}."
                )
                if modified_context_kwargs.get("cache_config", None) is None:
                    modified_context_kwargs["cache_config"] = new_cache_config
                else:
                    assert isinstance(modified_context_kwargs["cache_config"], BasicCacheConfig), (
                        f"cache_config must be BasicCacheConfig, but got "
                        f"{type(modified_context_kwargs['cache_config'])}."
                    )
                    modified_context_kwargs["cache_config"].update(**new_cache_config.as_dict())
            # Modify calibrator_config
            if new_calibrator_config is not None:
                assert isinstance(new_calibrator_config, CalibratorConfig), (
                    f"calibrator_config must be CalibratorConfig, but got "
                    f"{type(new_calibrator_config)}."
                )
                if modified_context_kwargs.get("calibrator_config", None) is None:
                    modified_context_kwargs["calibrator_config"] = new_calibrator_config
                else:
                    assert isinstance(
                        modified_context_kwargs["calibrator_config"], CalibratorConfig
                    ), (
                        f"calibrator_config must be CalibratorConfig, but got "
                        f"{type(modified_context_kwargs['calibrator_config'])}."
                    )
                    modified_context_kwargs["calibrator_config"].update(
                        **new_calibrator_config.as_dict()
                    )
        return modified_context_kwargs

    @classmethod
    def _config_messages(cls, logging: bool = True, **contexts_kwargs):
        cache_config: BasicCacheConfig = contexts_kwargs.get("cache_config", None)
        calibrator_config: CalibratorConfig = contexts_kwargs.get("calibrator_config", None)
        message = ""
        if cache_config is not None:
            message = f"Collected Context Config: {cache_config.strify()}"
            if calibrator_config is not None:
                message += f", Calibrator Config: {calibrator_config.strify(details=True)}"
            else:
                message += ", Calibrator Config: None"
        if logging:
            logger.info(message)
        return message

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
                block_adapter,
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
        block_adapter: BlockAdapter,
    ) -> torch.nn.Module:
        dummy_blocks = torch.nn.ModuleList()

        original_forward = transformer.forward

        assert isinstance(dummy_blocks_names, list)

        if _accelerate_is_availble:
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
        else:
            _hf_hook = None

        # TODO: remove group offload hooks the re-apply after cache applied.
        # hooks = _diffusers_hook.hooks.copy(); _diffusers_hook.hooks.clear()
        # re-apply hooks to transformer after cache applied.
        # from diffusers.hooks.hooks import HookFunctionReference, HookRegistry
        # from diffusers.hooks.group_offloading import apply_group_offloading
        context_manager: ContextManager = block_adapter.pipe._context_manager
        assert isinstance(context_manager, ContextManager._supported_managers)
        # NOTE: Also assign context manager to transformer for transformer-only case
        transformer._context_manager = context_manager  # instance level
        transformer._context_names = unique_blocks_name  # instance level

        def new_forward(self, *args, **kwargs):
            with ExitStack() as stack:
                for name, context_name in zip(
                    blocks_name,
                    unique_blocks_name,
                ):
                    stack.enter_context(
                        unittest.mock.patch.object(self, name, unified_blocks[context_name])
                    )
                for dummy_name in dummy_blocks_names:
                    stack.enter_context(unittest.mock.patch.object(self, dummy_name, dummy_blocks))
                outputs = original_forward(*args, **kwargs)

                if context_manager.persistent_context and context_manager.is_pre_refreshed():
                    cls.apply_stats_hooks(block_adapter)

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
                unified_blocks_bind_context[block_adapter.unique_blocks_name[i][j]] = (
                    torch.nn.ModuleList(
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
                                cache_context=block_adapter.unique_blocks_name[i][j],
                                context_manager=block_adapter.pipe._context_manager,
                                cache_type=cache_config.cache_type,
                            )
                        ]
                    )
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

            block_adapter.transformer[i]._forward_pattern = block_adapter.forward_pattern
            block_adapter.transformer[i]._has_separate_cfg = block_adapter.has_separate_cfg
            block_adapter.transformer[i]._context_kwargs = contexts_kwargs[params_shift]

            blocks = block_adapter.blocks[i]
            for j in range(len(blocks)):
                blocks[j]._forward_pattern = block_adapter.forward_pattern[i][j]
                blocks[j]._context_kwargs = contexts_kwargs[params_shift + j]

            params_shift += len(blocks)

    @classmethod
    @torch.compiler.disable
    def apply_stats_hooks(
        cls,
        block_adapter: BlockAdapter,
    ):
        from ..cache_blocks import (
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
            torch.nn.Module,  # Transformer-only
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
            if hasattr(transformer, "_context_manager"):
                context_manager = transformer._context_manager
                if isinstance(context_manager, ContextManager._supported_managers):
                    context_manager.clear_contexts()
                try:
                    del transformer._context_manager
                except Exception:
                    pass
            if hasattr(transformer, "_context_names"):
                del transformer._context_names

        def _release_pipeline_hooks(pipe):
            if hasattr(pipe, "_original_call"):
                original_call = pipe.__class__._original_call
                pipe.__class__.__call__ = original_call
                del pipe.__class__._original_call
            if hasattr(pipe, "_context_manager"):
                context_manager = pipe._context_manager
                if isinstance(context_manager, ContextManager._supported_managers):
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
        from ..cache_blocks import (
            remove_stats,
        )

        cls.release_hooks(pipe_or_adapter, remove_stats, remove_stats, remove_stats)

        # maybe release parallelism stats
        from cache_dit.parallelism import (
            remove_parallelism_stats,
        )

        cls.release_hooks(
            pipe_or_adapter,
            remove_parallelism_stats,
            remove_parallelism_stats,
            remove_parallelism_stats,
        )

    @classmethod
    def release_hooks(
        cls,
        pipe_or_adapter: Union[
            DiffusionPipeline,
            BlockAdapter,
        ],
        _release_blocks: Optional[Callable] = None,
        _release_transformer: Optional[Callable] = None,
        _release_pipeline: Optional[Callable] = None,
    ):
        if isinstance(pipe_or_adapter, DiffusionPipeline):
            pipe = pipe_or_adapter
            if _release_pipeline is not None:
                _release_pipeline(pipe)
            if hasattr(pipe, "transformer"):
                if _release_transformer is not None:
                    _release_transformer(pipe.transformer)
            if hasattr(pipe, "transformer_2"):  # Wan 2.2
                if _release_transformer is not None:
                    _release_transformer(pipe.transformer_2)
        elif isinstance(pipe_or_adapter, BlockAdapter):
            adapter = pipe_or_adapter
            BlockAdapter.assert_normalized(adapter)
            if _release_pipeline is not None:
                _release_pipeline(adapter.pipe)
            for transformer in BlockAdapter.flatten(adapter.transformer):
                if _release_transformer is not None:
                    _release_transformer(transformer)
            for blocks in BlockAdapter.flatten(adapter.blocks):
                if _release_blocks is not None:
                    _release_blocks(blocks)
        elif isinstance(pipe_or_adapter, torch.nn.Module):
            transformer = pipe_or_adapter
            if _release_transformer is not None:
                _release_transformer(transformer)
            for blocks in BlockAdapter.find_blocks(transformer):
                if _release_blocks is not None:
                    _release_blocks(blocks)

    @classmethod
    def maybe_refresh_context(
        cls,
        transformer: torch.nn.Module,
        **force_refresh_kwargs,
    ):
        verbose = force_refresh_kwargs.pop("verbose", False)
        # Get context manager from transformer
        if not hasattr(transformer, "_context_manager"):
            logger.warning(
                "Transformer has no attribute '_context_manager', skip refreshing cache context."
            )
            return
        context_manager: ContextManager = transformer._context_manager
        assert isinstance(context_manager, ContextManager._supported_managers)
        if not context_manager.persistent_context:
            logger.warning(
                "Transformer's context manager is not persistent, skip refreshing cache context."
            )
            return
        context_names: List[str] = getattr(transformer, "_context_names", [])
        if not context_names:
            logger.warning(
                "Transformer has no attribute '_context_names' or it's empty, "
                "skip refreshing cache context."
            )
            return

        for context_name in context_names:
            current_context = context_manager.get_context(context_name)
            old_init_kwargs = getattr(current_context, "_init_kwargs", {})  # type: dict
            new_init_kwargs = copy.deepcopy(old_init_kwargs)
            # Remove old context
            context_manager.remove_context(context_name)
            new_init_kwargs = cls._modify_context_params(
                force_refresh_kwargs,
                new_init_kwargs,
            )
            # Re-create new context with old init kwargs updated by
            # force_refresh_kwargs.
            context_manager.reset_context(
                context_name,
                **new_init_kwargs,
            )
            if verbose:
                logger.info(
                    f"✅ Refreshed cache context: {context_name}, "
                    f"{cls._config_messages(logging=False, **new_init_kwargs)}"
                )
            # reset _context_kwargs for transformer
            if hasattr(transformer, "_context_kwargs"):
                # Will overwrite the _context_kwargs by last context kwargs.
                # Only used for strify utilization.
                transformer._context_kwargs = new_init_kwargs
