import torch

import inspect
import unittest
import functools
import dataclasses

from typing import Any, Tuple, List, Optional
from contextlib import ExitStack
from diffusers import DiffusionPipeline
from cache_dit.cache_factory.patch.flux import (
    maybe_patch_flux_transformer,
)
from cache_dit.cache_factory import CacheType
from cache_dit.cache_factory import ForwardPattern
from cache_dit.cache_factory.cache_blocks import (
    cache_context,
    DBCachedTransformerBlocks,
)
from cache_dit.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class BlockAdapter:
    pipe: DiffusionPipeline = None
    transformer: torch.nn.Module = None
    blocks: torch.nn.ModuleList = None
    # transformer_blocks, blocks, etc.
    blocks_name: str = None
    dummy_blocks_names: list[str] = dataclasses.field(default_factory=list)
    # flags to control auto block adapter
    auto: bool = False
    allow_prefixes: List[str] = dataclasses.field(
        default_factory=lambda: [
            "transformer",
            "single_transformer",
            "blocks",
            "layers",
        ]
    )
    check_prefixes: bool = True
    allow_suffixes: List[str] = dataclasses.field(
        default_factory=lambda: ["TransformerBlock"]
    )
    check_suffixes: bool = False
    blocks_policy: str = dataclasses.field(
        default="max", metadata={"allowed_values": ["max", "min"]}
    )

    def __post_init__(self):
        self.maybe_apply_patch()

    def maybe_apply_patch(self):
        # Process some specificial cases, specific for transformers
        # that has different forward patterns between single_transformer_blocks
        # and transformer_blocks , such as Flux (diffusers < 0.35.0).
        if self.transformer.__class__.__name__.startswith("Flux"):
            self.transformer = maybe_patch_flux_transformer(
                self.transformer,
                blocks=self.blocks,
            )

    @staticmethod
    def auto_block_adapter(
        adapter: "BlockAdapter",
        forward_pattern: Optional[ForwardPattern] = None,
    ) -> "BlockAdapter":
        assert adapter.auto, (
            "Please manually set `auto` to True, or, manually "
            "set all the transformer blocks configuration."
        )
        assert adapter.pipe is not None, "adapter.pipe can not be None."
        pipe = adapter.pipe

        assert hasattr(pipe, "transformer"), "pipe.transformer can not be None."

        transformer = pipe.transformer

        # "transformer_blocks", "blocks", "single_transformer_blocks", "layers"
        blocks, blocks_name = BlockAdapter.find_blocks(
            transformer=transformer,
            allow_prefixes=adapter.allow_prefixes,
            allow_suffixes=adapter.allow_suffixes,
            check_prefixes=adapter.check_prefixes,
            check_suffixes=adapter.check_suffixes,
            blocks_policy=adapter.blocks_policy,
            forward_pattern=forward_pattern,
        )

        return BlockAdapter(
            pipe=pipe,
            transformer=transformer,
            blocks=blocks,
            blocks_name=blocks_name,
        )

    @staticmethod
    def check_block_adapter(adapter: "BlockAdapter") -> bool:
        if (
            isinstance(adapter.pipe, DiffusionPipeline)
            and adapter.transformer is not None
            and adapter.blocks is not None
            and adapter.blocks_name is not None
            and isinstance(adapter.blocks, torch.nn.ModuleList)
        ):
            return True

        logger.warning("Check block adapter failed!")
        return False

    @staticmethod
    def find_blocks(
        transformer: torch.nn.Module,
        allow_prefixes: List[str] = [
            "transformer",
            "single_transformer",
            "blocks",
            "layers",
        ],
        allow_suffixes: List[str] = [
            "TransformerBlock",
        ],
        check_prefixes: bool = True,
        check_suffixes: bool = False,
        **kwargs,
    ) -> Tuple[torch.nn.ModuleList, str]:
        # Check prefixes
        if check_prefixes:
            blocks_names = []
            for attr_name in dir(transformer):
                for prefix in allow_prefixes:
                    if attr_name.startswith(prefix):
                        blocks_names.append(attr_name)
        else:
            blocks_names = dir(transformer)

        # Check ModuleList
        valid_names = []
        valid_count = []
        forward_pattern = kwargs.get("forward_pattern", None)
        for blocks_name in blocks_names:
            if blocks := getattr(transformer, blocks_name, None):
                if isinstance(blocks, torch.nn.ModuleList):
                    block = blocks[0]
                    block_cls_name = block.__class__.__name__
                    # Check suffixes
                    if isinstance(block, torch.nn.Module) and (
                        any(
                            (
                                block_cls_name.endswith(allow_suffix)
                                for allow_suffix in allow_suffixes
                            )
                        )
                        or (not check_suffixes)
                    ):
                        # May check forward pattern
                        if forward_pattern is not None:
                            if BlockAdapter.match_blocks_pattern(
                                blocks,
                                forward_pattern,
                                logging=False,
                            ):
                                valid_names.append(blocks_name)
                                valid_count.append(len(blocks))
                        else:
                            valid_names.append(blocks_name)
                            valid_count.append(len(blocks))

        if not valid_names:
            raise ValueError(
                "Auto selected transformer blocks failed, please set it manually."
            )

        final_name = valid_names[0]
        final_count = valid_count[0]
        block_policy = kwargs.get("blocks_policy", "max")

        for blocks_name, count in zip(valid_names, valid_count):
            blocks = getattr(transformer, blocks_name)
            logger.info(
                f"Auto selected transformer blocks: {blocks_name}, "
                f"class: {blocks[0].__class__.__name__}, "
                f"num blocks: {count}"
            )
            if block_policy == "max":
                if final_count < count:
                    final_count = count
                    final_name = blocks_name
            else:
                if final_count > count:
                    final_count = count
                    final_name = blocks_name

        final_blocks = getattr(transformer, final_name)

        logger.info(
            f"Final selected transformer blocks: {final_name}, "
            f"class: {final_blocks[0].__class__.__name__}, "
            f"num blocks: {final_count}, block_policy: {block_policy}."
        )

        return final_blocks, final_name

    @staticmethod
    def match_block_pattern(
        block: torch.nn.Module,
        forward_pattern: ForwardPattern,
    ) -> bool:
        assert (
            forward_pattern.Supported
            and forward_pattern in ForwardPattern.supported_patterns()
        ), f"Pattern {forward_pattern} is not support now!"

        forward_parameters = set(
            inspect.signature(block.forward).parameters.keys()
        )
        num_outputs = str(
            inspect.signature(block.forward).return_annotation
        ).count("torch.Tensor")

        in_matched = True
        out_matched = True
        if num_outputs > 0 and len(forward_pattern.Out) != num_outputs:
            # output pattern not match
            out_matched = False

        for required_param in forward_pattern.In:
            if required_param not in forward_parameters:
                in_matched = False

        return in_matched and out_matched

    @staticmethod
    def match_blocks_pattern(
        transformer_blocks: torch.nn.ModuleList,
        forward_pattern: ForwardPattern,
        logging: bool = True,
    ) -> bool:
        assert (
            forward_pattern.Supported
            and forward_pattern in ForwardPattern.supported_patterns()
        ), f"Pattern {forward_pattern} is not support now!"

        assert isinstance(transformer_blocks, torch.nn.ModuleList)

        pattern_matched_states = []
        for block in transformer_blocks:
            pattern_matched_states.append(
                BlockAdapter.match_block_pattern(
                    block,
                    forward_pattern,
                )
            )

        pattern_matched = all(pattern_matched_states)  # all block match
        if pattern_matched and logging:
            block_cls_name = transformer_blocks[0].__class__.__name__
            logger.info(
                f"Match Block Forward Pattern: {block_cls_name}, {forward_pattern}"
                f"\nIN:{forward_pattern.In}, OUT:{forward_pattern.Out})"
            )

        return pattern_matched


@dataclasses.dataclass
class UnifiedCacheParams:
    block_adapter: BlockAdapter = None
    forward_pattern: ForwardPattern = ForwardPattern.Pattern_0


class UnifiedCacheAdapter:
    _supported_pipelines = [
        "Flux",
        "Mochi",
        "CogVideoX",
        "Wan",
        "HunyuanVideo",
        "QwenImage",
        "LTXVideo",
        "Allegro",
        "CogView3Plus",
        "CogView4",
        "Cosmos",
        "EasyAnimate",
        "SkyReelsV2",
        "SD3",
    ]

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    @classmethod
    def is_supported(cls, pipe: DiffusionPipeline) -> bool:
        pipe_cls_name: str = pipe.__class__.__name__
        for prefix in cls._supported_pipelines:
            if pipe_cls_name.startswith(prefix):
                return True
        return False

    @classmethod
    def get_params(cls, pipe: DiffusionPipeline) -> UnifiedCacheParams:
        pipe_cls_name: str = pipe.__class__.__name__
        if pipe_cls_name.startswith("Flux"):
            from diffusers import FluxTransformer2DModel

            assert isinstance(pipe.transformer, FluxTransformer2DModel)
            return UnifiedCacheParams(
                block_adapter=BlockAdapter(
                    pipe=pipe,
                    transformer=pipe.transformer,
                    blocks=(
                        pipe.transformer.transformer_blocks
                        + pipe.transformer.single_transformer_blocks
                    ),
                    blocks_name="transformer_blocks",
                    dummy_blocks_names=["single_transformer_blocks"],
                ),
                forward_pattern=ForwardPattern.Pattern_1,
            )
        elif pipe_cls_name.startswith("Mochi"):
            from diffusers import MochiTransformer3DModel

            assert isinstance(pipe.transformer, MochiTransformer3DModel)
            return UnifiedCacheParams(
                block_adapter=BlockAdapter(
                    pipe=pipe,
                    transformer=pipe.transformer,
                    blocks=pipe.transformer.transformer_blocks,
                    blocks_name="transformer_blocks",
                    dummy_blocks_names=[],
                ),
                forward_pattern=ForwardPattern.Pattern_0,
            )
        elif pipe_cls_name.startswith("CogVideoX"):
            from diffusers import CogVideoXTransformer3DModel

            assert isinstance(pipe.transformer, CogVideoXTransformer3DModel)
            return UnifiedCacheParams(
                block_adapter=BlockAdapter(
                    pipe=pipe,
                    transformer=pipe.transformer,
                    blocks=pipe.transformer.transformer_blocks,
                    blocks_name="transformer_blocks",
                    dummy_blocks_names=[],
                ),
                forward_pattern=ForwardPattern.Pattern_0,
            )
        elif pipe_cls_name.startswith("Wan"):
            from diffusers import (
                WanTransformer3DModel,
                WanVACETransformer3DModel,
            )

            assert isinstance(
                pipe.transformer,
                (WanTransformer3DModel, WanVACETransformer3DModel),
            )
            return UnifiedCacheParams(
                block_adapter=BlockAdapter(
                    pipe=pipe,
                    transformer=pipe.transformer,
                    blocks=pipe.transformer.blocks,
                    blocks_name="blocks",
                    dummy_blocks_names=[],
                ),
                forward_pattern=ForwardPattern.Pattern_2,
            )
        elif pipe_cls_name.startswith("HunyuanVideo"):
            from diffusers import HunyuanVideoTransformer3DModel

            assert isinstance(pipe.transformer, HunyuanVideoTransformer3DModel)
            return UnifiedCacheParams(
                block_adapter=BlockAdapter(
                    pipe=pipe,
                    blocks=(
                        pipe.transformer.transformer_blocks
                        + pipe.transformer.single_transformer_blocks
                    ),
                    blocks_name="transformer_blocks",
                    dummy_blocks_names=["single_transformer_blocks"],
                ),
                forward_pattern=ForwardPattern.Pattern_0,
            )
        elif pipe_cls_name.startswith("QwenImage"):
            from diffusers import QwenImageTransformer2DModel

            assert isinstance(pipe.transformer, QwenImageTransformer2DModel)
            return UnifiedCacheParams(
                block_adapter=BlockAdapter(
                    pipe=pipe,
                    transformer=pipe.transformer,
                    blocks=pipe.transformer.transformer_blocks,
                    blocks_name="transformer_blocks",
                    dummy_blocks_names=[],
                ),
                forward_pattern=ForwardPattern.Pattern_1,
            )
        elif pipe_cls_name.startswith("LTXVideo"):
            from diffusers import LTXVideoTransformer3DModel

            assert isinstance(pipe.transformer, LTXVideoTransformer3DModel)
            return UnifiedCacheParams(
                block_adapter=BlockAdapter(
                    pipe=pipe,
                    transformer=pipe.transformer,
                    blocks=pipe.transformer.transformer_blocks,
                    blocks_name="transformer_blocks",
                    dummy_blocks_names=[],
                ),
                forward_pattern=ForwardPattern.Pattern_2,
            )
        elif pipe_cls_name.startswith("Allegro"):
            from diffusers import AllegroTransformer3DModel

            assert isinstance(pipe.transformer, AllegroTransformer3DModel)
            return UnifiedCacheParams(
                block_adapter=BlockAdapter(
                    pipe=pipe,
                    transformer=pipe.transformer,
                    blocks=pipe.transformer.transformer_blocks,
                    blocks_name="transformer_blocks",
                    dummy_blocks_names=[],
                ),
                forward_pattern=ForwardPattern.Pattern_2,
            )
        elif pipe_cls_name.startswith("CogView3Plus"):
            from diffusers import CogView3PlusTransformer2DModel

            assert isinstance(pipe.transformer, CogView3PlusTransformer2DModel)
            return UnifiedCacheParams(
                block_adapter=BlockAdapter(
                    pipe=pipe,
                    transformer=pipe.transformer,
                    blocks=pipe.transformer.transformer_blocks,
                    blocks_name="transformer_blocks",
                    dummy_blocks_names=[],
                ),
                forward_pattern=ForwardPattern.Pattern_0,
            )
        elif pipe_cls_name.startswith("CogView4"):
            from diffusers import CogView4Transformer2DModel

            assert isinstance(pipe.transformer, CogView4Transformer2DModel)
            return UnifiedCacheParams(
                block_adapter=BlockAdapter(
                    pipe=pipe,
                    transformer=pipe.transformer,
                    blocks=pipe.transformer.transformer_blocks,
                    blocks_name="transformer_blocks",
                    dummy_blocks_names=[],
                ),
                forward_pattern=ForwardPattern.Pattern_0,
            )
        elif pipe_cls_name.startswith("Cosmos"):
            from diffusers import CosmosTransformer3DModel

            assert isinstance(pipe.transformer, CosmosTransformer3DModel)
            return UnifiedCacheParams(
                block_adapter=BlockAdapter(
                    pipe=pipe,
                    transformer=pipe.transformer,
                    blocks=pipe.transformer.transformer_blocks,
                    blocks_name="transformer_blocks",
                    dummy_blocks_names=[],
                ),
                forward_pattern=ForwardPattern.Pattern_2,
            )
        elif pipe_cls_name.startswith("EasyAnimate"):
            from diffusers import EasyAnimateTransformer3DModel

            assert isinstance(pipe.transformer, EasyAnimateTransformer3DModel)
            return UnifiedCacheParams(
                block_adapter=BlockAdapter(
                    pipe=pipe,
                    transformer=pipe.transformer,
                    blocks=pipe.transformer.transformer_blocks,
                    blocks_name="transformer_blocks",
                    dummy_blocks_names=[],
                ),
                forward_pattern=ForwardPattern.Pattern_0,
            )
        elif pipe_cls_name.startswith("SkyReelsV2"):
            from diffusers import SkyReelsV2Transformer3DModel

            assert isinstance(pipe.transformer, SkyReelsV2Transformer3DModel)
            return UnifiedCacheParams(
                block_adapter=BlockAdapter(
                    pipe=pipe,
                    transformer=pipe.transformer,
                    blocks=pipe.transformer.blocks,
                    blocks_name="blocks",
                    dummy_blocks_names=[],
                ),
                forward_pattern=ForwardPattern.Pattern_2,
            )
        elif pipe_cls_name.startswith("SD3"):
            from diffusers import SD3Transformer2DModel

            assert isinstance(pipe.transformer, SD3Transformer2DModel)
            return UnifiedCacheParams(
                block_adapter=BlockAdapter(
                    pipe=pipe,
                    transformer=pipe.transformer,
                    blocks=pipe.transformer.transformer_blocks,
                    blocks_name="transformer_blocks",
                    dummy_blocks_names=[],
                ),
                forward_pattern=ForwardPattern.Pattern_1,
            )
        else:
            raise ValueError(f"Unknown pipeline class name: {pipe_cls_name}")

    @classmethod
    def apply(
        cls,
        pipe: DiffusionPipeline = None,
        block_adapter: BlockAdapter = None,
        forward_pattern: ForwardPattern = ForwardPattern.Pattern_0,
        **cache_context_kwargs,
    ) -> DiffusionPipeline:
        assert (
            pipe is not None or block_adapter is not None
        ), "pipe or block_adapter can not both None!"

        if pipe is not None:
            if cls.is_supported(pipe):
                logger.info(
                    f"{pipe.__class__.__name__} is officially supported by cache-dit. "
                    "Use it's pre-defined BlockAdapter directly!"
                )
                params = cls.get_params(pipe)
                return cls.cachify(
                    params.block_adapter,
                    forward_pattern=params.forward_pattern,
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
                forward_pattern=forward_pattern,
                **cache_context_kwargs,
            )

    @classmethod
    def cachify(
        cls,
        block_adapter: BlockAdapter,
        *,
        forward_pattern: ForwardPattern = ForwardPattern.Pattern_0,
        **cache_context_kwargs,
    ) -> DiffusionPipeline:

        if block_adapter.auto:
            block_adapter = BlockAdapter.auto_block_adapter(
                block_adapter,
                forward_pattern,
            )

        if BlockAdapter.check_block_adapter(block_adapter):
            # Apply cache on pipeline: wrap cache context
            cls.create_context(
                block_adapter.pipe,
                **cache_context_kwargs,
            )
            # Apply cache on transformer: mock cached transformer blocks
            cls.mock_blocks(
                block_adapter,
                forward_pattern=forward_pattern,
            )
            cls.patch_params(
                block_adapter,
                forward_pattern=forward_pattern,
                **cache_context_kwargs,
            )
        return block_adapter.pipe

    @classmethod
    def patch_params(
        cls,
        block_adapter: BlockAdapter,
        forward_pattern: ForwardPattern = None,
        **cache_context_kwargs,
    ):
        block_adapter.transformer._forward_pattern = forward_pattern
        block_adapter.transformer._cache_context_kwargs = cache_context_kwargs
        block_adapter.pipe.__class__._cache_context_kwargs = (
            cache_context_kwargs
        )

    @classmethod
    def has_separate_cfg(
        cls,
        pipe_or_transformer: DiffusionPipeline | Any,
    ) -> bool:
        cls_name = pipe_or_transformer.__class__.__name__
        if cls_name.startswith("QwenImage"):
            return True
        elif cls_name.startswith("Wan"):
            return True
        return False

    @classmethod
    def check_context_kwargs(cls, pipe, **cache_context_kwargs):
        # Check cache_context_kwargs
        if not cache_context_kwargs["do_separate_cfg"]:
            # Check cfg for some specific case if users don't set it as True
            cache_context_kwargs["do_separate_cfg"] = cls.has_separate_cfg(pipe)

        if cache_type := cache_context_kwargs.pop("cache_type", None):
            assert (
                cache_type == CacheType.DBCache
            ), "Custom cache setting only support for DBCache now!"

        return cache_context_kwargs

    @classmethod
    def create_context(
        cls,
        pipe: DiffusionPipeline,
        **cache_context_kwargs,
    ) -> DiffusionPipeline:
        if getattr(pipe, "_is_cached", False):
            return pipe

        # Check cache_context_kwargs
        cache_context_kwargs = cls.check_context_kwargs(
            pipe,
            **cache_context_kwargs,
        )
        # Apply cache on pipeline: wrap cache context
        cache_kwargs, _ = cache_context.collect_cache_kwargs(
            default_attrs={},
            **cache_context_kwargs,
        )
        original_call = pipe.__class__.__call__

        @functools.wraps(original_call)
        def new_call(self, *args, **kwargs):
            with cache_context.cache_context(
                cache_context.create_cache_context(
                    **cache_kwargs,
                )
            ):
                return original_call(self, *args, **kwargs)

        pipe.__class__.__call__ = new_call
        pipe.__class__._is_cached = True
        return pipe

    @classmethod
    def mock_blocks(
        cls,
        block_adapter: BlockAdapter,
        forward_pattern: ForwardPattern = ForwardPattern.Pattern_0,
    ) -> torch.nn.Module:

        if getattr(block_adapter.transformer, "_is_cached", False):
            return block_adapter.transformer

        # Check block forward pattern matching
        assert BlockAdapter.match_blocks_pattern(
            block_adapter.blocks,
            forward_pattern=forward_pattern,
        ), (
            "No block forward pattern matched, "
            f"supported lists: {ForwardPattern.supported_patterns()}"
        )

        # Apply cache on transformer: mock cached transformer blocks
        cached_blocks = torch.nn.ModuleList(
            [
                DBCachedTransformerBlocks(
                    block_adapter.blocks,
                    transformer=block_adapter.transformer,
                    forward_pattern=forward_pattern,
                )
            ]
        )
        dummy_blocks = torch.nn.ModuleList()

        original_forward = block_adapter.transformer.forward

        assert isinstance(block_adapter.dummy_blocks_names, list)

        @functools.wraps(original_forward)
        def new_forward(self, *args, **kwargs):
            with ExitStack() as stack:
                stack.enter_context(
                    unittest.mock.patch.object(
                        self,
                        block_adapter.blocks_name,
                        cached_blocks,
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
