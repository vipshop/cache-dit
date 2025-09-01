import torch

import unittest
import functools
import dataclasses

# from typing import Any, Tuple, List
from contextlib import ExitStack
from diffusers import DiffusionPipeline
from cache_dit.cache_factory import CacheType
from cache_dit.cache_factory import cache_context
from cache_dit.cache_factory import ForwardPattern
from cache_dit.cache_factory import BlockAdapter
from cache_dit.cache_factory import BlockAdapterRegistry
from cache_dit.cache_factory import DBCachedBlocks

from cache_dit.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class UnifiedCacheParams:
    block_adapter: BlockAdapter = None
    # forward_pattern: ForwardPattern = ForwardPattern.Pattern_0


class UnifiedCacheAdapter:
    # _supported_pipelines = [
    #     "Flux",
    #     "Mochi",
    #     "CogVideoX",
    #     "Wan",
    #     "HunyuanVideo",
    #     "QwenImage",
    #     "LTXVideo",
    #     "Allegro",
    #     "CogView3Plus",
    #     "CogView4",
    #     "Cosmos",
    #     "EasyAnimate",
    #     "SkyReelsV2",
    #     "SD3",
    #     "ConsisID",
    #     "DiT",
    #     "Amused",
    #     "Bria",
    #     "HunyuanDiT",
    #     "HunyuanDiTPAG",
    #     "Lumina",
    #     "Lumina2",
    #     "OmniGen",
    #     "PixArt",
    #     "Sana",
    #     "ShapE",
    #     "StableAudio",
    #     "VisualCloze",
    #     "AuraFlow",
    #     "Chroma",
    #     "HiDream",
    # ]

    def __call__(self, *args, **kwargs):
        return self.apply(*args, **kwargs)

    # @classmethod
    # def supported_pipelines(cls) -> Tuple[int, List[str]]:
    #     return len(cls._supported_pipelines), [
    #         p + "*" for p in cls._supported_pipelines
    #     ]

    # @classmethod
    # def is_supported(cls, pipe: DiffusionPipeline) -> bool:
    #     pipe_cls_name: str = pipe.__class__.__name__
    #     for prefix in cls._supported_pipelines:
    #         if pipe_cls_name.startswith(prefix):
    #             return True
    #     return False

    @classmethod
    def get_params(cls, pipe: DiffusionPipeline) -> UnifiedCacheParams:
        pipe_cls_name: str = pipe.__class__.__name__

        if pipe_cls_name.startswith("Flux"):
            from diffusers import FluxTransformer2DModel
            from cache_dit.cache_factory.patch_functors import FluxPatchFunctor

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
                    patch_functor=FluxPatchFunctor(),
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
            if getattr(pipe, "transformer_2", None):
                # Wan 2.2, cache for low-noise transformer
                assert isinstance(
                    pipe.transformer_2,
                    (WanTransformer3DModel, WanVACETransformer3DModel),
                )
                return UnifiedCacheParams(
                    block_adapter=BlockAdapter(
                        pipe=pipe,
                        transformer=pipe.transformer_2,
                        blocks=pipe.transformer_2.blocks,
                        blocks_name="blocks",
                        dummy_blocks_names=[],
                    ),
                    forward_pattern=ForwardPattern.Pattern_2,
                )
            else:
                # Wan 2.1
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

        elif pipe_cls_name.startswith("ConsisID"):
            from diffusers import ConsisIDTransformer3DModel

            assert isinstance(pipe.transformer, ConsisIDTransformer3DModel)
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

        elif pipe_cls_name.startswith("DiT"):
            from diffusers import DiTTransformer2DModel

            assert isinstance(pipe.transformer, DiTTransformer2DModel)
            return UnifiedCacheParams(
                block_adapter=BlockAdapter(
                    pipe=pipe,
                    transformer=pipe.transformer,
                    blocks=pipe.transformer.transformer_blocks,
                    blocks_name="transformer_blocks",
                    dummy_blocks_names=[],
                ),
                forward_pattern=ForwardPattern.Pattern_3,
            )

        elif pipe_cls_name.startswith("Amused"):
            from diffusers import UVit2DModel

            assert isinstance(pipe.transformer, UVit2DModel)
            return UnifiedCacheParams(
                block_adapter=BlockAdapter(
                    pipe=pipe,
                    transformer=pipe.transformer,
                    blocks=pipe.transformer.transformer_layers,
                    blocks_name="transformer_layers",
                    dummy_blocks_names=[],
                ),
                forward_pattern=ForwardPattern.Pattern_3,
            )

        elif pipe_cls_name.startswith("Bria"):
            from diffusers import BriaTransformer2DModel

            assert isinstance(pipe.transformer, BriaTransformer2DModel)
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
                forward_pattern=ForwardPattern.Pattern_0,
            )

        elif pipe_cls_name.startswith("HunyuanDiT"):
            from diffusers import HunyuanDiT2DModel, HunyuanDiT2DControlNetModel

            assert isinstance(
                pipe.transformer,
                (HunyuanDiT2DModel, HunyuanDiT2DControlNetModel),
            )
            return UnifiedCacheParams(
                block_adapter=BlockAdapter(
                    pipe=pipe,
                    transformer=pipe.transformer,
                    blocks=pipe.transformer.blocks,
                    blocks_name="blocks",
                    dummy_blocks_names=[],
                ),
                forward_pattern=ForwardPattern.Pattern_3,
            )

        elif pipe_cls_name.startswith("HunyuanDiTPAG"):
            from diffusers import HunyuanDiT2DModel

            assert isinstance(pipe.transformer, HunyuanDiT2DModel)
            return UnifiedCacheParams(
                block_adapter=BlockAdapter(
                    pipe=pipe,
                    transformer=pipe.transformer,
                    blocks=pipe.transformer.blocks,
                    blocks_name="blocks",
                    dummy_blocks_names=[],
                ),
                forward_pattern=ForwardPattern.Pattern_3,
            )

        elif pipe_cls_name.startswith("Lumina"):
            from diffusers import LuminaNextDiT2DModel

            assert isinstance(pipe.transformer, LuminaNextDiT2DModel)
            return UnifiedCacheParams(
                block_adapter=BlockAdapter(
                    pipe=pipe,
                    transformer=pipe.transformer,
                    blocks=pipe.transformer.layers,
                    blocks_name="layers",
                    dummy_blocks_names=[],
                ),
                forward_pattern=ForwardPattern.Pattern_3,
            )

        elif pipe_cls_name.startswith("Lumina2"):
            from diffusers import Lumina2Transformer2DModel

            assert isinstance(pipe.transformer, Lumina2Transformer2DModel)
            return UnifiedCacheParams(
                block_adapter=BlockAdapter(
                    pipe=pipe,
                    transformer=pipe.transformer,
                    blocks=pipe.transformer.layers,
                    blocks_name="layers",
                    dummy_blocks_names=[],
                ),
                forward_pattern=ForwardPattern.Pattern_3,
            )

        elif pipe_cls_name.startswith("OmniGen"):
            from diffusers import OmniGenTransformer2DModel

            assert isinstance(pipe.transformer, OmniGenTransformer2DModel)
            return UnifiedCacheParams(
                block_adapter=BlockAdapter(
                    pipe=pipe,
                    transformer=pipe.transformer,
                    blocks=pipe.transformer.layers,
                    blocks_name="layers",
                    dummy_blocks_names=[],
                ),
                forward_pattern=ForwardPattern.Pattern_3,
            )

        elif pipe_cls_name.startswith("PixArt"):
            from diffusers import PixArtTransformer2DModel

            assert isinstance(pipe.transformer, PixArtTransformer2DModel)
            return UnifiedCacheParams(
                block_adapter=BlockAdapter(
                    pipe=pipe,
                    transformer=pipe.transformer,
                    blocks=pipe.transformer.transformer_blocks,
                    blocks_name="transformer_blocks",
                    dummy_blocks_names=[],
                ),
                forward_pattern=ForwardPattern.Pattern_3,
            )

        elif pipe_cls_name.startswith("Sana"):
            from diffusers import SanaTransformer2DModel

            assert isinstance(pipe.transformer, SanaTransformer2DModel)
            return UnifiedCacheParams(
                block_adapter=BlockAdapter(
                    pipe=pipe,
                    transformer=pipe.transformer,
                    blocks=pipe.transformer.transformer_blocks,
                    blocks_name="transformer_blocks",
                    dummy_blocks_names=[],
                ),
                forward_pattern=ForwardPattern.Pattern_3,
            )

        elif pipe_cls_name.startswith("ShapE"):
            from diffusers import PriorTransformer

            assert isinstance(pipe.prior, PriorTransformer)
            return UnifiedCacheParams(
                block_adapter=BlockAdapter(
                    pipe=pipe,
                    transformer=pipe.prior,
                    blocks=pipe.prior.transformer_blocks,
                    blocks_name="transformer_blocks",
                    dummy_blocks_names=[],
                ),
                forward_pattern=ForwardPattern.Pattern_3,
            )

        elif pipe_cls_name.startswith("StableAudio"):
            from diffusers import StableAudioDiTModel

            assert isinstance(pipe.transformer, StableAudioDiTModel)
            return UnifiedCacheParams(
                block_adapter=BlockAdapter(
                    pipe=pipe,
                    transformer=pipe.transformer,
                    blocks=pipe.transformer.transformer_blocks,
                    blocks_name="transformer_blocks",
                    dummy_blocks_names=[],
                ),
                forward_pattern=ForwardPattern.Pattern_3,
            )

        elif pipe_cls_name.startswith("VisualCloze"):
            from diffusers import FluxTransformer2DModel
            from cache_dit.cache_factory.patch_functors import FluxPatchFunctor

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
                    patch_functor=FluxPatchFunctor(),
                ),
                forward_pattern=ForwardPattern.Pattern_1,
            )

        elif pipe_cls_name.startswith("AuraFlow"):
            from diffusers import AuraFlowTransformer2DModel

            assert isinstance(pipe.transformer, AuraFlowTransformer2DModel)
            return UnifiedCacheParams(
                block_adapter=BlockAdapter(
                    pipe=pipe,
                    transformer=pipe.transformer,
                    # Only support caching single_transformer_blocks for AuraFlow now.
                    # TODO: Support AuraFlowPatchFunctor.
                    blocks=pipe.transformer.single_transformer_blocks,
                    blocks_name="single_transformer_blocks",
                    dummy_blocks_names=[],
                ),
                forward_pattern=ForwardPattern.Pattern_3,
            )

        elif pipe_cls_name.startswith("Chroma"):
            from diffusers import ChromaTransformer2DModel
            from cache_dit.cache_factory.patch_functors import (
                ChromaPatchFunctor,
            )

            assert isinstance(pipe.transformer, ChromaTransformer2DModel)
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
                    patch_functor=ChromaPatchFunctor(),
                ),
                forward_pattern=ForwardPattern.Pattern_1,
            )

        elif pipe_cls_name.startswith("HiDream"):
            from diffusers import HiDreamImageTransformer2DModel

            assert isinstance(pipe.transformer, HiDreamImageTransformer2DModel)
            return UnifiedCacheParams(
                block_adapter=BlockAdapter(
                    pipe=pipe,
                    transformer=pipe.transformer,
                    # Only support caching single_stream_blocks for HiDream now.
                    # TODO: Support HiDreamPatchFunctor.
                    blocks=pipe.transformer.single_stream_blocks,
                    blocks_name="single_stream_blocks",
                    dummy_blocks_names=[],
                ),
                forward_pattern=ForwardPattern.Pattern_3,
            )

        else:
            raise ValueError(f"Unknown pipeline class name: {pipe_cls_name}")

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
            # Apply cache on pipeline: wrap cache context
            cls.create_context(
                block_adapter.pipe,
                **cache_context_kwargs,
            )
            # Apply cache on transformer: mock cached transformer blocks
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

    @classmethod
    def check_context_kwargs(cls, pipe, **cache_context_kwargs):
        # Check cache_context_kwargs
        if not cache_context_kwargs["do_separate_cfg"]:
            # Check cfg for some specific case if users don't set it as True
            cache_context_kwargs["do_separate_cfg"] = (
                BlockAdapterRegistry.has_separate_cfg(pipe)
            )

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
    ) -> torch.nn.Module:

        if getattr(block_adapter.transformer, "_is_cached", False):
            return block_adapter.transformer

        # Check block forward pattern matching
        assert BlockAdapter.match_blocks_pattern(
            block_adapter.blocks,
            forward_pattern=block_adapter.forward_pattern,
        ), (
            "No block forward pattern matched, "
            f"supported lists: {ForwardPattern.supported_patterns()}"
        )

        # Apply cache on transformer: mock cached transformer blocks
        cached_blocks = torch.nn.ModuleList(
            [
                DBCachedBlocks(
                    block_adapter.blocks,
                    transformer=block_adapter.transformer,
                    forward_pattern=block_adapter.forward_pattern,
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
