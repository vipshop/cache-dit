import torch

import inspect
import unittest
import functools
import dataclasses

from enum import Enum
from contextlib import ExitStack
from diffusers import DiffusionPipeline
from cache_dit.cache_factory.patch.flux import (
    maybe_patch_flux_transformer,
)
from cache_dit.cache_factory.cache_blocks import (
    cache_context,
    DBCachedTransformerBlocks,
)
from cache_dit.logger import init_logger

logger = init_logger(__name__)


class CacheType(Enum):
    NONE = "NONE"
    DBCache = "Dual_Block_Cache"

    @staticmethod
    def type(cache_type: "CacheType | str") -> "CacheType":
        if isinstance(cache_type, CacheType):
            return cache_type
        return CacheType.cache_type(cache_type)

    @staticmethod
    def cache_type(cache_type: "CacheType | str") -> "CacheType":
        if cache_type is None:
            return CacheType.NONE

        if isinstance(cache_type, CacheType):
            return cache_type

        elif cache_type.lower() in (
            "dual_block_cache",
            "db_cache",
            "dbcache",
            "db",
        ):
            return CacheType.DBCache
        elif cache_type.lower() in (
            "none_cache",
            "nonecache",
            "no_cache",
            "nocache",
            "none",
            "no",
        ):
            return CacheType.NONE
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")

    @staticmethod
    def block_range(start: int, end: int, step: int = 1) -> list[int]:
        if start > end or end <= 0 or step <= 1:
            return []
        # Always compute 0 and end - 1 blocks for DB Cache
        return list(
            sorted(set([0] + list(range(start, end, step)) + [end - 1]))
        )

    @staticmethod
    def default_options(cache_type: "CacheType | str") -> dict:
        _no_options = {
            "cache_type": CacheType.NONE,
        }

        _Fn_compute_blocks = 8
        _Bn_compute_blocks = 0

        _db_options = {
            "cache_type": CacheType.DBCache,
            "residual_diff_threshold": 0.12,
            "warmup_steps": 8,
            "max_cached_steps": -1,  # -1 means no limit
            "Fn_compute_blocks": _Fn_compute_blocks,
            "Bn_compute_blocks": _Bn_compute_blocks,
            "max_Fn_compute_blocks": 16,
            "max_Bn_compute_blocks": 16,
            "Fn_compute_blocks_ids": [],  # 0, 1, 2, ..., 7, etc.
            "Bn_compute_blocks_ids": [],  # 0, 1, 2, ..., 7, etc.
        }

        if cache_type == CacheType.DBCache:
            return _db_options
        elif cache_type == CacheType.NONE:
            return _no_options
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")


@dataclasses.dataclass
class UnifiedCacheParams:
    pipe: DiffusionPipeline = None
    transformer: torch.nn.Module = None
    blocks: torch.nn.ModuleList = None
    # transformer_blocks, blocks, etc.
    blocks_name: str = None
    dummy_blocks_names: list[str] = dataclasses.field(default_factory=list)
    return_hidden_states_first: bool = True
    return_hidden_states_only: bool = False


class UnifiedCacheAdapter:
    _supported_pipelines = [
        "Flux",
        "Mochi",
        "CogVideoX",
        "Wan",
        "HunyuanVideo",
        "QwenImage",
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
                pipe=pipe,
                transformer=pipe.transformer,
                blocks=(
                    pipe.transformer.transformer_blocks
                    + pipe.transformer.single_transformer_blocks
                ),
                blocks_name="transformer_blocks",
                dummy_blocks_names=["single_transformer_blocks"],
                return_hidden_states_first=False,
                return_hidden_states_only=False,
            )
        elif pipe_cls_name.startswith("Mochi"):
            from diffusers import MochiTransformer3DModel

            assert isinstance(pipe.transformer, MochiTransformer3DModel)
            return UnifiedCacheParams(
                pipe=pipe,
                transformer=pipe.transformer,
                blocks=pipe.transformer.transformer_blocks,
                blocks_name="transformer_blocks",
                dummy_blocks_names=[],
                return_hidden_states_first=True,
                return_hidden_states_only=False,
            )
        elif pipe_cls_name.startswith("CogVideoX"):
            from diffusers import CogVideoXTransformer3DModel

            assert isinstance(pipe.transformer, CogVideoXTransformer3DModel)
            return UnifiedCacheParams(
                pipe=pipe,
                transformer=pipe.transformer,
                blocks=pipe.transformer.transformer_blocks,
                blocks_name="transformer_blocks",
                dummy_blocks_names=[],
                return_hidden_states_first=True,
                return_hidden_states_only=False,
            )
        elif pipe_cls_name.startswith("Wan"):
            from diffusers import WanTransformer3DModel

            assert isinstance(pipe.transformer, WanTransformer3DModel)
            return UnifiedCacheParams(
                pipe=pipe,
                transformer=pipe.transformer,
                blocks=pipe.transformer.blocks,
                blocks_name="blocks",
                dummy_blocks_names=[],
                return_hidden_states_first=True,
                return_hidden_states_only=True,
            )
        elif pipe_cls_name.startswith("HunyuanVideo"):
            from diffusers import HunyuanVideoTransformer3DModel

            assert isinstance(pipe.transformer, HunyuanVideoTransformer3DModel)
            return UnifiedCacheParams(
                pipe=pipe,
                blocks=(
                    pipe.transformer.transformer_blocks
                    + pipe.transformer.single_transformer_blocks
                ),
                blocks_name="transformer_blocks",
                dummy_blocks_names=["single_transformer_blocks"],
                return_hidden_states_first=True,
                return_hidden_states_only=False,
            )
        elif pipe_cls_name.startswith("QwenImage"):
            from diffusers import QwenImageTransformer2DModel

            assert isinstance(pipe.transformer, QwenImageTransformer2DModel)
            return UnifiedCacheParams(
                pipe=pipe,
                transformer=pipe.transformer,
                blocks=pipe.transformer.transformer_blocks,
                blocks_name="transformer_blocks",
                dummy_blocks_names=[],
                return_hidden_states_first=False,
                return_hidden_states_only=False,
            )
        else:
            raise ValueError(f"Unknown pipeline class name: {pipe_cls_name}")

    @classmethod
    def apply(
        cls,
        pipe: DiffusionPipeline,
        *,
        transformer: torch.nn.Module = None,
        blocks: torch.nn.ModuleList = None,
        # transformer_blocks, blocks, etc.
        blocks_name: str = None,
        dummy_blocks_names: list[str] = [],
        return_hidden_states_first: bool = True,
        return_hidden_states_only: bool = False,
        **cache_context_kwargs,
    ) -> DiffusionPipeline:
        if cls.is_supported(pipe) and (transformer is None or blocks is None):
            params = cls.get_params(pipe)
            return cls.cachify(
                params.pipe,
                params.transformer,
                params.blocks,
                blocks_name=params.blocks_name,
                dummy_blocks_names=params.dummy_blocks_names,
                return_hidden_states_first=params.return_hidden_states_first,
                return_hidden_states_only=params.return_hidden_states_only,
                **cache_context_kwargs,
            )
        else:
            return cls.cachify(
                pipe,
                transformer,
                blocks,
                blocks_name=blocks_name,
                dummy_blocks_names=dummy_blocks_names,
                return_hidden_states_first=return_hidden_states_first,
                return_hidden_states_only=return_hidden_states_only,
                **cache_context_kwargs,
            )

    @classmethod
    def cachify(
        cls,
        pipe: DiffusionPipeline,
        transformer: torch.nn.Module,
        blocks: torch.nn.ModuleList,
        *,
        # transformer_blocks, blocks, etc.
        blocks_name: str = None,
        dummy_blocks_names: list[str] = [],
        # (encoder_hidden_states, hidden_states) or
        # (hidden_states, encoder_hidden_states)
        return_hidden_states_first: bool = True,
        return_hidden_states_only: bool = False,
        **cache_context_kwargs,
    ) -> DiffusionPipeline:
        if (
            isinstance(pipe, DiffusionPipeline)
            and transformer is not None
            and blocks is not None
            and isinstance(blocks, torch.nn.ModuleList)
        ):
            assert isinstance(blocks, torch.nn.ModuleList)

            # Apply cache on pipeline: wrap cache context
            cls.create_context(pipe, **cache_context_kwargs)
            # Apply cache on transformer: mock cached transformer blocks
            cls.mock_blocks(
                transformer,
                blocks,
                blocks_name=blocks_name,
                dummy_blocks_names=dummy_blocks_names,
                return_hidden_states_first=return_hidden_states_first,
                return_hidden_states_only=return_hidden_states_only,
            )

        return pipe

    @classmethod
    def create_context(
        cls,
        pipe: DiffusionPipeline,
        **cache_context_kwargs,
    ) -> DiffusionPipeline:
        if getattr(pipe, "_is_cached", False):
            return pipe

        # Check cache_context_kwargs
        if not cache_context_kwargs:
            logger.warning(
                "cache_context_kwargs is empty, use default cache options!"
            )
            cache_context_kwargs = CacheType.default_options(CacheType.DBCache)

        if cache_type := cache_context_kwargs.pop("cache_type", None):
            assert (
                cache_type == CacheType.DBCache
            ), "Custom cache setting only support for DBCache now!"

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
        transformer: torch.nn.Module,
        blocks: torch.nn.ModuleList,
        blocks_name: str = None,
        dummy_blocks_names: list[str] = [],
        return_hidden_states_first: bool = True,
        return_hidden_states_only: bool = False,
    ) -> torch.nn.Module:
        if getattr(transformer, "_is_cached", False):
            return transformer

        # Firstly, process some specificial cases (TODO: more patches)
        if transformer.__class__.__name__.startswith("Flux"):
            transformer = maybe_patch_flux_transformer(
                transformer,
                blocks=blocks,
            )

        # Check block forward pattern matching
        assert cls.match_pattern(blocks), (
            "No block forward pattern matched, "
            f"supported lists: {cls.supported_patterns()}"
        )

        # Apply cache on transformer: mock cached transformer blocks
        cached_blocks = torch.nn.ModuleList(
            [
                DBCachedTransformerBlocks(
                    blocks,
                    transformer=transformer,
                    return_hidden_states_first=return_hidden_states_first,
                    return_hidden_states_only=return_hidden_states_only,
                )
            ]
        )
        dummy_blocks = torch.nn.ModuleList()

        original_forward = transformer.forward

        assert isinstance(dummy_blocks_names, list)
        if blocks_name is None:
            blocks_name = cls.find_blocks_name(transformer)
            assert blocks_name is not None

        @functools.wraps(original_forward)
        def new_forward(self, *args, **kwargs):
            with ExitStack() as stack:
                stack.enter_context(
                    unittest.mock.patch.object(
                        self,
                        blocks_name,
                        cached_blocks,
                    )
                )
                for dummy_name in dummy_blocks_names:
                    stack.enter_context(
                        unittest.mock.patch.object(
                            self,
                            dummy_name,
                            dummy_blocks,
                        )
                    )
                return original_forward(*args, **kwargs)

        transformer.forward = new_forward.__get__(transformer)
        transformer._is_cached = True

        return transformer

    @classmethod
    def make_pattern(cls, in_params: list, out_params: list) -> dict:
        return {"IN": in_params, "OUT": out_params}

    @classmethod
    def supported_patterns(cls):
        # TODO: support more cache patterns.
        return [
            cls.make_pattern(
                ["hidden_states", "encoder_hidden_states"],
                ["hidden_states", "encoder_hidden_states"],
            ),
            cls.make_pattern(
                ["hidden_states", "encoder_hidden_states"],
                ["hidden_states"],
            ),
        ]

    @classmethod
    def match_pattern(cls, transformer_blocks: torch.nn.ModuleList) -> bool:
        pattern_matched = True
        pattern_ids = []
        for block in transformer_blocks:
            forward_parameters = set(
                inspect.signature(block.forward).parameters.keys()
            )
            num_outputs = str(
                inspect.signature(block.forward).return_annotation
            ).count("torch.Tensor")

            matched_pattern_id = None
            param_matched = False
            for i, pattern in enumerate(cls.supported_patterns()):
                if param_matched:
                    break

                if num_outputs > 0 and len(pattern["OUT"]) != num_outputs:
                    # output pattern not match
                    break

                for required_param in pattern["IN"]:
                    if required_param not in forward_parameters:
                        break

                param_matched = True
                if param_matched:
                    matched_pattern_id = i  # first pattern

            if matched_pattern_id is not None:
                pattern_ids.append(matched_pattern_id)
            else:
                pattern_matched = False
                break

        if pattern_matched:
            unique_pattern_ids = set(pattern_ids)
            if len(unique_pattern_ids) > 1:
                pattern_matched = False
            else:
                pattern_id = list(unique_pattern_ids)[0]
                pattern = cls.supported_patterns()[pattern_id]
                logger.info(
                    f"Match cache pattern: IN({pattern['IN']}, OUT({pattern['OUT']}))"
                )

        return pattern_matched

    @classmethod
    def find_blocks_name(cls, transformer):
        blocks_name = None
        allow_prefixes = ["transformer", "blocks"]
        for attr_name in dir(transformer):
            if blocks_name is None:
                for prefix in allow_prefixes:
                    # transformer_blocks, blocks
                    if attr_name.startswith(prefix):
                        blocks_name = attr_name
                        logger.info(f"Auto selected blocks name: {blocks_name}")
                        break
        if blocks_name is None:
            logger.warning(
                "Auto selected blocks name failed, please set it manually."
            )
        return blocks_name
