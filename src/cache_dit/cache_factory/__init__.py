import inspect
import unittest
import functools
from contextlib import ExitStack

import torch
from diffusers import DiffusionPipeline
from cache_dit.cache_factory.adapters import CacheType
from cache_dit.cache_factory.adapters import apply_cache_on_pipe
from cache_dit.cache_factory.utils import load_cache_options_from_yaml
from cache_dit.cache_factory.patch.flux import (
    maybe_patch_flux_transformer,
)
from cache_dit.cache_factory.dual_block_cache import (
    cache_context,
    DBCachedTransformerBlocks,
)
from cache_dit.logger import init_logger

logger = init_logger(__name__)


def make_pattern(in_params: list, out_params: list) -> dict:
    return {"IN": in_params, "OUT": out_params}


def supported_patterns():
    # TODO: support more cache patterns.
    return [
        make_pattern(
            ["hidden_states", "encoder_hidden_states"],
            ["hidden_states", "encoder_hidden_states"],
        ),
    ]


def load_options(path: str):
    """cache_dit.load_options(cache_config.yaml)"""
    return load_cache_options_from_yaml(path)


def match_pattern(transformer_blocks: torch.nn.ModuleList) -> bool:
    pattern_matched = True
    pattern_ids = []
    for block in transformer_blocks:
        forward_parameters = set(
            inspect.signature(block.forward).parameters.keys()
        )

        matched_pattern_id = None
        for j, pattern in enumerate(supported_patterns()):
            param_matched = True
            for required_param in pattern["IN"]:
                if required_param not in forward_parameters:
                    param_matched = False
                    break
            if param_matched:
                matched_pattern_id = j  # last pattern
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
            pattern = supported_patterns()[pattern_id]
            logger.info(
                f"Match cache pattern: IN ({pattern['IN']}),"
                f"OUT ({pattern['OUT']})"
            )

    return pattern_matched


def enable_cache(
    pipe: DiffusionPipeline,
    *,
    transformer: torch.nn.Module = None,
    blocks: torch.nn.ModuleList = None,
    blocks_name: str = "transformer_blocks",
    dummy_blocks_names: list[str] = [],
    # (encoder_hidden_states, hidden_states) or
    # (hidden_states, encoder_hidden_states)
    return_hidden_states_first: bool = False,
    **cache_options_kwargs,
) -> DiffusionPipeline:
    if isinstance(pipe, DiffusionPipeline) and (
        transformer is None
        or blocks is None
        or not isinstance(blocks, torch.nn.ModuleList)
    ):
        return apply_cache_on_pipe(pipe, **cache_options_kwargs)
    elif (
        isinstance(pipe, DiffusionPipeline)
        and transformer is not None
        and blocks is not None
        and isinstance(blocks, torch.nn.ModuleList)
    ):
        # support custom cache setting for models that match the
        # supported block forward patterns.
        logger.info(
            f"Using custom cache setting for pipe: {pipe.__class__.__name__}, "
            f"transfomer: {transformer.__class__.__name__}"
        )
        if getattr(pipe, "_is_cached", False) or getattr(
            transformer, "_is_cached", False
        ):
            return pipe

        assert isinstance(blocks, torch.nn.ModuleList)

        assert (
            cache_options_kwargs.pop("cache_type", CacheType.NONE)
            == CacheType.DBCache
        ), "Custom cache setting if only support for DBCache now!"

        # Apply cache on pipeline: wrap cache context
        original_call = pipe.__class__.__call__

        @functools.wraps(original_call)
        def new_call(self, *args, **kwargs):
            with cache_context.cache_context(
                cache_context.create_cache_context(
                    **cache_options_kwargs,
                )
            ):
                return original_call(self, *args, **kwargs)

        pipe.__class__.__call__ = new_call
        pipe.__class__._is_cached = True

        # Apply cache on transformer: mock cached transformer blocks
        # process some specificial cases
        if transformer.__class__.__name__.startswith("Flux"):
            transformer = maybe_patch_flux_transformer(
                transformer,
                blocks=blocks,
            )

        assert match_pattern(blocks), (
            "No block forward pattern matched, "
            f"supported lists: {supported_patterns()}"
        )

        cached_blocks = torch.nn.ModuleList(
            [
                DBCachedTransformerBlocks(
                    blocks,
                    transformer=transformer,
                    return_hidden_states_first=return_hidden_states_first,
                )
            ]
        )
        dummy_blocks = torch.nn.ModuleList()

        original_forward = transformer.forward

        assert isinstance(dummy_blocks_names, list)

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

        return pipe
