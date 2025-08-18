import functools
import unittest

import torch
from diffusers import HunyuanVideoPipeline, HunyuanVideoTransformer3DModel

from cache_dit.cache_factory.dynamic_block_prune import (
    prune_context,
    DBPrunedTransformerBlocks,
)
from cache_dit.logger import init_logger


logger = init_logger(__name__)


def apply_db_prune_on_transformer(
    transformer: HunyuanVideoTransformer3DModel,
):
    if getattr(transformer, "_is_pruned", False):
        return transformer

    pruned_transformer_blocks = torch.nn.ModuleList(
        [
            DBPrunedTransformerBlocks(
                transformer.transformer_blocks
                + transformer.single_transformer_blocks,
                transformer=transformer,
            )
        ]
    )
    dummy_single_transformer_blocks = torch.nn.ModuleList()

    original_forward = transformer.forward

    @functools.wraps(original_forward)
    def new_forward(
        self,
        *args,
        **kwargs,
    ):
        with (
            unittest.mock.patch.object(
                self,
                "transformer_blocks",
                pruned_transformer_blocks,
            ),
            unittest.mock.patch.object(
                self,
                "single_transformer_blocks",
                dummy_single_transformer_blocks,
            ),
        ):
            return original_forward(
                *args,
                **kwargs,
            )

    transformer.forward = new_forward.__get__(transformer)
    transformer._is_pruned = True

    return transformer


def apply_db_prune_on_pipe(
    pipe: HunyuanVideoPipeline,
    *,
    shallow_patch: bool = False,
    residual_diff_threshold=0.06,
    downsample_factor=1,
    warmup_steps=0,
    max_cached_steps=-1,
    **kwargs,
):
    cache_kwargs, kwargs = prune_context.collect_prune_kwargs(
        default_attrs={
            "residual_diff_threshold": residual_diff_threshold,
            "downsample_factor": downsample_factor,
            "warmup_steps": warmup_steps,
            "max_cached_steps": max_cached_steps,
        },
        **kwargs,
    )
    if not getattr(pipe, "_is_pruned", False):
        original_call = pipe.__class__.__call__

        @functools.wraps(original_call)
        def new_call(self, *args, **kwargs):
            with prune_context.prune_context(
                prune_context.create_prune_context(
                    **cache_kwargs,
                )
            ):
                return original_call(self, *args, **kwargs)

        pipe.__class__.__call__ = new_call
        pipe.__class__._is_pruned = True

    if not shallow_patch:
        apply_db_prune_on_transformer(pipe.transformer, **kwargs)

    return pipe
