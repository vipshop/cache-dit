import functools
import unittest

import torch
from diffusers import DiffusionPipeline, FluxTransformer2DModel

from cache_dit.cache_factory.patch.flux import maybe_patch_flux_transformer
from cache_dit.cache_factory.dynamic_block_prune import (
    prune_context,
    DBPrunedTransformerBlocks,
)


def apply_db_prune_on_transformer(
    transformer: FluxTransformer2DModel,
):
    if getattr(transformer, "_is_pruned", False):
        return transformer

    transformer = maybe_patch_flux_transformer(transformer)

    cached_transformer_blocks = torch.nn.ModuleList(
        [
            DBPrunedTransformerBlocks(
                transformer.transformer_blocks
                + transformer.single_transformer_blocks,
                transformer=transformer,
                return_hidden_states_first=False,
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
                cached_transformer_blocks,
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
    pipe: DiffusionPipeline,
    *,
    residual_diff_threshold=0.05,
    downsample_factor=1,
    warmup_steps=0,
    max_pruned_steps=-1,
    **kwargs,
):
    cache_kwargs, kwargs = prune_context.collect_prune_kwargs(
        default_attrs={
            "residual_diff_threshold": residual_diff_threshold,
            "downsample_factor": downsample_factor,
            "warmup_steps": warmup_steps,
            "max_pruned_steps": max_pruned_steps,
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

    apply_db_prune_on_transformer(pipe.transformer)

    return pipe
