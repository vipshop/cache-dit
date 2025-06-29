import functools
import unittest

import torch
from diffusers import CogVideoXTransformer3DModel, DiffusionPipeline

from cache_dit.cache_factory.dynamic_block_prune import prune_context


def apply_db_prune_on_transformer(
    transformer: CogVideoXTransformer3DModel,
):
    if getattr(transformer, "_is_pruned", False):
        return transformer

    cached_transformer_blocks = torch.nn.ModuleList(
        [
            prune_context.DBPrunedTransformerBlocks(
                transformer.transformer_blocks,
                transformer=transformer,
            )
        ]
    )

    original_forward = transformer.forward

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        *args,
        **kwargs,
    ):
        with unittest.mock.patch.object(
            self,
            "transformer_blocks",
            cached_transformer_blocks,
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
    shallow_patch: bool = False,
    residual_diff_threshold=0.04,
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

    if not shallow_patch:
        apply_db_prune_on_transformer(pipe.transformer, **kwargs)

    return pipe
