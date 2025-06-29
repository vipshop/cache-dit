import functools
import unittest

import torch
from diffusers import DiffusionPipeline, WanTransformer3DModel

from cache_dit.cache_factory.dynamic_block_prune import prune_context


def apply_db_prune_on_transformer(
    transformer: WanTransformer3DModel,
):
    if getattr(transformer, "_is_pruned", False):
        return transformer

    blocks = torch.nn.ModuleList(
        [
            prune_context.DBPrunedTransformerBlocks(
                transformer.blocks,
                transformer=transformer,
                return_hidden_states_only=True,
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
            "blocks",
            blocks,
        ):
            return original_forward(
                *args,
                **kwargs,
            )

    transformer.forward = new_forward.__get__(transformer)

    transformer._is_pruned = True

    return transformer


def apply_cache_on_pipe(
    pipe: DiffusionPipeline,
    *,
    shallow_patch: bool = False,
    residual_diff_threshold=0.03,
    downsample_factor=1,
    # SLG is not supported in WAN with DBPrune yet
    # slg_layers=None,
    # slg_start: float = 0.0,
    # slg_end: float = 0.1,
    warmup_steps=0,
    max_cached_steps=-1,
    **kwargs,
):
    cache_kwargs, kwargs = prune_context.collect_prune_kwargs(
        default_attrs={
            "residual_diff_threshold": residual_diff_threshold,
            "downsample_factor": downsample_factor,
            # "enable_alter_cache": True,
            # "slg_layers": slg_layers,
            # "slg_start": slg_start,
            # "slg_end": slg_end,
            "num_inference_steps": kwargs.get("num_inference_steps", 50),
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
