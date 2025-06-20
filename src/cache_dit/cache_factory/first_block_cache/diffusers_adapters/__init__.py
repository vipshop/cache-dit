# Adapted from: https://github.com/chengzeyi/ParaAttention/tree/main/src/para_attn/first_block_cache/__init__.py

import importlib
from typing import Callable

from diffusers import DiffusionPipeline


def apply_fb_cache_on_transformer(transformer, *args, **kwargs):
    transformer_cls_name: str = transformer.__class__.__name__
    if transformer_cls_name.startswith("Flux"):
        adapter_name = "flux"
    elif transformer_cls_name.startswith("Mochi"):
        adapter_name = "mochi"
    elif transformer_cls_name.startswith("CogVideoX"):
        adapter_name = "cogvideox"
    elif transformer_cls_name.startswith("Wan"):
        adapter_name = "wan"
    elif transformer_cls_name.startswith("HunyuanVideo"):
        adapter_name = "hunyuan_video"
    else:
        raise ValueError(
            f"Unknown transformer class name: {transformer_cls_name}"
        )

    adapter_module = importlib.import_module(f".{adapter_name}", __package__)
    apply_cache_on_transformer_fn = getattr(
        adapter_module, "apply_cache_on_transformer"
    )
    return apply_cache_on_transformer_fn(transformer, *args, **kwargs)


def apply_fb_cache_on_pipe(pipe: DiffusionPipeline, *args, **kwargs):
    assert isinstance(pipe, DiffusionPipeline)

    pipe_cls_name: str = pipe.__class__.__name__
    if pipe_cls_name.startswith("Flux"):
        adapter_name = "flux"
    elif pipe_cls_name.startswith("Mochi"):
        adapter_name = "mochi"
    elif pipe_cls_name.startswith("CogVideoX"):
        adapter_name = "cogvideox"
    elif pipe_cls_name.startswith("Wan"):
        adapter_name = "wan"
    elif pipe_cls_name.startswith("HunyuanVideo"):
        adapter_name = "hunyuan_video"
    else:
        raise ValueError(f"Unknown pipeline class name: {pipe_cls_name}")

    adapter_module = importlib.import_module(f".{adapter_name}", __package__)
    apply_cache_on_pipe_fn = getattr(adapter_module, "apply_cache_on_pipe")
    return apply_cache_on_pipe_fn(pipe, *args, **kwargs)


# re-export functions for compatibility
apply_cache_on_transformer: Callable = apply_fb_cache_on_transformer
apply_cache_on_pipe: Callable = apply_fb_cache_on_pipe
