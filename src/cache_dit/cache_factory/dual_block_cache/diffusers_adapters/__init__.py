import importlib

from diffusers import DiffusionPipeline


def apply_db_cache_on_transformer(
    transformer,
):
    if getattr(transformer, "_is_cached", False):
        return transformer

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
    elif transformer_cls_name.startswith("QwenImage"):
        adapter_name = "qwen_image"
    else:
        raise ValueError(
            f"Unknown transformer class name: {transformer_cls_name}"
        )

    adapter_module = importlib.import_module(f".{adapter_name}", __package__)
    apply_db_cache_on_transformer_fn = getattr(
        adapter_module, "apply_db_cache_on_transformer"
    )

    # mock transformer_blocks -> cached_transformer_blocks
    return apply_db_cache_on_transformer_fn(transformer)


def apply_db_cache_on_pipe(
    pipe: DiffusionPipeline,
    *args,
    **kwargs,
):
    assert isinstance(pipe, DiffusionPipeline)
    if getattr(pipe, "_is_cached", False):
        return pipe

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
    elif pipe_cls_name.startswith("QwenImage"):
        adapter_name = "qwen_image"
    else:
        raise ValueError(f"Unknown pipeline class name: {pipe_cls_name}")

    adapter_module = importlib.import_module(f".{adapter_name}", __package__)
    apply_db_cache_on_pipe_fn = getattr(
        adapter_module, "apply_db_cache_on_pipe"
    )
    return apply_db_cache_on_pipe_fn(pipe, *args, **kwargs)
