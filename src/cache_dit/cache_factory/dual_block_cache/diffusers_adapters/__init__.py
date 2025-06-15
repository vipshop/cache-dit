import importlib

from diffusers import DiffusionPipeline


def apply_db_cache_on_transformer(transformer, *args, **kwargs):
    transformer_cls_name: str = transformer.__class__.__name__
    if transformer_cls_name.startswith("Flux"):
        adapter_name = "flux"
    elif transformer_cls_name.startswith("Mochi"):
        adapter_name = "mochi"
    elif transformer_cls_name.startswith("CogVideoX"):
        adapter_name = "cogvideox"
    else:
        raise ValueError(
            f"Unknown transformer class name: {transformer_cls_name}"
        )

    adapter_module = importlib.import_module(f".{adapter_name}", __package__)
    apply_db_cache_on_transformer_fn = getattr(
        adapter_module, "apply_db_cache_on_transformer"
    )
    return apply_db_cache_on_transformer_fn(transformer, *args, **kwargs)


def apply_db_cache_on_pipe(pipe: DiffusionPipeline, *args, **kwargs):
    assert isinstance(pipe, DiffusionPipeline)

    pipe_cls_name: str = pipe.__class__.__name__
    if pipe_cls_name.startswith("Flux"):
        adapter_name = "flux"
    elif pipe_cls_name.startswith("Mochi"):
        adapter_name = "mochi"
    elif pipe_cls_name.startswith("CogVideoX"):
        adapter_name = "cogvideox"
    else:
        raise ValueError(f"Unknown pipeline class name: {pipe_cls_name}")

    adapter_module = importlib.import_module(f".{adapter_name}", __package__)
    apply_db_cache_on_pipe_fn = getattr(
        adapter_module, "apply_db_cache_on_pipe"
    )
    return apply_db_cache_on_pipe_fn(pipe, *args, **kwargs)
