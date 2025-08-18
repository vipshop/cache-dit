import importlib
import functools

from diffusers import DiffusionPipeline
from cache_dit.cache_factory.dynamic_block_prune import prune_context


def apply_db_prune_on_transformer(
    transformer,
    *,
    residual_diff_threshold=0.05,
    downsample_factor=1,
    warmup_steps=0,
    max_pruned_steps=-1,
    **kwargs,
):
    if getattr(transformer, "_is_pruned", False):
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
    apply_db_prune_on_transformer_fn = getattr(
        adapter_module, "apply_db_prune_on_transformer"
    )
    # mock transformer_blocks -> pruned_transformer_blocks
    transformer = apply_db_prune_on_transformer_fn(transformer)

    # create prune context
    prune_kwargs, kwargs = prune_context.collect_prune_kwargs(
        default_attrs={
            "residual_diff_threshold": residual_diff_threshold,
            "downsample_factor": downsample_factor,
            "warmup_steps": warmup_steps,
            "max_pruned_steps": max_pruned_steps,
        },
        **kwargs,
    )
    assert getattr(transformer, "_is_pruned", False)

    pruned_forward = transformer.forward

    @functools.wraps(pruned_forward)
    def new_pruned_forward(
        self,
        *args,
        **kwargs,
    ):
        with prune_context.prune_context(
            prune_context.create_prune_context(
                **prune_kwargs,
            )
        ):
            return pruned_forward(
                *args,
                **kwargs,
            )

    transformer.forward = new_pruned_forward.__get__(transformer)

    return transformer


def apply_db_prune_on_pipe(
    pipe: DiffusionPipeline,
    *args,
    **kwargs,
):
    assert isinstance(pipe, DiffusionPipeline)
    if getattr(pipe, "_is_pruned", False):
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
    apply_db_prune_on_pipe_fn = getattr(
        adapter_module, "apply_db_prune_on_pipe"
    )
    return apply_db_prune_on_pipe_fn(pipe, *args, **kwargs)
