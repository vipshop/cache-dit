import torch

from typing import Optional
from cache_dit.logger import init_logger

from diffusers.models.modeling_utils import ModelMixin

from cache_dit.parallelism.backend import ParallelismBackend
from cache_dit.parallelism.config import ParallelismConfig

logger = init_logger(__name__)


def maybe_enable_parallelism_for_transformer(
    transformer: torch.nn.Module | ModelMixin,
    parallelism_config: Optional[ParallelismConfig],
) -> torch.nn.Module:
    assert isinstance(transformer, (torch.nn.Module, ModelMixin)), (
        "transformer must be an instance of torch.nn.Module or ModelMixin, "
        f"but got {type(transformer)}"
    )

    if parallelism_config is None:
        return transformer

    # Currently, we can dispatch the parallelism based on the backend type.
    if parallelism_config.backend == ParallelismBackend.NATIVE_HYBRID:
        return maybe_enable_hybrid_parallelism_for_transformer(
            transformer=transformer,
            parallelism_config=parallelism_config,
        )
    elif parallelism_config.backend == ParallelismBackend.NATIVE_DIFFUSER:
        return maybe_enable_context_parallelism_for_transformer(
            transformer=transformer,
            parallelism_config=parallelism_config,
        )
    elif parallelism_config.backend == ParallelismBackend.NATIVE_PYTORCH:
        return maybe_enable_tensor_parallelism_for_transformer(
            transformer=transformer,
            parallelism_config=parallelism_config,
        )
    else:
        raise ValueError(f"{parallelism_config.backend} backend is not supported yet")


def maybe_enable_hybrid_parallelism_for_transformer(
    transformer: torch.nn.Module | ModelMixin,
    parallelism_config: ParallelismConfig,
):
    assert parallelism_config.hybrid_enabled(), "hybrid_enabled() must be True for HYBRID backend."
    # 0. First enable context parallelism
    transformer = maybe_enable_context_parallelism_for_transformer(
        transformer=transformer,
        parallelism_config=parallelism_config,
    )
    # 1. Then enable tensor parallelism
    transformer = maybe_enable_tensor_parallelism_for_transformer(
        transformer=transformer,
        parallelism_config=parallelism_config,
    )
    transformer._is_parallelized = True  # type: ignore[attr-defined]
    # Use `parallelism` not `parallel` to avoid name conflict with diffusers.
    transformer._parallelism_config = parallelism_config  # type: ignore[attr-defined]
    logger.info(
        f"Parallelize Transformer: {transformer.__class__.__name__}, "
        f"id:{id(transformer)}, {parallelism_config.strify(True)}"
    )
    return transformer


def maybe_enable_context_parallelism_for_transformer(
    transformer: torch.nn.Module | ModelMixin,
    parallelism_config: ParallelismConfig,
) -> torch.nn.Module:
    assert isinstance(transformer, (torch.nn.Module, ModelMixin)), (
        "transformer must be an instance of torch.nn.Module or ModelMixin, "
        f"but got {type(transformer)}"
    )

    if parallelism_config is None:
        return transformer

    assert isinstance(parallelism_config, ParallelismConfig), (
        "parallelism_config must be an instance of ParallelismConfig"
        f" but got {type(parallelism_config)}"
    )

    # Ensure the backend is correct, NAITIVE_DIFFUSER or HYBRID
    assert parallelism_config.backend in (
        ParallelismBackend.NATIVE_DIFFUSER,
        ParallelismBackend.NATIVE_HYBRID,
    ), (
        "parallelism_config.backend must be ParallelismBackend.NATIVE_DIFFUSER "
        f"or ParallelismBackend.NATIVE_HYBRID but got {parallelism_config.backend}"
    )

    if parallelism_config.cp_enabled() or parallelism_config.hybrid_enabled():
        from .context_parallelism import maybe_enable_context_parallelism

        transformer = maybe_enable_context_parallelism(
            transformer,
            parallelism_config,
        )
        if not parallelism_config.hybrid_enabled():
            transformer._is_parallelized = True  # type: ignore[attr-defined]
            # Use `parallelism` not `parallel` to avoid name conflict with diffusers.
            transformer._parallelism_config = parallelism_config  # type: ignore[attr-defined]
            logger.info(
                f"Parallelize Transformer: {transformer.__class__.__name__}, "
                f"id:{id(transformer)}, {parallelism_config.strify(True)}"
            )
    else:
        raise ValueError(
            "NATIVE_DIFFUSER backend only support context parallelism now. "
            "Please set ulysses_size or ring_size in parallelism_config."
        )
    return transformer


def maybe_enable_tensor_parallelism_for_transformer(
    transformer: torch.nn.Module | ModelMixin,
    parallelism_config: ParallelismConfig,
) -> torch.nn.Module:
    assert isinstance(transformer, (torch.nn.Module, ModelMixin)), (
        "transformer must be an instance of torch.nn.Module or ModelMixin, "
        f"but got {type(transformer)}"
    )

    if parallelism_config is None:
        return transformer

    # Ensure the backend is correct, NATIVE_PYTORCH or HYBRID
    assert parallelism_config.backend in (
        ParallelismBackend.NATIVE_PYTORCH,
        ParallelismBackend.NATIVE_HYBRID,
    ), (
        "parallelism_config.backend must be ParallelismBackend.NATIVE_PYTORCH "
        f"or ParallelismBackend.NATIVE_HYBRID but got {parallelism_config.backend}"
    )

    assert isinstance(parallelism_config, ParallelismConfig), (
        "parallelism_config must be an instance of ParallelismConfig"
        f" but got {type(parallelism_config)}"
    )

    if parallelism_config.tp_enabled() or parallelism_config.hybrid_enabled():
        from .tensor_parallelism import maybe_enable_tensor_parallelism

        transformer = maybe_enable_tensor_parallelism(
            transformer=transformer,
            parallelism_config=parallelism_config,
        )
        if not parallelism_config.hybrid_enabled():
            transformer._is_parallelized = True  # type: ignore[attr-defined]
            # Use `parallelism` not `parallel` to avoid name conflict with diffusers.
            transformer._parallelism_config = parallelism_config  # type: ignore[attr-defined]
            logger.info(
                f"Parallelize Transformer: {transformer.__class__.__name__}, "
                f"id:{id(transformer)}, {parallelism_config.strify(True)}"
            )
    else:
        raise ValueError(
            "NATIVE_PYTORCH only supported tensor parallelism now. "
            "Please set tp_size > 1 for tensor parallelism."
        )
    return transformer
