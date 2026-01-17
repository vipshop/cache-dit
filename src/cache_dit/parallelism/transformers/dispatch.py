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
    # Now, The context parallelism is only supported in NATIVE_DIFFUSER backend,
    # and the tensor parallelism is only supported in NATIVE_PYTORCH backend.
    if parallelism_config.backend == ParallelismBackend.NATIVE_DIFFUSER:
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


def maybe_enable_context_parallelism_for_transformer(
    transformer: torch.nn.Module | ModelMixin,
    parallelism_config: Optional[ParallelismConfig],
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

    assert parallelism_config.backend == ParallelismBackend.NATIVE_DIFFUSER, (
        f"parallelism backend must be {ParallelismBackend.NATIVE_DIFFUSER}, "
        f"but got {parallelism_config.backend}"
    )

    if parallelism_config.ulysses_size is not None or parallelism_config.ring_size is not None:
        from .context_parallelism import maybe_enable_context_parallelism

        transformer = maybe_enable_context_parallelism(
            transformer,
            parallelism_config,
        )
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
    parallelism_config: Optional[ParallelismConfig],
) -> torch.nn.Module:
    assert isinstance(transformer, (torch.nn.Module, ModelMixin)), (
        "transformer must be an instance of torch.nn.Module or ModelMixin, "
        f"but got {type(transformer)}"
    )

    if parallelism_config is None:
        return transformer

    assert parallelism_config.backend == ParallelismBackend.NATIVE_PYTORCH, (
        "parallelism_config.backend must be ParallelismBackend.NATIVE_PYTORCH "
        f"but got {parallelism_config.backend}"
    )

    assert isinstance(parallelism_config, ParallelismConfig), (
        "parallelism_config must be an instance of ParallelismConfig"
        f" but got {type(parallelism_config)}"
    )
    assert parallelism_config.ulysses_size is None and parallelism_config.ring_size is None, (
        "Ulysses/Ring parallelism is not supported in Native_PyTorch backend. "
        "Please set it to None in parallelism_config."
    )

    if parallelism_config.tp_size is not None and parallelism_config.tp_size > 1:
        from .tensor_parallelism import maybe_enable_tensor_parallelism

        transformer = maybe_enable_tensor_parallelism(
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

    else:
        raise ValueError(
            "NATIVE_PYTORCH only supported tensor parallelism now. "
            "Please set tp_size > 1 for tensor parallelism."
        )
    return transformer
