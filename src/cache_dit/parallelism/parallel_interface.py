import torch
from cache_dit.parallelism.parallel_backend import ParallelismBackend
from cache_dit.parallelism.parallel_config import ParallelismConfig
from cache_dit.logger import init_logger

logger = init_logger(__name__)


def enable_parallelism(
    transformer: torch.nn.Module,
    parallelism_config: ParallelismConfig,
) -> torch.nn.Module:
    assert isinstance(transformer, torch.nn.Module), (
        "transformer must be an instance of torch.nn.Module, "
        f"but got {type(transformer)}"
    )
    if getattr(transformer, "_is_parallelized", False):
        logger.warning(
            "The transformer is already parallelized. "
            "Skipping parallelism enabling."
        )
        return transformer

    if parallelism_config.backend == ParallelismBackend.NATIVE_DIFFUSER:
        from cache_dit.parallelism.backends.native_diffusers import (
            maybe_enable_parallelism,
        )

        transformer = maybe_enable_parallelism(
            transformer,
            parallelism_config,
        )
    elif parallelism_config.backend == ParallelismBackend.NATIVE_PYTORCH:
        from cache_dit.parallelism.backends.native_pytorch import (
            maybe_enable_parallelism,
        )

        transformer = maybe_enable_parallelism(
            transformer,
            parallelism_config,
        )
    else:
        raise ValueError(
            f"Parallel backend {parallelism_config.backend} is not supported yet."
        )

    transformer._is_parallelized = True  # type: ignore[attr-defined]
    # Use `parallelism` not `parallel` to avoid name conflict with diffusers.
    transformer._parallelism_config = parallelism_config  # type: ignore[attr-defined]
    logger.info(
        f"Enabled parallelism: {parallelism_config.strify(True)}, "
        f"transformer id:{id(transformer)}"
    )
    return transformer


def remove_parallelism_stats(
    transformer: torch.nn.Module,
) -> torch.nn.Module:
    if not getattr(transformer, "_is_parallelized", False):
        logger.warning(
            "The transformer is not parallelized. "
            "Skipping removing parallelism."
        )
        return transformer

    if hasattr(transformer, "_is_parallelized"):
        del transformer._is_parallelized  # type: ignore[attr-defined]
    if hasattr(transformer, "_parallelism_config"):
        del transformer._parallelism_config  # type: ignore[attr-defined]
    return transformer
