from typing import Optional

import torch

from diffusers.models.modeling_utils import ModelMixin

from cache_dit.parallelism.parallel_backend import ParallelismBackend
from cache_dit.parallelism.parallel_config import ParallelismConfig

from cache_dit.logger import init_logger

logger = init_logger(__name__)


def maybe_enable_parallelism(
    transformer: torch.nn.Module | ModelMixin,
    parallelism_config: Optional[ParallelismConfig],
) -> torch.nn.Module:
    assert isinstance(transformer, torch.nn.Module), (
        "transformer must be an instance of torch.nn.Module, " f"but got {type(transformer)}"
    )
    assert isinstance(transformer, ModelMixin), (
        "transformer must be an instance of diffusers' ModelMixin, " f"but got {type(transformer)}"
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
    else:
        raise ValueError(
            "NATIVE_PYTORCH only supported tensor parallelism now. "
            "Please set tp_size > 1 for tensor parallelism."
        )
    return transformer
