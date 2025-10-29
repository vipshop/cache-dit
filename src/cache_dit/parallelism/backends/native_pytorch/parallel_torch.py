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
        "transformer must be an instance of torch.nn.Module, "
        f"but got {type(transformer)}"
    )
    assert isinstance(transformer, ModelMixin), (
        "transformer must be an instance of diffusers' ModelMixin, "
        f"but got {type(transformer)}"
    )
    if parallelism_config is None:
        return transformer

    assert isinstance(parallelism_config, ParallelismConfig), (
        "parallelism_config must be an instance of ParallelismConfig"
        f" but got {type(parallelism_config)}"
    )

    if (
        parallelism_config.backend == ParallelismBackend.NATIVE_PYTORCH
        and parallelism_config.tp_size > 1
    ):
        from .tensor_parallelism import maybe_enable_tensor_parallelism

        transformer = maybe_enable_tensor_parallelism(
            transformer=transformer,
            parallelism_config=parallelism_config,
        )
    return transformer
