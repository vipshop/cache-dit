try:
    import einops
except ImportError:
    raise ImportError(
        "Metrics functionality requires the 'parallelism' extra dependencies. "
        "Install with:\npip install cache-dit[parallelism]"
    )

import torch
from typing import Optional
from diffusers.models.modeling_utils import ModelMixin
from cache_dit.parallelism.parallel_backend import ParallelismBackend
from cache_dit.parallelism.parallel_config import ParallelismConfig
from cache_dit.logger import init_logger
from .tp_planners import *

logger = init_logger(__name__)


def maybe_enable_tensor_parallelism(
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

    assert parallelism_config.backend == ParallelismBackend.NATIVE_PYTORCH, (
        "parallelism_config.backend must be ParallelismBackend.NATIVE_PYTORCH "
        f"but got {parallelism_config.backend}"
    )

    extra_parallel_kwargs = {}
    if parallelism_config.parallel_kwargs is not None:
        extra_parallel_kwargs = parallelism_config.parallel_kwargs

    return TensorParallelismPlannerRegister.get_planner(transformer)().apply(
        transformer=transformer,
        parallelism_config=parallelism_config,
        **extra_parallel_kwargs,
    )
