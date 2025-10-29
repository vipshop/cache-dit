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
from .tp_plan_registers import TensorParallelismPlanerRegister
from cache_dit.logger import init_logger

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

    return TensorParallelismPlanerRegister.get_planer(transformer)().apply(
        transformer=transformer,
        parallelism_config=parallelism_config,
    )
