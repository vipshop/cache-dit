try:
    import einops
except ImportError:
    raise ImportError(
        "parallelism functionality requires the 'parallelism' extra dependencies. "
        "Install with:\npip install cache-dit[parallelism]"
    )

import torch
from typing import Optional
from diffusers.models.modeling_utils import ModelMixin
from cache_dit.parallelism.config import ParallelismConfig
from cache_dit.logger import init_logger

try:
    from .tp_plan_registers import TensorParallelismPlannerRegister
    from .tp_planners import _activate_tp_planners

    _activate_tp_planners()
except ImportError as e:
    raise ImportError(e)

logger = init_logger(__name__)


def maybe_enable_tensor_parallelism(
    transformer: torch.nn.Module | ModelMixin,
    parallelism_config: Optional[ParallelismConfig],
) -> torch.nn.Module:
    assert isinstance(transformer, (torch.nn.Module, ModelMixin)), (
        "transformer must be an instance of torch.nn.Module or ModelMixin, "
        f"but got {type(transformer)}"
    )

    assert isinstance(transformer, ModelMixin), (
        "transformer must be an instance of diffusers' ModelMixin, " f"but got {type(transformer)}"
    )
    if parallelism_config is None:
        return transformer

    extra_parallel_kwargs = {}
    if parallelism_config.parallel_kwargs is not None:
        extra_parallel_kwargs = parallelism_config.parallel_kwargs

    if not parallelism_config.tp_enabled():
        return transformer

    return TensorParallelismPlannerRegister.get_planner(transformer)().apply(
        transformer=transformer,
        parallelism_config=parallelism_config,
        **extra_parallel_kwargs,
    )
