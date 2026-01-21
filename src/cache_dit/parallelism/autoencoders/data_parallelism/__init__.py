import torch
from typing import Optional
from cache_dit.parallelism.config import ParallelismConfig
from cache_dit.logger import init_logger

try:
    from .dp_plan_registers import AutoEncoderDataParallelismPlannerRegister
    from .dp_planners import _activate_auto_encoder_dp_planners

    _activate_auto_encoder_dp_planners()
except ImportError as e:
    raise ImportError(e)

logger = init_logger(__name__)


def maybe_enable_data_parallelism(
    auto_encoder: torch.nn.Module,
    parallelism_config: Optional[ParallelismConfig],
) -> torch.nn.Module:
    assert isinstance(
        auto_encoder, torch.nn.Module
    ), f"auto_encoder must be an instance of torch.nn.Module, but got {type(auto_encoder)}"

    if parallelism_config is None:
        return auto_encoder

    # We don't check backend here because auto encoder may use different
    # parallelism backend with transformer.

    extra_parallel_kwargs = {}
    if parallelism_config.parallel_kwargs is not None:
        extra_parallel_kwargs = parallelism_config.parallel_kwargs

    return AutoEncoderDataParallelismPlannerRegister.get_planner(auto_encoder)().apply(
        auto_encoder=auto_encoder,
        parallelism_config=parallelism_config,
        **extra_parallel_kwargs,
    )
