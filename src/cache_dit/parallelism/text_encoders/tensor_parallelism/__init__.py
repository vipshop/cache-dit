try:
    import einops  # noqa: F401
except ImportError:
    raise ImportError(
        "parallelism functionality requires the 'parallelism' extra dependencies. "
        "Install with:\npip install cache-dit[parallelism]"
    )

import torch
from typing import Optional
from cache_dit.parallelism.config import ParallelismConfig
from cache_dit.logger import init_logger

try:
    from .tp_plan_registers import TextEncoderTensorParallelismPlannerRegister
    from .tp_planners import _activate_text_encoder_tp_planners

    _activate_text_encoder_tp_planners()
except ImportError as e:
    raise ImportError(e)

logger = init_logger(__name__)


def maybe_enable_tensor_parallelism(
    text_encoder: torch.nn.Module,
    parallelism_config: Optional[ParallelismConfig],
) -> torch.nn.Module:
    assert isinstance(
        text_encoder, torch.nn.Module
    ), f"text_encoder must be an instance of torch.nn.Module, but got {type(text_encoder)}"

    if parallelism_config is None:
        return text_encoder

    # We don't check backend here because text encoder may use different
    # parallelism backend with transformer.

    extra_parallel_kwargs = {}
    if parallelism_config.parallel_kwargs is not None:
        extra_parallel_kwargs = parallelism_config.parallel_kwargs

    return TextEncoderTensorParallelismPlannerRegister.get_planner(text_encoder)().apply(
        text_encoder=text_encoder,
        parallelism_config=parallelism_config,
        **extra_parallel_kwargs,
    )
