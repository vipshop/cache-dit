import torch
from typing import Optional

from diffusers.models.modeling_utils import ModelMixin
from cache_dit.parallelism.config import ParallelismConfig
from ....logger import init_logger

try:
    from ...attention import (
        _ExtendedContextParallelConfig,
        _enable_context_parallelism_ext,
        _maybe_register_custom_attn_backends,
        _is_diffusers_parallelism_available,
        enable_ulysses_anything,
        enable_ulysses_float8,
    )
    from .cp_plan_registers import ControlNetContextParallelismPlannerRegister
    from .cp_planners import _activate_controlnet_cp_planners

    _maybe_register_custom_attn_backends()
    _activate_controlnet_cp_planners()
except ImportError as e:
    raise ImportError(e)


logger = init_logger(__name__)


def maybe_enable_context_parallelism(
    controlnet: torch.nn.Module,
    parallelism_config: Optional[ParallelismConfig],
) -> torch.nn.Module:
    assert isinstance(controlnet, ModelMixin), (
        "controlnet must be an instance of diffusers' ModelMixin, " f"but got {type(controlnet)}"
    )
    if parallelism_config is None:
        return controlnet

    assert isinstance(parallelism_config, ParallelismConfig), (
        "parallelism_config must be an instance of ParallelismConfig"
        f" but got {type(parallelism_config)}"
    )
    assert _is_diffusers_parallelism_available(), (
        "Context parallelism requires the 'diffusers>=0.36.dev0'."
        "Please install latest version of diffusers from source"
    )

    if parallelism_config.cp_enabled():
        assert (
            not parallelism_config.hybrid_enabled()
        ), "Hybrid parallelism is not supported for ControlNet now."
        # Prepare extra context parallelism config, e.g, convert_to_fp32,
        # rotate_method for ring attention.
        cp_config = _ExtendedContextParallelConfig(
            ulysses_degree=parallelism_config.ulysses_size,
            ring_degree=parallelism_config.ring_size,
            convert_to_fp32=parallelism_config.ring_convert_to_fp32,
            rotate_method=parallelism_config.ring_rotate_method,
        )

        if parallelism_config.ulysses_anything:
            enable_ulysses_anything()

        if parallelism_config.ulysses_float8:
            enable_ulysses_float8()

        # Prefer custom cp_plan if provided
        cp_plan = parallelism_config.cp_plan
        if cp_plan is not None:
            logger.info(f"Using custom context parallelism plan: {cp_plan}")
        else:
            # Try get context parallelism plan from register if not provided
            cp_plan = ControlNetContextParallelismPlannerRegister.get_planner(controlnet)().apply(
                controlnet=controlnet, parallelism_config=parallelism_config
            )

            _enable_context_parallelism_ext(controlnet, config=cp_config, cp_plan=cp_plan)

    return controlnet
