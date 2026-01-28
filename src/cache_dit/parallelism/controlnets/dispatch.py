import torch

from typing import Optional
from cache_dit.logger import init_logger

from diffusers.models.modeling_utils import ModelMixin
from cache_dit.parallelism.backend import ParallelismBackend
from cache_dit.parallelism.config import ParallelismConfig
from .context_parallelism import maybe_enable_context_parallelism

logger = init_logger(__name__)


def maybe_enable_parallelism_for_controlnet(
    controlnet: torch.nn.Module | ModelMixin,
    parallelism_config: Optional[ParallelismConfig],
) -> torch.nn.Module:
    assert isinstance(controlnet, (torch.nn.Module, ModelMixin)), (
        "controlnet must be an instance of torch.nn.Module or ModelMixin, "
        f"but got {type(controlnet)}"
    )

    if parallelism_config is None:
        return controlnet

    if parallelism_config.backend != ParallelismBackend.NATIVE_DIFFUSER:
        logger.warning(
            f"Parallelism backend {parallelism_config.backend} is not supported "
            "for ControlNet now, skip context parallelism for ControlNet."
        )
        return controlnet

    if parallelism_config.cp_enabled():
        controlnet = maybe_enable_context_parallelism(
            controlnet=controlnet,
            parallelism_config=parallelism_config,
        )
        controlnet._is_parallelized = True  # type: ignore[attr-defined]
        # Use `parallelism` not `parallel` to avoid name conflict with diffusers.
        controlnet._parallelism_config = parallelism_config  # type: ignore[attr-defined]
        logger.info(
            f"Parallelize ControlNet: {controlnet.__class__.__name__}, "
            f"id:{id(controlnet)}, {parallelism_config.strify(True)}"
        )
    else:
        logger.warning(
            "Please set ulysses_size or ring_size in parallelism_config to enable "
            "context parallelism for ControlNet. Skipping parallelism for ControlNet."
        )
    return controlnet
