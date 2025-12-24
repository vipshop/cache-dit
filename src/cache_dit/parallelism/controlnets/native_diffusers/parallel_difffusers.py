import torch

from typing import Optional
from cache_dit.logger import init_logger

logger = init_logger(__name__)


from cache_dit.parallelism.parallel_backend import ParallelismBackend
from cache_dit.parallelism.parallel_config import ParallelismConfig
from .context_parallelism import maybe_enable_context_parallelism


def maybe_enable_parallelism_for_controlnet(
    controlnet: torch.nn.Module,
    parallelism_config: Optional[ParallelismConfig],
) -> torch.nn.Module:
    assert isinstance(
        controlnet, torch.nn.Module
    ), f"controlnet must be an instance of torch.nn.Module, but got {type(controlnet)}"
    if parallelism_config is None:
        return controlnet

    if parallelism_config.backend != ParallelismBackend.NATIVE_DIFFUSER:
        logger.warning(
            f"Parallelism backend {parallelism_config.backend} is not supported "
            "for ControlNet now, skip context parallelism for ControlNet."
        )
        return controlnet

    if parallelism_config.ulysses_size is not None or parallelism_config.ring_size is not None:
        controlnet = maybe_enable_context_parallelism(
            controlnet,
            parallelism_config,
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
