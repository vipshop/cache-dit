import torch

from typing import Optional
from cache_dit.logger import init_logger

logger = init_logger(__name__)


from diffusers.models.modeling_utils import ModelMixin
from cache_dit.parallelism.parallel_backend import ParallelismBackend
from cache_dit.parallelism.parallel_config import ParallelismConfig
from .context_parallelism import maybe_enable_context_parallelism


def maybe_enable_parallelism(
    transformer: torch.nn.Module,
    parallelism_config: Optional[ParallelismConfig],
) -> torch.nn.Module:
    assert isinstance(transformer, ModelMixin), (
        "transformer must be an instance of diffusers' ModelMixin, " f"but got {type(transformer)}"
    )
    if parallelism_config is None:
        return transformer

    assert isinstance(parallelism_config, ParallelismConfig), (
        "parallelism_config must be an instance of ParallelismConfig"
        f" but got {type(parallelism_config)}"
    )

    assert parallelism_config.backend == ParallelismBackend.NATIVE_DIFFUSER, (
        f"parallelism backend must be {ParallelismBackend.NATIVE_DIFFUSER}, "
        f"but got {parallelism_config.backend}"
    )

    if parallelism_config.ulysses_size is not None or parallelism_config.ring_size is not None:
        transformer = maybe_enable_context_parallelism(
            transformer,
            parallelism_config,
        )
    else:
        raise ValueError(
            "NATIVE_DIFFUSER backend only support context parallelism now. "
            "Please set ulysses_size or ring_size in parallelism_config."
        )
    return transformer
