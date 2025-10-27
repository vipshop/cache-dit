from typing import Optional

import torch

from cache_dit.logger import init_logger

logger = init_logger(__name__)


try:
    from diffusers import ContextParallelConfig

    def native_diffusers_parallelism_available() -> bool:
        return True

except ImportError:
    ContextParallelConfig = None

    def native_diffusers_parallelism_available() -> bool:
        return False


from diffusers.models.modeling_utils import ModelMixin

from cache_dit.parallelism.parallel_backend import ParallelismBackend
from cache_dit.parallelism.parallel_config import ParallelismConfig


def maybe_enable_parallelism(
    transformer: torch.nn.Module,
    parallelism_config: Optional[ParallelismConfig],
) -> torch.nn.Module:
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
        from torch.distributed import DeviceMesh, init_device_mesh

        from cache_dit.parallelism.tp.flux.parallelize import dit_apply_tp

        tp_mesh: DeviceMesh = init_device_mesh(
            device_type="cuda",
            mesh_shape=[parallelism_config.tp_size],
        )

        assert transformer.__class__.__name__.startswith(
            "Flux"
        ), "tensor parallelism currently only supports Flux models."
        transformer = dit_apply_tp(
            transformer,
            tp_mesh=tp_mesh,
        )

    return transformer
