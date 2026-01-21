from typing import Optional

import torch

from cache_dit.parallelism.config import ParallelismConfig

from cache_dit.logger import init_logger

logger = init_logger(__name__)


def maybe_enable_parallelism_for_auto_encoder(
    auto_encoder: torch.nn.Module,
    parallelism_config: Optional[ParallelismConfig],
) -> torch.nn.Module:
    assert isinstance(
        auto_encoder, torch.nn.Module
    ), f"auto_encoder must be an instance of torch.nn.Module, but got {type(auto_encoder)}"
    if getattr(auto_encoder, "_is_parallelized", False):
        logger.warning("The auto encoder is already parallelized. Skipping parallelism enabling.")
        return auto_encoder

    if parallelism_config is None:
        return auto_encoder

    from .data_parallelism import maybe_enable_data_parallelism

    auto_encoder = maybe_enable_data_parallelism(
        auto_encoder=auto_encoder,
        parallelism_config=parallelism_config,
    )

    auto_encoder._is_parallelized = True  # type: ignore[attr-defined]
    auto_encoder._parallelism_config = parallelism_config  # type: ignore[attr-defined]

    logger.info(
        f"Parallelize Auto Encoder: {auto_encoder.__class__.__name__}, "
        f"id:{id(auto_encoder)}, {parallelism_config.strify(True, vae=True)}"
    )

    return auto_encoder
