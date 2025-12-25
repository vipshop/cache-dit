from typing import Optional

import torch

from cache_dit.parallelism.config import ParallelismConfig

from cache_dit.logger import init_logger

logger = init_logger(__name__)


def maybe_enable_parallelism_for_text_encoder(
    text_encoder: torch.nn.Module,
    parallelism_config: Optional[ParallelismConfig],
) -> torch.nn.Module:
    assert isinstance(
        text_encoder, torch.nn.Module
    ), f"text_encoder must be an instance of torch.nn.Module, but got {type(text_encoder)}"
    if getattr(text_encoder, "_is_parallelized", False):
        logger.warning("The text encoder is already parallelized. Skipping parallelism enabling.")
        return text_encoder

    if parallelism_config is None:
        return text_encoder

    from .tensor_parallelism import maybe_enable_tensor_parallelism

    text_encoder = maybe_enable_tensor_parallelism(
        text_encoder=text_encoder,
        parallelism_config=parallelism_config,
    )

    text_encoder._is_parallelized = True  # type: ignore[attr-defined]
    text_encoder._parallelism_config = parallelism_config  # type: ignore[attr-defined]

    logger.info(
        f"Parallelize Text Encoder: {text_encoder.__class__.__name__}, "
        f"id:{id(text_encoder)}, {parallelism_config.strify(True, text_encoder=True)}"
    )

    return text_encoder
