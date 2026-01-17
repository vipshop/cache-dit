from typing import List, Optional, Tuple
import torch
from diffusers import DiffusionPipeline
from cache_dit.logger import init_logger

logger = init_logger(__name__)


def get_text_encoder_from_pipe(
    pipe: DiffusionPipeline,
) -> Tuple[Optional[torch.nn.Module], Optional[str]]:
    pipe_cls_name = pipe.__class__.__name__
    if (
        hasattr(pipe, "text_encoder_2")
        and not pipe_cls_name.startswith("Hunyuan")
        and not pipe_cls_name.startswith("Kandinsky")
    ):
        # Specific for FluxPipeline, FLUX.1-dev
        return getattr(pipe, "text_encoder_2"), "text_encoder_2"
    elif hasattr(pipe, "text_encoder_3"):  # HiDream pipeline
        return getattr(pipe, "text_encoder_3"), "text_encoder_3"
    elif hasattr(pipe, "text_encoder"):  # General case
        return getattr(pipe, "text_encoder"), "text_encoder"
    else:
        return None, None


def prepare_extra_parallel_modules(
    pipe: DiffusionPipeline,
    parallel_text_encoder: bool = False,
    parallel_vae: bool = False,
) -> List[torch.nn.Module]:
    extra_parallel_modules = []

    if parallel_text_encoder:
        text_encoder, encoder_name = get_text_encoder_from_pipe(pipe)
        if text_encoder is not None:
            extra_parallel_modules.append(text_encoder)
            logger.info(
                f"Added {encoder_name} ({text_encoder.__class__.__name__}) to extra_parallel_modules"
            )
        else:
            logger.warning(
                "parallel_text_encoder is enabled but no text encoder found in the pipeline."
            )

    if parallel_vae:
        if hasattr(pipe, "vae") and pipe.vae is not None:
            extra_parallel_modules.append(pipe.vae)
            logger.info(f"Added vae ({pipe.vae.__class__.__name__}) to extra_parallel_modules")
        else:
            logger.warning("parallel_vae is enabled but no VAE found in the pipeline.")

    return extra_parallel_modules
