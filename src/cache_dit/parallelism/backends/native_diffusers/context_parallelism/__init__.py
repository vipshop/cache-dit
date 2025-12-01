import torch
from typing import Optional

from diffusers.models.modeling_utils import ModelMixin
from cache_dit.parallelism.parallel_backend import ParallelismBackend
from cache_dit.parallelism.parallel_config import ParallelismConfig
from cache_dit.logger import init_logger
from ..utils import (
    native_diffusers_parallelism_available,
    ContextParallelConfig,
)
from .attention import maybe_resigter_native_attention_backend
from .attention import enable_ulysses_anything
from .cp_planners import *

try:
    maybe_resigter_native_attention_backend()
except ImportError as e:
    raise ImportError(e)

logger = init_logger(__name__)


def maybe_enable_context_parallelism(
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

    if (
        parallelism_config.backend == ParallelismBackend.NATIVE_DIFFUSER
        and native_diffusers_parallelism_available()
    ):
        cp_config = None
        if parallelism_config.ulysses_size is not None or parallelism_config.ring_size is not None:
            cp_config = ContextParallelConfig(
                ulysses_degree=parallelism_config.ulysses_size,
                ring_degree=parallelism_config.ring_size,
            )
        if cp_config is not None:
            experimental_ulysses_anything = parallelism_config.parallel_kwargs.get(
                "experimental_ulysses_anything", False
            )
            if experimental_ulysses_anything:
                enable_ulysses_anything()

            attention_backend = parallelism_config.parallel_kwargs.get("attention_backend", None)
            if hasattr(transformer, "enable_parallelism"):
                if hasattr(transformer, "set_attention_backend"):
                    # native, _native_cudnn, flash, etc.
                    if attention_backend is None:
                        # Default to native for context parallelism due to:
                        # - attn mask support (re-registered in cache-dit)
                        # - general compatibility with various models
                        transformer.set_attention_backend("native")
                        logger.warning(
                            "attention_backend is None, set default attention backend "
                            "to _native_cudnn for parallelism because of the issue: "
                            "https://github.com/huggingface/diffusers/pull/12443"
                        )
                    else:
                        transformer.set_attention_backend(attention_backend)
                        logger.info(
                            "Found attention_backend from config, set attention "
                            f"backend to: {attention_backend}"
                        )
                # Prefer custom cp_plan if provided
                cp_plan = parallelism_config.parallel_kwargs.get("cp_plan", None)
                if cp_plan is not None:
                    logger.info(f"Using custom context parallelism plan: {cp_plan}")
                else:
                    # Try get context parallelism plan from register if not provided
                    extra_parallel_kwargs = {}
                    if parallelism_config.parallel_kwargs is not None:
                        extra_parallel_kwargs = parallelism_config.parallel_kwargs
                    cp_plan = ContextParallelismPlannerRegister.get_planner(transformer)().apply(
                        transformer=transformer, **extra_parallel_kwargs
                    )

                transformer.enable_parallelism(config=cp_config, cp_plan=cp_plan)
                _maybe_patch_native_parallel_config(transformer)
            else:
                raise ValueError(
                    f"{transformer.__class__.__name__} does not support context parallelism."
                )

    return transformer


def _maybe_patch_native_parallel_config(
    transformer: torch.nn.Module,
) -> torch.nn.Module:

    cls_name = transformer.__class__.__name__
    if not cls_name.startswith("Nunchaku"):
        return transformer

    from diffusers import FluxTransformer2DModel, QwenImageTransformer2DModel

    try:
        from nunchaku.models.transformers.transformer_flux_v2 import (
            NunchakuFluxTransformer2DModelV2,
            NunchakuFluxAttention,
            NunchakuFluxFA2Processor,
        )
        from nunchaku.models.transformers.transformer_qwenimage import (
            NunchakuQwenAttention,
            NunchakuQwenImageNaiveFA2Processor,
            NunchakuQwenImageTransformer2DModel,
        )
    except ImportError:
        raise ImportError(
            "NunchakuFluxTransformer2DModelV2 or NunchakuQwenImageTransformer2DModel "
            "requires the 'nunchaku' package. Please install nunchaku before using "
            "the context parallelism for nunchaku 4-bits models."
        )
    assert isinstance(
        transformer,
        (
            NunchakuFluxTransformer2DModelV2,
            FluxTransformer2DModel,
        ),
    ) or isinstance(
        transformer,
        (
            NunchakuQwenImageTransformer2DModel,
            QwenImageTransformer2DModel,
        ),
    ), (
        "transformer must be an instance of NunchakuFluxTransformer2DModelV2 "
        f"or NunchakuQwenImageTransformer2DModel, but got {type(transformer)}"
    )
    config = transformer._parallel_config

    attention_classes = (
        NunchakuFluxAttention,
        NunchakuFluxFA2Processor,
        NunchakuQwenAttention,
        NunchakuQwenImageNaiveFA2Processor,
    )
    for module in transformer.modules():
        if not isinstance(module, attention_classes):
            continue
        processor = getattr(module, "processor", None)
        if processor is None or not hasattr(processor, "_parallel_config"):
            continue
        processor._parallel_config = config

    return transformer
