import torch
from typing import Optional

from diffusers.models.modeling_utils import ModelMixin
from cache_dit.parallelism.backend import ParallelismBackend
from cache_dit.parallelism.config import ParallelismConfig
from cache_dit.logger import init_logger

try:
    from ...attention import (
        _ExtendedContextParallelConfig,
        _maybe_register_custom_attn_backends,
        _is_diffusers_parallelism_available,
        enable_ulysses_anything,
        enable_ulysses_float8,
    )
    from .cp_plan_registers import ContextParallelismPlannerRegister
    from .cp_planners import _activate_cp_planners

    _maybe_register_custom_attn_backends()
    _activate_cp_planners()
except ImportError as e:
    raise ImportError(e)


logger = init_logger(__name__)


def maybe_enable_context_parallelism(
    transformer: torch.nn.Module | ModelMixin,
    parallelism_config: Optional[ParallelismConfig],
) -> torch.nn.Module:
    assert isinstance(transformer, (torch.nn.Module, ModelMixin)), (
        "transformer must be an instance of torch.nn.Module or ModelMixin, "
        f"but got {type(transformer)}"
    )
    if parallelism_config is None:
        return transformer

    assert isinstance(parallelism_config, ParallelismConfig), (
        "parallelism_config must be an instance of ParallelismConfig"
        f" but got {type(parallelism_config)}"
    )

    if (
        parallelism_config.backend == ParallelismBackend.NATIVE_DIFFUSER
        and _is_diffusers_parallelism_available()
    ):
        cp_config = None
        if parallelism_config.ulysses_size is not None or parallelism_config.ring_size is not None:
            # Prepare extra context parallelism config, e.g, convert_to_fp32,
            # rotate_method for ring attention.
            cp_config = _ExtendedContextParallelConfig(
                ulysses_degree=parallelism_config.ulysses_size,
                ring_degree=parallelism_config.ring_size,
                convert_to_fp32=parallelism_config.parallel_kwargs.get(
                    "ring_convert_to_fp32", True
                ),
                rotate_method=parallelism_config.parallel_kwargs.get(
                    "ring_rotate_method", "allgather"
                ),
            )
        if cp_config is not None:
            experimental_ulysses_anything = parallelism_config.parallel_kwargs.get(
                "experimental_ulysses_anything", False
            )
            # Float8 all_to_all for Ulysses Attention/Ulysses Anything Attention
            experimental_ulysses_float8 = parallelism_config.parallel_kwargs.get(
                "experimental_ulysses_float8", False
            )

            # Must call enable_ulysses_anything before enable_ulysses_float8.
            if experimental_ulysses_anything:
                enable_ulysses_anything()

            if experimental_ulysses_float8:
                enable_ulysses_float8()

            if hasattr(transformer, "enable_parallelism"):
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
                _maybe_patch_native_parallel_config(transformer, **extra_parallel_kwargs)
            else:
                raise ValueError(
                    f"{transformer.__class__.__name__} does not support context parallelism."
                )

    return transformer


def _maybe_patch_native_parallel_config(
    transformer: torch.nn.Module,
    **kwargs,
) -> torch.nn.Module:

    cls_name = transformer.__class__.__name__
    if not cls_name.startswith("Nunchaku"):
        return transformer

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
        from nunchaku.models.transformers.transformer_zimage import (
            NunchakuZImageTransformer2DModel,
            NunchakuZSingleStreamAttnProcessor,
            NunchakuZImageAttention,
        )
    except ImportError:
        raise ImportError(
            "NunchakuZImageTransformer2DModel, NunchakuFluxTransformer2DModelV2 and "
            "NunchakuQwenImageTransformer2DModel requires the 'nunchaku' package. "
            "Please install nunchaku>=1.10 before using the context parallelism for "
            "nunchaku 4-bits models."
        )

    assert isinstance(
        transformer,
        (
            NunchakuFluxTransformer2DModelV2,
            NunchakuQwenImageTransformer2DModel,
            NunchakuZImageTransformer2DModel,
        ),
    )
    config = getattr(transformer, "_parallel_config", None)
    if config is None:
        raise logger.warning(
            f"The transformer {cls_name} does not have _parallel_config attribute. "
            "Skipping patching native parallel config."
        )

    attention_classes = (
        NunchakuFluxAttention,
        NunchakuFluxFA2Processor,
        NunchakuQwenAttention,
        NunchakuQwenImageNaiveFA2Processor,
        NunchakuZImageAttention,
        NunchakuZSingleStreamAttnProcessor,
    )
    for module in transformer.modules():
        if not isinstance(module, attention_classes):
            continue
        processor = getattr(module, "processor", None)
        if processor is None or not hasattr(processor, "_parallel_config"):
            continue
        if getattr(processor, "_parallel_config", None) is not None:
            logger.warning(
                f"The attention processor {processor.__class__.__name__} already has "
                "_parallel_config attribute set. Skipping patching native parallel config."
            )
            continue
        processor._parallel_config = config

    return transformer
