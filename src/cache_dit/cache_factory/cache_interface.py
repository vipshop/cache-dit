from typing import Any, Tuple, List, Union, Optional
from diffusers import DiffusionPipeline
from cache_dit.cache_factory.cache_types import CacheType
from cache_dit.cache_factory.block_adapters import BlockAdapter
from cache_dit.cache_factory.block_adapters import BlockAdapterRegistry
from cache_dit.cache_factory.cache_adapters import CachedAdapter
from cache_dit.cache_factory.cache_contexts import BasicCacheConfig
from cache_dit.cache_factory.cache_contexts import CalibratorConfig
from cache_dit.cache_factory.params_modifier import ParamsModifier

from cache_dit.logger import init_logger

logger = init_logger(__name__)


def enable_cache(
    # DiffusionPipeline or BlockAdapter
    pipe_or_adapter: Union[
        DiffusionPipeline,
        BlockAdapter,
    ],
    # Basic DBCache config: BasicCacheConfig
    cache_config: BasicCacheConfig = BasicCacheConfig(),
    # Calibrator config: TaylorSeerCalibratorConfig, etc.
    calibrator_config: Optional[CalibratorConfig] = None,
    # Modify cache context params for specific blocks.
    params_modifiers: Optional[
        Union[
            ParamsModifier,
            List[ParamsModifier],
            List[List[ParamsModifier]],
        ]
    ] = None,
    # Other cache context kwargs: Deprecated cache kwargs
    **kwargs,
) -> Union[
    DiffusionPipeline,
    BlockAdapter,
]:
    r"""
    Unified Cache API for  almost Any Diffusion Transformers (with Transformer Blocks
    that match the specific Input and Output patterns).

    For a good balance between performance and precision, DBCache is configured by default
    with F8B0, 8 warmup steps, and unlimited cached steps.

    Args:
        pipe_or_adapter (`DiffusionPipeline` or `BlockAdapter`, *required*):
            The standard Diffusion Pipeline or custom BlockAdapter (from cache-dit or user-defined).
            For example: cache_dit.enable_cache(FluxPipeline(...)). Please check https://github.com/vipshop/cache-dit/blob/main/docs/BlockAdapter.md
            for the usgae of BlockAdapter.
        cache_config (`BasicCacheConfig`, *required*, defaults to BasicCacheConfig()):
            Basic DBCache config for cache context, defaults to BasicCacheConfig(). The configurable params listed belows:
                Fn_compute_blocks: (`int`, *required*, defaults to 8):
                    Specifies that `DBCache` uses the **first n** Transformer blocks to fit the information
                    at time step t, enabling the calculation of a more stable L1 diff and delivering more
                    accurate information to subsequent blocks. Please check https://github.com/vipshop/cache-dit/blob/main/docs/DBCache.md
                    for more details of DBCache.
                Bn_compute_blocks: (`int`, *required*, defaults to 0):
                    Further fuses approximate information in the **last n** Transformer blocks to enhance
                    prediction accuracy. These blocks act as an auto-scaler for approximate hidden states
                    that use residual cache.
                residual_diff_threshold (`float`, *required*, defaults to 0.08):
                    the value of residual diff threshold, a higher value leads to faster performance at the
                    cost of lower precision.
                max_warmup_steps (`int`, *required*, defaults to 8):
                    DBCache does not apply the caching strategy when the number of running steps is less than
                    or equal to this value, ensuring the model sufficiently learns basic features during warmup.
                max_cached_steps (`int`, *required*, defaults to -1):
                    DBCache disables the caching strategy when the previous cached steps exceed this value to
                    prevent precision degradation.
                max_continuous_cached_steps (`int`, *required*, defaults to -1):
                    DBCache disables the caching strategy when the previous continous cached steps exceed this value to
                    prevent precision degradation.
                enable_separate_cfg (`bool`, *required*,  defaults to None):
                    Whether to do separate cfg or not, such as Wan 2.1, Qwen-Image. For model that fused CFG
                    and non-CFG into single forward step, should set enable_separate_cfg as False, for example:
                    CogVideoX, HunyuanVideo, Mochi, etc.
                cfg_compute_first (`bool`, *required*,  defaults to False):
                    Compute cfg forward first or not, default False, namely, 0, 2, 4, ..., -> non-CFG step;
                    1, 3, 5, ... -> CFG step.
                cfg_diff_compute_separate (`bool`, *required*,  defaults to True):
                    Compute separate diff values for CFG and non-CFG step, default True. If False, we will
                    use the computed diff from current non-CFG transformer step for current CFG step.
        calibrator_config (`CalibratorConfig`, *optional*, defaults to None):
            Config for calibrator, if calibrator_config is not None, means that user want to use DBCache
            with specific calibrator, such as taylorseer, foca, and so on.
        params_modifiers ('ParamsModifier', *optional*, defaults to None):
            Modify cache context params for specific blocks. The configurable params listed belows:
                cache_config (`BasicCacheConfig`, *required*, defaults to BasicCacheConfig()):
                    The same as 'cache_config' param in cache_dit.enable_cache() interface.
                calibrator_config (`CalibratorConfig`, *optional*, defaults to None):
                    The same as 'calibrator_config' param in cache_dit.enable_cache() interface.
                **kwargs: (`dict`, *optional*, defaults to {}):
                    The same as 'kwargs' param in cache_dit.enable_cache() interface.
        kwargs (`dict`, *optional*, defaults to {})
            Other cache context kwargs, please check https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/cache_factory/cache_contexts/cache_context.py
            for more details.

    Examples:
    ```py
    >>> import cache_dit
    >>> from diffusers import DiffusionPipeline
    >>> pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image") # Can be any diffusion pipeline
    >>> cache_dit.enable_cache(pipe) # One-line code with default cache options.
    >>> output = pipe(...) # Just call the pipe as normal.
    >>> stats = cache_dit.summary(pipe) # Then, get the summary of cache acceleration stats.
    >>> cache_dit.disable_cache(pipe) # Disable cache and run original pipe.
    """
    # Collect cache context kwargs
    cache_context_kwargs = {}
    if (cache_type := cache_context_kwargs.pop("cache_type", None)) is not None:
        if cache_type == CacheType.NONE:
            return pipe_or_adapter

    # WARNING: Deprecated cache config params. These parameters are now retained
    # for backward compatibility but will be removed in the future.
    deprecated_cache_kwargs = {
        "Fn_compute_blocks": kwargs.get("Fn_compute_blocks", None),
        "Bn_compute_blocks": kwargs.get("Bn_compute_blocks", None),
        "max_warmup_steps": kwargs.get("max_warmup_steps", None),
        "max_cached_steps": kwargs.get("max_cached_steps", None),
        "max_continuous_cached_steps": kwargs.get(
            "max_continuous_cached_steps", None
        ),
        "residual_diff_threshold": kwargs.get("residual_diff_threshold", None),
        "enable_separate_cfg": kwargs.get("enable_separate_cfg", None),
        "cfg_compute_first": kwargs.get("cfg_compute_first", None),
        "cfg_diff_compute_separate": kwargs.get(
            "cfg_diff_compute_separate", None
        ),
    }

    deprecated_cache_kwargs = {
        k: v for k, v in deprecated_cache_kwargs.items() if v is not None
    }

    if deprecated_cache_kwargs:
        logger.warning(
            "Manually settup DBCache context without BasicCacheConfig is "
            "deprecated and will be removed in the future, please use "
            "`cache_config` parameter instead!"
        )
        if cache_config is not None:
            cache_config.update(**deprecated_cache_kwargs)
        else:
            cache_config = BasicCacheConfig(**deprecated_cache_kwargs)

    if cache_config is not None:
        cache_context_kwargs["cache_config"] = cache_config

    # WARNING: Deprecated taylorseer params. These parameters are now retained
    # for backward compatibility but will be removed in the future.
    if (
        kwargs.get("enable_taylorseer", None) is not None
        or kwargs.get("enable_encoder_taylorseer", None) is not None
    ):
        logger.warning(
            "Manually settup TaylorSeer calibrator without TaylorSeerCalibratorConfig is "
            "deprecated and will be removed in the future, please use "
            "`calibrator_config` parameter instead!"
        )
        from cache_dit.cache_factory.cache_contexts.calibrators import (
            TaylorSeerCalibratorConfig,
        )

        calibrator_config = TaylorSeerCalibratorConfig(
            enable_calibrator=kwargs.get("enable_taylorseer"),
            enable_encoder_calibrator=kwargs.get("enable_encoder_taylorseer"),
            calibrator_cache_type=kwargs.get(
                "taylorseer_cache_type", "residual"
            ),
            taylorseer_order=kwargs.get("taylorseer_order", 1),
        )

    if calibrator_config is not None:
        cache_context_kwargs["calibrator_config"] = calibrator_config

    if params_modifiers is not None:
        cache_context_kwargs["params_modifiers"] = params_modifiers

    if isinstance(pipe_or_adapter, (DiffusionPipeline, BlockAdapter)):
        return CachedAdapter.apply(
            pipe_or_adapter,
            **cache_context_kwargs,
        )
    else:
        raise ValueError(
            f"type: {type(pipe_or_adapter)} is not valid, "
            "Please pass DiffusionPipeline or BlockAdapter"
            "for the 1's position param: pipe_or_adapter"
        )


def disable_cache(
    pipe_or_adapter: Union[
        DiffusionPipeline,
        BlockAdapter,
    ],
):
    CachedAdapter.maybe_release_hooks(pipe_or_adapter)
    logger.warning(
        f"Cache Acceleration is disabled for: "
        f"{pipe_or_adapter.__class__.__name__}."
    )


def supported_pipelines(
    **kwargs,
) -> Tuple[int, List[str]]:
    return BlockAdapterRegistry.supported_pipelines(**kwargs)


def get_adapter(
    pipe: DiffusionPipeline | str | Any,
) -> BlockAdapter:
    return BlockAdapterRegistry.get_adapter(pipe)
