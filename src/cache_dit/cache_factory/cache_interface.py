from typing import Any, Tuple, List, Union, Optional
from diffusers import DiffusionPipeline
from cache_dit.cache_factory.cache_types import CacheType
from cache_dit.cache_factory.block_adapters import BlockAdapter
from cache_dit.cache_factory.block_adapters import BlockAdapterRegistry
from cache_dit.cache_factory.cache_adapters import CachedAdapter
from cache_dit.cache_factory.cache_contexts import BasicCacheConfig
from cache_dit.cache_factory.cache_contexts import DBCacheConfig
from cache_dit.cache_factory.cache_contexts import DBPruneConfig
from cache_dit.cache_factory.cache_contexts import CalibratorConfig
from cache_dit.cache_factory.params_modifier import ParamsModifier
from cache_dit.parallelism import ParallelismConfig
from cache_dit.parallelism import enable_parallelism

from cache_dit.logger import init_logger

logger = init_logger(__name__)


def enable_cache(
    # DiffusionPipeline or BlockAdapter
    pipe_or_adapter: Union[
        DiffusionPipeline,
        BlockAdapter,
    ],
    # BasicCacheConfig, DBCacheConfig, DBPruneConfig, etc.
    cache_config: Optional[
        Union[
            BasicCacheConfig,
            DBCacheConfig,
            DBPruneConfig,
        ]
    ] = None,
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
    # Config for Parallelism
    parallelism_config: Optional[ParallelismConfig] = None,
    # Other cache context kwargs: Deprecated cache kwargs
    **kwargs,
) -> Union[
    DiffusionPipeline,
    BlockAdapter,
]:
    r"""
    The `enable_cache` function serves as a unified caching interface designed to optimize the performance
    of diffusion transformer models by implementing an intelligent caching mechanism known as `DBCache`.
    This API is engineered to be compatible with nearly `all` diffusion transformer architectures that
    feature transformer blocks adhering to standard input-output patterns, eliminating the need for
    architecture-specific modifications.

    By strategically caching intermediate outputs of transformer blocks during the diffusion process,
    `DBCache` significantly reduces redundant computations without compromising generation quality.
    The caching mechanism works by tracking residual differences between consecutive steps, allowing
    the model to reuse previously computed features when these differences fall below a configurable
    threshold. This approach maintains a balance between computational efficiency and output precision.

    The default configuration (`F8B0, 8 warmup steps, unlimited cached steps`) is carefully tuned to
    provide an optimal tradeoff for most common use cases. The "F8B0" configuration indicates that
    the first 8 transformer blocks are used to compute stable feature differences, while no final
    blocks are employed for additional fusion. The warmup phase ensures the model establishes
    sufficient feature representation before caching begins, preventing potential degradation of
    output quality.

    This function seamlessly integrates with both standard diffusion pipelines and custom block
    adapters, making it versatile for various deployment scenarios—from research prototyping to
    production environments where inference speed is critical. By abstracting the complexity of
    caching logic behind a simple interface, it enables developers to enhance model performance
    with minimal code changes.

    Args:
        pipe_or_adapter (`DiffusionPipeline` or `BlockAdapter`, *required*):
            The standard Diffusion Pipeline or custom BlockAdapter (from cache-dit or user-defined).
            For example: cache_dit.enable_cache(FluxPipeline(...)). Please check https://github.com/vipshop/cache-dit/blob/main/docs/BlockAdapter.md
            for the usgae of BlockAdapter.

        cache_config (`BasicCacheConfig`, *required*, defaults to BasicCacheConfig()):
            Basic DBCache config for cache context, defaults to BasicCacheConfig(). The configurable params listed belows:
                Fn_compute_blocks: (`int`, *required*, defaults to 8):
                    Specifies that `DBCache` uses the**first n**Transformer blocks to fit the information at time step t,
                    enabling the calculation of a more stable L1 difference and delivering more accurate information
                    to subsequent blocks. Please check https://github.com/vipshop/cache-dit/blob/main/docs/DBCache.md
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
                warmup_interval (`int`, *required*, defaults to 1):
                    Skip interval in warmup steps, e.g., when warmup_interval is 2, only 0, 2, 4, ... steps
                    in warmup steps will be computed, others will use dynamic cache.
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
                    Whether to compute cfg forward first, default is False, meaning:
                    0, 2, 4, ..., -> non-CFG step;
                    1, 3, 5, ... -> CFG step.
                cfg_diff_compute_separate (`bool`, *required*,  defaults to True):
                    Whether to compute separate difference values for CFG and non-CFG steps, default is True.
                    If False, we will use the computed difference from the current non-CFG transformer step
                    for the current CFG step.

        calibrator_config (`CalibratorConfig`, *optional*, defaults to None):
            Config for calibrator. If calibrator_config is not None, it means the user wants to use DBCache
            with a specific calibrator, such as taylorseer, foca, and so on.

        params_modifiers ('ParamsModifier', *optional*, defaults to None):
            Modify cache context params for specific blocks. The configurable params listed belows:
                cache_config (`BasicCacheConfig`, *required*, defaults to BasicCacheConfig()):
                    The same as 'cache_config' param in cache_dit.enable_cache() interface.
                calibrator_config (`CalibratorConfig`, *optional*, defaults to None):
                    The same as 'calibrator_config' param in cache_dit.enable_cache() interface.
                **kwargs: (`dict`, *optional*, defaults to {}):
                    The same as 'kwargs' param in cache_dit.enable_cache() interface.

        parallelism_config (`ParallelismConfig`, *optional*, defaults to None):
            Config for Parallelism. If parallelism_config is not None, it means the user wants to enable
            parallelism for cache-dit. Please check https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/parallelism/parallel_config.py
            for more details of ParallelismConfig.
                ulysses_size: (`int`, *optional*, defaults to None):
                    The size of Ulysses cluster. If ulysses_size is not None, enable Ulysses style parallelism.
                ring_size: (`int`, *optional*, defaults to None):
                    The size of ring for ring parallelism. If ring_size is not None, enable ring attention.

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
    # Precheck for compatibility of different configurations
    if cache_config is None:
        if parallelism_config is None:
            # Set default cache config only when parallelism is not enabled
            logger.info("Using default DBCacheConfig for cache context.")
            cache_config = DBCacheConfig()
        else:
            # Allow empty cache_config when parallelism is enabled
            logger.info(
                "Parallelism is enabled, please manually set cache_config "
                "to avoid potential compatibility issues."
            )

    # Collect cache context kwargs
    context_kwargs = {}
    if (cache_type := context_kwargs.get("cache_type", None)) is not None:
        if cache_type == CacheType.NONE:
            return pipe_or_adapter

    # NOTE: Deprecated cache config params. These parameters are now retained
    # for backward compatibility but will be removed in the future.
    deprecated_kwargs = {
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

    deprecated_kwargs = {
        k: v for k, v in deprecated_kwargs.items() if v is not None
    }

    if deprecated_kwargs:
        logger.warning(
            "Manually settup DBCache context without BasicCacheConfig is "
            "deprecated and will be removed in the future, please use "
            "`cache_config` parameter instead!"
        )
        if cache_config is not None:
            cache_config.update(**deprecated_kwargs)
        else:
            cache_config = BasicCacheConfig(**deprecated_kwargs)

    if cache_config is not None:
        context_kwargs["cache_config"] = cache_config

    # NOTE: Deprecated taylorseer params. These parameters are now retained
    # for backward compatibility but will be removed in the future.
    if cache_config is not None and (
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
        context_kwargs["calibrator_config"] = calibrator_config

    if params_modifiers is not None:
        context_kwargs["params_modifiers"] = params_modifiers

    if cache_config is not None:
        if isinstance(pipe_or_adapter, (DiffusionPipeline, BlockAdapter)):
            pipe_or_adapter = CachedAdapter.apply(
                pipe_or_adapter,
                **context_kwargs,
            )
        else:
            raise ValueError(
                f"type: {type(pipe_or_adapter)} is not valid, "
                "Please pass DiffusionPipeline or BlockAdapter"
                "for the 1's position param: pipe_or_adapter"
            )
    else:
        logger.warning(
            "cache_config is None, skip enabling cache for "
            f"{pipe_or_adapter.__class__.__name__}."
        )

    # NOTE: Users should always enable parallelism after applying
    # cache to avoid hooks conflict.
    if parallelism_config is not None:
        assert isinstance(
            parallelism_config, ParallelismConfig
        ), "parallelism_config should be of type ParallelismConfig."
        if isinstance(pipe_or_adapter, DiffusionPipeline):
            transformer = pipe_or_adapter.transformer
        else:
            assert BlockAdapter.assert_normalized(pipe_or_adapter)
            transformers = BlockAdapter.flatten(pipe_or_adapter.transformer)
            if len(transformers) > 1:
                logger.warning(
                    "Multiple transformers are detected in the "
                    "BlockAdapter, all transfomers will be "
                    "enabled for parallelism."
                )
            for i, transformer in enumerate(transformers):
                # Enable parallelism for the transformer inplace
                transformers[i] = enable_parallelism(
                    transformer, parallelism_config
                )
    return pipe_or_adapter


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
