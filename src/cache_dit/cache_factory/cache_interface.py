from typing import Any, Tuple, List
from diffusers import DiffusionPipeline
from cache_dit.cache_factory.cache_types import CacheType
from cache_dit.cache_factory.block_adapters import BlockAdapter
from cache_dit.cache_factory.block_adapters import BlockAdapterRegistry
from cache_dit.cache_factory.cache_adapters import UnifiedCacheAdapter

from cache_dit.logger import init_logger

logger = init_logger(__name__)


def enable_cache(
    # DiffusionPipeline or BlockAdapter
    pipe_or_adapter: DiffusionPipeline | BlockAdapter | Any,
    # Cache context kwargs
    Fn_compute_blocks: int = 8,
    Bn_compute_blocks: int = 0,
    max_warmup_steps: int = 8,
    max_cached_steps: int = -1,
    max_continuous_cached_steps: int = -1,
    residual_diff_threshold: float = 0.08,
    # Cache CFG or not
    do_separate_cfg: bool = False,
    cfg_compute_first: bool = False,
    cfg_diff_compute_separate: bool = True,
    # Hybird TaylorSeer
    enable_taylorseer: bool = False,
    enable_encoder_taylorseer: bool = False,
    taylorseer_cache_type: str = "residual",
    taylorseer_order: int = 2,
    **other_cache_context_kwargs,
) -> DiffusionPipeline | Any:
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
        Fn_compute_blocks (`int`, *required*, defaults to 8):
            Specifies that `DBCache` uses the **first n** Transformer blocks to fit the information
            at time step t, enabling the calculation of a more stable L1 diff and delivering more
            accurate information to subsequent blocks. Please check https://github.com/vipshop/cache-dit/blob/main/docs/DBCache.md
            for more details of DBCache.
        Bn_compute_blocks: (`int`, *required*, defaults to 0):
            Further fuses approximate information in the **last n** Transformer blocks to enhance
            prediction accuracy. These blocks act as an auto-scaler for approximate hidden states
            that use residual cache.
        max_warmup_steps (`int`, *required*, defaults to 8):
            DBCache does not apply the caching strategy when the number of running steps is less than
            or equal to this value, ensuring the model sufficiently learns basic features during warmup.
        max_cached_steps (`int`, *required*, defaults to -1):
            DBCache disables the caching strategy when the previous cached steps exceed this value to
            prevent precision degradation.
        max_continuous_cached_steps (`int`, *required*, defaults to -1):
            DBCache disables the caching strategy when the previous continous cached steps exceed this value to
            prevent precision degradation.
        residual_diff_threshold (`float`, *required*, defaults to 0.08):
            he value of residual diff threshold, a higher value leads to faster performance at the
            cost of lower precision.
        do_separate_cfg (`bool`, *required*,  defaults to False):
            Whether to do separate cfg or not, such as Wan 2.1, Qwen-Image. For model that fused CFG
            and non-CFG into single forward step, should set do_separate_cfg as False, for example:
            CogVideoX, HunyuanVideo, Mochi, etc.
        cfg_compute_first (`bool`, *required*,  defaults to False):
            Compute cfg forward first or not, default False, namely, 0, 2, 4, ..., -> non-CFG step;
            1, 3, 5, ... -> CFG step.
        cfg_diff_compute_separate (`bool`, *required*,  defaults to True):
            Compute spearate diff values for CFG and non-CFG step, default True. If False, we will
            use the computed diff from current non-CFG transformer step for current CFG step.
        enable_taylorseer (`bool`, *required*,  defaults to False):
            Enable the hybird TaylorSeer for hidden_states or not. We have supported the
            [TaylorSeers: From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers](https://arxiv.org/pdf/2503.06923) algorithm
            to further improve the precision of DBCache in cases where the cached steps are large,
            namely, **Hybrid TaylorSeer + DBCache**. At timesteps with significant intervals,
            the feature similarity in diffusion models decreases substantially, significantly
            harming the generation quality.
        enable_encoder_taylorseer (`bool`, *required*,  defaults to False):
            Enable the hybird TaylorSeer for encoder_hidden_states or not.
        taylorseer_cache_type (`str`, *required*,  defaults to `residual`):
            The TaylorSeer implemented in cache-dit supports both `hidden_states` and `residual` as cache type.
        taylorseer_order (`int`, *required*, defaults to 2):
            The order of taylorseer, higher values of n_derivatives will lead to longer computation time,
            but may improve precision significantly.
        other_cache_kwargs: (`dict`, *optional*, defaults to {})
            Other cache context kwargs, please check https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/cache_factory/cache_context.py
            for more details.

    Examples:
    ```py
    >>> import cache_dit
    >>> from diffusers import DiffusionPipeline
    >>> pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image") # Can be any diffusion pipeline
    >>> cache_dit.enable_cache(pipe) # One-line code with default cache options.
    >>> output = pipe(...) # Just call the pipe as normal.
    >>> stats = cache_dit.summary(pipe) # Then, get the summary of cache acceleration stats.
    """

    # Collect cache context kwargs
    cache_context_kwargs = other_cache_context_kwargs.copy()
    cache_context_kwargs["cache_type"] = CacheType.DBCache
    cache_context_kwargs["Fn_compute_blocks"] = Fn_compute_blocks
    cache_context_kwargs["Bn_compute_blocks"] = Bn_compute_blocks
    cache_context_kwargs["max_warmup_steps"] = max_warmup_steps
    cache_context_kwargs["max_cached_steps"] = max_cached_steps
    cache_context_kwargs["max_continuous_cached_steps"] = (
        max_continuous_cached_steps
    )
    cache_context_kwargs["residual_diff_threshold"] = residual_diff_threshold
    cache_context_kwargs["do_separate_cfg"] = do_separate_cfg
    cache_context_kwargs["cfg_compute_first"] = cfg_compute_first
    cache_context_kwargs["cfg_diff_compute_separate"] = (
        cfg_diff_compute_separate
    )
    cache_context_kwargs["enable_taylorseer"] = enable_taylorseer
    cache_context_kwargs["enable_encoder_taylorseer"] = (
        enable_encoder_taylorseer
    )
    cache_context_kwargs["taylorseer_cache_type"] = taylorseer_cache_type
    if "taylorseer_kwargs" in cache_context_kwargs:
        cache_context_kwargs["taylorseer_kwargs"][
            "n_derivatives"
        ] = taylorseer_order
    else:
        cache_context_kwargs["taylorseer_kwargs"] = {
            "n_derivatives": taylorseer_order
        }

    if isinstance(pipe_or_adapter, BlockAdapter):
        return UnifiedCacheAdapter.apply(
            pipe=None,
            block_adapter=pipe_or_adapter,
            **cache_context_kwargs,
        )
    elif isinstance(pipe_or_adapter, DiffusionPipeline):
        return UnifiedCacheAdapter.apply(
            pipe=pipe_or_adapter,
            block_adapter=None,
            **cache_context_kwargs,
        )
    else:
        raise ValueError(
            f"type: {type(pipe_or_adapter)} is not valid, "
            "Please pass DiffusionPipeline or BlockAdapter"
            "for the 1's position param: pipe_or_adapter"
        )


def supported_pipelines(
    **kwargs,
) -> Tuple[int, List[str]]:
    return BlockAdapterRegistry.supported_pipelines(**kwargs)


def get_adapter(
    pipe: DiffusionPipeline | str | Any,
) -> BlockAdapter:
    return BlockAdapterRegistry.get_adapter(pipe)
