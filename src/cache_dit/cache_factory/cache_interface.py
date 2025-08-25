from diffusers import DiffusionPipeline
from cache_dit.cache_factory.forward_pattern import ForwardPattern
from cache_dit.cache_factory.cache_types import CacheType
from cache_dit.cache_factory.cache_adapters import BlockAdapterParams
from cache_dit.cache_factory.cache_adapters import UnifiedCacheAdapter

from cache_dit.logger import init_logger

logger = init_logger(__name__)


def enable_cache(
    # BlockAdapter & forward pattern
    pipe_or_adapter: DiffusionPipeline | BlockAdapterParams,
    forward_pattern: ForwardPattern = ForwardPattern.Pattern_0,
    # Cache context kwargs
    cache_type: CacheType = CacheType.DBCache,
    Fn_compute_blocks: int = 8,
    Bn_compute_blocks: int = 0,
    warmup_steps: int = 8,
    max_cached_steps: int = -1,
    residual_diff_threshold: float = 0.08,
    # Cache CFG or not
    do_separate_cfg: bool = False,
    cfg_compute_first: bool = False,
    cfg_diff_compute_separate: bool = False,
    # Hybird TaylorSeer
    enable_taylorseer: bool = False,
    enable_encoder_taylorseer: bool = False,
    taylorseer_cache_type: str = "residual",
    taylorseer_order: int = 2,
    **other_cache_kwargs,
) -> DiffusionPipeline:

    # Collect cache context kwargs
    cache_context_kwargs = other_cache_kwargs.copy()
    cache_context_kwargs["cache_type"] = cache_type
    cache_context_kwargs["Fn_compute_blocks"] = Fn_compute_blocks
    cache_context_kwargs["Bn_compute_blocks"] = Bn_compute_blocks
    cache_context_kwargs["warmup_steps"] = warmup_steps
    cache_context_kwargs["max_cached_steps"] = max_cached_steps
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

    if isinstance(pipe_or_adapter, BlockAdapterParams):
        return UnifiedCacheAdapter.apply(
            pipe=None,
            adapter_params=pipe_or_adapter,
            forward_pattern=forward_pattern,
            **cache_context_kwargs,
        )
    elif isinstance(pipe_or_adapter, DiffusionPipeline):
        return UnifiedCacheAdapter.apply(
            pipe=pipe_or_adapter,
            adapter_params=None,
            forward_pattern=forward_pattern,
            **cache_context_kwargs,
        )
    else:
        raise ValueError(
            "Please pass DiffusionPipeline or BlockAdapterParams"
            "(BlockAdapter) for the 1 position param: pipe_or_adapter"
        )
