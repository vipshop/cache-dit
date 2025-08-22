import torch
from typing import Dict, List
from diffusers import DiffusionPipeline
from cache_dit.cache_factory.forward_pattern import ForwardPattern
from cache_dit.cache_factory.cache_types import CacheType
from cache_dit.cache_factory.cache_adapters import BlockAdapterParams
from cache_dit.cache_factory.cache_adapters import UnifiedCacheAdapter
from cache_dit.cache_factory.utils import load_cache_options_from_yaml

from cache_dit.logger import init_logger

logger = init_logger(__name__)


def load_options(path: str):
    return load_cache_options_from_yaml(path)


def cache_type(
    type_hint: "CacheType | str",
) -> CacheType:
    return CacheType.type(cache_type=type_hint)


def default_options(
    cache_type: CacheType = CacheType.DBCache,
) -> Dict:
    return CacheType.default_options(cache_type)


def block_range(
    start: int,
    end: int,
    step: int = 1,
) -> List[int]:
    return CacheType.block_range(
        start,
        end,
        step,
    )


def enable_cache(
    pipe_or_adapter: DiffusionPipeline | BlockAdapterParams,
    forward_pattern: ForwardPattern = ForwardPattern.Pattern_0,
    **cache_options_kwargs,
) -> DiffusionPipeline:
    if isinstance(pipe_or_adapter, BlockAdapterParams):
        return UnifiedCacheAdapter.apply(
            pipe=None,
            adapter_params=pipe_or_adapter,
            forward_pattern=forward_pattern,
            **cache_options_kwargs,
        )
    elif isinstance(pipe_or_adapter, DiffusionPipeline):
        return UnifiedCacheAdapter.apply(
            pipe=pipe_or_adapter,
            adapter_params=None,
            forward_pattern=forward_pattern,
            **cache_options_kwargs,
        )
    else:
        raise ValueError(
            "Please pass DiffusionPipeline or BlockAdapterParams"
            "(BlockAdapter) for the 1 position param: pipe_or_adapter"
        )
