import torch
from typing import Dict, List
from diffusers import DiffusionPipeline
from cache_dit.cache_factory.forward_pattern import ForwardPattern
from cache_dit.cache_factory.cache_adapters import CacheType
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
    pipe: DiffusionPipeline = None,
    adapter_params: BlockAdapterParams = None,
    forward_pattern: ForwardPattern = ForwardPattern.Pattern_0,
    **cache_options_kwargs,
) -> DiffusionPipeline:
    return UnifiedCacheAdapter.apply(
        pipe,
        adapter_params,
        forward_pattern=forward_pattern,
        **cache_options_kwargs,
    )
