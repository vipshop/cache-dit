import torch
from diffusers import DiffusionPipeline
from cache_dit.cache_factory.cache_adapters import CacheType
from cache_dit.cache_factory.cache_adapters import UnifiedCacheAdapter
from cache_dit.cache_factory.utils import load_cache_options_from_yaml

from cache_dit.logger import init_logger

logger = init_logger(__name__)


def load_options(path: str):
    """cache_dit.load_options(cache_config.yaml)"""
    return load_cache_options_from_yaml(path)


def enable_cache(
    pipe: DiffusionPipeline,
    *,
    transformer: torch.nn.Module = None,
    blocks: torch.nn.ModuleList = None,
    # transformer_blocks, blocks, etc.
    blocks_name: str = None,
    dummy_blocks_names: list[str] = [],
    return_hidden_states_first: bool = True,
    return_hidden_states_only: bool = False,
    **cache_options_kwargs,
) -> DiffusionPipeline:
    return UnifiedCacheAdapter.apply(
        pipe,
        transformer,
        blocks,
        blocks_name=blocks_name,
        dummy_blocks_names=dummy_blocks_names,
        return_hidden_states_first=return_hidden_states_first,
        return_hidden_states_only=return_hidden_states_only,
        **cache_options_kwargs,
    )
