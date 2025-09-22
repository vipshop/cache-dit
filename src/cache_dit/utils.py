import torch
import dataclasses
import diffusers
import builtins as __builtin__
import contextlib

import numpy as np
from pprint import pprint
from diffusers import DiffusionPipeline

from typing import Dict, Any, List, Union
from cache_dit.cache_factory import CacheType
from cache_dit.cache_factory import BlockAdapter
from cache_dit.cache_factory import BasicCacheConfig
from cache_dit.cache_factory import CalibratorConfig
from cache_dit.logger import init_logger


logger = init_logger(__name__)


def dummy_print(*args, **kwargs):
    pass


@contextlib.contextmanager
def disable_print():
    origin_print = __builtin__.print
    __builtin__.print = dummy_print
    yield
    __builtin__.print = origin_print


@torch.compiler.disable
def is_diffusers_at_least_0_3_5() -> bool:
    return diffusers.__version__ >= "0.35.0"


@dataclasses.dataclass
class CacheStats:
    cache_options: dict = dataclasses.field(default_factory=dict)
    cached_steps: list[int] = dataclasses.field(default_factory=list)
    residual_diffs: dict[str, float] = dataclasses.field(default_factory=dict)
    cfg_cached_steps: list[int] = dataclasses.field(default_factory=list)
    cfg_residual_diffs: dict[str, float] = dataclasses.field(
        default_factory=dict
    )


def summary(
    adapter_or_others: Union[
        BlockAdapter,
        DiffusionPipeline,
        torch.nn.Module,
    ],
    details: bool = False,
    logging: bool = True,
    **kwargs,
) -> List[CacheStats]:
    if adapter_or_others is None:
        return [CacheStats()]

    if not isinstance(adapter_or_others, BlockAdapter):
        if not isinstance(adapter_or_others, DiffusionPipeline):
            transformer = adapter_or_others
            transformer_2 = None
        else:
            transformer = adapter_or_others.transformer
            transformer_2 = None
            if hasattr(adapter_or_others, "transformer_2"):
                transformer_2 = adapter_or_others.transformer_2

        if not BlockAdapter.is_cached(transformer):
            return [CacheStats()]

        blocks_stats: List[CacheStats] = []
        for blocks in BlockAdapter.find_blocks(transformer):
            blocks_stats.append(
                _summary(
                    blocks,
                    details=details,
                    logging=logging,
                    **kwargs,
                )
            )

        if transformer_2 is not None:
            for blocks in BlockAdapter.find_blocks(transformer_2):
                blocks_stats.append(
                    _summary(
                        blocks,
                        details=details,
                        logging=logging,
                        **kwargs,
                    )
                )

        blocks_stats.append(
            _summary(
                transformer,
                details=details,
                logging=logging,
                **kwargs,
            )
        )
        if transformer_2 is not None:
            blocks_stats.append(
                _summary(
                    transformer_2,
                    details=details,
                    logging=logging,
                    **kwargs,
                )
            )

        blocks_stats = [stats for stats in blocks_stats if stats.cache_options]

        return blocks_stats if len(blocks_stats) else [CacheStats()]

    adapter = adapter_or_others
    if not BlockAdapter.check_block_adapter(adapter):
        return [CacheStats()]

    blocks_stats = []
    flatten_blocks = BlockAdapter.flatten(adapter.blocks)
    for blocks in flatten_blocks:
        blocks_stats.append(
            _summary(
                blocks,
                details=details,
                logging=logging,
                **kwargs,
            )
        )

    blocks_stats = [stats for stats in blocks_stats if stats.cache_options]

    return blocks_stats if len(blocks_stats) else [CacheStats()]


def strify(
    adapter_or_others: Union[
        BlockAdapter,
        DiffusionPipeline,
        CacheStats,
        List[CacheStats],
        Dict[str, Any],
    ],
) -> str:
    if isinstance(adapter_or_others, BlockAdapter):
        stats = summary(adapter_or_others, logging=False)[-1]
        cache_options = stats.cache_options
        cached_steps = len(stats.cached_steps)
    elif isinstance(adapter_or_others, DiffusionPipeline):
        stats = summary(adapter_or_others, logging=False)[-1]
        cache_options = stats.cache_options
        cached_steps = len(stats.cached_steps)
    elif isinstance(adapter_or_others, CacheStats):
        stats = adapter_or_others
        cache_options = stats.cache_options
        cached_steps = len(stats.cached_steps)
    elif isinstance(adapter_or_others, list):
        stats = adapter_or_others[0]
        cache_options = stats.cache_options
        cached_steps = len(stats.cached_steps)
    elif isinstance(adapter_or_others, dict):

        # Assume cache_context_kwargs
        cache_options = adapter_or_others
        cached_steps = None
        cache_type = cache_options.get("cache_type", CacheType.NONE)

        if cache_type == CacheType.NONE:
            return "NONE"
    else:
        raise ValueError(
            "Please set pipe_or_stats param as one of: "
            "DiffusionPipeline | CacheStats | Dict[str, Any]"
        )

    if not cache_options:
        return "NONE"

    def basic_cache_str():
        cache_config: BasicCacheConfig = cache_options.get("cache_config", None)
        if cache_config is not None:
            return cache_config.strify()
        return "NONE"

    def calibrator_str():
        calibrator_config: CalibratorConfig = cache_options.get(
            "calibrator_config", None
        )
        if calibrator_config is not None:
            return calibrator_config.strify()
        return "T0O0"

    cache_type_str = f"{basic_cache_str()}_{calibrator_str()}"

    if cached_steps:
        cache_type_str += f"_S{cached_steps}"

    return cache_type_str


def _summary(
    pipe_or_module: Union[
        DiffusionPipeline,
        torch.nn.Module,
    ],
    details: bool = False,
    logging: bool = True,
    **kwargs,
) -> CacheStats:
    cache_stats = CacheStats()

    if not isinstance(pipe_or_module, torch.nn.Module):
        assert hasattr(pipe_or_module, "transformer")
        module = pipe_or_module.transformer
        cls_name = module.__class__.__name__
    else:
        module = pipe_or_module

    cls_name = module.__class__.__name__
    if isinstance(module, torch.nn.ModuleList):
        cls_name = module[0].__class__.__name__

    if hasattr(module, "_cache_context_kwargs"):
        cache_options = module._cache_context_kwargs
        cache_stats.cache_options = cache_options
        if logging:
            print(f"\n🤗Cache Options: {cls_name}\n\n{cache_options}")
    else:
        if logging:
            logger.warning(f"Can't find Cache Options for: {cls_name}")

    if hasattr(module, "_cached_steps"):
        cached_steps: list[int] = module._cached_steps
        residual_diffs: dict[str, float] = dict(module._residual_diffs)
        cache_stats.cached_steps = cached_steps
        cache_stats.residual_diffs = residual_diffs

        if residual_diffs and logging:
            diffs_values = list(residual_diffs.values())
            qmin = np.min(diffs_values)
            q0 = np.percentile(diffs_values, 0)
            q1 = np.percentile(diffs_values, 25)
            q2 = np.percentile(diffs_values, 50)
            q3 = np.percentile(diffs_values, 75)
            q4 = np.percentile(diffs_values, 95)
            qmax = np.max(diffs_values)

            print(
                f"\n⚡️Cache Steps and Residual Diffs Statistics: {cls_name}\n"
            )

            print(
                "| Cache Steps | Diffs P00 | Diffs P25 | Diffs P50 | Diffs P75 | Diffs P95 | Diffs Min | Diffs Max |"
            )
            print(
                "|-------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|"
            )
            print(
                f"| {len(cached_steps):<11} | {round(q0, 3):<9} | {round(q1, 3):<9} "
                f"| {round(q2, 3):<9} | {round(q3, 3):<9} | {round(q4, 3):<9} "
                f"| {round(qmin, 3):<9} | {round(qmax, 3):<9} |"
            )
            print("")

            if details:
                print(f"📚Cache Steps and Residual Diffs Details: {cls_name}\n")
                pprint(
                    f"Cache Steps: {len(cached_steps)}, {cached_steps}",
                )
                pprint(
                    f"Residual Diffs: {len(residual_diffs)}, {residual_diffs}",
                    compact=True,
                )

    if hasattr(module, "_cfg_cached_steps"):
        cfg_cached_steps: list[int] = module._cfg_cached_steps
        cfg_residual_diffs: dict[str, float] = dict(module._cfg_residual_diffs)
        cache_stats.cfg_cached_steps = cfg_cached_steps
        cache_stats.cfg_residual_diffs = cfg_residual_diffs

        if cfg_residual_diffs and logging:
            cfg_diffs_values = list(cfg_residual_diffs.values())
            qmin = np.min(cfg_diffs_values)
            q0 = np.percentile(cfg_diffs_values, 0)
            q1 = np.percentile(cfg_diffs_values, 25)
            q2 = np.percentile(cfg_diffs_values, 50)
            q3 = np.percentile(cfg_diffs_values, 75)
            q4 = np.percentile(cfg_diffs_values, 95)
            qmax = np.max(cfg_diffs_values)

            print(
                f"\n⚡️CFG Cache Steps and Residual Diffs Statistics: {cls_name}\n"
            )

            print(
                "| CFG Cache Steps | Diffs P00 | Diffs P25 | Diffs P50 | Diffs P75 | Diffs P95 | Diffs Min | Diffs Max |"
            )
            print(
                "|-----------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|"
            )
            print(
                f"| {len(cfg_cached_steps):<15} | {round(q0, 3):<9} | {round(q1, 3):<9} "
                f"| {round(q2, 3):<9} | {round(q3, 3):<9} | {round(q4, 3):<9} "
                f"| {round(qmin, 3):<9} | {round(qmax, 3):<9} |"
            )
            print("")

            if details:
                print(
                    f"📚CFG Cache Steps and Residual Diffs Details: {cls_name}\n"
                )
                pprint(
                    f"CFG Cache Steps: {len(cfg_cached_steps)}, {cfg_cached_steps}",
                )
                pprint(
                    f"CFG Residual Diffs: {len(cfg_residual_diffs)}, {cfg_residual_diffs}",
                    compact=True,
                )

    return cache_stats
