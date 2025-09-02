import torch
import dataclasses
import diffusers
import numpy as np
from pprint import pprint
from diffusers import DiffusionPipeline

from typing import Dict, Any
from cache_dit.logger import init_logger


logger = init_logger(__name__)


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
    pipe_or_module: DiffusionPipeline | torch.nn.Module | Any,
    details: bool = False,
    logging: bool = True,
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


def strify(
    pipe_or_stats: DiffusionPipeline | CacheStats | Dict[str, Any],
) -> str:
    if isinstance(pipe_or_stats, DiffusionPipeline):
        stats = summary(pipe_or_stats, logging=False)
        cache_options = stats.cache_options
        cached_steps = len(stats.cached_steps)
    elif isinstance(pipe_or_stats, CacheStats):
        stats = pipe_or_stats
        cache_options = stats.cache_options
        cached_steps = len(stats.cached_steps)
    elif isinstance(pipe_or_stats, dict):
        from cache_dit.cache_factory import CacheType

        # Assume cache_context_kwargs
        cache_options = pipe_or_stats
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

    def get_taylorseer_order():
        taylorseer_order = 0
        if "taylorseer_kwargs" in cache_options:
            if "n_derivatives" in cache_options["taylorseer_kwargs"]:
                taylorseer_order = cache_options["taylorseer_kwargs"][
                    "n_derivatives"
                ]
        elif "taylorseer_order" in cache_options:
            taylorseer_order = cache_options["taylorseer_order"]
        return taylorseer_order

    cache_type_str = (
        f"DBCACHE_F{cache_options.get('Fn_compute_blocks', 1)}"
        f"B{cache_options.get('Bn_compute_blocks', 0)}_"
        f"W{cache_options.get('max_warmup_steps', 0)}"
        f"M{max(0, cache_options.get('max_cached_steps', -1))}"
        f"MC{max(0, cache_options.get('max_continuous_cached_steps', -1))}_"
        f"T{int(cache_options.get('enable_taylorseer', False))}"
        f"O{get_taylorseer_order()}_"
        f"R{cache_options.get('residual_diff_threshold', 0.08)}"
    )

    if cached_steps:
        cache_type_str += f"_S{cached_steps}"

    return cache_type_str
