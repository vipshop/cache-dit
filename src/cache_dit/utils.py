import torch
import dataclasses
import diffusers
import numpy as np
from pprint import pprint
from diffusers import DiffusionPipeline

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
    pipe: DiffusionPipeline, details: bool = False, logging: bool = True
):
    cache_stats = CacheStats()
    pipe_cls_name = pipe.__class__.__name__

    if hasattr(pipe, "_cache_options"):
        cache_options = pipe._cache_options
        cache_stats.cache_options = cache_options
        if logging:
            print(f"\n🤗Cache Options: {pipe_cls_name}\n\n{cache_options}")

    if hasattr(pipe.transformer, "_cached_steps"):
        cached_steps: list[int] = pipe.transformer._cached_steps
        residual_diffs: dict[str, float] = dict(
            pipe.transformer._residual_diffs
        )
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
                f"\n⚡️Cache Steps and Residual Diffs Statistics: {pipe_cls_name}\n"
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
                print(
                    f"📚Cache Steps and Residual Diffs Details: {pipe_cls_name}\n"
                )
                pprint(
                    f"Cache Steps: {len(cached_steps)}, {cached_steps}",
                )
                pprint(
                    f"Residual Diffs: {len(residual_diffs)}, {residual_diffs}",
                    compact=True,
                )

    if hasattr(pipe.transformer, "_cfg_cached_steps"):
        cfg_cached_steps: list[int] = pipe.transformer._cfg_cached_steps
        cfg_residual_diffs: dict[str, float] = dict(
            pipe.transformer._cfg_residual_diffs
        )
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
                f"\n⚡️CFG Cache Steps and Residual Diffs Statistics: {pipe_cls_name}\n"
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
                    f"📚CFG Cache Steps and Residual Diffs Details: {pipe_cls_name}\n"
                )
                pprint(
                    f"CFG Cache Steps: {len(cfg_cached_steps)}, {cfg_cached_steps}",
                )
                pprint(
                    f"CFG Residual Diffs: {len(cfg_residual_diffs)}, {cfg_residual_diffs}",
                    compact=True,
                )

    return cache_stats


def strify(pipe_or_stats: DiffusionPipeline | CacheStats):
    if not isinstance(pipe_or_stats, CacheStats):
        stats = summary(pipe_or_stats, logging=False)
    else:
        stats = pipe_or_stats

    cache_options = stats.cache_options
    cached_steps = len(stats.cached_steps)

    if not cache_options:
        return "NONE"

    cache_type_str = (
        f"DBCACHE_F{cache_options['Fn_compute_blocks']}"
        f"B{cache_options['Bn_compute_blocks']}"
        f"W{cache_options['warmup_steps']}"
        f"M{max(0, cache_options['max_cached_steps'])}"
        f"T{int(cache_options['enable_taylorseer'])}"
        f"O{cache_options['taylorseer_kwargs']['n_derivatives']}_"
        f"R{cache_options['residual_diff_threshold']}_"
        f"S{cached_steps}"  # skiped steps
    )

    return cache_type_str
