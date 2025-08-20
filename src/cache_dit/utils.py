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
    cached_steps: list[int] = dataclasses.field(default_factory=list)
    residual_diffs: dict[str, float] = dataclasses.field(default_factory=dict)
    cfg_cached_steps: list[int] = dataclasses.field(default_factory=list)
    cfg_residual_diffs: dict[str, float] = dataclasses.field(
        default_factory=dict
    )


def summary(pipe: DiffusionPipeline, details: bool = False):
    cache_stats = CacheStats()
    pipe_cls_name = pipe.__class__.__name__

    if hasattr(pipe, "_cache_options"):
        cache_options = pipe._cache_options
        print(f"\nCache Options: {pipe_cls_name}\n{cache_options}")

    if hasattr(pipe.transformer, "_cached_steps"):
        cached_steps: list[int] = pipe.transformer._cached_steps
        residual_diffs: dict[str, float] = pipe.transformer._residual_diffs
        cache_stats.cached_steps = cached_steps
        cache_stats.residual_diffs = residual_diffs

        if residual_diffs:
            diffs_values = list(residual_diffs.values())
            q0 = np.percentile(diffs_values, 0)
            q1 = np.percentile(diffs_values, 25)
            q2 = np.percentile(diffs_values, 50)
            q3 = np.percentile(diffs_values, 75)
            q4 = np.percentile(diffs_values, 95)

            print(
                f"\nCache Steps and Residual Diffs Statistics: {pipe_cls_name}\n"
            )

            print(
                "| Cache Steps | Diffs P00 | Diffs P25 | Diffs P50 | Diffs P75 | Diffs P95 |"
            )
            print(
                "|-------------|-----------|-----------|-----------|-----------|-----------|"
            )
            print(
                f"| {len(cached_steps):<11} | {round(q0, 3):<9} | {round(q1, 3):<9} "
                f"| {round(q2, 3):<9} | {round(q3, 3):<9} | {round(q4, 3):<9} |"
            )

            if details:
                print(
                    f"\nCache Steps and Residual Diffs Details: {pipe_cls_name}\n"
                )
                print("-" * 200)
                pprint(
                    f"Cache Steps: {len(cached_steps)}, {cached_steps}",
                    width=200,
                )
                pprint(
                    f"Residual Diffs: {len(residual_diffs)}, {residual_diffs}",
                    compact=True,
                    width=200,
                )
                print("-" * 200)

    if hasattr(pipe.transformer, "_cfg_cached_steps"):
        cfg_cached_steps: list[int] = pipe.transformer._cfg_cached_steps
        cfg_residual_diffs: dict[str, float] = (
            pipe.transformer._cfg_residual_diffs
        )
        cache_stats.cfg_cached_steps = cfg_cached_steps
        cache_stats.cfg_residual_diffs = cfg_residual_diffs

        if cfg_residual_diffs:
            cfg_diffs_values = list(cfg_residual_diffs.values())
            q0 = np.percentile(cfg_diffs_values, 0)
            q1 = np.percentile(cfg_diffs_values, 25)
            q2 = np.percentile(cfg_diffs_values, 50)
            q3 = np.percentile(cfg_diffs_values, 75)
            q4 = np.percentile(cfg_diffs_values, 95)

            print(
                f"\nCFG Cache Steps and Residual Diffs Statistics: {pipe_cls_name}\n"
            )

            print(
                "| CFG Cache Steps | Diffs P00 | Diffs P25 | Diffs P50 | Diffs P75 | Diffs P95 |"
            )
            print(
                "|-----------------|-----------|-----------|-----------|-----------|-----------|"
            )
            print(
                f"| {len(cfg_cached_steps):<15} | {round(q0, 3):<9} | {round(q1, 3):<9} "
                f"| {round(q2, 3):<9} | {round(q3, 3):<9} | {round(q4, 3):<9} |"
            )

            if details:
                print(
                    f"\nCFG Cache Steps and Residual Diffs Details: {pipe_cls_name}\n"
                )
                print("-" * 200)
                pprint(
                    f"CFG Cache Steps: {len(cfg_cached_steps)}, {cfg_cached_steps}",
                    width=200,
                )
                pprint(
                    f"CFG Residual Diffs: {len(cfg_residual_diffs)}, {cfg_residual_diffs}",
                    compact=True,
                    width=200,
                )
                print("-" * 200)

    return cache_stats
