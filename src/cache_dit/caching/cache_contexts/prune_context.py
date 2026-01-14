import torch
import logging
import dataclasses
from typing import List
from ..cache_types import CacheType
from .prune_config import DBPruneConfig
from .cache_context import CachedContext
from cache_dit.logger import init_logger


logger = init_logger(__name__)


@dataclasses.dataclass
class PrunedContext(CachedContext):
    # Overwrite the cache_config type for PrunedContext
    cache_config: DBPruneConfig = dataclasses.field(
        default_factory=DBPruneConfig,
    )
    # Specially for Dynamic Block Prune
    pruned_blocks: List[int] = dataclasses.field(default_factory=list)
    actual_blocks: List[int] = dataclasses.field(default_factory=list)
    cfg_pruned_blocks: List[int] = dataclasses.field(default_factory=list)
    cfg_actual_blocks: List[int] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        if not isinstance(self.cache_config, DBPruneConfig):
            raise ValueError("PrunedContext only supports DBPruneConfig as cache_config.")

        if self.cache_config.cache_type == CacheType.DBPrune:
            if (
                self.calibrator_config is not None
                and self.cache_config.force_reduce_calibrator_vram
            ):
                # May reduce VRAM usage for Dynamic Block Prune
                self.extra_cache_config.downsample_factor = max(
                    4, self.extra_cache_config.downsample_factor
                )

    def get_residual_diff_threshold(self):
        # Overwite this func for Dynamic Block Prune
        residual_diff_threshold = self.cache_config.residual_diff_threshold
        if isinstance(residual_diff_threshold, torch.Tensor):
            residual_diff_threshold = residual_diff_threshold.item()
        if self.cache_config.enable_dynamic_prune_threshold:
            # Compute the dynamic prune threshold based on the mean of the
            # residual diffs of the previous computed or pruned blocks.
            step = str(self.get_current_step())
            if int(step) >= 0 and str(step) in self.residual_diffs:
                assert isinstance(self.residual_diffs[step], list)
                # Use all the recorded diffs for this step
                # NOTE: Should we only use the last 5 diffs?
                diffs = self.residual_diffs[step][:5]
                diffs = [d for d in diffs if d > 0.0]
                if diffs:
                    mean_diff = sum(diffs) / len(diffs)
                    relaxed_diff = mean_diff * self.cache_config.dynamic_prune_threshold_relax_ratio
                    if self.cache_config.max_dynamic_prune_threshold is None:
                        max_dynamic_prune_threshold = 2 * residual_diff_threshold
                    else:
                        max_dynamic_prune_threshold = self.cache_config.max_dynamic_prune_threshold
                    if relaxed_diff < max_dynamic_prune_threshold:
                        # If the mean diff is less than twice the threshold,
                        # we can use it as the dynamic prune threshold.
                        residual_diff_threshold = (
                            relaxed_diff
                            if relaxed_diff > residual_diff_threshold
                            else residual_diff_threshold
                        )
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"Dynamic prune threshold for step {step}: "
                            f"{residual_diff_threshold:.6f}"
                        )
        return residual_diff_threshold

    def mark_step_begin(self):
        # Overwite this func for Dynamic Block Prune
        super().mark_step_begin()
        # Reset pruned_blocks and actual_blocks at the beginning
        # of each transformer step.
        if self.get_current_transformer_step() == 0:
            self.pruned_blocks.clear()
            self.actual_blocks.clear()

    def add_residual_diff(self, diff: float | torch.Tensor):
        # Overwite this func for Dynamic Block Prune
        if isinstance(diff, torch.Tensor):
            diff = diff.item()
        # step: executed_steps - 1, not transformer_steps - 1
        step = str(self.get_current_step())
        # For Dynamic Block Prune, we will record all the diffs for this step
        # Only add the diff if it is not already recorded for this step
        if not self.is_separate_cfg_step():
            if step not in self.residual_diffs:
                self.residual_diffs[step] = []
            self.residual_diffs[step].append(diff)
        else:
            if step not in self.cfg_residual_diffs:
                self.cfg_residual_diffs[step] = []
            self.cfg_residual_diffs[step].append(diff)

    def add_pruned_step(self):
        curr_cached_step = self.get_current_step()
        # Avoid adding the same step multiple times
        if not self.is_separate_cfg_step():
            if curr_cached_step not in self.cached_steps:
                self.add_cached_step()
        else:
            if curr_cached_step not in self.cfg_cached_steps:
                self.add_cached_step()

    def add_pruned_block(self, num_blocks):
        if not self.is_separate_cfg_step():
            self.pruned_blocks.append(num_blocks)
        else:
            self.cfg_pruned_blocks.append(num_blocks)

    def add_actual_block(self, num_blocks):
        if not self.is_separate_cfg_step():
            self.actual_blocks.append(num_blocks)
        else:
            self.cfg_actual_blocks.append(num_blocks)

    def get_pruned_blocks(self):
        return self.pruned_blocks.copy()

    def get_cfg_pruned_blocks(self):
        return self.cfg_pruned_blocks.copy()

    def get_actual_blocks(self):
        return self.actual_blocks.copy()

    def get_cfg_actual_blocks(self):
        return self.cfg_actual_blocks.copy()

    def get_pruned_steps(self):
        return self.get_cached_steps()

    def get_cfg_pruned_steps(self):
        return self.get_cfg_cached_steps()
