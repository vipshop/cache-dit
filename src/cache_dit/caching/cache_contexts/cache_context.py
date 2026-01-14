# The cache context codebase is adapted from FBCache. Over time its codebase
# diverged a lot, and context API is no longer compatible with FBCache.
import logging
import dataclasses
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Union, Tuple

import torch

from .cache_config import (
    BasicCacheConfig,
    ExtraCacheConfig,
    DBCacheConfig,
)
from .calibrators import (
    Calibrator,
    CalibratorBase,
    CalibratorConfig,
)
from cache_dit.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class CachedContext:
    name: str = "default"
    # Buffer for storing the residuals and other tensors
    buffers: Dict[str, Any] = dataclasses.field(default_factory=dict)
    # Basic Dual Block Cache Config
    cache_config: Union[
        BasicCacheConfig,
        DBCacheConfig,
    ] = dataclasses.field(
        default_factory=BasicCacheConfig,
    )
    # NOTE: Users should never use these extra configurations.
    extra_cache_config: ExtraCacheConfig = dataclasses.field(
        default_factory=ExtraCacheConfig,
    )
    # Calibrator config for Dual Block Cache: TaylorSeer, FoCa, etc.
    calibrator_config: Optional[CalibratorConfig] = None

    # Calibrators for both CFG and non-CFG
    calibrator: Optional[CalibratorBase] = None
    encoder_calibrator: Optional[CalibratorBase] = None
    cfg_calibrator: Optional[CalibratorBase] = None
    cfg_encoder_calibrator: Optional[CalibratorBase] = None

    # Record the steps that have been cached, both cached and non-cache
    executed_steps: int = 0  # cache + non-cache steps pippeline
    # steps for transformer, for CFG, transformer_executed_steps will
    # be double of executed_steps.
    transformer_executed_steps: int = 0

    # CFG & non-CFG cached/pruned steps
    cached_steps: List[int] = dataclasses.field(default_factory=list)
    residual_diffs: DefaultDict[str, float | list] = dataclasses.field(
        default_factory=lambda: defaultdict(float),
    )
    accumulated_residual_diff: float = 0.0
    continuous_cached_steps: int = 0
    cfg_cached_steps: List[int] = dataclasses.field(default_factory=list)
    cfg_residual_diffs: DefaultDict[str, float | list] = dataclasses.field(
        default_factory=lambda: defaultdict(float),
    )
    cfg_accumulated_residual_diff: float = 0.0
    cfg_continuous_cached_steps: int = 0

    def __post_init__(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.info(f"Created CachedContext: {self.name}")
        # Some checks for settings
        if self.cache_config.enable_separate_cfg:
            if self.cache_config.cfg_diff_compute_separate:
                assert self.cache_config.cfg_compute_first is False, (
                    "cfg_compute_first must set as False if "
                    "cfg_diff_compute_separate is enabled."
                )

        if self.calibrator_config is not None:
            if self.calibrator_config.enable_calibrator:
                self.calibrator = Calibrator(self.calibrator_config)
                if self.cache_config.enable_separate_cfg:
                    self.cfg_calibrator = Calibrator(self.calibrator_config)

            if self.calibrator_config.enable_encoder_calibrator:
                self.encoder_calibrator = Calibrator(self.calibrator_config)
                if self.cache_config.enable_separate_cfg:
                    self.cfg_encoder_calibrator = Calibrator(self.calibrator_config)

    def enable_calibrator(self):
        if self.calibrator_config is not None:
            return self.calibrator_config.enable_calibrator
        return False

    def enable_encoder_calibrator(self):
        if self.calibrator_config is not None:
            return self.calibrator_config.enable_encoder_calibrator
        return False

    def calibrator_cache_type(self):
        if self.calibrator_config is not None:
            return self.calibrator_config.calibrator_cache_type
        return "residual"

    def has_calibrators(self) -> bool:
        if self.calibrator_config is not None:
            return (
                self.calibrator_config.enable_calibrator
                or self.calibrator_config.enable_encoder_calibrator
            )
        return False

    def get_residual_diff_threshold(self):
        residual_diff_threshold = self.cache_config.residual_diff_threshold
        if self.extra_cache_config.l1_hidden_states_diff_threshold is not None:
            # Use the L1 hidden states diff threshold if set
            residual_diff_threshold = self.extra_cache_config.l1_hidden_states_diff_threshold
        if isinstance(residual_diff_threshold, torch.Tensor):
            residual_diff_threshold = residual_diff_threshold.item()
        return residual_diff_threshold

    def get_buffer(self, name):
        return self.buffers.get(name)

    def set_buffer(self, name, buffer):
        self.buffers[name] = buffer

    def remove_buffer(self, name):
        if name in self.buffers:
            del self.buffers[name]

    def clear_buffers(self):
        self.buffers.clear()

    def mark_step_begin(self):
        # Always increase transformer executed steps
        # incr     step: prev 0 -> 1; prev 1 -> 2
        # current  step: incr step - 1
        self.transformer_executed_steps += 1
        if not self.cache_config.enable_separate_cfg:
            self.executed_steps += 1
        else:
            # 0,1 -> 0 + 1, 2,3 -> 1 + 1, ...
            if not self.cache_config.cfg_compute_first:
                if not self.is_separate_cfg_step():
                    # transformer step: 0,2,4,...
                    self.executed_steps += 1
            else:
                if self.is_separate_cfg_step():
                    # transformer step: 0,2,4,...
                    self.executed_steps += 1

        # Reset the cached steps and residual diffs at the beginning
        # of each inference.
        if self.get_current_transformer_step() == 0:
            self.cached_steps.clear()
            self.residual_diffs.clear()
            self.cfg_cached_steps.clear()
            self.cfg_residual_diffs.clear()
            # Reset the calibrators cache at the beginning of each inference.
            # reset_cache will set the current step to -1 for calibrator,
            if self.has_calibrators():
                calibrator, encoder_calibrator = self.get_calibrators()
                if calibrator is not None:
                    calibrator.reset_cache()
                if encoder_calibrator is not None:
                    encoder_calibrator.reset_cache()
                cfg_calibrator, cfg_encoder_calibrator = self.get_cfg_calibrators()
                if cfg_calibrator is not None:
                    cfg_calibrator.reset_cache()
                if cfg_encoder_calibrator is not None:
                    cfg_encoder_calibrator.reset_cache()

        # mark_step_begin of calibrator must be called after the cache is reset.
        if self.has_calibrators():
            if self.cache_config.enable_separate_cfg:
                # Assume non-CFG steps: 0, 2, 4, 6, ...
                if not self.is_separate_cfg_step():
                    calibrator, encoder_calibrator = self.get_calibrators()
                    if calibrator is not None:
                        calibrator.mark_step_begin()
                    if encoder_calibrator is not None:
                        encoder_calibrator.mark_step_begin()
                else:
                    cfg_calibrator, cfg_encoder_calibrator = self.get_cfg_calibrators()
                    if cfg_calibrator is not None:
                        cfg_calibrator.mark_step_begin()
                    if cfg_encoder_calibrator is not None:
                        cfg_encoder_calibrator.mark_step_begin()
            else:
                calibrator, encoder_calibrator = self.get_calibrators()
                if calibrator is not None:
                    calibrator.mark_step_begin()
                if encoder_calibrator is not None:
                    encoder_calibrator.mark_step_begin()

    def get_calibrators(self) -> Tuple[CalibratorBase, CalibratorBase]:
        return self.calibrator, self.encoder_calibrator

    def get_cfg_calibrators(self) -> Tuple[CalibratorBase, CalibratorBase]:
        return self.cfg_calibrator, self.cfg_encoder_calibrator

    def add_residual_diff(self, diff: float | torch.Tensor):
        if isinstance(diff, torch.Tensor):
            diff = diff.item()
        # step: executed_steps - 1, not transformer_steps - 1
        step = str(self.get_current_step())
        # Only add the diff if it is not already recorded for this step
        if not self.is_separate_cfg_step():
            if step not in self.residual_diffs:
                self.residual_diffs[step] = diff
                if diff > 0.0:
                    self.accumulated_residual_diff += diff
        else:
            if step not in self.cfg_residual_diffs:
                self.cfg_residual_diffs[step] = diff
                if diff > 0.0:
                    self.cfg_accumulated_residual_diff += diff

    def get_residual_diffs(self):
        return self.residual_diffs.copy()

    def get_cfg_residual_diffs(self):
        return self.cfg_residual_diffs.copy()

    def get_accumulated_residual_diff(self):
        return self.accumulated_residual_diff

    def get_cfg_accumulated_residual_diff(self):
        return self.cfg_accumulated_residual_diff

    def add_cached_step(self):
        curr_cached_step = self.get_current_step()
        if not self.is_separate_cfg_step():
            if self.cached_steps:
                prev_cached_step = self.cached_steps[-1]
                if curr_cached_step - prev_cached_step == 1:
                    if self.continuous_cached_steps == 0:
                        self.continuous_cached_steps += 2
                    else:
                        self.continuous_cached_steps += 1
            else:
                self.continuous_cached_steps += 1

            self.cached_steps.append(curr_cached_step)
        else:
            if self.cfg_cached_steps:
                prev_cfg_cached_step = self.cfg_cached_steps[-1]
                if curr_cached_step - prev_cfg_cached_step == 1:
                    if self.cfg_continuous_cached_steps == 0:
                        self.cfg_continuous_cached_steps += 2
                    else:
                        self.cfg_continuous_cached_steps += 1
            else:
                self.cfg_continuous_cached_steps += 1

            self.cfg_cached_steps.append(curr_cached_step)

    def get_cached_steps(self):
        return self.cached_steps.copy()

    def get_cfg_cached_steps(self):
        return self.cfg_cached_steps.copy()

    def get_current_step(self):
        return self.executed_steps - 1

    def get_current_transformer_step(self):
        return self.transformer_executed_steps - 1

    def is_separate_cfg_step(self):
        if not self.cache_config.enable_separate_cfg:
            return False
        if self.cache_config.cfg_compute_first:
            # CFG steps: 0, 2, 4, 6, ...
            return self.get_current_transformer_step() % 2 == 0
        # CFG steps: 1, 3, 5, 7, ...
        return self.get_current_transformer_step() % 2 != 0

    @property
    def warmup_steps(self) -> List[int]:
        # Truncate the warmup steps if steps_compute_mask is provided
        if self.cache_config.steps_computation_mask is not None:
            first_continuous_compute_steps = 0  # first continuous compute steps
            for m in self.cache_config.steps_computation_mask:
                if m == 1:
                    first_continuous_compute_steps += 1
                else:
                    break
            max_warmup = min(
                self.cache_config.max_warmup_steps,
                first_continuous_compute_steps,
            )
            return list(
                range(
                    0,
                    max_warmup,
                    self.cache_config.warmup_interval,
                )
            )

        return list(
            range(
                0,
                self.cache_config.max_warmup_steps,
                self.cache_config.warmup_interval,
            )
        )

    def is_in_warmup(self):
        return self.get_current_step() in self.warmup_steps

    def is_in_full_compute_steps(self):
        if self.cache_config.steps_computation_mask is None:
            return False
        current_step = self.get_current_step()
        if current_step < len(self.cache_config.steps_computation_mask):
            return self.cache_config.steps_computation_mask[current_step] == 1
        return False

    def get_steps_computation_policy(self):
        # If enabled steps_computation_mask w/ static cache, maybe use at the very
        # beginning of cache blocks forward. TODO: maybe support NO-Fn blocks compute
        # first for static cache.
        return self.cache_config.steps_computation_policy
