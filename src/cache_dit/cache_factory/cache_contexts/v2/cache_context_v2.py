import logging
import dataclasses
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Union, Tuple

import torch

from cache_dit.cache_factory.cache_contexts.v2.calibrators import (
    Calibrator,
    CalibratorBase,
)
from cache_dit.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class CalibratorConfigV2:  # no V1
    enable_calibrator: bool = False
    enable_encoder_calibrator: bool = False
    calibrator_type: str = "taylorseer"  # taylorseer or foca, etc.
    calibrator_cache_type: str = "residual"  # residual or hidden_states
    calibrator_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def strify(self):
        if self.calibrator_type == "taylorseer":
            taylorseer_order = self.calibrator_kwargs.get("n_derivatives", 0)
            if taylorseer_order:
                return f"T1O{taylorseer_order}"
            return "T0O0"
        else:
            raise ValueError(
                f"calibrator {self.calibrator_type} is not supported now!"
            )


@dataclasses.dataclass
class CachedContextV2:  # Internal CachedContext Impl class
    name: str = "default"
    # Dual Block Cache with flexible FnBn configuration.
    Fn_compute_blocks: int = 1
    Bn_compute_blocks: int = 0
    # non compute blocks diff threshold, we don't skip the non
    # compute blocks if the diff >= threshold
    non_compute_blocks_diff_threshold: float = 0.08
    max_Fn_compute_blocks: int = -1
    max_Bn_compute_blocks: int = -1
    # L1 hidden states or residual diff threshold for Fn
    residual_diff_threshold: Union[torch.Tensor, float] = 0.05
    l1_hidden_states_diff_threshold: float = None
    important_condition_threshold: float = 0.0

    # Buffer for storing the residuals and other tensors
    buffers: Dict[str, Any] = dataclasses.field(default_factory=dict)
    incremental_name_counters: DefaultDict[str, int] = dataclasses.field(
        default_factory=lambda: defaultdict(int),
    )

    # Other settings
    downsample_factor: int = 1
    num_inference_steps: int = -1  # for future use
    max_warmup_steps: int = 0  # DON'T Cache in warmup steps
    # DON'T Cache if the number of cached steps >= max_cached_steps
    max_cached_steps: int = -1  # for both CFG and non-CFG
    max_continuous_cached_steps: int = -1  # the max continuous cached steps

    # Record the steps that have been cached, both cached and non-cache
    executed_steps: int = 0  # cache + non-cache steps pippeline
    # steps for transformer, for CFG, transformer_executed_steps will
    # be double of executed_steps.
    transformer_executed_steps: int = 0

    # Support calibrators in Dual Block Cache: TaylorSeer, FoCa, etc.
    calibrator_config: Optional[CalibratorConfigV2] = None
    calibrator: Optional[CalibratorBase] = None
    encoder_calibrator: Optional[CalibratorBase] = None

    # Support enable_separate_cfg, such as Wan 2.1,
    # Qwen-Image. For model that fused CFG and non-CFG into single
    # forward step, should set enable_separate_cfg as False.
    # For example: CogVideoX, HunyuanVideo, Mochi.
    enable_separate_cfg: bool = False
    # Compute cfg forward first or not, default False, namely,
    # 0, 2, 4, ..., -> non-CFG step; 1, 3, 5, ... -> CFG step.
    cfg_compute_first: bool = False
    # Compute separate diff values for CFG and non-CFG step,
    # default True. If False, we will use the computed diff from
    # current non-CFG transformer step for current CFG step.
    cfg_diff_compute_separate: bool = True
    cfg_calibrator: Optional[CalibratorBase] = None
    cfg_encoder_calibrator: Optional[CalibratorBase] = None

    # CFG & non-CFG cached steps
    cached_steps: List[int] = dataclasses.field(default_factory=list)
    residual_diffs: DefaultDict[str, float] = dataclasses.field(
        default_factory=lambda: defaultdict(float),
    )
    continuous_cached_steps: int = 0
    cfg_cached_steps: List[int] = dataclasses.field(default_factory=list)
    cfg_residual_diffs: DefaultDict[str, float] = dataclasses.field(
        default_factory=lambda: defaultdict(float),
    )
    cfg_continuous_cached_steps: int = 0

    def __post_init__(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.info(f"Created _CacheContextV2: {self.name}")
        # Some checks for settings
        if self.enable_separate_cfg:
            if self.cfg_diff_compute_separate:
                assert self.cfg_compute_first is False, (
                    "cfg_compute_first must set as False if "
                    "cfg_diff_compute_separate is enabled."
                )

        if self.calibrator_config.enable_calibrator:
            self.calibrator = Calibrator(
                **self.calibrator_config.calibrator_kwargs
            )
            if self.enable_separate_cfg:
                self.cfg_calibrator = Calibrator(
                    **self.calibrator_config.calibrator_kwargs
                )

        if self.calibrator_config.enable_encoder_calibrator:
            self.encoder_calibrator = Calibrator(
                **self.calibrator_config.calibrator_kwargs
            )
            if self.enable_separate_cfg:
                self.cfg_encoder_calibrator = Calibrator(
                    **self.calibrator_config.calibrator_kwargs
                )

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

    def get_residual_diff_threshold(self):
        residual_diff_threshold = self.residual_diff_threshold
        if self.l1_hidden_states_diff_threshold is not None:
            # Use the L1 hidden states diff threshold if set
            residual_diff_threshold = self.l1_hidden_states_diff_threshold
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
        if not self.enable_separate_cfg:
            self.executed_steps += 1
        else:
            # 0,1 -> 0 + 1, 2,3 -> 1 + 1, ...
            if not self.cfg_compute_first:
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
            if (
                self.calibrator_config.enable_calibrator
                or self.calibrator_config.enable_encoder_calibrator
            ):
                calibrator, encoder_calibrator = self.get_calibrators()
                if calibrator is not None:
                    calibrator.reset_cache()
                if encoder_calibrator is not None:
                    encoder_calibrator.reset_cache()
                cfg_calibrator, cfg_encoder_calibrator = (
                    self.get_cfg_calibrators()
                )
                if cfg_calibrator is not None:
                    cfg_calibrator.reset_cache()
                if cfg_encoder_calibrator is not None:
                    cfg_encoder_calibrator.reset_cache()

        # mark_step_begin of calibrator must be called after the cache is reset.
        if (
            self.calibrator_config.enable_calibrator
            or self.calibrator_config.enable_encoder_calibrator
        ):
            if self.enable_separate_cfg:
                # Assume non-CFG steps: 0, 2, 4, 6, ...
                if not self.is_separate_cfg_step():
                    calibrator, encoder_calibrator = self.get_calibrators()
                    if calibrator is not None:
                        calibrator.mark_step_begin()
                    if encoder_calibrator is not None:
                        encoder_calibrator.mark_step_begin()
                else:
                    cfg_calibrator, cfg_encoder_calibrator = (
                        self.get_cfg_calibrators()
                    )
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

    def add_residual_diff(self, diff):
        # step: executed_steps - 1, not transformer_steps - 1
        step = str(self.get_current_step())
        # Only add the diff if it is not already recorded for this step
        if not self.is_separate_cfg_step():
            if step not in self.residual_diffs:
                self.residual_diffs[step] = diff
        else:
            if step not in self.cfg_residual_diffs:
                self.cfg_residual_diffs[step] = diff

    def get_residual_diffs(self):
        return self.residual_diffs.copy()

    def get_cfg_residual_diffs(self):
        return self.cfg_residual_diffs.copy()

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
        if not self.enable_separate_cfg:
            return False
        if self.cfg_compute_first:
            # CFG steps: 0, 2, 4, 6, ...
            return self.get_current_transformer_step() % 2 == 0
        # CFG steps: 1, 3, 5, 7, ...
        return self.get_current_transformer_step() % 2 != 0

    def is_in_warmup(self):
        return self.get_current_step() < self.max_warmup_steps
