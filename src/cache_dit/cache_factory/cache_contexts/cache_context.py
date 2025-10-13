import logging
import dataclasses
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Union, Tuple

import torch

from cache_dit.cache_factory.cache_types import CacheType
from cache_dit.cache_factory.cache_contexts.calibrators import (
    Calibrator,
    CalibratorBase,
    CalibratorConfig,
)
from cache_dit.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class BasicCacheConfig:
    # Default: Dual Block Cache with Flexible FnBn configuration.
    cache_type: CacheType = CacheType.DBCache  # DBCache, DBPrune, NONE

    # Fn_compute_blocks: (`int`, *required*, defaults to 8):
    #     Specifies that `DBCache` uses the **first n** Transformer blocks to fit the information
    #     at time step t, enabling the calculation of a more stable L1 diff and delivering more
    #     accurate information to subsequent blocks. Please check https://github.com/vipshop/cache-dit/blob/main/docs/DBCache.md
    #     for more details of DBCache.
    Fn_compute_blocks: int = 8
    # Bn_compute_blocks: (`int`, *required*, defaults to 0):
    #     Further fuses approximate information in the **last n** Transformer blocks to enhance
    #     prediction accuracy. These blocks act as an auto-scaler for approximate hidden states
    #     that use residual cache.
    Bn_compute_blocks: int = 0
    # residual_diff_threshold (`float`, *required*, defaults to 0.08):
    #     the value of residual diff threshold, a higher value leads to faster performance at the
    #     cost of lower precision.
    residual_diff_threshold: Union[torch.Tensor, float] = 0.08
    # max_warmup_steps (`int`, *required*, defaults to 8):
    #     DBCache does not apply the caching strategy when the number of running steps is less than
    #     or equal to this value, ensuring the model sufficiently learns basic features during warmup.
    max_warmup_steps: int = 8  # DON'T Cache in warmup steps
    # warmup_interval (`int`, *required*, defaults to 1):
    #     Skip interval in warmup steps, e.g., when warmup_interval is 2, only 0, 2, 4, ... steps
    #     in warmup steps will be computed, others will use dynamic cache.
    warmup_interval: int = 1  # skip interval in warmup steps
    # max_cached_steps (`int`, *required*, defaults to -1):
    #     DBCache disables the caching strategy when the previous cached steps exceed this value to
    #     prevent precision degradation.
    max_cached_steps: int = -1  # for both CFG and non-CFG
    # max_continuous_cached_steps (`int`, *required*, defaults to -1):
    #     DBCache disables the caching strategy when the previous continous cached steps exceed this value to
    #     prevent precision degradation.
    max_continuous_cached_steps: int = -1  # the max continuous cached steps
    # enable_separate_cfg (`bool`, *required*,  defaults to None):
    #     Whether to do separate cfg or not, such as Wan 2.1, Qwen-Image. For model that fused CFG
    #     and non-CFG into single forward step, should set enable_separate_cfg as False, for example:
    #     CogVideoX, HunyuanVideo, Mochi, etc.
    enable_separate_cfg: Optional[bool] = None
    # cfg_compute_first (`bool`, *required*,  defaults to False):
    #     Compute cfg forward first or not, default False, namely, 0, 2, 4, ..., -> non-CFG step;
    #     1, 3, 5, ... -> CFG step.
    cfg_compute_first: bool = False
    # cfg_diff_compute_separate (`bool`, *required*,  defaults to True):
    #     Compute separate diff values for CFG and non-CFG step, default True. If False, we will
    #     use the computed diff from current non-CFG transformer step for current CFG step.
    cfg_diff_compute_separate: bool = True

    def update(self, **kwargs) -> "BasicCacheConfig":
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def strify(self) -> str:
        return (
            f"{self.cache_type}_"
            f"F{self.Fn_compute_blocks}"
            f"B{self.Bn_compute_blocks}_"
            f"W{self.max_warmup_steps}"
            f"I{self.warmup_interval}"
            f"M{max(0, self.max_cached_steps)}"
            f"MC{max(0, self.max_continuous_cached_steps)}_"
            f"R{self.residual_diff_threshold}"
        )


@dataclasses.dataclass
class ExtraCacheConfig:
    # Some other not very important settings for Dual Block Cache.
    # NOTE: These flags maybe deprecated in the future and users
    # should never use these extra configurations in their cases.

    # l1_hidden_states_diff_threshold (`float`, *optional*, defaults to None):
    #     The hidden states diff threshold for DBCache if use hidden_states as
    #     cache (not residual).
    l1_hidden_states_diff_threshold: float = None
    # important_condition_threshold (`float`, *optional*, defaults to 0.0):
    #     Only select the most important tokens while calculating the l1 diff.
    important_condition_threshold: float = 0.0
    # downsample_factor (`int`, *optional*, defaults to 1):
    #     Downsample factor for Fn buffer, in order the save GPU memory.
    downsample_factor: int = 1
    # num_inference_steps (`int`, *optional*, defaults to -1):
    #     num_inference_steps for DiffusionPipeline, for future use.
    num_inference_steps: int = -1


@dataclasses.dataclass
class DBCacheConfig(BasicCacheConfig):
    pass  # Just an alias for BasicCacheConfig


@dataclasses.dataclass
class DBPruneConfig(BasicCacheConfig):
    # Dyanamic Block Prune specific configurations
    cache_type: CacheType = CacheType.DBPrune  # DBPrune

    # enable_dynamic_prune_threshold (`bool`, *required*, defaults to False):
    #     Whether to enable the dynamic prune threshold or not. If True, we will
    #     compute the dynamic prune threshold based on the mean of the residual
    #     diffs of the previous computed or pruned blocks.
    #     But, also limit mean_diff to be at least 2x the residual_diff_threshold
    #     to avoid too aggressive pruning.
    enable_dynamic_prune_threshold: bool = False
    # max_dynamic_prune_threshold (`float`, *optional*, defaults to None):
    #     The max dynamic prune threshold, if not None, the dynamic prune threshold
    #     will not exceed this value. If None, we will limit it to be at least 2x
    #     the residual_diff_threshold to avoid too aggressive pruning.
    max_dynamic_prune_threshold: float = None
    # dynamic_prune_threshold_relax_ratio (`float`, *optional*, defaults to 1.25):
    #     The relax ratio for dynamic prune threshold, the dynamic prune threshold
    #     will be set as:
    #         dynamic_prune_threshold = mean_diff * dynamic_prune_threshold_relax_ratio
    #     to avoid too aggressive pruning.
    #     The default value is 1.25, which means the dynamic prune threshold will
    #     be 1.25 times the mean of the residual diffs of the previous computed
    #     or pruned blocks.
    #     Users can tune this value to achieve a better trade-off between speedup
    #     and precision. A higher value leads to more aggressive pruning
    #     and faster speedup, but may also lead to lower precision.
    dynamic_prune_threshold_relax_ratio: float = 1.25

    def strify(self) -> str:
        return (
            f"{self.cache_type}_"
            f"W{self.max_warmup_steps}"
            f"I{self.warmup_interval}"
            f"M{max(0, self.max_cached_steps)}"
            f"MC{max(0, self.max_continuous_cached_steps)}_"
            f"R{self.residual_diff_threshold}_"
            f"DPT{int(self.enable_dynamic_prune_threshold)}"
            f"D{self.dynamic_prune_threshold_relax_ratio}"
        )


@dataclasses.dataclass
class CachedContext:
    name: str = "default"
    # Buffer for storing the residuals and other tensors
    buffers: Dict[str, Any] = dataclasses.field(default_factory=dict)
    # Basic Dual Block Cache Config
    cache_config: Union[
        BasicCacheConfig,
        DBCacheConfig,
        DBPruneConfig,
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
    continuous_cached_steps: int = 0
    cfg_cached_steps: List[int] = dataclasses.field(default_factory=list)
    cfg_residual_diffs: DefaultDict[str, float | list] = dataclasses.field(
        default_factory=lambda: defaultdict(float),
    )
    cfg_continuous_cached_steps: int = 0

    # Specially for Dynamic Block Prune
    pruned_blocks: List[int] = dataclasses.field(default_factory=list)
    actual_blocks: List[int] = dataclasses.field(default_factory=list)
    cfg_pruned_blocks: List[int] = dataclasses.field(default_factory=list)
    cfg_actual_blocks: List[int] = dataclasses.field(default_factory=list)

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
                    self.cfg_encoder_calibrator = Calibrator(
                        self.calibrator_config
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

    def has_calibrators(self) -> bool:
        if self.calibrator_config is not None:
            return (
                self.calibrator_config.enable_calibrator
                or self.calibrator_config.enable_encoder_calibrator
            )
        return False

    def get_residual_diff_threshold(self):
        # Dynamic Block Prune
        if self.cache_config.cache_type == CacheType.DBPrune:
            residual_diff_threshold = self.cache_config.residual_diff_threshold
            if isinstance(residual_diff_threshold, torch.Tensor):
                residual_diff_threshold = residual_diff_threshold.item()
            if self.cache_config.enable_dynamic_prune_threshold:
                # Compute the dynamic prune threshold based on the mean of the
                # residual diffs of the previous computed or pruned blocks.
                step = self.get_current_step()
                if step >= 0 and step in self.residual_diffs:
                    assert isinstance(self.residual_diffs[step], list)
                    # Use all the recorded diffs for this step
                    # NOTE: Should we only use the last 5 diffs?
                    diffs = self.residual_diffs[step][:5]
                    diffs = [d for d in diffs if d > 0.0]
                    if diffs:
                        mean_diff = sum(diffs) / len(diffs)
                        relaxed_diff = (
                            mean_diff
                            * self.cache_config.dynamic_prune_threshold_relax_ratio
                        )
                        if (
                            self.cache_config.max_dynamic_prune_threshold
                            is None
                        ):
                            max_dynamic_prune_threshold = (
                                2 * residual_diff_threshold
                            )
                        else:
                            max_dynamic_prune_threshold = (
                                self.cache_config.max_dynamic_prune_threshold
                            )
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
        # Dual Block Cache
        elif self.cache_config.cache_type == CacheType.DBCache:
            residual_diff_threshold = self.cache_config.residual_diff_threshold
            if (
                self.extra_cache_config.l1_hidden_states_diff_threshold
                is not None
            ):
                # Use the L1 hidden states diff threshold if set
                residual_diff_threshold = (
                    self.extra_cache_config.l1_hidden_states_diff_threshold
                )
            if isinstance(residual_diff_threshold, torch.Tensor):
                residual_diff_threshold = residual_diff_threshold.item()
            return residual_diff_threshold
        else:
            raise ValueError(
                f"Unsupported cache type: {self.cache_config.cache_type}"
            )

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
            self.pruned_blocks.clear()
            self.actual_blocks.clear()
            # Reset the calibrators cache at the beginning of each inference.
            # reset_cache will set the current step to -1 for calibrator,
            if self.has_calibrators():
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

    def add_residual_diff(self, diff: float | torch.Tensor):
        if isinstance(diff, torch.Tensor):
            diff = diff.item()
        # step: executed_steps - 1, not transformer_steps - 1
        step = str(self.get_current_step())
        if self.cache_config.cache_type == CacheType.DBPrune:
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
        elif self.cache_config.cache_type == CacheType.DBCache:
            # Only add the diff if it is not already recorded for this step
            if not self.is_separate_cfg_step():
                if step not in self.residual_diffs:
                    self.residual_diffs[step] = diff
            else:
                if step not in self.cfg_residual_diffs:
                    self.cfg_residual_diffs[step] = diff
        else:
            raise ValueError(
                f"Unsupported cache type: {self.cache_config.cache_type}"
            )

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

    def add_pruned_step(self):
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

    def get_cached_steps(self):
        return self.cached_steps.copy()

    def get_cfg_cached_steps(self):
        return self.cfg_cached_steps.copy()

    def get_pruned_steps(self):
        return self.get_cached_steps()

    def get_cfg_pruned_steps(self):
        return self.get_cfg_cached_steps()

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
        return list(
            range(
                0,
                self.cache_config.max_warmup_steps,
                self.cache_config.warmup_interval,
            )
        )

    def is_in_warmup(self):
        return self.get_current_step() in self.warmup_steps
