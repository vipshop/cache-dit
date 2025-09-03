import logging
import dataclasses
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Union, Tuple

import torch

from cache_dit.cache_factory.cache_contexts.taylorseer import TaylorSeer
from cache_dit.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class CachedContext:  # Internal CachedContext Impl class
    name: str = "default"
    # Dual Block Cache
    # Fn=1, Bn=0, means FB Cache, otherwise, Dual Block Cache
    Fn_compute_blocks: int = 1
    Bn_compute_blocks: int = 0
    # We have added residual cache pattern for selected compute blocks
    Fn_compute_blocks_ids: List[int] = dataclasses.field(default_factory=list)
    Bn_compute_blocks_ids: List[int] = dataclasses.field(default_factory=list)
    # non compute blocks diff threshold, we don't skip the non
    # compute blocks if the diff >= threshold
    non_compute_blocks_diff_threshold: float = 0.08
    max_Fn_compute_blocks: int = -1
    max_Bn_compute_blocks: int = -1
    # L1 hidden states or residual diff threshold for Fn
    residual_diff_threshold: Union[torch.Tensor, float] = 0.05
    l1_hidden_states_diff_threshold: float = None
    important_condition_threshold: float = 0.0

    # Alter Cache Settings
    # Pattern: 0 F 1 T 2 F 3 T 4 F 5 T ...
    enable_alter_cache: bool = False
    is_alter_cache: bool = True
    # 1.0 means we always cache the residuals if alter_cache is enabled.
    alter_residual_diff_threshold: Optional[Union[torch.Tensor, float]] = 1.0

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

    # Support TaylorSeers in Dual Block Cache
    # Title: From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers
    # Url: https://arxiv.org/pdf/2503.06923
    enable_taylorseer: bool = False
    enable_encoder_taylorseer: bool = False
    # NOTE: use residual cache for taylorseer may incur precision loss
    taylorseer_cache_type: str = "hidden_states"  # residual or hidden_states
    taylorseer_order: int = 2  # The order for TaylorSeer
    taylorseer_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    taylorseer: Optional[TaylorSeer] = None
    encoder_tarlorseer: Optional[TaylorSeer] = None

    # Support enable_spearate_cfg, such as Wan 2.1,
    # Qwen-Image. For model that fused CFG and non-CFG into single
    # forward step, should set enable_spearate_cfg as False.
    # For example: CogVideoX, HunyuanVideo, Mochi.
    enable_spearate_cfg: bool = False
    # Compute cfg forward first or not, default False, namely,
    # 0, 2, 4, ..., -> non-CFG step; 1, 3, 5, ... -> CFG step.
    cfg_compute_first: bool = False
    # Compute spearate diff values for CFG and non-CFG step,
    # default True. If False, we will use the computed diff from
    # current non-CFG transformer step for current CFG step.
    cfg_diff_compute_separate: bool = True
    cfg_taylorseer: Optional[TaylorSeer] = None
    cfg_encoder_taylorseer: Optional[TaylorSeer] = None

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

    @torch.compiler.disable
    def __post_init__(self):
        if logger.isEnabledFor(logging.DEBUG):
            logger.info(f"Created _CacheContext: {self.name}")
        # Some checks for settings
        if self.enable_spearate_cfg:
            assert self.enable_alter_cache is False, (
                "enable_alter_cache must set as False if "
                "enable_spearate_cfg is enabled."
            )
            if self.cfg_diff_compute_separate:
                assert self.cfg_compute_first is False, (
                    "cfg_compute_first must set as False if "
                    "cfg_diff_compute_separate is enabled."
                )

        if "max_warmup_steps" not in self.taylorseer_kwargs:
            # If max_warmup_steps is not set in taylorseer_kwargs,
            # set the same as max_warmup_steps for DBCache
            self.taylorseer_kwargs["max_warmup_steps"] = (
                self.max_warmup_steps if self.max_warmup_steps > 0 else 1
            )

        # Overwrite the 'n_derivatives' by 'taylorseer_order', default: 2.
        self.taylorseer_kwargs["n_derivatives"] = self.taylorseer_order

        if self.enable_taylorseer:
            self.taylorseer = TaylorSeer(**self.taylorseer_kwargs)
            if self.enable_spearate_cfg:
                self.cfg_taylorseer = TaylorSeer(**self.taylorseer_kwargs)

        if self.enable_encoder_taylorseer:
            self.encoder_tarlorseer = TaylorSeer(**self.taylorseer_kwargs)
            if self.enable_spearate_cfg:
                self.cfg_encoder_taylorseer = TaylorSeer(
                    **self.taylorseer_kwargs
                )

    @torch.compiler.disable
    def get_residual_diff_threshold(self):
        if self.enable_alter_cache:
            residual_diff_threshold = self.alter_residual_diff_threshold
        else:
            residual_diff_threshold = self.residual_diff_threshold
            if self.l1_hidden_states_diff_threshold is not None:
                # Use the L1 hidden states diff threshold if set
                residual_diff_threshold = self.l1_hidden_states_diff_threshold
        if isinstance(residual_diff_threshold, torch.Tensor):
            residual_diff_threshold = residual_diff_threshold.item()
        return residual_diff_threshold

    @torch.compiler.disable
    def get_buffer(self, name):
        if self.enable_alter_cache and self.is_alter_cache:
            name = f"{name}_alter"
        return self.buffers.get(name)

    @torch.compiler.disable
    def set_buffer(self, name, buffer):
        if self.enable_alter_cache and self.is_alter_cache:
            name = f"{name}_alter"
        self.buffers[name] = buffer

    @torch.compiler.disable
    def remove_buffer(self, name):
        if self.enable_alter_cache and self.is_alter_cache:
            name = f"{name}_alter"
        if name in self.buffers:
            del self.buffers[name]

    @torch.compiler.disable
    def clear_buffers(self):
        self.buffers.clear()

    @torch.compiler.disable
    def mark_step_begin(self):
        # Always increase transformer executed steps
        # incr    step: prev 0 -> 1; prev 1 -> 2
        # current step: incr step - 1
        self.transformer_executed_steps += 1
        if not self.enable_spearate_cfg:
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

        if not self.enable_alter_cache:
            # 0 F 1 T 2 F 3 T 4 F 5 T ...
            self.is_alter_cache = not self.is_alter_cache

        # Reset the cached steps and residual diffs at the beginning
        # of each inference.
        if self.get_current_transformer_step() == 0:
            self.cached_steps.clear()
            self.residual_diffs.clear()
            self.cfg_cached_steps.clear()
            self.cfg_residual_diffs.clear()
            # Reset the TaylorSeers cache at the beginning of each inference.
            # reset_cache will set the current step to -1 for TaylorSeer,
            if self.enable_taylorseer or self.enable_encoder_taylorseer:
                taylorseer, encoder_taylorseer = self.get_taylorseers()
                if taylorseer is not None:
                    taylorseer.reset_cache()
                if encoder_taylorseer is not None:
                    encoder_taylorseer.reset_cache()
                cfg_taylorseer, cfg_encoder_taylorseer = (
                    self.get_cfg_taylorseers()
                )
                if cfg_taylorseer is not None:
                    cfg_taylorseer.reset_cache()
                if cfg_encoder_taylorseer is not None:
                    cfg_encoder_taylorseer.reset_cache()

        # mark_step_begin of TaylorSeer must be called after the cache is reset.
        if self.enable_taylorseer or self.enable_encoder_taylorseer:
            if self.enable_spearate_cfg:
                # Assume non-CFG steps: 0, 2, 4, 6, ...
                if not self.is_separate_cfg_step():
                    taylorseer, encoder_taylorseer = self.get_taylorseers()
                    if taylorseer is not None:
                        taylorseer.mark_step_begin()
                    if encoder_taylorseer is not None:
                        encoder_taylorseer.mark_step_begin()
                else:
                    cfg_taylorseer, cfg_encoder_taylorseer = (
                        self.get_cfg_taylorseers()
                    )
                    if cfg_taylorseer is not None:
                        cfg_taylorseer.mark_step_begin()
                    if cfg_encoder_taylorseer is not None:
                        cfg_encoder_taylorseer.mark_step_begin()
            else:
                taylorseer, encoder_taylorseer = self.get_taylorseers()
                if taylorseer is not None:
                    taylorseer.mark_step_begin()
                if encoder_taylorseer is not None:
                    encoder_taylorseer.mark_step_begin()

    def get_taylorseers(self) -> Tuple[TaylorSeer, TaylorSeer]:
        return self.taylorseer, self.encoder_tarlorseer

    def get_cfg_taylorseers(self) -> Tuple[TaylorSeer, TaylorSeer]:
        return self.cfg_taylorseer, self.cfg_encoder_taylorseer

    @torch.compiler.disable
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

    @torch.compiler.disable
    def get_residual_diffs(self):
        return self.residual_diffs.copy()

    @torch.compiler.disable
    def get_cfg_residual_diffs(self):
        return self.cfg_residual_diffs.copy()

    @torch.compiler.disable
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

    @torch.compiler.disable
    def get_cached_steps(self):
        return self.cached_steps.copy()

    @torch.compiler.disable
    def get_cfg_cached_steps(self):
        return self.cfg_cached_steps.copy()

    @torch.compiler.disable
    def get_current_step(self):
        return self.executed_steps - 1

    @torch.compiler.disable
    def get_current_transformer_step(self):
        return self.transformer_executed_steps - 1

    @torch.compiler.disable
    def is_separate_cfg_step(self):
        if not self.enable_spearate_cfg:
            return False
        if self.cfg_compute_first:
            # CFG steps: 0, 2, 4, 6, ...
            return self.get_current_transformer_step() % 2 == 0
        # CFG steps: 1, 3, 5, 7, ...
        return self.get_current_transformer_step() % 2 != 0

    @torch.compiler.disable
    def is_in_warmup(self):
        return self.get_current_step() < self.max_warmup_steps
