import logging
import contextlib
import dataclasses
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Union, Tuple

import torch
import torch.distributed as dist

from cache_dit.cache_factory.taylorseer import TaylorSeer
from cache_dit.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class DBCacheContext:
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
    num_inference_steps: int = -1  # un-used now
    warmup_steps: int = 0  # DON'T Cache in warmup steps
    # DON'T Cache if the number of cached steps >= max_cached_steps
    max_cached_steps: int = -1  # for both CFG and non-CFG

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
    taylorseer_kwargs: Dict[str, Any] = dataclasses.field(default_factory=dict)
    taylorseer: Optional[TaylorSeer] = None
    encoder_tarlorseer: Optional[TaylorSeer] = None

    # Support do_separate_cfg, such as Wan 2.1,
    # Qwen-Image. For model that fused CFG and non-CFG into single
    # forward step, should set do_separate_cfg as False.
    # For example: CogVideoX, HunyuanVideo, Mochi.
    do_separate_cfg: bool = False
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
    cfg_cached_steps: List[int] = dataclasses.field(default_factory=list)
    cfg_residual_diffs: DefaultDict[str, float] = dataclasses.field(
        default_factory=lambda: defaultdict(float),
    )

    @torch.compiler.disable
    def __post_init__(self):
        # Some checks for settings
        if self.do_separate_cfg:
            assert self.enable_alter_cache is False, (
                "enable_alter_cache must set as False if "
                "do_separate_cfg is enabled."
            )
            if self.cfg_diff_compute_separate:
                assert self.cfg_compute_first is False, (
                    "cfg_compute_first must set as False if "
                    "cfg_diff_compute_separate is enabled."
                )

        if "warmup_steps" not in self.taylorseer_kwargs:
            # If warmup_steps is not set in taylorseer_kwargs,
            # set the same as warmup_steps for DBCache
            self.taylorseer_kwargs["warmup_steps"] = (
                self.warmup_steps if self.warmup_steps > 0 else 1
            )

        # Only set n_derivatives as 2 or 3, which is enough for most cases.
        if "n_derivatives" not in self.taylorseer_kwargs:
            self.taylorseer_kwargs["n_derivatives"] = max(
                2, min(3, self.taylorseer_kwargs["warmup_steps"])
            )

        if self.enable_taylorseer:
            self.taylorseer = TaylorSeer(**self.taylorseer_kwargs)
            if self.do_separate_cfg:
                self.cfg_taylorseer = TaylorSeer(**self.taylorseer_kwargs)

        if self.enable_encoder_taylorseer:
            self.encoder_tarlorseer = TaylorSeer(**self.taylorseer_kwargs)
            if self.do_separate_cfg:
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
        if not self.do_separate_cfg:
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
            if self.do_separate_cfg:
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
        if not self.is_separate_cfg_step():
            self.cached_steps.append(self.get_current_step())
        else:
            self.cfg_cached_steps.append(self.get_current_step())

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
        if not self.do_separate_cfg:
            return False
        if self.cfg_compute_first:
            # CFG steps: 0, 2, 4, 6, ...
            return self.get_current_transformer_step() % 2 == 0
        # CFG steps: 1, 3, 5, 7, ...
        return self.get_current_transformer_step() % 2 != 0

    @torch.compiler.disable
    def is_in_warmup(self):
        return self.get_current_step() < self.warmup_steps


@torch.compiler.disable
def get_residual_diff_threshold():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.get_residual_diff_threshold()


@torch.compiler.disable
def get_buffer(name):
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.get_buffer(name)


@torch.compiler.disable
def set_buffer(name, buffer):
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    cache_context.set_buffer(name, buffer)


@torch.compiler.disable
def remove_buffer(name):
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    cache_context.remove_buffer(name)


@torch.compiler.disable
def mark_step_begin():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    cache_context.mark_step_begin()


@torch.compiler.disable
def get_current_step():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.get_current_step()


@torch.compiler.disable
def get_current_step_residual_diff():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    step = str(get_current_step())
    residual_diffs = get_residual_diffs()
    if step in residual_diffs:
        return residual_diffs[step]
    return None


@torch.compiler.disable
def get_current_step_cfg_residual_diff():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    step = str(get_current_step())
    cfg_residual_diffs = get_cfg_residual_diffs()
    if step in cfg_residual_diffs:
        return cfg_residual_diffs[step]
    return None


@torch.compiler.disable
def get_current_transformer_step():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.get_current_transformer_step()


@torch.compiler.disable
def get_cached_steps():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.get_cached_steps()


@torch.compiler.disable
def get_cfg_cached_steps():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.get_cfg_cached_steps()


@torch.compiler.disable
def get_max_cached_steps():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.max_cached_steps


@torch.compiler.disable
def add_cached_step():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    cache_context.add_cached_step()


@torch.compiler.disable
def add_residual_diff(diff):
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    cache_context.add_residual_diff(diff)


@torch.compiler.disable
def get_residual_diffs():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.get_residual_diffs()


@torch.compiler.disable
def get_cfg_residual_diffs():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.get_cfg_residual_diffs()


@torch.compiler.disable
def is_taylorseer_enabled():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.enable_taylorseer


@torch.compiler.disable
def is_encoder_taylorseer_enabled():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.enable_encoder_taylorseer


def get_taylorseers() -> Tuple[TaylorSeer, TaylorSeer]:
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.get_taylorseers()


def get_cfg_taylorseers() -> Tuple[TaylorSeer, TaylorSeer]:
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.get_cfg_taylorseers()


@torch.compiler.disable
def is_taylorseer_cache_residual():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.taylorseer_cache_type == "residual"


@torch.compiler.disable
def is_cache_residual():
    if is_taylorseer_enabled():
        # residual or hidden_states
        return is_taylorseer_cache_residual()
    return True


@torch.compiler.disable
def is_encoder_cache_residual():
    if is_encoder_taylorseer_enabled():
        # residual or hidden_states
        return is_taylorseer_cache_residual()
    return True


@torch.compiler.disable
def is_alter_cache_enabled():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.enable_alter_cache


@torch.compiler.disable
def is_alter_cache():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.is_alter_cache


@torch.compiler.disable
def is_in_warmup():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.is_in_warmup()


@torch.compiler.disable
def is_l1_diff_enabled():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return (
        cache_context.l1_hidden_states_diff_threshold is not None
        and cache_context.l1_hidden_states_diff_threshold > 0.0
    )


@torch.compiler.disable
def get_important_condition_threshold():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.important_condition_threshold


@torch.compiler.disable
def non_compute_blocks_diff_threshold():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.non_compute_blocks_diff_threshold


@torch.compiler.disable
def Fn_compute_blocks():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    assert (
        cache_context.Fn_compute_blocks >= 1
    ), "Fn_compute_blocks must be >= 1"
    if cache_context.max_Fn_compute_blocks > 0:
        # NOTE: Fn_compute_blocks can be 1, which means FB Cache
        # but it must be less than or equal to max_Fn_compute_blocks
        assert (
            cache_context.Fn_compute_blocks
            <= cache_context.max_Fn_compute_blocks
        ), (
            f"Fn_compute_blocks must be <= {cache_context.max_Fn_compute_blocks}, "
            f"but got {cache_context.Fn_compute_blocks}"
        )
    return cache_context.Fn_compute_blocks


@torch.compiler.disable
def Fn_compute_blocks_ids():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    assert (
        len(cache_context.Fn_compute_blocks_ids)
        <= cache_context.Fn_compute_blocks
    ), (
        "The num of Fn_compute_blocks_ids must be <= Fn_compute_blocks "
        f"{cache_context.Fn_compute_blocks}, but got "
        f"{len(cache_context.Fn_compute_blocks_ids)}"
    )
    return cache_context.Fn_compute_blocks_ids


@torch.compiler.disable
def Bn_compute_blocks():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    assert (
        cache_context.Bn_compute_blocks >= 0
    ), "Bn_compute_blocks must be >= 0"
    if cache_context.max_Bn_compute_blocks > 0:
        # NOTE: Bn_compute_blocks can be 0, which means FB Cache
        # but it must be less than or equal to max_Bn_compute_blocks
        assert (
            cache_context.Bn_compute_blocks
            <= cache_context.max_Bn_compute_blocks
        ), (
            f"Bn_compute_blocks must be <= {cache_context.max_Bn_compute_blocks}, "
            f"but got {cache_context.Bn_compute_blocks}"
        )
    return cache_context.Bn_compute_blocks


@torch.compiler.disable
def Bn_compute_blocks_ids():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    assert (
        len(cache_context.Bn_compute_blocks_ids)
        <= cache_context.Bn_compute_blocks
    ), (
        "The num of Bn_compute_blocks_ids must be <= Bn_compute_blocks "
        f"{cache_context.Bn_compute_blocks}, but got "
        f"{len(cache_context.Bn_compute_blocks_ids)}"
    )
    return cache_context.Bn_compute_blocks_ids


@torch.compiler.disable
def do_separate_cfg():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.do_separate_cfg


@torch.compiler.disable
def is_separate_cfg_step():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.is_separate_cfg_step()


@torch.compiler.disable
def cfg_diff_compute_separate():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.cfg_diff_compute_separate


_current_cache_context: DBCacheContext = None


def create_cache_context(*args, **kwargs):
    return DBCacheContext(*args, **kwargs)


def get_current_cache_context():
    return _current_cache_context


def set_current_cache_context(cache_context=None):
    global _current_cache_context
    _current_cache_context = cache_context


def collect_cache_kwargs(default_attrs: dict, **kwargs):
    # NOTE: This API will split kwargs into cache_kwargs and other_kwargs
    # default_attrs: specific settings for different pipelines
    cache_attrs = dataclasses.fields(DBCacheContext)
    cache_attrs = [
        attr
        for attr in cache_attrs
        if hasattr(
            DBCacheContext,
            attr.name,
        )
    ]
    cache_kwargs = {
        attr.name: kwargs.pop(
            attr.name,
            getattr(DBCacheContext, attr.name),
        )
        for attr in cache_attrs
    }

    def _safe_set_sequence_field(
        field_name: str,
        default_value: Any = None,
    ):
        if field_name not in cache_kwargs:
            cache_kwargs[field_name] = kwargs.pop(
                field_name,
                default_value,
            )

    # Manually set sequence fields, namely, Fn_compute_blocks_ids
    # and Bn_compute_blocks_ids, which are lists or sets.
    _safe_set_sequence_field("Fn_compute_blocks_ids", [])
    _safe_set_sequence_field("Bn_compute_blocks_ids", [])
    _safe_set_sequence_field("taylorseer_kwargs", {})

    for attr in cache_attrs:
        if attr.name in default_attrs:  # can be empty {}
            cache_kwargs[attr.name] = default_attrs[attr.name]

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Collected DBCache kwargs: {cache_kwargs}")

    return cache_kwargs, kwargs


@contextlib.contextmanager
def cache_context(cache_context):
    global _current_cache_context
    old_cache_context = _current_cache_context
    _current_cache_context = cache_context
    try:
        yield
    finally:
        _current_cache_context = old_cache_context


@torch.compiler.disable
def are_two_tensors_similar(
    t1: torch.Tensor,  # prev residual R(t-1,n) = H(t-1,n) - H(t-1,0)
    t2: torch.Tensor,  # curr residual R(t  ,n) = H(t  ,n) - H(t  ,0)
    *,
    threshold: float,
    parallelized: bool = False,
    prefix: str = "Fn",  # for debugging
):
    # Special case for threshold, 0.0 means the threshold is disabled, -1.0 means
    # the threshold is always enabled, -2.0 means the shape is not matched.
    if threshold <= 0.0:
        add_residual_diff(-0.0)
        return False

    if threshold >= 1.0:
        # If threshold is 1.0 or more, we consider them always similar.
        add_residual_diff(-1.0)
        return True

    if t1.shape != t2.shape:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{prefix}, shape error: {t1.shape} != {t2.shape}")
        add_residual_diff(-2.0)
        return False

    if all(
        (
            do_separate_cfg(),
            is_separate_cfg_step(),
            not cfg_diff_compute_separate(),
            get_current_step_residual_diff() is not None,
        )
    ):
        # Reuse computed diff value from non-CFG step
        diff = get_current_step_residual_diff()
    else:
        # Find the most significant token through t1 and t2, and
        # consider the diff of the significant token. The more significant,
        # the more important.
        condition_thresh = get_important_condition_threshold()
        if condition_thresh > 0.0:
            raw_diff = (t1 - t2).abs()  # [B, seq_len, d]
            token_m_df = raw_diff.mean(dim=-1)  # [B, seq_len]
            token_m_t1 = t1.abs().mean(dim=-1)  # [B, seq_len]
            # D = (t1 - t2) / t1 = 1 - (t2 / t1), if D = 0, then t1 = t2.
            token_diff = token_m_df / token_m_t1  # [B, seq_len]
            condition = token_diff > condition_thresh  # [B, seq_len]
            if condition.sum() > 0:
                condition = condition.unsqueeze(-1)  # [B, seq_len, 1]
                condition = condition.expand_as(raw_diff)  # [B, seq_len, d]
                mean_diff = raw_diff[condition].mean()
                mean_t1 = t1[condition].abs().mean()
            else:
                mean_diff = (t1 - t2).abs().mean()
                mean_t1 = t1.abs().mean()
        else:
            # Use the mean of the absolute difference of the tensors
            mean_diff = (t1 - t2).abs().mean()
            mean_t1 = t1.abs().mean()

        if parallelized:
            # TODO: May use async op
            dist.all_reduce(mean_diff, op=dist.ReduceOp.AVG)
            dist.all_reduce(mean_t1, op=dist.ReduceOp.AVG)

        # D = (t1 - t2) / t1 = 1 - (t2 / t1), if D = 0, then t1 = t2.
        # Futher, if we assume that (H(t,  0) - H(t-1,0)) ~ 0, then,
        # H(t-1,n) ~ H(t  ,n), which means the hidden states are similar.
        diff = (mean_diff / mean_t1).item()

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"{prefix}, diff: {diff:.6f}, threshold: {threshold:.6f}")

    add_residual_diff(diff)

    return diff < threshold


@torch.compiler.disable
def _debugging_set_buffer(prefix):
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"set {prefix}, "
            f"transformer step: {get_current_transformer_step()}, "
            f"executed step: {get_current_step()}"
        )


@torch.compiler.disable
def _debugging_get_buffer(prefix):
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"get {prefix}, "
            f"transformer step: {get_current_transformer_step()}, "
            f"executed step: {get_current_step()}"
        )


# Fn buffers
@torch.compiler.disable
def set_Fn_buffer(buffer: torch.Tensor, prefix: str = "Fn"):
    # Set hidden_states or residual for Fn blocks.
    # This buffer is only use for L1 diff calculation.
    downsample_factor = get_downsample_factor()
    if downsample_factor > 1:
        buffer = buffer[..., ::downsample_factor]
        buffer = buffer.contiguous()
    if is_separate_cfg_step():
        _debugging_set_buffer(f"{prefix}_buffer_cfg")
        set_buffer(f"{prefix}_buffer_cfg", buffer)
    else:
        _debugging_set_buffer(f"{prefix}_buffer")
        set_buffer(f"{prefix}_buffer", buffer)


@torch.compiler.disable
def get_Fn_buffer(prefix: str = "Fn"):
    if is_separate_cfg_step():
        _debugging_get_buffer(f"{prefix}_buffer_cfg")
        return get_buffer(f"{prefix}_buffer_cfg")
    _debugging_get_buffer(f"{prefix}_buffer")
    return get_buffer(f"{prefix}_buffer")


@torch.compiler.disable
def set_Fn_encoder_buffer(buffer: torch.Tensor, prefix: str = "Fn"):
    if is_separate_cfg_step():
        _debugging_set_buffer(f"{prefix}_encoder_buffer_cfg")
        set_buffer(f"{prefix}_encoder_buffer_cfg", buffer)
    else:
        _debugging_set_buffer(f"{prefix}_encoder_buffer")
        set_buffer(f"{prefix}_encoder_buffer", buffer)


@torch.compiler.disable
def get_Fn_encoder_buffer(prefix: str = "Fn"):
    if is_separate_cfg_step():
        _debugging_get_buffer(f"{prefix}_encoder_buffer_cfg")
        return get_buffer(f"{prefix}_encoder_buffer_cfg")
    _debugging_get_buffer(f"{prefix}_encoder_buffer")
    return get_buffer(f"{prefix}_encoder_buffer")


# Bn buffers
@torch.compiler.disable
def set_Bn_buffer(buffer: torch.Tensor, prefix: str = "Bn"):
    # Set hidden_states or residual for Bn blocks.
    # This buffer is use for hidden states approximation.
    if is_taylorseer_enabled():
        # taylorseer, encoder_taylorseer
        if is_separate_cfg_step():
            taylorseer, _ = get_cfg_taylorseers()
        else:
            taylorseer, _ = get_taylorseers()

        if taylorseer is not None:
            # Use TaylorSeer to update the buffer
            taylorseer.update(buffer)
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "TaylorSeer is enabled but not set in the cache context. "
                    "Falling back to default buffer retrieval."
                )
            if is_separate_cfg_step():
                _debugging_set_buffer(f"{prefix}_buffer_cfg")
                set_buffer(f"{prefix}_buffer_cfg", buffer)
            else:
                _debugging_set_buffer(f"{prefix}_buffer")
                set_buffer(f"{prefix}_buffer", buffer)
    else:
        if is_separate_cfg_step():
            _debugging_set_buffer(f"{prefix}_buffer_cfg")
            set_buffer(f"{prefix}_buffer_cfg", buffer)
        else:
            _debugging_set_buffer(f"{prefix}_buffer")
            set_buffer(f"{prefix}_buffer", buffer)


@torch.compiler.disable
def get_Bn_buffer(prefix: str = "Bn"):
    if is_taylorseer_enabled():
        # taylorseer, encoder_taylorseer
        if is_separate_cfg_step():
            taylorseer, _ = get_cfg_taylorseers()
        else:
            taylorseer, _ = get_taylorseers()

        if taylorseer is not None:
            return taylorseer.approximate_value()
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "TaylorSeer is enabled but not set in the cache context. "
                    "Falling back to default buffer retrieval."
                )
            # Fallback to default buffer retrieval
            if is_separate_cfg_step():
                _debugging_get_buffer(f"{prefix}_buffer_cfg")
                return get_buffer(f"{prefix}_buffer_cfg")
            _debugging_get_buffer(f"{prefix}_buffer")
            return get_buffer(f"{prefix}_buffer")
    else:
        if is_separate_cfg_step():
            _debugging_get_buffer(f"{prefix}_buffer_cfg")
            return get_buffer(f"{prefix}_buffer_cfg")
        _debugging_get_buffer(f"{prefix}_buffer")
        return get_buffer(f"{prefix}_buffer")


@torch.compiler.disable
def set_Bn_encoder_buffer(buffer: torch.Tensor, prefix: str = "Bn"):
    # This buffer is use for encoder hidden states approximation.
    if is_encoder_taylorseer_enabled():
        # taylorseer, encoder_taylorseer
        if is_separate_cfg_step():
            _, encoder_taylorseer = get_cfg_taylorseers()
        else:
            _, encoder_taylorseer = get_taylorseers()

        if encoder_taylorseer is not None:
            # Use TaylorSeer to update the buffer
            encoder_taylorseer.update(buffer)
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "TaylorSeer is enabled but not set in the cache context. "
                    "Falling back to default buffer retrieval."
                )
            if is_separate_cfg_step():
                _debugging_set_buffer(f"{prefix}_encoder_buffer_cfg")
                set_buffer(f"{prefix}_encoder_buffer_cfg", buffer)
            else:
                _debugging_set_buffer(f"{prefix}_encoder_buffer")
                set_buffer(f"{prefix}_encoder_buffer", buffer)
    else:
        if is_separate_cfg_step():
            _debugging_set_buffer(f"{prefix}_encoder_buffer_cfg")
            set_buffer(f"{prefix}_encoder_buffer_cfg", buffer)
        else:
            _debugging_set_buffer(f"{prefix}_encoder_buffer")
            set_buffer(f"{prefix}_encoder_buffer", buffer)


@torch.compiler.disable
def get_Bn_encoder_buffer(prefix: str = "Bn"):
    if is_encoder_taylorseer_enabled():
        if is_separate_cfg_step():
            _, encoder_taylorseer = get_cfg_taylorseers()
        else:
            _, encoder_taylorseer = get_taylorseers()

        if encoder_taylorseer is not None:
            # Use TaylorSeer to approximate the value
            return encoder_taylorseer.approximate_value()
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "TaylorSeer is enabled but not set in the cache context. "
                    "Falling back to default buffer retrieval."
                )
            # Fallback to default buffer retrieval
            if is_separate_cfg_step():
                _debugging_get_buffer(f"{prefix}_encoder_buffer_cfg")
                return get_buffer(f"{prefix}_encoder_buffer_cfg")
            _debugging_get_buffer(f"{prefix}_encoder_buffer")
            return get_buffer(f"{prefix}_encoder_buffer")
    else:
        if is_separate_cfg_step():
            _debugging_get_buffer(f"{prefix}_encoder_buffer_cfg")
            return get_buffer(f"{prefix}_encoder_buffer_cfg")
        _debugging_get_buffer(f"{prefix}_encoder_buffer")
        return get_buffer(f"{prefix}_encoder_buffer")


@torch.compiler.disable
def apply_hidden_states_residual(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    prefix: str = "Bn",
    encoder_prefix: str = "Bn_encoder",
):
    # Allow Bn and Fn prefix to be used for residual cache.
    if "Bn" in prefix:
        hidden_states_prev = get_Bn_buffer(prefix)
    else:
        hidden_states_prev = get_Fn_buffer(prefix)

    assert hidden_states_prev is not None, f"{prefix}_buffer must be set before"

    if is_cache_residual():
        hidden_states = hidden_states_prev + hidden_states
    else:
        # If cache is not residual, we use the hidden states directly
        hidden_states = hidden_states_prev

    hidden_states = hidden_states.contiguous()

    if encoder_hidden_states is not None:
        if "Bn" in encoder_prefix:
            encoder_hidden_states_prev = get_Bn_encoder_buffer(encoder_prefix)
        else:
            encoder_hidden_states_prev = get_Fn_encoder_buffer(encoder_prefix)

        assert (
            encoder_hidden_states_prev is not None
        ), f"{prefix}_encoder_buffer must be set before"

        if is_encoder_cache_residual():
            encoder_hidden_states = (
                encoder_hidden_states_prev + encoder_hidden_states
            )
        else:
            # If encoder cache is not residual, we use the encoder hidden states directly
            encoder_hidden_states = encoder_hidden_states_prev

        encoder_hidden_states = encoder_hidden_states.contiguous()

    return hidden_states, encoder_hidden_states


@torch.compiler.disable
def get_downsample_factor():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.downsample_factor


@torch.compiler.disable
def get_can_use_cache(
    states_tensor: torch.Tensor,  # hidden_states or residual
    parallelized: bool = False,
    threshold: Optional[float] = None,  # can manually set threshold
    prefix: str = "Fn",
):
    if is_in_warmup():
        return False

    max_cached_steps = get_max_cached_steps()
    if not is_separate_cfg_step():
        cached_steps = get_cached_steps()
    else:
        cached_steps = get_cfg_cached_steps()

    if max_cached_steps >= 0 and (len(cached_steps) >= max_cached_steps):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"{prefix}, max_cached_steps reached: {max_cached_steps}, "
                "cannot use cache."
            )
        return False

    if threshold is None or threshold <= 0.0:
        threshold = get_residual_diff_threshold()
    if threshold <= 0.0:
        return False

    downsample_factor = get_downsample_factor()
    if downsample_factor > 1 and "Bn" not in prefix:
        states_tensor = states_tensor[..., ::downsample_factor]
        states_tensor = states_tensor.contiguous()

    # Allow Bn and Fn prefix to be used for diff calculation.
    if "Bn" in prefix:
        prev_states_tensor = get_Bn_buffer(prefix)
    else:
        prev_states_tensor = get_Fn_buffer(prefix)

    if not is_alter_cache_enabled():
        # Dynamic cache according to the residual diff
        can_use_cache = (
            prev_states_tensor is not None
            and are_two_tensors_similar(
                prev_states_tensor,
                states_tensor,
                threshold=threshold,
                parallelized=parallelized,
                prefix=prefix,
            )
        )
    else:
        # Only cache in the alter cache steps
        can_use_cache = (
            prev_states_tensor is not None
            and are_two_tensors_similar(
                prev_states_tensor,
                states_tensor,
                threshold=threshold,
                parallelized=parallelized,
                prefix=prefix,
            )
            and is_alter_cache()
        )
    return can_use_cache
