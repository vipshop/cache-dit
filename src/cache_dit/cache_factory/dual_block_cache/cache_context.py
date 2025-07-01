# Adapted from: https://github.com/chengzeyi/ParaAttention/tree/main/src/para_attn/first_block_cache/context.py

import logging
import contextlib
import dataclasses
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Union

import torch

import cache_dit.primitives as DP
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
    residual_diff_threshold: Union[torch.Tensor, float] = 0.0
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
    num_inference_steps: int = -1
    warmup_steps: int = 0  # DON'T Cache in warmup steps
    # DON'T Cache if the number of cached steps >= max_cached_steps
    max_cached_steps: int = -1

    # Statistics for botch alter cache and non-alter cache
    # Record the steps that have been cached, both alter cache and non-alter cache
    executed_steps: int = 0  # cache + non-cache steps
    cached_steps: List[int] = dataclasses.field(default_factory=list)
    residual_diffs: DefaultDict[str, float] = dataclasses.field(
        default_factory=lambda: defaultdict(float),
    )
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
    alter_taylorseer: Optional[TaylorSeer] = None
    alter_encoder_taylorseer: Optional[TaylorSeer] = None

    # TODO: Support SLG in Dual Block Cache
    # Skip Layer Guidance, SLG
    # https://github.com/huggingface/candle/issues/2588
    slg_layers: Optional[List[int]] = None
    slg_start: float = 0.0
    slg_end: float = 0.1

    @torch.compiler.disable
    def __post_init__(self):

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
            if self.enable_alter_cache:
                self.alter_taylorseer = TaylorSeer(**self.taylorseer_kwargs)

        if self.enable_encoder_taylorseer:
            self.encoder_tarlorseer = TaylorSeer(**self.taylorseer_kwargs)
            if self.enable_alter_cache:
                self.alter_encoder_taylorseer = TaylorSeer(
                    **self.taylorseer_kwargs
                )

    @torch.compiler.disable
    def get_incremental_name(self, name=None):
        if name is None:
            name = "default"
        idx = self.incremental_name_counters[name]
        self.incremental_name_counters[name] += 1
        return f"{name}_{idx}"

    @torch.compiler.disable
    def reset_incremental_names(self):
        self.incremental_name_counters.clear()

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
        if not self.enable_alter_cache:
            self.executed_steps += 1
        else:
            self.executed_steps += 1
            # 0 F 1 T 2 F 3 T 4 F 5 T ...
            self.is_alter_cache = not self.is_alter_cache

        # Reset the cached steps and residual diffs at the beginning
        # of each inference.
        if self.get_current_step() == 0:
            self.cached_steps.clear()
            self.residual_diffs.clear()
            self.reset_incremental_names()
            # Reset the TaylorSeers cache at the beginning of each inference.
            # reset_cache will set the current step to -1 for TaylorSeer,
            if self.enable_taylorseer or self.enable_encoder_taylorseer:
                taylorseer, encoder_taylorseer = self.get_taylorseers()
                if taylorseer is not None:
                    taylorseer.reset_cache()
                if encoder_taylorseer is not None:
                    encoder_taylorseer.reset_cache()

        # mark_step_begin of TaylorSeer must be called after the cache is reset.
        if self.enable_taylorseer or self.enable_encoder_taylorseer:
            taylorseer, encoder_taylorseer = self.get_taylorseers()
            if taylorseer is not None:
                taylorseer.mark_step_begin()
            if encoder_taylorseer is not None:
                encoder_taylorseer.mark_step_begin()

    @torch.compiler.disable
    def get_taylorseers(self):
        if self.enable_alter_cache and self.is_alter_cache:
            return self.alter_taylorseer, self.alter_encoder_taylorseer
        return self.taylorseer, self.encoder_tarlorseer

    @torch.compiler.disable
    def add_residual_diff(self, diff):
        step = str(self.get_current_step())
        if step not in self.residual_diffs:
            # Only add the diff if it is not already recorded for this step
            self.residual_diffs[step] = diff

    @torch.compiler.disable
    def get_residual_diffs(self):
        return self.residual_diffs.copy()

    @torch.compiler.disable
    def add_cached_step(self):
        self.cached_steps.append(self.get_current_step())

    @torch.compiler.disable
    def get_cached_steps(self):
        return self.cached_steps.copy()

    @torch.compiler.disable
    def get_current_step(self):
        return self.executed_steps - 1

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
def get_cached_steps():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.get_cached_steps()


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
def is_taylorseer_enabled():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.enable_taylorseer


@torch.compiler.disable
def is_encoder_taylorseer_enabled():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.enable_encoder_taylorseer


@torch.compiler.disable
def get_taylorseers():
    cache_context = get_current_cache_context()
    assert cache_context is not None, "cache_context must be set before"
    return cache_context.get_taylorseers()


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

    assert default_attrs is not None, "default_attrs must be set before"
    for attr in cache_attrs:
        if attr.name in default_attrs:
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
        mean_diff = DP.all_reduce_sync(mean_diff, "avg")
        mean_t1 = DP.all_reduce_sync(mean_t1, "avg")

    # D = (t1 - t2) / t1 = 1 - (t2 / t1), if D = 0, then t1 = t2.
    # Futher, if we assume that (H(t,  0) - H(t-1,0)) ~ 0, then,
    # H(t-1,n) ~ H(t  ,n), which means the hidden states are similar.
    diff = (mean_diff / mean_t1).item()

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"{prefix}, diff: {diff:.6f}, threshold: {threshold:.6f}")

    add_residual_diff(diff)

    return diff < threshold


# Fn buffers
@torch.compiler.disable
def set_Fn_buffer(buffer: torch.Tensor, prefix: str = "Fn"):
    # Set hidden_states or residual for Fn blocks.
    # This buffer is only use for L1 diff calculation.
    downsample_factor = get_downsample_factor()
    if downsample_factor > 1:
        buffer = buffer[..., ::downsample_factor]
        buffer = buffer.contiguous()
    set_buffer(f"{prefix}_buffer", buffer)


@torch.compiler.disable
def get_Fn_buffer(prefix: str = "Fn"):
    return get_buffer(f"{prefix}_buffer")


@torch.compiler.disable
def set_Fn_encoder_buffer(buffer: torch.Tensor, prefix: str = "Fn"):
    set_buffer(f"{prefix}_encoder_buffer", buffer)


@torch.compiler.disable
def get_Fn_encoder_buffer(prefix: str = "Fn"):
    return get_buffer(f"{prefix}_encoder_buffer")


# Bn buffers
@torch.compiler.disable
def set_Bn_buffer(buffer: torch.Tensor, prefix: str = "Bn"):
    # Set hidden_states or residual for Bn blocks.
    # This buffer is use for hidden states approximation.
    if is_taylorseer_enabled():
        # taylorseer, encoder_taylorseer
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
            set_buffer(f"{prefix}_buffer", buffer)
    else:
        set_buffer(f"{prefix}_buffer", buffer)


@torch.compiler.disable
def get_Bn_buffer(prefix: str = "Bn"):
    if is_taylorseer_enabled():
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
            return get_buffer(f"{prefix}_buffer")
    else:
        return get_buffer(f"{prefix}_buffer")


@torch.compiler.disable
def set_Bn_encoder_buffer(buffer: torch.Tensor, prefix: str = "Bn"):
    # This buffer is use for encoder hidden states approximation.
    if is_encoder_taylorseer_enabled():
        # taylorseer, encoder_taylorseer
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
            set_buffer(f"{prefix}_encoder_buffer", buffer)
    else:
        set_buffer(f"{prefix}_encoder_buffer", buffer)


@torch.compiler.disable
def get_Bn_encoder_buffer(prefix: str = "Bn"):
    if is_encoder_taylorseer_enabled():
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
            return get_buffer(f"{prefix}_encoder_buffer")
    else:
        return get_buffer(f"{prefix}_encoder_buffer")


@torch.compiler.disable
def apply_hidden_states_residual(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
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

    hidden_states = hidden_states.contiguous()
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
    cached_steps = get_cached_steps()
    max_cached_steps = get_max_cached_steps()
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


class DBCachedTransformerBlocks(torch.nn.Module):
    def __init__(
        self,
        transformer_blocks,
        single_transformer_blocks=None,
        *,
        transformer=None,
        return_hidden_states_first=True,
        return_hidden_states_only=False,
    ):
        super().__init__()

        self.transformer = transformer
        self.transformer_blocks = transformer_blocks
        self.single_transformer_blocks = single_transformer_blocks
        self.return_hidden_states_first = return_hidden_states_first
        self.return_hidden_states_only = return_hidden_states_only

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        original_hidden_states = hidden_states
        # Call first `n` blocks to process the hidden states for
        # more stable diff calculation.
        hidden_states, encoder_hidden_states = self.call_Fn_transformer_blocks(
            hidden_states,
            encoder_hidden_states,
            *args,
            **kwargs,
        )

        Fn_hidden_states_residual = hidden_states - original_hidden_states
        del original_hidden_states

        mark_step_begin()
        # Residual L1 diff or Hidden States L1 diff
        can_use_cache = get_can_use_cache(
            (
                Fn_hidden_states_residual
                if not is_l1_diff_enabled()
                else hidden_states
            ),
            parallelized=self._is_parallelized(),
            prefix=(
                "Fn_residual"
                if not is_l1_diff_enabled()
                else "Fn_hidden_states"
            ),
        )

        torch._dynamo.graph_break()
        if can_use_cache:
            torch._dynamo.graph_break()
            add_cached_step()
            del Fn_hidden_states_residual
            hidden_states, encoder_hidden_states = apply_hidden_states_residual(
                hidden_states,
                encoder_hidden_states,
                prefix=(
                    "Bn_residual" if is_cache_residual() else "Bn_hidden_states"
                ),
                encoder_prefix=(
                    "Bn_residual"
                    if is_encoder_cache_residual()
                    else "Bn_hidden_states"
                ),
            )
            torch._dynamo.graph_break()
            # Call last `n` blocks to further process the hidden states
            # for higher precision.
            hidden_states, encoder_hidden_states = (
                self.call_Bn_transformer_blocks(
                    hidden_states,
                    encoder_hidden_states,
                    *args,
                    **kwargs,
                )
            )
        else:
            torch._dynamo.graph_break()
            set_Fn_buffer(Fn_hidden_states_residual, prefix="Fn_residual")
            if is_l1_diff_enabled():
                # for hidden states L1 diff
                set_Fn_buffer(hidden_states, "Fn_hidden_states")
            del Fn_hidden_states_residual
            torch._dynamo.graph_break()
            (
                hidden_states,
                encoder_hidden_states,
                hidden_states_residual,
                encoder_hidden_states_residual,
            ) = self.call_Mn_transformer_blocks(  # middle
                hidden_states,
                encoder_hidden_states,
                *args,
                **kwargs,
            )
            torch._dynamo.graph_break()
            if is_cache_residual():
                set_Bn_buffer(
                    hidden_states_residual,
                    prefix="Bn_residual",
                )
            else:
                # TaylorSeer
                set_Bn_buffer(
                    hidden_states,
                    prefix="Bn_hidden_states",
                )
            if is_encoder_cache_residual():
                set_Bn_encoder_buffer(
                    encoder_hidden_states_residual,
                    prefix="Bn_residual",
                )
            else:
                # TaylorSeer
                set_Bn_encoder_buffer(
                    encoder_hidden_states,
                    prefix="Bn_hidden_states",
                )
            torch._dynamo.graph_break()
            # Call last `n` blocks to further process the hidden states
            # for higher precision.
            hidden_states, encoder_hidden_states = (
                self.call_Bn_transformer_blocks(
                    hidden_states,
                    encoder_hidden_states,
                    *args,
                    **kwargs,
                )
            )

        patch_cached_stats(self.transformer)
        torch._dynamo.graph_break()

        return (
            hidden_states
            if self.return_hidden_states_only
            else (
                (hidden_states, encoder_hidden_states)
                if self.return_hidden_states_first
                else (encoder_hidden_states, hidden_states)
            )
        )

    @torch.compiler.disable
    def _is_parallelized(self):
        # Compatible with distributed inference.
        return all(
            (
                self.transformer is not None,
                getattr(self.transformer, "_is_parallelized", False),
            )
        )

    @torch.compiler.disable
    def _is_in_cache_step(self):
        # Check if the current step is in cache steps.
        # If so, we can skip some Bn blocks and directly
        # use the cached values.
        return get_current_step() in get_cached_steps()

    @torch.compiler.disable
    def _Fn_transformer_blocks(self):
        # Select first `n` blocks to process the hidden states for
        # more stable diff calculation.
        # Fn: [0,...,n-1]
        selected_Fn_transformer_blocks = self.transformer_blocks[
            : Fn_compute_blocks()
        ]
        return selected_Fn_transformer_blocks

    @torch.compiler.disable
    def _Mn_single_transformer_blocks(self):  # middle blocks
        # M(N-2n): transformer_blocks [n,...] + single_transformer_blocks [0,...,N-n]
        selected_Mn_single_transformer_blocks = []
        if self.single_transformer_blocks is not None:
            if Bn_compute_blocks() == 0:  # WARN: x[:-0] = []
                selected_Mn_single_transformer_blocks = (
                    self.single_transformer_blocks
                )
            else:
                selected_Mn_single_transformer_blocks = (
                    self.single_transformer_blocks[: -Bn_compute_blocks()]
                )
        return selected_Mn_single_transformer_blocks

    @torch.compiler.disable
    def _Mn_transformer_blocks(self):  # middle blocks
        # M(N-2n): only transformer_blocks [n,...,N-n], middle
        if Bn_compute_blocks() == 0:  # WARN: x[:-0] = []
            selected_Mn_transformer_blocks = self.transformer_blocks[
                Fn_compute_blocks() :
            ]
        else:
            selected_Mn_transformer_blocks = self.transformer_blocks[
                Fn_compute_blocks() : -Bn_compute_blocks()
            ]
        return selected_Mn_transformer_blocks

    @torch.compiler.disable
    def _Bn_single_transformer_blocks(self):
        # Bn: single_transformer_blocks [N-n+1,...,N-1]
        selected_Bn_single_transformer_blocks = []
        if self.single_transformer_blocks is not None:
            selected_Bn_single_transformer_blocks = (
                self.single_transformer_blocks[-Bn_compute_blocks() :]
            )
        return selected_Bn_single_transformer_blocks

    @torch.compiler.disable
    def _Bn_transformer_blocks(self):
        # Bn: transformer_blocks [N-n+1,...,N-1]
        selected_Bn_transformer_blocks = self.transformer_blocks[
            -Bn_compute_blocks() :
        ]
        return selected_Bn_transformer_blocks

    def call_Fn_transformer_blocks(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        assert Fn_compute_blocks() <= len(self.transformer_blocks), (
            f"Fn_compute_blocks {Fn_compute_blocks()} must be less than "
            f"the number of transformer blocks {len(self.transformer_blocks)}"
        )
        for block in self._Fn_transformer_blocks():
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                *args,
                **kwargs,
            )
            if not isinstance(hidden_states, torch.Tensor):
                hidden_states, encoder_hidden_states = hidden_states
                if not self.return_hidden_states_first:
                    hidden_states, encoder_hidden_states = (
                        encoder_hidden_states,
                        hidden_states,
                    )

        return hidden_states, encoder_hidden_states

    def call_Mn_transformer_blocks(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        original_hidden_states = hidden_states
        original_encoder_hidden_states = encoder_hidden_states
        if self.single_transformer_blocks is not None:
            for block in self.transformer_blocks[Fn_compute_blocks() :]:
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    *args,
                    **kwargs,
                )
                if not isinstance(hidden_states, torch.Tensor):
                    hidden_states, encoder_hidden_states = hidden_states
                    if not self.return_hidden_states_first:
                        hidden_states, encoder_hidden_states = (
                            encoder_hidden_states,
                            hidden_states,
                        )

            hidden_states = torch.cat(
                [encoder_hidden_states, hidden_states], dim=1
            )
            for block in self._Mn_single_transformer_blocks():
                hidden_states = block(
                    hidden_states,
                    *args,
                    **kwargs,
                )
            encoder_hidden_states, hidden_states = hidden_states.split(
                [
                    encoder_hidden_states.shape[1],
                    hidden_states.shape[1] - encoder_hidden_states.shape[1],
                ],
                dim=1,
            )
        else:
            for block in self._Mn_transformer_blocks():
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    *args,
                    **kwargs,
                )
                if not isinstance(hidden_states, torch.Tensor):
                    hidden_states, encoder_hidden_states = hidden_states
                    if not self.return_hidden_states_first:
                        hidden_states, encoder_hidden_states = (
                            encoder_hidden_states,
                            hidden_states,
                        )

        # hidden_states_shape = hidden_states.shape
        # encoder_hidden_states_shape = encoder_hidden_states.shape
        hidden_states = (
            hidden_states.reshape(-1)
            .contiguous()
            .reshape(original_hidden_states.shape)
        )
        encoder_hidden_states = (
            encoder_hidden_states.reshape(-1)
            .contiguous()
            .reshape(original_encoder_hidden_states.shape)
        )

        # hidden_states = hidden_states.contiguous()
        # encoder_hidden_states = encoder_hidden_states.contiguous()

        hidden_states_residual = hidden_states - original_hidden_states
        encoder_hidden_states_residual = (
            encoder_hidden_states - original_encoder_hidden_states
        )

        hidden_states_residual = (
            hidden_states_residual.reshape(-1)
            .contiguous()
            .reshape(original_hidden_states.shape)
        )
        encoder_hidden_states_residual = (
            encoder_hidden_states_residual.reshape(-1)
            .contiguous()
            .reshape(original_encoder_hidden_states.shape)
        )

        return (
            hidden_states,
            encoder_hidden_states,
            hidden_states_residual,
            encoder_hidden_states_residual,
        )

    @torch.compiler.disable
    def _Bn_i_single_hidden_states_residual(
        self,
        Bn_i_hidden_states: torch.Tensor,
        Bn_i_original_hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor,
        original_encoder_hidden_states: torch.Tensor,
    ):
        # Split the Bn_i_hidden_states and Bn_i_original_hidden_states
        # into encoder_hidden_states and hidden_states.
        Bn_i_hidden_states, Bn_i_encoder_hidden_states = (
            self._split_Bn_i_single_hidden_states(
                Bn_i_hidden_states,
                original_hidden_states,
                original_encoder_hidden_states,
            )
        )
        # Split the Bn_i_original_hidden_states into encoder_hidden_states
        # and hidden_states.
        Bn_i_original_hidden_states, Bn_i_original_encoder_hidden_states = (
            self._split_Bn_i_single_hidden_states(
                Bn_i_original_hidden_states,
                original_hidden_states,
                original_encoder_hidden_states,
            )
        )

        # Compute the residuals for the Bn_i_hidden_states and
        # Bn_i_encoder_hidden_states.
        Bn_i_hidden_states_residual = (
            Bn_i_hidden_states - Bn_i_original_hidden_states
        )
        Bn_i_encoder_hidden_states_residual = (
            Bn_i_encoder_hidden_states - Bn_i_original_encoder_hidden_states
        )
        return (
            Bn_i_hidden_states_residual,
            Bn_i_encoder_hidden_states_residual,
        )

    @torch.compiler.disable
    def _split_Bn_i_single_hidden_states(
        self,
        Bn_i_hidden_states: torch.Tensor,
        original_hidden_states: torch.Tensor,
        original_encoder_hidden_states: torch.Tensor,
    ):
        # Split the Bn_i_hidden_states into encoder_hidden_states and hidden_states.
        Bn_i_encoder_hidden_states, Bn_i_hidden_states = (
            Bn_i_hidden_states.split(
                [
                    original_encoder_hidden_states.shape[1],
                    Bn_i_hidden_states.shape[1]
                    - original_encoder_hidden_states.shape[1],
                ],
                dim=1,
            )
        )
        # Reshape the Bn_i_hidden_states and Bn_i_encoder_hidden_states
        # to the original shape. This is necessary to ensure that the
        # residuals are computed correctly.
        Bn_i_hidden_states = (
            Bn_i_hidden_states.reshape(-1)
            .contiguous()
            .reshape(original_hidden_states.shape)
        )
        Bn_i_encoder_hidden_states = (
            Bn_i_encoder_hidden_states.reshape(-1)
            .contiguous()
            .reshape(original_encoder_hidden_states.shape)
        )
        return Bn_i_hidden_states, Bn_i_encoder_hidden_states

    def _compute_and_cache_single_transformer_block(
        self,
        # Block index in the transformer blocks
        # Bn: 8, block_id should be in [0, 8)
        block_id: int,
        # Helper inputs for hidden states split and reshape
        original_hidden_states: torch.Tensor,
        original_encoder_hidden_states: torch.Tensor,
        # Below are the inputs to the block
        block,  # The transformer block to be executed
        hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        # Helper function for `call_Bn_transformer_blocks`
        # Skip the blocks by reuse residual cache if they are not
        # in the Bn_compute_blocks_ids. NOTE: We should only skip
        # the specific Bn blocks in cache steps. Compute the block
        # and cache the residuals in non-cache steps.

        # Normal steps: Compute the block and cache the residuals.
        if not self._is_in_cache_step():
            Bn_i_original_hidden_states = hidden_states
            hidden_states = block(
                hidden_states,
                *args,
                **kwargs,
            )
            # Cache residuals for the non-compute Bn blocks for
            # subsequent cache steps.
            if block_id not in Bn_compute_blocks_ids():
                Bn_i_hidden_states = hidden_states
                (
                    Bn_i_hidden_states_residual,
                    Bn_i_encoder_hidden_states_residual,
                ) = self._Bn_i_single_hidden_states_residual(
                    Bn_i_hidden_states,
                    Bn_i_original_hidden_states,
                    original_hidden_states,
                    original_encoder_hidden_states,
                )

                # Save original_hidden_states for diff calculation.
                set_Bn_buffer(
                    Bn_i_original_hidden_states,
                    prefix=f"Bn_{block_id}_single_original",
                )
                set_Bn_encoder_buffer(
                    Bn_i_original_hidden_states,
                    prefix=f"Bn_{block_id}_single_original",
                )

                set_Bn_buffer(
                    Bn_i_hidden_states_residual,
                    prefix=f"Bn_{block_id}_single_residual",
                )
                set_Bn_encoder_buffer(
                    Bn_i_encoder_hidden_states_residual,
                    prefix=f"Bn_{block_id}_single_residual",
                )
                del Bn_i_hidden_states
                del Bn_i_hidden_states_residual
                del Bn_i_encoder_hidden_states_residual

            del Bn_i_original_hidden_states

        else:
            # Cache steps: Reuse the cached residuals.
            # Check if the block is in the Bn_compute_blocks_ids.
            if block_id in Bn_compute_blocks_ids():
                hidden_states = block(
                    hidden_states,
                    *args,
                    **kwargs,
                )
            else:
                # Skip the block if it is not in the Bn_compute_blocks_ids.
                # Use the cached residuals instead.
                # Check if can use the cached residuals.
                if get_can_use_cache(
                    hidden_states,  # curr step
                    parallelized=self._is_parallelized(),
                    threshold=non_compute_blocks_diff_threshold(),
                    prefix=f"Bn_{block_id}_single_original",  # prev step
                ):
                    Bn_i_original_hidden_states = hidden_states
                    (
                        Bn_i_original_hidden_states,
                        Bn_i_original_encoder_hidden_states,
                    ) = self._split_Bn_i_single_hidden_states(
                        Bn_i_original_hidden_states,
                        original_hidden_states,
                        original_encoder_hidden_states,
                    )
                    hidden_states, encoder_hidden_states = (
                        apply_hidden_states_residual(
                            Bn_i_original_hidden_states,
                            Bn_i_original_encoder_hidden_states,
                            prefix=(
                                f"Bn_{block_id}_single_residual"
                                if is_cache_residual()
                                else f"Bn_{block_id}_single_original"
                            ),
                            encoder_prefix=(
                                f"Bn_{block_id}_single_residual"
                                if is_encoder_cache_residual()
                                else f"Bn_{block_id}_single_original"
                            ),
                        )
                    )
                    hidden_states = torch.cat(
                        [encoder_hidden_states, hidden_states],
                        dim=1,
                    )
                    del Bn_i_original_hidden_states
                    del Bn_i_original_encoder_hidden_states
                else:
                    hidden_states = block(
                        hidden_states,
                        *args,
                        **kwargs,
                    )
        return hidden_states

    def _compute_and_cache_transformer_block(
        self,
        # Block index in the transformer blocks
        # Bn: 8, block_id should be in [0, 8)
        block_id: int,
        # Below are the inputs to the block
        block,  # The transformer block to be executed
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        # Helper function for `call_Bn_transformer_blocks`
        # Skip the blocks by reuse residual cache if they are not
        # in the Bn_compute_blocks_ids. NOTE: We should only skip
        # the specific Bn blocks in cache steps. Compute the block
        # and cache the residuals in non-cache steps.

        # Normal steps: Compute the block and cache the residuals.
        if not self._is_in_cache_step():
            Bn_i_original_hidden_states = hidden_states
            Bn_i_original_encoder_hidden_states = encoder_hidden_states
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                *args,
                **kwargs,
            )
            if not isinstance(hidden_states, torch.Tensor):
                hidden_states, encoder_hidden_states = hidden_states
                if not self.return_hidden_states_first:
                    hidden_states, encoder_hidden_states = (
                        encoder_hidden_states,
                        hidden_states,
                    )
            # Cache residuals for the non-compute Bn blocks for
            # subsequent cache steps.
            if block_id not in Bn_compute_blocks_ids():
                Bn_i_hidden_states_residual = (
                    hidden_states - Bn_i_original_hidden_states
                )
                Bn_i_encoder_hidden_states_residual = (
                    encoder_hidden_states - Bn_i_original_encoder_hidden_states
                )

                # Save original_hidden_states for diff calculation.
                set_Bn_buffer(
                    Bn_i_original_hidden_states,
                    prefix=f"Bn_{block_id}_original",
                )
                set_Bn_encoder_buffer(
                    Bn_i_original_encoder_hidden_states,
                    prefix=f"Bn_{block_id}_original",
                )

                set_Bn_buffer(
                    Bn_i_hidden_states_residual,
                    prefix=f"Bn_{block_id}_residual",
                )
                set_Bn_encoder_buffer(
                    Bn_i_encoder_hidden_states_residual,
                    prefix=f"Bn_{block_id}_residual",
                )
                del Bn_i_hidden_states_residual
                del Bn_i_encoder_hidden_states_residual

            del Bn_i_original_hidden_states
            del Bn_i_original_encoder_hidden_states

        else:
            # Cache steps: Reuse the cached residuals.
            # Check if the block is in the Bn_compute_blocks_ids.
            if block_id in Bn_compute_blocks_ids():
                hidden_states = block(
                    hidden_states,
                    encoder_hidden_states,
                    *args,
                    **kwargs,
                )
                if not isinstance(hidden_states, torch.Tensor):
                    hidden_states, encoder_hidden_states = hidden_states
                    if not self.return_hidden_states_first:
                        hidden_states, encoder_hidden_states = (
                            encoder_hidden_states,
                            hidden_states,
                        )
            else:
                # Skip the block if it is not in the Bn_compute_blocks_ids.
                # Use the cached residuals instead.
                # Check if can use the cached residuals.
                if get_can_use_cache(
                    hidden_states,  # curr step
                    parallelized=self._is_parallelized(),
                    threshold=non_compute_blocks_diff_threshold(),
                    prefix=f"Bn_{block_id}_original",  # prev step
                ):
                    hidden_states, encoder_hidden_states = (
                        apply_hidden_states_residual(
                            hidden_states,
                            encoder_hidden_states,
                            prefix=(
                                f"Bn_{block_id}_residual"
                                if is_cache_residual()
                                else f"Bn_{block_id}_original"
                            ),
                            encoder_prefix=(
                                f"Bn_{block_id}_residual"
                                if is_encoder_cache_residual()
                                else f"Bn_{block_id}_original"
                            ),
                        )
                    )
                else:
                    hidden_states = block(
                        hidden_states,
                        encoder_hidden_states,
                        *args,
                        **kwargs,
                    )
                    if not isinstance(hidden_states, torch.Tensor):
                        hidden_states, encoder_hidden_states = hidden_states
                        if not self.return_hidden_states_first:
                            hidden_states, encoder_hidden_states = (
                                encoder_hidden_states,
                                hidden_states,
                            )
        return hidden_states, encoder_hidden_states

    def call_Bn_transformer_blocks(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        if Bn_compute_blocks() == 0:
            return hidden_states, encoder_hidden_states

        original_hidden_states = hidden_states
        original_encoder_hidden_states = encoder_hidden_states
        if self.single_transformer_blocks is not None:
            assert Bn_compute_blocks() <= len(self.single_transformer_blocks), (
                f"Bn_compute_blocks {Bn_compute_blocks()} must be less than "
                f"the number of single transformer blocks {len(self.single_transformer_blocks)}"
            )

            torch._dynamo.graph_break()
            hidden_states = torch.cat(
                [encoder_hidden_states, hidden_states], dim=1
            )
            if len(Bn_compute_blocks_ids()) > 0:
                for i, block in enumerate(self._Bn_single_transformer_blocks()):
                    hidden_states = (
                        self._compute_and_cache_single_transformer_block(
                            i,
                            original_hidden_states,
                            original_encoder_hidden_states,
                            block,
                            hidden_states,
                            *args,
                            **kwargs,
                        )
                    )
            else:
                # Compute all Bn blocks if no specific Bn compute blocks ids are set.
                for block in self._Bn_single_transformer_blocks():
                    hidden_states = block(
                        hidden_states,
                        *args,
                        **kwargs,
                    )
            encoder_hidden_states, hidden_states = hidden_states.split(
                [
                    encoder_hidden_states.shape[1],
                    hidden_states.shape[1] - encoder_hidden_states.shape[1],
                ],
                dim=1,
            )
            torch._dynamo.graph_break()
        else:
            assert Bn_compute_blocks() <= len(self.transformer_blocks), (
                f"Bn_compute_blocks {Bn_compute_blocks()} must be less than "
                f"the number of transformer blocks {len(self.transformer_blocks)}"
            )
            torch._dynamo.graph_break()
            if len(Bn_compute_blocks_ids()) > 0:
                for i, block in enumerate(self._Bn_transformer_blocks()):
                    hidden_states, encoder_hidden_states = (
                        self._compute_and_cache_transformer_block(
                            i,
                            block,
                            hidden_states,
                            encoder_hidden_states,
                            *args,
                            **kwargs,
                        )
                    )
            else:
                # Compute all Bn blocks if no specific Bn compute blocks ids are set.
                for block in self._Bn_transformer_blocks():
                    hidden_states = block(
                        hidden_states,
                        encoder_hidden_states,
                        *args,
                        **kwargs,
                    )
                    if not isinstance(hidden_states, torch.Tensor):
                        hidden_states, encoder_hidden_states = hidden_states
                        if not self.return_hidden_states_first:
                            hidden_states, encoder_hidden_states = (
                                encoder_hidden_states,
                                hidden_states,
                            )
            torch._dynamo.graph_break()

        hidden_states = (
            hidden_states.reshape(-1)
            .contiguous()
            .reshape(original_hidden_states.shape)
        )
        encoder_hidden_states = (
            encoder_hidden_states.reshape(-1)
            .contiguous()
            .reshape(original_encoder_hidden_states.shape)
        )
        return hidden_states, encoder_hidden_states


@torch.compiler.disable
def patch_cached_stats(
    transformer,
):
    # Patch the cached stats to the transformer, the cached stats
    # will be reset for each calling of pipe.__call__(**kwargs).
    if transformer is None:
        return

    # TODO: Patch more cached stats to the transformer
    transformer._cached_steps = get_cached_steps()
    transformer._residual_diffs = get_residual_diffs()
