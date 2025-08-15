import logging
import contextlib
import dataclasses
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import torch

import cache_dit.primitives as primitives
from cache_dit.logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class DBPPruneContext:
    # Dyanmic Block Prune
    # Aleast compute first `Fn` and  last `Bn` blocks
    # FnBn designs are inspired by the Dual Block Cache
    Fn_compute_blocks: int = 8
    Bn_compute_blocks: int = 8
    # Non prune blocks IDs, e.g., [0, 1, 2, 3, 4, 5, 6, 7]
    non_prune_blocks_ids: List[int] = dataclasses.field(default_factory=list)
    # L1 hidden states or residual diff threshold for Fn
    residual_diff_threshold: Union[torch.Tensor, float] = 0.0
    l1_hidden_states_diff_threshold: float = None
    important_condition_threshold: float = 0.0
    # Compute the dynamic prune threshold based on the mean of the
    # residual diffs of the previous computed or pruned blocks.
    # But, also limit mean_diff to be at least 2x the residual_diff_threshold
    # to avoid too aggressive pruning.
    enable_dynamic_prune_threshold: bool = False
    max_dynamic_prune_threshold: float = None
    dynamic_prune_threshold_relax_ratio: float = 1.25
    # Residual cache update interval, in steps.
    residual_cache_update_interval: int = 1

    # Buffer for storing the residuals and other tensors
    buffers: Dict[str, Any] = dataclasses.field(default_factory=dict)

    # Other settings
    downsample_factor: int = 1
    num_inference_steps: int = -1
    warmup_steps: int = 0  # DON'T pruned in warmup steps
    # DON'T prune if the number of pruned steps >= max_pruned_steps
    max_pruned_steps: int = -1

    # Record the steps that have been cached, both cached and non-cache
    executed_steps: int = 0  # cache + non-cache steps pippeline
    # steps for transformer, for CFG, transformer_executed_steps will
    # be double of executed_steps.
    transformer_executed_steps: int = 0

    pruned_blocks: List[int] = dataclasses.field(default_factory=list)
    actual_blocks: List[int] = dataclasses.field(default_factory=list)
    # Residual diffs for each step, [step: list[float]]
    residual_diffs: Dict[str, List[float]] = dataclasses.field(
        default_factory=lambda: defaultdict(list),
    )

    @torch.compiler.disable
    def get_residual_diff_threshold(self):
        residual_diff_threshold = self.residual_diff_threshold
        if self.l1_hidden_states_diff_threshold is not None:
            # Use the L1 hidden states diff threshold if set
            residual_diff_threshold = self.l1_hidden_states_diff_threshold
        if isinstance(residual_diff_threshold, torch.Tensor):
            residual_diff_threshold = residual_diff_threshold.item()
        if self.enable_dynamic_prune_threshold:
            # Compute the dynamic prune threshold based on the mean of the
            # residual diffs of the previous computed or pruned blocks.
            step = self.get_current_step()
            if step >= 0 and step in self.residual_diffs:
                # TODO: Should we only use the last 5 diffs
                diffs = self.residual_diffs[step][:]
                diffs = [d for d in diffs if d > 0.0]
                if diffs:
                    mean_diff = sum(diffs) / len(diffs)
                    relaxed_diff = (
                        mean_diff * self.dynamic_prune_threshold_relax_ratio
                    )
                    if self.max_dynamic_prune_threshold is None:
                        max_dynamic_prune_threshold = (
                            2 * residual_diff_threshold
                        )
                    else:
                        max_dynamic_prune_threshold = (
                            self.max_dynamic_prune_threshold
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

    @torch.compiler.disable
    def get_buffer(self, name):
        return self.buffers.get(name)

    @torch.compiler.disable
    def set_buffer(self, name, buffer):
        self.buffers[name] = buffer

    @torch.compiler.disable
    def remove_buffer(self, name):
        if name in self.buffers:
            del self.buffers[name]

    @torch.compiler.disable
    def clear_buffers(self):
        self.buffers.clear()

    @torch.compiler.disable
    def mark_step_begin(self):
        self.executed_steps += 1
        if self.get_current_step() == 0:
            self.pruned_blocks.clear()
            self.actual_blocks.clear()
            self.residual_diffs.clear()

    @torch.compiler.disable
    def add_pruned_block(self, num_blocks):
        self.pruned_blocks.append(num_blocks)

    @torch.compiler.disable
    def add_actual_block(self, num_blocks):
        self.actual_blocks.append(num_blocks)

    @torch.compiler.disable
    def add_residual_diff(self, diff):
        if isinstance(diff, torch.Tensor):
            diff = diff.item()
        step = self.get_current_step()
        self.residual_diffs[step].append(diff)
        max_num_block_diffs = 1000
        # Avoid memory leak, keep only the last 1000 diffs
        if len(self.residual_diffs[step]) > max_num_block_diffs:
            self.residual_diffs[step] = self.residual_diffs[step][
                -max_num_block_diffs:
            ]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Step {step}, block: {len(self.residual_diffs[step])}, "
                f"residual diff: {diff:.6f}"
            )

    @torch.compiler.disable
    def get_current_step(self):
        return self.executed_steps - 1

    @torch.compiler.disable
    def is_in_warmup(self):
        return self.get_current_step() < self.warmup_steps


@torch.compiler.disable
def get_residual_diff_threshold():
    prune_context = get_current_prune_context()
    assert prune_context is not None, "prune_context must be set before"
    return prune_context.get_residual_diff_threshold()


@torch.compiler.disable
def get_buffer(name):
    prune_context = get_current_prune_context()
    assert prune_context is not None, "prune_context must be set before"
    return prune_context.get_buffer(name)


@torch.compiler.disable
def set_buffer(name, buffer):
    prune_context = get_current_prune_context()
    assert prune_context is not None, "prune_context must be set before"
    prune_context.set_buffer(name, buffer)


@torch.compiler.disable
def remove_buffer(name):
    prune_context = get_current_prune_context()
    assert prune_context is not None, "prune_context must be set before"
    prune_context.remove_buffer(name)


@torch.compiler.disable
def mark_step_begin():
    prune_context = get_current_prune_context()
    assert prune_context is not None, "prune_context must be set before"
    prune_context.mark_step_begin()


@torch.compiler.disable
def get_current_step():
    prune_context = get_current_prune_context()
    assert prune_context is not None, "prune_context must be set before"
    return prune_context.get_current_step()


@torch.compiler.disable
def get_max_pruned_steps():
    prune_context = get_current_prune_context()
    assert prune_context is not None, "prune_context must be set before"
    return prune_context.max_pruned_steps


@torch.compiler.disable
def add_pruned_block(num_blocks):
    assert (
        isinstance(num_blocks, int) and num_blocks >= 0
    ), "num_blocks must be a non-negative integer"
    prune_context = get_current_prune_context()
    assert prune_context is not None, "prune_context must be set before"
    prune_context.add_pruned_block(num_blocks)


@torch.compiler.disable
def get_pruned_blocks():
    prune_context = get_current_prune_context()
    assert prune_context is not None, "prune_context must be set before"
    return prune_context.pruned_blocks.copy()


@torch.compiler.disable
def add_actual_block(num_blocks):
    assert (
        isinstance(num_blocks, int) and num_blocks >= 0
    ), "num_blocks must be a non-negative integer"
    prune_context = get_current_prune_context()
    assert prune_context is not None, "prune_context must be set before"
    prune_context.add_actual_block(num_blocks)


@torch.compiler.disable
def get_actual_blocks():
    prune_context = get_current_prune_context()
    assert prune_context is not None, "prune_context must be set before"
    return prune_context.actual_blocks.copy()


@torch.compiler.disable
def get_pruned_steps():
    prune_context = get_current_prune_context()
    assert prune_context is not None, "prune_context must be set before"
    pruned_blocks = get_pruned_blocks()
    pruned_blocks = [x for x in pruned_blocks if x > 0]
    return len(pruned_blocks)


@torch.compiler.disable
def is_in_warmup():
    prune_context = get_current_prune_context()
    assert prune_context is not None, "prune_context must be set before"
    return prune_context.is_in_warmup()


@torch.compiler.disable
def is_l1_diff_enabled():
    prune_context = get_current_prune_context()
    assert prune_context is not None, "prune_context must be set before"
    return (
        prune_context.l1_hidden_states_diff_threshold is not None
        and prune_context.l1_hidden_states_diff_threshold > 0.0
    )


@torch.compiler.disable
def add_residual_diff(diff):
    prune_context = get_current_prune_context()
    assert prune_context is not None, "prune_context must be set before"
    prune_context.add_residual_diff(diff)


@torch.compiler.disable
def get_residual_diffs():
    prune_context = get_current_prune_context()
    assert prune_context is not None, "prune_context must be set before"
    # Return a copy of the residual diffs to avoid modification
    return prune_context.residual_diffs.copy()


@torch.compiler.disable
def get_important_condition_threshold():
    prune_context = get_current_prune_context()
    assert prune_context is not None, "prune_context must be set before"
    return prune_context.important_condition_threshold


@torch.compiler.disable
def residual_cache_update_interval():
    prune_context = get_current_prune_context()
    assert prune_context is not None, "prune_context must be set before"
    return prune_context.residual_cache_update_interval


@torch.compiler.disable
def Fn_compute_blocks():
    prune_context = get_current_prune_context()
    assert prune_context is not None, "prune_context must be set before"
    assert (
        prune_context.Fn_compute_blocks >= 0
    ), "Fn_compute_blocks must be >= 0"
    return prune_context.Fn_compute_blocks


@torch.compiler.disable
def Bn_compute_blocks():
    prune_context = get_current_prune_context()
    assert prune_context is not None, "prune_context must be set before"
    assert (
        prune_context.Bn_compute_blocks >= 0
    ), "Bn_compute_blocks must be >= 0"
    return prune_context.Bn_compute_blocks


@torch.compiler.disable
def get_non_prune_blocks_ids():
    prune_context = get_current_prune_context()
    assert prune_context is not None, "prune_context must be set before"
    return prune_context.non_prune_blocks_ids


_current_prune_context: DBPPruneContext = None


def create_prune_context(*args, **kwargs):
    return DBPPruneContext(*args, **kwargs)


def get_current_prune_context():
    return _current_prune_context


def set_current_prune_context(prune_context=None):
    global _current_prune_context
    _current_prune_context = prune_context


def collect_prune_kwargs(default_attrs: dict, **kwargs):
    # NOTE: This API will split kwargs into prune_kwargs and other_kwargs
    # default_attrs: specific settings for different pipelines
    prune_attrs = dataclasses.fields(DBPPruneContext)
    prune_attrs = [
        attr
        for attr in prune_attrs
        if hasattr(
            DBPPruneContext,
            attr.name,
        )
    ]
    prune_kwargs = {
        attr.name: kwargs.pop(
            attr.name,
            getattr(DBPPruneContext, attr.name),
        )
        for attr in prune_attrs
    }

    # Manually set sequence fields, such as non_prune_blocks_ids
    def _safe_set_sequence_field(
        field_name: str,
        default_value: Any = None,
    ):
        if field_name not in prune_kwargs:
            prune_kwargs[field_name] = kwargs.pop(
                field_name,
                default_value,
            )

    _safe_set_sequence_field("non_prune_blocks_ids", [])

    assert default_attrs is not None, "default_attrs must be set before"
    for attr in prune_attrs:
        if attr.name in default_attrs:
            prune_kwargs[attr.name] = default_attrs[attr.name]

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Collected DBPrune kwargs: {prune_kwargs}")

    return prune_kwargs, kwargs


@contextlib.contextmanager
def prune_context(prune_context):
    global _current_prune_context
    old_prune_context = _current_prune_context
    _current_prune_context = prune_context
    try:
        yield
    finally:
        _current_prune_context = old_prune_context


@torch.compiler.disable
def are_two_tensors_similar(
    t1: torch.Tensor,  # prev residual R(t-1,n) = H(t-1,n) - H(t-1,0)
    t2: torch.Tensor,  # curr residual R(t  ,n) = H(t  ,n) - H(t  ,0)
    *,
    threshold: float,
    parallelized: bool = False,
    name: str = "Bn",  # for debugging
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
            logger.debug(f"{name}, shape error: {t1.shape} != {t2.shape}")
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
        mean_diff = primitives.all_reduce_sync(mean_diff, "avg")
        mean_t1 = primitives.all_reduce_sync(mean_t1, "avg")

    # D = (t1 - t2) / t1 = 1 - (t2 / t1), if D = 0, then t1 = t2.
    # Futher, if we assume that (H(t,  0) - H(t-1,0)) ~ 0, then,
    # H(t-1,n) ~ H(t  ,n), which means the hidden states are similar.
    diff = (mean_diff / mean_t1).item()

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"{name}, diff: {diff:.6f}, threshold: {threshold:.6f}")

    add_residual_diff(diff)

    return diff < threshold


@torch.compiler.disable
def apply_hidden_states_residual(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    name: str = "Bn",
    encoder_name: str = "Bn_encoder",
):
    hidden_states_residual = get_buffer(f"{name}")

    assert hidden_states_residual is not None, f"{name} must be set before"
    hidden_states = hidden_states_residual + hidden_states

    encoder_hidden_states_residual = get_buffer(f"{encoder_name}")
    assert (
        encoder_hidden_states_residual is not None
    ), f"{encoder_name} must be set before"
    encoder_hidden_states = (
        encoder_hidden_states_residual + encoder_hidden_states
    )

    hidden_states = hidden_states.contiguous()
    encoder_hidden_states = encoder_hidden_states.contiguous()

    return hidden_states, encoder_hidden_states


@torch.compiler.disable
def get_downsample_factor():
    prune_context = get_current_prune_context()
    assert prune_context is not None, "prune_context must be set before"
    return prune_context.downsample_factor


@torch.compiler.disable
def get_can_use_prune(
    states_tensor: torch.Tensor,  # hidden_states or residual
    parallelized: bool = False,
    threshold: Optional[float] = None,  # can manually set threshold
    name: str = "Bn",
):
    if is_in_warmup():
        return False

    pruned_steps = get_pruned_steps()
    max_pruned_steps = get_max_pruned_steps()
    if max_pruned_steps >= 0 and (pruned_steps >= max_pruned_steps):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"{name}, max_pruned_steps reached: {max_pruned_steps}, "
                "cannot use prune."
            )
        return False

    if threshold is None or threshold <= 0.0:
        threshold = get_residual_diff_threshold()
    if threshold <= 0.0:
        return False

    downsample_factor = get_downsample_factor()
    prev_states_tensor = get_buffer(f"{name}")

    if downsample_factor > 1:
        states_tensor = states_tensor[..., ::downsample_factor]
        states_tensor = states_tensor.contiguous()
        if prev_states_tensor is not None:
            prev_states_tensor = prev_states_tensor[..., ::downsample_factor]
            prev_states_tensor = prev_states_tensor.contiguous()

    return prev_states_tensor is not None and are_two_tensors_similar(
        prev_states_tensor,
        states_tensor,
        threshold=threshold,
        parallelized=parallelized,
        name=name,
    )
