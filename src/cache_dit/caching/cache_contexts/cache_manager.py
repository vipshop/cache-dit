import logging
import contextlib
from typing import Dict, Optional, Tuple, Union, List

import torch
import torch.distributed as dist

from .calibrators import CalibratorBase
from .cache_context import CachedContext
from cache_dit.logger import init_logger

logger = init_logger(__name__)


class ContextNotExistError(Exception):
    pass


class CachedContextManager:

    def __init__(self, name: str = None, persistent_context: bool = False):
        self.name = name
        self._current_context: CachedContext = None
        self._cached_context_manager: Dict[str, CachedContext] = {}
        # Whether to create new context automatically when setting
        # a non-exist context name. Persistent context is useful when
        # the pipeline class is not provided and users want to use
        # cache-dit in a transformer-only way.
        self._persistent_context = persistent_context
        self._current_step_refreshed: bool = False

    @property
    def persistent_context(self) -> bool:
        return self._persistent_context

    @property
    def current_context(self) -> CachedContext:
        return self._current_context

    @property
    @torch.compiler.disable
    def current_step_refreshed(self) -> bool:
        return self._current_step_refreshed

    @torch.compiler.disable
    def is_pre_refreshed(self) -> bool:
        _context = self._current_context
        if _context is None:
            return False

        num_inference_steps = _context.cache_config.num_inference_steps
        if num_inference_steps is not None:
            current_step = _context.get_current_step()  # e.g, 0~49,50~99,...
            return current_step == num_inference_steps - 1
        # If num_inference_steps is None, always return True, thus will make
        # `apply_stats_hooks` called after each forward when persistent_context is True.
        # Otherwise, we will lost the accurate cached stats after each request.
        return True

    @torch.compiler.disable
    def new_context(self, *args, **kwargs) -> CachedContext:
        _context = CachedContext(*args, **kwargs)
        # NOTE: Patch args and kwargs for implicit refresh.
        _context._init_args = args  # maybe empty tuple: ()
        _context._init_kwargs = kwargs  # maybe empty dict: {}
        self._cached_context_manager[_context.name] = _context
        return _context

    @torch.compiler.disable
    def maybe_refresh(
        self,
        cached_context: Optional[CachedContext | str] = None,
    ) -> bool:
        if cached_context is None:
            _context = self._current_context
            assert _context is not None, "Current context is not set!"

        if isinstance(cached_context, CachedContext):
            _context = cached_context
        else:
            if cached_context not in self._cached_context_manager:
                raise ContextNotExistError("Context not exist!")
            _context = self._cached_context_manager[cached_context]

        num_inference_steps = _context.cache_config.num_inference_steps
        if num_inference_steps is not None:
            current_step = _context.get_current_step()  # e.g, 0~49,50~99,...
            # Another round of inference, need to refresh cache context.
            if current_step >= num_inference_steps:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Refreshing cache context '{_context.name}' "
                        f"as current step: {current_step} >= "
                        f"num_inference_steps: {num_inference_steps}."
                    )
                return True
        return False

    @torch.compiler.disable
    def set_context(
        self,
        cached_context: CachedContext | str,
        *args,
        **kwargs,
    ) -> CachedContext:
        if isinstance(cached_context, CachedContext):
            self._current_context = cached_context
        else:
            if cached_context not in self._cached_context_manager:
                if not self._persistent_context:
                    raise ContextNotExistError(
                        "Context not exist and persistent_context is False. Please "
                        "create new context first or set persistent_context=True."
                    )
                else:
                    # Create new context if not exist
                    if any((bool(args), bool(kwargs))):
                        kwargs["name"] = cached_context
                        self._current_context = self.new_context(*args, **kwargs)
                    else:
                        raise ValueError("To create new context, please provide args and kwargs.")
            else:
                self._current_context = self._cached_context_manager[cached_context]

        if self.maybe_refresh(self._current_context):
            if not any((bool(args), bool(kwargs))):
                assert hasattr(self._current_context, "_init_args")
                assert hasattr(self._current_context, "_init_kwargs")
                args = self._current_context._init_args
                kwargs = self._current_context._init_kwargs

            self._current_context = self.reset_context(self._current_context, *args, **kwargs)
            self._current_step_refreshed = True
        else:
            self._current_step_refreshed = False

        return self._current_context

    def get_context(self, name: str = None) -> CachedContext:
        if name is not None:
            if name not in self._cached_context_manager:
                raise ContextNotExistError("Context not exist!")
            return self._cached_context_manager[name]
        return self._current_context

    def reset_context(
        self,
        cached_context: CachedContext | str,
        *args,
        **kwargs,
    ) -> CachedContext:
        if isinstance(cached_context, CachedContext):
            old_context_name = cached_context.name
            if cached_context.name in self._cached_context_manager:
                cached_context.clear_buffers()
                del self._cached_context_manager[cached_context.name]
            # force use old_context name
            kwargs["name"] = old_context_name
            _context = self.new_context(*args, **kwargs)
        else:
            old_context_name = cached_context
            if cached_context in self._cached_context_manager:
                self._cached_context_manager[cached_context].clear_buffers()
                del self._cached_context_manager[cached_context]
            # force use old_context name
            kwargs["name"] = old_context_name
            _context = self.new_context(*args, **kwargs)
        return _context

    def remove_context(self, cached_context: CachedContext | str):
        if isinstance(cached_context, CachedContext):
            cached_context.clear_buffers()
            if cached_context.name in self._cached_context_manager:
                del self._cached_context_manager[cached_context.name]
        else:
            if cached_context in self._cached_context_manager:
                self._cached_context_manager[cached_context].clear_buffers()
                del self._cached_context_manager[cached_context]

    def clear_contexts(self):
        for context_name in list(self._cached_context_manager.keys()):
            self.remove_context(context_name)

    @contextlib.contextmanager
    def enter_context(self, cached_context: CachedContext | str):
        old_cached_context = self._current_context
        if isinstance(cached_context, CachedContext):
            self._current_context = cached_context
        else:
            self._current_context = self._cached_context_manager[cached_context]
        try:
            yield
        finally:
            self._current_context = old_cached_context

    @torch.compiler.disable
    def get_residual_diff_threshold(self) -> float:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.get_residual_diff_threshold()

    @torch.compiler.disable
    def get_buffer(self, name) -> torch.Tensor:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.get_buffer(name)

    @torch.compiler.disable
    def set_buffer(self, name, buffer):
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        cached_context.set_buffer(name, buffer)

    @torch.compiler.disable
    def remove_buffer(self, name):
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        cached_context.remove_buffer(name)

    @torch.compiler.disable
    def mark_step_begin(self):
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        cached_context.mark_step_begin()

    @torch.compiler.disable
    def get_current_step(self) -> int:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.get_current_step()

    @torch.compiler.disable
    def get_current_step_residual_diff(self) -> float:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        step = str(self.get_current_step())
        residual_diffs = self.get_residual_diffs()
        if step in residual_diffs:
            return residual_diffs[step]
        return None

    @torch.compiler.disable
    def get_current_step_cfg_residual_diff(self) -> float:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        step = str(self.get_current_step())
        cfg_residual_diffs = self.get_cfg_residual_diffs()
        if step in cfg_residual_diffs:
            return cfg_residual_diffs[step]
        return None

    @torch.compiler.disable
    def get_current_transformer_step(self) -> int:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.get_current_transformer_step()

    @torch.compiler.disable
    def get_cached_steps(self) -> List[int]:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.get_cached_steps()

    @torch.compiler.disable
    def get_cfg_cached_steps(self) -> List[int]:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.get_cfg_cached_steps()

    @torch.compiler.disable
    def get_max_cached_steps(self) -> int:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.cache_config.max_cached_steps

    @torch.compiler.disable
    def get_max_continuous_cached_steps(self) -> int:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.cache_config.max_continuous_cached_steps

    @torch.compiler.disable
    def get_continuous_cached_steps(self) -> int:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.continuous_cached_steps

    @torch.compiler.disable
    def get_cfg_continuous_cached_steps(self) -> int:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.cfg_continuous_cached_steps

    @torch.compiler.disable
    def add_cached_step(self):
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        cached_context.add_cached_step()

    @torch.compiler.disable
    def add_residual_diff(self, diff):
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        cached_context.add_residual_diff(diff)

    @torch.compiler.disable
    def get_residual_diffs(self) -> Dict[str, float]:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.get_residual_diffs()

    @torch.compiler.disable
    def get_cfg_residual_diffs(self) -> Dict[str, float]:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.get_cfg_residual_diffs()

    @torch.compiler.disable
    def get_accumulated_residual_diff(self) -> float:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.get_accumulated_residual_diff()

    @torch.compiler.disable
    def get_cfg_accumulated_residual_diff(self) -> float:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.get_cfg_accumulated_residual_diff()

    @torch.compiler.disable
    def max_accumulated_residual_diff_threshold(self) -> float:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.cache_config.max_accumulated_residual_diff_threshold

    @torch.compiler.disable
    def is_calibrator_enabled(self) -> bool:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.enable_calibrator()

    @torch.compiler.disable
    def is_encoder_calibrator_enabled(self) -> bool:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.enable_encoder_calibrator()

    def get_calibrator(self) -> Tuple[CalibratorBase, CalibratorBase]:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.get_calibrators()

    def get_cfg_calibrator(self) -> Tuple[CalibratorBase, CalibratorBase]:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.get_cfg_calibrators()

    @torch.compiler.disable
    def is_calibrator_cache_residual(self) -> bool:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.calibrator_cache_type() == "residual"

    @torch.compiler.disable
    def is_cache_residual(self) -> bool:
        if self.is_calibrator_enabled():
            # residual or hidden_states
            return self.is_calibrator_cache_residual()
        return True

    @torch.compiler.disable
    def is_encoder_cache_residual(self) -> bool:
        if self.is_encoder_calibrator_enabled():
            # residual or hidden_states
            return self.is_calibrator_cache_residual()
        return True

    @torch.compiler.disable
    def is_in_warmup(self) -> bool:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.is_in_warmup()

    @torch.compiler.disable
    def is_in_full_compute_steps(self) -> bool:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.is_in_full_compute_steps()

    @torch.compiler.disable
    def is_steps_computation_mask_enabled(self) -> bool:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.cache_config.steps_computation_mask is not None

    @torch.compiler.disable
    def get_steps_computation_policy(self) -> str:  # dynamic/static
        # If enabled steps_computation_mask w/ static cache, maybe use
        # at the very beginning of cache blocks forward. NO, Fn blocks
        # compute first.
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.get_steps_computation_policy()

    @torch.compiler.disable
    def is_l1_diff_enabled(self) -> bool:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return (
            cached_context.extra_cache_config.l1_hidden_states_diff_threshold is not None
            and cached_context.extra_cache_config.l1_hidden_states_diff_threshold > 0.0
        )

    @torch.compiler.disable
    def get_important_condition_threshold(self) -> float:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.extra_cache_config.important_condition_threshold

    @torch.compiler.disable
    def Fn_compute_blocks(self) -> int:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        assert cached_context.cache_config.Fn_compute_blocks >= 1, "Fn_compute_blocks must be >= 1"
        return cached_context.cache_config.Fn_compute_blocks

    @torch.compiler.disable
    def Bn_compute_blocks(self) -> int:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        assert cached_context.cache_config.Bn_compute_blocks >= 0, "Bn_compute_blocks must be >= 0"
        return cached_context.cache_config.Bn_compute_blocks

    @torch.compiler.disable
    def enable_separate_cfg(self) -> bool:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.cache_config.enable_separate_cfg

    @torch.compiler.disable
    def is_separate_cfg_step(self) -> bool:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.is_separate_cfg_step()

    @torch.compiler.disable
    def cfg_diff_compute_separate(self) -> bool:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.cache_config.cfg_diff_compute_separate

    @torch.compiler.disable
    def similarity(
        self,
        t1: torch.Tensor,  # prev residual R(t-1,n) = H(t-1,n) - H(t-1,0)
        t2: torch.Tensor,  # curr residual R(t  ,n) = H(t  ,n) - H(t  ,0)
        *,
        threshold: float,
        parallelized: bool = False,
        prefix: str = "Fn",  # for debugging
    ) -> bool:
        # Special case for threshold, 0.0 means the threshold is disabled, -1.0 means
        # the threshold is always enabled, -2.0 means the shape is not matched.
        if threshold <= 0.0:
            self.add_residual_diff(-0.0)
            return False

        if threshold >= 1.0:
            # If threshold is 1.0 or more, we consider them always similar.
            self.add_residual_diff(-1.0)
            return True

        if t1.shape != t2.shape:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"{prefix}, shape error: {t1.shape} != {t2.shape}")
            self.add_residual_diff(-2.0)
            return False

        if all(
            (
                self.enable_separate_cfg(),
                self.is_separate_cfg_step(),
                not self.cfg_diff_compute_separate(),
                self.get_current_step_residual_diff() is not None,
            )
        ):
            # Reuse computed diff value from non-CFG step
            diff = self.get_current_step_residual_diff()
        else:
            # Find the most significant token through t1 and t2, and
            # consider the diff of the significant token. The more significant,
            # the more important.
            condition_thresh = self.get_important_condition_threshold()
            if condition_thresh > 0.0:
                raw_diff = (t1 - t2).abs()  # [B, seq_len, d]
                token_m_df = raw_diff.mean(dim=-1)  # [B, seq_len]
                token_m_t1 = t1.abs().mean(dim=-1)  # [B, seq_len]
                # D = (t1 - t2) / t1 = 1 - (t2 / t1), if D = 0, then t1 = t2.
                token_diff = token_m_df / token_m_t1  # [B, seq_len]
                condition: torch.Tensor = token_diff > condition_thresh  # [B, seq_len]
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
                dist.all_reduce(mean_diff, op=dist.ReduceOp.AVG)
                dist.all_reduce(mean_t1, op=dist.ReduceOp.AVG)

            # D = (t1 - t2) / t1 = 1 - (t2 / t1), if D = 0, then t1 = t2.
            # Futher, if we assume that (H(t,  0) - H(t-1,0)) ~ 0, then,
            # H(t-1,n) ~ H(t  ,n), which means the hidden states are similar.
            diff = (mean_diff / mean_t1).item()

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"{prefix}, diff: {diff:.6f}, threshold: {threshold:.6f}")

        self.add_residual_diff(diff)

        return diff < threshold

    def _debugging_set_buffer(self, prefix):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"set {prefix}, "
                f"transformer step: {self.get_current_transformer_step()}, "
                f"executed step: {self.get_current_step()}"
            )

    def _debugging_get_buffer(self, prefix):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"get {prefix}, "
                f"transformer step: {self.get_current_transformer_step()}, "
                f"executed step: {self.get_current_step()}"
            )

    # Fn buffers
    @torch.compiler.disable
    def set_Fn_buffer(self, buffer: torch.Tensor, prefix: str = "Fn"):
        # DON'T set None Buffer
        if buffer is None:
            return
        # Set hidden_states or residual for Fn blocks.
        # This buffer is only use for L1 diff calculation.
        downsample_factor = self.get_downsample_factor()
        if downsample_factor > 1:
            buffer = buffer[..., ::downsample_factor]
            buffer = buffer.contiguous()
        if self.is_separate_cfg_step():
            self._debugging_set_buffer(f"{prefix}_buffer_cfg")
            self.set_buffer(f"{prefix}_buffer_cfg", buffer)
        else:
            self._debugging_set_buffer(f"{prefix}_buffer")
            self.set_buffer(f"{prefix}_buffer", buffer)

    @torch.compiler.disable
    def get_Fn_buffer(self, prefix: str = "Fn") -> torch.Tensor:
        if self.is_separate_cfg_step():
            self._debugging_get_buffer(f"{prefix}_buffer_cfg")
            return self.get_buffer(f"{prefix}_buffer_cfg")
        self._debugging_get_buffer(f"{prefix}_buffer")
        return self.get_buffer(f"{prefix}_buffer")

    @torch.compiler.disable
    def set_Fn_encoder_buffer(self, buffer: torch.Tensor, prefix: str = "Fn"):
        # DON'T set None Buffer
        if buffer is None:
            return
        if self.is_separate_cfg_step():
            self._debugging_set_buffer(f"{prefix}_encoder_buffer_cfg")
            self.set_buffer(f"{prefix}_encoder_buffer_cfg", buffer)
        else:
            self._debugging_set_buffer(f"{prefix}_encoder_buffer")
            self.set_buffer(f"{prefix}_encoder_buffer", buffer)

    @torch.compiler.disable
    def get_Fn_encoder_buffer(self, prefix: str = "Fn") -> torch.Tensor:
        if self.is_separate_cfg_step():
            self._debugging_get_buffer(f"{prefix}_encoder_buffer_cfg")
            return self.get_buffer(f"{prefix}_encoder_buffer_cfg")
        self._debugging_get_buffer(f"{prefix}_encoder_buffer")
        return self.get_buffer(f"{prefix}_encoder_buffer")

    # Bn buffers
    @torch.compiler.disable
    def set_Bn_buffer(self, buffer: torch.Tensor, prefix: str = "Bn"):
        # DON'T set None Buffer
        if buffer is None:
            return
        # Set hidden_states or residual for Bn blocks.
        # This buffer is use for hidden states approximation.
        if self.is_calibrator_enabled():
            # calibrator, encoder_calibrator
            if self.is_separate_cfg_step():
                calibrator, _ = self.get_cfg_calibrator()
            else:
                calibrator, _ = self.get_calibrator()

            if calibrator is not None:
                # Use calibrator to update the buffer
                calibrator.update(buffer, name=prefix)
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "calibrator is enabled but not set in the cache context. "
                        "Falling back to default buffer retrieval."
                    )
                if self.is_separate_cfg_step():
                    self._debugging_set_buffer(f"{prefix}_buffer_cfg")
                    self.set_buffer(f"{prefix}_buffer_cfg", buffer)
                else:
                    self._debugging_set_buffer(f"{prefix}_buffer")
                    self.set_buffer(f"{prefix}_buffer", buffer)
        else:
            if self.is_separate_cfg_step():
                self._debugging_set_buffer(f"{prefix}_buffer_cfg")
                self.set_buffer(f"{prefix}_buffer_cfg", buffer)
            else:
                self._debugging_set_buffer(f"{prefix}_buffer")
                self.set_buffer(f"{prefix}_buffer", buffer)

    @torch.compiler.disable
    def get_Bn_buffer(self, prefix: str = "Bn") -> torch.Tensor:
        if self.is_calibrator_enabled():
            # calibrator, encoder_calibrator
            if self.is_separate_cfg_step():
                calibrator, _ = self.get_cfg_calibrator()
            else:
                calibrator, _ = self.get_calibrator()

            if calibrator is not None:
                return calibrator.approximate(name=prefix)
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "calibrator is enabled but not set in the cache context. "
                        "Falling back to default buffer retrieval."
                    )
                # Fallback to default buffer retrieval
                if self.is_separate_cfg_step():
                    self._debugging_get_buffer(f"{prefix}_buffer_cfg")
                    return self.get_buffer(f"{prefix}_buffer_cfg")
                self._debugging_get_buffer(f"{prefix}_buffer")
                return self.get_buffer(f"{prefix}_buffer")
        else:
            if self.is_separate_cfg_step():
                self._debugging_get_buffer(f"{prefix}_buffer_cfg")
                return self.get_buffer(f"{prefix}_buffer_cfg")
            self._debugging_get_buffer(f"{prefix}_buffer")
            return self.get_buffer(f"{prefix}_buffer")

    @torch.compiler.disable
    def set_Bn_encoder_buffer(self, buffer: torch.Tensor | None, prefix: str = "Bn"):
        # DON'T set None Buffer
        if buffer is None:
            return

        # This buffer is use for encoder hidden states approximation.
        if self.is_encoder_calibrator_enabled():
            # calibrator, encoder_calibrator
            if self.is_separate_cfg_step():
                _, encoder_calibrator = self.get_cfg_calibrator()
            else:
                _, encoder_calibrator = self.get_calibrator()

            if encoder_calibrator is not None:
                # Use CalibratorBase to update the buffer
                encoder_calibrator.update(buffer, name=prefix)
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "CalibratorBase is enabled but not set in the cache context. "
                        "Falling back to default buffer retrieval."
                    )
                if self.is_separate_cfg_step():
                    self._debugging_set_buffer(f"{prefix}_encoder_buffer_cfg")
                    self.set_buffer(f"{prefix}_encoder_buffer_cfg", buffer)
                else:
                    self._debugging_set_buffer(f"{prefix}_encoder_buffer")
                    self.set_buffer(f"{prefix}_encoder_buffer", buffer)
        else:
            if self.is_separate_cfg_step():
                self._debugging_set_buffer(f"{prefix}_encoder_buffer_cfg")
                self.set_buffer(f"{prefix}_encoder_buffer_cfg", buffer)
            else:
                self._debugging_set_buffer(f"{prefix}_encoder_buffer")
                self.set_buffer(f"{prefix}_encoder_buffer", buffer)

    @torch.compiler.disable
    def get_Bn_encoder_buffer(self, prefix: str = "Bn") -> torch.Tensor:
        if self.is_encoder_calibrator_enabled():
            if self.is_separate_cfg_step():
                _, encoder_calibrator = self.get_cfg_calibrator()
            else:
                _, encoder_calibrator = self.get_calibrator()

            if encoder_calibrator is not None:
                # Use calibrator to approximate the value
                return encoder_calibrator.approximate(name=prefix)
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        "calibrator is enabled but not set in the cache context. "
                        "Falling back to default buffer retrieval."
                    )
                # Fallback to default buffer retrieval
                if self.is_separate_cfg_step():
                    self._debugging_get_buffer(f"{prefix}_encoder_buffer_cfg")
                    return self.get_buffer(f"{prefix}_encoder_buffer_cfg")
                self._debugging_get_buffer(f"{prefix}_encoder_buffer")
                return self.get_buffer(f"{prefix}_encoder_buffer")
        else:
            if self.is_separate_cfg_step():
                self._debugging_get_buffer(f"{prefix}_encoder_buffer_cfg")
                return self.get_buffer(f"{prefix}_encoder_buffer_cfg")
            self._debugging_get_buffer(f"{prefix}_encoder_buffer")
            return self.get_buffer(f"{prefix}_encoder_buffer")

    @torch.compiler.disable
    def apply_cache(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        prefix: str = "Bn",
        encoder_prefix: str = "Bn_encoder",
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        # Allow Bn and Fn prefix to be used for residual cache.
        if "Bn" in prefix:
            hidden_states_prev = self.get_Bn_buffer(prefix)
        else:
            hidden_states_prev = self.get_Fn_buffer(prefix)

        assert hidden_states_prev is not None, f"{prefix}_buffer must be set before"

        if self.is_cache_residual():
            hidden_states = hidden_states_prev + hidden_states
        else:
            # If cache is not residual, we use the hidden states directly
            hidden_states = hidden_states_prev

        hidden_states = hidden_states.contiguous()

        if encoder_hidden_states is not None:
            if "Bn" in encoder_prefix:
                encoder_hidden_states_prev = self.get_Bn_encoder_buffer(encoder_prefix)
            else:
                encoder_hidden_states_prev = self.get_Fn_encoder_buffer(encoder_prefix)

            if encoder_hidden_states_prev is not None:

                if self.is_encoder_cache_residual():
                    encoder_hidden_states = encoder_hidden_states_prev + encoder_hidden_states
                else:
                    # If encoder cache is not residual, we use the encoder hidden states directly
                    encoder_hidden_states = encoder_hidden_states_prev

            encoder_hidden_states = encoder_hidden_states.contiguous()

        return hidden_states, encoder_hidden_states

    @torch.compiler.disable
    def get_downsample_factor(self) -> float:
        cached_context = self.get_context()
        assert cached_context is not None, "cached_context must be set before"
        return cached_context.extra_cache_config.downsample_factor

    @torch.compiler.disable
    def can_cache(
        self,
        states_tensor: torch.Tensor,  # hidden_states or residual
        parallelized: bool = False,
        threshold: Optional[float] = None,  # can manually set threshold
        prefix: str = "Fn",
    ) -> bool:

        if self.is_in_warmup():
            return False

        # if enabled steps_computation_mask w/ dyanamic cache
        if self.is_steps_computation_mask_enabled():
            if self.is_in_full_compute_steps():
                return False
            else:  # In cache steps, dynaimic/static cache allowed
                if self.get_steps_computation_policy() == "static":
                    return True

        # max cached steps
        max_cached_steps = self.get_max_cached_steps()
        if not self.is_separate_cfg_step():
            cached_steps = self.get_cached_steps()
        else:
            cached_steps = self.get_cfg_cached_steps()

        if max_cached_steps >= 0 and (len(cached_steps) >= max_cached_steps):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"{prefix}, max_cached_steps reached: {max_cached_steps}, " "can not use cache."
                )
            return False

        # max continuous cached steps
        max_continuous_cached_steps = self.get_max_continuous_cached_steps()
        if not self.is_separate_cfg_step():
            continuous_cached_steps = self.get_continuous_cached_steps()
        else:
            continuous_cached_steps = self.get_cfg_continuous_cached_steps()

        if max_continuous_cached_steps >= 0 and (
            continuous_cached_steps >= max_continuous_cached_steps
        ):
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"{prefix}, max_continuous_cached_steps "
                    f"reached: {max_continuous_cached_steps}, "
                    "can not use cache."
                )
            # reset continuous cached steps stats
            cached_context = self.get_context()
            if not self.is_separate_cfg_step():
                cached_context.continuous_cached_steps = 0
            else:
                cached_context.cfg_continuous_cached_steps = 0
            return False

        # max accumulated residual diff threshold
        max_accumulated_residual_diff_threshold = self.max_accumulated_residual_diff_threshold()
        if (
            max_accumulated_residual_diff_threshold is not None
            and max_accumulated_residual_diff_threshold > 0.0
        ):
            if not self.is_separate_cfg_step():
                accumulated_residual_diff = self.get_accumulated_residual_diff()
            else:
                accumulated_residual_diff = self.get_cfg_accumulated_residual_diff()
            if accumulated_residual_diff >= max_accumulated_residual_diff_threshold:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"{prefix}, max_accumulated_residual_diff_threshold "
                        f"reached: {max_accumulated_residual_diff_threshold:.6f}, "
                        f"accumulated_residual_diff: {accumulated_residual_diff:.6f}, "
                        "can not use cache."
                    )
                return False

        if threshold is None or threshold <= 0.0:
            threshold = self.get_residual_diff_threshold()
        if threshold <= 0.0:
            return False

        downsample_factor = self.get_downsample_factor()
        if downsample_factor > 1 and "Bn" not in prefix:
            states_tensor = states_tensor[..., ::downsample_factor]
            states_tensor = states_tensor.contiguous()

        # Allow Bn and Fn prefix to be used for diff calculation.
        if "Bn" in prefix:
            prev_states_tensor = self.get_Bn_buffer(prefix)
        else:
            prev_states_tensor = self.get_Fn_buffer(prefix)

        # Dynamic cache according to the residual diff
        can_cache = prev_states_tensor is not None and self.similarity(
            prev_states_tensor,
            states_tensor,
            threshold=threshold,
            parallelized=parallelized,
            prefix=prefix,
        )
        return can_cache
