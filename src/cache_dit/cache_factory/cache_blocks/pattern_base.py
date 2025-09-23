import inspect
import asyncio
import torch
import torch.distributed as dist

from typing import List
from cache_dit.cache_factory.cache_contexts.cache_context import CachedContext
from cache_dit.cache_factory.cache_contexts.cache_manager import (
    CachedContextManager,
)
from cache_dit.cache_factory import ForwardPattern
from cache_dit.logger import init_logger

logger = init_logger(__name__)


class CachedBlocks_Pattern_Base(torch.nn.Module):
    _supported_patterns = [
        ForwardPattern.Pattern_0,
        ForwardPattern.Pattern_1,
        ForwardPattern.Pattern_2,
    ]

    def __init__(
        self,
        # 0. Transformer blocks configuration
        transformer_blocks: torch.nn.ModuleList,
        transformer: torch.nn.Module = None,
        forward_pattern: ForwardPattern = ForwardPattern.Pattern_0,
        check_forward_pattern: bool = True,
        check_num_outputs: bool = True,
        # 1. Cache context configuration
        cache_prefix: str = None,  # maybe un-need.
        cache_context: CachedContext | str = None,
        cache_manager: CachedContextManager = None,
        **kwargs,
    ):
        super().__init__()

        # 0. Transformer blocks configuration
        self.transformer = transformer
        self.transformer_blocks = transformer_blocks
        self.forward_pattern = forward_pattern
        self.check_forward_pattern = check_forward_pattern
        self.check_num_outputs = check_num_outputs
        # 1. Cache context configuration
        self.cache_prefix = cache_prefix
        self.cache_context = cache_context
        self.cache_manager = cache_manager
        self.pending_tasks: List[asyncio.Task] = []

        self._check_forward_pattern()
        logger.info(
            f"Match Cached Blocks: {self.__class__.__name__}, for "
            f"{self.cache_prefix}, cache_context: {self.cache_context}, "
            f"cache_manager: {self.cache_manager.name}."
        )

    def _check_forward_pattern(self):
        if not self.check_forward_pattern:
            logger.warning(
                f"Skipped Forward Pattern Check: {self.forward_pattern}"
            )
            return

        assert (
            self.forward_pattern.Supported
            and self.forward_pattern in self._supported_patterns
        ), f"Pattern {self.forward_pattern} is not supported now!"

        if self.transformer_blocks is not None:
            for block in self.transformer_blocks:
                # Special case for HiDreamBlock
                if hasattr(block, "block"):
                    if isinstance(block.block, torch.nn.Module):
                        block = block.block

                forward_parameters = set(
                    inspect.signature(block.forward).parameters.keys()
                )

                if self.check_num_outputs:
                    num_outputs = str(
                        inspect.signature(block.forward).return_annotation
                    ).count("torch.Tensor")

                    if num_outputs > 0:
                        assert len(self.forward_pattern.Out) == num_outputs, (
                            f"The number of block's outputs is {num_outputs} don't not "
                            f"match the number of the pattern: {self.forward_pattern}, "
                            f"Out: {len(self.forward_pattern.Out)}."
                        )

                for required_param in self.forward_pattern.In:
                    assert (
                        required_param in forward_parameters
                    ), f"The input parameters must contains: {required_param}."

    @torch.compiler.disable
    def _check_cache_params(self):
        assert self.cache_manager.Fn_compute_blocks() <= len(
            self.transformer_blocks
        ), (
            f"Fn_compute_blocks {self.cache_manager.Fn_compute_blocks()} must be less than "
            f"the number of transformer blocks {len(self.transformer_blocks)}"
        )
        assert self.cache_manager.Bn_compute_blocks() <= len(
            self.transformer_blocks
        ), (
            f"Bn_compute_blocks {self.cache_manager.Bn_compute_blocks()} must be less than "
            f"the number of transformer blocks {len(self.transformer_blocks)}"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        # Use it's own cache context.
        self.cache_manager.set_context(self.cache_context)
        self._check_cache_params()

        original_hidden_states = hidden_states
        # Call first `n` blocks to process the hidden states for
        # more stable diff calculation.
        hidden_states, encoder_hidden_states = self.call_Fn_blocks(
            hidden_states,
            encoder_hidden_states,
            *args,
            **kwargs,
        )

        Fn_hidden_states_residual = hidden_states - original_hidden_states
        del original_hidden_states

        self.cache_manager.mark_step_begin()
        # Residual L1 diff or Hidden States L1 diff
        can_use_cache = self.cache_manager.can_cache(
            (
                Fn_hidden_states_residual
                if not self.cache_manager.is_l1_diff_enabled()
                else hidden_states
            ),
            parallelized=self._is_parallelized(),
            prefix=(
                f"{self.cache_prefix}_Fn_residual"
                if not self.cache_manager.is_l1_diff_enabled()
                else f"{self.cache_prefix}_Fn_hidden_states"
            ),
        )

        torch._dynamo.graph_break()
        if can_use_cache:
            self.cache_manager.add_cached_step()
            del Fn_hidden_states_residual
            hidden_states, encoder_hidden_states = (
                self.cache_manager.apply_cache(
                    hidden_states,
                    encoder_hidden_states,
                    prefix=(
                        f"{self.cache_prefix}_Bn_residual"
                        if self.cache_manager.is_cache_residual()
                        else f"{self.cache_prefix}_Bn_hidden_states"
                    ),
                    encoder_prefix=(
                        f"{self.cache_prefix}_Bn_residual"
                        if self.cache_manager.is_encoder_cache_residual()
                        else f"{self.cache_prefix}_Bn_hidden_states"
                    ),
                )
            )
            torch._dynamo.graph_break()
            # Call last `n` blocks to further process the hidden states
            # for higher precision.
            hidden_states, encoder_hidden_states = self.call_Bn_blocks(
                hidden_states,
                encoder_hidden_states,
                *args,
                **kwargs,
            )
        else:
            self.cache_manager.set_Fn_buffer(
                Fn_hidden_states_residual,
                prefix=f"{self.cache_prefix}_Fn_residual",
            )
            if self.cache_manager.is_l1_diff_enabled():
                # for hidden states L1 diff
                self.cache_manager.set_Fn_buffer(
                    hidden_states,
                    f"{self.cache_prefix}_Fn_hidden_states",
                )
            del Fn_hidden_states_residual
            torch._dynamo.graph_break()
            (
                hidden_states,
                encoder_hidden_states,
                hidden_states_residual,
                encoder_hidden_states_residual,
            ) = self.call_Mn_blocks(  # middle
                hidden_states,
                encoder_hidden_states,
                *args,
                **kwargs,
            )
            torch._dynamo.graph_break()
            if self.cache_manager.is_cache_residual():
                self.cache_manager.set_Bn_buffer(
                    hidden_states_residual,
                    prefix=f"{self.cache_prefix}_Bn_residual",
                )
            else:
                self.cache_manager.set_Bn_buffer(
                    hidden_states,
                    prefix=f"{self.cache_prefix}_Bn_hidden_states",
                )

            if self.cache_manager.is_encoder_cache_residual():
                self.cache_manager.set_Bn_encoder_buffer(
                    encoder_hidden_states_residual,
                    prefix=f"{self.cache_prefix}_Bn_residual",
                )
            else:
                self.cache_manager.set_Bn_encoder_buffer(
                    encoder_hidden_states,
                    prefix=f"{self.cache_prefix}_Bn_hidden_states",
                )
            torch._dynamo.graph_break()
            # Call last `n` blocks to further process the hidden states
            # for higher precision.
            hidden_states, encoder_hidden_states = self.call_Bn_blocks(
                hidden_states,
                encoder_hidden_states,
                *args,
                **kwargs,
            )

        # patch cached stats for blocks or remove it.
        torch._dynamo.graph_break()

        return (
            hidden_states
            if self.forward_pattern.Return_H_Only
            else (
                (hidden_states, encoder_hidden_states)
                if self.forward_pattern.Return_H_First
                else (encoder_hidden_states, hidden_states)
            )
        )

    @torch.compiler.disable
    def _is_parallelized(self):
        # Compatible with distributed inference.
        return any(
            (
                all(
                    (
                        self.transformer is not None,
                        getattr(self.transformer, "_is_parallelized", False),
                    )
                ),
                (dist.is_initialized() and dist.get_world_size() > 1),
            )
        )

    @torch.compiler.disable
    def _is_in_cache_step(self):
        # Check if the current step is in cache steps.
        # If so, we can skip some Bn blocks and directly
        # use the cached values.
        return (
            self.cache_manager.get_current_step()
            in self.cache_manager.get_cached_steps()
        ) or (
            self.cache_manager.get_current_step()
            in self.cache_manager.get_cfg_cached_steps()
        )

    @torch.compiler.disable
    def _Fn_blocks(self):
        # Select first `n` blocks to process the hidden states for
        # more stable diff calculation.
        # Fn: [0,...,n-1]
        selected_Fn_blocks = self.transformer_blocks[
            : self.cache_manager.Fn_compute_blocks()
        ]
        return selected_Fn_blocks

    @torch.compiler.disable
    def _Mn_blocks(self):  # middle blocks
        # M(N-2n): only transformer_blocks [n,...,N-n], middle
        if self.cache_manager.Bn_compute_blocks() == 0:  # WARN: x[:-0] = []
            selected_Mn_blocks = self.transformer_blocks[
                self.cache_manager.Fn_compute_blocks() :
            ]
        else:
            selected_Mn_blocks = self.transformer_blocks[
                self.cache_manager.Fn_compute_blocks() : -self.cache_manager.Bn_compute_blocks()
            ]
        return selected_Mn_blocks

    @torch.compiler.disable
    def _Bn_blocks(self):
        # Bn: transformer_blocks [N-n+1,...,N-1]
        selected_Bn_blocks = self.transformer_blocks[
            -self.cache_manager.Bn_compute_blocks() :
        ]
        return selected_Bn_blocks

    def call_Fn_blocks(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        for block in self._Fn_blocks():
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                *args,
                **kwargs,
            )
            if not isinstance(hidden_states, torch.Tensor):
                hidden_states, encoder_hidden_states = hidden_states
                if not self.forward_pattern.Return_H_First:
                    hidden_states, encoder_hidden_states = (
                        encoder_hidden_states,
                        hidden_states,
                    )

        return hidden_states, encoder_hidden_states

    def call_Mn_blocks(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        original_hidden_states = hidden_states
        original_encoder_hidden_states = encoder_hidden_states
        for block in self._Mn_blocks():
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                *args,
                **kwargs,
            )
            if not isinstance(hidden_states, torch.Tensor):
                hidden_states, encoder_hidden_states = hidden_states
                if not self.forward_pattern.Return_H_First:
                    hidden_states, encoder_hidden_states = (
                        encoder_hidden_states,
                        hidden_states,
                    )

        # compute hidden_states residual
        hidden_states = hidden_states.contiguous()

        hidden_states_residual = hidden_states - original_hidden_states

        if (
            encoder_hidden_states is not None
            and original_encoder_hidden_states is not None
        ):
            encoder_hidden_states = encoder_hidden_states.contiguous()
            encoder_hidden_states_residual = (
                encoder_hidden_states - original_encoder_hidden_states
            )
        else:
            encoder_hidden_states_residual = None

        return (
            hidden_states,
            encoder_hidden_states,
            hidden_states_residual,
            encoder_hidden_states_residual,
        )

    def call_Bn_blocks(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        if self.cache_manager.Bn_compute_blocks() == 0:
            return hidden_states, encoder_hidden_states

        for block in self._Bn_blocks():
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                *args,
                **kwargs,
            )
            if not isinstance(hidden_states, torch.Tensor):
                hidden_states, encoder_hidden_states = hidden_states
                if not self.forward_pattern.Return_H_First:
                    hidden_states, encoder_hidden_states = (
                        encoder_hidden_states,
                        hidden_states,
                    )

        return hidden_states, encoder_hidden_states
