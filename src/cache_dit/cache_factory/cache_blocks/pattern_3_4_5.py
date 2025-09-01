import torch

from cache_dit.cache_factory import CachedContext
from cache_dit.cache_factory import ForwardPattern
from cache_dit.cache_factory.cache_blocks.utils import (
    patch_cached_stats,
)
from cache_dit.cache_factory.cache_blocks.pattern_base import (
    CachedBlocks_Pattern_Base,
)
from cache_dit.logger import init_logger

logger = init_logger(__name__)


class CachedBlocks_Pattern_3_4_5(CachedBlocks_Pattern_Base):
    _supported_patterns = [
        ForwardPattern.Pattern_3,
        ForwardPattern.Pattern_4,
        ForwardPattern.Pattern_5,
    ]

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        original_hidden_states = hidden_states
        # Call first `n` blocks to process the hidden states for
        # more stable diff calculation.
        # encoder_hidden_states: None Pattern 3, else 4, 5
        hidden_states, encoder_hidden_states = self.call_Fn_blocks(
            hidden_states,
            *args,
            **kwargs,
        )

        Fn_hidden_states_residual = hidden_states - original_hidden_states
        del original_hidden_states

        CachedContext.mark_step_begin()
        # Residual L1 diff or Hidden States L1 diff
        can_use_cache = CachedContext.get_can_use_cache(
            (
                Fn_hidden_states_residual
                if not CachedContext.is_l1_diff_enabled()
                else hidden_states
            ),
            parallelized=self._is_parallelized(),
            prefix=(
                "Fn_residual"
                if not CachedContext.is_l1_diff_enabled()
                else "Fn_hidden_states"
            ),
        )

        torch._dynamo.graph_break()
        if can_use_cache:
            CachedContext.add_cached_step()
            del Fn_hidden_states_residual
            hidden_states, encoder_hidden_states = (
                CachedContext.apply_hidden_states_residual(
                    hidden_states,
                    # None Pattern 3, else 4, 5
                    encoder_hidden_states,
                    prefix=(
                        "Bn_residual"
                        if CachedContext.is_cache_residual()
                        else "Bn_hidden_states"
                    ),
                    encoder_prefix=(
                        "Bn_residual"
                        if CachedContext.is_encoder_cache_residual()
                        else "Bn_hidden_states"
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
            CachedContext.set_Fn_buffer(
                Fn_hidden_states_residual, prefix="Fn_residual"
            )
            if CachedContext.is_l1_diff_enabled():
                # for hidden states L1 diff
                CachedContext.set_Fn_buffer(hidden_states, "Fn_hidden_states")
            del Fn_hidden_states_residual
            torch._dynamo.graph_break()
            (
                hidden_states,
                encoder_hidden_states,
                hidden_states_residual,
                # None Pattern 3, else 4, 5
                encoder_hidden_states_residual,
            ) = self.call_Mn_blocks(  # middle
                hidden_states,
                # None Pattern 3, else 4, 5
                encoder_hidden_states,
                *args,
                **kwargs,
            )
            torch._dynamo.graph_break()
            if CachedContext.is_cache_residual():
                CachedContext.set_Bn_buffer(
                    hidden_states_residual,
                    prefix="Bn_residual",
                )
            else:
                # TaylorSeer
                CachedContext.set_Bn_buffer(
                    hidden_states,
                    prefix="Bn_hidden_states",
                )
            if CachedContext.is_encoder_cache_residual():
                CachedContext.set_Bn_encoder_buffer(
                    # None Pattern 3, else 4, 5
                    encoder_hidden_states_residual,
                    prefix="Bn_residual",
                )
            else:
                # TaylorSeer
                CachedContext.set_Bn_encoder_buffer(
                    # None Pattern 3, else 4, 5
                    encoder_hidden_states,
                    prefix="Bn_hidden_states",
                )
            torch._dynamo.graph_break()
            # Call last `n` blocks to further process the hidden states
            # for higher precision.
            hidden_states, encoder_hidden_states = self.call_Bn_blocks(
                hidden_states,
                # None Pattern 3, else 4, 5
                encoder_hidden_states,
                *args,
                **kwargs,
            )

        patch_cached_stats(self.transformer)
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

    def call_Fn_blocks(
        self,
        hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        assert CachedContext.Fn_compute_blocks() <= len(
            self.transformer_blocks
        ), (
            f"Fn_compute_blocks {CachedContext.Fn_compute_blocks()} must be less than "
            f"the number of transformer blocks {len(self.transformer_blocks)}"
        )
        encoder_hidden_states = None  # Pattern 3
        for block in self._Fn_blocks():
            hidden_states = block(
                hidden_states,
                *args,
                **kwargs,
            )
            if not isinstance(hidden_states, torch.Tensor):  # Pattern 4, 5
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
        # None Pattern 3, else 4, 5
        encoder_hidden_states: torch.Tensor | None,
        *args,
        **kwargs,
    ):
        original_hidden_states = hidden_states
        original_encoder_hidden_states = encoder_hidden_states
        for block in self._Mn_blocks():
            hidden_states = block(
                hidden_states,
                *args,
                **kwargs,
            )
            if not isinstance(hidden_states, torch.Tensor):  # Pattern 4, 5
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
            original_encoder_hidden_states is not None
            and encoder_hidden_states is not None
        ):  # Pattern 4, 5
            encoder_hidden_states_residual = (
                encoder_hidden_states - original_encoder_hidden_states
            )
        else:
            encoder_hidden_states_residual = None  # Pattern 3

        return (
            hidden_states,
            encoder_hidden_states,
            hidden_states_residual,
            encoder_hidden_states_residual,
        )

    def call_Bn_blocks(
        self,
        hidden_states: torch.Tensor,
        # None Pattern 3, else 4, 5
        encoder_hidden_states: torch.Tensor | None,
        *args,
        **kwargs,
    ):
        if CachedContext.Bn_compute_blocks() == 0:
            return hidden_states, encoder_hidden_states

        assert CachedContext.Bn_compute_blocks() <= len(
            self.transformer_blocks
        ), (
            f"Bn_compute_blocks {CachedContext.Bn_compute_blocks()} must be less than "
            f"the number of transformer blocks {len(self.transformer_blocks)}"
        )
        if len(CachedContext.Bn_compute_blocks_ids()) > 0:
            raise ValueError(
                f"Bn_compute_blocks_ids is not support for "
                f"patterns: {self._supported_patterns}."
            )
        else:
            # Compute all Bn blocks if no specific Bn compute blocks ids are set.
            for block in self._Bn_blocks():
                hidden_states = block(
                    hidden_states,
                    *args,
                    **kwargs,
                )
                if not isinstance(hidden_states, torch.Tensor):  # Pattern 4,5
                    hidden_states, encoder_hidden_states = hidden_states
                    if not self.forward_pattern.Return_H_First:
                        hidden_states, encoder_hidden_states = (
                            encoder_hidden_states,
                            hidden_states,
                        )

        return hidden_states, encoder_hidden_states
