import inspect
import torch

from cache_dit.cache_factory.dual_block_cache import cache_context
from cache_dit.logger import init_logger

logger = init_logger(__name__)


class DBCachedTransformerBlocks(torch.nn.Module):
    def __init__(
        self,
        transformer_blocks: torch.nn.ModuleList,
        *,
        transformer: torch.nn.Module = None,
        return_hidden_states_first: bool = True,
        return_hidden_states_only: bool = False,
    ):
        super().__init__()

        self.transformer = transformer
        self.transformer_blocks = transformer_blocks
        self.return_hidden_states_first = return_hidden_states_first
        self.return_hidden_states_only = return_hidden_states_only
        self._check_forward_params()

    def _check_forward_params(self):
        self.required_parameters = [
            "hidden_states",
            "encoder_hidden_states",
        ]
        if self.transformer_blocks is not None:
            for block in self.transformer_blocks:
                forward_parameters = set(
                    inspect.signature(block.forward).parameters.keys()
                )
                for required_param in self.required_parameters:
                    assert (
                        required_param in forward_parameters
                    ), f"The input parameters must contains: {required_param}."

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

        cache_context.mark_step_begin()
        # Residual L1 diff or Hidden States L1 diff
        can_use_cache = cache_context.get_can_use_cache(
            (
                Fn_hidden_states_residual
                if not cache_context.is_l1_diff_enabled()
                else hidden_states
            ),
            parallelized=self._is_parallelized(),
            prefix=(
                "Fn_residual"
                if not cache_context.is_l1_diff_enabled()
                else "Fn_hidden_states"
            ),
        )

        torch._dynamo.graph_break()
        if can_use_cache:
            cache_context.add_cached_step()
            del Fn_hidden_states_residual
            hidden_states, encoder_hidden_states = (
                cache_context.apply_hidden_states_residual(
                    hidden_states,
                    encoder_hidden_states,
                    prefix=(
                        "Bn_residual"
                        if cache_context.is_cache_residual()
                        else "Bn_hidden_states"
                    ),
                    encoder_prefix=(
                        "Bn_residual"
                        if cache_context.is_encoder_cache_residual()
                        else "Bn_hidden_states"
                    ),
                )
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
            cache_context.set_Fn_buffer(
                Fn_hidden_states_residual, prefix="Fn_residual"
            )
            if cache_context.is_l1_diff_enabled():
                # for hidden states L1 diff
                cache_context.set_Fn_buffer(hidden_states, "Fn_hidden_states")
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
            if cache_context.is_cache_residual():
                cache_context.set_Bn_buffer(
                    hidden_states_residual,
                    prefix="Bn_residual",
                )
            else:
                # TaylorSeer
                cache_context.set_Bn_buffer(
                    hidden_states,
                    prefix="Bn_hidden_states",
                )
            if cache_context.is_encoder_cache_residual():
                cache_context.set_Bn_encoder_buffer(
                    encoder_hidden_states_residual,
                    prefix="Bn_residual",
                )
            else:
                # TaylorSeer
                cache_context.set_Bn_encoder_buffer(
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
        return (
            cache_context.get_current_step() in cache_context.get_cached_steps()
        ) or (
            cache_context.get_current_step()
            in cache_context.get_cfg_cached_steps()
        )

    @torch.compiler.disable
    def _Fn_transformer_blocks(self):
        # Select first `n` blocks to process the hidden states for
        # more stable diff calculation.
        # Fn: [0,...,n-1]
        selected_Fn_transformer_blocks = self.transformer_blocks[
            : cache_context.Fn_compute_blocks()
        ]
        return selected_Fn_transformer_blocks

    @torch.compiler.disable
    def _Mn_transformer_blocks(self):  # middle blocks
        # M(N-2n): only transformer_blocks [n,...,N-n], middle
        if cache_context.Bn_compute_blocks() == 0:  # WARN: x[:-0] = []
            selected_Mn_transformer_blocks = self.transformer_blocks[
                cache_context.Fn_compute_blocks() :
            ]
        else:
            selected_Mn_transformer_blocks = self.transformer_blocks[
                cache_context.Fn_compute_blocks() : -cache_context.Bn_compute_blocks()
            ]
        return selected_Mn_transformer_blocks

    @torch.compiler.disable
    def _Bn_transformer_blocks(self):
        # Bn: transformer_blocks [N-n+1,...,N-1]
        selected_Bn_transformer_blocks = self.transformer_blocks[
            -cache_context.Bn_compute_blocks() :
        ]
        return selected_Bn_transformer_blocks

    def call_Fn_transformer_blocks(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        assert cache_context.Fn_compute_blocks() <= len(
            self.transformer_blocks
        ), (
            f"Fn_compute_blocks {cache_context.Fn_compute_blocks()} must be less than "
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

        # compute hidden_states residual
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()

        hidden_states_residual = hidden_states - original_hidden_states
        encoder_hidden_states_residual = (
            encoder_hidden_states - original_encoder_hidden_states
        )

        return (
            hidden_states,
            encoder_hidden_states,
            hidden_states_residual,
            encoder_hidden_states_residual,
        )

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
            if block_id not in cache_context.Bn_compute_blocks_ids():
                Bn_i_hidden_states_residual = (
                    hidden_states - Bn_i_original_hidden_states
                )
                Bn_i_encoder_hidden_states_residual = (
                    encoder_hidden_states - Bn_i_original_encoder_hidden_states
                )

                # Save original_hidden_states for diff calculation.
                cache_context.set_Bn_buffer(
                    Bn_i_original_hidden_states,
                    prefix=f"Bn_{block_id}_original",
                )
                cache_context.set_Bn_encoder_buffer(
                    Bn_i_original_encoder_hidden_states,
                    prefix=f"Bn_{block_id}_original",
                )

                cache_context.set_Bn_buffer(
                    Bn_i_hidden_states_residual,
                    prefix=f"Bn_{block_id}_residual",
                )
                cache_context.set_Bn_encoder_buffer(
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
            if block_id in cache_context.Bn_compute_blocks_ids():
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
                if cache_context.get_can_use_cache(
                    hidden_states,  # curr step
                    parallelized=self._is_parallelized(),
                    threshold=cache_context.non_compute_blocks_diff_threshold(),
                    prefix=f"Bn_{block_id}_original",  # prev step
                ):
                    hidden_states, encoder_hidden_states = (
                        cache_context.apply_hidden_states_residual(
                            hidden_states,
                            encoder_hidden_states,
                            prefix=(
                                f"Bn_{block_id}_residual"
                                if cache_context.is_cache_residual()
                                else f"Bn_{block_id}_original"
                            ),
                            encoder_prefix=(
                                f"Bn_{block_id}_residual"
                                if cache_context.is_encoder_cache_residual()
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
        if cache_context.Bn_compute_blocks() == 0:
            return hidden_states, encoder_hidden_states

        assert cache_context.Bn_compute_blocks() <= len(
            self.transformer_blocks
        ), (
            f"Bn_compute_blocks {cache_context.Bn_compute_blocks()} must be less than "
            f"the number of transformer blocks {len(self.transformer_blocks)}"
        )
        if len(cache_context.Bn_compute_blocks_ids()) > 0:
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
    transformer._cached_steps = cache_context.get_cached_steps()
    transformer._residual_diffs = cache_context.get_residual_diffs()
    transformer._cfg_cached_steps = cache_context.get_cfg_cached_steps()
    transformer._cfg_residual_diffs = cache_context.get_cfg_residual_diffs()
