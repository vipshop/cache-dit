import inspect
import torch

from cache_dit.cache_factory.dynamic_block_prune import prune_context
from cache_dit.logger import init_logger

logger = init_logger(__name__)


class DBPrunedTransformerBlocks(torch.nn.Module):
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
        self.pruned_blocks_step: int = 0
        self._check_forward_params()

    def _check_forward_params(self):
        # NOTE: DBPrune only support blocks which have the pattern:
        # IN/OUT: (hidden_states, encoder_hidden_states)
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
        prune_context.mark_step_begin()
        self.pruned_blocks_step = 0
        original_hidden_states = hidden_states

        torch._dynamo.graph_break()
        hidden_states, encoder_hidden_states = self.call_blocks(
            hidden_states,
            encoder_hidden_states,
            *args,
            **kwargs,
        )

        del original_hidden_states
        torch._dynamo.graph_break()

        prune_context.add_pruned_block(self.pruned_blocks_step)
        prune_context.add_actual_block(self.num_transformer_blocks)
        patch_pruned_stats(self.transformer)

        return (
            hidden_states
            if self.return_hidden_states_only
            else (
                (hidden_states, encoder_hidden_states)
                if self.return_hidden_states_first
                else (encoder_hidden_states, hidden_states)
            )
        )

    @property
    @torch.compiler.disable
    def num_transformer_blocks(self):
        # Total number of transformer blocks.
        return len(self.transformer_blocks)

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
    def _non_prune_blocks_ids(self):
        # Never prune the first `Fn` and last `Bn` blocks.
        num_blocks = self.num_transformer_blocks
        Fn_compute_blocks_ = (
            prune_context.Fn_compute_blocks()
            if prune_context.Fn_compute_blocks() < num_blocks
            else num_blocks
        )
        Fn_compute_blocks_ids = list(range(Fn_compute_blocks_))
        Bn_compute_blocks_ = (
            prune_context.Bn_compute_blocks()
            if prune_context.Bn_compute_blocks() < num_blocks
            else num_blocks
        )
        Bn_compute_blocks_ids = list(
            range(
                num_blocks - Bn_compute_blocks_,
                num_blocks,
            )
        )
        non_prune_blocks_ids = list(
            set(
                Fn_compute_blocks_ids
                + Bn_compute_blocks_ids
                + prune_context.get_non_prune_blocks_ids()
            )
        )
        non_prune_blocks_ids = [
            d for d in non_prune_blocks_ids if d < num_blocks
        ]
        return sorted(non_prune_blocks_ids)

    @torch.compiler.disable
    def _should_update_residuals(self):
        # Wrap for non compiled mode.
        # Check if the current step is a multiple of
        # the residual cache update interval.
        return (
            prune_context.get_current_step()
            % prune_context.residual_cache_update_interval()
            == 0
        )

    @torch.compiler.disable
    def _get_can_use_prune(
        self,
        block_id: int,  # Block index in the transformer blocks
        hidden_states: torch.Tensor,  # hidden_states or residual
        name: str = "Bn_original",  # prev step name for single blocks
    ):
        # Wrap for non compiled mode.
        can_use_prune = False
        if block_id not in self._non_prune_blocks_ids():
            can_use_prune = prune_context.get_can_use_prune(
                hidden_states,  # curr step
                parallelized=self._is_parallelized(),
                name=name,  # prev step
            )
        self.pruned_blocks_step += int(can_use_prune)
        return can_use_prune

    def _compute_or_prune_block(
        self,
        block_id: int,  # Block index in the transformer blocks
        # Below are the inputs to the block
        block,  # The transformer block to be executed
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        # Helper function for `call_blocks`
        original_hidden_states = hidden_states
        original_encoder_hidden_states = encoder_hidden_states

        # block_id: global block index in the transformer blocks +
        # single_transformer_blocks
        can_use_prune = self._get_can_use_prune(
            block_id,
            hidden_states,  # hidden_states or residual
            name=f"{block_id}_original",  # prev step
        )
        # Prune steps: Prune current block and reuse the cached
        # residuals for hidden states approximate.
        if can_use_prune:
            hidden_states, encoder_hidden_states = (
                prune_context.apply_hidden_states_residual(
                    hidden_states,
                    encoder_hidden_states,
                    name=f"{block_id}_residual",
                    encoder_name=f"{block_id}_encoder_residual",
                )
            )
        else:
            # Normal steps: Compute the block and cache the residuals.
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

            # Save original_hidden_states for diff calculation.
            # May not be necessary to update the hidden
            # states and residuals each step?
            if self._should_update_residuals():
                # Cache residuals for the non-compute Bn blocks for
                # subsequent prune steps.
                hidden_states_residual = hidden_states - original_hidden_states
                encoder_hidden_states_residual = (
                    encoder_hidden_states - original_encoder_hidden_states
                )
                prune_context.set_buffer(
                    f"{block_id}_original",
                    original_hidden_states,
                )

                prune_context.set_buffer(
                    f"{block_id}_residual",
                    hidden_states_residual,
                )
                prune_context.set_buffer(
                    f"{block_id}_encoder_residual",
                    encoder_hidden_states_residual,
                )
                del hidden_states_residual
                del encoder_hidden_states_residual

        del original_hidden_states
        del original_encoder_hidden_states

        return hidden_states, encoder_hidden_states

    def call_blocks(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        for i, block in enumerate(self.transformer_blocks):
            hidden_states, encoder_hidden_states = self._compute_or_prune_block(
                i,
                block,
                hidden_states,
                encoder_hidden_states,
                *args,
                **kwargs,
            )

        return hidden_states, encoder_hidden_states


@torch.compiler.disable
def patch_pruned_stats(
    transformer,
):
    # Patch the pruned stats to the transformer, the pruned stats
    # will be reset for each calling of pipe.__call__(**kwargs).
    if transformer is None:
        return

    # TODO: Patch more pruned stats to the transformer
    transformer._pruned_blocks = prune_context.get_pruned_blocks()
    transformer._pruned_steps = prune_context.get_pruned_steps()
    transformer._residual_diffs = prune_context.get_residual_diffs()
    transformer._actual_blocks = prune_context.get_actual_blocks()

    transformer._cfg_pruned_blocks = prune_context.get_cfg_pruned_blocks()
    transformer._cfg_pruned_steps = prune_context.get_cfg_pruned_steps()
    transformer._cfg_residual_diffs = prune_context.get_cfg_residual_diffs()
    transformer._cfg_actual_blocks = prune_context.get_cfg_actual_blocks()
