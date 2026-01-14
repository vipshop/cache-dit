import torch

from ..forward_pattern import ForwardPattern
from ..cache_contexts.cache_manager import (
    ContextNotExistError,
)
from .pattern_base import (
    CachedBlocks_Pattern_Base,
)
from ..cache_contexts.prune_context import PrunedContext
from ..cache_contexts.prune_manager import (
    PrunedContextManager,
)
from ..cache_types import CacheType

from cache_dit.logger import init_logger

logger = init_logger(__name__)


class CachedBlocks_Pattern_3_4_5(CachedBlocks_Pattern_Base):
    _supported_patterns = [
        ForwardPattern.Pattern_3,
        ForwardPattern.Pattern_4,
        ForwardPattern.Pattern_5,
    ]

    def call_blocks(
        self,
        hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        # Call all blocks to process the hidden states without cache.
        new_encoder_hidden_states = None
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                *args,
                **kwargs,
            )
            hidden_states, new_encoder_hidden_states = self._process_block_outputs(hidden_states)

        return hidden_states, new_encoder_hidden_states

    @torch.compiler.disable
    def _process_block_outputs(
        self, hidden_states: torch.Tensor | tuple
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Process the outputs for the block.
        new_encoder_hidden_states = None
        if not isinstance(hidden_states, torch.Tensor):  # Pattern 4, 5
            if len(hidden_states) == 2:
                if isinstance(hidden_states[1], torch.Tensor):
                    hidden_states, new_encoder_hidden_states = hidden_states
                    if not self.forward_pattern.Return_H_First:
                        hidden_states, new_encoder_hidden_states = (
                            new_encoder_hidden_states,
                            hidden_states,
                        )
                elif isinstance(hidden_states[0], torch.Tensor):
                    hidden_states = hidden_states[0]
                else:
                    raise ValueError("Unexpected hidden_states format.")
            else:
                assert len(hidden_states) == 1, f"Unexpected output length: {len(hidden_states)}"
                hidden_states = hidden_states[0]
        return hidden_states, new_encoder_hidden_states

    @torch.compiler.disable
    def _process_forward_outputs(
        self,
        hidden_states: torch.Tensor,
        new_encoder_hidden_states: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, None]:
        if self.forward_pattern.Return_H_Only:
            return hidden_states
        else:
            if self.forward_pattern.Return_H_First:
                return (hidden_states, new_encoder_hidden_states)
            else:
                return (new_encoder_hidden_states, hidden_states)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        # Use it's own cache context.
        try:
            self.context_manager.set_context(self.cache_context)
            self._check_cache_params()
        except ContextNotExistError as e:
            logger.warning(f"context not exist: {e}, skip cache.")
            hidden_states, new_encoder_hidden_states = self.call_blocks(
                hidden_states,
                *args,
                **kwargs,
            )
            return self._process_forward_outputs(hidden_states, new_encoder_hidden_states)

        original_hidden_states = hidden_states
        # Call first `n` blocks to process the hidden states for
        # more stable diff calculation.
        hidden_states, new_encoder_hidden_states = self.call_Fn_blocks(
            hidden_states,
            *args,
            **kwargs,
        )

        Fn_hidden_states_residual = self._get_Fn_residual(original_hidden_states, hidden_states)
        del original_hidden_states

        self.context_manager.mark_step_begin()
        # Residual L1 diff or Hidden States L1 diff
        can_use_cache = self.context_manager.can_cache(
            (
                Fn_hidden_states_residual
                if not self.context_manager.is_l1_diff_enabled()
                else hidden_states
            ),
            parallelized=self._is_parallelized(),
            prefix=(
                f"{self.cache_prefix}_Fn_residual"
                if not self.context_manager.is_l1_diff_enabled()
                else f"{self.cache_prefix}_Fn_hidden_states"
            ),
        )

        torch._dynamo.graph_break()
        if can_use_cache:
            self.context_manager.add_cached_step()
            del Fn_hidden_states_residual
            hidden_states, new_encoder_hidden_states = self.context_manager.apply_cache(
                hidden_states,
                new_encoder_hidden_states,  # encoder_hidden_states not use cache
                prefix=(
                    f"{self.cache_prefix}_Bn_residual"
                    if self.context_manager.is_cache_residual()
                    else f"{self.cache_prefix}_Bn_hidden_states"
                ),
                encoder_prefix=(
                    f"{self.cache_prefix}_Bn_residual"
                    if self.context_manager.is_encoder_cache_residual()
                    else f"{self.cache_prefix}_Bn_hidden_states"
                ),
            )
            torch._dynamo.graph_break()
            # Call last `n` blocks to further process the hidden states
            # for higher precision.
            if self.context_manager.Bn_compute_blocks() > 0:
                hidden_states, new_encoder_hidden_states = self.call_Bn_blocks(
                    hidden_states,
                    *args,
                    **kwargs,
                )
        else:
            self.context_manager.set_Fn_buffer(
                Fn_hidden_states_residual,
                prefix=f"{self.cache_prefix}_Fn_residual",
            )
            if self.context_manager.is_l1_diff_enabled():
                # for hidden states L1 diff
                self.context_manager.set_Fn_buffer(
                    hidden_states,
                    f"{self.cache_prefix}_Fn_hidden_states",
                )
            del Fn_hidden_states_residual
            torch._dynamo.graph_break()
            old_encoder_hidden_states = new_encoder_hidden_states
            (
                hidden_states,
                new_encoder_hidden_states,
                hidden_states_residual,
            ) = self.call_Mn_blocks(  # middle
                hidden_states,
                *args,
                **kwargs,
            )

            torch._dynamo.graph_break()
            if self.context_manager.is_cache_residual():
                self.context_manager.set_Bn_buffer(
                    hidden_states_residual,
                    prefix=f"{self.cache_prefix}_Bn_residual",
                )
            else:
                self.context_manager.set_Bn_buffer(
                    hidden_states,
                    prefix=f"{self.cache_prefix}_Bn_hidden_states",
                )

            if new_encoder_hidden_states is not None:
                new_encoder_hidden_states_residual = (
                    new_encoder_hidden_states - old_encoder_hidden_states
                )
            if self.context_manager.is_encoder_cache_residual():
                if new_encoder_hidden_states is not None:
                    self.context_manager.set_Bn_encoder_buffer(
                        new_encoder_hidden_states_residual,
                        prefix=f"{self.cache_prefix}_Bn_residual",
                    )
            else:
                if new_encoder_hidden_states is not None:
                    self.context_manager.set_Bn_encoder_buffer(
                        new_encoder_hidden_states_residual,
                        prefix=f"{self.cache_prefix}_Bn_hidden_states",
                    )
            torch._dynamo.graph_break()
            # Call last `n` blocks to further process the hidden states
            # for higher precision.
            if self.context_manager.Bn_compute_blocks() > 0:
                hidden_states, new_encoder_hidden_states = self.call_Bn_blocks(
                    hidden_states,
                    *args,
                    **kwargs,
                )

        torch._dynamo.graph_break()

        return self._process_forward_outputs(
            hidden_states,
            new_encoder_hidden_states,
        )

    def call_Fn_blocks(
        self,
        hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        new_encoder_hidden_states = None
        for block in self._Fn_blocks():
            hidden_states = block(
                hidden_states,
                *args,
                **kwargs,
            )
            hidden_states, new_encoder_hidden_states = self._process_block_outputs(hidden_states)

        return hidden_states, new_encoder_hidden_states

    def call_Mn_blocks(
        self,
        hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        original_hidden_states = hidden_states
        new_encoder_hidden_states = None
        for block in self._Mn_blocks():
            hidden_states = block(
                hidden_states,
                *args,
                **kwargs,
            )

            hidden_states, new_encoder_hidden_states = self._process_block_outputs(hidden_states)

        # compute hidden_states residual
        hidden_states = hidden_states.contiguous()
        hidden_states_residual = hidden_states - original_hidden_states.to(hidden_states.device)

        return (
            hidden_states,
            new_encoder_hidden_states,
            hidden_states_residual,
        )

    def call_Bn_blocks(
        self,
        hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        new_encoder_hidden_states = None
        if self.context_manager.Bn_compute_blocks() == 0:
            return hidden_states, new_encoder_hidden_states

        for block in self._Bn_blocks():
            hidden_states = block(
                hidden_states,
                *args,
                **kwargs,
            )

            hidden_states, new_encoder_hidden_states = self._process_block_outputs(hidden_states)

        return hidden_states, new_encoder_hidden_states


class PrunedBlocks_Pattern_3_4_5(CachedBlocks_Pattern_3_4_5):
    _supported_patterns = [
        ForwardPattern.Pattern_3,
        ForwardPattern.Pattern_4,
        ForwardPattern.Pattern_5,
    ]
    pruned_blocks_step: int = 0  # number of pruned blocks in current step

    def __init__(
        self,
        # 0. Transformer blocks configuration
        transformer_blocks: torch.nn.ModuleList,
        transformer: torch.nn.Module = None,
        forward_pattern: ForwardPattern = ForwardPattern.Pattern_0,
        check_forward_pattern: bool = True,
        check_num_outputs: bool = True,
        # 1. Prune context configuration
        cache_prefix: str = None,  # maybe un-need.
        cache_context: PrunedContext | str = None,
        context_manager: PrunedContextManager = None,
        cache_type: CacheType = CacheType.DBPrune,
        **kwargs,
    ):
        super().__init__(
            # 0. Transformer blocks configuration
            transformer_blocks,
            transformer=transformer,
            forward_pattern=forward_pattern,
            check_forward_pattern=check_forward_pattern,
            check_num_outputs=check_num_outputs,
            # 1. Cache context configuration
            cache_prefix=cache_prefix,
            cache_context=cache_context,
            context_manager=context_manager,
            cache_type=cache_type,
            **kwargs,
        )
        assert isinstance(
            self.context_manager, PrunedContextManager
        ), "context_manager must be PrunedContextManager for PrunedBlocks."
        self.context_manager: PrunedContextManager = self.context_manager  # For type hint

    @torch.compiler.disable
    def _check_cache_type(self):
        assert (
            self.cache_type == CacheType.DBPrune
        ), f"Cache type {self.cache_type} is not supported for PrunedBlocks."

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        self.pruned_blocks_step: int = 0  # reset for each step

        # Use it's own cache context.
        try:
            self.context_manager.set_context(self.cache_context)
            self._check_cache_params()
        except ContextNotExistError as e:
            logger.warning(f"context not exist: {e}, skip prune.")
            hidden_states, new_encoder_hidden_states = self.call_blocks(
                hidden_states,
                *args,
                **kwargs,
            )
            return self._process_forward_outputs(hidden_states, new_encoder_hidden_states)

        self.context_manager.mark_step_begin()

        if self._check_if_context_parallel_enabled(self.transformer_blocks[0]):
            raise RuntimeError("Block level Context parallelism is not supported in PrunedBlocks.")

        # Call all blocks with prune strategy to process the hidden states.
        new_encoder_hidden_states = None
        for i, block in enumerate(self.transformer_blocks):
            hidden_states, new_encoder_hidden_states = self.compute_or_prune(
                i,
                block,
                hidden_states,
                new_encoder_hidden_states,
                *args,
                **kwargs,
            )

        self.context_manager.add_pruned_block(self.pruned_blocks_step)
        self.context_manager.add_actual_block(self.num_blocks)

        return self._process_forward_outputs(
            hidden_states,
            new_encoder_hidden_states,
        )

    @property
    @torch.compiler.disable
    def num_blocks(self):
        return len(self.transformer_blocks)

    @torch.compiler.disable
    def _skip_prune(self, block_id: int) -> bool:
        # Wrap for non compiled mode.
        return block_id in self.context_manager.get_non_prune_blocks_ids(self.num_blocks)

    @torch.compiler.disable
    def _maybe_prune(
        self,
        block_id: int,  # Block index in the transformer blocks
        hidden_states: torch.Tensor,  # hidden_states or residual
        prefix: str = "Bn_original",  # prev step name for single blocks
    ):
        # Wrap for non compiled mode.
        can_use_prune = False
        if not self._skip_prune(block_id):
            can_use_prune = self.context_manager.can_prune(
                hidden_states,  # curr step
                parallelized=self._is_parallelized(),
                prefix=prefix,  # prev step
            )
        self.pruned_blocks_step += int(can_use_prune)
        return can_use_prune

    def compute_or_prune(
        self,
        block_id: int,  # Block index in the transformer blocks
        # Below are the inputs to the block
        block,  # The transformer block to be executed
        hidden_states: torch.Tensor,
        new_encoder_hidden_states: torch.Tensor | None,
        *args,
        **kwargs,
    ):
        original_hidden_states = hidden_states
        original_encoder_hidden_states = new_encoder_hidden_states

        can_use_prune = self._maybe_prune(
            block_id,
            hidden_states,
            prefix=f"{self.cache_prefix}_{block_id}_Fn_original",
        )

        # Prune steps: Prune current block and reuse the cached
        # residuals for hidden states approximate.
        torch._dynamo.graph_break()
        if can_use_prune:
            self.context_manager.add_pruned_step()
            hidden_states, new_encoder_hidden_states = self.context_manager.apply_prune(
                hidden_states,
                new_encoder_hidden_states,
                prefix=(
                    f"{self.cache_prefix}_{block_id}_Bn_residual"
                    if self.context_manager.is_cache_residual()
                    else f"{self.cache_prefix}_{block_id}_Bn_hidden_states"
                ),
                encoder_prefix=(
                    f"{self.cache_prefix}_{block_id}_Bn_encoder_residual"
                    if self.context_manager.is_encoder_cache_residual()
                    else f"{self.cache_prefix}_{block_id}_Bn_encoder_hidden_states"
                ),
            )
            torch._dynamo.graph_break()
        else:
            # Normal steps: Compute the block and cache the residuals.
            hidden_states = block(
                hidden_states,
                *args,
                **kwargs,
            )
            hidden_states, new_encoder_hidden_states = self._process_block_outputs(
                hidden_states, new_encoder_hidden_states
            )
            if not self._skip_prune(block_id):
                hidden_states = hidden_states.contiguous()
                hidden_states_residual = hidden_states - original_hidden_states

                if (
                    new_encoder_hidden_states is not None
                    and original_encoder_hidden_states is not None
                ):
                    new_encoder_hidden_states = new_encoder_hidden_states.contiguous()
                    new_encoder_hidden_states_residual = (
                        new_encoder_hidden_states - original_encoder_hidden_states
                    )
                else:
                    new_encoder_hidden_states_residual = None

                self.context_manager.set_Fn_buffer(
                    original_hidden_states,
                    prefix=f"{self.cache_prefix}_{block_id}_Fn_original",
                )
                if self.context_manager.is_cache_residual():
                    self.context_manager.set_Bn_buffer(
                        hidden_states_residual,
                        prefix=f"{self.cache_prefix}_{block_id}_Bn_residual",
                    )
                else:
                    self.context_manager.set_Bn_buffer(
                        hidden_states,
                        prefix=f"{self.cache_prefix}_{block_id}_Bn_hidden_states",
                    )
                if new_encoder_hidden_states_residual is not None:
                    if self.context_manager.is_encoder_cache_residual():
                        self.context_manager.set_Bn_encoder_buffer(
                            new_encoder_hidden_states_residual,
                            prefix=f"{self.cache_prefix}_{block_id}_Bn_encoder_residual",
                        )
                    else:
                        self.context_manager.set_Bn_encoder_buffer(
                            new_encoder_hidden_states_residual,
                            prefix=f"{self.cache_prefix}_{block_id}_Bn_encoder_hidden_states",
                        )
            torch._dynamo.graph_break()

        return hidden_states, new_encoder_hidden_states
