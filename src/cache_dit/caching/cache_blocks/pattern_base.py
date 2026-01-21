import inspect
import logging
import torch
import torch.distributed as dist
from diffusers.hooks import HookRegistry
from ..cache_contexts.cache_context import CachedContext
from ..cache_contexts.prune_context import PrunedContext
from ..cache_contexts.cache_manager import (
    CachedContextManager,
    ContextNotExistError,
)
from ..cache_contexts.prune_manager import (
    PrunedContextManager,
)
from ..forward_pattern import ForwardPattern
from ..cache_types import CacheType
from cache_dit.logger import init_logger

logger = init_logger(__name__)

try:
    from diffusers.hooks.context_parallel import ContextParallelSplitHook
except ImportError:
    ContextParallelSplitHook = None
    logger.debug(
        "Context parallelism in cache-dit requires 'diffusers>=0.36.dev0.\n"
        "Please install latest version of diffusers from source via: \n"
        "pip3 install git+https://github.com/huggingface/diffusers.git"
    )


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
        context_manager: CachedContextManager = None,
        cache_type: CacheType = CacheType.DBCache,
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
        self.context_manager = context_manager
        self.cache_type = cache_type

        self._check_forward_pattern()
        self._check_cache_type()
        logger.info(
            f"Match Blocks: {self.__class__.__name__}, for "
            f"{self.cache_prefix}, cache_context: {self.cache_context}, "
            f"context_manager: {self.context_manager.name}."
        )

    def _check_forward_pattern(self):
        if not self.check_forward_pattern:
            logger.warning(f"Skipped Forward Pattern Check: {self.forward_pattern}")
            return

        assert (
            self.forward_pattern.Supported and self.forward_pattern in self._supported_patterns
        ), f"Pattern {self.forward_pattern} is not supported now!"

        if self.transformer_blocks is not None:
            for block in self.transformer_blocks:
                # Special case for HiDreamBlock
                if hasattr(block, "block"):
                    if isinstance(block.block, torch.nn.Module):
                        block = block.block

                forward_parameters = set(inspect.signature(block.forward).parameters.keys())

                if self.check_num_outputs:
                    num_outputs = str(inspect.signature(block.forward).return_annotation).count(
                        "torch.Tensor"
                    )

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
    def _check_cache_type(self):
        assert (
            self.cache_type == CacheType.DBCache
        ), f"Cache type {self.cache_type} is not supported for CachedBlocks."

    @torch.compiler.disable
    def _check_cache_params(self):
        self._check_cache_type()
        assert self.context_manager.Fn_compute_blocks() <= len(self.transformer_blocks), (
            f"Fn_compute_blocks {self.context_manager.Fn_compute_blocks()} must be less than "
            f"the number of transformer blocks {len(self.transformer_blocks)}"
        )
        assert self.context_manager.Bn_compute_blocks() <= len(self.transformer_blocks), (
            f"Bn_compute_blocks {self.context_manager.Bn_compute_blocks()} must be less than "
            f"the number of transformer blocks {len(self.transformer_blocks)}"
        )

    def call_blocks(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        # Call all blocks to process the hidden states without cache.
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                *args,
                **kwargs,
            )
            hidden_states, encoder_hidden_states = self._process_block_outputs(
                hidden_states, encoder_hidden_states
            )
        return hidden_states, encoder_hidden_states

    @torch.compiler.disable
    def _process_block_outputs(
        self,
        hidden_states: torch.Tensor | tuple,
        encoder_hidden_states: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not isinstance(hidden_states, torch.Tensor):
            hidden_states, encoder_hidden_states = hidden_states
            if not self.forward_pattern.Return_H_First:
                hidden_states, encoder_hidden_states = (
                    encoder_hidden_states,
                    hidden_states,
                )
        return hidden_states, encoder_hidden_states

    @torch.compiler.disable
    def _process_forward_outputs(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | torch.Tensor:
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
    def _check_if_context_parallel_enabled(
        self,
        module: torch.nn.Module,
    ) -> bool:
        if ContextParallelSplitHook is None:
            return False
        if hasattr(module, "_diffusers_hook"):
            _diffusers_hook: HookRegistry = module._diffusers_hook
            for hook in _diffusers_hook.hooks.values():
                if isinstance(hook, ContextParallelSplitHook):
                    return True
        return False

    def _get_Fn_residual(
        self,
        original_hidden_states: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # NOTE: Make cases compatible with context parallelism while using
        # block level cp plan, e.g., WanTransformer3DModel. The shape of
        # `original_hidden_states` and `hidden_states` after Fn maybe
        # different due to seqlen split in context parallelism.
        if self._check_if_context_parallel_enabled(self.transformer_blocks[0]) and (
            original_hidden_states.shape != hidden_states.shape
        ):
            # Force use `hidden_states` as the Fn states residual for subsequent
            # dynamic cache processing if the shape is different.
            Fn_hidden_states_residual = hidden_states
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Context parallelism is enabled in Fn blocks, and the shape of "
                    f"original_hidden_states {original_hidden_states.shape} and "
                    f"hidden_states {hidden_states.shape} are different after Fn blocks. "
                    f"Use hidden_states as Fn_hidden_states_residual directly."
                )
        else:
            Fn_hidden_states_residual = hidden_states - original_hidden_states.to(
                hidden_states.device
            )
        return Fn_hidden_states_residual

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        # Use it's own cache context.
        try:
            self.context_manager.set_context(self.cache_context)
            self._check_cache_params()
        except ContextNotExistError as e:
            logger.warning(f"Cache context not exist: {e}, skip cache.")
            # Call all blocks to process the hidden states.
            hidden_states, encoder_hidden_states = self.call_blocks(
                hidden_states,
                encoder_hidden_states,
                *args,
                **kwargs,
            )
            return self._process_forward_outputs(
                hidden_states,
                encoder_hidden_states,
            )

        original_hidden_states = hidden_states
        # Call first `n` blocks to process the hidden states for
        # more stable diff calculation.
        hidden_states, encoder_hidden_states = self.call_Fn_blocks(
            hidden_states,
            encoder_hidden_states,
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
            hidden_states, encoder_hidden_states = self.context_manager.apply_cache(
                hidden_states,
                encoder_hidden_states,
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
            hidden_states, encoder_hidden_states = self.call_Bn_blocks(
                hidden_states,
                encoder_hidden_states,
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

            if self.context_manager.is_encoder_cache_residual():
                self.context_manager.set_Bn_encoder_buffer(
                    encoder_hidden_states_residual,
                    prefix=f"{self.cache_prefix}_Bn_residual",
                )
            else:
                self.context_manager.set_Bn_encoder_buffer(
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

        return self._process_forward_outputs(
            hidden_states,
            encoder_hidden_states,
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
            self.context_manager.get_current_step() in self.context_manager.get_cached_steps()
        ) or (
            self.context_manager.get_current_step() in self.context_manager.get_cfg_cached_steps()
        )

    @torch.compiler.disable
    def _Fn_blocks(self):
        # Select first `n` blocks to process the hidden states for
        # more stable diff calculation.
        # Fn: [0,...,n-1]
        selected_Fn_blocks = self.transformer_blocks[: self.context_manager.Fn_compute_blocks()]
        return selected_Fn_blocks

    @torch.compiler.disable
    def _Mn_blocks(self):  # middle blocks
        # M(N-2n): only transformer_blocks [n,...,N-n], middle
        if self.context_manager.Bn_compute_blocks() == 0:  # WARN: x[:-0] = []
            selected_Mn_blocks = self.transformer_blocks[self.context_manager.Fn_compute_blocks() :]
        else:
            selected_Mn_blocks = self.transformer_blocks[
                self.context_manager.Fn_compute_blocks() : -self.context_manager.Bn_compute_blocks()
            ]
        return selected_Mn_blocks

    @torch.compiler.disable
    def _Bn_blocks(self):
        # Bn: transformer_blocks [N-n+1,...,N-1]
        selected_Bn_blocks = self.transformer_blocks[-self.context_manager.Bn_compute_blocks() :]
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
            hidden_states, encoder_hidden_states = self._process_block_outputs(
                hidden_states, encoder_hidden_states
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
            hidden_states, encoder_hidden_states = self._process_block_outputs(
                hidden_states, encoder_hidden_states
            )

        # compute hidden_states residual
        hidden_states = hidden_states.contiguous()

        hidden_states_residual = hidden_states - original_hidden_states

        if encoder_hidden_states is not None and original_encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.contiguous()
            encoder_hidden_states_residual = encoder_hidden_states - original_encoder_hidden_states
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
        if self.context_manager.Bn_compute_blocks() == 0:
            return hidden_states, encoder_hidden_states

        for block in self._Bn_blocks():
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                *args,
                **kwargs,
            )
            hidden_states, encoder_hidden_states = self._process_block_outputs(
                hidden_states, encoder_hidden_states
            )

        return hidden_states, encoder_hidden_states


class PrunedBlocks_Pattern_Base(CachedBlocks_Pattern_Base):
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
        encoder_hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        self.pruned_blocks_step: int = 0  # reset for each step

        # Use it's own cache context.
        try:
            self.context_manager.set_context(self.cache_context)
            self._check_cache_params()
        except ContextNotExistError as e:
            logger.warning(f"Cache context not exist: {e}, skip prune.")
            # Fallback to call all blocks to process the hidden states w/o prune.
            hidden_states, encoder_hidden_states = self.call_blocks(
                hidden_states,
                encoder_hidden_states,
                *args,
                **kwargs,
            )
            return self._process_forward_outputs(
                hidden_states,
                encoder_hidden_states,
            )

        self.context_manager.mark_step_begin()

        if self._check_if_context_parallel_enabled(self.transformer_blocks[0]):
            raise RuntimeError("Block level Context parallelism is not supported in PrunedBlocks.")

        # Call all blocks with prune strategy to process the hidden states.
        for i, block in enumerate(self.transformer_blocks):
            hidden_states, encoder_hidden_states = self.compute_or_prune(
                i,
                block,
                hidden_states,
                encoder_hidden_states,
                *args,
                **kwargs,
            )

        self.context_manager.add_pruned_block(self.pruned_blocks_step)
        self.context_manager.add_actual_block(self.num_blocks)

        return self._process_forward_outputs(
            hidden_states,
            encoder_hidden_states,
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
        encoder_hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ):
        original_hidden_states = hidden_states
        original_encoder_hidden_states = encoder_hidden_states

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
            hidden_states, encoder_hidden_states = self.context_manager.apply_prune(
                hidden_states,
                encoder_hidden_states,
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
                encoder_hidden_states,
                *args,
                **kwargs,
            )
            hidden_states, encoder_hidden_states = self._process_block_outputs(
                hidden_states, encoder_hidden_states
            )
            if not self._skip_prune(block_id):
                hidden_states = hidden_states.contiguous()
                hidden_states_residual = hidden_states - original_hidden_states

                if encoder_hidden_states is not None and original_encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.contiguous()
                    encoder_hidden_states_residual = (
                        encoder_hidden_states - original_encoder_hidden_states
                    )
                else:
                    encoder_hidden_states_residual = None

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
                if encoder_hidden_states_residual is not None:
                    if self.context_manager.is_encoder_cache_residual():
                        self.context_manager.set_Bn_encoder_buffer(
                            encoder_hidden_states_residual,
                            prefix=f"{self.cache_prefix}_{block_id}_Bn_encoder_residual",
                        )
                    else:
                        self.context_manager.set_Bn_encoder_buffer(
                            encoder_hidden_states_residual,
                            prefix=f"{self.cache_prefix}_{block_id}_Bn_encoder_hidden_states",
                        )
            torch._dynamo.graph_break()

        return hidden_states, encoder_hidden_states
