import torch

from ..forward_pattern import ForwardPattern
from ..cache_contexts.cache_manager import (
  ContextNotExistError, )
from .pattern_base import (
  CachedBlocks_Pattern_Base, )

from ...logger import init_logger

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
      self, hidden_states: torch.Tensor | tuple) -> tuple[torch.Tensor, torch.Tensor | None]:
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
      (Fn_hidden_states_residual
       if not self.context_manager.is_l1_diff_enabled() else hidden_states),
      parallelized=self._is_parallelized(),
      prefix=(f"{self.cache_prefix}_Fn_residual" if not self.context_manager.is_l1_diff_enabled()
              else f"{self.cache_prefix}_Fn_hidden_states"),
    )

    if can_use_cache:
      self.context_manager.add_cached_step()
      del Fn_hidden_states_residual
      hidden_states, new_encoder_hidden_states = self.context_manager.apply_cache(
        hidden_states,
        new_encoder_hidden_states,  # encoder_hidden_states not use cache
        prefix=(f"{self.cache_prefix}_Bn_residual" if self.context_manager.is_cache_residual() else
                f"{self.cache_prefix}_Bn_hidden_states"),
        encoder_prefix=(f"{self.cache_prefix}_Bn_residual"
                        if self.context_manager.is_encoder_cache_residual() else
                        f"{self.cache_prefix}_Bn_hidden_states"),
      )

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
        new_encoder_hidden_states_residual = (new_encoder_hidden_states - old_encoder_hidden_states)
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

      # Call last `n` blocks to further process the hidden states
      # for higher precision.
      if self.context_manager.Bn_compute_blocks() > 0:
        hidden_states, new_encoder_hidden_states = self.call_Bn_blocks(
          hidden_states,
          *args,
          **kwargs,
        )

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
