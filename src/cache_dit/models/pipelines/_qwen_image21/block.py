"""Fused forward for one Qwen-Image 2.1 transformer block.

The block class is left untouched: only the instance's `forward` is rebound, so
`_repeated_blocks = ["QwenImage21TransformerBlock"]`, the cached-blocks wrapper and the
forward-pattern signature check all keep seeing the class they expect.

Two call conventions reach this function and both must work:

- diffusers' own `QwenImage21Transformer2DModel.forward` passes a single layer's
  `QwenImage21KVLayerCache` (keyword calls, and 10 positional args from the gradient
  checkpointing branch);
- cache-dit's patched forward passes the `QwenImage21KVCache` container, because the
  cached-blocks wrapper forwards the same kwargs to every layer.

The container is therefore detected by duck typing rather than by position.
"""

from typing import Any, Optional

import torch

from .fused_ops import apply_gated_residual, apply_ln_modulate, apply_silu_mul


@torch.compiler.disable
def _resolve_layer_cache(block: torch.nn.Module, kv_caches: Any) -> Any:
  """Resolve one block's own layer cache outside the compiled region.

  `block._idx` is a per-block Python int, so tracing the lookup makes dynamo specialize a
  separate graph per block; with 32 blocks that trips `recompile_limit` and every block
  past the limit silently falls back to eager (measured: +11% per step). Keeping the
  lookup behind `torch.compiler.disable` keeps all blocks on one graph.

  :param block: Block whose per-layer cache is needed.
  :param kv_caches: Container forwarded by cache-dit's patched transformer forward.
  :returns: The layer cache bound to `block`.
  """

  return kv_caches.get_layer(block._idx)


def fused_block_forward(
  self: torch.nn.Module,
  hidden_states: torch.Tensor,
  modulation: torch.Tensor,
  rotary_emb: Optional[torch.Tensor] = None,
  attention_mask: Any = None,
  target_token_mask: Optional[torch.Tensor] = None,
  layer_cache: Any = None,
  kv_cache_mode: Optional[str] = None,
  cache_write_slice: Optional[slice] = None,
  segments: Optional[list] = None,
  key_valid: Optional[torch.Tensor] = None,
  *,
  kv_caches: Any = None,
  cp_meta: Optional[tuple] = None,
) -> torch.Tensor:
  """Run one block with fused LayerNorm/modulation and gated residuals.

  The attention call is passed through unchanged, so `dispatch_attention_fn` keeps
  receiving exactly the tensors the stock pipeline hands it.

  :param self: The patched `QwenImage21TransformerBlock`.
  :param hidden_states: `(B, S, D)` residual stream.
  :param modulation: `(B (+ 1), 4D)` shared modulation for all blocks.
  :param rotary_emb: `(S, D // 2)` complex64 rotary table.
  :param attention_mask: Block-causal or padding mask, whatever the caller built.
  :param target_token_mask: `(S,)` bool mask selecting target-image rows.
  :param layer_cache: Single-layer cache, or the container on the cached path.
  :param kv_cache_mode: `"extract"`, `"cached"` or `None`.
  :param cache_write_slice: Prefix slice to store on the extract step.
  :param segments: Prefix segments consumed by the stock attention processor.
  :param key_valid: Padding mask over keys.
  :param kv_caches: Cache container forwarded by cache-dit's patched forward.
  :param cp_meta: Context-parallel `(comm, pad, size)` tuple, when CP is enabled.
  :returns: The updated residual stream.
  """

  if kv_caches is not None:
    layer_cache = kv_caches
  if layer_cache is not None and hasattr(layer_cache, "get_layer"):
    layer_cache = _resolve_layer_cache(self, layer_cache)

  mod1, mod2 = modulation.chunk(2, dim=-1)

  img_modulated, img_gate1 = apply_ln_modulate(hidden_states, mod1, target_token_mask,
                                               self.img_norm1.eps)
  attn_kwargs = dict(
    attention_mask=attention_mask,
    rotary_emb=rotary_emb,
    layer_cache=layer_cache,
    kv_cache_mode=kv_cache_mode,
    cache_write_slice=cache_write_slice,
    segments=segments,
    key_valid=key_valid,
  )
  if cp_meta is not None:
    # Only the context-parallel processor accepts it; the stock one would raise on the kwarg.
    attn_kwargs["cp_meta"] = cp_meta
  attn_output = self.attn(hidden_states=img_modulated, **attn_kwargs)
  hidden_states = apply_gated_residual(hidden_states, img_gate1, attn_output)

  img_modulated2, img_gate2 = apply_ln_modulate(hidden_states, mod2, target_token_mask,
                                                self.img_norm2.eps)
  hidden_states = apply_gated_residual(hidden_states, img_gate2, self.img_mlp(img_modulated2))

  if hidden_states.dtype == torch.float16:
    hidden_states = hidden_states.clip(-65504, 65504)

  return hidden_states


def fused_swiglu_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
  """Run one `QwenImage21SwiGLUFeedForward` with the fused `silu * up` kernel.

  Same expression as upstream (`out(silu(gate_layer(x)) * proj(x))`), with the two
  projections' outputs combined in one vectorised kernel instead of a separate SiLU and
  multiply pass.

  :param self: The patched `QwenImage21SwiGLUFeedForward`.
  :param hidden_states: `(B, S, D)` block input.
  :returns: `(B, S, D)` FFN output.
  """

  return self.out(apply_silu_mul(self.gate_layer(hidden_states), self.proj(hidden_states)))


def install_fused_modules(transformer: torch.nn.Module) -> None:
  """Bind the fused forwards on every block, SwiGLU branch and attention processor.

  Idempotent. The attention processor is only swapped when the fused q/k norm + RoPE is both
  enabled and useful: a flex processor builds its own `BlockMask` and is left alone, and the
  fused processor hands context-parallel calls back to the stock implementation.

  :param transformer: Qwen-Image 2.1 transformer whose modules should be rebound.
  """

  from diffusers.models.transformers.transformer_qwenimage21 import (
    QwenImage21AttnProcessor,
    QwenImage21FlexAttnProcessor,
  )

  from ....logger import init_logger
  from .attention_processor import FusedQwenImage21AttnProcessor
  from .fused_ops import ACTIVE_OPS

  logger = init_logger(__name__)
  flex_blocks = 0
  for index, block in enumerate(transformer.transformer_blocks):
    block._idx = index
    block.forward = fused_block_forward.__get__(block)
    block.img_mlp.forward = fused_swiglu_forward.__get__(block.img_mlp)

    processor = block.attn.processor
    if isinstance(processor, QwenImage21FlexAttnProcessor):
      flex_blocks += 1
      continue
    if "qk_norm_rope" in ACTIVE_OPS and type(processor) is QwenImage21AttnProcessor:
      block.attn.processor = FusedQwenImage21AttnProcessor()

  if flex_blocks:
    logger.info(f"Qwen-Image 2.1 fused attention keeps {flex_blocks} flex attention processors "
                "on the stock implementation (they own their prefill BlockMask).")


__all__ = ["fused_block_forward", "fused_swiglu_forward", "install_fused_modules"]
