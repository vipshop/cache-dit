"""Fused q/k norm + RoPE, installed per attention instance.

The attention call itself is not fused: it still goes to whichever backend
``dispatch_attention_fn`` selects, with the same backend and parallel-config attributes the
stock processor passes, so context parallelism and tensor parallelism keep working through the
exact same call site.

Why a per-instance subclass rather than a patched module function: diffusers inlines the q/k
preparation in the processor's ``__call__`` and has no module-level helper to rebind, and
patching the class would leak the fusion into every other FLUX.2 transformer in the process.
The subclass keeps ``isinstance`` checks, the class-level ``_attention_backend`` /
``_parallel_config`` defaults and any attribute cache-dit assigns to ``module.processor`` (its
context-parallel runtime walks the modules and writes ``_cp_config``/``_parallel_config`` onto
the processor instance, which is then handed to ``dispatch_attention_fn`` unchanged).

The KV-cache processors are deliberately not replaced: those belong to the edit variants, where
attention consumes and stores reference K/V, and their contract is wider than the seams fused
here. They keep the stock path and a single log line records it.
"""

from typing import Any, Optional

import torch
from diffusers.models.transformers.transformer_flux2 import (
  Flux2AttnProcessor,
  Flux2ParallelSelfAttnProcessor,
  dispatch_attention_fn,
)

from ....logger import init_logger
from .fused_ops import apply_qk_norm_rope, apply_qk_norm_rope_join, apply_silu_mul

logger = init_logger(__name__)

_ORIGINAL_ATTRIBUTE = "_flux2_klein_original_processor"


def _rotary_tables(image_rotary_emb: Optional[Any]) -> tuple[Any, Any]:
  """Split the model's rotary argument into its `(cos, sin)` pair.

  :param image_rotary_emb: `(cos, sin)` tuple, or `None` when the caller skips the rotation.
  :returns: The two tables, or `(None, None)`.
  """

  if image_rotary_emb is None:
    return None, None
  return image_rotary_emb[0], image_rotary_emb[1]


class _FusedFlux2AttnProcessor(Flux2AttnProcessor):
  """Double-stream processor: fused q/k norm + RoPE, joint streams written in one pass."""

  def __call__(
    self,
    attn: torch.nn.Module,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    image_rotary_emb: Optional[Any] = None,
  ) -> Any:
    """Same contract as `Flux2AttnProcessor.__call__`, with the fused q/k seam.

    :param attn: The `Flux2Attention` module owning the projections and norms.
    :param hidden_states: `(B, S_img, D)` image stream.
    :param encoder_hidden_states: `(B, S_txt, D)` text stream.
    :param attention_mask: Optional mask, forwarded unchanged.
    :param image_rotary_emb: `(cos, sin)` tables for the joint sequence.
    :returns: The attention output, or `(image, text)` when a text stream is present.
    """

    from diffusers.models.transformers.transformer_flux2 import _get_qkv_projections

    query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
      attn, hidden_states, encoder_hidden_states)

    head_dim = attn.head_dim
    query = query.unflatten(-1, (-1, head_dim))
    key = key.unflatten(-1, (-1, head_dim))
    value = value.unflatten(-1, (-1, head_dim))

    cos, sin = _rotary_tables(image_rotary_emb)
    if attn.added_kv_proj_dim is not None:
      encoder_query = encoder_query.unflatten(-1, (-1, head_dim))
      encoder_key = encoder_key.unflatten(-1, (-1, head_dim))
      encoder_value = encoder_value.unflatten(-1, (-1, head_dim))
      if cos is not None:
        # The joined operator needs one epsilon for both streams; the model builds all four
        # norms from the same value, but check instead of assuming it.
        if attn.norm_q.eps == attn.norm_added_q.eps and attn.norm_k.eps == attn.norm_added_k.eps:
          query = apply_qk_norm_rope_join(query,
                                          attn.norm_q.weight,
                                          encoder_query,
                                          attn.norm_added_q.weight,
                                          cos,
                                          sin,
                                          attn.norm_q.eps,
                                          image_norm=attn.norm_q,
                                          text_norm=attn.norm_added_q)
          key = apply_qk_norm_rope_join(key,
                                        attn.norm_k.weight,
                                        encoder_key,
                                        attn.norm_added_k.weight,
                                        cos,
                                        sin,
                                        attn.norm_k.eps,
                                        image_norm=attn.norm_k,
                                        text_norm=attn.norm_added_k)
        else:
          query = torch.cat([
            apply_qk_norm_rope(encoder_query,
                               attn.norm_added_q.weight,
                               cos,
                               sin,
                               0,
                               attn.norm_added_q.eps,
                               norm=attn.norm_added_q),
            apply_qk_norm_rope(query,
                               attn.norm_q.weight,
                               cos,
                               sin,
                               encoder_query.shape[1],
                               attn.norm_q.eps,
                               norm=attn.norm_q)
          ],
                            dim=1)
          key = torch.cat([
            apply_qk_norm_rope(encoder_key,
                               attn.norm_added_k.weight,
                               cos,
                               sin,
                               0,
                               attn.norm_added_k.eps,
                               norm=attn.norm_added_k),
            apply_qk_norm_rope(key,
                               attn.norm_k.weight,
                               cos,
                               sin,
                               encoder_key.shape[1],
                               attn.norm_k.eps,
                               norm=attn.norm_k)
          ],
                          dim=1)
      else:
        query = torch.cat([attn.norm_added_q(encoder_query), attn.norm_q(query)], dim=1)
        key = torch.cat([attn.norm_added_k(encoder_key), attn.norm_k(key)], dim=1)
      value = torch.cat([encoder_value, value], dim=1)
    elif cos is not None:
      query = apply_qk_norm_rope(query,
                                 attn.norm_q.weight,
                                 cos,
                                 sin,
                                 0,
                                 attn.norm_q.eps,
                                 norm=attn.norm_q)
      key = apply_qk_norm_rope(key,
                               attn.norm_k.weight,
                               cos,
                               sin,
                               0,
                               attn.norm_k.eps,
                               norm=attn.norm_k)
    else:
      query = attn.norm_q(query)
      key = attn.norm_k(key)

    hidden_states = dispatch_attention_fn(
      query,
      key,
      value,
      attn_mask=attention_mask,
      backend=self._attention_backend,
      parallel_config=self._parallel_config,
    )
    hidden_states = hidden_states.flatten(2, 3)
    hidden_states = hidden_states.to(query.dtype)

    if encoder_hidden_states is not None:
      encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
        [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]],
        dim=1)
      encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)

    if encoder_hidden_states is not None:
      return hidden_states, encoder_hidden_states
    return hidden_states


class _FusedFlux2ParallelSelfAttnProcessor(Flux2ParallelSelfAttnProcessor):
  """Single-stream processor: fused q/k norm + RoPE over the packed QKV/MLP projection."""

  def __call__(
    self,
    attn: torch.nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    image_rotary_emb: Optional[Any] = None,
  ) -> torch.Tensor:
    """Same contract as `Flux2ParallelSelfAttnProcessor.__call__`, with the fused q/k seam.

    :param attn: The `Flux2ParallelSelfAttention` module.
    :param hidden_states: `(B, S, D)` joint stream.
    :param attention_mask: Optional mask, forwarded unchanged.
    :param image_rotary_emb: `(cos, sin)` tables for the joint sequence.
    :returns: `(B, S, D)` block output.
    """

    hidden_states_proj = attn.to_qkv_mlp_proj(hidden_states)

    qkv_dim = 3 * attn.inner_dim
    mlp_dim = attn.mlp_hidden_dim * attn.mlp_mult_factor
    local_qkv = hidden_states_proj.shape[-1] * qkv_dim // (qkv_dim + mlp_dim)
    qkv, mlp_hidden_states = torch.split(hidden_states_proj,
                                         [local_qkv, hidden_states_proj.shape[-1] - local_qkv],
                                         dim=-1)

    query, key, value = qkv.chunk(3, dim=-1)

    query = query.unflatten(-1, (-1, attn.head_dim))
    key = key.unflatten(-1, (-1, attn.head_dim))
    value = value.unflatten(-1, (-1, attn.head_dim))

    cos, sin = _rotary_tables(image_rotary_emb)
    if cos is not None:
      query = apply_qk_norm_rope(query,
                                 attn.norm_q.weight,
                                 cos,
                                 sin,
                                 0,
                                 attn.norm_q.eps,
                                 norm=attn.norm_q)
      key = apply_qk_norm_rope(key,
                               attn.norm_k.weight,
                               cos,
                               sin,
                               0,
                               attn.norm_k.eps,
                               norm=attn.norm_k)
    else:
      query = attn.norm_q(query)
      key = attn.norm_k(key)

    attn_output = dispatch_attention_fn(
      query,
      key,
      value,
      attn_mask=attention_mask,
      backend=self._attention_backend,
      parallel_config=self._parallel_config,
    )

    attn_output = attn_output.flatten(2, 3)
    attn_output = attn_output.to(query.dtype)

    mlp_hidden_states = apply_silu_mul(mlp_hidden_states)

    hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=-1)
    return attn.to_out(hidden_states)


def _swap_processor(attn: torch.nn.Module, fused_cls: type, stock_cls: type) -> bool:
  """Replace one attention module's processor with the fused subclass instance.

  :param attn: Attention module owning the processor.
  :param fused_cls: Fused processor class to install.
  :param stock_cls: Class the current processor must be, so KV variants are left alone.
  :returns: Whether a replacement happened.
  """

  processor = getattr(attn, "processor", None)
  if processor is None or isinstance(processor, fused_cls):
    return False
  if type(processor) is not stock_cls:
    return False
  fused = fused_cls()
  for attribute in ("_attention_backend", "_parallel_config", "_cp_config"):
    if hasattr(processor, attribute):
      setattr(fused, attribute, getattr(processor, attribute))
  setattr(attn, _ORIGINAL_ATTRIBUTE, processor)
  attn.processor = fused
  return True


def install_attention_patch(transformer: torch.nn.Module) -> None:
  """Install the fused processors on every block of one transformer.

  Idempotent. Instances whose processor is a KV-cache variant keep the stock path.

  :param transformer: FLUX.2-Klein transformer whose attention modules should be upgraded.
  """

  installed = 0
  skipped = 0
  for block in transformer.transformer_blocks:
    if _swap_processor(block.attn, _FusedFlux2AttnProcessor, Flux2AttnProcessor):
      installed += 1
    elif getattr(block.attn, _ORIGINAL_ATTRIBUTE, None) is None:
      skipped += 1
  for block in transformer.single_transformer_blocks:
    if _swap_processor(block.attn, _FusedFlux2ParallelSelfAttnProcessor,
                       Flux2ParallelSelfAttnProcessor):
      installed += 1
    elif getattr(block.attn, _ORIGINAL_ATTRIBUTE, None) is None:
      skipped += 1

  if skipped:
    logger.info(f"FLUX.2 Klein keeps the stock attention processor on {skipped} block(s); "
                "their q/k norm and RoPE are not fused (KV-cache/edit variants).")
  if installed:
    logger.info(f"FLUX.2 Klein fused q/k norm + RoPE installed on {installed} attention module(s).")


def uninstall_attention_patch(transformer: torch.nn.Module) -> None:
  """Restore the stock processors. Used by tests and by the operator switch.

  :param transformer: Transformer whose attention modules should be restored.
  """

  blocks = list(transformer.transformer_blocks) + list(transformer.single_transformer_blocks)
  for block in blocks:
    original = getattr(block.attn, _ORIGINAL_ATTRIBUTE, None)
    if original is None:
      continue
    block.attn.processor = original
    delattr(block.attn, _ORIGINAL_ATTRIBUTE)


def attention_patch_is_installed(transformer: torch.nn.Module) -> bool:
  """Return whether any block currently runs a fused processor.

  :param transformer: Transformer to inspect.
  :returns: `True` when at least one block was upgraded.
  """

  blocks = list(transformer.transformer_blocks) + list(transformer.single_transformer_blocks)
  return any(
    isinstance(block.attn.processor, (_FusedFlux2AttnProcessor,
                                      _FusedFlux2ParallelSelfAttnProcessor)) for block in blocks)


__all__ = [
  "attention_patch_is_installed",
  "install_attention_patch",
  "uninstall_attention_patch",
]
