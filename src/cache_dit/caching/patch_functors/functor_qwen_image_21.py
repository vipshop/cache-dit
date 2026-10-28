import torch
import math
from typing import Any, Optional

from diffusers import QwenImage21Transformer2DModel
from diffusers.models.transformers.transformer_qwenimage21 import (
  _IMG_TOKENS_PER_SLOT,
  QwenImage21FlexAttnProcessor,
  QwenImage21KVCache,
  QwenImage21TransformerBlock,
  _qwenimage21_prefix_segments,
  build_qwenimage21_block_causal_mask,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils.peft_utils import apply_lora_scale

from .functor_base import PatchFunctor
from ...logger import init_logger

logger = init_logger(__name__)


def _cp_context(transformer: QwenImage21Transformer2DModel) -> tuple[int, int, Any]:
  """Return `(rank, world_size, cp_config)` for the active context-parallel runtime.

  :param transformer: Transformer instance that may carry `_cp_config`.
  :returns: The Ulysses-local rank, CP world size, and runtime config (or `None` when disabled).
  """

  cp_config = getattr(transformer, "_cp_config", None)
  if cp_config is None:
    return 0, 1, None
  world_size = int(getattr(cp_config, "ulysses_degree", 1) or 1)
  if world_size <= 1:
    return 0, 1, None
  rank = getattr(cp_config, "_ulysses_local_rank", None)
  if rank is None:
    raise RuntimeError("Qwen-Image 2.1 CP is enabled before the Ulysses mesh rank is initialized.")
  return int(rank), world_size, cp_config


def _cp_comm(transformer: QwenImage21Transformer2DModel):
  """Return the lazily created Ulysses communicator bound to the CP group.

  :param transformer: Transformer instance with an initialized `_cp_config`.
  :returns: The cached `_All2AllComm` instance.
  """

  cp_config = transformer._cp_config
  comm = getattr(transformer, "_qwenimage21_cp_comm", None)
  if comm is None or getattr(transformer, "_qwenimage21_cp_comm_config", None) is not cp_config:
    # The communicator binds to one process group: rebuilding the parallelism config must
    # rebuild it too, otherwise the shard size and the group disagree.
    from ...distributed.core._distributed_primitives import _All2AllComm

    comm = _All2AllComm(cp_config)
    transformer._qwenimage21_cp_comm = comm
    transformer._qwenimage21_cp_comm_config = cp_config
    logger.info(f"Qwen-Image 2.1 context parallelism: cp_size={cp_config.ulysses_degree}, "
                "full-sequence shards + head-sharded prefix KV cache.")
  return comm


def _cp_shard(x: torch.Tensor, rank: int, size: int, dim: int) -> tuple[torch.Tensor, int]:
  """Pad and evenly shard a sequence along `dim`.

  :param x: Sequence-shaped tensor.
  :param rank: Ulysses-local rank.
  :param size: CP world size.
  :param dim: Dimension to shard.
  :returns: The local shard and number of padding entries in the global sequence.
  """

  pad = (-x.shape[dim]) % size
  if pad:
    pad_shape = list(x.shape)
    pad_shape[dim] = pad
    x = torch.cat([x, x.new_zeros(pad_shape)], dim=dim)
  local = x.shape[dim] // size
  return x.narrow(dim, rank * local, local), pad


def _cp_gather(x: torch.Tensor, seq_len: int, comm) -> torch.Tensor:
  """Gather local sequence shards and remove global padding.

  :param x: Local padded sequence shard.
  :param seq_len: Unpadded global sequence length.
  :param comm: Communicator returned by :func:`_cp_comm`.
  :returns: The full, unpadded sequence tensor.
  """

  return comm.all_gather_tensor_dim(x, dim=1)[:, :seq_len]


class QwenImage21PatchFunctor(PatchFunctor):
  """Make Qwen-Image 2.1 compatible with cache-dit's cached blocks.

  The model keeps a prefix (text and condition-image) KV cache that is indexed by the
  transformer block index. Cache-dit replaces `transformer.transformer_blocks` with a
  single-element wrapper list, so the original loop runs once with `index_block == 0`
  and every real block would read and write layer 0's K/V. The result does not crash,
  it collapses to color noise, which is why the cache-aware slice has to happen inside
  the block (same fix as sglang's native Qwen-Image 2.1 DiT).

  The transformer forward is copied rather than wrapped for two reasons:

  - it reads `block.attn.processor` to decide whether the block-causal mask must be
    built as a flex `BlockMask`, and the wrapper list has no `.attn`;
  - the per-layer cache has to be forwarded as a whole container instead of one layer.
  """

  def _apply(
    self,
    transformer: QwenImage21Transformer2DModel,
    blocks: torch.nn.ModuleList = None,
    **kwargs,
  ) -> QwenImage21Transformer2DModel:

    real_blocks = transformer.transformer_blocks
    for index, block in enumerate(real_blocks):
      assert isinstance(block, QwenImage21TransformerBlock)
      block._idx = index
      block.forward = __patch_block_forward__.__get__(block)

    # Keep a reference to the real blocks: the forward needs the attention processors while
    # `transformer_blocks` is temporarily swapped for the cached wrapper. A tuple, not the
    # ModuleList itself, so the blocks are not registered a second time and `state_dict()`
    # stays free of duplicate keys.
    transformer._qwenimage21_blocks = tuple(real_blocks)
    transformer.forward = __patch_transformer_forward__.__get__(transformer)

    if getattr(transformer, "_old_forward", None) is not None:
      # accelerate hooks captured the pre-patch forward and `mock_transformer` prefers
      # `_old_forward` when `_hf_hook` is set. Leaving it stale would bypass this patch
      # and silently reuse layer 0 for every block.
      transformer._old_forward = transformer.forward

    transformer._is_patched = True
    return transformer


def __patch_block_forward__(
  self: QwenImage21TransformerBlock,
  hidden_states: torch.Tensor,
  modulation: torch.Tensor,
  rotary_emb: Optional[torch.Tensor] = None,
  attention_mask: Any = None,
  target_token_mask: Optional[torch.Tensor] = None,
  kv_caches: Optional[QwenImage21KVCache] = None,
  kv_cache_mode: Optional[str] = None,
  cache_write_slice: Optional[slice] = None,
  segments: Optional[list] = None,
  key_valid: Optional[torch.Tensor] = None,
  cp_meta: Optional[tuple] = None,
) -> torch.Tensor:
  # Patched in cache-dit: the cached wrapper forwards the same kwarg to every layer, so
  # the per-layer cache is resolved here instead of in the transformer block loop.
  layer_cache = kv_caches.get_layer(self._idx) if kv_caches is not None else None

  mod1, mod2 = modulation.chunk(2, dim=-1)

  img_modulated, img_gate1 = self._modulate(self.img_norm1(hidden_states), mod1, target_token_mask)
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
  hidden_states = hidden_states + img_gate1.tanh() * attn_output

  img_modulated2, img_gate2 = self._modulate(self.img_norm2(hidden_states), mod2, target_token_mask)
  hidden_states = hidden_states + img_gate2.tanh() * self.img_mlp(img_modulated2)

  if hidden_states.dtype == torch.float16:
    hidden_states = hidden_states.clip(-65504, 65504)

  return hidden_states


@apply_lora_scale("attention_kwargs")
def __patch_transformer_forward__(
  self: QwenImage21Transformer2DModel,
  hidden_states: torch.Tensor,
  encoder_hidden_states: torch.Tensor,
  timestep: torch.Tensor,
  img_shapes: list[list[tuple[int, int, int]]],
  img_mask: torch.Tensor,
  encoder_hidden_states_mask: Optional[torch.Tensor] = None,
  attention_kwargs: Optional[dict] = None,
  kv_cache: Optional[QwenImage21KVCache] = None,
  kv_cache_mode: Optional[str] = None,
  return_dict: bool = True,
) -> torch.Tensor | Transformer2DModelOutput:

  batch_size = hidden_states.shape[0]
  if kv_cache is not None and not self.config.causal_condition:
    raise ValueError(
      "kv_cache requires `causal_condition=True`. The cache is only valid because text and condition-image "
      "tokens modulate from t=0, which makes their activations independent of the denoising step.")
  if kv_cache is not None and kv_cache_mode not in ("extract", "cached"):
    raise ValueError(
      f"kv_cache_mode must be 'extract' or 'cached' when kv_cache is provided, got {kv_cache_mode!r}."
    )
  if kv_cache is None and kv_cache_mode is not None:
    raise ValueError(
      f"kv_cache_mode is {kv_cache_mode!r} but no kv_cache was passed to hold the prefix.")

  hidden_states = self.img_in(hidden_states)
  encoder_hidden_states = self.txt_in(encoder_hidden_states)

  repeats = torch.where(img_mask, _IMG_TOKENS_PER_SLOT, 1)[0]
  image_pad_mask = torch.repeat_interleave(img_mask[0], repeats)

  target_tokens = math.prod(img_shapes[0][-1])
  joint_hidden_states = torch.cat(
    [
      encoder_hidden_states,
      encoder_hidden_states.new_zeros(batch_size, target_tokens // 4,
                                      encoder_hidden_states.shape[2]),
    ],
    dim=1,
  )
  joint_hidden_states = joint_hidden_states.repeat_interleave(repeats, dim=1)
  joint_hidden_states[:, image_pad_mask] = hidden_states

  rotary_emb = self.pos_embed(img_shapes[0], image_pad_mask, device=hidden_states.device)
  image_ids, target_token_mask = self.build_token_metadata(image_pad_mask, img_shapes[0])

  timestep = timestep.to(hidden_states.dtype)
  if self.config.causal_condition:
    timestep = torch.cat([timestep, timestep.new_zeros(1)], dim=0)
    modulation_mask = target_token_mask
  else:
    modulation_mask = None
  temb = self.time_text_embed(timestep, hidden_states)
  modulation = self.modulation(temb)

  joint_key_valid = None
  if encoder_hidden_states_mask is not None:
    joint_key_valid = torch.ones(batch_size,
                                 image_pad_mask.shape[0],
                                 dtype=torch.bool,
                                 device=hidden_states.device)
    text_positions = (~image_pad_mask).nonzero(as_tuple=True)[0]
    vlm_text_positions = ~img_mask[0][:encoder_hidden_states_mask.shape[1]]
    joint_key_valid[:, text_positions] = encoder_hidden_states_mask.bool()[:, vlm_text_positions]

  prefix_len = int((~target_token_mask).sum())

  cp_rank, cp_size, cp_config = _cp_context(self)
  cp_enabled = cp_size > 1

  if kv_cache_mode == "cached":
    joint_hidden_states = joint_hidden_states[:, prefix_len:]
    rotary_emb = rotary_emb[prefix_len:]
    modulation_mask = modulation_mask[prefix_len:]
    attention_mask = None if joint_key_valid is None else joint_key_valid[:, None, None, :]
    cache_write_slice = None
    block_segments, block_key_valid = None, None
  else:
    # `transformer_blocks` may be swapped for the cached wrapper, so read the processors
    # from the real block list that this functor captured at patch time.
    processors = [block.attn.processor for block in self._qwenimage21_blocks]
    needs_block_mask = any(
      isinstance(processor, QwenImage21FlexAttnProcessor) for processor in processors)
    attention_mask = (build_qwenimage21_block_causal_mask(
      image_ids, joint_key_valid, batch_size, hidden_states.device) if needs_block_mask else None)
    block_segments = (None if all(
      isinstance(processor, QwenImage21FlexAttnProcessor)
      for processor in processors) else _qwenimage21_prefix_segments(image_ids, prefix_len))
    cache_write_slice = slice(0, prefix_len) if kv_cache_mode == "extract" else None
    block_key_valid = joint_key_valid

  # `norm_out` runs on the gathered sequence: the mask must cover whatever the blocks return,
  # which is the full joint sequence on prefill and target-only rows on cached steps.
  modulation_mask_full = modulation_mask

  cp_meta = None
  cp_seq_len = joint_hidden_states.shape[1]
  if cp_enabled:
    joint_hidden_states, cp_pad = _cp_shard(joint_hidden_states, cp_rank, cp_size, dim=1)
    rotary_emb, rotary_pad = _cp_shard(rotary_emb, cp_rank, cp_size, dim=0)
    assert rotary_pad == cp_pad
    if modulation_mask is not None:
      modulation_mask, mask_pad = _cp_shard(modulation_mask, cp_rank, cp_size, dim=0)
      assert mask_pad == cp_pad
    cp_meta = (_cp_comm(self), cp_pad, cp_size)

  # Iterate the attribute, not the captured list: cache-dit swaps `transformer_blocks`
  # for the cached wrapper and the wrapper drives the real blocks internally.
  for block in self.transformer_blocks:
    if torch.is_grad_enabled() and self.gradient_checkpointing:
      joint_hidden_states = self._gradient_checkpointing_func(
        block,
        joint_hidden_states,
        modulation,
        rotary_emb,
        attention_mask,
        modulation_mask,
        kv_cache,
        kv_cache_mode,
        cache_write_slice,
        block_segments,
        block_key_valid,
        cp_meta,
      )
    else:
      joint_hidden_states = block(
        hidden_states=joint_hidden_states,
        modulation=modulation,
        rotary_emb=rotary_emb,
        attention_mask=attention_mask,
        target_token_mask=modulation_mask,
        kv_caches=kv_cache,
        kv_cache_mode=kv_cache_mode,
        cache_write_slice=cache_write_slice,
        segments=block_segments,
        key_valid=block_key_valid,
        cp_meta=cp_meta,
      )

  if cp_enabled:
    # The pipeline slices `noise_pred[:, -latents.size(1):]`, so hand back the full sequence.
    joint_hidden_states = _cp_gather(joint_hidden_states, cp_seq_len, cp_meta[0])

  joint_hidden_states = self.norm_out(joint_hidden_states, temb, modulation_mask_full)
  output = self.proj_out(joint_hidden_states)

  if not return_dict:
    return (output, )

  return Transformer2DModelOutput(sample=output)
