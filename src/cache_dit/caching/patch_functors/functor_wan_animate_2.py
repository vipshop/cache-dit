import torch
from typing import Any, List, Optional, Tuple, Union

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_wan_animate_2 import (
  WanAnimate2KVCache,
  WanAnimate2KVLayerCache,
  WanAnimate2Transformer3DModel,
  WanAnimate2TransformerBlock,
)

from .functor_base import PatchFunctor
from ...logger import init_logger

logger = init_logger(__name__)

# The upstream block loop skips this layer on the unconditional branch of CFG:
# `if is_uncondtion and idx == 9: continue` (diffusers `transformer_wan_animate_2.py:909-911`).
# Cache-dit drives every block from one wrapper call, so the skip is replayed inside block 9.
_UNCOND_SKIP_LAYER = 9


class WanAnimate2PatchFunctor(PatchFunctor):
  """Make Wan-Animate-2 compatible with cache-dit's cached blocks.

  The model keeps one reference K/V cache per transformer layer and resolves it inside the block
  loop
  (`kv_cache.get(idx)`), and it skips layer 9 on the unconditional CFG branch. Cache-dit replaces
  `transformer.blocks` with a single-element wrapper list driven by one shared kwargs set, so both
  per-block behaviours would silently break: every layer would read and write layer 0's K/V (which
  collapses to colour noise instead of raising) and the unconditional branch would run one extra
  layer. Both are therefore pushed down into the block, mirroring the Qwen-Image 2.1 functor.

  The transformer forward is copied rather than wrapped because the two passes drive the blocks
  differently: `kv_cache_mode="extract"` (the per-segment reference pass) must run every block
  directly so that all layers write their pre-RoPE K/V, while `"cached"` must iterate whatever
  `self.blocks` currently is (the cached wrapper). Upstream: `diffusers/models/transformers/
  transformer_wan_animate_2.py:766-926`.
  """

  def _apply(
    self,
    transformer: WanAnimate2Transformer3DModel,
    blocks: torch.nn.ModuleList = None,
    **kwargs,
  ) -> WanAnimate2Transformer3DModel:

    real_blocks = transformer.blocks
    for index, block in enumerate(real_blocks):
      assert isinstance(block, WanAnimate2TransformerBlock)
      block._idx = index
      block._wan_animate2_block_forward = block.forward
      block.forward = __patch_block_forward__.__get__(block)

    # A tuple, not the ModuleList: the blocks are already registered under `transformer.blocks`, and
    # registering them a second time would duplicate every key in `state_dict()`.
    transformer._wan_animate2_blocks = tuple(real_blocks)
    # Kept for reference / debugging only. `transformer._original_forward` belongs to cache-dit and,
    # with accelerate hooks installed, already points at the patched forward.
    transformer._wan_animate2_original_forward = transformer.forward
    transformer.forward = __patch_transformer_forward__.__get__(transformer)

    if getattr(transformer, "_old_forward", None) is not None:
      # accelerate captured the pre-patch forward and `mock_transformer` prefers `_old_forward` when
      # `_hf_hook` is set. Leaving it stale would bypass this patch entirely.
      transformer._old_forward = transformer.forward

    transformer._is_patched = True
    return transformer


def __patch_block_forward__(
  self: WanAnimate2TransformerBlock,
  hidden_states: torch.Tensor,
  temb: torch.Tensor,
  encoder_hidden_states: torch.Tensor,
  kv_cache: Union[WanAnimate2KVCache, WanAnimate2KVLayerCache, None] = None,
  kv_cache_mode: Optional[str] = None,
  rotary_emb: Optional[torch.Tensor] = None,
  grid_sizes: Optional[torch.Tensor] = None,
  encoder_hidden_states_image: Optional[torch.Tensor] = None,
  reference_rotary_emb: Optional[torch.Tensor] = None,
  reference_grid_sizes: Optional[torch.Tensor] = None,
  attention_mask: Any = None,
  origin_latent_frames: Optional[int] = None,
  origin_latent_hw: Optional[int] = None,
  is_uncondtion: bool = False,
) -> torch.Tensor:

  # Replay of the upstream `if is_uncondtion and idx == 9: continue`.
  if is_uncondtion and self._idx == _UNCOND_SKIP_LAYER:
    return hidden_states

  # The cached wrapper forwards one shared kwargs set, so the per-layer cache is resolved here. A
  # plain layer cache is still accepted to keep the block callable from the stock loop.
  if isinstance(kv_cache, WanAnimate2KVLayerCache):
    layer_cache = kv_cache
  elif kv_cache is None:
    layer_cache = None
  else:
    layer_cache = kv_cache.get(self._idx)

  return self._wan_animate2_block_forward(
    hidden_states=hidden_states,
    temb=temb,
    encoder_hidden_states=encoder_hidden_states,
    kv_cache=layer_cache,
    kv_cache_mode=kv_cache_mode,
    rotary_emb=rotary_emb,
    grid_sizes=grid_sizes,
    encoder_hidden_states_image=encoder_hidden_states_image,
    reference_rotary_emb=reference_rotary_emb,
    reference_grid_sizes=reference_grid_sizes,
    attention_mask=attention_mask,
    origin_latent_frames=origin_latent_frames,
    origin_latent_hw=origin_latent_hw,
  )


def __patch_transformer_forward__(
  self: WanAnimate2Transformer3DModel,
  hidden_states: List[torch.Tensor],
  timestep: torch.Tensor,
  encoder_hidden_states: List[torch.Tensor],
  condition_latents: List[torch.Tensor],
  kv_cache: Optional[WanAnimate2KVCache] = None,
  kv_cache_mode: Optional[str] = None,
  seq_len: Optional[int] = None,
  encoder_hidden_states_image: Optional[torch.Tensor] = None,
  offset_grid_sizes: Optional[torch.Tensor] = None,
  reference_grid_sizes: Optional[torch.Tensor] = None,
  origin_len: Optional[int] = None,
  origin_area: Optional[List[int]] = None,
  is_uncondtion: bool = False,
  return_dict: bool = True,
) -> Union[Transformer2DModelOutput, Tuple[List[torch.Tensor]]]:

  if kv_cache_mode not in ("extract", "cached"):
    raise ValueError(f"`kv_cache_mode` must be either 'extract' or 'cached', got {kv_cache_mode}.")

  # Upstream derives the RoPE device from `patch_embedding.weight`, which layerwise offload parks on
  # CPU while the latents stay on the compute device; derive it from the inputs instead.
  device = hidden_states[0].device

  # 1. Patch embedding. `grid_sizes` describes whichever stream this pass is running over.
  hidden_states = [torch.cat([u, v], dim=0) for u, v in zip(hidden_states, condition_latents)]
  hidden_states = [self.patch_embedding(u.unsqueeze(0)) for u in hidden_states]
  grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in hidden_states])
  hidden_states = [u.flatten(2).transpose(1, 2) for u in hidden_states]
  if any(u.size(1) != seq_len for u in hidden_states):
    raise ValueError(
      f"Each sample must hold exactly `seq_len` ({seq_len}) tokens, got "
      f"{[u.size(1) for u in hidden_states]}. Self-attention here is either dense and unmasked or "
      f"driven "
      f"by a block mask built from the grid, so a padded sequence would not line up.")
  hidden_states = torch.cat(hidden_states)

  # 2. Rotary embeddings for the reference stream. The `refer_offset_*` config values place the
  # reference tokens on a RoPE grid disjoint from the generation tokens; -1 means "use the
  # reference grid size for that axis", resolved per call from the reference grid, which arrives as
  # `offset_grid_sizes` under "extract" and as `reference_grid_sizes` under "cached".
  reference_grid = offset_grid_sizes if kv_cache_mode == "extract" else reference_grid_sizes
  refer_offsets = tuple(offset if offset >= 0 else reference_grid[0][axis].item()
                        for axis, offset in enumerate((self.config.refer_offset_t,
                                                       self.config.refer_offset_h,
                                                       self.config.refer_offset_w)))
  freqs_ref = self._rope_freqs(refer_offsets, device)

  # 3. Time and context embeddings. The reference pass is modulated at a fixed timestep.
  timestep_input = timestep * 0 + 1 if kv_cache_mode == "extract" else timestep
  temb = self.time_embedding(
    self.timesteps_proj(timestep_input).to(dtype=next(self.time_embedding.parameters()).dtype))
  timestep_proj = self.time_projection(temb).unflatten(1, (6, self.config.dim))

  encoder_hidden_states = self.text_embedding(
    torch.stack([
      torch.cat([u, u.new_zeros(self.config.text_len - u.size(0), u.size(1))])
      for u in encoder_hidden_states
    ]))
  encoder_hidden_states_image = (self.img_emb(encoder_hidden_states_image)
                                 if self.config.use_img_emb else None)

  # 4. Per-mode block arguments.
  if kv_cache_mode == "extract":
    block_kwargs = {
      "rotary_emb": freqs_ref,
      "grid_sizes": grid_sizes,
    }
  else:
    # Latent geometry of the full video, which is what the block mask is built over. The current
    # segment is scattered into a buffer of this size inside the attention processor.
    origin_latent_frames = origin_len // 4 + 1
    origin_latent_hw = origin_area[0] * origin_area[1] // 256

    block_mask_id = (origin_latent_frames, origin_latent_hw)
    if block_mask_id not in self.block_masks:
      self.block_masks[block_mask_id] = self.create_mask(origin_latent_frames, origin_latent_hw,
                                                         hidden_states.device)

    block_kwargs = {
      "rotary_emb": self._rope_freqs((0, 0, 0), device),
      "grid_sizes": grid_sizes,
      "reference_rotary_emb": freqs_ref,
      "reference_grid_sizes": reference_grid_sizes,
      "attention_mask": self.block_masks[block_mask_id],
      "origin_latent_frames": origin_latent_frames,
      "origin_latent_hw": origin_latent_hw,
    }

  block_kwargs.update(
    temb=timestep_proj,
    encoder_hidden_states=encoder_hidden_states,
    encoder_hidden_states_image=encoder_hidden_states_image,
    kv_cache_mode=kv_cache_mode,
    is_uncondtion=is_uncondtion,
  )

  # 5. Transformer blocks. "extract" iterates the real blocks so that the reference pass always runs
  # every layer and writes its pre-RoPE K/V, whatever `self.blocks` currently is; "cached" iterates
  # the attribute, which cache-dit swaps for the cached wrapper.
  hidden_states, cp_meta = _maybe_cp_shard(self, hidden_states)

  blocks = self._wan_animate2_blocks if kv_cache_mode == "extract" else self.blocks
  for block in blocks:
    if torch.is_grad_enabled() and self.gradient_checkpointing:
      hidden_states = self._gradient_checkpointing_func(
        block,
        hidden_states,
        kv_cache=kv_cache,
        **block_kwargs,
      )
    else:
      hidden_states = block(hidden_states, kv_cache=kv_cache, **block_kwargs)

  if cp_meta is not None:
    hidden_states = _maybe_cp_gather(cp_meta, hidden_states)

  # 6. Output. Under `kv_cache_mode="extract"` the meaningful product is the populated cache; the
  # sample is returned anyway so both modes have the same return type.
  hidden_states = self.head(hidden_states, temb)
  output = [u.float() for u in self.unpatchify(hidden_states, grid_sizes)]

  if not return_dict:
    return (output, )
  return Transformer2DModelOutput(sample=output)


def _cp_world_size(transformer: WanAnimate2Transformer3DModel) -> int:
  """Return the context-parallel world size bound to the transformer.

  :param transformer: Transformer instance that may carry `_cp_config`.
  :returns: The Ulysses degree, or 1 when context parallelism is disabled.
  """

  cp_config = getattr(transformer, "_cp_config", None)
  if cp_config is None:
    return 1
  return int(getattr(cp_config, "ulysses_degree", 1) or 1)


def _maybe_cp_shard(
  transformer: WanAnimate2Transformer3DModel,
  hidden_states: torch.Tensor,
) -> Optional[tuple]:
  """Pad and shard the token sequence for context parallelism.

  Attention runs on the gathered full sequence, so only `hidden_states` is sharded here; RoPE,
  `grid_sizes` and the flex block mask keep their full-video geometry inside the processor.

  :param transformer: Transformer instance carrying `_cp_config` when CP is enabled.
  :param hidden_states: Full `[B, S, D]` token sequence.
  :returns: `(local_hidden_states, cp_meta)`; `cp_meta` is `None` when CP is disabled.
  """

  cp_size = _cp_world_size(transformer)
  if cp_size <= 1:
    return hidden_states, None
  cp_rank = getattr(transformer._cp_config, "_ulysses_local_rank", None)
  if cp_rank is None:
    raise RuntimeError("Wan-Animate-2 CP is enabled before the Ulysses mesh rank is initialized.")
  seq_len = hidden_states.shape[1]
  pad = (-seq_len) % cp_size
  if pad:
    padding = hidden_states.new_zeros(hidden_states.shape[:1] + (pad, ) + hidden_states.shape[2:])
    hidden_states = torch.cat([hidden_states, padding], dim=1)
  local = hidden_states.shape[1] // cp_size
  shard = hidden_states.narrow(1, cp_rank * local, local).contiguous()
  return shard, (transformer, pad, cp_size, seq_len)


def wan_animate_2_cp_comm(transformer: WanAnimate2Transformer3DModel):
  """Return the Ulysses communicator bound to the active CP group.

  :param transformer: Transformer instance with an initialized `_cp_config`.
  :returns: The cached `_All2AllComm` instance, or `None` when CP is disabled.
  """

  cp_config = getattr(transformer, "_cp_config", None)
  if cp_config is None:
    return None
  comm = getattr(transformer, "_wan_animate2_cp_comm", None)
  if comm is None or getattr(transformer, "_wan_animate2_cp_comm_config", None) is not cp_config:
    # The communicator binds to one process group: rebuilding the parallelism config must rebuild it
    # too, otherwise the shard size and the group disagree.
    from ...distributed.core._distributed_primitives import _All2AllComm

    comm = _All2AllComm(cp_config)
    transformer._wan_animate2_cp_comm = comm
    transformer._wan_animate2_cp_comm_config = cp_config
  return comm


def _maybe_cp_gather(cp_meta: tuple, hidden_states: torch.Tensor) -> torch.Tensor:
  """Gather the sharded token sequence back to the full resolution.

  :param cp_meta: Tuple returned by `_maybe_cp_shard`.
  :param hidden_states: Local `[B, S / cp_size, D]` token sequence.
  :returns: The gathered `[B, S, D]` sequence without padding.
  """

  transformer, pad, cp_size, seq_len = cp_meta
  if cp_size <= 1:
    return hidden_states
  comm = wan_animate_2_cp_comm(transformer)
  return comm.all_gather_tensor_dim(hidden_states, dim=1)[:, :seq_len]
