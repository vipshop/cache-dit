"""Boogu-Image transformer: blocks, double-stream stage and the 2D model.

Adapted from Boogu-Image (`boogu/models/transformers/transformer_boogu.py`),
Apache-2.0. Context parallelism is handled inside the double-stream block (see
`_boogu_local_double_stream_inputs`) instead of being patched on from outside;
TeaCache / TaylorSeer are removed because cache-dit's own cache owns that role.
"""

import itertools
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.loaders.single_file_model import FromOriginalModelMixin
from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import (
  USE_PEFT_BACKEND,
  logging,
  scale_lora_layers,
  unscale_lora_layers,
)
from einops import rearrange

from .attention_processor import (
  BooguImageAttnProcessor,
  BooguImageAttnProcessorFlash2Varlen,
  BooguImageDoubleStreamSelfAttnProcessor,
  BooguImageDoubleStreamSelfAttnProcessorFlash2Varlen,
)
from .block_lumina2 import (
  Lumina2CombinedTimestepCaptionEmbedding,
  LuminaFeedForward,
  LuminaLayerNormContinuous,
  LuminaRMSNormZero,
)
from .fused_ops import apply_gated_rms_norm
from .fused_ops import apply_scaled_rms_norm
from .rms_norm import RMSNorm
from .rope import BooguImageDoubleStreamRotaryPosEmbed

logger = logging.get_logger(__name__)


def _tensor_split_bounds(length: int, world: int, rank: int) -> Tuple[int, int]:
  """Return the `[start, end)` chunk of ``torch.tensor_split(length, world)[rank]``.

  Mirrors `_PartitionAnythingSharder` so the block's local length lists agree
  with the framework sharder that owns the single-stream split.

  :param length: Global length along the sharded dimension.
  :param world: Number of Ulysses ranks.
  :param rank: This rank's position in the Ulysses group.
  :returns: The half-open `(start, end)` chunk bounds.
  """

  base = length // world
  rem = length % world
  start = rank * base + min(rank, rem)
  end = start + base + (1 if rank < rem else 0)
  return start, end


def _gather_sequence(tensor: torch.Tensor, cp_config) -> torch.Tensor:
  """All-gather a locally sharded sequence tensor back to global order.

  :param tensor: Locally sharded tensor.
  :param cp_config: Runtime CP config providing the mesh and the UAA flag.
  :returns: The full tensor in rank order.
  """

  from ....distributed.core._context_parallel import _EquipartitionSharder
  from ....distributed.core._context_parallel import _PartitionAnythingSharder

  if cp_config.ulysses_anything:
    return _PartitionAnythingSharder.unshard_anything(tensor, 1, cp_config._flattened_mesh)
  return _EquipartitionSharder.unshard(tensor, 1, cp_config._flattened_mesh)


def _boogu_local_double_stream_inputs(
  cp_config,
  img_hidden_states: torch.Tensor,
  instruct_hidden_states: torch.Tensor,
  image_rotary_emb: torch.Tensor,
  rotary_emb: torch.Tensor,
  encoder_seq_lengths: Optional[List[int]],
  seq_lengths: Optional[List[int]],
) -> Optional[Tuple]:
  """Slice the double-stream inputs down to this rank's Ulysses chunk.

  Returns `None` when the replicated path must run instead: no CP, a batch
  larger than one, or a padded instruction stream (the fixed-offset rope stitch
  below assumes unpadded streams).

  :param cp_config: Runtime CP config, or `None`.
  :param img_hidden_states: Image stream, full or already local.
  :param instruct_hidden_states: Instruction stream, full or already local.
  :param image_rotary_emb: Global rope table of the image stream.
  :param rotary_emb: Global joint rope table laid out as `[instruct | image]`.
  :param encoder_seq_lengths: Global per-sample instruction lengths.
  :param seq_lengths: Global per-sample joint lengths.
  :returns: The local `(img, instruct, img_rope, joint_rope, enc_lens,
    seq_lens)` tuple, or `None` for the replicated path.
  """

  if cp_config is None or cp_config.ulysses_degree <= 1:
    return None
  if not encoder_seq_lengths:
    return None
  if img_hidden_states.shape[0] != 1:
    # The CP hooks have already split both streams at the first block, so there
    # is nothing to fall back to: running the replicated path here would attend
    # over local chunks without any communication.
    raise NotImplementedError("Boogu-Image context parallelism runs batch size 1; got a batch of "
                              f"{img_hidden_states.shape[0]}. Run the batch as separate calls, or "
                              "disable context parallelism.")

  world = cp_config.ulysses_degree
  rank = int(getattr(cp_config, "_ulysses_local_rank", 0) or 0)
  # The python length lists are never split by the CP hooks, so they always
  # carry the global sizes (batch 1 means row 0 is the whole story).
  l_instruct = encoder_seq_lengths[0]
  l_img = seq_lengths[0] - l_instruct
  if not cp_config.ulysses_anything:
    uneven = [
      f"{name}={length}" for name, length in (("instruction", l_instruct), ("image", l_img),
                                              ("joint", l_instruct + l_img)) if length % world != 0
    ]
    if uneven:
      raise ValueError("Boogu-Image context parallelism needs the instruction, image and joint "
                       f"sequence lengths to divide evenly across {world} ranks, got "
                       f"{', '.join(uneven)}. Pass `--ulysses-anything` to shard uneven "
                       "lengths.")
  inst_start, inst_end = _tensor_split_bounds(l_instruct, world, rank)
  img_start, img_end = _tensor_split_bounds(l_img, world, rank)
  enc_local = inst_end - inst_start
  img_local_len = img_end - img_start

  inst_width = instruct_hidden_states.shape[1]
  is_full = inst_width == l_instruct
  is_local = inst_width == enc_local and img_hidden_states.shape[1] == img_local_len
  if is_full:
    # The CP hooks split both streams at the first block; a full-width input
    # means the plan did not run, so stay on the replicated path.
    logger.debug(
      "[CP] Boogu double-stream received full-width streams (%d); keeping the "
      "replicated path.", inst_width)
    return None
  if not is_local:
    # Padded instruction stream (valid length < tensor width) breaks the
    # fixed-offset rope slicing below.
    logger.debug(
      "[CP] Boogu double-stream falls back to the replicated path: instruction "
      "width %d matches neither the global length %d nor the local chunk %d.",
      inst_width,
      l_instruct,
      enc_local,
    )
    return None

  # Every layer receives the GLOBAL ropes from transformer.forward, so the local
  # ropes are always sliced from them. The joint rope is laid out per sample as
  # `[instruct | combined-img]`, so the local rope is stitched from two
  # segments rather than sliced contiguously.
  local_image_rotary_emb = image_rotary_emb[:, img_start:img_end]
  local_rotary_emb = torch.cat(
    [
      rotary_emb[:, inst_start:inst_end],
      rotary_emb[:, l_instruct + img_start:l_instruct + img_end],
    ],
    dim=1,
  )
  return (
    img_hidden_states,
    instruct_hidden_states,
    local_image_rotary_emb,
    local_rotary_emb,
    [enc_local],
    [enc_local + img_local_len],
  )


class BooguImageTransformerBlock(nn.Module):
  """
    Basic Boogu-Image transformer block: attention + MLP + RMSNorm.
    """

  def __init__(
    self,
    dim: int,
    num_attention_heads: int,
    num_kv_heads: int,
    multiple_of: int,
    ffn_dim_multiplier: float,
    norm_eps: float,
    modulation: bool = True,
  ) -> None:
    """Initialize the transformer block."""
    super().__init__()
    self.head_dim = dim // num_attention_heads
    self.modulation = modulation

    try:
      processor = BooguImageAttnProcessorFlash2Varlen()
    except ImportError:
      processor = BooguImageAttnProcessor()

    # Initialize attention layer
    self.attn = Attention(
      query_dim=dim,
      cross_attention_dim=None,
      dim_head=dim // num_attention_heads,
      qk_norm="rms_norm",
      heads=num_attention_heads,
      kv_heads=num_kv_heads,
      eps=1e-5,
      bias=False,
      out_bias=False,
      processor=processor,
    )

    # Initialize feed-forward network
    self.feed_forward = LuminaFeedForward(
      dim=dim,
      inner_dim=4 * dim,
      multiple_of=multiple_of,
      ffn_dim_multiplier=ffn_dim_multiplier,
    )

    # Initialize normalization layers
    if modulation:
      self.norm1 = LuminaRMSNormZero(embedding_dim=dim,
                                     norm_eps=norm_eps,
                                     norm_elementwise_affine=True)
    else:
      self.norm1 = RMSNorm(dim, eps=norm_eps)

    self.ffn_norm1 = RMSNorm(dim, eps=norm_eps)
    self.norm2 = RMSNorm(dim, eps=norm_eps)
    self.ffn_norm2 = RMSNorm(dim, eps=norm_eps)

    self.initialize_weights()

  def initialize_weights(self) -> None:
    """Initialize linear weights and modulation parameters."""
    nn.init.xavier_uniform_(self.attn.to_q.weight)
    nn.init.xavier_uniform_(self.attn.to_k.weight)
    nn.init.xavier_uniform_(self.attn.to_v.weight)
    nn.init.xavier_uniform_(self.attn.to_out[0].weight)

    nn.init.xavier_uniform_(self.feed_forward.linear_1.weight)
    nn.init.xavier_uniform_(self.feed_forward.linear_2.weight)
    nn.init.xavier_uniform_(self.feed_forward.linear_3.weight)

    if self.modulation:
      nn.init.zeros_(self.norm1.linear.weight)
      nn.init.zeros_(self.norm1.linear.bias)

  def forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    image_rotary_emb: torch.Tensor,
    temb: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """Forward pass of the transformer block.

    Args:
        hidden_states: Input hidden states tensor
        attention_mask: Attention mask tensor
        image_rotary_emb: Rotary embeddings for image tokens
        temb: Optional timestep embedding tensor

    Returns:
        torch.Tensor: Output hidden states after transformer block processing
    """

    if self.modulation:
      if temb is None:
        raise ValueError("temb must be provided when modulation is enabled")
      norm_hidden_states, gate_msa, scale_mlp, gate_mlp = self.norm1(hidden_states, temb)

      attn_output = self.attn(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_hidden_states,
        attention_mask=attention_mask,
        image_rotary_emb=image_rotary_emb,
      )
      hidden_states = apply_gated_rms_norm(attn_output, self.norm2, gate_msa, hidden_states)
      mlp_output = self.feed_forward(apply_scaled_rms_norm(hidden_states, self.ffn_norm1,
                                                           scale_mlp))
      hidden_states = apply_gated_rms_norm(mlp_output, self.ffn_norm2, gate_mlp, hidden_states)
    else:
      norm_hidden_states = self.norm1(hidden_states)
      attn_output = self.attn(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_hidden_states,
        attention_mask=attention_mask,
        image_rotary_emb=image_rotary_emb,
      )
      hidden_states = hidden_states + self.norm2(attn_output)
      mlp_output = self.feed_forward(self.ffn_norm1(hidden_states))
      hidden_states = hidden_states + self.ffn_norm2(mlp_output)

    return hidden_states


class BooguImageNoiseRefinerTransformerBlock(BooguImageTransformerBlock):
  pass


class BooguImageRefImgRefinerTransformerBlock(BooguImageTransformerBlock):
  pass


class BooguImageContextRefinerTransformerBlock(BooguImageTransformerBlock):
  pass


class BooguImageSingleStreamTransformerBlock(BooguImageTransformerBlock):
  pass


class BooguImageDoubleStreamTransformerBlock(nn.Module):
  """Boogu-Image double-stream block.

  Here "double-stream" is the same idea as a "dual-stream" layer: instruction tokens and image
  tokens are processed in parallel streams.
  """

  def __init__(
    self,
    dim: int,
    num_attention_heads: int,
    num_kv_heads: int,
    multiple_of: int,
    ffn_dim_multiplier: float,
    norm_eps: float,
    modulation: bool = True,
    index: int = 0,
  ) -> None:
    """Initialize the double stream transformer block.

    :param index: Zero-based position of this block in `double_stream_layers`.
    """
    super().__init__()
    self.head_dim = dim // num_attention_heads
    self.num_attention_heads = num_attention_heads
    self.modulation = modulation
    self.hidden_size = dim
    self.index = index

    try:
      processor = BooguImageAttnProcessorFlash2Varlen()
    except ImportError:
      processor = BooguImageAttnProcessor()

    try:
      double_stream_processor = (BooguImageDoubleStreamSelfAttnProcessorFlash2Varlen(
        head_dim=self.head_dim,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        qkv_bias=False,
      ))
    except ImportError:
      double_stream_processor = BooguImageDoubleStreamSelfAttnProcessor(
        head_dim=self.head_dim,
        num_attention_heads=num_attention_heads,
        num_kv_heads=num_kv_heads,
        qkv_bias=False,
      )

    # Image stream components.
    self.img_instruct_attn = Attention(
      query_dim=dim,
      cross_attention_dim=None,
      dim_head=dim // num_attention_heads,
      qk_norm="rms_norm",
      heads=num_attention_heads,
      kv_heads=num_kv_heads,
      eps=1e-5,
      bias=False,
      out_bias=False,
      processor=double_stream_processor,
    )

    self.img_self_attn = Attention(
      query_dim=dim,
      cross_attention_dim=None,
      dim_head=dim // num_attention_heads,
      qk_norm="rms_norm",
      heads=num_attention_heads,
      kv_heads=num_kv_heads,
      eps=1e-5,
      bias=False,
      out_bias=False,
      processor=processor,
    )

    self.img_feed_forward = LuminaFeedForward(
      dim=dim,
      inner_dim=4 * dim,
      multiple_of=multiple_of,
      ffn_dim_multiplier=ffn_dim_multiplier,
    )

    if modulation:
      # Image modulation terms: cross-attn, MLP, self-attn.
      self.img_norm1 = LuminaRMSNormZero(embedding_dim=dim,
                                         norm_eps=norm_eps,
                                         norm_elementwise_affine=True)
      self.img_norm2 = LuminaRMSNormZero(embedding_dim=dim,
                                         norm_eps=norm_eps,
                                         norm_elementwise_affine=True)
      self.img_norm3 = LuminaRMSNormZero(embedding_dim=dim,
                                         norm_eps=norm_eps,
                                         norm_elementwise_affine=True)
    else:
      self.img_norm1 = RMSNorm(dim, eps=norm_eps)
      self.img_norm2 = RMSNorm(dim, eps=norm_eps)
      self.img_norm3 = RMSNorm(dim, eps=norm_eps)

    self.img_ffn_norm1 = RMSNorm(dim, eps=norm_eps)
    self.img_attn_norm = RMSNorm(dim, eps=norm_eps)
    self.img_self_attn_norm = RMSNorm(dim, eps=norm_eps)
    self.img_ffn_norm2 = RMSNorm(dim, eps=norm_eps)

    # Instruction stream components.
    self.instruct_feed_forward = LuminaFeedForward(
      dim=dim,
      inner_dim=4 * dim,
      multiple_of=multiple_of,
      ffn_dim_multiplier=ffn_dim_multiplier,
    )

    if modulation:
      # Instruction modulation terms: cross-attn, MLP.
      self.instruct_norm1 = LuminaRMSNormZero(embedding_dim=dim,
                                              norm_eps=norm_eps,
                                              norm_elementwise_affine=True)
      self.instruct_norm2 = LuminaRMSNormZero(embedding_dim=dim,
                                              norm_eps=norm_eps,
                                              norm_elementwise_affine=True)
    else:
      self.instruct_norm1 = RMSNorm(dim, eps=norm_eps)
      self.instruct_norm2 = RMSNorm(dim, eps=norm_eps)

    self.instruct_ffn_norm1 = RMSNorm(dim, eps=norm_eps)
    self.instruct_attn_norm = RMSNorm(dim, eps=norm_eps)
    self.instruct_ffn_norm2 = RMSNorm(dim, eps=norm_eps)

    self.initialize_weights()

    # double_stream_processor owns its own q/k/v projections.
    for param in self.img_instruct_attn.to_q.parameters():
      param.requires_grad = False
    for param in self.img_instruct_attn.to_k.parameters():
      param.requires_grad = False
    for param in self.img_instruct_attn.to_v.parameters():
      param.requires_grad = False

    del self.img_instruct_attn.to_k
    del self.img_instruct_attn.to_v
    del self.img_instruct_attn.to_q

  def initialize_weights(self) -> None:
    """Initialize linear weights and modulation parameters."""
    nn.init.xavier_uniform_(self.img_instruct_attn.to_out[0].weight)

    # Keep Xavier init consistent across Boogu-Image blocks.
    nn.init.xavier_uniform_(self.img_self_attn.to_q.weight)
    nn.init.xavier_uniform_(self.img_self_attn.to_k.weight)
    nn.init.xavier_uniform_(self.img_self_attn.to_v.weight)
    nn.init.xavier_uniform_(self.img_self_attn.to_out[0].weight)

    nn.init.xavier_uniform_(self.img_feed_forward.linear_1.weight)
    nn.init.xavier_uniform_(self.img_feed_forward.linear_2.weight)
    nn.init.xavier_uniform_(self.img_feed_forward.linear_3.weight)

    nn.init.xavier_uniform_(self.instruct_feed_forward.linear_1.weight)
    nn.init.xavier_uniform_(self.instruct_feed_forward.linear_2.weight)
    nn.init.xavier_uniform_(self.instruct_feed_forward.linear_3.weight)

    # Initialize modulation parameters
    if self.modulation:
      nn.init.zeros_(self.img_norm1.linear.weight)
      nn.init.zeros_(self.img_norm1.linear.bias)
      nn.init.zeros_(self.img_norm2.linear.weight)
      nn.init.zeros_(self.img_norm2.linear.bias)
      nn.init.zeros_(self.img_norm3.linear.weight)
      nn.init.zeros_(self.img_norm3.linear.bias)

      nn.init.zeros_(self.instruct_norm1.linear.weight)
      nn.init.zeros_(self.instruct_norm1.linear.bias)
      nn.init.zeros_(self.instruct_norm2.linear.weight)
      nn.init.zeros_(self.instruct_norm2.linear.bias)

  def forward(
      self,
      img_hidden_states: torch.Tensor,  # [B, L_img, D] - Image tokens (ref_img + noise_img)
      instruct_hidden_states: torch.Tensor,  # [B, L_instruct, D] - Instruction tokens
      img_attention_mask: torch.Tensor,  # [B, L_img] - Mask for [ref_img + noise_img]
      joint_attention_mask: torch.Tensor,  # [B, L_total] - Mask for [instruct + img]
      image_rotary_emb: torch.Tensor,  # [B, L_img, head_dim] - RoPE for [ref + noise]
      rotary_emb: torch.Tensor,  # [B, L_total, head_dim] - RoPE for [instruct + img]
      temb: Optional[torch.Tensor] = None,  # [B, 1024] - Timestep embeddings
      encoder_seq_lengths: Optional[
        List[int]] = None,  # [B] - Instruction sequence lengths per sample
      seq_lengths: Optional[List[int]] = None,  # [B] - Total sequence lengths per sample
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run one dual-stream (double-stream) block step.

    The block owns the context-parallel bookkeeping for this stage because two
    things cannot be expressed with cache-dit's declarative hook plan: the python
    length lists (`encoder_seq_lengths` / `seq_lengths`) that hooks leave global,
    and the joint rope layout (`[instruct | combined-img]` per sample, so the
    local rope is a two-segment stitch rather than a contiguous slice).

    Data flow per forward: the CP hooks split both streams once at the first
    block, so every block receives and returns the same (local) width — that is
    what keeps a cached block stack's `output - input` residual shape invariant.
    The block only rewrites the rope tables and the two python length lists for
    the local chunk; the transformer gathers both streams after the stack.

    :param img_hidden_states: Image stream `[B, L_img, D]` (full or local chunk).
    :param instruct_hidden_states: Instruction stream `[B, L_instruct, D]`.
    :param img_attention_mask: Padding mask for the image stream.
    :param joint_attention_mask: Padding mask for the joint `[instruct + img]`.
    :param image_rotary_emb: RoPE table of the image stream.
    :param rotary_emb: RoPE table of the joint sequence.
    :param temb: Timestep embedding, required when modulation is enabled.
    :param encoder_seq_lengths: Global per-sample instruction lengths.
    :param seq_lengths: Global per-sample joint lengths.
    :returns: The updated `(img_hidden_states, instruct_hidden_states)` pair.
    """

    if self.modulation and temb is None:
      raise ValueError("temb must be provided when modulation is enabled")

    joint_processor = getattr(self.img_instruct_attn, "processor", None)
    self_processor = getattr(self.img_self_attn, "processor", None)
    cp_config = getattr(joint_processor, "_cp_config", None)
    cp_local = _boogu_local_double_stream_inputs(
      cp_config,
      img_hidden_states,
      instruct_hidden_states,
      image_rotary_emb,
      rotary_emb,
      encoder_seq_lengths,
      seq_lengths,
    )
    if cp_local is not None:
      (img_hidden_states, instruct_hidden_states, image_rotary_emb, rotary_emb, encoder_seq_lengths,
       seq_lengths) = cp_local
    # Both processors only take the Ulysses path while this stage really runs on
    # local chunks; on the replicated fallback they must stay on the local path.
    cp_on = cp_local is not None
    if joint_processor is not None:
      joint_processor._boogu_cp_eligible = cp_on
    if self_processor is not None:
      self_processor._boogu_cp_eligible = cp_on

    # Extract dimensions
    batch_size = img_hidden_states.shape[0]
    L_instruct = instruct_hidden_states.shape[1]  # Instruction sequence length
    L_img = img_hidden_states.shape[1]  # Image sequence length (ref_img + noise_img)

    if self.modulation:
      # Step 1: modulation for both streams.
      img_norm1_out, img_gate_msa, img_scale_mlp, img_gate_mlp = self.img_norm1(
        img_hidden_states, temb)
      img_norm2_out, img_shift_mlp, _, _ = self.img_norm2(img_hidden_states, temb)
      img_norm3_out, img_gate_self, _, _ = self.img_norm3(img_hidden_states, temb)

      (
        instruct_norm1_out,
        instruct_gate_msa,
        instruct_scale_mlp,
        instruct_gate_mlp,
      ) = self.instruct_norm1(instruct_hidden_states, temb)
      instruct_norm2_out, instruct_shift_mlp, _, _ = self.instruct_norm2(
        instruct_hidden_states, temb)

      # Step 2: joint attention on [instruct + img].
      # Call processor directly because Attention.forward does not expose these dual-stream args.
      joint_attn_out = self.img_instruct_attn.processor(
        attn=self.img_instruct_attn,
        img_hidden_states=img_norm1_out,
        instruct_hidden_states=instruct_norm1_out,
        joint_attention_mask=joint_attention_mask,
        rotary_emb=rotary_emb,
        encoder_seq_lengths=encoder_seq_lengths,
        seq_lengths=seq_lengths,
      )

      # Split back into instruction/image segments: narrow the joint output
      # directly when B=1 has no padding on either stream.
      if (batch_size == 1 and encoder_seq_lengths[0] == L_instruct
          and seq_lengths[0] - encoder_seq_lengths[0] == L_img):
        instruct_attn_out = joint_attn_out[:, :L_instruct]
        img_attn_out = joint_attn_out[:, L_instruct:seq_lengths[0]]
      else:
        instruct_attn_out = instruct_hidden_states.new_zeros(batch_size, L_instruct,
                                                             self.hidden_size)
        img_attn_out = img_hidden_states.new_zeros(batch_size, L_img, self.hidden_size)
        for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
          instruct_attn_out[i, :encoder_seq_len] = joint_attn_out[i, :encoder_seq_len]
          img_attn_out[i, :seq_len - encoder_seq_len] = joint_attn_out[i, encoder_seq_len:seq_len]

      # Step 3: image self-attention.
      img_self_attn_out = self.img_self_attn(
        hidden_states=img_norm3_out,
        encoder_hidden_states=img_norm3_out,
        attention_mask=img_attention_mask,
        image_rotary_emb=image_rotary_emb,
      )

      # Step 4: residual updates.
      # gated residual: fused norm+gate epilogue when
      # VOMNI_ENABLE_CUSTOM_KERNELS=1, else the eager chain
      img_hidden_states = apply_gated_rms_norm(img_attn_out, self.img_attn_norm, img_gate_msa,
                                               img_hidden_states)
      img_hidden_states = apply_gated_rms_norm(
        img_self_attn_out,
        self.img_self_attn_norm,
        img_gate_self,
        img_hidden_states,
      )

      img_mlp_input = (1 + img_scale_mlp.unsqueeze(1)) * img_norm2_out + img_shift_mlp.unsqueeze(1)
      img_mlp_out = self.img_feed_forward(self.img_ffn_norm1(img_mlp_input))
      img_hidden_states = apply_gated_rms_norm(img_mlp_out, self.img_ffn_norm2, img_gate_mlp,
                                               img_hidden_states)

      instruct_hidden_states = apply_gated_rms_norm(
        instruct_attn_out,
        self.instruct_attn_norm,
        instruct_gate_msa,
        instruct_hidden_states,
      )

      instruct_mlp_input = (
        1 + instruct_scale_mlp.unsqueeze(1)) * instruct_norm2_out + instruct_shift_mlp.unsqueeze(1)
      instruct_mlp_out = self.instruct_feed_forward(self.instruct_ffn_norm1(instruct_mlp_input))
      instruct_hidden_states = apply_gated_rms_norm(
        instruct_mlp_out,
        self.instruct_ffn_norm2,
        instruct_gate_mlp,
        instruct_hidden_states,
      )

    else:
      # Non-modulated branch used by context-style blocks.
      img_norm1_out = self.img_norm1(img_hidden_states)
      img_norm3_out = self.img_norm3(img_hidden_states)
      instruct_norm1_out = self.instruct_norm1(instruct_hidden_states)

      # Same processor path as above.
      joint_attn_out = self.img_instruct_attn.processor(
        attn=self.img_instruct_attn,
        img_hidden_states=img_norm1_out,
        instruct_hidden_states=instruct_norm1_out,
        joint_attention_mask=joint_attention_mask,
        rotary_emb=rotary_emb,
        encoder_seq_lengths=encoder_seq_lengths,
        seq_lengths=seq_lengths,
      )

      # narrow the joint output directly when B=1 has no padding
      if (batch_size == 1 and encoder_seq_lengths[0] == L_instruct
          and seq_lengths[0] - encoder_seq_lengths[0] == L_img):
        instruct_attn_out = joint_attn_out[:, :L_instruct]
        img_attn_out = joint_attn_out[:, L_instruct:seq_lengths[0]]
      else:
        instruct_attn_out = instruct_hidden_states.new_zeros(batch_size, L_instruct,
                                                             self.hidden_size)
        img_attn_out = img_hidden_states.new_zeros(batch_size, L_img, self.hidden_size)
        for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
          instruct_attn_out[i, :encoder_seq_len] = joint_attn_out[i, :encoder_seq_len]
          img_attn_out[i, :seq_len - encoder_seq_len] = joint_attn_out[i, encoder_seq_len:seq_len]

      img_self_attn_out = self.img_self_attn(
        hidden_states=img_norm3_out,
        encoder_hidden_states=img_norm3_out,
        attention_mask=img_attention_mask,
        image_rotary_emb=image_rotary_emb,
      )

      img_hidden_states = img_hidden_states + self.img_attn_norm(img_attn_out)
      img_hidden_states = img_hidden_states + self.img_self_attn_norm(img_self_attn_out)
      img_norm2_out = self.img_norm2(img_hidden_states)
      img_mlp_out = self.img_feed_forward(self.img_ffn_norm1(img_norm2_out))
      img_hidden_states = img_hidden_states + self.img_ffn_norm2(img_mlp_out)

      instruct_hidden_states = instruct_hidden_states + self.instruct_attn_norm(instruct_attn_out)
      instruct_norm2_out = self.instruct_norm2(instruct_hidden_states)
      instruct_mlp_out = self.instruct_feed_forward(self.instruct_ffn_norm1(instruct_norm2_out))
      instruct_hidden_states = instruct_hidden_states + self.instruct_ffn_norm2(instruct_mlp_out)

    return img_hidden_states, instruct_hidden_states


class BooguImageTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin,
                                   FromOriginalModelMixin):
  """Boogu-Image transformer with mixed stream topology.

  Early layers use double-stream (aka dual-stream) processing, then switch to single-stream joint
  processing.
  """

  _supports_gradient_checkpointing = True
  _no_split_modules = [
    "BooguImageTransformerBlock",
    "BooguImageNoiseRefinerTransformerBlock",
    "BooguImageRefImgRefinerTransformerBlock",
    "BooguImageContextRefinerTransformerBlock",
    "BooguImageSingleStreamTransformerBlock",
    "BooguImageDoubleStreamTransformerBlock",
    "PromptEmbedding",
    "nn.Embedding",
  ]
  # Only the single-stream stack takes part in regional compilation and
  # regional quantization. Upstream keeps the noise/ref-image refiners and the
  # double-stream layers out of regional compilation for numerical stability,
  # and they are a small minority of the FLOPs anyway.
  _repeated_blocks = ["BooguImageSingleStreamTransformerBlock"]
  _skip_layerwise_casting_patterns = ["x_embedder", "norm", "embedding"]

  @register_to_config
  def __init__(
    self,
    patch_size: int = 2,
    in_channels: int = 16,
    out_channels: Optional[int] = None,
    hidden_size: int = 2304,
    num_layers: int = 26,
    num_double_stream_layers: int = 2,
    num_refiner_layers: int = 2,
    num_attention_heads: int = 24,
    num_kv_heads: int = 8,
    multiple_of: int = 256,
    ffn_dim_multiplier: Optional[float] = None,
    norm_eps: float = 1e-5,
    axes_dim_rope: Tuple[int, int, int] = (40, 40, 40),
    axes_lens: Tuple[int, int, int] = (2048, 1664, 1664),
    # instruction_feat_dim: int = 1024,
    instruction_feature_configs: Dict[str, Any] = dict(
      instruction_feat_dim=1024,
      reduce_type="mean",
      num_instruction_feat_layers=1,
    ),
    prompt_tuning_configs: Dict[str, Any] = dict(use_prompt_tuning=False),
    timestep_scale: float = 1.0,
  ) -> None:
    """Initialize the Boogu-Image mixed single-double stream transformer model."""
    super().__init__()

    # Validate configuration
    if (hidden_size // num_attention_heads) != sum(axes_dim_rope):
      raise ValueError(f"hidden_size // num_attention_heads ({hidden_size // num_attention_heads}) "
                       f"must equal sum(axes_dim_rope) ({sum(axes_dim_rope)})")

    if num_double_stream_layers > num_layers:
      raise ValueError(
        f"num_double_stream_layers ({num_double_stream_layers}) cannot be greater than "
        f"num_layers ({num_layers})")

    self.out_channels = out_channels or in_channels
    self.num_double_stream_layers = num_double_stream_layers
    self.num_single_stream_layers = num_layers - num_double_stream_layers
    self.instruction_feature_configs = instruction_feature_configs
    self.prompt_tuning_configs = prompt_tuning_configs
    self.preprocessed_instruction_feat_dim = (
      self.cal_preprocessed_instruction_feat_dim(instruction_feature_configs))

    # Initialize embeddings
    self.rope_embedder = BooguImageDoubleStreamRotaryPosEmbed(
      theta=10000,
      axes_dim=axes_dim_rope,
      axes_lens=axes_lens,
      patch_size=patch_size,
    )

    self.x_embedder = nn.Linear(
      in_features=patch_size * patch_size * in_channels,
      out_features=hidden_size,
    )

    self.ref_image_patch_embedder = nn.Linear(
      in_features=patch_size * patch_size * in_channels,
      out_features=hidden_size,
    )

    self.time_caption_embed = Lumina2CombinedTimestepCaptionEmbedding(
      hidden_size=hidden_size,
      instruction_feat_dim=self.preprocessed_instruction_feat_dim,
      norm_eps=norm_eps,
      timestep_scale=timestep_scale,
    )

    # Refiner layers.
    self.noise_refiner = nn.ModuleList([
      BooguImageNoiseRefinerTransformerBlock(
        hidden_size,
        num_attention_heads,
        num_kv_heads,
        multiple_of,
        ffn_dim_multiplier,
        norm_eps,
        modulation=True,
      ) for _ in range(num_refiner_layers)
    ])

    self.ref_image_refiner = nn.ModuleList([
      BooguImageRefImgRefinerTransformerBlock(
        hidden_size,
        num_attention_heads,
        num_kv_heads,
        multiple_of,
        ffn_dim_multiplier,
        norm_eps,
        modulation=True,
      ) for _ in range(num_refiner_layers)
    ])

    self.context_refiner = nn.ModuleList([
      BooguImageContextRefinerTransformerBlock(
        hidden_size,
        num_attention_heads,
        num_kv_heads,
        multiple_of,
        ffn_dim_multiplier,
        norm_eps,
        modulation=False,
      ) for _ in range(num_refiner_layers)
    ])

    # Mixed architecture: dual-stream first, then single-stream.
    # Here "double-stream" and "dual-stream" mean the same thing.
    self.double_stream_layers = nn.ModuleList([
      BooguImageDoubleStreamTransformerBlock(
        hidden_size,
        num_attention_heads,
        num_kv_heads,
        multiple_of,
        ffn_dim_multiplier,
        norm_eps,
        modulation=True,
        index=index,
      ) for index in range(num_double_stream_layers)
    ])

    # Single-stream layers process the fused sequence.
    self.single_stream_layers = nn.ModuleList([
      BooguImageSingleStreamTransformerBlock(
        hidden_size,
        num_attention_heads,
        num_kv_heads,
        multiple_of,
        ffn_dim_multiplier,
        norm_eps,
        modulation=True,
      ) for _ in range(self.num_single_stream_layers)
    ])

    # Output norm and projection.
    self.norm_out = LuminaLayerNormContinuous(
      embedding_dim=hidden_size,
      conditioning_embedding_dim=min(hidden_size, 1024),
      elementwise_affine=False,
      eps=1e-6,
      bias=True,
      out_dim=patch_size * patch_size * self.out_channels,
    )

    # Distinguish multiple reference images.
    self.image_index_embedding = nn.Parameter(torch.randn(5,
                                                          hidden_size))  # support max 5 ref images

    self.gradient_checkpointing = False

    self.initialize_weights()

    self.layers = list(self.double_stream_layers) + list(self.single_stream_layers)

  def initialize_weights(self) -> None:
    """Initialize the weights of the model.

    Uses Xavier uniform initialization for linear layers.
    """
    nn.init.xavier_uniform_(self.x_embedder.weight)
    nn.init.constant_(self.x_embedder.bias, 0.0)

    nn.init.xavier_uniform_(self.ref_image_patch_embedder.weight)
    nn.init.constant_(self.ref_image_patch_embedder.bias, 0.0)

    nn.init.zeros_(self.norm_out.linear_1.weight)
    nn.init.zeros_(self.norm_out.linear_1.bias)
    nn.init.zeros_(self.norm_out.linear_2.weight)
    nn.init.zeros_(self.norm_out.linear_2.bias)

    nn.init.normal_(self.image_index_embedding, std=0.02)

  def img_patch_embed_and_refine(
    self,
    hidden_states,
    ref_image_hidden_states,
    padded_img_mask,
    padded_ref_img_mask,
    noise_rotary_emb,
    ref_img_rotary_emb,
    l_effective_ref_img_len,
    l_effective_img_len,
    temb,
  ):
    """Embed image patches and run the refiner blocks."""
    batch_size = len(hidden_states)
    max_combined_img_len = max([
      img_len + sum(ref_img_len)
      for img_len, ref_img_len in zip(l_effective_img_len, l_effective_ref_img_len)
    ])

    hidden_states = self.x_embedder(hidden_states)
    ref_image_hidden_states = self.ref_image_patch_embedder(ref_image_hidden_states)

    for i in range(batch_size):
      shift = 0
      for j, ref_img_len in enumerate(l_effective_ref_img_len[i]):
        ref_image_hidden_states[i, shift:shift + ref_img_len, :] = (
          ref_image_hidden_states[i, shift:shift + ref_img_len, :] + self.image_index_embedding[j])
        shift += ref_img_len

    for layer in self.noise_refiner:
      hidden_states = layer(hidden_states, padded_img_mask, noise_rotary_emb, temb)

    flat_l_effective_ref_img_len = list(itertools.chain(*l_effective_ref_img_len))
    num_ref_images = len(flat_l_effective_ref_img_len)
    max_ref_img_len = max(flat_l_effective_ref_img_len) if flat_l_effective_ref_img_len else 0

    # Text-to-image yields a zero-length reference row. The refiner would then
    # run on `[batch, 0, hidden]` (a zero-length attention call) and write
    # nothing back, so skipping it is numerically identical and avoids feeding a
    # degenerate sequence to the backend.
    if num_ref_images > 0 and max_ref_img_len > 0:
      batch_ref_img_mask = ref_image_hidden_states.new_zeros(num_ref_images,
                                                             max_ref_img_len,
                                                             dtype=torch.bool)
      batch_ref_image_hidden_states = ref_image_hidden_states.new_zeros(
        num_ref_images, max_ref_img_len, self.config.hidden_size)
      batch_ref_img_rotary_emb = hidden_states.new_zeros(
        num_ref_images,
        max_ref_img_len,
        ref_img_rotary_emb.shape[-1],
        dtype=ref_img_rotary_emb.dtype,
      )
      batch_temb = temb.new_zeros(num_ref_images, *temb.shape[1:], dtype=temb.dtype)

      # Flatten reference images into a temporary batch.
      idx = 0
      for i in range(batch_size):
        shift = 0
        for ref_img_len in l_effective_ref_img_len[i]:
          batch_ref_img_mask[idx, :ref_img_len] = True
          batch_ref_image_hidden_states[idx, :ref_img_len] = (ref_image_hidden_states[i,
                                                                                      shift:shift +
                                                                                      ref_img_len])
          batch_ref_img_rotary_emb[idx, :ref_img_len] = ref_img_rotary_emb[i, shift:shift +
                                                                           ref_img_len]
          batch_temb[idx] = temb[i]
          shift += ref_img_len
          idx += 1

      # Refine each reference-image sample.
      for layer in self.ref_image_refiner:
        batch_ref_image_hidden_states = layer(
          batch_ref_image_hidden_states,
          batch_ref_img_mask,
          batch_ref_img_rotary_emb,
          batch_temb,
        )

      # Restore reference-image sequence layout.
      idx = 0
      for i in range(batch_size):
        shift = 0
        for ref_img_len in l_effective_ref_img_len[i]:
          ref_image_hidden_states[i, shift:shift +
                                  ref_img_len] = (batch_ref_image_hidden_states[idx, :ref_img_len])
          shift += ref_img_len
          idx += 1

    combined_img_hidden_states = hidden_states.new_zeros(batch_size, max_combined_img_len,
                                                         self.config.hidden_size)
    for i, (ref_img_len, img_len) in enumerate(zip(l_effective_ref_img_len, l_effective_img_len)):
      combined_img_hidden_states[i, :sum(ref_img_len)] = ref_image_hidden_states[
        i, :sum(ref_img_len)]
      combined_img_hidden_states[i, sum(ref_img_len):sum(ref_img_len) +
                                 img_len] = hidden_states[i, :img_len]

    return combined_img_hidden_states

  def flat_and_pad_to_seq(self, hidden_states, ref_image_hidden_states):
    """Flatten patch tokens and pad to batched sequences."""
    batch_size = len(hidden_states)
    p = self.config.patch_size
    device = hidden_states[0].device

    img_sizes = [(img.size(1), img.size(2)) for img in hidden_states]
    l_effective_img_len = [(H // p) * (W // p) for (H, W) in img_sizes]

    if ref_image_hidden_states is not None:
      ref_img_sizes = [[(img.size(1), img.size(2)) for img in imgs] if imgs is not None else None
                       for imgs in ref_image_hidden_states]
      l_effective_ref_img_len = [
        [(ref_img_size[0] // p) * (ref_img_size[1] // p)
         for ref_img_size in _ref_img_sizes] if _ref_img_sizes is not None else [0]
        for _ref_img_sizes in ref_img_sizes
      ]
    else:
      ref_img_sizes = [None for _ in range(batch_size)]
      l_effective_ref_img_len = [[0] for _ in range(batch_size)]

    max_ref_img_len = max([sum(ref_img_len) for ref_img_len in l_effective_ref_img_len])
    max_img_len = max(l_effective_img_len)

    # Reference-image patch embeddings.
    flat_ref_img_hidden_states = []
    for i in range(batch_size):
      if ref_img_sizes[i] is not None:
        imgs = []
        for ref_img in ref_image_hidden_states[i]:
          C, H, W = ref_img.size()
          ref_img = rearrange(ref_img, "c (h p1) (w p2) -> (h w) (p1 p2 c)", p1=p, p2=p)
          imgs.append(ref_img)

        img = torch.cat(imgs, dim=0)
        flat_ref_img_hidden_states.append(img)
      else:
        flat_ref_img_hidden_states.append(None)

    # Noise-image patch embeddings.
    flat_hidden_states = []
    for i in range(batch_size):
      img = hidden_states[i]
      C, H, W = img.size()

      img = rearrange(img, "c (h p1) (w p2) -> (h w) (p1 p2 c)", p1=p, p2=p)
      flat_hidden_states.append(img)

    padded_ref_img_hidden_states = torch.zeros(
      batch_size,
      max_ref_img_len,
      flat_hidden_states[0].shape[-1],
      device=device,
      dtype=flat_hidden_states[0].dtype,
    )
    padded_ref_img_mask = torch.zeros(batch_size, max_ref_img_len, dtype=torch.bool, device=device)
    for i in range(batch_size):
      if ref_img_sizes[i] is not None:
        padded_ref_img_hidden_states[i, :sum(l_effective_ref_img_len[i])] = (
          flat_ref_img_hidden_states[i])
        padded_ref_img_mask[i, :sum(l_effective_ref_img_len[i])] = True

    padded_hidden_states = torch.zeros(
      batch_size,
      max_img_len,
      flat_hidden_states[0].shape[-1],
      device=device,
      dtype=flat_hidden_states[0].dtype,
    )
    padded_img_mask = torch.zeros(batch_size, max_img_len, dtype=torch.bool, device=device)
    for i in range(batch_size):
      padded_hidden_states[i, :l_effective_img_len[i]] = flat_hidden_states[i]
      padded_img_mask[i, :l_effective_img_len[i]] = True

    return (
      padded_hidden_states,
      padded_ref_img_hidden_states,
      padded_img_mask,
      padded_ref_img_mask,
      l_effective_ref_img_len,
      l_effective_img_len,
      ref_img_sizes,
      img_sizes,
    )

  def cal_preprocessed_instruction_feat_dim(self, instruction_feature_configs: Dict[str, Any]):
    num_instruction_feat_layers = max(
      instruction_feature_configs.get("num_instruction_feat_layers", 1), 1)
    instruction_feat_dim = instruction_feature_configs.get("instruction_feat_dim", 4096)
    reduce_type = instruction_feature_configs.get("reduce_type", "concat")
    if "cat" in reduce_type.lower():
      return num_instruction_feat_layers * instruction_feat_dim
    elif "mean" in reduce_type.lower():
      return instruction_feat_dim
    else:
      raise ValueError(f"Invalid reduce_type: {reduce_type}")

  def preprocess_instruction_hidden_states(self, raw_instruction_hidden_states,
                                           instruction_feature_configs: Dict[str, Any]):
    num_instruction_feat_layers = max(
      instruction_feature_configs.get("num_instruction_feat_layers", 1), 1)
    reduce_type = instruction_feature_configs.get("reduce_type", "concat")

    instruction_hidden_states = None
    if isinstance(raw_instruction_hidden_states, torch.Tensor):
      instruction_hidden_states = raw_instruction_hidden_states
    elif isinstance(raw_instruction_hidden_states, (list, tuple)):
      assert len(raw_instruction_hidden_states) == num_instruction_feat_layers
      if "cat" in reduce_type.lower():
        instruction_hidden_states = torch.cat(raw_instruction_hidden_states, dim=-1)
      elif "mean" in reduce_type.lower():
        instruction_hidden_states = torch.mean(torch.stack(raw_instruction_hidden_states), dim=0)
      else:
        raise ValueError(f"Invalid reduce_type: {reduce_type}")
    else:
      raise ValueError(
        f"Invalid type of raw_instruction_hidden_states, expected torch.Tensor or list, but got {type(raw_instruction_hidden_states)}"
      )

    assert (self.preprocessed_instruction_feat_dim == instruction_hidden_states.shape[-1])

    return instruction_hidden_states

  def _boogu_cp_active(self) -> bool:
    """Whether cache-dit context parallelism owns this transformer's layout.

    cache-dit sets `_cp_config` on the transformer it parallelizes; under
    context parallelism the sequence layout is per-rank and cache-dit owns
    mask flattening, so the batch-size-1 mask flags must stay off.

    :returns: True when a cache-dit context parallel config is attached.
    """
    return getattr(self, "_cp_config", None) is not None

  def forward(
    self,
    hidden_states: Union[torch.Tensor, List[torch.Tensor]],
    timestep: torch.Tensor,
    instruction_hidden_states: torch.Tensor,
    freqs_cis: torch.Tensor,
    instruction_attention_mask: torch.Tensor,
    ref_image_hidden_states: Optional[List[List[torch.Tensor]]] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = False,
  ) -> Union[torch.Tensor, Transformer2DModelOutput]:
    """
        Forward pass:
        context/refiner -> dual-stream (double-stream) -> fusion -> single-stream -> projection.
        """
    instruction_hidden_states = self.preprocess_instruction_hidden_states(
      instruction_hidden_states, self.instruction_feature_configs)

    if attention_kwargs is not None:
      attention_kwargs = attention_kwargs.copy()
      lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
      lora_scale = 1.0

    if USE_PEFT_BACKEND:
      # weight the lora layers by setting `lora_scale` for each PEFT layer
      scale_lora_layers(self, lora_scale)
    else:
      if (attention_kwargs is not None and attention_kwargs.get("scale", None) is not None):
        logger.warning(
          "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

    # === 1. Initial processing (same as original Boogu-Image) ===
    batch_size = len(hidden_states)
    is_hidden_states_tensor = isinstance(hidden_states, torch.Tensor)

    if is_hidden_states_tensor:
      assert hidden_states.ndim == 4
      hidden_states = [_hidden_states for _hidden_states in hidden_states]

    device = hidden_states[0].device

    # Timestep and instruction embedding.
    temb, instruction_hidden_states = self.time_caption_embed(timestep, instruction_hidden_states,
                                                              hidden_states[0].dtype)

    # Flatten and pad token sequences.
    (
      hidden_states,
      ref_image_hidden_states,
      img_mask,
      ref_img_mask,
      l_effective_ref_img_len,
      l_effective_img_len,
      ref_img_sizes,
      img_sizes,
    ) = self.flat_and_pad_to_seq(hidden_states, ref_image_hidden_states)

    # Build rotary embeddings and sequence lengths.
    (
      context_rotary_emb,
      ref_img_rotary_emb,
      noise_rotary_emb,
      rotary_emb,
      encoder_seq_lengths,
      seq_lengths,
      combined_img_rotary_emb,
      combined_img_seq_lengths,
    ) = self.rope_embedder(
      freqs_cis,
      instruction_attention_mask,
      l_effective_ref_img_len,
      l_effective_img_len,
      ref_img_sizes,
      img_sizes,
      device,
    )

    # Context refinement (replicated under CP: its sequences are short).
    for layer in self.context_refiner:
      instruction_hidden_states = layer(
        instruction_hidden_states,
        instruction_attention_mask,
        context_rotary_emb,
      )

    # Image patch embedding and refinement.
    combined_img_hidden_states = self.img_patch_embed_and_refine(
      hidden_states,
      ref_image_hidden_states,
      img_mask,
      ref_img_mask,
      noise_rotary_emb,
      ref_img_rotary_emb,
      l_effective_ref_img_len,
      l_effective_img_len,
      temb,
    )

    # Dual-stream (double-stream) stage.
    instruct_hidden_states = instruction_hidden_states
    img_hidden_states = combined_img_hidden_states

    # Joint mask for [instruct + image].
    max_seq_len = max(seq_lengths)
    joint_attention_mask = hidden_states.new_zeros(batch_size, max_seq_len, dtype=torch.bool)
    for i, seq_len in enumerate(seq_lengths):
      joint_attention_mask[i, :seq_len] = True

    # Run dual-stream blocks.
    if self.num_double_stream_layers > 0:
      # Image-only mask for [ref + noise].
      max_img_len = max(combined_img_seq_lengths)
      img_attention_mask = hidden_states.new_zeros(batch_size, max_img_len, dtype=torch.bool)
      for i, img_seq_len in enumerate(combined_img_seq_lengths):
        img_attention_mask[i, :img_seq_len] = True

      for layer in self.double_stream_layers:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
          img_hidden_states, instruct_hidden_states = (self._gradient_checkpointing_func(
            layer,
            img_hidden_states,
            instruct_hidden_states,
            img_attention_mask,
            joint_attention_mask,
            combined_img_rotary_emb,
            rotary_emb,
            temb,
            encoder_seq_lengths,
            seq_lengths,
          ))
        else:
          img_hidden_states, instruct_hidden_states = layer(
            img_hidden_states,
            instruct_hidden_states,
            img_attention_mask,
            joint_attention_mask,
            combined_img_rotary_emb,
            rotary_emb,
            temb,
            encoder_seq_lengths,
            seq_lengths,
          )

    # Under context parallelism every double-stream block keeps its inputs and
    # outputs at the local chunk width (so the cache residual stays shape
    # invariant); gather once here, before the fusion loop needs full streams.
    cp_config = getattr(self, "_cp_config", None)
    if cp_config is not None and img_hidden_states.shape[1] < max(combined_img_seq_lengths):
      img_hidden_states = _gather_sequence(img_hidden_states, cp_config)
      instruct_hidden_states = _gather_sequence(instruct_hidden_states, cp_config)

    # Fuse streams to joint sequence.
    joint_hidden_states = hidden_states.new_zeros(batch_size, max(seq_lengths),
                                                  self.config.hidden_size)
    for i, (encoder_seq_len, seq_len) in enumerate(zip(encoder_seq_lengths, seq_lengths)):
      joint_hidden_states[i, :encoder_seq_len] = instruct_hidden_states[i, :encoder_seq_len]
      joint_hidden_states[i, encoder_seq_len:seq_len] = img_hidden_states[i, :seq_len -
                                                                          encoder_seq_len]

    # Single-stream stage.
    hidden_states = joint_hidden_states

    for layer in self.single_stream_layers:
      if torch.is_grad_enabled() and self.gradient_checkpointing:
        hidden_states = self._gradient_checkpointing_func(layer, hidden_states,
                                                          joint_attention_mask, rotary_emb, temb)
      else:
        hidden_states = layer(hidden_states, joint_attention_mask, rotary_emb, temb)

    # Same split-outside / gather-outside contract as the double-stream stage:
    # every cached block stays shape invariant, and the single gather happens
    # here before the output projection and unpatchify need full sequences.
    if cp_config is not None and hidden_states.shape[1] < max(seq_lengths):
      hidden_states = _gather_sequence(hidden_states, cp_config)

    # Output projection.
    hidden_states = self.norm_out(hidden_states, temb)

    # Reshape back to image format.
    p = self.config.patch_size
    output = []
    for i, (img_size, img_len,
            seq_len) in enumerate(zip(img_sizes, l_effective_img_len, seq_lengths)):
      height, width = img_size
      img_tokens = hidden_states[i][seq_len - img_len:seq_len]
      img_output = rearrange(
        img_tokens,
        "(h w) (p1 p2 c) -> c (h p1) (w p2)",
        h=height // p,
        w=width // p,
        p1=p,
        p2=p,
      )
      output.append(img_output)

    if is_hidden_states_tensor:
      output = torch.stack(output, dim=0)

    # Reset LoRA scaling.
    if USE_PEFT_BACKEND:
      unscale_lora_layers(self, lora_scale)

    if not return_dict:
      return output
    return Transformer2DModelOutput(sample=output)
