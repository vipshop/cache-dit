"""Context-parallel plan for LTX-2 video transformers.

This planner reuses the LTXVideo implementation pattern, but adjusts the plan and patched helper
logic for LTX-2-specific timestep and mask behavior.
"""

import functools
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.transformer_ltx2 import (
  LTX2Attention,
  LTX2AudioVideoAttnProcessor,
  LTX2VideoTransformer3DModel,
  apply_interleaved_rotary_emb,
  apply_split_rotary_emb,
)
from ...attention import _dispatch_attention_fn
from ...distributed.core import (
  _ContextParallelInput,
  _ContextParallelModelPlan,
  _ContextParallelOutput,
)
from ...logger import init_logger
from ..config import ParallelismConfig
from .register import (
  ContextParallelismPlanner,
  ContextParallelismPlannerRegister,
)

logger = init_logger(__name__)


@ContextParallelismPlannerRegister.register("LTX2")
class LTX2ContextParallelismPlanner(ContextParallelismPlanner):

  def _apply(
    self,
    transformer: Optional[torch.nn.Module | ModelMixin] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:
    assert transformer is not None, "Transformer must be provided."
    assert isinstance(
      transformer,
      LTX2VideoTransformer3DModel), "Transformer must be an instance of LTX2VideoTransformer3DModel"

    # LTX2ImageToVideoPipeline passes `timestep` as a 2D `(B, seq_len)` tensor named
    # `video_timestep`. The cache-dit CP plan shards that tensor correctly for Ulysses / Ring CP.

    # Patch attention-mask preparation so head sharding and global-sequence padding stay aligned
    # under CP.
    LTX2Attention.prepare_attention_mask = __patch__LTX2Attention_prepare_attention_mask__  # type: ignore[assignment]
    LTX2AudioVideoAttnProcessor.__call__ = __patch__LTX2AudioVideoAttnProcessor__call__  # type: ignore[assignment]

    rope_type = getattr(getattr(transformer, "config", None), "rope_type", "interleaved")
    if rope_type == "split":
      # split RoPE returns (B, H, T, D/2), shard along T dim
      rope_expected_dims = 4
      rope_split_dim = 2
    else:
      # interleaved RoPE returns (B, T, D), shard along T dim
      rope_expected_dims = 3
      rope_split_dim = 1

    _cp_plan: _ContextParallelModelPlan = {
      "": {
        # Shard video/audio latents across sequence
        "hidden_states":
        _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        "audio_hidden_states":
        _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        # Shard prompt embeds across sequence
        "encoder_hidden_states":
        _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        "audio_encoder_hidden_states":
        _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        # IMPORTANT: shard video timestep (B, seq_len) to match sharded hidden_states
        "timestep":
        _ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
        # NOTE: do NOT shard attention masks; handled in patched attention processor
      },
      # Split RoPE outputs to match CP-sharded sequence length
      "rope": {
        0:
        _ContextParallelInput(split_dim=rope_split_dim,
                              expected_dims=rope_expected_dims,
                              split_output=True),
        1:
        _ContextParallelInput(split_dim=rope_split_dim,
                              expected_dims=rope_expected_dims,
                              split_output=True),
      },
      "audio_rope": {
        0:
        _ContextParallelInput(split_dim=rope_split_dim,
                              expected_dims=rope_expected_dims,
                              split_output=True),
        1:
        _ContextParallelInput(split_dim=rope_split_dim,
                              expected_dims=rope_expected_dims,
                              split_output=True),
      },
      "cross_attn_rope": {
        0:
        _ContextParallelInput(split_dim=rope_split_dim,
                              expected_dims=rope_expected_dims,
                              split_output=True),
        1:
        _ContextParallelInput(split_dim=rope_split_dim,
                              expected_dims=rope_expected_dims,
                              split_output=True),
      },
      "cross_attn_audio_rope": {
        0:
        _ContextParallelInput(split_dim=rope_split_dim,
                              expected_dims=rope_expected_dims,
                              split_output=True),
        1:
        _ContextParallelInput(split_dim=rope_split_dim,
                              expected_dims=rope_expected_dims,
                              split_output=True),
      },
      # Gather outputs before returning
      "proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
      "audio_proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
    }

    return _cp_plan


# Upstream references for future syncs:
# - https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_ltx2.py
# - https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py


@functools.wraps(LTX2Attention.prepare_attention_mask)
def __patch__LTX2Attention_prepare_attention_mask__(
  self: LTX2Attention,
  attention_mask: torch.Tensor,
  target_length: int,
  batch_size: int,
  out_dim: int = 3,
  # NOTE: Allow specifying head_size for CP
  head_size: Optional[int] = None,
) -> torch.Tensor:
  # Relative to upstream diffusers, this helper accepts an explicit `head_size`. Under Context
  # Parallelism each rank owns only `attn.heads // world_size` heads, so repeating the mask with the
  # full `self.heads` would break the sharded attention shape contract.
  if head_size is None:
    head_size = self.heads
  if attention_mask is None:
    return attention_mask

  current_length: int = attention_mask.shape[-1]
  if current_length != target_length:
    if attention_mask.device.type == "mps":
      padding_shape = (attention_mask.shape[0], attention_mask.shape[1], target_length)
      padding = torch.zeros(padding_shape, dtype=attention_mask.dtype, device=attention_mask.device)
      attention_mask = torch.cat([attention_mask, padding], dim=2)
    else:
      attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

  if out_dim == 3:
    if attention_mask.shape[0] < batch_size * head_size:
      attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
  elif out_dim == 4:
    attention_mask = attention_mask.unsqueeze(1)
    attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

  return attention_mask


@functools.wraps(LTX2AudioVideoAttnProcessor.__call__)
def __patch__LTX2AudioVideoAttnProcessor__call__(
  self: LTX2AudioVideoAttnProcessor,
  attn: "LTX2Attention",
  hidden_states: torch.Tensor,
  encoder_hidden_states: Optional[torch.Tensor] = None,
  attention_mask: Optional[torch.Tensor] = None,
  query_rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
  key_rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
  # Diffusers prepares the attention mask from the local `sequence_length` and reshapes it with the
  # full `attn.heads`. Under Context Parallelism the sequence is sharded across ranks, while the
  # mask usually still represents the global sequence. This patch therefore expands
  # `target_length` back to the global sequence length and repeats the mask with
  # `attn.heads // world_size`.
  batch_size, sequence_length, _ = (hidden_states.shape if encoder_hidden_states is None else
                                    encoder_hidden_states.shape)

  if attention_mask is not None:
    if self._cp_config is None:
      attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
      attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
    else:
      cp_config = self._cp_config
      if cp_config is not None and cp_config._world_size > 1:
        head_size = attn.heads // cp_config._world_size
        attention_mask = attn.prepare_attention_mask(
          attention_mask,
          sequence_length * cp_config._world_size,
          batch_size,
          3,
          head_size,
        )
        attention_mask = attention_mask.view(batch_size, head_size, -1, attention_mask.shape[-1])
      else:
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

  if encoder_hidden_states is None:
    encoder_hidden_states = hidden_states

  query = attn.to_q(hidden_states)
  key = attn.to_k(encoder_hidden_states)
  value = attn.to_v(encoder_hidden_states)

  query = attn.norm_q(query)
  key = attn.norm_k(key)

  if query_rotary_emb is not None:
    # Keep RoPE logic identical to upstream: for v2a/a2v cross-attn, K can use separate RoPE.
    if attn.rope_type == "interleaved":
      query = apply_interleaved_rotary_emb(query, query_rotary_emb)
      key = apply_interleaved_rotary_emb(
        key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb)
    elif attn.rope_type == "split":
      query = apply_split_rotary_emb(query, query_rotary_emb)
      key = apply_split_rotary_emb(
        key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb)

  query = query.unflatten(2, (attn.heads, -1))
  key = key.unflatten(2, (attn.heads, -1))
  value = value.unflatten(2, (attn.heads, -1))

  hidden_states = _dispatch_attention_fn(
    query,
    key,
    value,
    attn_mask=attention_mask,
    dropout_p=0.0,
    is_causal=False,
    backend=self._attention_backend,
    cp_config=self._cp_config,
  )
  hidden_states = hidden_states.flatten(2, 3)
  hidden_states = hidden_states.to(query.dtype)

  hidden_states = attn.to_out[0](hidden_states)
  hidden_states = attn.to_out[1](hidden_states)
  return hidden_states
