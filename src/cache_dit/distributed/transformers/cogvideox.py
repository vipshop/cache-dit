import functools
from typing import Optional, Tuple

import torch
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.cogvideox_transformer_3d import (
  CogVideoXAttnProcessor2_0, )

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


@ContextParallelismPlannerRegister.register("CogVideoX")
class CogVideoXContextParallelismPlanner(ContextParallelismPlanner):

  def _apply(
    self,
    transformer: Optional[torch.nn.Module | ModelMixin] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:

    CogVideoXAttnProcessor2_0.__call__ = __patch_CogVideoXAttnProcessor2_0__call__
    # Also need to patch the parallel config and attention backend
    if not hasattr(CogVideoXAttnProcessor2_0, "_parallel_config"):
      CogVideoXAttnProcessor2_0._parallel_config = None
    if not hasattr(CogVideoXAttnProcessor2_0, "_attention_backend"):
      CogVideoXAttnProcessor2_0._attention_backend = None

    _cp_plan = {
      "transformer_blocks.0": {
        "hidden_states":
        _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        "encoder_hidden_states":
        _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
      },
      "transformer_blocks.*": {
        "image_rotary_emb": [
          _ContextParallelInput(split_dim=0, expected_dims=2, split_output=False),
          _ContextParallelInput(split_dim=0, expected_dims=2, split_output=False),
        ],
      },
      "proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
    }
    return _cp_plan


@functools.wraps(CogVideoXAttnProcessor2_0.__call__)
def __patch_CogVideoXAttnProcessor2_0__call__(
  self: CogVideoXAttnProcessor2_0,
  attn: Attention,
  hidden_states: torch.Tensor,
  encoder_hidden_states: torch.Tensor,
  attention_mask: Optional[torch.Tensor] = None,
  image_rotary_emb: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
  text_seq_length = encoder_hidden_states.size(1)

  hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

  batch_size, sequence_length, _ = hidden_states.shape

  # NOTE(DefTruth): attention mask is always None in CogVideoX
  if attention_mask is not None:
    attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
    attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

  query = attn.to_q(hidden_states)
  key = attn.to_k(hidden_states)
  value = attn.to_v(hidden_states)

  inner_dim = key.shape[-1]
  head_dim = inner_dim // attn.heads

  # NOTE(DefTruth): no transpose
  query = query.view(batch_size, -1, attn.heads, head_dim)
  key = key.view(batch_size, -1, attn.heads, head_dim)
  value = value.view(batch_size, -1, attn.heads, head_dim)

  if attn.norm_q is not None:
    query = attn.norm_q(query)
  if attn.norm_k is not None:
    key = attn.norm_k(key)

  # Apply RoPE if needed
  if image_rotary_emb is not None:
    query[:, text_seq_length:] = apply_rotary_emb(
      query[:, text_seq_length:],
      image_rotary_emb,
      sequence_dim=1,
    )
    if not attn.is_cross_attention:
      key[:, text_seq_length:] = apply_rotary_emb(
        key[:, text_seq_length:],
        image_rotary_emb,
        sequence_dim=1,
      )

  # NOTE(DefTruth): Apply dispatch_attention_fn instead of sdpa directly
  hidden_states = _dispatch_attention_fn(
    query,
    key,
    value,
    attn_mask=attention_mask,
    dropout_p=0.0,
    is_causal=False,
    backend=getattr(self, "_attention_backend", None),
    cp_config=getattr(self, "_cp_config", None),
  )
  hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)

  # linear proj
  hidden_states = attn.to_out[0](hidden_states)
  # dropout
  hidden_states = attn.to_out[1](hidden_states)

  encoder_hidden_states, hidden_states = hidden_states.split(
    [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1)
  return hidden_states, encoder_hidden_states
