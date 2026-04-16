import functools
from typing import Optional, Tuple

import torch
from diffusers.models.modeling_utils import ModelMixin
from ...distributed.core import (
  _ContextParallelInput,
  _ContextParallelModelPlan,
  _ContextParallelOutput,
)

try:
  from diffusers.models.transformers.transformer_chronoedit import (
    WanAttention as ChronoEditWanAttention,
    WanAttnProcessor as ChronoEditWanAttnProcessor,
    _get_added_kv_projections,
    _get_qkv_projections,
  )
except ImportError as exc:
  raise ImportError(
    "ChronoEdit context parallelism requires diffusers with transformer_chronoedit support. "
    "Please install a recent diffusers version from source: \n"
    "pip3 install git+https://github.com/huggingface/diffusers.git") from exc

from ...attention import _dispatch_attention_fn
from ...logger import init_logger
from ..config import ParallelismConfig
from .register import (
  ContextParallelismPlanner,
  ContextParallelismPlannerRegister,
)

logger = init_logger(__name__)


@ContextParallelismPlannerRegister.register("ChronoEditTransformer3D")
class ChronoEditContextParallelismPlanner(ContextParallelismPlanner):

  def _apply(
    self,
    transformer: Optional[torch.nn.Module | ModelMixin] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:

    ChronoEditWanAttnProcessor.__call__ = __patch_ChronoEditWanAttnProcessor__call__
    _cp_plan = {
      "rope": {
        0: _ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
        1: _ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
      },
      "blocks.0": {
        "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
      },
      "proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
    }
    return _cp_plan


@functools.wraps(ChronoEditWanAttnProcessor.__call__)
def __patch_ChronoEditWanAttnProcessor__call__(
  self: ChronoEditWanAttnProcessor,
  attn: ChronoEditWanAttention,
  hidden_states: torch.Tensor,
  encoder_hidden_states: Optional[torch.Tensor] = None,
  attention_mask: Optional[torch.Tensor] = None,
  rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
  encoder_hidden_states_img = None
  if attn.add_k_proj is not None:
    # 512 is the context length of the text encoder, hardcoded for now
    image_context_length = encoder_hidden_states.shape[1] - 512
    encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
    encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

  query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

  query = attn.norm_q(query)
  key = attn.norm_k(key)

  query = query.unflatten(2, (attn.heads, -1))
  key = key.unflatten(2, (attn.heads, -1))
  value = value.unflatten(2, (attn.heads, -1))

  if rotary_emb is not None:

    def apply_rotary_emb(
      hidden_states: torch.Tensor,
      freqs_cos: torch.Tensor,
      freqs_sin: torch.Tensor,
    ):
      x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
      cos = freqs_cos[..., 0::2]
      sin = freqs_sin[..., 1::2]
      out = torch.empty_like(hidden_states)
      out[..., 0::2] = x1 * cos - x2 * sin
      out[..., 1::2] = x1 * sin + x2 * cos
      return out.type_as(hidden_states)

    query = apply_rotary_emb(query, *rotary_emb)
    key = apply_rotary_emb(key, *rotary_emb)

  # I2V task
  hidden_states_img = None
  if encoder_hidden_states_img is not None:
    key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
    key_img = attn.norm_added_k(key_img)

    key_img = key_img.unflatten(2, (attn.heads, -1))
    value_img = value_img.unflatten(2, (attn.heads, -1))

    hidden_states_img = _dispatch_attention_fn(
      query,
      key_img,
      value_img,
      attn_mask=None,
      dropout_p=0.0,
      is_causal=False,
      backend=self._attention_backend,
      # FIXME(DefTruth): Since the key/value in cross-attention depends
      # solely on encoder_hidden_states_img (img), the (q_chunk * k) * v
      # computation can be parallelized independently. Thus, there is
      # no need to pass the config here.
      cp_config=None,
    )
    hidden_states_img = hidden_states_img.flatten(2, 3)
    hidden_states_img = hidden_states_img.type_as(query)

  hidden_states = _dispatch_attention_fn(
    query,
    key,
    value,
    attn_mask=attention_mask,
    dropout_p=0.0,
    is_causal=False,
    backend=self._attention_backend,
    # FIXME(DefTruth): Since the key/value in cross-attention depends
    # solely on encoder_hidden_states (text), the (q_chunk * k) * v
    # computation can be parallelized independently. Thus, there is
    # no need to pass the config here.
    cp_config=(self._cp_config if encoder_hidden_states is None else None),
  )
  hidden_states = hidden_states.flatten(2, 3)
  hidden_states = hidden_states.type_as(query)

  if hidden_states_img is not None:
    hidden_states = hidden_states + hidden_states_img

  hidden_states = attn.to_out[0](hidden_states)
  hidden_states = attn.to_out[1](hidden_states)
  return hidden_states
