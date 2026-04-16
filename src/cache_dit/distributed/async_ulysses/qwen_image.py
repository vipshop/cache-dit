import functools
from typing import Optional

import torch
from diffusers.models.transformers.transformer_qwenimage import (
  Attention,
  QwenDoubleStreamAttnProcessor2_0,
  apply_rotary_emb_qwen,
)

from ...attention import _dispatch_attention_fn
from .common import require_cp_config
from .registry import AsyncUlyssesPlanner, AsyncUlyssesRegistry, MethodPatchSpec
from ..core import _All2AllComm

__all__ = ["QwenImageAsyncUlyssesPlanner"]


@AsyncUlyssesRegistry.register(("QwenImage", "QwenImageTransformer2DModel"))
class QwenImageAsyncUlyssesPlanner(AsyncUlyssesPlanner):

  @classmethod
  def get_method_patches(cls) -> tuple[MethodPatchSpec, ...]:
    return (MethodPatchSpec(QwenDoubleStreamAttnProcessor2_0, "__call__",
                            cls._build_qwen_attn_patch), )

  @staticmethod
  def _async_ulysses_attn_qwen(
    self: QwenDoubleStreamAttnProcessor2_0,
    attn: Attention,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor = None,
    encoder_hidden_states_mask: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.FloatTensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
  ) -> torch.FloatTensor:
    del encoder_hidden_states_mask
    if encoder_hidden_states is None:
      raise ValueError(
        "QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream)")

    cp_config = require_cp_config(self, "QwenDoubleStreamAttnProcessor2_0")
    seq_txt = encoder_hidden_states.shape[1]

    img_value = attn.to_v(hidden_states)
    txt_value = attn.add_v_proj(encoder_hidden_states)
    img_value = img_value.unflatten(-1, (attn.heads, -1))
    txt_value = txt_value.unflatten(-1, (attn.heads, -1))
    joint_value = torch.cat([txt_value, img_value], dim=1)

    comm = _All2AllComm(cp_config)
    joint_value_wait = comm.send_v(joint_value)

    img_query = attn.to_q(hidden_states)
    txt_query = attn.add_q_proj(encoder_hidden_states)
    img_query = img_query.unflatten(-1, (attn.heads, -1))
    txt_query = txt_query.unflatten(-1, (attn.heads, -1))
    if attn.norm_q is not None:
      img_query = attn.norm_q(img_query)
    if attn.norm_added_q is not None:
      txt_query = attn.norm_added_q(txt_query)
    if image_rotary_emb is not None:
      img_freqs, txt_freqs = image_rotary_emb
      img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
      txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
    joint_query = torch.cat([txt_query, img_query], dim=1)
    joint_query_wait = comm.send_q(joint_query)

    img_key = attn.to_k(hidden_states)
    txt_key = attn.add_k_proj(encoder_hidden_states)
    img_key = img_key.unflatten(-1, (attn.heads, -1))
    txt_key = txt_key.unflatten(-1, (attn.heads, -1))
    if attn.norm_k is not None:
      img_key = attn.norm_k(img_key)
    if attn.norm_added_k is not None:
      txt_key = attn.norm_added_k(txt_key)
    if image_rotary_emb is not None:
      img_freqs, txt_freqs = image_rotary_emb
      img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
      txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)
    joint_key = torch.cat([txt_key, img_key], dim=1)
    joint_key_wait = comm.send_k(joint_key)

    joint_value = joint_value_wait.wait()
    joint_query = joint_query_wait.wait()
    joint_key = joint_key_wait.wait()

    out = _dispatch_attention_fn(
      joint_query,
      joint_key,
      joint_value,
      attn_mask=attention_mask,
      dropout_p=0.0,
      is_causal=False,
      backend=self._attention_backend,
      cp_config=None,
    )

    joint_hidden_states = comm.send_o(out).wait()
    joint_hidden_states = joint_hidden_states.flatten(2, 3)
    joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

    txt_attn_output = joint_hidden_states[:, :seq_txt, :]
    img_attn_output = joint_hidden_states[:, seq_txt:, :]
    img_attn_output = attn.to_out[0](img_attn_output)
    if len(attn.to_out) > 1:
      img_attn_output = attn.to_out[1](img_attn_output)
    txt_attn_output = attn.to_add_out(txt_attn_output)
    return img_attn_output, txt_attn_output

  @staticmethod
  def _build_qwen_attn_patch(original):

    @functools.wraps(original)
    def wrapper(self,
                attn,
                hidden_states,
                encoder_hidden_states=None,
                encoder_hidden_states_mask=None,
                attention_mask=None,
                image_rotary_emb=None):
      cp_config = getattr(self, "_cp_config", None)
      if cp_config is not None and cp_config.ulysses_degree > 1:
        return QwenImageAsyncUlyssesPlanner._async_ulysses_attn_qwen(
          self,
          attn,
          hidden_states,
          encoder_hidden_states,
          encoder_hidden_states_mask,
          attention_mask,
          image_rotary_emb,
        )
      return original(
        self,
        attn,
        hidden_states,
        encoder_hidden_states,
        encoder_hidden_states_mask,
        attention_mask,
        image_rotary_emb,
      )

    return wrapper
