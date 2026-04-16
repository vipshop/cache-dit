import functools
from typing import Any, Dict, Optional, Tuple

try:
  import torch
  from diffusers.models.transformers.transformer_longcat_image import (
    LongCatImageAttention,
    LongCatImageAttnProcessor,
    LongCatImageSingleTransformerBlock,
    apply_rotary_emb,
  )
except ImportError:
  _longcat_available = False
else:
  _longcat_available = True

from ...attention import _dispatch_attention_fn
from .common import maybe_wait, require_cp_config
from .registry import AsyncUlyssesPlanner, AsyncUlyssesRegistry, MethodPatchSpec
from ..core import _All2AllComm

__all__ = ["LongCatImageAsyncUlyssesPlanner"]

if _longcat_available:

  @AsyncUlyssesRegistry.register("LongCatImageTransformer2DModel")
  class LongCatImageAsyncUlyssesPlanner(AsyncUlyssesPlanner):

    @classmethod
    def get_method_patches(cls) -> tuple[MethodPatchSpec, ...]:
      return (
        MethodPatchSpec(LongCatImageAttnProcessor, "__call__", cls._build_longcat_attn_patch),
        MethodPatchSpec(LongCatImageSingleTransformerBlock, "forward",
                        cls._build_longcat_single_block_patch),
      )

    @staticmethod
    def _async_ulysses_attn_longcat(
      self: LongCatImageAttnProcessor,
      attn: LongCatImageAttention,
      hidden_states: torch.Tensor,
      encoder_hidden_states: torch.Tensor = None,
      attention_mask: Optional[torch.Tensor] = None,
      image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
      cp_config = require_cp_config(self, "LongCatImageAttnProcessor")

      value = attn.to_v(hidden_states)
      value = value.unflatten(-1, (attn.heads, -1))
      if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_value = attn.add_v_proj(encoder_hidden_states)
        encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
        value = torch.cat([encoder_value, value], dim=1)

      comm = _All2AllComm(cp_config)
      value_wait = comm.send_v(value)

      query = attn.to_q(hidden_states)
      query = query.unflatten(-1, (attn.heads, -1))
      query = attn.norm_q(query)
      if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
        encoder_query = attn.norm_added_q(encoder_query)
        query = torch.cat([encoder_query, query], dim=1)
      if image_rotary_emb is not None:
        query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
      query_wait = comm.send_q(query)

      key = attn.to_k(hidden_states)
      key = key.unflatten(-1, (attn.heads, -1))
      key = attn.norm_k(key)
      if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
        encoder_key = attn.norm_added_k(encoder_key)
        key = torch.cat([encoder_key, key], dim=1)
      if image_rotary_emb is not None:
        key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
      key_wait = comm.send_k(key)

      value = value_wait.wait()
      query = query_wait.wait()
      key = key_wait.wait()

      out = _dispatch_attention_fn(
        query,
        key,
        value,
        attn_mask=attention_mask,
        backend=self._attention_backend,
        cp_config=None,
      )

      if encoder_hidden_states is not None:
        out = comm.send_o(out).wait()
        hidden_states = out.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)
        encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
          [
            encoder_hidden_states.shape[1],
            hidden_states.shape[1] - encoder_hidden_states.shape[1],
          ],
          dim=1,
        )
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        return hidden_states, encoder_hidden_states

      return comm.send_o(out)

    @staticmethod
    def _build_longcat_attn_patch(original):

      @functools.wraps(original)
      def wrapper(self,
                  attn,
                  hidden_states,
                  encoder_hidden_states=None,
                  attention_mask=None,
                  image_rotary_emb=None):
        cp_config = getattr(self, "_cp_config", None)
        if cp_config is not None and cp_config.ulysses_degree > 1:
          return LongCatImageAsyncUlyssesPlanner._async_ulysses_attn_longcat(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
          )
        return original(
          self,
          attn,
          hidden_states,
          encoder_hidden_states=encoder_hidden_states,
          attention_mask=attention_mask,
          image_rotary_emb=image_rotary_emb,
        )

      return wrapper

    @staticmethod
    def _build_longcat_single_block_patch(original):

      @functools.wraps(original)
      def wrapper(
        self: LongCatImageSingleTransformerBlock,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
      ) -> Tuple[torch.Tensor, torch.Tensor]:
        text_seq_len = encoder_hidden_states.shape[1]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)

        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output_wait = self.attn(
          hidden_states=norm_hidden_states,
          image_rotary_emb=image_rotary_emb,
          **joint_attention_kwargs,
        )
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

        attn_output = maybe_wait(attn_output_wait)
        attn_output = attn_output.contiguous()
        if attn_output.ndim == 4:
          attn_output = attn_output.flatten(2, 3)

        hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
        gate = gate.unsqueeze(1)
        hidden_states = gate * self.proj_out(hidden_states)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
          hidden_states = hidden_states.clip(-65504, 65504)
        return hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]

      return wrapper
