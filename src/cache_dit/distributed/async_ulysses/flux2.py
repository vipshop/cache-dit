import functools

import torch
from diffusers.models.transformers.transformer_flux2 import (
  Flux2AttnProcessor,
  Flux2Attention,
  Flux2ParallelSelfAttnProcessor,
  Flux2ParallelSelfAttention,
  apply_rotary_emb,
)

from ...attention import _dispatch_attention_fn
from .common import require_cp_config
from .registry import AsyncUlyssesPlanner, AsyncUlyssesRegistry, MethodPatchSpec
from ..core import _All2AllComm

__all__ = ["Flux2AsyncUlyssesPlanner"]


@AsyncUlyssesRegistry.register("Flux2Transformer2DModel")
class Flux2AsyncUlyssesPlanner(AsyncUlyssesPlanner):

  @classmethod
  def get_method_patches(cls) -> tuple[MethodPatchSpec, ...]:
    return (
      MethodPatchSpec(Flux2AttnProcessor, "__call__", cls._build_flux2_attn_patch),
      MethodPatchSpec(Flux2ParallelSelfAttnProcessor, "__call__", cls._build_flux2_self_attn_patch),
    )

  @staticmethod
  def _async_ulysses_attn_flux2(
    self: Flux2AttnProcessor,
    attn: Flux2Attention,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    attention_mask: torch.Tensor | None = None,
    image_rotary_emb: torch.Tensor | None = None,
  ) -> torch.Tensor:
    cp_config = require_cp_config(self, "Flux2AttnProcessor")

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
    out_wait = comm.send_o(out)
    out = out_wait.wait()

    hidden_states = out.flatten(2, 3)
    hidden_states = hidden_states.to(query.dtype)
    if encoder_hidden_states is not None:
      encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
        [
          encoder_hidden_states.shape[1],
          hidden_states.shape[1] - encoder_hidden_states.shape[1],
        ],
        dim=1,
      )
      encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)
    return (hidden_states,
            encoder_hidden_states) if encoder_hidden_states is not None else hidden_states

  @staticmethod
  def _async_ulysses_self_attn_flux2(
    self: Flux2ParallelSelfAttnProcessor,
    attn: Flux2ParallelSelfAttention,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    image_rotary_emb: torch.Tensor | None = None,
  ) -> torch.Tensor:
    cp_config = require_cp_config(self, "Flux2ParallelSelfAttnProcessor")

    hidden_states = attn.to_qkv_mlp_proj(hidden_states)
    qkv, mlp_hidden_states = torch.split(
      hidden_states,
      [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor],
      dim=-1,
    )
    query, key, value = qkv.chunk(3, dim=-1)

    value = value.unflatten(-1, (attn.heads, -1))
    comm = _All2AllComm(cp_config)
    value_wait = comm.send_v(value)

    query = query.unflatten(-1, (attn.heads, -1))
    query = attn.norm_q(query)
    if image_rotary_emb is not None:
      query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
    query_wait = comm.send_q(query)

    key = key.unflatten(-1, (attn.heads, -1))
    key = attn.norm_k(key)
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
    out_wait = comm.send_o(out)
    mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)
    out = out_wait.wait()

    hidden_states = out.flatten(2, 3)
    hidden_states = hidden_states.to(query.dtype)
    hidden_states = torch.cat([hidden_states, mlp_hidden_states], dim=-1)
    return attn.to_out(hidden_states)

  @staticmethod
  def _build_flux2_attn_patch(original):

    @functools.wraps(original)
    def wrapper(self,
                attn,
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=None,
                image_rotary_emb=None):
      cp_config = getattr(self, "_cp_config", None)
      if cp_config is not None and cp_config.ulysses_degree > 1:
        return Flux2AsyncUlyssesPlanner._async_ulysses_attn_flux2(
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
  def _build_flux2_self_attn_patch(original):

    @functools.wraps(original)
    def wrapper(self, attn, hidden_states, attention_mask=None, image_rotary_emb=None):
      cp_config = getattr(self, "_cp_config", None)
      if cp_config is not None and cp_config.ulysses_degree > 1:
        return Flux2AsyncUlyssesPlanner._async_ulysses_self_attn_flux2(
          self,
          attn,
          hidden_states,
          attention_mask=attention_mask,
          image_rotary_emb=image_rotary_emb,
        )
      return original(
        self,
        attn,
        hidden_states,
        attention_mask=attention_mask,
        image_rotary_emb=image_rotary_emb,
      )

    return wrapper
