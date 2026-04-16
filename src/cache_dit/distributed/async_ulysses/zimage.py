import functools
from typing import Optional

import torch
from diffusers.models.transformers.transformer_z_image import Attention, ZSingleStreamAttnProcessor

from ...attention import _dispatch_attention_fn
from ...platforms import current_platform
from .registry import AsyncUlyssesPlanner, AsyncUlyssesRegistry, MethodPatchSpec
from .common import require_cp_config
from ..core import _All2AllComm

__all__ = ["ZImageAsyncUlyssesPlanner"]


@AsyncUlyssesRegistry.register(("ZImage", "ZImageTransformer2DModel", "Lumina2"))
class ZImageAsyncUlyssesPlanner(AsyncUlyssesPlanner):

  @classmethod
  def get_method_patches(cls) -> tuple[MethodPatchSpec, ...]:
    return (MethodPatchSpec(ZSingleStreamAttnProcessor, "__call__", cls._build_zimage_attn_patch), )

  @staticmethod
  def _apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast(current_platform.device_type, enabled=False):
      x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
      freqs_cis = freqs_cis.unsqueeze(2)
      x_out = torch.view_as_real(x * freqs_cis).flatten(3)
      return x_out.type_as(x_in)

  @staticmethod
  def _async_ulysses_attn_zimage(
    self: ZSingleStreamAttnProcessor,
    attn: Attention,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    freqs_cis: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    del encoder_hidden_states
    cp_config = require_cp_config(self, "ZSingleStreamAttnProcessor")
    dtype = hidden_states.dtype

    query = attn.to_q(hidden_states)
    query = query.unflatten(-1, (attn.heads, -1))
    if attn.norm_q is not None:
      query = attn.norm_q(query)
    if freqs_cis is not None:
      query = ZImageAsyncUlyssesPlanner._apply_rotary_emb(query, freqs_cis)

    comm = _All2AllComm(cp_config)
    query_wait = comm.send_q(query)

    key = attn.to_k(hidden_states)
    key = key.unflatten(-1, (attn.heads, -1))
    if attn.norm_k is not None:
      key = attn.norm_k(key)
    if freqs_cis is not None:
      key = ZImageAsyncUlyssesPlanner._apply_rotary_emb(key, freqs_cis)
    key_wait = comm.send_k(key)

    value = attn.to_v(hidden_states)
    value = value.unflatten(-1, (attn.heads, -1))
    value_wait = comm.send_v(value)

    query = query_wait.wait()
    key = key_wait.wait()
    value = value_wait.wait()
    query, key = query.to(dtype), key.to(dtype)

    if attention_mask is not None and attention_mask.ndim == 2:
      attention_mask = attention_mask[:, None, None, :]

    out = _dispatch_attention_fn(
      query,
      key,
      value,
      attn_mask=attention_mask,
      dropout_p=0.0,
      is_causal=False,
      backend=self._attention_backend,
      cp_config=None,
    )
    hidden_states = comm.send_o(out).wait()
    hidden_states = hidden_states.flatten(2, 3)
    hidden_states = hidden_states.to(dtype)

    output = attn.to_out[0](hidden_states)
    if len(attn.to_out) > 1:
      output = attn.to_out[1](output)
    return output

  @staticmethod
  def _build_zimage_attn_patch(original):

    @functools.wraps(original)
    def wrapper(self,
                attn,
                hidden_states,
                encoder_hidden_states=None,
                attention_mask=None,
                freqs_cis=None):
      cp_config = getattr(self, "_cp_config", None)
      if cp_config is not None and cp_config.ulysses_degree > 1:
        return ZImageAsyncUlyssesPlanner._async_ulysses_attn_zimage(
          self,
          attn,
          hidden_states,
          encoder_hidden_states,
          attention_mask,
          freqs_cis,
        )
      return original(self, attn, hidden_states, encoder_hidden_states, attention_mask, freqs_cis)

    return wrapper
