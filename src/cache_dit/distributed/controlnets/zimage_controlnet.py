import torch
import functools
from typing import Optional
from diffusers.models.modeling_utils import ModelMixin
from diffusers import ZImageControlNetModel
from diffusers.models.controlnets.controlnet_z_image import (
  ZSingleStreamAttnProcessor,
  Attention,
)
from ...attention import _dispatch_attention_fn

from ...distributed.core import (
  _All2AllComm,
  _ContextParallelInput,
  _ContextParallelModelPlan,
  _ContextParallelOutput,
)
from .register import (
  ControlNetContextParallelismPlanner,
  ControlNetContextParallelismPlannerRegister,
  ParallelismConfig,
)
from cache_dit.platforms import current_platform

from ...logger import init_logger

logger = init_logger(__name__)


@ControlNetContextParallelismPlannerRegister.register("ZImageControlNetModel")
class ZImageControlNetContextParallelismPlanner(ControlNetContextParallelismPlanner):

  def _apply(
    self,
    controlnet: Optional[torch.nn.Module | ModelMixin] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:

    # NOTE: Diffusers native CP plan still not supported for ZImageControlNetModel
    self._cp_planner_preferred_native_diffusers = False

    if controlnet is not None and self._cp_planner_preferred_native_diffusers:
      assert isinstance(
        controlnet,
        ZImageControlNetModel), "controlnet must be an instance of ZImageControlNetModel"
      if hasattr(controlnet, "_cp_plan"):
        if controlnet._cp_plan is not None:
          return controlnet._cp_plan

    if parallelism_config.ulysses_async:
      ZSingleStreamAttnProcessor.__call__ = (
        __patch_ZSingleStreamAttnProcessor_ulysses_async__call__)

      logger.info("Enabled experimental Async QKV Projection with Ulysses style "
                  "Context Parallelism for ZImageControlNetModel.")

    # The cp plan for ZImage ControlNet is very complicated, I [HATE] it.
    n_control_layers = len(controlnet.control_layers)  # 15
    n_control_noise_refiner_layers = len(controlnet.control_noise_refiner)  # 2
    _cp_plan = {
      "control_noise_refiner.0": {
        "c": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
      },
      "control_noise_refiner.*": {
        "x": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        "freqs_cis": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
      },
      f"control_noise_refiner.{n_control_noise_refiner_layers - 1}":
      _ContextParallelOutput(gather_dim=2, expected_dims=4),
      "control_layers.0": {
        "c": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
      },
      "control_layers.*": {
        "x": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        "freqs_cis": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
      },
      f"control_layers.{n_control_layers - 1}":
      _ContextParallelOutput(gather_dim=2, expected_dims=4),
    }
    return _cp_plan


# NOTE: Support Async Ulysses QKV projection for Z-Image ControlNet
def _ulysses_attn_with_async_qkv_proj_zimage_controlnet(
  self: ZSingleStreamAttnProcessor,
  attn: Attention,
  hidden_states: torch.Tensor,
  encoder_hidden_states: Optional[torch.Tensor] = None,
  attention_mask: Optional[torch.Tensor] = None,
  freqs_cis: Optional[torch.Tensor] = None,
) -> torch.Tensor:
  cp_config = getattr(self, "_cp_config", None)
  if cp_config is None:
    raise RuntimeError(
      "ZSingleStreamAttnProcessor is missing _cp_config during async Ulysses attention.")

  # Apply RoPE
  def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast(current_platform.device_type, enabled=False):
      x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
      freqs_cis = freqs_cis.unsqueeze(2)
      x_out = torch.view_as_real(x * freqs_cis).flatten(3)
      return x_out.type_as(x_in)  # todo

  dtype = hidden_states.dtype
  query = attn.to_q(hidden_states)  # type: torch.Tensor
  query = query.unflatten(-1, (attn.heads, -1))
  if attn.norm_q is not None:  # Apply Norms
    query = attn.norm_q(query)
  if freqs_cis is not None:  # Apply RoPE
    query = apply_rotary_emb(query, freqs_cis)

  comm = _All2AllComm(cp_config)

  # Async all to all for query
  query_wait = comm.send_q(query)

  key = attn.to_k(hidden_states)  # type: torch.Tensor
  key = key.unflatten(-1, (attn.heads, -1))
  if attn.norm_k is not None:  # Apply Norms
    key = attn.norm_k(key)
  if freqs_cis is not None:  # Apply RoPE
    key = apply_rotary_emb(key, freqs_cis)

  # Async all to all for key
  key_wait = comm.send_k(key)

  value = attn.to_v(hidden_states)  # type: torch.Tensor
  value = value.unflatten(-1, (attn.heads, -1))

  # Async all to all for value
  value_wait = comm.send_v(value)

  # Ensure the query, key, value are ready
  query = query_wait.wait()
  key = key_wait.wait()
  value = value_wait.wait()

  # Cast to correct dtype
  query, key = query.to(dtype), key.to(dtype)

  # From [batch, seq_len] to [batch, 1, 1, seq_len] -> broadcast to [batch, heads, seq_len, seq_len]
  if attention_mask is not None and attention_mask.ndim == 2:
    attention_mask = attention_mask[:, None, None, :]

  # Compute joint attention
  out = _dispatch_attention_fn(
    query,
    key,
    value,
    attn_mask=attention_mask,
    dropout_p=0.0,
    is_causal=False,
    backend=self._attention_backend,
    cp_config=None,  # set to None to avoid double parallelism
  )  # (B, S_GLOBAL, H_LOCAL, D)

  out_wait = comm.send_o(out)  # (B, S_LOCAL, H_GLOBAL, D)
  hidden_states = out_wait.wait()  # type: torch.Tensor

  # Reshape back
  hidden_states = hidden_states.flatten(2, 3)
  hidden_states = hidden_states.to(dtype)

  output = attn.to_out[0](hidden_states)
  if len(attn.to_out) > 1:  # dropout
    output = attn.to_out[1](output)

  return output


ZSingleStreamAttnProcessor_original__call__ = ZSingleStreamAttnProcessor.__call__


@functools.wraps(ZSingleStreamAttnProcessor_original__call__)
def __patch_ZSingleStreamAttnProcessor_ulysses_async__call__(
  self: ZSingleStreamAttnProcessor,
  attn: Attention,
  hidden_states: torch.Tensor,
  encoder_hidden_states: Optional[torch.Tensor] = None,
  attention_mask: Optional[torch.Tensor] = None,
  freqs_cis: Optional[torch.Tensor] = None,
) -> torch.Tensor:
  cp_config = getattr(self, "_cp_config", None)
  if cp_config is not None and cp_config.ulysses_degree > 1:
    return _ulysses_attn_with_async_qkv_proj_zimage_controlnet(
      self,
      attn,
      hidden_states,
      encoder_hidden_states,
      attention_mask,
      freqs_cis,
    )
  else:
    return ZSingleStreamAttnProcessor_original__call__(
      self,
      attn,
      hidden_states,
      encoder_hidden_states,
      attention_mask,
      freqs_cis,
    )
