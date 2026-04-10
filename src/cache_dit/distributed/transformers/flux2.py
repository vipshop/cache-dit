import functools
from typing import Dict, List, Optional, Tuple

import torch
from diffusers import Flux2Transformer2DModel
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.transformer_flux2 import (
  Flux2AttnProcessor,
  Flux2Attention,
  Flux2KVAttnProcessor,
  Flux2KVParallelSelfAttnProcessor,
  Flux2ParallelSelfAttnProcessor,
  Flux2ParallelSelfAttention,
  Flux2SingleTransformerBlock,
  Flux2TransformerBlock,
  apply_rotary_emb,
)
from einops import rearrange
from torch.distributed import DeviceMesh
from torch.distributed.tensor.parallel import (
  ColwiseParallel,
  ParallelStyle,
  RowwiseParallel,
  parallelize_module,
)

from ...attention import _dispatch_attention_fn
from ...distributed import (
  _All2AllComm,
  _ContextParallelInput,
  _ContextParallelModelPlan,
  _ContextParallelOutput,
)
from ...logger import init_logger
from ...platforms import current_platform
from ..config import ParallelismConfig
from ..utils import maybe_empty_cache, shard_div_attr
from .register import (
  ContextParallelismPlanner,
  ContextParallelismPlannerRegister,
  TensorParallelismPlanner,
  TensorParallelismPlannerRegister,
)

logger = init_logger(__name__)


@ContextParallelismPlannerRegister.register("Flux2Transformer2DModel")
class Flux2ContextParallelismPlanner(ContextParallelismPlanner):

  def _apply(
    self,
    transformer: Optional[torch.nn.Module | ModelMixin] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:

    self._cp_planner_preferred_native_diffusers = False

    if parallelism_config.ulysses_async:
      assert not _is_klein_kv(transformer), "Async Ulysses is not supported for Klein-KV now."
      Flux2AttnProcessor.__call__ = __patch_flux2_attn_processor__
      Flux2ParallelSelfAttnProcessor.__call__ = __patch_flux2_self_attn_processor__

    if transformer is not None and self._cp_planner_preferred_native_diffusers:
      assert isinstance(
        transformer,
        Flux2Transformer2DModel), "Transformer must be an instance of Flux2Transformer2DModel"
      if hasattr(transformer, "_cp_plan"):
        if transformer._cp_plan is not None:
          return transformer._cp_plan

    # Use custom CP plan in cache-dit for better control and flexibility.
    if not _is_klein_kv(transformer):
      _cp_plan = {
        "": {
          "hidden_states":
          _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
          "encoder_hidden_states":
          _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
          "img_ids":
          _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
          "txt_ids":
          _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
      }
    else:
      num_double_blocks = len(transformer.transformer_blocks)
      num_single_blocks = len(transformer.single_transformer_blocks)
      _cp_plan = {
        "pos_embed": {
          "ids": _ContextParallelInput(split_dim=0, expected_dims=2, split_output=False)
        },
        "transformer_blocks.0": {
          "hidden_states":
          _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
          "encoder_hidden_states":
          _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "transformer_blocks.*": {
          # Will auto skip splitting in cached mode (the dim of temb_mod_img is 2)
          "temb_mod_img": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        f"transformer_blocks.{num_double_blocks - 1}": (
          _ContextParallelOutput(gather_dim=1, expected_dims=3),  # Gather encoder hidden states
          _ContextParallelOutput(gather_dim=1, expected_dims=3),  # Gather hidden states
        ),
        "single_transformer_blocks.0": {
          "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "single_transformer_blocks.*": {
          # Will auto skip splitting in cached mode (the dim of temb_mod is 2)
          "temb_mod": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        f"single_transformer_blocks.{num_single_blocks - 1}":
        _ContextParallelOutput(gather_dim=1, expected_dims=3),
        "norm_out": {
          "x": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False)
        },
        "proj_out":
        _ContextParallelOutput(gather_dim=1, expected_dims=3),
      }
    return _cp_plan


def _is_klein_kv(transformer: Flux2Transformer2DModel) -> bool:
  attn: Flux2Attention = transformer.transformer_blocks[0].attn
  return isinstance(attn.processor, Flux2KVAttnProcessor) or isinstance(
    attn.processor, Flux2KVParallelSelfAttnProcessor)


# Implements async Ulysses communication for Attention module when context parallelism
# is enabled with Ulysses degree > 1. The async communication allows overlapping
# communication with computation for better performance.
def _async_ulysses_attn_flux2(
  self: Flux2AttnProcessor,
  attn: "Flux2Attention",
  hidden_states: torch.Tensor,
  encoder_hidden_states: torch.Tensor = None,
  attention_mask: torch.Tensor | None = None,
  image_rotary_emb: torch.Tensor | None = None,
) -> torch.Tensor:
  # Manually expand _get_qkv_projections to support async Ulysses communication.
  cp_config = getattr(self, "_cp_config", None)
  if cp_config is None:
    raise RuntimeError("Flux2AttnProcessor is missing _cp_config during async Ulysses attention.")

  value = attn.to_v(hidden_states)  # type: torch.Tensor
  value = value.unflatten(-1, (attn.heads, -1))
  if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
    encoder_value = attn.add_v_proj(encoder_hidden_states)  # type: torch.Tensor
    encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
    value = torch.cat([encoder_value, value], dim=1)

  comm = _All2AllComm(cp_config).init_meta(value)

  # Async all to all for value
  value_wait = comm.send_v(value)

  query = attn.to_q(hidden_states)  # type: torch.Tensor
  query = query.unflatten(-1, (attn.heads, -1))
  query = attn.norm_q(query)
  if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
    encoder_query = attn.add_q_proj(encoder_hidden_states)  # type: torch.Tensor
    encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
    encoder_query = attn.norm_added_q(encoder_query)
    query = torch.cat([encoder_query, query], dim=1)
  if image_rotary_emb is not None:
    query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)

  # Async all to all for query
  query_wait = comm.send_q(query)

  key = attn.to_k(hidden_states)  # type: torch.Tensor
  key = key.unflatten(-1, (attn.heads, -1))
  key = attn.norm_k(key)

  if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
    encoder_key = attn.add_k_proj(encoder_hidden_states)  # type: torch.Tensor
    encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
    encoder_key = attn.norm_added_k(encoder_key)
    key = torch.cat([encoder_key, key], dim=1)
  if image_rotary_emb is not None:
    key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
  # Async all to all for key
  key_wait = comm.send_k(key)

  # Ensure the query, key, value are ready
  value = value_wait.wait()
  query = query_wait.wait()
  key = key_wait.wait()

  out = _dispatch_attention_fn(
    query,
    key,
    value,
    attn_mask=attention_mask,
    backend=self._attention_backend,
    cp_config=None,  # set to None to avoid double parallelism
  )  # (B, S_GLOBAL, H_LOCAL, D)
  # Must be sync all to all for out when encoder_hidden_states is used
  out_wait = comm.send_o(out)  # (B, S_LOCAL, H_GLOBAL, D)
  out = out_wait.wait()  # type: torch.Tensor

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

  if encoder_hidden_states is not None:
    return hidden_states, encoder_hidden_states
  else:
    return hidden_states


flux2_attn_processor__call__ = Flux2AttnProcessor.__call__


@functools.wraps(flux2_attn_processor__call__)
def __patch_flux2_attn_processor__(
  self: Flux2AttnProcessor,
  attn: "Flux2Attention",
  hidden_states: torch.Tensor,
  encoder_hidden_states: torch.Tensor = None,
  attention_mask: torch.Tensor | None = None,
  image_rotary_emb: torch.Tensor | None = None,
) -> torch.Tensor:
  cp_config = getattr(self, "_cp_config", None)
  if cp_config is not None and cp_config.ulysses_degree > 1:
    return _async_ulysses_attn_flux2(
      self,
      attn,
      hidden_states,
      encoder_hidden_states=encoder_hidden_states,
      attention_mask=attention_mask,
      image_rotary_emb=image_rotary_emb,
    )
  else:
    return flux2_attn_processor__call__(
      self,
      attn,
      hidden_states,
      encoder_hidden_states=encoder_hidden_states,
      attention_mask=attention_mask,
      image_rotary_emb=image_rotary_emb,
    )


def _async_ulysses_self_attn_flux2(
  self: Flux2ParallelSelfAttnProcessor,
  attn: "Flux2ParallelSelfAttention",
  hidden_states: torch.Tensor,
  attention_mask: torch.Tensor | None = None,
  image_rotary_emb: torch.Tensor | None = None,
) -> torch.Tensor:
  cp_config = getattr(self, "_cp_config", None)
  if cp_config is None:
    raise RuntimeError(
      "Flux2ParallelSelfAttnProcessor is missing _cp_config during async Ulysses attention.")

  # Parallel in (QKV + MLP in) projection
  hidden_states = attn.to_qkv_mlp_proj(hidden_states)
  qkv, mlp_hidden_states = torch.split(
    hidden_states, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1)

  # Handle the attention logic
  query, key, value = qkv.chunk(3, dim=-1)

  value = value.unflatten(-1, (attn.heads, -1))
  comm = _All2AllComm(cp_config).init_meta(value)
  # Async all to all for value
  value_wait = comm.send_v(value)

  query = query.unflatten(-1, (attn.heads, -1))  # type: torch.Tensor
  query = attn.norm_q(query)
  if image_rotary_emb is not None:
    query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
  # Async all to all for query
  query_wait = comm.send_q(query)

  key = key.unflatten(-1, (attn.heads, -1))
  key = attn.norm_k(key)
  if image_rotary_emb is not None:
    key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
  # Async all to all for key
  key_wait = comm.send_k(key)

  # Ensure the query, key, value are ready
  value = value_wait.wait()
  query = query_wait.wait()
  key = key_wait.wait()

  out = _dispatch_attention_fn(
    query,
    key,
    value,
    attn_mask=attention_mask,
    backend=self._attention_backend,
    cp_config=None,  # set to None to avoid double parallelism
  )  # (B, S_GLOBAL, H_LOCAL, D)
  # Must be sync all to all for out when encoder_hidden_states is used
  out_wait = comm.send_o(out)  # (B, S_LOCAL, H_GLOBAL, D)

  # Handle the feedforward (FF) logic, overlap with attention output communication
  mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)

  out = out_wait.wait()  # type: torch.Tensor

  hidden_states = out.flatten(2, 3)
  hidden_states = hidden_states.to(query.dtype)

  # Concatenate and parallel output projection
  hidden_states = torch.cat([hidden_states, mlp_hidden_states], dim=-1)
  hidden_states = attn.to_out(hidden_states)

  return hidden_states


flux2_self_attn_processor__call__ = Flux2ParallelSelfAttnProcessor.__call__


@functools.wraps(flux2_self_attn_processor__call__)
def __patch_flux2_self_attn_processor__(
  self: Flux2ParallelSelfAttnProcessor,
  attn: "Flux2ParallelSelfAttention",
  hidden_states: torch.Tensor,
  attention_mask: torch.Tensor | None = None,
  image_rotary_emb: torch.Tensor | None = None,
) -> torch.Tensor:
  cp_config = getattr(self, "_cp_config", None)
  if cp_config is not None and cp_config.ulysses_degree > 1:
    return _async_ulysses_self_attn_flux2(
      self,
      attn,
      hidden_states,
      attention_mask=attention_mask,
      image_rotary_emb=image_rotary_emb,
    )
  else:
    return flux2_self_attn_processor__call__(
      self,
      attn,
      hidden_states,
      attention_mask=attention_mask,
      image_rotary_emb=image_rotary_emb,
    )


@TensorParallelismPlannerRegister.register("Flux2Transformer")
class Flux2TensorParallelismPlanner(TensorParallelismPlanner):

  def _apply(
    self,
    transformer: torch.nn.Module,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    tp_mesh = self.mesh(parallelism_config=parallelism_config)
    transformer, layer_plans = self.parallelize_transformer(
      transformer=transformer,
      tp_mesh=tp_mesh,
    )

    return transformer, layer_plans

  @classmethod
  def rerangege_swiglu_weight(cls, weight: torch.Tensor, tp_size: int):
    weight = rearrange(weight, "r (g h d) -> r (h g d)", g=2, h=tp_size)
    return weight

  @classmethod
  def rearrange_feedforward_weight(cls, block: Flux2TransformerBlock, tp_size: int):

    block.ff.linear_in.weight.data = cls.rerangege_swiglu_weight(block.ff.linear_in.weight.data.T,
                                                                 tp_size).T
    block.ff_context.linear_in.weight.data = cls.rerangege_swiglu_weight(
      block.ff_context.linear_in.weight.data.T, tp_size).T

  @classmethod
  def rearrange_singleblock_weight(cls, block: Flux2SingleTransformerBlock, tp_size: int):
    attn = block.attn
    to_qkv_mlp_proj_weight = attn.to_qkv_mlp_proj.weight.data.T
    qkv, mlp = torch.split(
      to_qkv_mlp_proj_weight,
      [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor],
      dim=-1,
    )

    mlp = cls.rerangege_swiglu_weight(mlp, tp_size)

    def rerangege_qkv_weight(weight: torch.Tensor, tp_size: int):
      weight = rearrange(weight, "r (g h d) -> r (h g d)", g=3, h=tp_size)
      return weight

    qkv = rerangege_qkv_weight(qkv, tp_size)
    qkv = rearrange(qkv, "r (h d) -> r h d", h=tp_size)
    mlp = rearrange(mlp, "r (h d) -> r h d", h=tp_size)
    to_qkv_mlp_proj_weight = torch.cat([qkv, mlp], dim=-1)
    to_qkv_mlp_proj_weight = to_qkv_mlp_proj_weight.flatten(1)
    attn.to_qkv_mlp_proj.weight.data = to_qkv_mlp_proj_weight.T

    # rearrange out projection weight
    out_weight = attn.to_out.weight.data.T
    # FLUX.2-dev, FLUX.2-klein, divide by 4
    attn_out_dim = out_weight.shape[0] // 4
    attn_out_weight = out_weight[:attn_out_dim, ...]
    mlp_out_weight = out_weight[attn_out_dim:, ...]

    attn_out_weight = rearrange(attn_out_weight, "(g d) c -> g d c", g=tp_size)
    mlp_out_weight = rearrange(mlp_out_weight, "(g d) c -> g d c", g=tp_size)

    new_out_weight = torch.cat([attn_out_weight, mlp_out_weight], dim=1)
    new_out_weight = rearrange(new_out_weight, "g d c -> (g d) c")
    attn.to_out.weight.data = new_out_weight.T

  def parallelize_transformer(
    self,
    transformer: Flux2Transformer2DModel,
    tp_mesh: DeviceMesh,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    tp_size = tp_mesh.get_group().size()
    layer_plans = []

    for _, block in transformer.transformer_blocks.named_children():
      # moving to cuda speed up the rearrangement process significantly
      old_device = next(block.parameters()).device
      block.to(current_platform.device_type)
      self.rearrange_feedforward_weight(block, tp_size)
      block.to(old_device)
      shard_div_attr(block.attn, "heads", tp_size)
      layer_plan = {
        "attn.to_q": ColwiseParallel(),
        "attn.to_k": ColwiseParallel(),
        "attn.to_v": ColwiseParallel(),
        "attn.to_out.0": RowwiseParallel(),
        "attn.add_q_proj": ColwiseParallel(),
        "attn.add_k_proj": ColwiseParallel(),
        "attn.add_v_proj": ColwiseParallel(),
        "attn.to_add_out": RowwiseParallel(),
        "ff.linear_in": ColwiseParallel(),
        "ff.linear_out": RowwiseParallel(),
        "ff_context.linear_in": ColwiseParallel(),
        "ff_context.linear_out": RowwiseParallel(),
      }

      parallelize_module(
        module=block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_plan,
      )
      layer_plans.append(layer_plan)
    maybe_empty_cache()

    for _, block in transformer.single_transformer_blocks.named_children():
      # moving to cuda speed up the rearrangement process significantly
      old_device = next(block.parameters()).device
      block.to(current_platform.device_type)
      self.rearrange_singleblock_weight(block, tp_size)
      block.to(old_device)
      shard_div_attr(block.attn, "heads", tp_size)
      shard_div_attr(block.attn, "inner_dim", tp_size)
      shard_div_attr(block.attn, "mlp_hidden_dim", tp_size)
      layer_plan = {
        "attn.to_qkv_mlp_proj": ColwiseParallel(),
        "attn.to_out": RowwiseParallel(),
      }

      parallelize_module(
        module=block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_plan,
      )
      layer_plans.append(layer_plan)
    maybe_empty_cache()

    return transformer, layer_plans
