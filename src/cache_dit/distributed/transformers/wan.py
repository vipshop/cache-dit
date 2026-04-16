import functools
from typing import Dict, List, Optional, Tuple, Union

import torch
from diffusers.models.modeling_utils import ModelMixin

from diffusers.models.transformers.transformer_wan import (
  WanAttention,
  WanAttnProcessor,
  _get_added_kv_projections,
  _get_qkv_projections,
)
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor.parallel import (
  ColwiseParallel,
  ParallelStyle,
  RowwiseParallel,
  parallelize_module,
)

from ...attention import _dispatch_attention_fn
from ...distributed.core import (
  _ContextParallelInput,
  _ContextParallelModelPlan,
  _ContextParallelOutput,
)
from ...logger import init_logger
from ..config import ParallelismConfig
from ..utils import shard_div_attr
from .register import (
  ContextParallelismPlanner,
  ContextParallelismPlannerRegister,
  TensorParallelismPlanner,
  TensorParallelismPlannerRegister,
)

logger = init_logger(__name__)


@ContextParallelismPlannerRegister.register("WanTransformer3D")
class WanContextParallelismPlanner(ContextParallelismPlanner):

  def _apply(
    self,
    transformer: Optional[torch.nn.Module | ModelMixin] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:

    WanAttnProcessor.__call__ = __patch_WanAttnProcessor__call__
    _cp_plan = {
      "rope": {
        0: _ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
        1: _ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
      },
      "blocks.0": {
        "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
      },
      "proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
      # Wan 2.2 TI2V: https://github.com/huggingface/diffusers/pull/12562
      "": {
        "timestep": _ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
      },
    }
    return _cp_plan


@functools.wraps(WanAttnProcessor.__call__)
def __patch_WanAttnProcessor__call__(
  self: WanAttnProcessor,
  attn: WanAttention,
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


@ContextParallelismPlannerRegister.register("WanVACETransformer3D")
class WanVACEContextParallelismPlanner(ContextParallelismPlanner):

  def _apply(
    self,
    transformer: Optional[torch.nn.Module | ModelMixin] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:

    _cp_plan = {
      "rope": {
        0: _ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
        1: _ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
      },
      "vace_blocks.0": {
        "control_hidden_states":
        _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
      },
      "vace_blocks.*": {
        "hidden_states":
        _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        "encoder_hidden_states":
        _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
      },
      "blocks.0": {
        "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
      },
      "blocks.*": {
        "encoder_hidden_states":
        _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
      },
      "proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
    }
    return _cp_plan


class DistributedRMSNorm(nn.Module):

  def __init__(
    self,
    tp_mesh: DeviceMesh,
    normalized_shape: Union[int, list[int], torch.Size],
    eps: Optional[float],
    elementwise_affine: bool,
    weight: torch.nn.parameter.Parameter,
  ):
    super().__init__()
    self.tp_mesh = tp_mesh
    self.elementwise_affine = elementwise_affine
    self.normalized_shape = normalized_shape
    self.eps = eps
    if self.elementwise_affine:
      assert weight is not None
    self.weight = weight

  @classmethod
  def from_rmsnorm(cls, tp_mesh: DeviceMesh, rmsnorm: nn.RMSNorm):
    if not isinstance(rmsnorm, int):
      assert len(rmsnorm.normalized_shape) == 1

    if rmsnorm.weight is not None:
      tp_size = tp_mesh.get_group().size()
      tp_rank = tp_mesh.get_group().rank()
      weight = rmsnorm.weight.chunk(tp_size, dim=0)[tp_rank]
    else:
      weight = None
    norm = cls(
      tp_mesh=tp_mesh,
      normalized_shape=rmsnorm.normalized_shape,
      eps=rmsnorm.eps,
      elementwise_affine=rmsnorm.elementwise_affine,
      weight=weight,
    )
    return norm

  def forward(self, x):
    if self.elementwise_affine:
      assert x.shape[-1] == self.weight.shape[0]
    mean_square = torch.mean(x * x, dim=-1, keepdim=True)
    torch.distributed.all_reduce(
      mean_square,
      op=torch.distributed.ReduceOp.AVG,
      group=self.tp_mesh.get_group(),
    )
    root_mean_square = torch.sqrt(mean_square + self.eps)
    x_normed = x / root_mean_square
    if self.elementwise_affine:
      x_normed = x_normed * self.weight.to(device=x.device)
    assert x_normed.device.type != "cpu"
    return x_normed


@TensorParallelismPlannerRegister.register("ChronoEdit")
@TensorParallelismPlannerRegister.register("Wan")
class WanTensorParallelismPlanner(TensorParallelismPlanner):

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

  def parallelize_transformer(
    self,
    transformer: nn.Module,
    tp_mesh: DeviceMesh,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:

    def prepare_block(block: nn.Module):
      tp_size = tp_mesh.size()
      shard_div_attr(block.attn1, "heads", tp_size)
      shard_div_attr(block.attn2, "heads", tp_size)
      layer_plan = {
        "attn1.to_q": ColwiseParallel(),
        "attn1.to_k": ColwiseParallel(),
        "attn1.to_v": ColwiseParallel(),
        "attn1.to_out.0": RowwiseParallel(),
        "attn2.to_q": ColwiseParallel(),
        "attn2.to_k": ColwiseParallel(),
        "attn2.to_v": ColwiseParallel(),
        "attn2.to_out.0": RowwiseParallel(),
        "ffn.net.0.proj": ColwiseParallel(),
        "ffn.net.2": RowwiseParallel(),
      }
      if getattr(block.attn2, "add_k_proj", None):
        layer_plan["attn2.add_k_proj"] = ColwiseParallel()
      if getattr(block.attn2, "add_v_proj", None):
        layer_plan["attn2.add_v_proj"] = ColwiseParallel()
      parallelize_module(
        module=block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_plan,
      )

      block.attn1.norm_q = DistributedRMSNorm.from_rmsnorm(tp_mesh, block.attn1.norm_q)
      block.attn1.norm_k = DistributedRMSNorm.from_rmsnorm(tp_mesh, block.attn1.norm_k)
      block.attn2.norm_q = DistributedRMSNorm.from_rmsnorm(tp_mesh, block.attn2.norm_q)
      block.attn2.norm_k = DistributedRMSNorm.from_rmsnorm(tp_mesh, block.attn2.norm_k)
      if getattr(block.attn2, "norm_added_k", None):
        block.attn2.norm_added_k = DistributedRMSNorm.from_rmsnorm(tp_mesh,
                                                                   block.attn2.norm_added_k)
      return layer_plan

    layer_plans = []
    for _, block in transformer.blocks.named_children():
      layer_plans.append(prepare_block(block))

    if hasattr(transformer, "vace_blocks"):
      for _, block in transformer.vace_blocks.named_children():
        layer_plans.append(prepare_block(block))
    return transformer, layer_plans
