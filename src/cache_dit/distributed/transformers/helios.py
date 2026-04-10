from typing import Dict, List, Optional, Tuple, Union

import torch
from diffusers import HeliosTransformer3DModel
from diffusers.models.modeling_utils import ModelMixin
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor.parallel import (
  ColwiseParallel,
  ParallelStyle,
  RowwiseParallel,
  parallelize_module,
)

from ...distributed import (
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


@ContextParallelismPlannerRegister.register("HeliosTransformer3DModel")
class HeliosContextParallelismPlanner(ContextParallelismPlanner):

  def _apply(
    self,
    transformer: Optional[torch.nn.Module | ModelMixin] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:

    self._cp_planner_preferred_native_diffusers = False

    if transformer is not None and self._cp_planner_preferred_native_diffusers:
      assert isinstance(
        transformer,
        HeliosTransformer3DModel), "Transformer must be an instance of HeliosTransformer3DModel"
      if hasattr(transformer, "_cp_plan"):
        if transformer._cp_plan is not None:
          return transformer._cp_plan

    # NOTE(DefTruth): This cp plan here is ugly but it works, we  will optimize it in the future.
    num_blocks = len(transformer.blocks)
    # NOTE: Due to the complex concat and split ops for history hidden states and current hidden
    # states in Helios, we have to pinned the sharding strategy at 'attn' and 'ffn' level, this
    # will lead to sub-optimal performance because of the extra all-gather and scatter communication
    # overhead, we will optimize it in the future by supporting more flexible sharding strategy.
    _cp_plan = {
      # Input split at attn level and ffn level.
      "blocks.*.attn1": {
        "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        "rotary_emb": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
      },
      "blocks.*.attn2": {
        "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
      },
      "blocks.*.ffn": {
        "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
      },
      # Output gather at attn level and ffn level.
      **{
        f"blocks.{i}.attn1": _ContextParallelOutput(gather_dim=1, expected_dims=3)
        for i in range(num_blocks)
      },
      **{
        f"blocks.{i}.attn2": _ContextParallelOutput(gather_dim=1, expected_dims=3)
        for i in range(num_blocks)
      },
      **{
        f"blocks.{i}.ffn": _ContextParallelOutput(gather_dim=1, expected_dims=3)
        for i in range(num_blocks)
      },
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


@TensorParallelismPlannerRegister.register("HeliosTransformer3DModel")
class HeliosTensorParallelismPlanner(TensorParallelismPlanner):

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
    return transformer, layer_plans
