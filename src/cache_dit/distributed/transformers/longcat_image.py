from typing import Dict, List, Optional, Tuple

import torch
from diffusers.models.modeling_utils import ModelMixin

try:
  from diffusers.models.transformers.transformer_longcat_image import (
    LongCatImageSingleTransformerBlock, )  # requires diffusers>=0.37.0.dev0

  _longcat_image_is_available = True
except ImportError:
  _longcat_image_is_available = False
from einops import rearrange
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed._tensor import Replicate
from torch.distributed.tensor.parallel import (
  ColwiseParallel,
  ParallelStyle,
  RowwiseParallel,
  parallelize_module,
)

from ...distributed.core import (
  _ContextParallelInput,
  _ContextParallelModelPlan,
  _ContextParallelOutput,
)
from ...logger import init_logger
from ..async_ulysses import AsyncUlyssesRegistry
from ..config import ParallelismConfig
from ..utils import shard_div_attr
from .register import (
  ContextParallelismPlanner,
  ContextParallelismPlannerRegister,
  TensorParallelismPlanner,
  TensorParallelismPlannerRegister,
)

logger = init_logger(__name__)


@ContextParallelismPlannerRegister.register("LongCatImageTransformer2DModel")
class LongCatImageContextParallelismPlanner(ContextParallelismPlanner):

  def _apply(
    self,
    transformer: Optional[torch.nn.Module | ModelMixin] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:

    if not _longcat_image_is_available:
      logger.warning(
        "Diffusers LongCatImageTransformer2DModel or related classes are not found. "
        "Please install diffusers>=0.37.0.dev0 from source. Skipping CP plan for LongCatImage.")
      return transformer

    if parallelism_config.ulysses_async and transformer is not None:
      AsyncUlyssesRegistry.apply(transformer)

    _cp_plan = {
      "": {
        "hidden_states":
        _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        "encoder_hidden_states":
        _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        "img_ids":
        _ContextParallelInput(split_dim=0, expected_dims=2, split_output=False),
        "txt_ids":
        _ContextParallelInput(split_dim=0, expected_dims=2, split_output=False),
      },
      "proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
    }
    return _cp_plan


@TensorParallelismPlannerRegister.register("LongCatImageTransformer2DModel")
class LongCatImageTensorParallelismPlanner(TensorParallelismPlanner):

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
    layer_plans = []
    for _, block in transformer.transformer_blocks.named_children():
      shard_div_attr(block.attn, "heads", tp_mesh.size())
      layer_plan = {
        "attn.to_q": ColwiseParallel(),
        "attn.to_k": ColwiseParallel(),
        "attn.to_v": ColwiseParallel(),
        "attn.to_out.0": RowwiseParallel(),
        "ff.net.0.proj": ColwiseParallel(),
        "ff.net.2": RowwiseParallel(),
        "attn.add_q_proj": ColwiseParallel(),
        "attn.add_k_proj": ColwiseParallel(),
        "attn.add_v_proj": ColwiseParallel(),
        "attn.to_add_out": RowwiseParallel(),
        "ff_context.net.0.proj": ColwiseParallel(),
        "ff_context.net.2": RowwiseParallel(),
      }

      if getattr(block.norm1, "linear", None) is not None:
        layer_plan["norm1.linear"] = ColwiseParallel(output_layouts=Replicate())
      if getattr(block.norm1_context, "linear", None) is not None:
        layer_plan["norm1_context.linear"] = ColwiseParallel(output_layouts=Replicate())
      parallelize_module(
        module=block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_plan,
      )
      layer_plans.append(layer_plan)

    # NOTE: special handling for LongCatImageSingleTransformerBlock, we have to
    # rearrange the proj_out weight because it contains both out and down
    # projection weights in a single matrix.
    def rearrange_proj_out_weight(single_block: LongCatImageSingleTransformerBlock, tp_group_size):
      # rowwise
      hidden_dim = single_block.attn.to_q.weight.shape[0]
      requires_grad = single_block.proj_out.weight.requires_grad
      linear2_weight_data = single_block.proj_out.weight.data.T.detach().clone()
      out_weight = linear2_weight_data[:hidden_dim, ...]
      out_weight = rearrange(out_weight, "(G D) C -> G D C", G=tp_group_size)
      down_weight = linear2_weight_data.data[hidden_dim:, ...]
      down_weight = rearrange(down_weight, "(G D) C -> G D C", G=tp_group_size)
      new_linear2_weight = torch.cat([out_weight, down_weight], dim=1)
      new_linear2_weight = rearrange(new_linear2_weight, "G D C -> (G D) C")
      single_block.proj_out.weight.data.copy_(new_linear2_weight.T)
      single_block.proj_out.weight.requires_grad_(requires_grad)

    for _, block in transformer.single_transformer_blocks.named_children():
      rearrange_proj_out_weight(block, tp_mesh.size())
      shard_div_attr(block.attn, "heads", tp_mesh.size())
      layer_plan = {
        "attn.to_q": ColwiseParallel(),
        "attn.to_k": ColwiseParallel(),
        "attn.to_v": ColwiseParallel(),
        "proj_mlp": ColwiseParallel(),
        "proj_out": RowwiseParallel(),
      }
      if getattr(block.norm, "linear", None) is not None:
        layer_plan["norm.linear"] = ColwiseParallel(output_layouts=Replicate())
      parallelize_module(
        module=block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_plan,
      )
      layer_plans.append(layer_plan)
    return transformer, layer_plans
