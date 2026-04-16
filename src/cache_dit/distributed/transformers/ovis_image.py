from typing import Dict, List, Optional, Tuple

import torch
from diffusers.models.modeling_utils import ModelMixin

try:
  from diffusers import OvisImageTransformer2DModel
  from diffusers.models.transformers.transformer_ovis_image import (
    OvisImageSingleTransformerBlock,
    OvisImageTransformerBlock,
  )
except ImportError:
  raise ImportError("OvisImageTransformer2DModel requires the 'diffusers>=0.36.dev0'."
                    "Please install latest version of diffusers from source: \n"
                    "pip3 install git+https://github.com/huggingface/diffusers.git")
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


@ContextParallelismPlannerRegister.register("OvisImageTransformer2DModel")
class OvisImageContextParallelismPlanner(ContextParallelismPlanner):

  def _apply(
    self,
    transformer: Optional[torch.nn.Module | ModelMixin] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:

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


@TensorParallelismPlannerRegister.register("OvisImage")
class OvisImageTensorParallelismPlanner(TensorParallelismPlanner):

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
    assert isinstance(transformer, OvisImageTransformer2DModel)
    layer_plans = []

    for _, block in transformer.transformer_blocks.named_children():
      assert isinstance(block, OvisImageTransformerBlock)
      rearrange_ffn_0_swiglu_proj_weight(block.ff.net[0].proj, tp_mesh.size())
      rearrange_ffn_0_swiglu_proj_weight(block.ff_context.net[0].proj, tp_mesh.size())
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

    for _, block in transformer.single_transformer_blocks.named_children():
      assert isinstance(block, OvisImageSingleTransformerBlock)
      rearrange_proj_out_weight(block, tp_mesh.size())
      shard_div_attr(block.attn, "heads", tp_mesh.size())
      rearrange_proj_mlp_weight(block, tp_mesh.size())
      shard_div_attr(block, "mlp_hidden_dim", tp_mesh.size())
      # Compute order: proj_mlp, to_q, to_k, to_v, proj_out
      # proj_mlp: dim -> self.mlp_hidden_dim * 2 -> split by mlp_hidden_dim
      layer_plan = {
        "proj_mlp": ColwiseParallel(),
        "attn.to_q": ColwiseParallel(),
        "attn.to_k": ColwiseParallel(),
        "attn.to_v": ColwiseParallel(),
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


# NOTE: Special handling for OvisImageSingleTransformerBlock, we have to rearrange the
# proj_out weight because it contains both out and down projection weights in a single matrix.
def rearrange_proj_out_weight(single_block: OvisImageSingleTransformerBlock, tp_group_size):
  # Rowwise: rearrange the proj_out weight for RowwiseParallel, (M,K)x(K,N), permute at K (in_dim)
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


def rearrange_proj_mlp_weight(single_block: OvisImageSingleTransformerBlock, tp_group_size):
  # Colwise: rearrange the proj_mlp weight for ColwiseParallel, (M,K)x(K,N), permute at N (out_dim)
  # Original tensor shape: [*, Hd + Gd], where Hd = Gd (Hd and Gd have the same dimension size)
  # Linear transformation definition: y = x * A^T, where
  #   A: [out_dim, in_dim]  (transformation matrix)
  #   x: [*, in_dim]        (input tensor, * denotes arbitrary leading dimensions)
  #
  # Tensor Parallel (TP) dimension permutation logic:
  # 1. Split Hd and Gd evenly according to the TP group size (tp_group_size)
  #    - When tp_group_size=2: Split [..., Hd+Gd] into [..., (Hd/2+Gd/2) + (Hd/2+Gd/2)]
  #    - When tp_group_size=4: Split [..., Hd+Gd] into [..., (Hd/4+Gd/4)*4]
  #      Expanded form: [..., Hd/4+Gd/4 + Hd/4+Gd/4 + Hd/4+Gd/4 + Hd/4+Gd/4]
  # 2. Perform dimension permutation and rearrangement on the split tensor
  # 3. Reshape the tensor back to the original shape [..., (Hd + Gd)] finally
  mlp_hidden_dim = single_block.proj_mlp.weight.shape[0] // 2
  requires_grad = single_block.proj_mlp.weight.requires_grad
  linear1_weight_data = single_block.proj_mlp.weight.data.T.detach().clone()  # [in_dim, out_dim]
  new_linear1_weight = torch.zeros_like(linear1_weight_data)
  part1_linear1_weight_data = linear1_weight_data[..., :mlp_hidden_dim]
  part2_linear1_weight_data = linear1_weight_data[..., mlp_hidden_dim:]
  split_size = mlp_hidden_dim // tp_group_size
  for i in range(tp_group_size):
    start_idx = i * split_size
    end_idx = (i + 1) * split_size
    new_linear1_weight[..., i * 2 * split_size:(i * 2 + 1) *
                       split_size] = (part1_linear1_weight_data[..., start_idx:end_idx])
    new_linear1_weight[..., (i * 2 + 1) * split_size:(i * 2 + 2) *
                       split_size] = (part2_linear1_weight_data[..., start_idx:end_idx])

  single_block.proj_mlp.weight.data.copy_(new_linear1_weight.T)  # [out_dim, in_dim]
  single_block.proj_mlp.weight.requires_grad_(requires_grad)


# Ovis-Image use SwiGLU: self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)
# hidden_states = self.proj(hidden_states); hidden_states, gate = hidden_states.chunk(2, dim=-1)
# reference: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/activations.py#L140
def rearrange_ffn_0_swiglu_proj_weight(proj: torch.nn.Linear, tp_group_size):
  # Colwise: rearrange the proj_mlp weight for ColwiseParallel, (M,K)x(K,N), permute at N (out_dim)
  # Original tensor shape: [*, Hd + Gd], where Hd = Gd (Hd and Gd have the same dimension size)
  # Linear transformation definition: y = x * A^T, where
  #   A: [out_dim, in_dim]  (transformation matrix)
  #   x: [*, in_dim]        (input tensor, * denotes arbitrary leading dimensions)
  #
  # Tensor Parallel (TP) dimension permutation logic:
  # 1. Split Hd and Gd evenly according to the TP group size (tp_group_size)
  #    - When tp_group_size=2: Split [..., Hd+Gd] into [..., (Hd/2+Gd/2) + (Hd/2+Gd/2)]
  #    - When tp_group_size=4: Split [..., Hd+Gd] into [..., (Hd/4+Gd/4)*4]
  #      Expanded form: [..., Hd/4+Gd/4 + Hd/4+Gd/4 + Hd/4+Gd/4 + Hd/4+Gd/4]
  # 2. Perform dimension permutation and rearrangement on the split tensor
  # 3. Reshape the tensor back to the original shape [..., (Hd + Gd)] finally
  dim_out = proj.weight.shape[0] // 2
  requires_grad = proj.weight.requires_grad
  linear1_weight_data = proj.weight.data.T.detach().clone()  # [in_dim, out_dim]
  new_linear1_weight = torch.zeros_like(linear1_weight_data)
  part1_linear1_weight_data = linear1_weight_data[..., :dim_out]
  part2_linear1_weight_data = linear1_weight_data[..., dim_out:]
  split_size = dim_out // tp_group_size
  for i in range(tp_group_size):
    start_idx = i * split_size
    end_idx = (i + 1) * split_size
    new_linear1_weight[..., i * 2 * split_size:(i * 2 + 1) *
                       split_size] = (part1_linear1_weight_data[..., start_idx:end_idx])
    new_linear1_weight[..., (i * 2 + 1) * split_size:(i * 2 + 2) *
                       split_size] = (part2_linear1_weight_data[..., start_idx:end_idx])

  proj.weight.data.copy_(new_linear1_weight.T)  # [out_dim, in_dim]
  proj.weight.requires_grad_(requires_grad)
