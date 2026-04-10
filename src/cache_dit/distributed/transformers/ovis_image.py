import functools
from typing import Any, Dict, List, Optional, Tuple

import torch
from diffusers.models.modeling_utils import ModelMixin

try:
  from diffusers import OvisImageTransformer2DModel
  from diffusers.models.transformers.transformer_ovis_image import (
    OvisImageAttention,
    OvisImageAttnProcessor,
    OvisImageSingleTransformerBlock,
    OvisImageTransformerBlock,
    apply_rotary_emb,
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

from ...attention import _dispatch_attention_fn
from ...distributed import (
  _All2AllComm,
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


@ContextParallelismPlannerRegister.register("OvisImageTransformer2DModel")
class OvisImageContextParallelismPlanner(ContextParallelismPlanner):

  def _apply(
    self,
    transformer: Optional[torch.nn.Module | ModelMixin] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:

    if parallelism_config.ulysses_async:
      OvisImageAttnProcessor.__call__ = __patch_ovis_attn_processor__
      OvisImageSingleTransformerBlock.forward = __patch_ovis_single_block__

    if transformer is not None and self._cp_planner_preferred_native_diffusers:
      assert isinstance(transformer, OvisImageTransformer2DModel
                        ), "Transformer must be an instance of OvisImageTransformer2DModel"
      if hasattr(transformer, "_cp_plan"):
        if transformer._cp_plan is not None:
          return transformer._cp_plan

    # Otherwise, use the custom CP plan defined here, this maybe
    # a little different from the native diffusers implementation
    # for some models.
    _cp_plan = {
      # Here is a Transformer level CP plan for OvisImage, which will
      # only apply the only 1 split hook (pre_forward) on the forward
      # of Transformer, and gather the output after Transformer forward.
      # Pattern of transformer forward, split_output=False:
      #     un-split input -> splited input (inside transformer)
      # Pattern of the transformer_blocks, single_transformer_blocks:
      #     splited input (previous splited output) -> to_qkv/...
      #     -> all2all
      #     -> attn (local head, full seqlen)
      #     -> all2all
      #     -> splited output
      # The `hidden_states` and `encoder_hidden_states` will still keep
      # itself splited after block forward (namely, automatic split by
      # the all2all comm op after attn) for the all blocks.
      # img_ids and txt_ids will only be splited once at the very beginning,
      # and keep splited through the whole transformer forward. The all2all
      # comm op only happens on the `out` tensor after local attn not on
      # img_ids and txt_ids.
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
      # Then, the final proj_out will gather the splited output.
      #     splited input (previous splited output)
      #     -> all gather
      #     -> un-split output
      "proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
    }
    return _cp_plan


# Implements async Ulysses communication for Attention module when context parallelism
# is enabled with Ulysses degree > 1. The async communication allows overlapping
# communication with computation for better performance.
def _async_ulysses_attn_ovis(
  self: OvisImageAttnProcessor,
  attn: OvisImageAttention,
  hidden_states: torch.Tensor,
  encoder_hidden_states: torch.Tensor = None,
  attention_mask: Optional[torch.Tensor] = None,
  image_rotary_emb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
  cp_config = getattr(self, "_cp_config", None)
  if cp_config is None:
    raise RuntimeError(
      "OvisImageAttnProcessor is missing _cp_config during async Ulysses attention.")

  value = attn.to_v(hidden_states)  # type: torch.Tensor
  value = value.unflatten(-1, (attn.heads, -1))
  if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
    encoder_value = attn.add_v_proj(encoder_hidden_states)  # type: torch.Tensor
    encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
    value = torch.cat([encoder_value, value], dim=1)

  comm = _All2AllComm(cp_config).init_meta(value)

  # Async all to all for value
  value_wait = comm.send_v(value)

  query = attn.to_q(hidden_states)
  query = query.unflatten(-1, (attn.heads, -1))  # type: torch.Tensor
  query = attn.norm_q(query)
  if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
    encoder_query = attn.add_q_proj(encoder_hidden_states)
    encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))  # type: torch.Tensor
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
    encoder_key = attn.add_k_proj(encoder_hidden_states)
    encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))  # type: torch.Tensor
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

  if encoder_hidden_states is not None:
    # Must be sync all to all for out when encoder_hidden_states is used
    out_wait = comm.send_o(out)  # (B, S_LOCAL, H_GLOBAL, D)
    out = out_wait.wait()  # type: torch.Tensor

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
  else:
    # Can be async all to all for out when no encoder_hidden_states
    out_wait = comm.send_o(out)  # (B, S_LOCAL, H_GLOBAL, D)
    return out_wait


ovis_attn_processor__call__ = OvisImageAttnProcessor.__call__


@functools.wraps(ovis_attn_processor__call__)
def __patch_ovis_attn_processor__(
  self: OvisImageAttnProcessor,
  attn: "OvisImageAttention",
  hidden_states: torch.Tensor,
  encoder_hidden_states: torch.Tensor = None,
  attention_mask: Optional[torch.Tensor] = None,
  image_rotary_emb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
  cp_config = getattr(self, "_cp_config", None)
  if cp_config is not None and cp_config.ulysses_degree > 1:
    return _async_ulysses_attn_ovis(
      self,
      attn,
      hidden_states,
      encoder_hidden_states=encoder_hidden_states,
      attention_mask=attention_mask,
      image_rotary_emb=image_rotary_emb,
    )

  # Otherwise, use the original call for non-ulysses case
  return ovis_attn_processor__call__(
    self,
    attn,
    hidden_states,
    encoder_hidden_states=encoder_hidden_states,
    attention_mask=attention_mask,
    image_rotary_emb=image_rotary_emb,
  )


@functools.wraps(OvisImageSingleTransformerBlock.forward)
def __patch_ovis_single_block__(
  self: OvisImageSingleTransformerBlock,
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
  # Perform attention with Ulysses async QKV proj, the attn_output
  # may be is an instance of AsyncCollectiveTensor.
  attn_output_wait = self.attn(
    hidden_states=norm_hidden_states,
    image_rotary_emb=image_rotary_emb,
    **joint_attention_kwargs,
  )
  # NOTE: Enable the out all2all overlap with mlp computation
  mlp_hidden_states, mlp_hidden_gate = torch.split(self.proj_mlp(norm_hidden_states),
                                                   [self.mlp_hidden_dim, self.mlp_hidden_dim],
                                                   dim=-1)
  mlp_hidden_states = self.act_mlp(mlp_hidden_gate) * mlp_hidden_states

  # NOTE: Then ensure the attn_output is ready
  if not isinstance(attn_output_wait, torch.Tensor):
    attn_output = attn_output_wait()  # type: torch.Tensor
  else:
    attn_output = attn_output_wait
  attn_output = attn_output.contiguous()
  if attn_output.ndim == 4:
    attn_output = attn_output.flatten(2, 3)

  hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
  gate = gate.unsqueeze(1)
  hidden_states = gate * self.proj_out(hidden_states)
  hidden_states = residual + hidden_states
  if hidden_states.dtype == torch.float16:
    hidden_states = hidden_states.clip(-65504, 65504)

  encoder_hidden_states, hidden_states = (
    hidden_states[:, :text_seq_len],
    hidden_states[:, text_seq_len:],
  )
  return encoder_hidden_states, hidden_states


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
