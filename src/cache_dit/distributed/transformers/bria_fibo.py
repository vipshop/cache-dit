"""BriaFIBO distributed parallelism planners (CP and TP)."""

from typing import Dict, List, Tuple

import torch
import torch.distributed as dist
from diffusers.models.transformers.transformer_bria_fibo import (
  BriaFiboTransformer2DModel, )
from einops import rearrange
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
from ..config import ParallelismConfig
from ..utils import shard_div_attr
from .register import (
  ContextParallelismPlanner,
  ContextParallelismPlannerRegister,
  TensorParallelismPlanner,
  TensorParallelismPlannerRegister,
)

logger = init_logger(__name__)


def _build_ulysses_mask_perm(local_txt: int, local_img: int, cp_config, device) -> torch.Tensor:
  """Build the permutation mapping the original ``[text, image]`` order to the Ulysses all-to-all
  order ``[text_0, img_0, text_1, img_1, ...]``.

  Under Ulysses CP each rank holds a local sequence ``[text_local, image_local]``.
  The all-to-all concatenates these in rank order, so the global sequence the
  attention actually sees is ``cat_r([text_r, image_r])`` — NOT the original
  ``[text_global, image_global]``.  A 2D padding mask built in the original
  order must therefore be permuted (on both query and key dims) to match.

  :param local_txt: local text sequence length on this rank (post CP split).
  :param local_img: local image sequence length on this rank (post CP split).
  :param cp_config: the active ``_ContextParallelConfig``.
  :param device: device for the returned index tensor.
  :returns: 1D long tensor ``perm`` s.t. ``mask[..., perm, :][..., :, perm]``
      reorders the mask into all-to-all order.
  """
  group = cp_config._ulysses_mesh.get_group()
  world_size = dist.get_world_size(group)
  gathered = [None] * world_size
  dist.all_gather_object(gathered, (int(local_txt), int(local_img)), group=group)
  txt_sizes = [g[0] for g in gathered]
  img_sizes = [g[1] for g in gathered]
  global_txt = sum(txt_sizes)

  perm: List[int] = []
  txt_off = 0
  img_off = 0
  for r in range(world_size):
    perm.extend(range(txt_off, txt_off + txt_sizes[r]))
    base = global_txt + img_off
    perm.extend(range(base, base + img_sizes[r]))
    txt_off += txt_sizes[r]
    img_off += img_sizes[r]
  return torch.tensor(perm, device=device, dtype=torch.long)


def _patch_bria_fibo_mask_permute(transformer: BriaFiboTransformer2DModel) -> None:
  """Patch the transformer forward to permute the 2D attention mask under CP.

  BriaFIBO passes a full ``[B, 1, S, S]`` text-padding mask through
  ``joint_attention_kwargs``.  Ulysses all-to-all reorders the sequence, so
  the mask must be permuted (both query and key dims) into all-to-all order
  before it reaches the attention backend.  Everything else (local RoPE from
  split ids, local hidden/encoder states) is already handled correctly by the
  unmodified forward, so this patch only fixes the mask and delegates.
  """
  orig_forward = transformer.forward

  def patched_forward(
    hidden_states,
    encoder_hidden_states=None,
    text_encoder_layers=None,
    pooled_projections=None,
    timestep=None,
    img_ids=None,
    txt_ids=None,
    guidance=None,
    joint_attention_kwargs=None,
    return_dict=True,
  ):
    cp = getattr(transformer, "_cp_config", None)
    if (cp is not None and getattr(cp, "_world_size", 1) > 1
        and joint_attention_kwargs is not None):
      mask = joint_attention_kwargs.get("attention_mask")
      if mask is not None and mask.dim() == 4:
        perm = _build_ulysses_mask_perm(encoder_hidden_states.shape[1], hidden_states.shape[1], cp,
                                        mask.device)
        mask = mask.index_select(-2, perm).index_select(-1, perm)
        joint_attention_kwargs = {**joint_attention_kwargs, "attention_mask": mask}
    return orig_forward(
      hidden_states=hidden_states,
      encoder_hidden_states=encoder_hidden_states,
      text_encoder_layers=text_encoder_layers,
      pooled_projections=pooled_projections,
      timestep=timestep,
      img_ids=img_ids,
      txt_ids=txt_ids,
      guidance=guidance,
      joint_attention_kwargs=joint_attention_kwargs,
      return_dict=return_dict,
    )

  transformer.forward = patched_forward


@ContextParallelismPlannerRegister.register("BriaFiboTransformer2DModel")
class BriaFiboContextParallelismPlanner(ContextParallelismPlanner):
  """Transformer-level CP (Flux pattern) with attention-mask permutation.

  ``hidden_states``, ``encoder_hidden_states``, ``img_ids`` and ``txt_ids``
  are split at the root; each ``caption_projection`` output is split so the
  per-block text injection stays consistent with the local
  ``encoder_hidden_states``.  BriaFIBO also feeds a 2D ``[B, 1, S, S]``
  text-padding mask via ``joint_attention_kwargs``; because Ulysses all-to-all
  reorders the sequence, a forward patch permutes that mask (both query and
  key dims) into all-to-all order.  A single gather at ``proj_out`` restores
  the full sequence.
  """

  def _apply(
    self,
    transformer=None,
    parallelism_config=None,
    **kwargs,
  ) -> _ContextParallelModelPlan:
    _patch_bria_fibo_mask_permute(transformer)

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
      "caption_projection.*": {
        0: _ContextParallelInput(split_dim=1, expected_dims=3, split_output=True),
      },
      "proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
    }
    return _cp_plan


@TensorParallelismPlannerRegister.register("BriaFiboTransformer2DModel")
class BriaFiboTensorParallelismPlanner(TensorParallelismPlanner):
  """TP planner for BriaFIBO — standard MHA with dual block lists."""

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
  def rearrange_proj_out_weight(cls, block, tp_size: int):
    """Rearrange ``proj_out`` weight for row-wise TP.

    The ``proj_out`` in single-transformer blocks fuses attention output
    (first ``dim`` columns) and MLP output (remaining ``mlp_hidden_dim``
    columns).  RowwiseParallel splits the input dimension, so the column
    order must be interleaved by tp-group to keep attn/mlp features
    grouped per GPU.
    """
    hidden_dim = block.attn.to_q.weight.shape[0]
    requires_grad = block.proj_out.weight.requires_grad
    weight_t = block.proj_out.weight.data.T.detach().clone()  # [in_features, out_features]

    out_weight = weight_t[:hidden_dim, ...]  # [hidden_dim, out_features]
    out_weight = rearrange(out_weight, "(G D) C -> G D C", G=tp_size)

    down_weight = weight_t[hidden_dim:, ...]  # [mlp_hidden_dim, out_features]
    down_weight = rearrange(down_weight, "(G D) C -> G D C", G=tp_size)

    new_weight = torch.cat([out_weight, down_weight], dim=1)  # [tp, (dim+mlp_hidden_dim)/tp, out]
    new_weight = rearrange(new_weight, "G D C -> (G D) C")  # [in_features, out_features]

    block.proj_out.weight.data.copy_(new_weight.T)
    block.proj_out.weight.requires_grad_(requires_grad)

  def parallelize_transformer(
    self,
    transformer: BriaFiboTransformer2DModel,
    tp_mesh: DeviceMesh,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    tp_size = tp_mesh.get_group().size()
    layer_plans = []

    for _, block in transformer.transformer_blocks.named_children():
      shard_div_attr(block.attn, "heads", tp_size)
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
      self.rearrange_proj_out_weight(block, tp_size)
      shard_div_attr(block.attn, "heads", tp_size)
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
