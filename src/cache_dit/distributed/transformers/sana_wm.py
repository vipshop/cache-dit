"""SANA-WM distributed parallelism planners.

Tensor parallelism only. Context parallelism is **structurally unavailable** on SANA-WM and the
planner refuses it explicitly instead of letting the registry fail with a generic message.
"""

from typing import Dict, List, Tuple

import torch
from torch.distributed import DeviceMesh
from torch.distributed._tensor import Replicate
from torch.distributed.tensor.parallel import (
  ColwiseParallel,
  ParallelStyle,
  RowwiseParallel,
  parallelize_module,
)

from ..config import ParallelismConfig
from .register import (
  ContextParallelismPlanner,
  ContextParallelismPlannerRegister,
  TensorParallelismPlanner,
  TensorParallelismPlannerRegister,
)


@ContextParallelismPlannerRegister.register("SanaWM")
class SanaWMContextParallelismPlanner(ContextParallelismPlanner):
  """Refusal planner: SANA-WM has no valid context-parallel decomposition.

  Both attention branches of `SanaVideoMSCamCtrlBlock` are recurrences **along the temporal
  axis**: `BidirectionalGDNUCPESinglePathLiteLA` runs a bidirectional gated delta rule whose
  per-pixel state is carried from frame to frame, and the camera branch does the same in UCPE ray
  space. `GLUMBConvTemp` then mixes frames again through its temporal convolution. Sharding the
  token sequence along `T` therefore needs sequential state/halo exchange between ranks *inside*
  a block -- not the all-to-all that Ulysses or the ring rotation of Ring attention implement.

  Two independent implementations reached the same conclusion: vllm-omni pins
  `skip_sequence_parallel=True` for this model, and sglang raises when `sp_degree != 1`
  ("Stage-1 GDN/GLUMBConvTemp span frames and require halo/state exchange before latents can be
  sharded"). Registering a refusal keeps the error actionable: the generic registry message
  ("No planner registered under name") would suggest a missing integration rather than an
  impossible one.
  """

  def _apply(
    self,
    transformer: torch.nn.Module | None = None,
    parallelism_config: ParallelismConfig | None = None,
    **kwargs,
  ):
    name = type(transformer).__name__ if transformer is not None else "SanaWM"
    raise NotImplementedError(
      f"Context parallelism is not available for SANA-WM ({name}). Its blocks carry a "
      "bidirectional GDN recurrence along the temporal axis plus a cross-frame temporal "
      "convolution (`GLUMBConvTemp`), so a sequence shard would need per-frame state and halo "
      "exchange between ranks, which neither Ulysses nor Ring provides. Use tensor parallelism "
      "(`--parallel tp`) for the multi-GPU run instead; vllm-omni and sglang made the same call.")


def _refuse_refiner(transformer: torch.nn.Module, what: str) -> None:
  """Reject the stage-2 refiner explicitly instead of failing on a missing attribute.

  `SanaWMLTX2RefinerTransformer3DModel` shares the `SanaWM` planner prefix but has no `blocks`
  stack and, more importantly, runs the LTX-2 head-sharded attention path with its own
  KV-cache-aware processors. Sharding it needs the LTX-2 head/norm/RoPE treatment plus processor
  support, which is a follow-up rather than a mechanical extension of this plan.

  :param transformer: Transformer the planner was selected for.
  :param what: Feature name used in the error message.
  :raises NotImplementedError: When `transformer` is the stage-2 refiner.
  """
  from diffusers.models.transformers.transformer_sana_wm_refiner import (
    SanaWMLTX2RefinerTransformer3DModel, )

  if isinstance(transformer, SanaWMLTX2RefinerTransformer3DModel):
    raise NotImplementedError(
      f"{what.capitalize()} is not available for the SANA-WM stage-2 refiner "
      "(`SanaWMLTX2RefinerTransformer3DModel`). Its blocks reuse the LTX-2 attention layout with "
      "KV-cache-aware processors, so sharding them needs the LTX-2 head/norm/RoPE path plus "
      "processor support. Run stage 1 with the parallel config and keep the refiner on one GPU.")


def sana_wm_block_plan(block: torch.nn.Module) -> Dict[str, ParallelStyle]:
  """Build the TP plan for one `SanaVideoMSCamCtrlBlock`.

  :param block: A stage-1 transformer block.
  :returns: The `parallelize_module` plan for this block.
  """
  plan: Dict[str, ParallelStyle] = {
    # Fused `[Q|K|V]` output: shard the rows, gather the activations back to full width.
    "attn.qkv": ColwiseParallel(output_layouts=Replicate()),
    "attn.proj": RowwiseParallel(input_layouts=Replicate()),
    # Per-frame GDN gates: `(B, N, C) -> (B, T, HW, heads)`, broadcast over the spatial axis.
    "attn.beta_proj": ColwiseParallel(output_layouts=Replicate()),
    "attn.gate_proj": ColwiseParallel(output_layouts=Replicate()),
    # UCPE camera branch. `cam_heads == heads` and `cam_head_dim == head_dim`, so it is the same
    # shape as the main branch and is added back into the same residual stream.
    "attn.q_proj_cam": ColwiseParallel(output_layouts=Replicate()),
    "attn.k_proj_cam": ColwiseParallel(output_layouts=Replicate()),
    "attn.v_proj_cam": ColwiseParallel(output_layouts=Replicate()),
    "attn.out_proj_cam": RowwiseParallel(input_layouts=Replicate()),
    # Text cross-attention: `q_linear` / `kv_linear` reshape with the *global* head count.
    "cross_attn.q_linear": ColwiseParallel(output_layouts=Replicate()),
    "cross_attn.kv_linear": ColwiseParallel(output_layouts=Replicate()),
    "cross_attn.proj": RowwiseParallel(input_layouts=Replicate()),
  }
  if getattr(block.attn, "output_gate", None) is not None:
    # Multiplies the attention output elementwise, so it has to stay full width. Absent when the
    # block was built with `use_output_gate=False`.
    plan["attn.output_gate"] = ColwiseParallel(output_layouts=Replicate())
  if getattr(block, "plucker_proj", None) is not None:
    # Post-attention Plucker injection: `x = x + plucker_proj(plucker_emb)`.
    plan["plucker_proj"] = ColwiseParallel(output_layouts=Replicate())
  return plan


@TensorParallelismPlannerRegister.register("SanaWM")
class SanaWMTensorParallelismPlanner(TensorParallelismPlanner):
  """Tensor parallelism for the SANA-WM stage-1 DiT (`SanaWMTransformer3DModel`).

  Strategy: shard the attention weights, keep every activation full width (the PRX recipe).

  The block's attention cannot be head-sharded without rewriting its forward. `attn.qkv` is one
  fused `Linear(C, 3C)` whose output is reshaped as `(B, N, 3, heads, dim)`, the Q/K RMSNorm runs
  over the **full channel** axis, and the short convolution reshapes the projections back with
  `channels = x.shape[-1]`. A `Shard(-1)` output would break all three. Sharding the weights while
  all-gathering the output back to full width (`output_layouts=Replicate()`) keeps the fused
  reshape, the global RMS and the depthwise temporal convolution untouched, so the plan is a
  **memory** optimization: it halves the attention weight footprint and stays numerically exact
  (single-card vs TP2 differs only by the all-gather reduction order).

  Unsharded on purpose:

  * `mlp` (`GLUMBConvTemp`) is `Conv2d` end to end -- `torch.distributed.tensor.parallel` only
    covers `nn.Linear`, and the inverted bottleneck is ~half of the block GEMM budget, so TP
    cannot become a speed win here regardless of the attention plan.
  * `final_layer.linear` maps channels straight onto `patch * out_channels`, which a shard would
    scramble.
  * The embedders (`x_embedder` / `y_embedder` / `raymap_embedder` / `time_proj` / `t_embedder` /
    `t_block`) run once per forward and hold a negligible share of the weights.

  The GDN gates (`beta_proj` / `gate_proj`), the shared output gate and the camera projections are
  sharded with a replicated output so the elementwise products stay full width.
  """

  def _apply(
    self,
    transformer: torch.nn.Module,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    _refuse_refiner(transformer, "tensor parallelism")
    tp_mesh = self.mesh(parallelism_config=parallelism_config)
    transformer, layer_plans = self.parallelize_transformer(
      transformer=transformer,
      tp_mesh=tp_mesh,
    )
    return transformer, layer_plans

  def parallelize_transformer(
    self,
    transformer: torch.nn.Module,
    tp_mesh: DeviceMesh,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    layer_plans: List[Dict[str, ParallelStyle]] = []
    for _, block in transformer.blocks.named_children():
      layer_plan = sana_wm_block_plan(block)
      parallelize_module(
        module=block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_plan,
      )
      layer_plans.append(layer_plan)
    return transformer, layer_plans
