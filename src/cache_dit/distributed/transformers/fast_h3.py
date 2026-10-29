"""FastH3 transformer distributed parallelism planners (CP + TP).

FastH3 keeps MiniMax-H3's packed-sequence layout, so the context-parallel plan is inherited
unchanged: the split still happens at the first block, `rope` output rows and `adaln_indices` are
split alongside it, and the two output heads gather back. The video sparse attention itself needs
the full sequence to rank tiles, so the attention processor rejects context parallelism with an
explicit error for now.

The tensor-parallel plan is the MiniMax-H3 one plus the FastH3 compression gate: `to_gate_compress`
is a plain `hidden -> heads * head_dim` projection whose output is reshaped per head, so it shards
column-wise exactly like `to_q`. Every rank then owns `heads / tp_size` gate rows, which is what the
processor's `unflatten(-1, (attn.heads, -1))` expects after `shard_div_attr` updated `heads`.
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

from ..utils import shard_div_attr
from .minimax_h3 import (
  MiniMaxH3ContextParallelismPlanner,
  MiniMaxH3TensorParallelismPlanner,
)
from .register import (
  ContextParallelismPlannerRegister,
  TensorParallelismPlannerRegister,
)

__all__ = [
  "FastH3ContextParallelismPlanner",
  "FastH3TensorParallelismPlanner",
]


@ContextParallelismPlannerRegister.register("FastH3Transformer3DModel")
class FastH3ContextParallelismPlanner(MiniMaxH3ContextParallelismPlanner):
  r"""Context parallelism planner for `FastH3Transformer3DModel`.

  The plan is the MiniMax-H3 one: the packed sequence is sharded once at the first block, the
  per-row tensors (`rope` outputs, `adaln_indices`, `timestep_indices`) are split with it, and the
  two output heads gather the rows back. Only the attention path differs, and the FastH3 processor
  raises while video sparse attention is enabled because VSA has to rank tiles over the full
  sequence and the compression gate would have to ride the same all-to-all as Q/K/V.
  """

  def _apply(
    self,
    transformer: torch.nn.Module,
    parallelism_config,
    **kwargs,
  ) -> dict:
    if getattr(transformer, "enable_vsa", False):
      raise NotImplementedError(
        "FastH3 runs video sparse attention, which is not context-parallel yet: tile selection "
        "needs the whole sequence and the compression gate has to travel with Q/K/V. Disable VSA "
        "(`set_vsa_enabled(False)`) to use context parallelism, or run without --parallel.")
    return super()._apply(transformer, parallelism_config, **kwargs)


@TensorParallelismPlannerRegister.register("FastH3Transformer3DModel")
class FastH3TensorParallelismPlanner(MiniMaxH3TensorParallelismPlanner):
  r"""Tensor parallelism planner for `FastH3Transformer3DModel`.

  Identical to the MiniMax-H3 plan for the block stack, with one extra entry:
  `attn.to_gate_compress` shards column-wise. A dense FastH3 checkpoint (`-Dense-DataFree`) has no
  such module, and `parallelize_module` raises on an unknown FQN, so the entry is conditional.
  """

  def parallelize_transformer(
    self,
    transformer: torch.nn.Module,
    tp_mesh: DeviceMesh,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    """Shard the block stack, the compression gates and the output modulation.

    :param transformer: FastH3 transformer.
    :param tp_mesh: TP device mesh.
    :returns: The sharded transformer and the applied layer plans.
    """
    if getattr(transformer, "enable_vsa", False):
      # The sparse path calls index_select / gather / topk / BlockMask.from_kv_blocks and then adds
      # a gate gathered per tile; none of those has a DTensor strategy yet, and the failure would
      # surface inside DTensor dispatch. Refuse up front, symmetrically with the CP planner.
      raise NotImplementedError(
        "FastH3 runs video sparse attention, which is not tensor-parallel yet: tile selection "
        "indexes the full sequence and the block mask is a local index tensor. Call "
        "`set_vsa_enabled(False)` to use tensor parallelism, or drop --parallel-tp.")
    tp_size = tp_mesh.size()
    heads = transformer.config.num_attention_heads
    head_dim = transformer.config.attention_head_dim
    layer_plans: List[Dict[str, ParallelStyle]] = []

    for block in transformer.transformer_blocks:
      if getattr(block.attn, "fused_projections", False):
        raise ValueError("FastH3 does not support fused QKV projections under tensor parallelism; "
                         "keep `to_q` / `to_k` / `to_v` separate.")
      self.rearrange_ffn_weight(block, tp_size)
      shard_div_attr(block.attn, "heads", tp_size)
      layer_plan: Dict[str, ParallelStyle] = {
        "attn.to_q": ColwiseParallel(),
        "attn.to_k": ColwiseParallel(),
        "attn.to_v": ColwiseParallel(),
        "attn.to_out.0": RowwiseParallel(),
        "ff.net.0.proj": ColwiseParallel(),
        "ff.net.2": RowwiseParallel(),
        "adaln_proj.linear": ColwiseParallel(output_layouts=Replicate()),
      }
      if getattr(block.attn, "to_gate_compress", None) is not None:
        layer_plan["attn.to_gate_compress"] = ColwiseParallel()
      parallelize_module(module=block, device_mesh=tp_mesh, parallelize_plan=layer_plan)
      layer_plans.append(layer_plan)

    norm_out_plan = {"linear": ColwiseParallel(output_layouts=Replicate())}
    parallelize_module(module=transformer.norm_out,
                       device_mesh=tp_mesh,
                       parallelize_plan=norm_out_plan)
    layer_plans.append(norm_out_plan)

    local_q_rows = transformer.transformer_blocks[0].attn.to_q.weight.to_local().shape[0]
    if local_q_rows != (heads // tp_size) * head_dim:
      raise ValueError(f"Sharded Q projection has {local_q_rows} rows per rank, expected "
                       f"{heads // tp_size} heads x {head_dim} channels.")
    return transformer, layer_plans
