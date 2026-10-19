"""FLUX.3 transformer distributed parallelism planner (TP).

FLUX.3 has four block stacks that share one block layout (`Flux3ModeTransformerBlock` for the three
per-token-mode streams, `Flux3SingleTransformerBlock` for the joint stack), so TP shards all of them
with the same layer plan.

Context parallelism is intentionally not implemented: the joint stack attends over
`cat([text, image(, reference)])`, but the per-mode stacks that feed it run on three separate
streams of different lengths (text is right-padded to a multiple of 80, the image grid is
`H/16 x W/16`, the references are their own grids). A contiguous Ulysses split of the joint
sequence would cut across the mode boundary, so every rank would need a different, padded local
layout of all three streams, plus matching splits of `joint_rotary_emb` and the per-mode
modulations. That requires reassembling the whole forward, which the hook-based CP design rules out.
"""

from typing import Dict, List, Tuple

import torch

from torch.distributed import DeviceMesh
from torch.distributed.tensor.parallel import (
  ColwiseParallel,
  ParallelStyle,
  RowwiseParallel,
  parallelize_module,
)

from ..utils import shard_div_attr
from .register import (
  ContextParallelismPlanner,
  ContextParallelismPlannerRegister,
  TensorParallelismPlanner,
  TensorParallelismPlannerRegister,
)


@ContextParallelismPlannerRegister.register("Flux3Transformer2DModel")
class Flux3ContextParallelismPlanner(ContextParallelismPlanner):
  """Placeholder planner: FLUX.3 cannot be context-parallelized (see the module docstring)."""

  def _apply(self, transformer: torch.nn.Module, parallelism_config, **kwargs):
    raise ValueError(
      "FLUX.3 does not support context parallelism: its joint stack attends over "
      "cat([text, image(, reference)]) while the per-mode stacks that feed it run on three "
      "separate streams of different lengths, so the joint sequence cannot be split contiguously "
      "across ranks with a hook-based plan. Use tensor parallelism (--parallel tp) or more GPUs "
      "instead.")


@TensorParallelismPlannerRegister.register("Flux3Transformer2DModel")
class Flux3TensorParallelismPlanner(TensorParallelismPlanner):
  """Tensor parallelism planner for `Flux3Transformer2DModel`."""

  def _apply(
    self,
    transformer: torch.nn.Module,
    parallelism_config,
    **kwargs,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    tp_mesh = self.mesh(parallelism_config=parallelism_config)
    transformer, layer_plans = self.parallelize_transformer(
      transformer=transformer,
      tp_mesh=tp_mesh,
    )
    return transformer, layer_plans

  def _block_lists(self, transformer: torch.nn.Module) -> List[torch.nn.ModuleList]:
    """All block stacks of the transformer: the three per-mode stacks and the joint stack.

    :param transformer: FLUX.3 transformer.
    :returns: The block stacks to shard.
    """
    return [
      transformer.txt_mode_blocks,
      *transformer.mode_blocks.values(),
      transformer.single_transformer_blocks,
    ]

  def rearrange_ffn_weight(self, block: torch.nn.Module, tp_size: int) -> None:
    """Rearrange the fused gate/up projection so each TP shard can apply its own SwiGLU.

    `Flux3FeedForward.linear_in` fuses the SwiGLU gate and up projections into one weight of shape
    `[2 * inner_dim, dim]` (`gate = x[..., :inner_dim]`, `up = x[..., inner_dim:]`). A plain
    `ColwiseParallel` would give each rank one contiguous chunk, i.e. the gate rows to the first
    rank and the up rows to the second, and the local SwiGLU would then mix gate with gate.

    The weight is therefore reordered to `[gate_0, up_0, gate_1, up_1, ...]`, one `(gate, up)`
    pair per TP chunk, so that each rank holds both halves of the elementwise product for its own
    `inner_dim / tp_size` slice of channels. `linear_out` (`RowwiseParallel`) splits its input
    columns contiguously in the same chunk order, so the two stay aligned.

    :param block: Transformer block whose `ff.linear_in` weight is rearranged in place.
    :param tp_size: TP world size.
    """
    linear_in = block.ff.linear_in
    weight = linear_in.weight.data
    if weight.shape[0] % (2 * tp_size) != 0:
      raise ValueError(f"Cannot split the fused SwiGLU projection of shape {tuple(weight.shape)} "
                       f"into {tp_size} equal (gate, up) chunks.")
    gate, up = weight.chunk(2, dim=0)
    gate_chunks = gate.chunk(tp_size, dim=0)
    up_chunks = up.chunk(tp_size, dim=0)
    linear_in.weight.data = torch.cat(
      [torch.cat([g, u], dim=0) for g, u in zip(gate_chunks, up_chunks)],
      dim=0,
    ).contiguous()

    if linear_in.bias is not None:
      gate, up = linear_in.bias.data.chunk(2, dim=0)
      linear_in.bias.data = torch.cat(
        [torch.cat([g, u], dim=0) for g, u in zip(gate.chunk(tp_size), up.chunk(tp_size))],
        dim=0,
      ).contiguous()

  def parallelize_transformer(
    self,
    transformer: torch.nn.Module,
    tp_mesh: DeviceMesh,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    """Shard every block stack of the transformer.

    :param transformer: FLUX.3 transformer.
    :param tp_mesh: TP device mesh.
    :returns: The sharded transformer and the applied layer plans.
    """
    tp_size = tp_mesh.size()
    layer_plans = []

    for blocks in self._block_lists(transformer):
      for block in blocks:
        self.rearrange_ffn_weight(block, tp_size)
        # The attention processor reshapes Q/K/V with `attn.heads`, which must follow the sharded
        # projection size (24 -> 12 for tp_size=2).
        shard_div_attr(block.attn, "heads", tp_size)
        layer_plan = {
          "attn.to_q": ColwiseParallel(),
          "attn.to_k": ColwiseParallel(),
          "attn.to_v": ColwiseParallel(),
          "attn.to_out.0": RowwiseParallel(),
          "ff.linear_in": ColwiseParallel(),
          "ff.linear_out": RowwiseParallel(),
        }
        parallelize_module(module=block, device_mesh=tp_mesh, parallelize_plan=layer_plan)
        layer_plans.append(layer_plan)

    return transformer, layer_plans
