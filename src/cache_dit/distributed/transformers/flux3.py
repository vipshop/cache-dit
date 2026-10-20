"""FLUX.3 transformer distributed parallelism planners (CP + TP).

FLUX.3 has four block stacks that share one block layout (`Flux3ModeTransformerBlock` for the three
per-token-mode streams, `Flux3SingleTransformerBlock` for the joint stack), so TP shards all of them
with the same layer plan.

The two planners follow the two different shapes of the model. TP shards the weights of every block
stack. CP shards the token streams of the root forward only; see
`Flux3ContextParallelismPlanner` for why a root split is enough although every block rebuilds the
joint sequence with `torch.cat`.
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

from ..core import (
  _ContextParallelInput,
  _ContextParallelModelPlan,
  _ContextParallelOutput,
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
  """Context parallelism planner for `Flux3Transformer2DModel`.

  The split is a single root split: the three token streams (`encoder_hidden_states` = text mode,
  `hidden_states` = image mode, `image_cond_hidden_states` = reference mode) and their position-id
  tables are sharded along the sequence dim, and the velocity prediction is gathered at `proj_out`.

  This is sound even though every block rebuilds the joint sequence with `torch.cat` (the joint
  stack attends over `cat([text, image(, reference)])` and the block itself over
  `cat([text, content])`): attention has no mask and RoPE is applied per token from the ids, so a
  block's attention over the *union* of the ranks' contiguous chunks equals the full-sequence
  attention. A rank's content is a per-stream subsequence concatenated in the block's own order
  (`[text_c_i, image_c_i, reference_c_i]`), not a slice of the global order, which is exactly why
  the order-invariance of the unmasked attention matters. The per-mode modulations are per-token
  broadcasts of one timestep embedding and are recomputed from the local lengths.
  """

  def _apply(
    self,
    transformer: torch.nn.Module,
    parallelism_config,
    **kwargs,
  ) -> "_ContextParallelModelPlan":
    if getattr(parallelism_config, "ulysses_async", False):
      raise ValueError("Async Ulysses (`--ulysses-async`) is not supported for "
                       "Flux3Transformer2DModel yet: the async registry has no patch for its "
                       "attention processor. Drop the flag or use another model.")
    _cp_plan = {
      "": {
        "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3),
        "encoder_hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3),
        "image_cond_hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3),
        "img_ids": _ContextParallelInput(split_dim=0, expected_dims=2),
        "txt_ids": _ContextParallelInput(split_dim=0, expected_dims=2),
        "image_cond_ids": _ContextParallelInput(split_dim=0, expected_dims=2),
      },
      "proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
    }
    return _cp_plan


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
