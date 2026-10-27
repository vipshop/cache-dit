from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.distributed import DeviceMesh
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


@ContextParallelismPlannerRegister.register("MOSSImageTransformer")
class MOSSImageContextParallelismPlanner(ContextParallelismPlanner):
  """Hook-based context parallelism for MOSS-Image.

  Every sample is one joint `[image, text]` sequence, so the split happens at the denoising stack
  instead of at the root:

  * `layers.0` splits `hidden_states` on dim 1 with 3 dims. cache-dit's Fn-residual replay reads
    the split hook of `transformer_blocks[0]`, so the input split has to live on the first block.
  * `rope_split` output-splits the RoPE table with the same partition. The table is shared by every
    block (only block 0's argument could be hooked), and the shrink must follow the framework
    sharder so `--ulysses-anything` stays consistent. It is applied after text refinement, which
    consumes the full-length table.
  * `sequence_gather` gathers dim 1 back before the forward slices out the image tokens, so the
    velocity keeps the layout the pipeline expects.

  Text refinement, the timestep MLP and the modulation/head projections are sequence independent or
  run on the full text sequence, so they stay replicated.

  Ring attention is rejected: it needs equally sized local chunks on every rank, while this joint
  length (`image + text` tokens) is usually not divisible by the CP size, and
  `--ulysses-anything` only covers the Ulysses path. Use Ulysses.
  """

  def _apply(
    self,
    transformer: Optional[torch.nn.Module] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:
    assert transformer is not None, "Transformer must be provided."
    assert parallelism_config is not None, "ParallelismConfig must be provided."
    if parallelism_config.ring_size is not None and parallelism_config.ring_size > 1:
      raise NotImplementedError(
        "MOSS-Image context parallelism does not support Ring attention: Ring needs equally sized "
        "local chunks, and the joint `image + text` length is usually not divisible by the CP size "
        "(`--ulysses-anything` only covers Ulysses). Use `--parallel ulysses` instead.")

    _cp_plan: _ContextParallelModelPlan = {
      "rope_split": {
        0: _ContextParallelInput(split_dim=1, expected_dims=3, split_output=True),
      },
      "layers.0": {
        "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3),
      },
      "sequence_gather": _ContextParallelOutput(gather_dim=1, expected_dims=3),
    }
    logger.debug(f"Generated MOSS-Image CP plan: {_cp_plan}")
    return _cp_plan


@TensorParallelismPlannerRegister.register("MOSSImageTransformer")
class MOSSImageTensorParallelismPlanner(TensorParallelismPlanner):
  """Tensor-parallel planner for MOSS-Image.

  Each block is `proj.q/k/v/o` plus a gated FFN, all plain `nn.Linear`:

  * `proj.q/k/v` are `ColwiseParallel` and `heads` is divided by `tp_size`, so the block reshapes
    with the local head count and the per-head `q_norm`/`k_norm` (`_RMSNorm(head_dim)`) stay
    replicated. `proj.o` is `RowwiseParallel` and all-reduces the head partial sums.
  * `proj.ffn_up_gate` packs `(hidden, gate)` into one `Linear(dim, 2 * ffn_dim)` whose two halves
    are consumed by `chunk(2, -1)`, so a plain column shard would give a rank only half of each
    operand. The weight rows are rearranged into `[hidden_0, gate_0, hidden_1, gate_1, ...]`
    before `parallelize_module`, which keeps `chunk(2, -1)` valid on the local shard.
  * `proj.ffn_down` is `RowwiseParallel`.

  Everything outside `layers` (embeddings, timestep MLP, shared modulation, output head and the
  norms) stays replicated: the residual stream is full-length after each all-reduce, so those
  modules see the same shapes as on one GPU.
  """

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
    transformer: torch.nn.Module,
    tp_mesh: DeviceMesh,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    from .ovis_image import rearrange_ffn_0_swiglu_proj_weight

    tp_size = tp_mesh.size()
    layer_plans: List[Dict[str, ParallelStyle]] = []
    for block in transformer.layers:
      assert isinstance(block, nn.Module), "MOSS-Image layers must hold `nn.Module` blocks."
      ffn_up_gate = block.proj["ffn_up_gate"]
      # The rearrange helper writes `2 * tp_size` equally sized row groups; a partial group would
      # silently leave zero rows behind.
      if ffn_up_gate.weight.shape[0] % (2 * tp_size) != 0:
        raise ValueError(
          f"[TP] Unsupported tp_size={tp_size}: the fused `ffn_up_gate` output dimension "
          f"{ffn_up_gate.weight.shape[0]} must be divisible by 2 * tp_size.")
      # Rearrange before parallelize_module: afterwards the weight is a DTensor shard.
      rearrange_ffn_0_swiglu_proj_weight(ffn_up_gate, tp_size)
      # `heads` drives `unflatten(-1, (heads, -1))`, so it must follow the local shard.
      shard_div_attr(block, "heads", tp_size)
      layer_plan: Dict[str, ParallelStyle] = {
        "proj.q": ColwiseParallel(),
        "proj.k": ColwiseParallel(),
        "proj.v": ColwiseParallel(),
        "proj.o": RowwiseParallel(),
        "proj.ffn_up_gate": ColwiseParallel(),
        "proj.ffn_down": RowwiseParallel(),
      }
      parallelize_module(module=block, device_mesh=tp_mesh, parallelize_plan=layer_plan)
      layer_plans.append(layer_plan)

    return transformer, layer_plans
