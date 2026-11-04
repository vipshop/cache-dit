"""Context- and tensor-parallel planners for LongLive 2.0.

The transformer is a Wan2.2-TI2V-5B block stack with a causal KV cache, so both plans are the Wan
ones: the sequence is split once at `blocks.0` (RoPE is recomputed on the full chunk and output-split
by the hook), and every projection that Wan shards is sharded here as well.

Context parallelism and the chunk boundary do not interact: the attention cache is a global-coordinate
buffer and both of its cursors (`current_start` / `current_end`) are computed by the pipeline from the
latent frame index, never from the local sequence length. The Ulysses all-to-all only rearranges heads
and sequence layout inside one forward, so the K/V written by chunk `i` and read by chunk `i+1` keep
their global addressing on every rank. Two divisibility preconditions are enforced by
`LongLive2Transformer3DModel.forward` / `init_kv_cache`: the chunk token count and the post-TP head
count must both divide by the Ulysses size.

Ring CP is rejected: it needs equal chunks per rank *and* would have to gather a cache that is already
laid out for the full clip, so the combination buys nothing over Ulysses here.
"""

from typing import Dict, List, Optional, Tuple

import torch
from diffusers.models.modeling_utils import ModelMixin
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
from .wan import DistributedRMSNorm

logger = init_logger(__name__)


@ContextParallelismPlannerRegister.register("LongLive2")
class LongLive2ContextParallelismPlanner(ContextParallelismPlanner):

  def _apply(
    self,
    transformer: Optional[torch.nn.Module | ModelMixin] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:

    if int(getattr(parallelism_config, "ring_size", 1) or 1) > 1:
      # Measured, not inferred: forcing Wan's ring plan crashes inside the KV write with
      # "The expanded size of the tensor (768) must match the existing size (384) ... Target sizes:
      # [1, 768, 24, 128]" - the cache is laid out for the chunk's global coordinates while ring only
      # ever holds a rotated slice of them.
      raise NotImplementedError(
        "LongLive 2.0 supports Ulysses context parallelism only: its causal KV cache is written at "
        "exact global token coordinates, so Ring's equal-chunk rotation makes the cache slice "
        "(chunk/ring_size tokens) and the rotated sequence disagree - measured to raise a shape "
        "error inside the cache write. Re-run with `--parallel ulysses`.")
    if bool(getattr(parallelism_config, "ulysses_anything", False)):
      # Measured: bypassing this check lets a 2-rank run finish, but the output is bit-identical to
      # plain Ulysses (30.16 dB / 0.7222 SSIM against the single-GPU reference, same as CP2), so the
      # flag is swallowed by the custom attention processor instead of buying extra CP sizes.
      raise NotImplementedError(
        "LongLive 2.0 rejects `--ulysses-anything`: the custom causal attention processor has no "
        "UAA padding branch, so the flag is silently ignored (measured: bit-identical to plain "
        "Ulysses). Pick an Ulysses size that divides both the head count and the chunk's latent "
        "frame count instead.")

    _cp_plan = {
      # The two RoPE tables: recomputed on the full chunk, then sliced to the local sequence.
      "rope": {
        0: _ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
        1: _ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
      },
      "blocks.0": {
        "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
      },
      "proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
      # Per-token timestep of the Wan2.2 TI2V conditioning path.
      "": {
        "timestep": _ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
      },
    }
    return _cp_plan


@TensorParallelismPlannerRegister.register("LongLive2")
class LongLive2TensorParallelismPlanner(TensorParallelismPlanner):

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
  ) -> Tuple[nn.Module, List[Dict[str, ParallelStyle]]]:

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
      parallelize_module(
        module=block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_plan,
      )

      block.attn1.norm_q = DistributedRMSNorm.from_rmsnorm(tp_mesh, block.attn1.norm_q)
      block.attn1.norm_k = DistributedRMSNorm.from_rmsnorm(tp_mesh, block.attn1.norm_k)
      block.attn2.norm_q = DistributedRMSNorm.from_rmsnorm(tp_mesh, block.attn2.norm_q)
      block.attn2.norm_k = DistributedRMSNorm.from_rmsnorm(tp_mesh, block.attn2.norm_k)
      return layer_plan

    layer_plans = []
    for _, block in transformer.blocks.named_children():
      layer_plans.append(prepare_block(block))
    return transformer, layer_plans
