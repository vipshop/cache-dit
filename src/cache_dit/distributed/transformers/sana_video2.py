"""SANA-Video 2.0 distributed parallelism planners (CP + TP).

The transformer is cache-dit's vendored copy (`cache_dit.models.pipelines.sana_video2`), so the
planners can rely on the exact module layout instead of diffusers attribute names.

* CP shards the token axis. Two properties make it work: the gated linear attention builds a
  per-head ``value @ key^T`` state that is a **sum over all tokens**, so a sequence shard only
  needs one fp32 all-reduce per layer to become the global state (done inside
  `GatedLinearAttention`), and the dense anchor layers plus the text cross-attention run through
  `_dispatch_attention_fn`, which owns the Ulysses all-to-all for the head split.
* TP shards the SwiGLU feed-forward network only. Attention cannot be head-sharded without
  rewriting the block forward: `attn.qkv` is one fused `Linear(C, 3C)`, the Q/K RMSNorm spans the
  full channel axis, the gated linear attention reshapes back with `channels = x.shape[-1]`, and
  `beta_proj` / `output_gate` are per-channel gates multiplied onto a full-width tensor.
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

from ...distributed.core import (
  _ContextParallelInput,
  _ContextParallelOutput,
)
from ...logger import init_logger
from ..config import ParallelismConfig
from .register import (
  ContextParallelismPlanner,
  ContextParallelismPlannerRegister,
  TensorParallelismPlanner,
  TensorParallelismPlannerRegister,
)

logger = init_logger(__name__)


@ContextParallelismPlannerRegister.register("SanaVideo2Transformer3DModel")
class SanaVideo2ContextParallelismPlanner(ContextParallelismPlanner):
  """Context parallelism for `SanaVideo2Transformer3DModel`.

  The token split happens on markers rather than on the Transformer arguments: `forward` embeds the
  full latent grid first (a 1x1x1 Conv3d, negligible next to the block stack) and only then hands
  the tokens to the Attention-Residual loop, whose depth buffers must hold the *local* shard. The
  two RoPE tables are split by the same hooks so every rank rotates its own position window, and
  the shard is gathered before the latent grid is rebuilt.

  Head divisibility: the dense anchor layers have 10 heads, so `ulysses_size` must divide 10 (2 is
  fine, 4 needs `--ulysses-anything`); the gated linear layers have 20 heads and only need the
  state all-reduce, which is degree-agnostic.
  """

  def _apply(
    self,
    transformer: torch.nn.Module | None = None,
    parallelism_config: ParallelismConfig | None = None,
    **kwargs,
  ):
    ulysses = 1
    ring = 1
    anything = False
    if parallelism_config is not None:
      ulysses = int(parallelism_config.ulysses_size or 1)
      ring = int(parallelism_config.ring_size or 1)
      anything = bool(getattr(parallelism_config, "ulysses_anything", False))
    # Ring CP rotates the token axis instead of splitting heads, so only the Ulysses degree has to
    # divide the head count. `ulysses_anything` is forbidden together with ring > 1 by the CP config,
    # so the hint below only mentions it when it is actually available.
    if transformer is not None and ulysses > 1 and not anything:
      softmax_heads = min(
        int(block.attn.heads) for block in transformer.blocks
        if getattr(block.attn, "dim", None) == transformer.config.softmax_head_dim)
      if softmax_heads % ulysses:
        hint = ("Pass `--ulysses-anything` to shard them unevenly, or use a smaller `--ulysses`."
                if ring == 1 else ("Use a smaller `--ulysses`: `--ulysses-anything` is not "
                                   "available together with `--ring`."))
        raise ValueError(
          f"SANA-Video 2.0 dense anchor layers have {softmax_heads} heads, which is not divisible "
          f"by the Ulysses degree {ulysses}. {hint}")
    _cp_plan = {
      "sequence_split": {
        0: _ContextParallelInput(split_dim=1, expected_dims=3, split_output=True),
      },
      "rope_linear_split": {
        0: _ContextParallelInput(split_dim=2, expected_dims=4, split_output=True),
      },
      "rope_softmax_split": {
        0: _ContextParallelInput(split_dim=2, expected_dims=4, split_output=True),
      },
      "sequence_gather": _ContextParallelOutput(gather_dim=1, expected_dims=3),
    }
    return _cp_plan


def sana_video2_block_plan(block: torch.nn.Module) -> Dict[str, ParallelStyle]:
  """Build the TP plan for one `SanaVideo2Block`.

  :param block: A SANA-Video 2.0 transformer block.
  :returns: The `parallelize_module` plan for this block.
  """
  return {
    # SwiGLU: `silu(gate_proj(x)) * up_proj(x)` is elementwise, so the sharded outputs line up
    # without communication, and `down_proj` all-reduces back to the full residual stream.
    "mlp.gate_proj": ColwiseParallel(),
    "mlp.up_proj": ColwiseParallel(),
    "mlp.down_proj": RowwiseParallel(),
  }


@TensorParallelismPlannerRegister.register("SanaVideo2Transformer3DModel")
class SanaVideo2TensorParallelismPlanner(TensorParallelismPlanner):
  """Tensor parallelism for `SanaVideo2Transformer3DModel`.

  Strategy: shard the SwiGLU FFN, keep the attention replicated.

  The FFN carries ~2.5B of the 4.5B parameters (4x SwiGLU over 32 layers), so it is both the
  memory and the GEMM win. Attention is left replicated on purpose: head-sharding `attn.qkv` would
  split a fused `Linear(C, 3C)`, break the full-channel Q/K RMSNorm and the `channels =
  x.shape[-1]` reshape of the gated linear attention, and split `output_gate` / `beta_proj` away
  from the full-width tensor they multiply. Sharding it with `Replicate()` layouts instead (the
  SANA-WM recipe) buys memory but adds an all-gather per projection, so it is not part of the
  shipped plan.

  Everything outside the stack (`x_embedder`, `t_embedder`, `t_block`, `y_embedder`,
  `attn_res`, `final_layer`) stays replicated: the residual stream is full length after each FFN
  all-reduce.
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
    layer_plans: List[Dict[str, ParallelStyle]] = []
    for _, block in transformer.blocks.named_children():
      layer_plan = sana_video2_block_plan(block)
      parallelize_module(
        module=block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_plan,
      )
      layer_plans.append(layer_plan)
    return transformer, layer_plans
