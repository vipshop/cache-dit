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


@ContextParallelismPlannerRegister.register("LingBotVideoTransformer")
class LingBotVideoContextParallelismPlanner(ContextParallelismPlanner):
  """Hook-based context parallelism for LingBot-Video Dense.

  Every sample is one joint `[video; text]` sequence (`video` tokens first, then the text tokens), so
  the split happens at the denoising stack instead of at the root:

  * `blocks.0` splits `hidden_states` on dim 1 with 3 dims. cache-dit's Fn-residual replay reads the
    split hook of the first block, and the split hook resolves its argument by name, so
    `LingBotVideoBlock.forward` keeps `hidden_states` as its first parameter.
  * `cp_rotary` / `cp_temb6` / `cp_temb_input` output-split the RoPE table, the token-wise
    modulation and the timestep conditioning with the same partition. They are computed before the
    block loop and consumed per token, so they must follow the framework sharder to keep
    `--ulysses-anything` consistent. `temb6` is split while it is still 3-D: the block loop rebuilds
    it as `(B*S, 6D)` afterwards.
  * `cp_out` gathers dim 1 back after `proj_out`, i.e. on `prod(patch_size) * out_channels = 64`
    wide tokens instead of the 2048 wide residual stream, which cuts the all-gather volume 32x. The
    forward then slices the video tokens and unpatchifies.

  The patch embedder, the text embedder, the timestep MLP and the final modulation are sequence
  independent or run on the full text sequence, so they stay replicated.

  Ring attention is rejected: it needs equally sized local chunks on every rank, the joint length is
  usually not divisible by the CP size, and `--ulysses-anything` only covers the Ulysses path. The
  vendored transformer also does not implement upstream's pad-to-divisible path, so a Ulysses run on
  a non-divisible joint length needs `--ulysses-anything`.
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
        "LingBot-Video context parallelism does not support Ring attention: Ring needs equally sized "
        "local chunks, and the joint `video + text` length is usually not divisible by the CP size "
        "(`--ulysses-anything` only covers Ulysses). Use `--parallel ulysses` instead.")

    _cp_plan: _ContextParallelModelPlan = {
      "cp_rotary": {
        0: _ContextParallelInput(split_dim=1, expected_dims=3, split_output=True),
      },
      "cp_temb_input": {
        0: _ContextParallelInput(split_dim=1, expected_dims=3, split_output=True),
      },
      "cp_temb6": {
        0: _ContextParallelInput(split_dim=1, expected_dims=3, split_output=True),
      },
      "blocks.0": {
        "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3),
      },
      "cp_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
    }
    logger.debug(f"Generated LingBot-Video CP plan: {_cp_plan}")
    return _cp_plan


@TensorParallelismPlannerRegister.register("LingBotVideoTransformer")
class LingBotVideoTensorParallelismPlanner(TensorParallelismPlanner):
  """Tensor-parallel planner for LingBot-Video Dense.

  Each block is a joint-attention stack plus a gated FFN, all plain `nn.Linear`:

  * `attn.to_q/k/v` are `ColwiseParallel` and `attn.num_heads` is divided by `tp_size`, so the block
    reshapes with the local head count and the per-head `norm_q`/`norm_k` (RMSNorm over `head_dim`)
    stay replicated. `attn.to_out` is `RowwiseParallel` and all-reduces the head partial sums.
  * `ffn.gate_proj` / `ffn.up_proj` are `ColwiseParallel`, `ffn.down_proj` is `RowwiseParallel`.
    `intermediate_size=6144` is divisible by 2/4/8.

  Everything outside `blocks` (patch embedder, text embedder, timestep MLP, `time_modulation`,
  `norm_out_modulation`, `proj_out`, the per-block `scale_shift_table` and every norm) stays
  replicated: the residual stream is full width after each all-reduce, and the token-wise modulation
  is `chunk(6, -1)`-ed, so each rank needs the whole `6 * hidden_size` vector.
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
    tp_size = tp_mesh.size()
    layer_plans: List[Dict[str, ParallelStyle]] = []
    for block in transformer.blocks:
      assert isinstance(block, nn.Module), "LingBot-Video blocks must be `nn.Module`."
      attn = block.attn
      for name in ("gate_proj", "up_proj"):
        out_features = getattr(block.ffn, name).weight.shape[0]
        if out_features % tp_size != 0:
          raise ValueError(
            f"[TP] Unsupported tp_size={tp_size}: the FFN output dimension {out_features} must be "
            f"divisible by tp_size.")
      # `num_heads` drives `unflatten(2, (heads, head_dim))`, so it must follow the local shard.
      shard_div_attr(attn, "num_heads", tp_size)
      layer_plan: Dict[str, ParallelStyle] = {
        "attn.to_q": ColwiseParallel(),
        "attn.to_k": ColwiseParallel(),
        "attn.to_v": ColwiseParallel(),
        "attn.to_out": RowwiseParallel(),
        "ffn.gate_proj": ColwiseParallel(),
        "ffn.up_proj": ColwiseParallel(),
        "ffn.down_proj": RowwiseParallel(),
      }
      parallelize_module(module=block, device_mesh=tp_mesh, parallelize_plan=layer_plan)
      layer_plans.append(layer_plan)

    return transformer, layer_plans
