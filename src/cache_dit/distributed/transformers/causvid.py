"""Parallel planners for CausVid-Wan (causal Wan 2.1 student).

Context parallelism follows the LingBot/FastVideo strategy for causal video diffusion: the sequence
shard stays a contiguous slice of the current chunk, the KV cache is **partitioned by head** (each
rank owns the full history of its own heads, so the cross-chunk tokens never cross a rank boundary),
and the per-block attention performs the Ulysses all-to-all itself.

That is why the plan below has no attention-processor patch: `CausVidCausalAttnProcessor` reads
`_cp_config` from the processor instance (cache-dit attaches it to every diffusers attention module)
and exchanges q/k/v inside `attn1`. `attn2` (text cross-attention) needs no communication because the
text sequence is replicated on every rank.
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor.parallel import ParallelStyle

from ...distributed.core import (
  _ContextParallelInput,
  _ContextParallelModelPlan,
  _ContextParallelOutput,
)
from ...logger import init_logger
from ..config import ParallelismConfig
from .register import (
  ContextParallelismPlanner,
  ContextParallelismPlannerRegister,
  TensorParallelismPlannerRegister,
)
from .wan import WanTensorParallelismPlanner

logger = init_logger(__name__)


@ContextParallelismPlannerRegister.register("CausVidTransformer3D")
class CausVidContextParallelismPlanner(ContextParallelismPlanner):
  """Ulysses CP for the chunk-causal Wan student.

  * `rope` output-splits the rotary tables of the current chunk, so each rank rotates its own
    contiguous token slice with the correct global frame positions.
  * `blocks.0` splits `hidden_states`; cache-dit replays the first block's split hook for the Fn
    residual, which is why the split must hang on `blocks.0` and not on the root.
  * `proj_out` gathers after the projection, i.e. on `out_channels * prod(patch_size) = 64` wide
    tokens instead of the 1536-wide residual stream. `norm_out` runs before it on the local shard,
    which is exact because Wan2.1's modulation is per batch, not per token.

  Ring attention is rejected: the KV cache is addressed by global token coordinates and every rank
  must see the whole chunk it writes, so a rotating KV ring has no consistent slot mapping. Head
  counts and chunk lengths must divide by the Ulysses size; `--ulysses-anything` is rejected because
  the padded rows would land inside the KV slots.
  """

  def _apply(
    self,
    transformer: Optional[torch.nn.Module] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:
    assert transformer is not None, "Transformer must be provided."
    assert parallelism_config is not None, "ParallelismConfig must be provided."
    ring_size = parallelism_config.ring_size or 1
    if ring_size > 1:
      raise NotImplementedError(
        "CausVid-Wan context parallelism does not support Ring attention: the causal KV cache is "
        "addressed by global token coordinates, so every rank must hold the complete chunk it "
        "writes. Use `--parallel ulysses` instead.")
    if parallelism_config.ulysses_anything:
      raise NotImplementedError(
        "CausVid-Wan does not support `--ulysses-anything`: padding the chunk to a divisible length "
        "would write padded rows into the causal KV slots and corrupt every later chunk. Choose a "
        "`--num-frames` whose chunk token count divides by the Ulysses size instead.")

    tp_size = parallelism_config.tp_size or 1
    ulysses_size = parallelism_config.ulysses_size or 1
    heads = transformer.config.num_attention_heads // tp_size
    if heads % ulysses_size != 0:
      raise ValueError(
        f"CausVid-Wan CP needs the post-TP head count to divide by the Ulysses size, got "
        f"{transformer.config.num_attention_heads} / {tp_size} = {heads} head(s) over "
        f"{ulysses_size} rank(s).")

    _cp_plan: _ContextParallelModelPlan = {
      "rope": {
        0: _ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
        1: _ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
      },
      "blocks.0": {
        "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
      },
      "proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
    }
    logger.debug(f"Generated CausVid-Wan CP plan: {_cp_plan}")
    return _cp_plan


@TensorParallelismPlannerRegister.register("CausVidTransformer3D")
class CausVidTensorParallelismPlanner(WanTensorParallelismPlanner):
  """Tensor parallelism for the causal Wan blocks.

  The module tree is the diffusers Wan block (`attn1.to_q/k/v`, `attn1.to_out.0`, `ffn.net.*`), so
  the Wan plan applies unchanged: heads are divided on both attentions, the per-head RMSNorms become
  distributed norms, and the KV cache picks up the local head count because it is allocated after the
  plan has been applied.
  """

  def parallelize_transformer(
    self,
    transformer: nn.Module,
    tp_mesh: DeviceMesh,
  ) -> Tuple[nn.Module, List[Dict[str, ParallelStyle]]]:
    return super().parallelize_transformer(transformer=transformer, tp_mesh=tp_mesh)
