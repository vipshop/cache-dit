"""Context and tensor parallelism for LongCat-Video-Avatar.

The transformer is cache-dit's vendored copy (`cache_dit.models.pipelines.longcat_video_avatar`), so
the planners own both the hook plan and the CP-aware code paths inside the model.
"""

from typing import Dict, List, Tuple

import torch
from torch import nn
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
from .register import (
  ContextParallelismPlanner,
  ContextParallelismPlannerRegister,
  TensorParallelismPlanner,
  TensorParallelismPlannerRegister,
)

logger = init_logger(__name__)


@ContextParallelismPlannerRegister.register("LongCatVideoAvatarTransformer3DModel")
class LongCatVideoAvatarContextParallelismPlanner(ContextParallelismPlanner):
  """Frame-aligned context parallelism for LongCat-Video-Avatar.

  The token axis of this model is `frame * height * width`, and the audio cross-attention reshapes it
  back into `(frame, height * width)` before every attention call. A rank therefore has to own whole
  latent frames — a split inside a frame is not representable — which is why the planner requires the
  latent frame count to be divisible by the CP size and rejects the uneven Ulysses mode.

  The plan is a single-point split at the denoising stack:

  * `blocks.0` splits `hidden_states` on the token axis. Everything before the loop (patch embedder,
    timestep MLP, text embedder, audio projection, the 3-D RoPE table) stays full-length on every
    rank: it is cheap, has no attention, and the model slices the timestep embedding and the audio
    tokens to the local frame range itself (they are Python-side tensors of `tuple` shape, so hooks
    cannot reach them).
  * `final_layer` gathers the token axis back. Gathering after `final_layer` instead of after the
    last block moves `prod(patch_size) * out_channels = 64` wide features instead of the 4096 wide
    residual stream, i.e. 64x less all-gather volume.

  Only the self-attention all-to-all's: the text cross-attention is replicated on every rank and the
  audio cross-attention is per-frame, so both run with `parallel_config=None` (a second collective
  there would concatenate the replicated keys and produce a 2x long text/audio sequence).
  """

  def _apply(
    self,
    transformer: torch.nn.Module = None,
    parallelism_config: ParallelismConfig = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:
    assert transformer is not None, "Transformer must be provided."
    assert parallelism_config is not None, "ParallelismConfig must be provided."
    if parallelism_config.ring_size is not None and parallelism_config.ring_size > 1:
      raise NotImplementedError(
        "LongCat-Video-Avatar context parallelism does not support Ring attention: the audio "
        "cross-attention reshapes the token axis by frame, so every rank needs whole latent frames, "
        "and Ring's wrap-around schedule has no gain here. Use `--parallel ulysses` instead.")
    if getattr(parallelism_config, "ulysses_anything", False):
      raise NotImplementedError(
        "LongCat-Video-Avatar does not support `--ulysses-anything`: an uneven sequence split cuts "
        "through a latent frame, and the per-frame audio cross-attention cannot be reshaped from a "
        "partial frame. Pick a clip whose latent frame count is divisible by the CP size.")

    _cp_plan: _ContextParallelModelPlan = {
      "blocks.0": {
        "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3),
      },
      "final_layer": _ContextParallelOutput(gather_dim=1, expected_dims=3),
    }
    logger.debug(f"Generated LongCat-Video-Avatar CP plan: {_cp_plan}")
    return _cp_plan


@TensorParallelismPlannerRegister.register("LongCatVideoAvatarTransformer3DModel")
class LongCatVideoAvatarTensorParallelismPlanner(TensorParallelismPlanner):
  """Tensor parallelism for LongCat-Video-Avatar.

  Every projection in the stack is a plain `nn.Linear`, and three of them are packed: the self
  attention packs Q/K/V into one `Linear(dim, 3 * dim)`, and both cross-attentions pack their KV into
  `Linear(in, 2 * dim)`. The attention processors slice those packed outputs by head, so they need
  the **complete** tensor and a plain-`ColwiseParallel` shard would corrupt the reshape. The plan
  therefore keeps the processors' view of the projections intact:

  * `attn.qkv`, `cross_attn.q_linear`, `cross_attn.kv_linear`, `audio_cross_attn.q_linear`,
    `audio_cross_attn.kv_linear`: `ColwiseParallel(output_layouts=Replicate())`, i.e. each rank
    computes a slice and the output is all-gathered back to full width;
  * `attn.proj`, `cross_attn.proj`, `audio_cross_attn.proj`: `RowwiseParallel(input_layouts=Replicate())`,
    because their input is the full (all-gathered) attention output;
  * `ffn.w1` / `ffn.w3`: `ColwiseParallel()`, `ffn.w2`: `RowwiseParallel()` — the SwiGLU pair is a
    clean Colwise -> Rowwise chain, no replicate needed;
  * no `shard_div_attr`: with `Replicate()` outputs the processors see the original head count and
    `head_dim`, so dividing `num_heads` would break the unflatten.

  Everything outside the stack (patch embedder, timestep MLP, text embedder, audio projection, the two
  adaLN modulation heads and `final_layer`) stays replicated: the residual stream is full width after
  each all-reduce.
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
    """Shard every block of the denoising stack.

    :param transformer: the vendored avatar transformer.
    :param tp_mesh: device mesh of the TP group.
    :returns:`(transformer, per-block layer plans)`.
    """

    tp_size = tp_mesh.size()
    layer_plans: List[Dict[str, ParallelStyle]] = []
    for block in transformer.blocks:
      assert isinstance(block, nn.Module), "LongCat-Video-Avatar blocks must be `nn.Module`."
      for module_name, linear in (
        ("attn.qkv", block.attn.qkv),
        ("attn.proj", block.attn.proj),
        ("cross_attn.q_linear", block.cross_attn.q_linear),
        ("cross_attn.kv_linear", block.cross_attn.kv_linear),
        ("cross_attn.proj", block.cross_attn.proj),
        ("audio_cross_attn.q_linear", block.audio_cross_attn.q_linear),
        ("audio_cross_attn.kv_linear", block.audio_cross_attn.kv_linear),
        ("audio_cross_attn.proj", block.audio_cross_attn.proj),
      ):
        if linear.out_features % tp_size != 0:
          raise ValueError(
            f"[TP] Unsupported tp_size={tp_size}: `{module_name}` has {linear.out_features} output "
            f"features, which is not divisible by tp_size.")

      layer_plan = {
        "attn.qkv": ColwiseParallel(output_layouts=Replicate()),
        "attn.proj": RowwiseParallel(input_layouts=Replicate()),
        "cross_attn.q_linear": ColwiseParallel(output_layouts=Replicate()),
        "cross_attn.kv_linear": ColwiseParallel(output_layouts=Replicate()),
        "cross_attn.proj": RowwiseParallel(input_layouts=Replicate()),
        "audio_cross_attn.q_linear": ColwiseParallel(output_layouts=Replicate()),
        "audio_cross_attn.kv_linear": ColwiseParallel(output_layouts=Replicate()),
        "audio_cross_attn.proj": RowwiseParallel(input_layouts=Replicate()),
        "ffn.w1": ColwiseParallel(),
        "ffn.w3": ColwiseParallel(),
        "ffn.w2": RowwiseParallel(),
      }
      parallelize_module(module=block, device_mesh=tp_mesh, parallelize_plan=layer_plan)
      layer_plans.append(layer_plan)

    return transformer, layer_plans
