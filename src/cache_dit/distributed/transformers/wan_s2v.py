"""Context- and tensor-parallelism planners for Wan2.2-S2V.

The CP plan and the TP plan cannot reuse the Wan-family ones: the class name
(`WanS2VTransformer3DModel`) does not match the CP prefix `"WanTransformer3D"`, and the TP plan of
Wan has no notion of the audio injector, which is a stack of cross-attention layers that lives
outside `blocks`.
"""

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
from ..utils import shard_div_attr
from .register import (
  ContextParallelismPlanner,
  ContextParallelismPlannerRegister,
  TensorParallelismPlanner,
  TensorParallelismPlannerRegister,
)


@ContextParallelismPlannerRegister.register("WanS2V")
class WanS2VContextParallelismPlanner(ContextParallelismPlanner):
  """Context parallelism for Wan2.2-S2V (Ulysses).

  Split points, and why they are where they are:

  - `blocks.0.hidden_states` carries the whole packed sequence (video tokens, reference-image
    tokens, motion tokens) and is the only sequence tensor the block loop reads.
  - `proj_out` gathers the sequence back. The reference-image and motion tokens are still in the
    sequence at that point so that every rank holds the same number of tokens; the transformer
    drops them right after the gather.

  The RoPE table is sharded inside `transformer.forward`, not by this plan: it is one tensor shared
  by every block, so a `blocks.0` input hook would leave blocks 1..N-1 rotating their shard with
  the wrong positions, and an output hook on `rope` would miss the motion-token rotary embedding
  that `inject_motion` concatenates afterwards.

  The audio injection rebuilds whole frames from the gathered sequence on its own (see
  :class:`cache_dit.models.pipelines.wan_s2v.transformer.AudioInjector`), and the segment boundary
  of the modulation is recomputed per rank inside `forward`, so nothing else needs to be split.
  `ulysses_anything` is not supported: padding the sequence would break the frame-aligned audio
  injection.
  """

  def _apply(
    self,
    transformer: Optional[torch.nn.Module] = None,
    parallelism_config=None,
    **kwargs,
  ) -> _ContextParallelModelPlan:
    if getattr(parallelism_config, "ulysses_anything", False):
      raise NotImplementedError(
        "Wan2.2-S2V does not support `--ulysses-anything`: the audio injector all-gathers an equal "
        "`local_len` from every rank and then rebuilds frames from the unpadded offsets, so padded "
        "splits hand it lengths it cannot reconcile. Use a CP degree that divides the sequence.")

    _cp_plan = {
      "blocks.0": {
        "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
      },
      "proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
    }
    return _cp_plan


@TensorParallelismPlannerRegister.register("WanS2V")
class WanS2VTensorParallelismPlanner(TensorParallelismPlanner):
  """Tensor parallelism for Wan2.2-S2V.

  Same sharding as Wan for the transformer blocks, plus the audio injector: its cross-attention
  layers are ordinary `WanAttention` modules whose K/V inputs are replicated per frame, so they
  shard exactly like `attn2`. The AdaIN / pre-norm layers stay replicated because their output is a
  full-width modulation of the hidden states.
  """

  def _apply(
    self,
    transformer: torch.nn.Module,
    parallelism_config=None,
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
    # Imported here, not at module scope: importing `.wan` registers the Wan planners, and the
    # registry matches the first registered prefix, so "WanS2V" would lose to "Wan".
    from .wan import DistributedRMSNorm

    tp_size = tp_mesh.size()

    def attention_plan(attn: nn.Module) -> Dict[str, ParallelStyle]:
      """Shard one `WanAttention` in place and return its module-local plan."""
      shard_div_attr(attn, "heads", tp_size)
      plan = {
        "to_q": ColwiseParallel(),
        "to_k": ColwiseParallel(),
        "to_v": ColwiseParallel(),
        "to_out.0": RowwiseParallel(),
      }
      if getattr(attn, "add_k_proj", None) is not None:
        plan["add_k_proj"] = ColwiseParallel()
        plan["add_v_proj"] = ColwiseParallel()

      # `norm_q` / `norm_k` are RMS norms over the full head width; after sharding the heads each
      # rank only holds a slice, so the mean-square has to be reduced across the TP group.
      attn.norm_q = DistributedRMSNorm.from_rmsnorm(tp_mesh, attn.norm_q)
      attn.norm_k = DistributedRMSNorm.from_rmsnorm(tp_mesh, attn.norm_k)
      if getattr(attn, "norm_added_k", None) is not None:
        attn.norm_added_k = DistributedRMSNorm.from_rmsnorm(tp_mesh, attn.norm_added_k)
      return plan

    def prepare_block(block: nn.Module) -> Dict[str, ParallelStyle]:
      layer_plan = {}
      for name, style in attention_plan(block.attn1).items():
        layer_plan[f"attn1.{name}"] = style
      for name, style in attention_plan(block.attn2).items():
        layer_plan[f"attn2.{name}"] = style
      layer_plan["ffn.net.0.proj"] = ColwiseParallel()
      layer_plan["ffn.net.2"] = RowwiseParallel()
      parallelize_module(module=block, device_mesh=tp_mesh, parallelize_plan=layer_plan)
      return layer_plan

    layer_plans = []
    for _, block in transformer.blocks.named_children():
      layer_plans.append(prepare_block(block))

    audio_injector = getattr(transformer, "audio_injector", None)
    if audio_injector is not None:
      for name, injector in audio_injector.injector.named_children():
        local_plan = attention_plan(injector)
        parallelize_module(module=injector, device_mesh=tp_mesh, parallelize_plan=local_plan)
        layer_plans.append({
          f"audio_injector.injector.{name}.{key}": style
          for key, style in local_plan.items()
        })

    return transformer, layer_plans
