"""Boogu-Image distributed parallelism planners (CP and TP).

Context parallelism covers all three stages of the vendored transformer:

* ``noise_refiner`` and ``ref_image_refiner`` — plain blocks with a 2D mask, so
  the declarative hook plan splits the hidden states at the first block, splits
  the per-layer RoPE table, and gathers after the last block.
* ``double_stream_layers`` — the block owns its bookkeeping (python length
  lists, the joint ``[instruct | image]`` RoPE layout, and the replicated
  fallback when the batch is not 1); see
  ``models/pipelines/boogu_image/transformer_boogu.py``.
* ``single_stream_layers`` — the fused joint sequence, handled by the hook plan.

``context_refiner`` stays replicated: its sequences are short and it consumes a
caller-supplied mask that may carry padding.
"""

from typing import Dict, List, Optional, Tuple

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

# Block stacks that the CP hook plan splits; `context_refiner` is deliberately
# absent (replicated).
_CP_REFINER_STACKS = ("noise_refiner", "ref_image_refiner")


def _set_cp_eligible_processors(transformer: nn.Module) -> None:
  """Set `_boogu_cp_eligible` on exactly the processors the CP plan shards.

  `_runtime` attaches `_cp_config` to every diffusers `Attention` processor, so
  eligibility is what keeps an unsplit stack (the context refiner) on the local
  path. The double-stream processors are toggled per forward by their block and
  start disabled here.

  :param transformer: The Boogu transformer being parallelized.
  """

  for module in transformer.modules():
    processor = getattr(module, "processor", None)
    if processor is not None:
      processor._boogu_cp_eligible = False

  for stack in _CP_REFINER_STACKS + ("single_stream_layers", ):
    for layer in getattr(transformer, stack, None) or []:
      processor = getattr(getattr(layer, "attn", None), "processor", None)
      if processor is not None:
        processor._boogu_cp_eligible = True


@ContextParallelismPlannerRegister.register("BooguImageTransformer")
class BooguImageContextParallelismPlanner(ContextParallelismPlanner):

  def _apply(
    self,
    transformer: Optional[torch.nn.Module] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:
    if transformer is None:
      return {}

    if getattr(parallelism_config, "ulysses_async", False):
      raise ValueError("Boogu-Image does not implement the async Ulysses path; drop "
                       "`ulysses_async` (synchronous Ulysses fully supports this model).")

    _set_cp_eligible_processors(transformer)

    plan: _ContextParallelModelPlan = {}

    # Refiner stacks: the mask stays full and the chunks keep rank order.
    for stack in _CP_REFINER_STACKS:
      layers = getattr(transformer, stack, None)
      if not layers:
        continue
      if len(layers) == 1:
        # A single-layer stack needs the split and the gather on the same module
        # key; the framework's list form registers both hooks there.
        plan[f"{stack}.0"] = [
          {
            "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3),
            "image_rotary_emb": _ContextParallelInput(split_dim=1, expected_dims=3),
          },
          _ContextParallelOutput(gather_dim=1, expected_dims=3),
        ]
        continue
      plan[f"{stack}.0"] = {
        "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3),
      }
      plan[f"{stack}.*"] = {
        "image_rotary_emb": _ContextParallelInput(split_dim=1, expected_dims=3),
      }
      plan[f"{stack}.{len(layers) - 1}"] = _ContextParallelOutput(gather_dim=1, expected_dims=3)

    # Double-stream stack: split both streams once at the first block and keep
    # every block shape invariant (local in -> local out), so caching a block
    # stack under CP still sees `output - input` of the same width. The joint
    # rope table is NOT split: the block stitches the local rope from the global
    # `[instruct | image]` layout itself. The transformer gathers both streams
    # after the stack, right before the fusion loop.
    double_stream_layers = getattr(transformer, "double_stream_layers", None)
    if double_stream_layers:
      plan["double_stream_layers.0"] = {
        "img_hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3),
        "instruct_hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3),
      }

    single_stream_layers = getattr(transformer, "single_stream_layers", None)
    if not single_stream_layers:
      return plan

    # `attention_mask` is not split: every rank holds the identical full mask and
    # Ulysses consumes it at full-sequence granularity.
    plan["single_stream_layers.0"] = {
      "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3),
    }
    plan["single_stream_layers.*"] = {
      "image_rotary_emb": _ContextParallelInput(split_dim=1, expected_dims=3),
    }
    return plan


def _dtensor_safe_swiglu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  """Pure-PyTorch SwiGLU that tolerates DTensor-sharded inputs.

  flash_attn's fused activation does not understand DTensor placements and dies
  with an illegal memory access once a `ColwiseParallel` feeds it a sharded
  tensor.

  :param x: Gate operand.
  :param y: Value operand.
  :returns: `silu(x) * y` in the input dtype.
  """

  return torch.nn.functional.silu(x.float()).to(x.dtype) * y


@TensorParallelismPlannerRegister.register("BooguImageTransformer")
class BooguImageTensorParallelismPlanner(TensorParallelismPlanner):

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

  @staticmethod
  def _single_stream_layer_plan(block: nn.Module) -> Dict[str, ParallelStyle]:
    """Shard the FFN and the modulation projection of one block.

    Attention projections stay replicated: `num_kv_heads=7` is not divisible by
    `tp_size=2`, and the all-gather a `Replicate()` workaround needs costs more
    than the sharded GEMMs save (measured on the same checkpoint: FFN-only
    108.9s vs 115.7s end to end, PSNR 50.2 vs 49.9).

    :param block: A single-stream / refiner block.
    :returns: The per-module parallel plan.
    """

    layer_plan: Dict[str, ParallelStyle] = {
      "feed_forward.linear_1": ColwiseParallel(),
      "feed_forward.linear_3": ColwiseParallel(),
      "feed_forward.linear_2": RowwiseParallel(),
    }
    if getattr(block.norm1, "linear", None) is not None:
      layer_plan["norm1.linear"] = ColwiseParallel(output_layouts=Replicate())
    return layer_plan

  @staticmethod
  def _double_stream_layer_plan(block: nn.Module) -> Dict[str, ParallelStyle]:
    """Shard both FFNs and all modulation projections of a double-stream block.

    :param block: A double-stream block.
    :returns: The per-module parallel plan.
    """

    layer_plan: Dict[str, ParallelStyle] = {
      "img_feed_forward.linear_1": ColwiseParallel(),
      "img_feed_forward.linear_3": ColwiseParallel(),
      "img_feed_forward.linear_2": RowwiseParallel(),
      "instruct_feed_forward.linear_1": ColwiseParallel(),
      "instruct_feed_forward.linear_3": ColwiseParallel(),
      "instruct_feed_forward.linear_2": RowwiseParallel(),
    }
    for mod_name in ("img_norm1", "img_norm2", "img_norm3", "instruct_norm1", "instruct_norm2"):
      if getattr(getattr(block, mod_name, None), "linear", None) is not None:
        layer_plan[f"{mod_name}.linear"] = ColwiseParallel(output_layouts=Replicate())
    return layer_plan

  def parallelize_transformer(
    self,
    transformer: nn.Module,
    tp_mesh: DeviceMesh,
  ) -> Tuple[nn.Module, List[Dict[str, ParallelStyle]]]:
    layer_plans: List[Dict[str, ParallelStyle]] = []

    for block_list_name in ("noise_refiner", "ref_image_refiner", "context_refiner"):
      block_list = getattr(transformer, block_list_name, None)
      if block_list is None:
        continue
      for _, block in block_list.named_children():
        self._make_ffn_dtensor_safe(block)
        layer_plan = self._single_stream_layer_plan(block)
        parallelize_module(module=block, device_mesh=tp_mesh, parallelize_plan=layer_plan)
        layer_plans.append(layer_plan)

    for _, block in transformer.double_stream_layers.named_children():
      self._make_ffn_dtensor_safe(block)
      layer_plan = self._double_stream_layer_plan(block)
      parallelize_module(module=block, device_mesh=tp_mesh, parallelize_plan=layer_plan)
      layer_plans.append(layer_plan)

    for _, block in transformer.single_stream_layers.named_children():
      self._make_ffn_dtensor_safe(block)
      layer_plan = self._single_stream_layer_plan(block)
      parallelize_module(module=block, device_mesh=tp_mesh, parallelize_plan=layer_plan)
      layer_plans.append(layer_plan)

    return transformer, layer_plans

  @staticmethod
  def _make_ffn_dtensor_safe(block: nn.Module) -> None:
    """Swap flash_attn's fused SwiGLU for the PyTorch one before sharding.

    :param block: Block whose feed-forward modules are about to be parallelized.
    """

    for ffn_name in ("feed_forward", "img_feed_forward", "instruct_feed_forward"):
      ffn = getattr(block, ffn_name, None)
      if ffn is None or not hasattr(ffn, "swiglu"):
        continue
      ffn.swiglu = _dtensor_safe_swiglu
