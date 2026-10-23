"""Context/tensor parallel plans for Bria Fibo 2.0 transformers.

Fibo 2.0 is a single-stream DiT (`layers`) with a Z-Image style block contract: the block takes
`x`, `attn_mask`, `freqs_cis` and a per-token `noise_mask`, and returns `x`. Both position ids and
the noisy/clean token selector are therefore block *inputs*, which lets context parallelism split
them at the block boundary instead of patching the RoPE builder (`RopeEmbedder` is not a module and
caches its tables).
"""

from typing import Dict, List, Optional, Tuple

import torch
from diffusers.models.modeling_utils import ModelMixin
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


@ContextParallelismPlannerRegister.register("BriaFibo2Transformer2DModel")
class BriaFibo2ContextParallelismPlanner(ContextParallelismPlanner):
  """CP planner for Bria Fibo 2.0 — per-layer split/gather around the packed token sequence."""

  def _apply(
    self,
    transformer: Optional[torch.nn.Module | ModelMixin] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:
    assert transformer is not None, "Transformer must be provided."

    # Context parallelism owns the block loop, so the patch functor has to be in place *before* the
    # hooks are installed: without it the loop is never collapsed by the cache wrapper (this path)
    # and the per-block text bundles are lost once a cache is enabled on top of it. The functor is
    # idempotent (`_is_patched`), so the cache path can apply it as well.
    if not getattr(transformer, "_is_patched", False):
      from ...caching.patch_functors import BriaFibo2PatchFunctor

      BriaFibo2PatchFunctor().apply(transformer)

    if parallelism_config.ring_size is not None and parallelism_config.ring_size > 1:
      raise NotImplementedError(
        "Ring attention is not supported for Bria Fibo 2.0: when a batch pads its gist tokens to "
        "the largest count, the resulting 2-D mask describes the global key axis, which only "
        "Ulysses keeps valid. Use `--parallel ulysses` instead.")
    if parallelism_config.ulysses_anything:
      raise NotImplementedError(
        "ulysses_anything is not supported for Bria Fibo 2.0: the packed sequence mixes gist, image "
        "and context tokens, so shards of uneven length no longer line up with the RoPE table. "
        "Choose a cp_size that divides the packed token count.")

    n_layers = len(transformer.layers)

    # The packed sequence is `[gist | image | context]` (or `[image | context | gist]` when a batch
    # pads its gist tokens). Splitting it contiguously at every block keeps the token order intact,
    # so Ulysses' all-to-all restores the global order and the key-side `attn_mask` stays full length
    # (`BriaFibo2AttnProcessor` broadcasts it over the query axis).
    #
    # `freqs_cis` and `noise_mask` are re-passed to *every* block by the transformer forward, so they
    # need the wildcard entry; only `x` would otherwise stay sharded. The output gather also runs per
    # layer: `final_layer` slices the packed sequence with a global `image_start`, so the sequence has
    # to be whole again before the loop ends — including on a cached step, where the middle blocks
    # (and therefore a gather attached to the last block only) never run.
    #
    # The price is two collectives per layer instead of one `proj_out` gather. `noise_refiner` and
    # `perceiver` run before the split and stay replicated on every rank.
    _cp_plan: _ContextParallelModelPlan = {
      "layers.*": {
        "x": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        "freqs_cis": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        "noise_mask": _ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
      },
      **{
        f"layers.{i}": _ContextParallelOutput(gather_dim=1, expected_dims=3)
        for i in range(n_layers)
      },
    }

    return _cp_plan


@TensorParallelismPlannerRegister.register("BriaFibo2Transformer2DModel")
class BriaFibo2TensorParallelismPlanner(TensorParallelismPlanner):
  """TP planner for Bria Fibo 2.0 — MHA heads plus the SwiGLU FFN, modulation replicated."""

  def _apply(
    self,
    transformer: torch.nn.Module,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    # Same reason as the CP planner: sharding first would make `PatchFunctor.apply` skip the model,
    # and a later `enable_cache` would then collapse the loop without the per-block text bundles.
    if not getattr(transformer, "_is_patched", False):
      from ...caching.patch_functors import BriaFibo2PatchFunctor

      BriaFibo2PatchFunctor().apply(transformer)

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
    tp_size = tp_mesh.get_group().size()

    def tp_shard_block(block) -> Dict[str, ParallelStyle]:
      shard_div_attr(block.attention, "heads", tp_size)
      layer_plan: Dict[str, ParallelStyle] = {
        "attention.to_q": ColwiseParallel(),
        "attention.to_k": ColwiseParallel(),
        "attention.to_v": ColwiseParallel(),
        "attention.to_out.0": RowwiseParallel(),
        # SwiGLU FFN: w1/w3 fan out into the sharded hidden dim, w2 all-reduces back.
        "feed_forward.w1": ColwiseParallel(),
        "feed_forward.w3": ColwiseParallel(),
        "feed_forward.w2": RowwiseParallel(),
        # `adaLN_modulation` stays replicated: its output is consumed as `chunk(4, dim=-1)`, which a
        # contiguous output shard would split at the wrong boundaries.
      }
      if block.cross_attention is not None:
        shard_div_attr(block.cross_attention, "heads", tp_size)
        layer_plan.update({
          "cross_attention.to_q": ColwiseParallel(),
          "cross_attention.to_k": ColwiseParallel(),
          "cross_attention.to_v": ColwiseParallel(),
          "cross_attention.to_out.0": RowwiseParallel(),
        })
      parallelize_module(module=block, device_mesh=tp_mesh, parallelize_plan=layer_plan)
      return layer_plan

    # `noise_refiner` and `perceiver` keep a full copy: both run on the pre-loop sequence (2 layers
    # and a handful of gist queries), so sharding them would only add collectives.
    layer_plans = [tp_shard_block(block) for block in transformer.layers]

    return transformer, layer_plans
