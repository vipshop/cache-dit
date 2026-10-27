"""Ideogram-4 distributed parallelism planners (CP + TP).

Ideogram-4 packs the text-conditioning tokens and the patchified image latents into one
sequence (``[left-pad][text][image]``) and hands the transformer a block-diagonal attention
mask derived from ``segment_ids``.  Every per-token operand is already a plain sequence
tensor at the transformer entry, so context parallelism splits the packed sequence there
instead of patching ``forward``: ``hidden_states``, ``encoder_hidden_states``,
``position_ids`` and ``indicator`` are sharded on their sequence axis, the block loop runs
on the local chunk, and ``final_layer`` gathers the full sequence back for the velocity
prediction.

``segment_ids`` is deliberately **not** split.  The mask built inside the forward has shape
``(B, 1, L, L)`` and describes the global key axis, while Ulysses all-to-all concatenates
the per-rank contiguous chunks in rank order.  The recovered global order therefore equals
the original packed order, so the full mask stays valid and no mask permutation is needed.
"""

import functools
from typing import Dict, List, Optional, Tuple

import torch
from torch.distributed import DeviceMesh
from torch.distributed.tensor.parallel import (
  ColwiseParallel,
  ParallelStyle,
  RowwiseParallel,
  parallelize_module,
)

from diffusers.models.transformers.transformer_ideogram4 import (
  Ideogram4AttnProcessor,
  _rotate_half,
)

from ...attention import _dispatch_attention_fn
from ...logger import init_logger
from ..config import ParallelismConfig
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

logger = init_logger(__name__)


def _patch_attention_processor_for_cp() -> None:
  """Monkey-patch ``Ideogram4AttnProcessor.__call__`` onto cache-dit's CP dispatch.

  Upstream calls diffusers' ``dispatch_attention_fn`` with ``parallel_config``, which only
  understands diffusers' own ``ParallelConfig``.  The fork keeps the exact upstream math
  order -- Q/K projection, ``norm_q``/``norm_k`` RMSNorm, then MRoPE on the
  ``(B, L, H, D)`` layout -- and only swaps the dispatch for
  :func:`cache_dit.attention._dispatch_attention_fn`, which receives the CP config and runs
  the Ulysses all-to-all.  ``_rotate_half`` is imported from the diffusers module rather
  than duplicated, so a future upstream RoPE change surfaces here instead of silently
  diverging from the reference path.

  The fork is inert while the processor carries no ``_cp_config`` (a TP-only or cp-free
  run): it hands control back to the saved upstream ``__call__`` and stays bit-exact.  The
  patch is class-level and idempotent, so it also reaches processors of a second attention
  tree (e.g. ``unconditional_transformer``) that this runtime never armed.
  """
  if getattr(Ideogram4AttnProcessor, "_cp_patched", False):
    return

  _original_call = Ideogram4AttnProcessor.__call__

  @functools.wraps(_original_call)
  def _patched_call(
    self: Ideogram4AttnProcessor,
    attn: torch.nn.Module,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    image_rotary_emb: Tuple[torch.Tensor, torch.Tensor],
  ) -> torch.Tensor:
    cp_config = getattr(self, "_cp_config", None)
    if cp_config is None:
      return _original_call(self, attn, hidden_states, attention_mask, image_rotary_emb)

    query = attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
    key = attn.to_k(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
    value = attn.to_v(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))

    query = attn.norm_q(query)
    key = attn.norm_k(key)

    # MRoPE is applied in the (B, L, num_heads, head_dim) layout; cos/sin broadcast over heads.
    cos, sin = image_rotary_emb
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    query = (query * cos) + (_rotate_half(query) * sin)
    key = (key * cos) + (_rotate_half(key) * sin)

    # The mask is the full (B, 1, L, L) segment mask: Ulysses recovers the global key order.
    hidden_states = _dispatch_attention_fn(
      query,
      key,
      value,
      attn_mask=attention_mask,
      backend=getattr(self, "_attention_backend", None),
      cp_config=cp_config,
    )
    hidden_states = hidden_states.flatten(2, 3)
    return attn.to_out[0](hidden_states)

  Ideogram4AttnProcessor.__call__ = _patched_call
  Ideogram4AttnProcessor._cp_patched = True


@ContextParallelismPlannerRegister.register("Ideogram4Transformer2DModel")
class Ideogram4ContextParallelismPlanner(ContextParallelismPlanner):
  """Context-parallel planner for Ideogram-4.

  The plan is a single root split plus one gather:

  - root forward: ``hidden_states``, ``encoder_hidden_states`` and ``position_ids`` are
    split on dim 1 with 3 dims, ``indicator`` on dim 1 with 2 dims.  Everything the forward
    derives from them before the block loop (token masks, ``input_proj``,
    ``llm_cond_norm``/``llm_cond_proj``, ``embed_image_indicator``, ``rotary_emb``) is
    per-token, so it stays local and no other tensor needs a hook.
  - ``segment_ids`` is not split: the ``(B, 1, L, L)`` mask has to describe the global key
    axis, and Ulysses' rank-ordered chunk concat restores the original packed order.
  - ``final_layer`` gathers dim 1 back to the full sequence, so the returned velocity keeps
    the packed layout the pipeline slices the image region from.
  - ``timestep``-derived ``adaln_input`` is ``(B, 1, adaln_dim)``, i.e. sequence
    independent, and is recomputed identically on every rank.

  Ring attention is rejected because the model always builds an ``attn_mask`` and the Ring
  path needs ``return_lse``, which the native backend refuses together with a mask.
  ``ulysses_anything`` is rejected because the mask describes the equipartitioned global
  key axis: uneven shards would break both the mask length and the pack layout.
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
        "Ideogram-4 context parallelism does not support Ring attention: the segment mask is "
        "always passed to attention, and Ring needs `return_lse`, which the native backend "
        "rejects together with a mask. Use Ulysses (`--parallel ulysses`) instead.")
    if parallelism_config.ulysses_anything:
      raise NotImplementedError(
        "Ideogram-4 context parallelism does not support `--ulysses-anything`: the "
        "`(B, 1, L, L)` segment mask describes the equipartitioned global key axis, so uneven "
        "shards would desynchronize the mask length from the all-to-all layout. Choose a "
        "cp_size that divides both the packed conditional length and the image-only "
        "unconditional length.")
    backend = parallelism_config.attention_backend
    if backend not in (None, "native", "_sdpa_cudnn"):
      raise NotImplementedError(
        f"Ideogram-4 context parallelism requires a mask-capable native backend, got {backend!r}."
        " Use `--attn native`.")

    _patch_attention_processor_for_cp()

    _cp_plan: _ContextParallelModelPlan = {
      "": {
        "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3),
        "encoder_hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3),
        "position_ids": _ContextParallelInput(split_dim=1, expected_dims=3),
        "indicator": _ContextParallelInput(split_dim=1, expected_dims=2),
      },
      "final_layer": _ContextParallelOutput(gather_dim=1, expected_dims=3),
    }
    return _cp_plan


@TensorParallelismPlannerRegister.register("Ideogram4Transformer2DModel")
class Ideogram4TensorParallelismPlanner(TensorParallelismPlanner):
  """Tensor-parallel planner for Ideogram-4.

  Every block is a plain pre-norm transformer block: MHA with separate
  ``to_q``/``to_k``/``to_v`` (``num_heads=18``, ``head_dim=256``) and a SwiGLU
  ``feed_forward``.  Heads divide by ``tp_size``, so ``shard_div_attr`` shrinks
  ``attention.num_heads`` and the processor's ``unflatten(-1, (num_heads, head_dim))`` keeps
  ``head_dim`` intact for the Q/K RMSNorm and MRoPE.  ``to_out.0`` and
  ``feed_forward.w2`` are ``RowwiseParallel`` and all-reduce the partial outputs.

  ``adaln_modulation`` stays replicated on purpose: its output is consumed as
  ``mod.chunk(4, dim=-1)`` into scale/gate pairs, and a contiguous column shard would move
  those chunk boundaries.  The same reasoning keeps the entry/exit projections
  (``input_proj``, ``llm_cond_proj``, ``llm_cond_norm``, ``final_layer``) replicated -- they
  are a handful of GEMMs on the full hidden size, so sharding them would only add
  collectives to the critical path.
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
    for _, block in transformer.layers.named_children():
      shard_div_attr(block.attention, "num_heads", tp_size)
      layer_plan: Dict[str, ParallelStyle] = {
        "attention.to_q": ColwiseParallel(),
        "attention.to_k": ColwiseParallel(),
        "attention.to_v": ColwiseParallel(),
        "attention.to_out.0": RowwiseParallel(),
        "feed_forward.w1": ColwiseParallel(),
        "feed_forward.w3": ColwiseParallel(),
        "feed_forward.w2": RowwiseParallel(),
      }
      parallelize_module(module=block, device_mesh=tp_mesh, parallelize_plan=layer_plan)
      layer_plans.append(layer_plan)
    return transformer, layer_plans
