"""HY-Nomas transformer distributed parallelism planners (TP, CP).

HY-Nomas denoises one 54-layer double-stream stack (`img` video stream + replicated `txt` text
stream) and keeps a **persistent** per-layer KV cache across segment-autoregressive steps.  Two
properties of that shape drive the tensor-parallel plan:

* every projection is a plain ``nn.Linear`` on the block itself (``img_attn_q`` / ``txt_mlp.fc1`` /
  ``img_mod.linear`` — checkpoint names, no fused QKV, no nested attention module), so the plan is a
  flat list of column/row shards;
* the head constants the block reshapes with (``block.heads`` / ``block.attn_heads``) are owned by
  :meth:`NomasTransformer3DModel.set_parallel`, which already divides them by ``tp_size`` and
  ``tp_size * sp_size`` respectively.  The planner therefore **must not** call ``shard_div_attr`` on
  those blocks: doing so would divide the metadata twice.
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch.distributed import DeviceMesh
from torch.distributed._tensor import Replicate
from torch.distributed.tensor.parallel import (
  ColwiseParallel,
  ParallelStyle,
  RowwiseParallel,
  parallelize_module,
)

from ...logger import init_logger
from ..config import ParallelismConfig
from ..parallel import (
  _ContextParallelInput,
  _ContextParallelModelPlan,
  _ContextParallelOutput,
)
from .register import (
  ContextParallelismPlanner,
  ContextParallelismPlannerRegister,
  TensorParallelismPlanner,
  TensorParallelismPlannerRegister,
)

logger = init_logger(__name__)

# Streams of the double-stream block, in the checkpoint's prefix order.
_STREAMS = ("img", "txt")


@ContextParallelismPlannerRegister.register("Nomas")
class NomasContextParallelismPlanner(ContextParallelismPlanner):
  r"""Context-parallel planner for HY-Nomas.

  The video stream is sharded on the token axis and the text stream stays replicated: the block
  concatenates ``[video | text]`` for attention, so a split text stream would have to be gathered
  before every block.  Attention itself is already head/token exchanged inside the block
  (``ulysses_all_to_all`` over the CP sub-mesh), which is why the hook plan only has to move the
  sequence once:

  * **split** on ``double_blocks.0`` — ``img_in`` runs on the full sequence (cheap, no attention, and
    the RoPE table is sliced to this rank's range by ``prepare_segment_context``), then the first
    block receives the local video tokens.
  * **gather** on ``final_layer`` — it sits *outside* the block container, so a cached step (which
    skips blocks) still gathers and the output keeps the full token count.

  Because ``final_layer`` also runs on the KV-write pass, every segment pays one extra all-gather in
  addition to the per-block exchanges.

  Ring attention and ``--ulysses-anything`` are rejected: every rank's cache holds the **full**
  segment row count (``video_tokens + text_tokens``; the capacity is derived from the whole
  segment, not divided by ``sp_size``) and only its own head slice, which is exactly what the
  block's Ulysses all-to-all leaves behind.  A ring/variable partition would give ranks different
  row counts and break that layout.
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
        "HY-Nomas context parallelism does not support Ring attention: the segment KV cache stores "
        "an equal number of rows per rank, which only holds for the Ulysses equipartition. Use "
        "`--parallel ulysses`.")
    if parallelism_config.ulysses_anything:
      raise NotImplementedError(
        "HY-Nomas context parallelism does not support `--ulysses-anything`: the video tokens are "
        "split evenly and the cache capacity is derived from that split. Use the default Ulysses "
        "partition.")

    sp_size = parallelism_config.ulysses_size or 1
    tp_size = parallelism_config.tp_size or 1
    heads = transformer.config.num_attention_heads
    shards = tp_size * sp_size
    if heads % shards:
      raise ValueError(f"HY-Nomas has {heads} attention heads, which is not divisible by "
                       f"tp_size * ulysses_size = {shards}.")

    blocks = transformer.double_blocks
    if len(blocks) != transformer.config.num_layers:
      raise ValueError(f"HY-Nomas context parallelism expects {transformer.config.num_layers} "
                       f"double-stream blocks, found {len(blocks)}; the stack is probably already "
                       f"wrapped by the cache runtime.")
    final_layer = getattr(transformer, "final_layer", None)
    if final_layer is None:
      raise ValueError("HY-Nomas context parallelism needs `transformer.final_layer` to gather.")
    if any(final_layer is block for block in blocks):
      raise AssertionError("final_layer must stay outside the cached block container: the gather "
                           "has to run on a cached step too.")

    # Head metadata only; the CP process group arrives later via `_cp_config`.
    transformer.set_context_parallel(sp_size)

    _cp_plan: _ContextParallelModelPlan = {
      "double_blocks.0": {
        "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3),
      },
      "final_layer": _ContextParallelOutput(gather_dim=1, expected_dims=3),
    }
    logger.debug("Generated HY-Nomas CP plan (sp_size=%d): %s", sp_size, _cp_plan)
    return _cp_plan


@TensorParallelismPlannerRegister.register("Nomas")
class NomasTensorParallelismPlanner(TensorParallelismPlanner):
  r"""Tensor-parallel planner for HY-Nomas.

  Each ``NomasDoubleStreamBlock`` owns, per stream, a separate ``q`` / ``k`` / ``v`` projection, an
  output projection, a two-layer MLP and one modulation projection:

  * ``{stream}_attn_{q,k,v}`` — ``ColwiseParallel``: the block immediately does
    ``.unflatten(-1, (self.heads, -1))``, and ``set_parallel`` has already divided ``heads`` by
    ``tp_size``, so the per-rank head count keeps ``head_dim`` at its release value.
  * ``{stream}_attn_proj`` / ``{stream}_mlp.fc2`` — ``RowwiseParallel``: they close the residual and
    all-reduce back to the full feature width.
  * ``{stream}_mlp.fc1`` — ``ColwiseParallel``.
  * ``{stream}_mod.linear`` — ``ColwiseParallel(output_layouts=Replicate())``: the block reshapes the
    projection with ``.chunk(6, -1)`` and applies the six pieces to a **full-width** activation, so
    the modulation output must be all-gathered.  ``Replicate`` is a correctness requirement here,
    not an optimization; the sharded weight still halves the resident parameters.
  * ``{stream}_attn_{q,k}_norm`` — untouched: ``NomasRMSNorm(d // heads)`` normalises the last axis
    of a ``[B, S, heads, head_dim]`` tensor, i.e. per head over ``head_dim``.  ``head_dim`` is frozen
    across ranks, so the module is replicated and no distributed RMSNorm is needed.

  ``img_in`` / ``txt_in`` / ``time_in`` / ``final_layer`` / ``action_modules`` stay replicated: they
  either produce or consume the full feature width (``final_layer.linear`` writes ``out_channels``
  per token, and the camera attention projects ``hidden_size -> 1024``), and the block stack
  dominates both compute and resident memory.
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

  @staticmethod
  def layer_plan() -> Dict[str, ParallelStyle]:
    """Return the shard plan of one double-stream block.

    :returns: Plan entries keyed by attribute paths inside the block.
    """
    plan: Dict[str, ParallelStyle] = {}
    for stream in _STREAMS:
      for name in ("q", "k", "v"):
        plan[f"{stream}_attn_{name}"] = ColwiseParallel()
      plan[f"{stream}_attn_proj"] = RowwiseParallel()
      plan[f"{stream}_mlp.fc1"] = ColwiseParallel()
      plan[f"{stream}_mlp.fc2"] = RowwiseParallel()
      # Full-width consumer: `.chunk(6, -1)` of a sharded output would misalign the six tables.
      plan[f"{stream}_mod.linear"] = ColwiseParallel(output_layouts=Replicate())
    return plan

  def parallelize_transformer(
    self,
    transformer: torch.nn.Module,
    tp_mesh: DeviceMesh,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    tp_size = tp_mesh.size()
    blocks = transformer.double_blocks
    if len(blocks) != transformer.config.num_layers:
      raise ValueError(f"HY-Nomas tensor parallelism expects {transformer.config.num_layers} "
                       f"double-stream blocks, found {len(blocks)}; the stack is probably already "
                       f"wrapped by the cache runtime.")

    # Owns `block.heads` (heads // tp) and `block.attn_heads` (heads // (tp * sp)), and fails fast
    # when the release head count is not divisible.  Do not call `shard_div_attr` afterwards: the
    # metadata would be divided twice.
    transformer.set_tensor_parallel(tp_size)

    template = self.layer_plan()
    layer_plans: List[Dict[str, ParallelStyle]] = []
    for block in blocks:
      layer_plan = dict(template)
      parallelize_module(
        module=block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_plan,
      )
      layer_plans.append(layer_plan)
    logger.debug("Generated HY-Nomas TP plan (tp_size=%d, %d blocks)", tp_size, len(blocks))
    return transformer, layer_plans
