"""MiniMax-H3-Image transformer distributed parallelism planners (CP + TP).

MiniMax-H3-Image is MiniMax-H3 **without its audio stream**: one packed 1-D sequence of text rows,
reference-image rows and generated rows runs through a single stack of 50 blocks, and a block reads
per-row `adaln_indices` to pick its modulation from a `(timestep, modality)` table. There is no
cross-attention, no attention mask and no classifier-free guidance, so every step is one forward
pass.

Two things differ from the video model and drive both plans:

* the checkpoint is distilled with Parallel Decoding Distillation, so it carries **one output head
  per sampling step** (`pdd_proj_out[i]`, six of them) instead of a single `proj_out` /
  `audio_proj_out` pair;
* `forward` scatters the rows into the packed buffer with `index_copy` over *global* row indices,
  so the sequence can only be split once the buffer exists — at the first block.

The planners therefore reuse the MiniMax-H3 shape: CP replaces the two output-head gathers with the
six per-step heads, and TP reuses the video block plan verbatim (same attention, same fused SwiGLU
FFN, same `adaln_proj` / `norm_out` modulation tables).
"""

from typing import Dict, List, Tuple

import torch
from torch.distributed import DeviceMesh
from torch.distributed.tensor.parallel import ParallelStyle

from ...logger import init_logger
from ..config import ParallelismConfig
from ..parallel import _ContextParallelInput, _ContextParallelOutput
from .minimax_h3 import MiniMaxH3TensorParallelismPlanner
from .register import (
  ContextParallelismPlanner,
  ContextParallelismPlannerRegister,
  TensorParallelismPlannerRegister,
)

logger = init_logger(__name__)


@ContextParallelismPlannerRegister.register("MiniMaxH3ImageTransformer2DModel")
class MiniMaxH3ImageContextParallelismPlanner(ContextParallelismPlanner):
  r"""Context parallelism planner for `MiniMaxH3ImageTransformer2DModel`.

  **The split happens at the first block, not at the root forward.** `forward` scatters the text and
  image streams into the packed buffer with `index_copy` over *global* row indices (`text_indices`,
  `image_indices`), so its inputs cannot be sharded: a sliced `hidden_states` would be scattered with
  indices that address the full sequence. Everything the block stack then reads per row is sharded
  alongside the packed buffer instead:

  * `rope` returns one `(cos, sin)` row of angles per row of the sequence, so both outputs are split
    (`split_output=True`);
  * `transformer_blocks.0` splits `hidden_states` once, after the buffer is built;
  * every block splits its own copy of `adaln_indices`, since each one reads it from `forward`'s
    unsharded tensor;
  * `norm_out` splits `timestep_indices` for the same reason;
  * every per-step output head gathers its rows back before `forward` selects the image rows with
    global indices again.

  **The plan order matters.** The cached path replays the first split hook of the first block to
  re-derive the residual of a skipped step, so `transformer_blocks.0` has to register
  `hidden_states` before `transformer_blocks.*` registers `adaln_indices` on the same block.
  Otherwise the replayed hook would slice a 3-D hidden state with a 1-D plan, silently leave the
  hidden state unsharded, and change what the cache compares. Keep the key order below.

  **The gather sits on the output heads, not on the block stack.** `norm_out` and every head are
  row-wise (an RMSNorm, a per-row `index_select` modulation and a `Linear`), so computing them on the
  local shard and gathering afterwards is numerically identical — and it moves `(B, S, 96)` instead
  of `(B, S, 5376)`, i.e. 56x less traffic at bfloat16. Only one of the six heads runs per step; the
  other five hooks simply never fire, which is correct (every rank runs the same step index).

  `position_ids` and `token_tags` must **not** be split: `forward` validates their shapes against the
  full sequence and uses them for the global scatter.

  The packed sequence is `image_rows + text_rows`, and the text length follows the prompt, so it
  rarely divides by the CP size — prefer `--ulysses-anything` unless the length is known to divide.
  """

  def _apply(
    self,
    transformer: torch.nn.Module,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> dict:
    if getattr(parallelism_config, "ulysses_async", False):
      raise ValueError("Async Ulysses (`--ulysses-async`) is not supported for "
                       "MiniMaxH3ImageTransformer2DModel: the async registry has no patch for its "
                       "attention processor. Drop the flag or use another model.")

    ulysses_size = parallelism_config.ulysses_size or 1
    ring_size = parallelism_config.ring_size or 1
    tp_size = parallelism_config.tp_size or 1
    heads = transformer.config.num_attention_heads

    # CP runs before TP, and Ulysses shards the heads of the *local* attention, so the constraint is
    # on the post-TP head count: 56 heads give 28 (tp2) or 14 (tp4). Both divide by cp2; only the tp2
    # count divides by cp4 as well.
    if ulysses_size > 1 and not parallelism_config.ulysses_anything:
      local_heads = heads // tp_size
      if local_heads % ulysses_size != 0:
        raise ValueError(
          f"MiniMaxH3ImageTransformer2DModel has {heads} attention heads, i.e. {local_heads} per "
          f"rank after tp_size={tp_size}, which is not divisible by ulysses_size={ulysses_size}. "
          "Ulysses shards heads, so pick a combination that divides, e.g. tp2+cp2 (28 heads) or "
          "tp4+cp2 (14 heads), or enable --ulysses-anything.")

    if (ulysses_size > 1 or ring_size > 1) and not parallelism_config.ulysses_anything:
      logger.warning(
        "MiniMaxH3ImageTransformer2DModel packs text + image rows into one sequence whose length "
        "follows the prompt, so it rarely divides by the context-parallel size; pass "
        "--ulysses-anything unless that length is known to divide.")

    return {
      # `rope` computes one row of angles per row of the packed sequence.
      "rope": {
        0: _ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
        1: _ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
      },
      # Must stay first on block 0: the cached path replays this hook.
      "transformer_blocks.0": {
        "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3),
      },
      "transformer_blocks.*": {
        "adaln_indices": _ContextParallelInput(split_dim=0, expected_dims=1),
      },
      "norm_out": {
        "timestep_indices": _ContextParallelInput(split_dim=0, expected_dims=1),
      },
      # Per-step output heads; the wildcard expands to the whole `pdd_proj_out` ModuleList.
      "pdd_proj_out.*": _ContextParallelOutput(gather_dim=1, expected_dims=3),
    }


@TensorParallelismPlannerRegister.register("MiniMaxH3ImageTransformer2DModel")
class MiniMaxH3ImageTensorParallelismPlanner(MiniMaxH3TensorParallelismPlanner):
  r"""Tensor parallelism planner for `MiniMaxH3ImageTransformer2DModel`.

  The block stack is identical to MiniMax-H3's (same attention, same fused SwiGLU feed-forward, same
  `adaln_proj` modulation table), so the video plan is reused as-is: `to_q` / `to_k` / `to_v` and
  `ff.net.0.proj` colwise, `to_out.0` and `ff.net.2` rowwise, and the `(timestep, modality)` table
  colwise with a replicated output because the block does `view(-1, 6 * hidden).chunk(6)` and then
  `index_select`s rows by `adaln_indices`.

  `token_refiner`, `proj_in`, `context_embedder`, `time_embedder` and the per-step output heads stay
  replicated. The real reason is that the residual stream is always `Replicate` under this plan —
  `to_out.0` and `ff.net.2` are both `RowwiseParallel`, which all-reduces — so `norm_out` and the
  heads can read the full-width hidden state; the six heads being 3.1M parameters each is only an
  extra argument. Do not turn `to_out.0` into a colwise projection without revisiting this.

  `shard_div_attr(attn, "heads", tp_size)` comes along from the inherited plan for property
  consistency only: this model's attention processor reshapes Q/K/V with
  `unflatten(-1, (-1, attn.head_dim))` and never reads `attn.heads`.
  """

  def _apply(
    self,
    transformer: torch.nn.Module,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    # The inherited plan shards the separate `to_q` / `to_k` / `to_v` projections, and its own guard
    # names the video transformer; fail here with this model's name instead of leaking that one.
    if getattr(transformer.transformer_blocks[0].attn, "fused_projections", False):
      raise ValueError(
        f"{type(transformer).__name__} has `attn.fused_projections` set (by "
        "`fuse_qkv_projections()`), which this plan does not shard: the block runs one `to_qkv` "
        "projection and would reshape Q/K/V with the already-sharded `heads`. Keep the projections "
        "separate.")

    tp_mesh: DeviceMesh = self.mesh(parallelism_config=parallelism_config)
    return self.parallelize_transformer(transformer=transformer, tp_mesh=tp_mesh)
