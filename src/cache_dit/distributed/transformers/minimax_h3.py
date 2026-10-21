"""MiniMax-H3 transformer distributed parallelism planners (CP + TP).

MiniMax-H3 denoises **one packed sequence** through a single stack of blocks: text rows, keyframe
conditioning rows, audio rows and video rows all live in the same 1-D sequence, and a block reads
per-row `adaln_indices` to pick its modulation from a `(timestep, modality)` table. There is no
cross-attention, no attention mask and no classifier-free guidance, so every step is one forward
pass.

The two planners follow that shape:

* CP shards the packed sequence once, at the first block, and carries the per-row tensors that the
  block stack, the output norm and the rotary embedding read.
* TP shards the attention and FFN projections of every block, keeps the token refiner and the
  per-modality input / output projections replicated, and shards the AdaLN tables while
  all-gathering their output.
"""

from typing import Dict, List, Tuple

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
from ..core import (
  _ContextParallelInput,
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


@ContextParallelismPlannerRegister.register("MiniMaxH3Transformer3DModel")
class MiniMaxH3ContextParallelismPlanner(ContextParallelismPlanner):
  r"""Context parallelism planner for `MiniMaxH3Transformer3DModel`.

  **The split happens at the first block, not at the root forward.** `forward` scatters the three
  input streams into the packed buffer with `index_copy` over *global* row indices
  (`video_indices`, `audio_indices`, `text_indices`), so its inputs cannot be sharded: a sliced
  `hidden_states` would be scattered with indices that address the full sequence. Everything the
  block stack then reads per row is sharded alongside the packed buffer instead:

  * `rope` returns one `(cos, sin)` row of angles per row of the sequence, so both outputs are
    split (`split_output=True`);
  * `transformer_blocks.0` splits `hidden_states` once, after the buffer is built;
  * every block splits its own copy of `adaln_indices`, since each one reads it from `forward`'s
    unsharded tensor;
  * `norm_out` splits `timestep_indices` for the same reason;
  * both output heads gather their rows back before `forward` selects the video / audio rows with
    global indices again.

  **The plan order matters.** The cached path replays the first split hook of the first block to
  re-derive the residual of a skipped step, so `transformer_blocks.0` has to register
  `hidden_states` before `transformer_blocks.*` registers `adaln_indices` on the same block.
  Otherwise the replayed hook would slice a 3-D hidden state with a 1-D plan, silently leave the
  hidden state unsharded, and change what the cache compares. Keep the key order below.

  `position_ids`, `token_tags` and `timestep_indices` must **not** be split: `forward` validates
  their shapes against the full sequence and uses them for the global scatter.

  The packed sequence is `video_rows + audio_rows + text_rows`, and the text length follows the
  prompt, so it rarely divides by the CP size — prefer `--ulysses-anything` unless the length is
  known to divide.
  """

  def _apply(
    self,
    transformer: torch.nn.Module,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> dict:
    if getattr(parallelism_config, "ulysses_async", False):
      raise ValueError("Async Ulysses (`--ulysses-async`) is not supported for "
                       "MiniMaxH3Transformer3DModel: the async registry has no patch for its "
                       "attention processor. Drop the flag or use another model.")

    ulysses_size = parallelism_config.ulysses_size or 1
    ring_size = parallelism_config.ring_size or 1
    tp_size = parallelism_config.tp_size or 1
    heads = transformer.config.num_attention_heads

    # CP runs before TP, and Ulysses shards the heads of the *local* attention, so the constraint
    # is on the post-TP head count: 56 heads give 28 (tp2) or 14 (tp4). Both divide by cp2; only the
    # tp2 count divides by cp4 as well.
    if ulysses_size > 1 and not parallelism_config.ulysses_anything:
      local_heads = heads // tp_size
      if local_heads % ulysses_size != 0:
        raise ValueError(
          f"MiniMaxH3Transformer3DModel has {heads} attention heads, i.e. {local_heads} per rank "
          f"after tp_size={tp_size}, which is not divisible by ulysses_size={ulysses_size}. "
          "Ulysses shards heads, so pick a combination that divides, e.g. tp2+cp2 (28 heads) or "
          "tp4+cp2 (14 heads), or enable --ulysses-anything.")

    if (ulysses_size > 1 or ring_size > 1) and not parallelism_config.ulysses_anything:
      logger.warning(
        "MiniMaxH3Transformer3DModel packs text + audio + video rows into one sequence whose length "
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
      "proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
      "audio_proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
    }


@TensorParallelismPlannerRegister.register("MiniMaxH3Transformer3DModel")
class MiniMaxH3TensorParallelismPlanner(TensorParallelismPlanner):
  r"""Tensor parallelism planner for `MiniMaxH3Transformer3DModel`.

  Only the block stack is sharded. `token_refiner` stays replicated because `forward` scatters its
  output with `index_copy` over the full packed sequence, so sharding the text stream would only
  add an all-gather. The per-modality input and output projections (`proj_in`, `audio_proj_in`,
  `context_embedder`, `proj_out`, `audio_proj_out`) and the time embedding are replicated for the
  same reason: they produce or consume the full-width packed buffer.

  The AdaLN tables are sharded with `output_layouts=Replicate()`. Each one is a
  `Linear(time_embed_dim, 6 * hidden * 3)` — 24.2 GiB over the 50 blocks — and its output has to be
  complete because the block does `view(-1, 6 * hidden).chunk(6)` and then `index_select`s rows by
  `adaln_indices`. `Replicate()` all-gathers the partial outputs, so the weight is still divided by
  the TP size while the block sees a full table. `norm_out.linear` follows the same pattern for its
  `chunk(2)`.
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

  def rearrange_ffn_weight(self, block: torch.nn.Module, tp_size: int) -> None:
    r"""Interleave the fused SwiGLU projection into one `(up, gate)` pair per TP chunk.

    `block.ff` is a diffusers `FeedForward(..., activation_fn="swiglu")`, whose `SwiGLU.proj` is a
    `Linear(dim, 2 * inner_dim)` followed by `hidden_states, gate = chunk(2, dim=-1)` — the
    **first** half is the up projection and the **second** half is the gate. A plain
    `ColwiseParallel` would hand one contiguous chunk to each rank, i.e. the up rows to one rank and
    the gate rows to another, and each local SwiGLU would then multiply the gate half with itself:
    the shapes stay valid and the output is silently wrong.

    Reordering to `[up_0, gate_0, up_1, gate_1, ...]` makes every rank's contiguous shard hold both
    halves of its own `inner_dim / tp_size` channels, so its local `chunk(2)` yields
    `(up_i, gate_i)`. `ff.net[2]` (`RowwiseParallel`) shards its input columns in the same chunk
    order, which keeps the two aligned.

    :param block: Transformer block whose `ff.net[0].proj` is rearranged in place.
    :param tp_size: TP world size.
    """
    proj = block.ff.net[0].proj
    if getattr(proj, "_cache_dit_swiglu_rearranged", False):
      # The interleave is an involution: applying it twice restores `[up*, gate*]` and every rank
      # would then mix gate with gate, without any shape error.
      raise RuntimeError("The fused SwiGLU projection of this block was already rearranged for "
                         "tensor parallelism.")
    weight = proj.weight.data
    if weight.shape[0] % (2 * tp_size) != 0:
      raise ValueError(f"Cannot split the fused SwiGLU projection of shape {tuple(weight.shape)} "
                       f"into {tp_size} equal (up, gate) chunks.")
    up, gate = weight.chunk(2, dim=0)
    # Set the guard before the first mutation: the interleave is an involution, so a half-applied
    # rearrangement is worse than a failed one.
    proj._cache_dit_swiglu_rearranged = True
    proj.weight.data = torch.cat(
      [
        torch.cat([u, g], dim=0)
        for u, g in zip(up.chunk(tp_size, dim=0), gate.chunk(tp_size, dim=0))
      ],
      dim=0,
    ).contiguous()

    if proj.bias is not None:
      up, gate = proj.bias.data.chunk(2, dim=0)
      proj.bias.data = torch.cat(
        [
          torch.cat([u, g], dim=0)
          for u, g in zip(up.chunk(tp_size, dim=0), gate.chunk(tp_size, dim=0))
        ],
        dim=0,
      ).contiguous()

  def parallelize_transformer(
    self,
    transformer: torch.nn.Module,
    tp_mesh: DeviceMesh,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    """Shard the block stack and the output modulation of the transformer.

    :param transformer: MiniMax-H3 transformer.
    :param tp_mesh: TP device mesh.
    :returns: The sharded transformer and the applied layer plans.
    """
    tp_size = tp_mesh.size()
    heads = transformer.config.num_attention_heads
    head_dim = transformer.config.attention_head_dim
    layer_plans = []

    for block in transformer.transformer_blocks:
      if getattr(block.attn, "fused_projections", False):
        raise ValueError(
          "MiniMaxH3Transformer3DModel has `attn.fused_projections` set (by "
          "`fuse_qkv_projections()`), which this plan does not shard: the block runs one `to_qkv` "
          "projection and would reshape Q/K/V with the already-sharded `heads`. Keep the "
          "projections separate.")
      self.rearrange_ffn_weight(block, tp_size)
      # The attention processor reshapes Q/K/V with `attn.heads` and `attn.head_dim`, so `heads`
      # has to follow the sharded projection (56 -> 28 for tp_size=2).
      shard_div_attr(block.attn, "heads", tp_size)
      layer_plan = {
        "attn.to_q": ColwiseParallel(),
        "attn.to_k": ColwiseParallel(),
        "attn.to_v": ColwiseParallel(),
        "attn.to_out.0": RowwiseParallel(),
        "ff.net.0.proj": ColwiseParallel(),
        "ff.net.2": RowwiseParallel(),
        "adaln_proj.linear": ColwiseParallel(output_layouts=Replicate()),
      }
      parallelize_module(module=block, device_mesh=tp_mesh, parallelize_plan=layer_plan)
      layer_plans.append(layer_plan)

    # The shared output norm projects the timestep embedding to `(shift, scale)` and every row
    # picks from it, so its output has to stay complete as well.
    norm_out_plan = {"linear": ColwiseParallel(output_layouts=Replicate())}
    parallelize_module(module=transformer.norm_out,
                       device_mesh=tp_mesh,
                       parallelize_plan=norm_out_plan)
    layer_plans.append(norm_out_plan)

    # Guard against a later edit that shards Q/K/V without updating `heads`. `DTensor.shape` is the
    # global shape, so the check has to read the local shard.
    local_q_rows = transformer.transformer_blocks[0].attn.to_q.weight.to_local().shape[0]
    if local_q_rows != (heads // tp_size) * head_dim:
      raise ValueError(f"Sharded Q projection has {local_q_rows} rows per rank, expected "
                       f"{heads // tp_size} heads x {head_dim} channels.")
    return transformer, layer_plans
