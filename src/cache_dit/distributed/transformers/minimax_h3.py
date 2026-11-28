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
from torch.distributed.tensor import Shard
from torch.distributed.tensor.parallel import (
  ColwiseParallel,
  ParallelStyle,
  RowwiseParallel,
  parallelize_module,
)

from ...logger import init_logger
from ..config import ParallelismConfig
from ..linear_sharding import (
  local_parameter,
  local_shard,
  shard_linear_weights,
  validate_local_weight_shards,
)
from ..parallel import (
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
  prompt, so it rarely divides by the CP size — prefer `ulysses_anything=true` in
  `--parallel-params` unless the length is known to divide.
  """

  def _apply(
    self,
    transformer: torch.nn.Module,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> dict:
    if getattr(parallelism_config, "ulysses_async", False):
      raise ValueError("Async Ulysses (`ulysses_async=true` in `--parallel-params`) is not "
                       "supported for MiniMaxH3Transformer3DModel: the async registry has no patch "
                       "for its attention processor. Drop the flag or use another model.")

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
          "tp4+cp2 (14 heads), or enable `ulysses_anything=true` in `--parallel-params`.")

    if (ulysses_size > 1 or ring_size > 1) and not parallelism_config.ulysses_anything:
      logger.warning(
        "MiniMaxH3Transformer3DModel packs text + audio + video rows into one sequence whose length "
        "follows the prompt, so it rarely divides by the context-parallel size; pass "
        "`ulysses_anything=true` in `--parallel-params` unless that length is known to divide.")

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


@TensorParallelismPlannerRegister.register("MiniMaxH3Transformer3DModel", local_weight_shards=True)
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
    *,
    local_shard_mesh: DeviceMesh | None = None,
    **kwargs,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    if local_shard_mesh is None:
      return self.parallelize_transformer(transformer, self.mesh(parallelism_config))
    return self.parallelize_transformer(transformer, local_shard_mesh, local_weight_shards=True)

  @staticmethod
  def _block_plan() -> Dict[str, ParallelStyle]:
    return {
      "attn.to_q": ColwiseParallel(),
      "attn.to_k": ColwiseParallel(),
      "attn.to_v": ColwiseParallel(),
      "attn.to_out.0": RowwiseParallel(),
      "ff.net.0.proj": ColwiseParallel(),
      "ff.net.2": RowwiseParallel(),
      "adaln_proj.linear": ColwiseParallel(output_layouts=Replicate()),
    }

  @staticmethod
  def _initialize_locally(path: str) -> bool:
    # Native distribution keeps the FFN output projection's measured peak lower.
    return path != "ff.net.2"

  def _preflight_local_weight_shards(
    self,
    transformer: torch.nn.Module,
    parallelism_config: ParallelismConfig,
  ) -> DeviceMesh:
    tp_mesh = self.mesh(parallelism_config)
    replica_mesh = parallelism_config._flat_mesh if parallelism_config.hybrid_enabled() else tp_mesh
    tp_size = tp_mesh.size()
    head_dim = transformer.config.attention_head_dim
    paths = self._block_plan()

    def projections():
      for i, block in enumerate(transformer.transformer_blocks):
        self._check_attention(block)
        self._check_swiglu(block.ff.net[0].proj, tp_size)
        heads = block.attn.heads
        if heads % tp_size != 0:
          raise ValueError(
            f"transformer_blocks.{i}.attn.heads must be divisible by tp_size={tp_size}.")
        rows = block.attn.to_q.weight.shape[0]
        if rows != heads * head_dim:
          raise ValueError(f"transformer_blocks.{i}.attn.to_q has {rows} rows, "
                           f"expected {heads * head_dim}.")
        for path in paths:
          yield f"transformer_blocks.{i}.{path}", self._initialize_locally(path)
      yield "norm_out.linear", True

    validate_local_weight_shards(transformer, projections(), replica_mesh)
    return tp_mesh

  @staticmethod
  def _check_attention(block: torch.nn.Module) -> None:
    if block.attn.fused_projections:
      raise ValueError(
        "MiniMaxH3Transformer3DModel has `attn.fused_projections` set (by "
        "`fuse_qkv_projections()`), which this plan does not shard: the block runs one `to_qkv` "
        "projection and would reshape Q/K/V with the already-sharded `heads`. Keep the "
        "projections separate.")

  @staticmethod
  def _check_swiglu(proj: torch.nn.Linear, tp_size: int) -> None:
    if getattr(proj, "_cache_dit_swiglu_rearranged", False):
      raise RuntimeError("The fused SwiGLU projection of this block was already rearranged for "
                         "tensor parallelism.")
    if proj.weight.shape[0] % (2 * tp_size) != 0:
      raise ValueError(
        f"Cannot split the fused SwiGLU projection of shape {tuple(proj.weight.shape)} "
        f"into {tp_size} equal (up, gate) chunks.")

  def rearrange_ffn_weight(self, block: torch.nn.Module, tp_size: int) -> None:
    r"""Interleave the fused SwiGLU projection into one `(up, gate)` pair per TP chunk.

    Diffusers' `SwiGLU.proj` stores all up rows followed by all gate rows. Reordering them to
    `[up_0, gate_0, up_1, gate_1, ...]` gives each rank both halves of its own channels for the
    local `chunk(2)`. The rowwise `ff.net[2]` takes the matching input columns.

    :param block: Transformer block whose `ff.net[0].proj` is rearranged in place.
    :param tp_size: TP world size.
    """
    proj = block.ff.net[0].proj
    self._check_swiglu(proj, tp_size)
    # Mark before the first mutation so a partially rearranged projection cannot be reused.
    proj._cache_dit_swiglu_rearranged = True
    for param in proj.parameters(recurse=False):
      up, gate = param.data.chunk(2, dim=0)
      shards = []
      for up_shard, gate_shard in zip(up.chunk(tp_size, dim=0), gate.chunk(tp_size, dim=0)):
        shards.extend((up_shard, gate_shard))
      param.data = torch.cat(shards, dim=0)

  @staticmethod
  def _shard_swiglu(proj: torch.nn.Linear, tp_mesh: DeviceMesh) -> None:
    proj._cache_dit_swiglu_rearranged = True
    for name, param in proj.named_parameters(recurse=False):
      up, gate = param.detach().chunk(2, dim=0)
      up = local_shard(up, tp_mesh, dim=0)
      gate = local_shard(gate, tp_mesh, dim=0)
      rows = up.size(0)
      local = torch.empty((2 * rows, *up.shape[1:]), dtype=param.dtype, device=tp_mesh.device_type)
      local[:rows].copy_(up)
      local[rows:].copy_(gate)
      proj.register_parameter(name, local_parameter(param, local, tp_mesh, Shard(0)))

  def parallelize_transformer(
    self,
    transformer: torch.nn.Module,
    tp_mesh: DeviceMesh,
    *,
    local_weight_shards: bool = False,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    """Shard the block stack and the output modulation of the transformer.

    :param transformer: MiniMax-H3 transformer.
    :param tp_mesh: TP device mesh.
    :param local_weight_shards: Initialize weights from preflighted CPU projections.
    :returns: The sharded transformer and the applied layer plans.
    """
    tp_size = tp_mesh.size()
    blocks = transformer.transformer_blocks
    layer_plan = self._block_plan()
    # The output norm needs the full shift and scale table.
    norm_out_plan = {"linear": ColwiseParallel(output_layouts=Replicate())}

    # The attention processor reshapes Q/K/V using the local head count.
    for block in blocks:
      if not local_weight_shards:
        self._check_attention(block)
        self.rearrange_ffn_weight(block, tp_size)
        shard_div_attr(block.attn, "heads", tp_size)
        parallelize_module(module=block, device_mesh=tp_mesh, parallelize_plan=layer_plan)
        continue

      shard_div_attr(block.attn, "heads", tp_size)
      # Distribute the native FFN output before placing the AdaLN shard.
      for path, style in layer_plan.items():
        linear = block.get_submodule(path)
        if path == "ff.net.0.proj":
          self._shard_swiglu(linear, tp_mesh)
        elif self._initialize_locally(path):
          shard_linear_weights(linear, tp_mesh, style)
        parallelize_module(linear, tp_mesh, style)

    if local_weight_shards:
      shard_linear_weights(transformer.norm_out.linear, tp_mesh, norm_out_plan["linear"])
    parallelize_module(module=transformer.norm_out,
                       device_mesh=tp_mesh,
                       parallelize_plan=norm_out_plan)

    if not local_weight_shards:
      heads = transformer.config.num_attention_heads
      head_dim = transformer.config.attention_head_dim
      local_q_rows = blocks[0].attn.to_q.weight.to_local().shape[0]
      if local_q_rows != (heads // tp_size) * head_dim:
        raise ValueError(f"Sharded Q projection has {local_q_rows} rows per rank, expected "
                         f"{heads // tp_size} heads x {head_dim} channels.")
    return transformer, [layer_plan] * len(blocks) + [norm_out_plan]
