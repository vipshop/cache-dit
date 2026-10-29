"""Stable Audio 3 DiT distributed parallelism planners (CP + TP).

`StableAudio3DiTModel` denoises a `(batch, 256, T)` waveform-latent stream through
`transformer.transformer_blocks`, where `T = ceil(duration * 44100 / 4096)` (108 frames for 10 s).
Inside `forward` the sequence is transposed to `(batch, T, 1536)`, the 64 learned `memory_tokens` are
prepended, and one full RoPE table is built for the resulting `64 + T` tokens; every block then
consumes `hidden_states` plus keyword arguments that never change across the loop. That structure is
what makes the block stack cacheable and what the planners below exploit.

* CP shards the sequence **once, after the memory tokens are prepended** (block 0's input) and
  gathers it back on the last block, so the Ulysses rank-concatenation rebuilds the exact
  `[memory | audio]` order. The RoPE table is built for the full sequence by `forward` and is split
  on the same token axis by the plan itself, which makes the trailing `freqs[-seq_len:]` slice inside
  diffusers' `_apply_rotary_partial` a no-op. The cross-attention keeps its replicated text context:
  its processor is patched to run **without** a CP config, otherwise Ulysses would concatenate the
  text tokens once per rank.
* TP shards the fused projections. `to_qkv` (`chunk(5)`), `cross_attn.to_q` (`chunk(2)`),
  `to_kv` (`chunk(3)`) and `ff.proj_in` (`chunk(2)`) all split their output into equal sections, so a
  plain `ColwiseParallel` would cut through a section boundary. The planner therefore reorders each
  fused weight into "shard-part major" order first: every rank then holds
  `[section_0_part_r | section_1_part_r | ...]` and the existing `chunk(n)` calls yield that rank's own
  slice of every section. `heads` is divided to match, while `dim_heads` stays intact. The RMS QK
  norms normalize a single head (`dim_heads`), so unlike LongCat-AudioDiT they stay local and do not
  need a distributed variant.
"""

import functools
import inspect
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

_TP_PERMUTED_FLAG = "_cache_dit_sa3_tp_permuted"


def _num_memory_tokens(transformer: nn.Module) -> int:
  """Return the number of memory tokens prepended inside `forward`.

  :param transformer: `StableAudio3DiTModel` instance.
  :returns: `transformer.config.num_memory_tokens` (0 when the checkpoint disables them).
  """

  return int(getattr(transformer.config, "num_memory_tokens", 0) or 0)


def _cp_size(parallelism_config: ParallelismConfig) -> int:
  """Return the sequence-parallel world size (Ulysses or Ring, not TP).

  :param parallelism_config: Requested parallelism configuration.
  :returns: Number of ranks the latent sequence is split across.
  """

  return max(1, int(parallelism_config.ulysses_size or 1), int(parallelism_config.ring_size or 1))


def _is_cp_armed(processor: nn.Module) -> bool:
  """Report whether a processor is currently attached to a multi-rank CP group.

  :param processor: Attention processor instance.
  :returns: `True` when a multi-rank `_cp_config` / `_parallel_config` is present.
  """

  cp_config = getattr(processor, "_cp_config", None)
  if cp_config is None:
    cp_config = getattr(processor, "_parallel_config", None)
  if cp_config is None:
    return False
  world_size = getattr(cp_config, "_world_size", None)
  if world_size is None:
    world_size = getattr(cp_config, "world_size", 1)
  return int(world_size or 1) > 1


_CROSS_ATTN_PATCH_ORIGINALS: Dict[type, callable] = {}


def _install_cp_cross_attn_patch() -> None:
  """Keep the cross attention local by hiding the CP config from its processor.

  The text/duration context is replicated on every rank, so letting Ulysses run inside the
  cross attention concatenates the text tokens once per rank: K grows to `cp_size * text_len` while
  the (already dropped) mask stays at `text_len`, and the attention either dies on a shape mismatch or
  silently attends to a duplicated context. The self attention keeps the config because it owns the
  sharded latent sequence.

  The patch is class level and armed only when the instance carries a multi-rank CP config, so the
  same processor class stays bit-identical in a cp-free run (and in the VAE, which builds its
  attention from a different processor class entirely).
  """

  try:
    from diffusers.models.transformers.transformer_stable_audio3 import (
      StableAudio3CrossAttnProcessor, )
  except ImportError:  # diffusers without Stable Audio 3
    return

  processor_cls = StableAudio3CrossAttnProcessor
  if processor_cls in _CROSS_ATTN_PATCH_ORIGINALS:
    return

  original = processor_cls.__call__
  _CROSS_ATTN_PATCH_ORIGINALS[processor_cls] = original

  @functools.wraps(original)
  def __patch__cross_attn_call__(self, *args, **kwargs):
    saved_original = _CROSS_ATTN_PATCH_ORIGINALS.get(processor_cls)
    if saved_original is None:
      raise RuntimeError("cache-dit: the Stable Audio 3 cross attention processor patch lost its "
                         "upstream `__call__` reference.")
    if not _is_cp_armed(self):
      return saved_original(self, *args, **kwargs)
    parallel_config = getattr(self, "_parallel_config", None)
    cp_config = getattr(self, "_cp_config", None)
    self._parallel_config = None
    self._cp_config = None
    try:
      return saved_original(self, *args, **kwargs)
    finally:
      self._parallel_config = parallel_config
      self._cp_config = cp_config

  processor_cls.__call__ = __patch__cross_attn_call__


def _wrap_inpaint_guard(transformer: nn.Module) -> None:
  """Reject `local_add_cond` (inpainting) before `forward` runs under CP.

  The inpaint pipeline passes a `(batch, local_add_cond_dim, T)` conditioning tensor that each block
  adds to the **audio** positions only (`hidden_states[:, num_memory_tokens:] + local`). Under CP the
  hidden states are a contiguous shard of `[memory | audio]` while `local_add_cond` is aligned to the
  audio axis alone, so the two shard boundaries do not line up unless the split happens to fall on the
  memory-token edge: `hidden_states[:, num_memory_tokens:]` then has a local length while `local` is
  still `(batch, dim, T)`, so the add fails on a broadcast error deep inside a block. Rejecting up front
  turns that into one readable message.

  :param transformer: `StableAudio3DiTModel` instance.
  """

  if getattr(transformer, "_cache_dit_sa3_cp_guard", False):
    return
  original_forward = transformer.forward
  signature = inspect.signature(original_forward)

  def forward(self, *args, **kwargs):
    bound = signature.bind(*args, **kwargs)
    if bound.arguments.get("local_add_cond") is not None:
      raise NotImplementedError(
        "Context parallelism for Stable Audio 3 does not support the inpainting pipeline yet: "
        "`local_add_cond` is aligned to the audio frames while the CP shard also covers the "
        "prepended memory tokens, so the two partitions cannot share one split. Use the inpaint "
        "pipeline on a single GPU.")
    return original_forward(*bound.args, **bound.kwargs)

  forward.__signature__ = signature
  transformer.forward = forward.__get__(transformer, type(transformer))
  transformer._cache_dit_sa3_cp_guard = True


@ContextParallelismPlannerRegister.register("StableAudio3DiTModel")
class StableAudio3ContextParallelismPlanner(ContextParallelismPlanner):

  def _apply(
    self,
    transformer: Optional[torch.nn.Module] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:
    """Build the Stable Audio 3 context parallel plan.

    :param transformer: `StableAudio3DiTModel` instance.
    :param parallelism_config: Requested parallelism configuration.
    :returns: The context parallel plan.
    """

    if transformer is None or parallelism_config is None:
      return {}

    ring_size = int(parallelism_config.ring_size or 1)
    ulysses_size = int(parallelism_config.ulysses_size or 1)
    if ring_size > 1:
      # Ring attention would need the same memory-token layout and rope window, but the ring backend
      # is unverified here (the flash-attention ring path also rejects the fp32 parity harness), so
      # Ulysses is required until a ring row is measured.
      raise NotImplementedError(
        f"Stable Audio 3 context parallelism only supports Ulysses (got ring_size={ring_size}, "
        f"ulysses_size={ulysses_size}). Use `--parallel ulysses`.")

    cp_size = _cp_size(parallelism_config)
    tp_size = max(1, int(parallelism_config.tp_size or 1))
    heads = int(transformer.config.num_heads)
    if cp_size > 1 and (heads // tp_size) % cp_size != 0:
      raise ValueError(
        f"Stable Audio 3 has {heads} attention heads, i.e. {heads // tp_size} per rank after "
        f"tp_size={tp_size}, which is not divisible by the sequence parallel size {cp_size}. "
        "Ulysses shards heads across the sequence-parallel ranks.")

    blocks = transformer.transformer_blocks
    last_index = len(blocks) - 1
    if last_index < 1:
      raise ValueError(
        "Stable Audio 3 context parallelism needs at least two transformer blocks: the sequence is "
        "split on block 0 and gathered on the last block's output, which would be the same module "
        "with a single block.")

    _install_cp_cross_attn_patch()
    _wrap_inpaint_guard(transformer)

    # The cached path only executes the tail blocks inside the Bn group, so a plan that gathers on the
    # last block needs `Bn_compute_blocks >= 1`; the cache layer validates it against this attribute.
    transformer._cp_requires_min_bn_blocks = 1

    _cp_plan: _ContextParallelModelPlan = {
      # Must stay first on block 0: the cached path replays this hook to re-derive the residual.
      # The input is `(batch, num_memory_tokens + T, dim)` — the memory tokens are already prepended,
      # so the shard is a contiguous window of the real attention sequence. Splitting any earlier
      # (e.g. at `proj_in`) would prepend one copy of the memory tokens per rank and change the
      # softmax normalization.
      "transformer_blocks.0": {
        "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3),
      },
      # The full `(64 + T, rot_dim)` table is built before the loop, so it has to be cut on the token
      # axis with the same boundaries as `hidden_states`. Passing the local window down makes the
      # `freqs[-seq_len:]` slice inside `_apply_rotary_partial` an identity, which is why diffusers
      # does not need to be patched.
      "transformer_blocks.*": {
        "rope": _ContextParallelInput(split_dim=0, expected_dims=2),
      },
      # `forward` drops the 64 memory tokens and runs `proj_out` / `postprocess_conv` on the full
      # sequence, so the gather has to restore it on the last block's output.
      f"transformer_blocks.{last_index}": _ContextParallelOutput(gather_dim=1, expected_dims=3),
    }
    logger.debug(f"Generated Stable Audio 3 CP plan: {_cp_plan}")
    return _cp_plan


def _check_fused_rows(module: nn.Linear, sections: int, tp_size: int, name: str) -> None:
  """Validate that a fused projection can be split into `sections` x `tp_size` equal row blocks.

  :param module: Fused `nn.Linear`.
  :param sections: Number of equal output sections the processor recovers with `chunk(sections)`.
  :param tp_size: Tensor parallel world size.
  :param name: Dotted module name, used in the error message.
  :raises ValueError: When the output features cannot be split evenly.
  """

  if tp_size <= 1:
    return
  out_features = module.weight.shape[0]
  if out_features % (sections * tp_size) != 0:
    raise ValueError(
      f"Stable Audio 3 TP cannot shard `{name}`: its {out_features} output features split into "
      f"{sections} sections that are not divisible by tp_size={tp_size}.")


def _permute_fused_rows(module: nn.Linear, sections: int, tp_size: int, name: str) -> None:
  """Reorder a fused projection's output rows into "shard-part major" order.

  The fused projections of a Stable Audio 3 block emit `sections` equal slices that the processor
  recovers with `chunk(sections, dim=-1)`. A plain column shard cuts through those boundaries, so the
  weight rows are reordered to `[part_0: all sections | part_1: all sections | ...]` first: the
  uniform shard a rank then receives is exactly `[section_0_part_r | section_1_part_r | ...]`, which
  the existing `chunk(sections)` turns into that rank's slice of every section. Row permutation is
  exact for a linear layer (each output row is independent), so `tp_size == 1` is a no-op.

  :param module: Fused `nn.Linear` whose output is chunked into `sections` equal slices.
  :param sections: Number of equal output sections (5 for `to_qkv`, 3 for `to_kv`, 2 elsewhere).
  :param tp_size: Tensor parallel world size.
  :param name: Dotted module name, used in error messages.
  """

  if tp_size <= 1:
    return
  _check_fused_rows(module, sections, tp_size, name)
  weight = module.weight
  out_features, in_features = weight.shape
  per_section = out_features // sections
  weight.data.copy_(
    weight.data.view(sections, tp_size, per_section // tp_size,
                     in_features).permute(1, 0, 2, 3).reshape(out_features, in_features))
  if module.bias is not None:
    bias = module.bias
    bias.data.copy_(
      bias.data.view(sections, tp_size, per_section // tp_size).permute(1, 0,
                                                                        2).reshape(out_features))


@TensorParallelismPlannerRegister.register("StableAudio3DiTModel")
class StableAudio3TensorParallelismPlanner(TensorParallelismPlanner):

  def _apply(
    self,
    transformer: torch.nn.Module,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    """Build the Stable Audio 3 tensor parallel plan.

    :param transformer: `StableAudio3DiTModel` instance.
    :param parallelism_config: Requested parallelism configuration.
    :returns: The parallelized transformer and the applied layer plans.
    """

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
    """Shard every block, leaving the step-invariant conditioning stack replicated.

    :param transformer: `StableAudio3DiTModel` instance.
    :param tp_mesh: Tensor parallel mesh.
    :returns: The parallelized transformer and the applied layer plans.
    """

    tp_size = int(tp_mesh.size())
    if tp_size > 1 and getattr(transformer, _TP_PERMUTED_FLAG, False):
      raise RuntimeError("Stable Audio 3 TP: the fused weights were already reordered once; "
                         "`parallelize_transformer` must not be applied twice.")

    # The section counts below mirror the processor's `chunk(n)` calls, which are *differential*
    # layouts (5 / 2 / 3 / 2). A checkpoint trained with `use_differential_attention=False` uses
    # 3 / 1 / 2 / 2 instead, so the reordering would silently mix sections: reject it instead.
    for block in transformer.transformer_blocks:
      if not getattr(block.self_attn, "use_differential", True):
        raise NotImplementedError(
          "Stable Audio 3 tensor parallelism currently requires `use_differential_attention=True`: "
          "the fused-projection reordering is derived from the differential `chunk(5/2/3)` layout.")

    # Validate every fused projection before mutating anything, so a failed attempt leaves the
    # transformer untouched.
    for block in transformer.transformer_blocks:
      if tp_size > 1:
        _check_fused_rows(block.self_attn.to_qkv, 5, tp_size, "self_attn.to_qkv")
        _check_fused_rows(block.cross_attn.to_q, 2, tp_size, "cross_attn.to_q")
        _check_fused_rows(block.cross_attn.to_kv, 3, tp_size, "cross_attn.to_kv")
        _check_fused_rows(block.ff.proj_in, 2, tp_size, "ff.proj_in")

    layer_plans = []
    for block in transformer.transformer_blocks:
      # `heads` drives `unflatten(-1, (heads, dim_heads))` on the locally sharded Q/K/V output, so it
      # has to follow the shard. `dim_heads` stays intact, and the QK RMS norms normalize one head
      # (`dim_heads` elements) *after* the unflatten, i.e. entirely inside a rank — unlike
      # LongCat-AudioDiT they need no distributed variant.
      shard_div_attr(block.self_attn, "heads", tp_size)
      shard_div_attr(block.cross_attn, "heads", tp_size)

      _permute_fused_rows(block.self_attn.to_qkv, 5, tp_size, "self_attn.to_qkv")
      _permute_fused_rows(block.cross_attn.to_q, 2, tp_size, "cross_attn.to_q")
      _permute_fused_rows(block.cross_attn.to_kv, 3, tp_size, "cross_attn.to_kv")
      _permute_fused_rows(block.ff.proj_in, 2, tp_size, "ff.proj_in")

      # `to_out` / `ff.proj_out` take the **unsharded** feature axis as input, and the reordering above
      # keeps each rank's heads (and gate/up halves) inside one contiguous window of it, so a plain
      # rowwise shard already lines up with the local activation: no column reordering is needed.
      layer_plan = {
        "self_attn.to_qkv": ColwiseParallel(),
        "self_attn.to_out": RowwiseParallel(),
        "cross_attn.to_q": ColwiseParallel(),
        "cross_attn.to_kv": ColwiseParallel(),
        "cross_attn.to_out": RowwiseParallel(),
        "ff.proj_in": ColwiseParallel(),
        "ff.proj_out": RowwiseParallel(),
      }
      parallelize_module(
        module=block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_plan,
      )
      layer_plans.append(layer_plan)

    # `global_cond_embedder` emits the 6-way AdaLN modulation that is chunked right afterwards,
    # `to_cond_embed` / `to_global_embed` / `to_timestep_embed` produce step-invariant conditioning and
    # the 1x1 convolutions act on the latent channels: all tiny, all replicated.
    if tp_size > 1:
      setattr(transformer, _TP_PERMUTED_FLAG, True)
    return transformer, layer_plans
