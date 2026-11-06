"""MOVA distributed parallelism planners (CP + TP).

MOVA denoises two towers per layer: a 40-layer video tower split across two experts (high-noise and
low-noise, only one active per step) and a 30-layer audio tower, coupled by a bidirectional bridge.

That shape drives both planners:

* **CP shards only the video stream.** The audio sequence is 403 tokens (odd, and 0.2 % of the video
  compute), so it stays replicated: its self attention is computed redundantly on every rank (exact),
  the audio->video direction keeps a complete KV for free, and only the video->audio direction needs
  the video keys/values back, which it all-gathers *after* projecting them locally and rotating them
  with the local rotary slice (see ``models/pipelines/mova/bridge.py``). Gathering the projected
  ``k``/``v`` (1536 channels each) instead of the 5120-channel hidden states is 1.67x less traffic
  and, more importantly, avoids rebuilding the full activation inside a block.
* **TP shards the projections of every fused block.** Q/K/V and the FFN first layer are column-wise,
  the output projections and the FFN second layer are row-wise. The QK RMSNorm normalizes the *full*
  projection width, so both attention norms and the two bridge conditioners' norms move to
  :class:`~cache_dit.distributed.utils._DistributedRMSNorm`; ``head_dim`` never follows the shard.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor.parallel import (
  ColwiseParallel,
  ParallelStyle,
  RowwiseParallel,
  parallelize_module,
)

from ...attention.backends.register import _AttnBackendRegistry
from ...logger import init_logger
from ..config import ParallelismConfig
from ..core import _ContextParallelInput, _ContextParallelModelPlan, _ContextParallelOutput
from ..utils import _DistributedRMSNorm, shard_div_attr
from .register import (
  ContextParallelismPlanner,
  ContextParallelismPlannerRegister,
  TensorParallelismPlanner,
  TensorParallelismPlannerRegister,
)

logger = init_logger(__name__)

__all__ = ["MOVAContextParallelismPlanner", "MOVATensorParallelismPlanner"]


@ContextParallelismPlannerRegister.register("MOVAStack")
class MOVAContextParallelismPlanner(ContextParallelismPlanner):
  r"""Hook-based context parallelism for the fused MOVA stack (Ulysses only).

  * ``blocks.0`` splits the video hidden states once; the audio stream is not listed and therefore
    stays replicated.
  * the **last** block's output is gathered. The second output slot is ``None`` on purpose: it tells
    the framework not to gather the (replicated) audio stream.
  * the fused stack pre-splits the video rotary tables in its own ``forward`` instead of listing them
    here. A wildcard hook would also split the bridge's cross-modal table, which the video->audio
    direction rotates *before* gathering its keys — splitting it would double-shard the sequence.

  Because the gather sits on the last block, **CP with caching requires ``Bn_compute_blocks >= 1``**:
  with the ``DBCacheConfig`` default of 0 the last block is skipped on a cached step (nothing gathers)
  and, on a computed step, it lands in the middle group whose residual subtraction would mix a
  gathered tensor with a local one.
  """

  def _apply(
    self,
    transformer: Optional[nn.Module] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:
    del kwargs
    assert transformer is not None, "Transformer must be provided."
    assert parallelism_config is not None, "ParallelismConfig must be provided."

    if parallelism_config.ring_size is not None and parallelism_config.ring_size > 1:
      raise NotImplementedError(
        "MOVA context parallelism does not support Ring attention: the video->audio bridge "
        "all-gathers the projected keys/values, which assumes the plain Ulysses equipartition.")
    if parallelism_config.ulysses_anything:
      raise NotImplementedError(
        "MOVA context parallelism does not support `--ulysses-anything`: the key/value gather "
        "assumes equally sized shards.")

    ulysses_size = parallelism_config.ulysses_size or 1
    tp_size = parallelism_config.tp_size or 1
    for tower in ("video_dit", "video_dit_2"):
      module = getattr(transformer, tower, None)
      if module is None:
        continue
      heads = module.config.num_heads
      if (heads // tp_size) % ulysses_size != 0:
        raise ValueError(
          f"MOVA {tower} has {heads} attention heads, i.e. {heads // tp_size} per rank after "
          f"tp_size={tp_size}, which is not divisible by ulysses_size={ulysses_size}.")
    audio_heads = transformer.audio_dit.config.num_heads
    if (audio_heads // tp_size) % ulysses_size != 0:
      logger.info(f"MOVA audio tower has {audio_heads} heads; it stays replicated under context "
                  f"parallelism, so ulysses_size={ulysses_size} does not need to divide them.")

    backend, _ = _AttnBackendRegistry.get_active_backend()
    if not _AttnBackendRegistry.is_context_parallel_available(backend):
      raise ValueError(
        f"The active attention backend ({backend.value}) does not implement context parallelism; "
        "cache-dit would silently drop the CP config and every rank would attend only its own "
        "shard. Use `--attn native` (or another CP-capable backend) with `--parallel ulysses`.")

    blocks = transformer.blocks
    last_index = len(blocks) - 1
    if last_index < 1:
      raise ValueError(
        "MOVA context parallelism needs at least two fused blocks: the video stream is split on "
        "block 0 and gathered on the last block's output.")

    transformer._cp_requires_min_bn_blocks = 1

    _cp_plan: _ContextParallelModelPlan = {
      "blocks.0": {
        "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3),
      },
      # Second slot None => the anchored audio stream is replicated and must not be gathered.
      f"blocks.{last_index}": (_ContextParallelOutput(gather_dim=1, expected_dims=3), None),
    }
    logger.debug(f"Generated MOVA CP plan: {_cp_plan}")
    return _cp_plan


@TensorParallelismPlannerRegister.register("MOVAStack")
class MOVATensorParallelismPlanner(TensorParallelismPlanner):
  r"""Tensor-parallel planner for the fused MOVA stack.

  Every fused block owns up to five sub-modules that shard on the same head grid: two video experts,
  the audio tower's block, and the two bridge conditioners. The video tower has 40 heads and the
  audio tower 12, so a common ``tp_size`` divides both (``tp_size`` in 1/2/4).

  The bridge conditioners cross the two widths (``q``: 5120 or 1536, ``kv``: the other one), and a
  column-wise shard of ``k``/``v`` gives each rank the KV of the same heads it owns for ``q`` — the
  pair stays head-aligned without any extra communication. ``ConditionalCrossAttentionBlock.y_norm``
  normalizes the *conditioning* tower's width, which is never sharded along the query axis, so it
  stays replicated.
  """

  def _apply(
    self,
    transformer: torch.nn.Module,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    del kwargs
    tp_mesh = self.mesh(parallelism_config=parallelism_config)
    return self.parallelize_transformer(transformer=transformer, tp_mesh=tp_mesh)

  @staticmethod
  def _shard_qk_norms(attn: nn.Module, tp_mesh: DeviceMesh, tp_size: int, context: str) -> None:
    """Replace the attention's QK RMSNorm with the distributed variant.

    :param attn: Attention module owning ``norm_q`` / ``norm_k``.
    :param tp_mesh: Tensor parallel mesh.
    :param tp_size: Tensor parallel degree.
    :param context: Dotted path, used in the error message.
    """

    for attr in ("norm_q", "norm_k"):
      norm = getattr(attn, attr, None)
      if norm is None:
        continue
      width = norm.weight.shape[0] if getattr(norm, "weight", None) is not None else None
      if width is not None and width % tp_size != 0:
        raise ValueError(f"[TP] {context}.{attr} normalizes {width} channels, which is not "
                         f"divisible by tp_size={tp_size}.")
      setattr(attn, attr, _DistributedRMSNorm.from_rmsnorm(tp_mesh, norm))

  def _attention_plan(self, prefix: str) -> Dict[str, ParallelStyle]:
    """Shard plan of one attention module (``<prefix>.q/k/v/o``).

    :param prefix: Dotted path of the attention inside the fused block.
    :returns: The plan entries for that attention.
    """

    return {
      f"{prefix}.q": ColwiseParallel(),
      f"{prefix}.k": ColwiseParallel(),
      f"{prefix}.v": ColwiseParallel(),
      f"{prefix}.o": RowwiseParallel(),
    }

  def _block_plan(self, block: nn.Module, tower: str, tp_mesh: DeviceMesh,
                  tp_size: int) -> Dict[str, ParallelStyle]:
    """Shard plan of one tower block (self attention, text cross attention, FFN).

    :param block: Fused block owning the tower block.
    :param tower: Attribute name of the tower block inside the fused block.
    :param tp_mesh: Tensor parallel mesh.
    :param tp_size: Tensor parallel degree.
    :returns: The plan entries for that tower block.
    """

    sub: nn.Module = getattr(block, tower)
    plan: Dict[str, ParallelStyle] = {}
    for attn_name in ("self_attn", "cross_attn"):
      attn = getattr(sub, attn_name)
      heads = getattr(attn, "num_heads")
      if heads % tp_size != 0:
        raise ValueError(
          f"[TP] {tower}.{attn_name} has {heads} heads, not divisible by tp_size={tp_size}.")
      # The processor reshapes with both the module's head count and the inner wrapper's; `head_dim`
      # is frozen because `rope_apply_head_dim` slices with it.
      shard_div_attr(attn, "num_heads", tp_size)
      shard_div_attr(attn.attn, "num_heads", tp_size)
      self._shard_qk_norms(attn, tp_mesh, tp_size, f"{tower}.{attn_name}")
      plan.update(self._attention_plan(f"{tower}.{attn_name}"))
    plan[f"{tower}.ffn.0"] = ColwiseParallel()
    plan[f"{tower}.ffn.2"] = RowwiseParallel()
    return plan

  def _conditioner_plan(self, block: nn.Module, name: str, tp_mesh: DeviceMesh,
                        tp_size: int) -> Dict[str, ParallelStyle]:
    """Shard plan of one bridge conditioner.

    :param block: Fused block owning the conditioner.
    :param name: Attribute name of the conditioner inside the fused block.
    :param tp_mesh: Tensor parallel mesh.
    :param tp_size: Tensor parallel degree.
    :returns: The plan entries for that conditioner.
    """

    inner = getattr(block, name).inner
    heads = inner.num_heads
    if heads % tp_size != 0:
      raise ValueError(f"[TP] {name}.inner has {heads} heads, not divisible by tp_size={tp_size}.")
    shard_div_attr(inner, "num_heads", tp_size)
    # The attention wrapper reshapes with its own copy of the head count, exactly like the tower
    # blocks' `attn.attn`: leaving it unsharded slices the halved projection into twice as many heads
    # of half `head_dim`, which keeps every shape legal and silently corrupts the attention.
    shard_div_attr(inner.attn, "num_heads", tp_size)
    self._shard_qk_norms(inner, tp_mesh, tp_size, f"{name}.inner")
    return self._attention_plan(f"{name}.inner")

  def parallelize_transformer(
    self,
    transformer: torch.nn.Module,
    tp_mesh: DeviceMesh,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    """Shard every fused block.

    :param transformer: The `MOVAStack`.
    :param tp_mesh: Tensor parallel mesh.
    :returns: The parallelized transformer and the applied layer plans.
    """

    tp_size = tp_mesh.size()
    layer_plans: List[Dict[str, ParallelStyle]] = []

    for block in transformer.blocks:
      plan: Dict[str, ParallelStyle] = {}
      for tower in ("video_block", "video_block_2"):
        if getattr(block, tower, None) is not None:
          plan.update(self._block_plan(block, tower, tp_mesh, tp_size))
      if block.audio_block is not None:
        plan.update(self._block_plan(block, "audio_block", tp_mesh, tp_size))
      for conditioner in ("a2v_conditioner", "v2a_conditioner"):
        if getattr(block, conditioner, None) is not None:
          plan.update(self._conditioner_plan(block, conditioner, tp_mesh, tp_size))
      if not plan:
        continue
      parallelize_module(module=block, device_mesh=tp_mesh, parallelize_plan=plan)
      layer_plans.append(plan)

    first = transformer.blocks[0].video_block.self_attn
    local_rows = first.q.weight.to_local().shape[0]
    expected = (transformer.video_dit.config.dim // tp_size)
    if local_rows != expected:
      raise ValueError(
        f"Sharded video Q projection has {local_rows} rows per rank, expected {expected}.")
    return transformer, layer_plans
