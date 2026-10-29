import math
import torch
from typing import Any, Dict, List, Optional, Tuple

from diffusers.models.attention_dispatch import AttentionBackendName
from diffusers.models.attention_dispatch import (
  dispatch_attention_fn as _diffusers_dispatch_attention_fn, )
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.transformer_wan_animate_2 import (
  WanAnimate2AttnProcessor,
  WanAnimate2Transformer3DModel,
  _get_qkv_projections,
  rope_apply,
)
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
  ColwiseParallel,
  ParallelStyle,
  RowwiseParallel,
  parallelize_module,
)

from .register import (
  ContextParallelismPlanner,
  ContextParallelismPlannerRegister,
  TensorParallelismPlanner,
  TensorParallelismPlannerRegister,
)
from ..config import ParallelismConfig
from ..utils import shard_div_attr
from ...attention import _dispatch_attention_fn
from ...logger import init_logger

logger = init_logger(__name__)

_ATTN_PROCESSOR_PATCHED = False


def _patch_attn_processor_for_cp() -> None:
  """Run the reference and generation passes with full-sequence Ulysses attention.

  Wan-Animate-2 packs the generation tokens and the cached reference tokens into a 128-aligned
  `[generation | reference]` buffer whose indices come from the flex `BlockMask`, and the mask is
  built over the *full* video geometry. The sequence therefore cannot stay sharded inside the
  attention: Q/K/V are projected on the local shard, redistributed from `[B, S_LOCAL, H_GLOBAL, D]`
  to `[B, S_GLOBAL, H_LOCAL, D]`, attended over the full packed layout (so the mask keeps its
  meaning), and the output is scattered back to the local shard. Ring attention is not usable for
  the same reason: it splits K/V across ranks while the mask indexes the full `[generation |
  reference]` K/V on one rank.

  The processor reads its communicator from `self._cp_comm`, which only the CP planner sets, so a
  transformer without CP keeps the stock behaviour bit for bit.
  """

  global _ATTN_PROCESSOR_PATCHED
  if _ATTN_PROCESSOR_PATCHED:
    return

  original_call = WanAnimate2AttnProcessor.__call__

  def cp_call(
    self,
    attn: nn.Module,
    hidden_states: torch.Tensor,
    rotary_emb: torch.Tensor,
    grid_sizes: torch.Tensor,
    kv_cache: Optional[torch.nn.Module] = None,
    kv_cache_mode: Optional[str] = None,
    rope_stride: int = 1,
    reference_rotary_emb: Optional[torch.Tensor] = None,
    reference_grid_sizes: Optional[torch.Tensor] = None,
    reference_rope_stride: int = 1,
    attention_mask: Any = None,
    origin_latent_frames: Optional[int] = None,
    origin_latent_hw: Optional[int] = None,
  ) -> torch.Tensor:

    cp_size = int(getattr(getattr(self, "_cp_config", None), "ulysses_degree", 1) or 1)
    if cp_size <= 1:
      return original_call(
        self,
        attn,
        hidden_states,
        rotary_emb=rotary_emb,
        grid_sizes=grid_sizes,
        kv_cache=kv_cache,
        kv_cache_mode=kv_cache_mode,
        rope_stride=rope_stride,
        reference_rotary_emb=reference_rotary_emb,
        reference_grid_sizes=reference_grid_sizes,
        reference_rope_stride=reference_rope_stride,
        attention_mask=attention_mask,
        origin_latent_frames=origin_latent_frames,
        origin_latent_hw=origin_latent_hw,
      )

    # The communicator binds to one process group, so it is rebuilt whenever the config changes.
    comm = getattr(self, "_cp_comm", None)
    if comm is None or getattr(self, "_cp_comm_config", None) is not self._cp_config:
      from ...distributed.core._distributed_primitives import _All2AllComm

      comm = _All2AllComm(self._cp_config)
      self._cp_comm = comm
      self._cp_comm_config = self._cp_config

    local_sequence = hidden_states.shape[1]
    full_sequence = local_sequence * cp_size
    # `grid_sizes` describes the current segment, so the trailing rows of the gathered sequence are
    # padding that `rope_apply` passes through untouched.
    valid_length = int(torch.prod(grid_sizes[0]).item())
    cp_pad = full_sequence - valid_length
    if cp_pad < 0:
      raise ValueError(f"Gathered sequence ({full_sequence}) is shorter than the segment "
                       f"({valid_length}); the CP shard size does not match the segment geometry.")

    query, key, value = _get_qkv_projections(attn, hidden_states, None)
    query = attn.norm_q(query)
    key = attn.norm_k(key)
    query = query.unflatten(2, (attn.heads, -1))
    key = key.unflatten(2, (attn.heads, -1))
    value = value.unflatten(2, (attn.heads, -1))

    local_heads = query.shape[2]
    if local_heads % cp_size != 0:
      raise NotImplementedError(
        f"Wan-Animate-2 CP requires the local attention head count after TP ({local_heads}) to be "
        f"divisible by ulysses_size ({cp_size}).")

    # Metadata carries the local sequence length on some backends, and the reference and generation
    # passes have different lengths, so refresh it on every call.
    comm.init_meta(query)
    query = comm.send_q(query).wait()
    key = comm.send_k(key).wait()
    value = comm.send_v(value).wait()

    # From here on the layout is the one the stock processor expects: full segment length with
    # local heads. Everything below mirrors upstream `transformer_wan_animate_2.py:198-273`.
    if kv_cache_mode == "extract":
      # Drop the all-to-all padding before anything else. The dense reference path has no mask, so
      # padded rows would enter the softmax denominator and, from layer 1 on, leak into the
      # reference
      # K/V written for the generation pass: the result would depend on `seq_len % cp_size`.
      query = query[:, :valid_length]
      key = key[:, :valid_length]
      value = value[:, :valid_length]
      kv_cache.store(key.clone(), value.clone())

      query = rope_apply(query, grid_sizes, rotary_emb, rope_stride).type_as(value)
      key = rope_apply(key, grid_sizes, rotary_emb, rope_stride).type_as(value)

      hidden_states = _dispatch_attention_fn(
        query,
        key,
        value,
        attn_mask=None,
        backend=self._attention_backend,
        parallel_config=None,
      )
    elif kv_cache_mode == "cached":
      query = rope_apply(query, grid_sizes, rotary_emb, rope_stride).type_as(value)
      key = rope_apply(key, grid_sizes, rotary_emb, rope_stride).type_as(value)

      key_ref, value_ref = kv_cache.get()
      key_ref = rope_apply(key_ref, reference_grid_sizes, reference_rotary_emb,
                           reference_rope_stride).type_as(value)

      frames, height, width = grid_sizes[0].tolist()
      ref_frames, ref_height, ref_width = reference_grid_sizes[0].tolist()
      hw, ref_hw = height * width, ref_height * ref_width
      heads, head_dim = query.shape[2], query.shape[3]
      batch_size = query.shape[0]

      packed_length = math.ceil((origin_latent_frames + 1) * origin_latent_hw / 128) * 128
      packed_ref_length = math.ceil(origin_latent_frames * origin_latent_hw / 128) * 128

      query_packed = query.new_zeros(batch_size, packed_length, heads, head_dim)
      key_packed = key.new_zeros(batch_size, packed_length + packed_ref_length, heads, head_dim)
      value_packed = value.new_zeros(batch_size, packed_length + packed_ref_length, heads, head_dim)

      generation = slice(0, frames * origin_latent_hw)
      query_packed[:, generation].view(batch_size, frames, origin_latent_hw, heads,
                                       head_dim)[:, :, :hw] = query[:, :valid_length].view(
                                         batch_size, frames, hw, heads, head_dim)
      key_packed[:, generation].view(batch_size, frames, origin_latent_hw, heads,
                                     head_dim)[:, :, :hw] = key[:, :valid_length].view(
                                       batch_size, frames, hw, heads, head_dim)
      value_packed[:, generation].view(batch_size, frames, origin_latent_hw, heads,
                                       head_dim)[:, :, :hw] = value[:, :valid_length].view(
                                         batch_size, frames, hw, heads, head_dim)

      reference = slice(packed_length, packed_length + ref_frames * origin_latent_hw)
      key_packed[:, reference].view(batch_size, ref_frames, origin_latent_hw, heads,
                                    head_dim)[:, :, :ref_hw] = key_ref.view(
                                      batch_size, ref_frames, ref_hw, heads,
                                      head_dim)[:, :, :ref_hw]
      value_packed[:, reference].view(batch_size, ref_frames, origin_latent_hw, heads,
                                      head_dim)[:, :, :ref_hw] = value_ref.view(
                                        batch_size, ref_frames, ref_hw, heads,
                                        head_dim)[:, :, :ref_hw]

      # The packed [generation | reference] scoring is expressed as a flex `BlockMask`, which
      # cache-dit's backend registry does not provide, so this branch dispatches through diffusers
      # with an explicit flex backend. Communication is already done above, hence
      # `parallel_config=None`.
      hidden_states = _diffusers_dispatch_attention_fn(
        query_packed,
        key_packed,
        value_packed,
        attn_mask=attention_mask,
        backend=AttentionBackendName.FLEX,
        parallel_config=None,
      )

      hidden_states = hidden_states[:, generation].view(batch_size, frames, origin_latent_hw, heads,
                                                        head_dim)[:, :, :hw]
      hidden_states = hidden_states.reshape(batch_size, valid_length, heads, head_dim)
    else:
      raise ValueError(
        f"`kv_cache_mode` must be either 'extract' or 'cached', got {kv_cache_mode}.")

    # The extract branch attends over the gathered (already padded) sequence, so trim back to the
    # segment length before restoring the padding the output all-to-all expects.
    hidden_states = hidden_states[:, :valid_length]
    if cp_pad:
      pad_shape = list(hidden_states.shape)
      pad_shape[1] = cp_pad
      hidden_states = torch.cat([hidden_states, hidden_states.new_zeros(pad_shape)], dim=1)
    hidden_states = comm.send_o(hidden_states).wait()
    hidden_states = hidden_states.flatten(2, 3).type_as(query)
    hidden_states = attn.to_out[0](hidden_states)
    return attn.to_out[1](hidden_states)

  WanAnimate2AttnProcessor.__call__ = cp_call
  _ATTN_PROCESSOR_PATCHED = True


@ContextParallelismPlannerRegister.register("WanAnimate2")
class WanAnimate2ContextParallelismPlanner(ContextParallelismPlanner):
  """Context parallelism for Wan-Animate-2 with full-sequence attention and sharded heads."""

  def _apply(
    self,
    transformer: Optional[torch.nn.Module | ModelMixin] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> Dict:
    if parallelism_config.ring_size is not None and parallelism_config.ring_size > 1:
      raise NotImplementedError(
        "Ring attention is not supported for Wan-Animate-2: the in-context flex block mask indexes "
        "the full [generation | reference] K/V on every rank, which ring attention splits across "
        "ranks. Use `--parallel ulysses` instead.")
    if parallelism_config.ulysses_anything:
      raise NotImplementedError(
        "ulysses_anything is not supported for Wan-Animate-2: only the standard padded Ulysses "
        "path "
        "keeps the reference K/V cache consistent with the flex block mask.")

    from ...caching.patch_functors import WanAnimate2PatchFunctor

    # The split/gather lives in the patch functor, so context parallelism needs it even when the
    # cache engine is disabled; the functor is idempotent for the cache path.
    if not getattr(transformer, "_is_patched", False):
      WanAnimate2PatchFunctor().apply(transformer)
    if not getattr(transformer, "_is_patched", False):
      # `PatchFunctor.apply` bails out on an already parallelized transformer. The processor would
      # then all-to-all a sequence that was never sharded, i.e. wrong results instead of an error.
      raise RuntimeError("Wan-Animate-2 context parallelism requires the WanAnimate2PatchFunctor. "
                         "Enable caching before parallelizing the transformer.")

    _patch_attn_processor_for_cp()
    logger.info("Enabled Wan-Animate-2 context parallelism: full-sequence attention with "
                f"head-sharded reference KV (ulysses_size={parallelism_config.ulysses_size}).")
    return {}


@TensorParallelismPlannerRegister.register("WanAnimate2")
class WanAnimate2TensorParallelismPlanner(TensorParallelismPlanner):
  """Tensor parallelism for Wan-Animate-2: no GQA, so heads split evenly across ranks."""

  def _apply(
    self,
    transformer: WanAnimate2Transformer3DModel,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> Tuple[WanAnimate2Transformer3DModel, List[Dict[str, ParallelStyle]]]:
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
    # registry matches the first prefix, so "Wan" must be registered *after* "WanAnimate2".
    from .wan import DistributedRMSNorm

    def prepare_block(block: nn.Module):
      tp_size = tp_mesh.size()
      for attn in (block.self_attn, block.cross_attn):
        if getattr(attn, "fused_projections", False):
          # `to_qkv` / `to_kv` pack the projections, and a Colwise plan cannot split them per head.
          attn.unfuse_projections()
        shard_div_attr(attn, "heads", tp_size)

      layer_plan = {
        "self_attn.to_q": ColwiseParallel(),
        "self_attn.to_k": ColwiseParallel(),
        "self_attn.to_v": ColwiseParallel(),
        "self_attn.to_out.0": RowwiseParallel(),
        "cross_attn.to_q": ColwiseParallel(),
        "cross_attn.to_k": ColwiseParallel(),
        "cross_attn.to_v": ColwiseParallel(),
        "cross_attn.to_out.0": RowwiseParallel(),
        "ffn.0": ColwiseParallel(),
        "ffn.2": RowwiseParallel(),
      }
      if getattr(block.cross_attn, "add_k_proj", None) is not None:
        layer_plan["cross_attn.add_k_proj"] = ColwiseParallel()
      if getattr(block.cross_attn, "add_v_proj", None) is not None:
        layer_plan["cross_attn.add_v_proj"] = ColwiseParallel()

      parallelize_module(module=block, device_mesh=tp_mesh, parallelize_plan=layer_plan)

      # `norm_q` / `norm_k` normalize over the full 5120-wide hidden dimension (before the head
      # unflatten), so a colwise-sharded projection must normalize with the global RMS statistics.
      for attn in (block.self_attn, block.cross_attn):
        attn.norm_q = DistributedRMSNorm.from_rmsnorm(tp_mesh, attn.norm_q)
        attn.norm_k = DistributedRMSNorm.from_rmsnorm(tp_mesh, attn.norm_k)
        if getattr(attn, "norm_added_k", None) is not None:
          attn.norm_added_k = DistributedRMSNorm.from_rmsnorm(tp_mesh, attn.norm_added_k)
      return layer_plan

    layer_plans = []
    for _, block in transformer.blocks.named_children():
      layer_plans.append(prepare_block(block))
    return transformer, layer_plans
