"""Motif-Video context & tensor parallelism planners.

CP follows the HunyuanVideo 1.0 shape: single-point split at `transformer_blocks.0`
(image + text streams), a `rope` output-split for the per-rank image segment, and a
`proj_out` gather. The joint attention mask is NOT hooked: the patch functor's forward
permutes it into all-to-all order `[H_0, E_0, H_1, E_1, ...]` (full length), and the
DDT decoder query stream is chunked to the local segment there as well.

Both attention processors are class-level patched to route through cache-dit's
`_dispatch_attention_fn` with `cp_config`; unarmed processors delegate to the untouched
upstream `__call__` so any attention tree the CP runtime never armed stays bit-identical.
The single-stream cross-attention runs through Ulysses as asymmetric attention (its Q is
the local image stream while K/V come from the local text segment; the templated Ulysses
kernels all-to-all Q and K/V independently), and its text mask is re-extracted from the
permuted global mask in interleaved-rank order to match the K/V layout.

TP (v1) parallelizes only the 12 dual-stream blocks: the single-stream `attn` is
`pre_only=True` (no `to_out`), so its output is head-sharded while the cross-attention
`to_out.0` (Rowwise) returns the full width — `attn_output + cross_output` inside
`MotifVideoSingleTransformerBlock.forward` would mix shard widths. The single-stream
fused `proj_out` (attn+mlp) can be split later as a dedicated experiment.
"""

import functools
from typing import Optional

import torch

from diffusers.models.transformers.transformer_motif_video import (
  MotifVideoAttnProcessor2_0,
  MotifVideoCrossAttnProcessor2_0,
)
from ...distributed.core import (
  _ContextParallelInput,
  _ContextParallelModelPlan,
  _ContextParallelOutput,
)
from ...attention import _dispatch_attention_fn
from ...caching.patch_functors import MotifVideoPatchFunctor
from ..config import ParallelismConfig
from ..utils import shard_div_attr
from .register import (
  ContextParallelismPlanner,
  ContextParallelismPlannerRegister,
  TensorParallelismPlanner,
  TensorParallelismPlannerRegister,
)

# Upstream `__call__` implementations captured before the class-level CP patches are
# installed. The patched `__call__` hands control back to them for any processor without
# a multi-rank `_cp_config`, so the patch can never rewrite non-CP math.
_CP_PATCH_ORIGINALS: dict = {}


def _cp_is_armed(processor) -> bool:
  """Whether `processor` was armed by the CP runtime with a multi-rank `_cp_config`."""
  cp_config = getattr(processor, "_cp_config", None)
  return cp_config is not None and (cp_config._world_size or 0) > 1


def _ensure_patched(transformer: torch.nn.Module) -> None:
  """Apply the Motif-Video patch functor if it has not run yet (idempotent).

  The CP plan and the TP layer plans address the `_motif_single_*` sub-modules the functor
  creates; when parallelism is applied before `enable_cache`, the functor would otherwise be
  skipped (`check_parallelized`), so every planner applies it itself first.
  """

  if not getattr(transformer, "_is_patched", False):
    MotifVideoPatchFunctor().apply(transformer)


def _install_cp_processor_patches() -> None:
  """Install the class-level attention patches the Motif-Video CP path needs."""
  for processor_cls in (MotifVideoAttnProcessor2_0, MotifVideoCrossAttnProcessor2_0):
    _CP_PATCH_ORIGINALS.setdefault(processor_cls, processor_cls.__call__)
  MotifVideoAttnProcessor2_0.__call__ = __patch__MotifVideoAttnProcessor2_0__call__  # type: ignore[assignment]
  MotifVideoCrossAttnProcessor2_0.__call__ = __patch__MotifVideoCrossAttnProcessor2_0__call__  # type: ignore[assignment]


@ContextParallelismPlannerRegister.register("MotifVideoTransformer3DModel")
class MotifVideoContextParallelismPlanner(ContextParallelismPlanner):

  def _apply(
    self,
    transformer: Optional[torch.nn.Module] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:
    assert transformer is not None, "Transformer must be provided."

    if parallelism_config is not None and getattr(parallelism_config, "ulysses_anything", False):
      # The patched forward permutes the attention mask with an even
      # `torch.chunk(world_size)` split and the cross-attention text-mask extraction derives
      # each rank's text segment from that same equipartition; UAA's uneven split would
      # silently misalign both. Same guard as `BriaFibo2ContextParallelismPlanner`.
      raise NotImplementedError(
        "Motif-Video CP does not support --ulysses-anything: the attention-mask permutation "
        "and the cross-attention text-mask extraction assume an even sequence split. "
        "Pick resolutions whose image-token count and text length divide by ulysses_size.")

    # The plan (and the patched forward) address the functor's sub-module attributes.
    _ensure_patched(transformer)
    _install_cp_processor_patches()

    _cp_plan: _ContextParallelModelPlan = {
      # (freqs_cos, freqs_sin): each [N_img, head_dim], split on the token axis so every
      # rank gets its image segment; text tokens never carry RoPE in Motif-Video.
      "rope": {
        0: _ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
        1: _ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
      },
      # Single-point split on the first dual-stream block: image + text streams.
      "transformer_blocks.0": {
        "hidden_states":
        _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        "encoder_hidden_states":
        _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
      },
      # Gather the prediction back to the full sequence for the output head.
      "proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
    }
    return _cp_plan


# Upstream references for future syncs:
# - https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_motif_video.py
@functools.wraps(MotifVideoAttnProcessor2_0.__call__)
def __patch__MotifVideoAttnProcessor2_0__call__(
  self,
  attn,
  hidden_states: torch.Tensor,
  encoder_hidden_states: Optional[torch.Tensor] = None,
  attention_mask: Optional[torch.Tensor] = None,
  image_rotary_emb: Optional[torch.Tensor] = None,
):
  original = _CP_PATCH_ORIGINALS.get(MotifVideoAttnProcessor2_0)
  if original is not None and not _cp_is_armed(self):
    return original(self, attn, hidden_states, encoder_hidden_states, attention_mask,
                    image_rotary_emb)

  from diffusers.models.embeddings import apply_rotary_emb

  # Concatenate hidden states with encoder hidden states for joint attention if needed
  if attn.add_q_proj is None and encoder_hidden_states is not None:
    hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

  # Project QKV
  query = attn.to_q(hidden_states)
  key = attn.to_k(hidden_states)
  value = attn.to_v(hidden_states)

  query = query.unflatten(2, (attn.heads, -1))
  key = key.unflatten(2, (attn.heads, -1))
  value = value.unflatten(2, (attn.heads, -1))

  # Normalize QK
  if attn.norm_q is not None:
    query = attn.norm_q(query)
  if attn.norm_k is not None:
    key = attn.norm_k(key)

  # Apply RoPE to the image segment only (encoder tokens carry no RoPE)
  if image_rotary_emb is not None:
    if attn.add_q_proj is None and encoder_hidden_states is not None:
      split_idx = -encoder_hidden_states.shape[1]
      query = torch.cat(
        [
          apply_rotary_emb(query[:, :split_idx, :, :], image_rotary_emb, sequence_dim=1),
          query[:, split_idx:, :, :],
        ],
        dim=1,
      )
      key = torch.cat(
        [
          apply_rotary_emb(key[:, :split_idx, :, :], image_rotary_emb, sequence_dim=1),
          key[:, split_idx:, :, :],
        ],
        dim=1,
      )
    else:
      query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
      key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

  # Add encoder conditioning QKV projections and normalization
  if attn.add_q_proj is not None and encoder_hidden_states is not None:
    encoder_query = attn.add_q_proj(encoder_hidden_states)
    encoder_key = attn.add_k_proj(encoder_hidden_states)
    encoder_value = attn.add_v_proj(encoder_hidden_states)

    encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
    encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
    encoder_value = encoder_value.unflatten(2, (attn.heads, -1))

    if attn.norm_added_q is not None:
      encoder_query = attn.norm_added_q(encoder_query)
    if attn.norm_added_k is not None:
      encoder_key = attn.norm_added_k(encoder_key)

    query = torch.cat([query, encoder_query], dim=1)
    key = torch.cat([key, encoder_key], dim=1)
    value = torch.cat([value, encoder_value], dim=1)

  # Compute attention: route through cache-dit's CP-aware dispatch. The attention mask
  # is already in all-to-all order (permuted by the patch functor's forward, full length).
  hidden_states = _dispatch_attention_fn(
    query,
    key,
    value,
    attn_mask=attention_mask,
    backend=getattr(self, "_attention_backend", None),
    cp_config=getattr(self, "_cp_config", None),
  )
  hidden_states = hidden_states.flatten(2, 3)
  hidden_states = hidden_states.to(query.dtype)

  # Apply output projections and split encoder states
  if encoder_hidden_states is not None:
    hidden_states, encoder_hidden_states = (
      hidden_states[:, :-encoder_hidden_states.shape[1]],
      hidden_states[:, -encoder_hidden_states.shape[1]:],
    )

    if getattr(attn, "to_out", None) is not None:
      hidden_states = attn.to_out[0](hidden_states)
      hidden_states = attn.to_out[1](hidden_states)

    if getattr(attn, "to_add_out", None) is not None:
      encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

    return hidden_states, encoder_hidden_states

  if getattr(attn, "to_out", None) is not None:
    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)
  return hidden_states


@functools.wraps(MotifVideoCrossAttnProcessor2_0.__call__)
def __patch__MotifVideoCrossAttnProcessor2_0__call__(
  self,
  attn,
  hidden_states: torch.Tensor,
  encoder_hidden_states: torch.Tensor,
  attention_mask: Optional[torch.Tensor] = None,
  image_rotary_emb: Optional[torch.Tensor] = None,
  image_embed_seq_len: int = 0,
):
  original = _CP_PATCH_ORIGINALS.get(MotifVideoCrossAttnProcessor2_0)
  if original is not None and not _cp_is_armed(self):
    return original(self, attn, hidden_states, encoder_hidden_states, attention_mask,
                    image_rotary_emb, image_embed_seq_len)

  from diffusers.models.embeddings import apply_rotary_emb

  txt_kv = encoder_hidden_states[:, image_embed_seq_len:, :]

  text_mask = None
  if attention_mask is not None:
    # Under CP the joint mask arrives permuted into all-to-all order
    # `[H_0, E_0, H_1, E_1, ...]` (full length) while this rank's cross-attention K/V
    # cover only the local text segment. After the K/V all-to-all the kernel sees the
    # text tokens of every rank concatenated `[E_0, E_1, ...]`, i.e. the interleaved
    # text segments of the permuted mask — extract them in that order.
    cp_config = getattr(self, "_cp_config", None)
    if cp_config is not None and cp_config._world_size > 1:
      world_size = cp_config._world_size
      chunk = attention_mask.shape[-1] // world_size
      q_len = hidden_states.shape[1]
      text_mask = torch.cat(
        [attention_mask[..., i * chunk + q_len:(i + 1) * chunk] for i in range(world_size)],
        dim=-1,
      )
    else:
      text_mask = attention_mask[:, :, :, image_embed_seq_len - encoder_hidden_states.shape[1]:]

  query = attn.to_q(hidden_states)
  key = attn.to_k(txt_kv)
  value = attn.to_v(txt_kv)

  query = query.unflatten(2, (attn.heads, -1))
  key = key.unflatten(2, (attn.heads, -1))
  value = value.unflatten(2, (attn.heads, -1))

  if attn.norm_q is not None:
    query = attn.norm_q(query)
  if attn.norm_k is not None:
    key = attn.norm_k(key)

  if image_rotary_emb is not None:
    query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)

  # Cross-attention is asymmetric (Q = local image stream, K/V = local text segment):
  # the templated Ulysses kernels all-to-all Q and K/V independently, so this routes
  # through CP dispatch directly.
  hidden_states = _dispatch_attention_fn(
    query,
    key,
    value,
    attn_mask=text_mask,
    backend=getattr(self, "_attention_backend", None),
    cp_config=getattr(self, "_cp_config", None),
  )
  hidden_states = hidden_states.flatten(2, 3)
  hidden_states = hidden_states.to(query.dtype)

  hidden_states = attn.to_out[0](hidden_states)
  hidden_states = attn.to_out[1](hidden_states)
  return hidden_states


@TensorParallelismPlannerRegister.register("MotifVideoTransformer3DModel")
class MotifVideoTensorParallelismPlanner(TensorParallelismPlanner):
  """TP over the dual-stream blocks only (v1).

  The single-stream blocks stay replicated: their `attn` is `pre_only=True` (no `to_out`),
  so a head-sharded attention output (inner/tp) would be summed with the cross-attention
  `to_out.0` Rowwise output (full inner) inside the block — a guaranteed shape mismatch.
  Splitting the fused single-stream `proj_out` is left as a follow-up experiment.
  """

  def _apply(
    self,
    transformer: Optional[torch.nn.Module] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ):
    assert transformer is not None, "Transformer must be provided."
    _ensure_patched(transformer)

    from torch.distributed.tensor.parallel import (
      ColwiseParallel,
      RowwiseParallel,
      parallelize_module,
    )
    from torch.distributed._tensor import Replicate

    tp_mesh = self.mesh(parallelism_config=parallelism_config)

    transformer, layer_plans = self.parallelize_transformer(
      transformer=transformer,
      tp_mesh=tp_mesh,
      colwise_cls=ColwiseParallel,
      rowwise_cls=RowwiseParallel,
      replicate_cls=Replicate,
      parallelize_fn=parallelize_module,
    )
    return transformer, layer_plans

  def parallelize_transformer(
    self,
    transformer,
    tp_mesh,
    colwise_cls,
    rowwise_cls,
    replicate_cls,
    parallelize_fn,
  ):
    layer_plans = []

    # Dual-stream blocks: 12 heads x 128, pure MHA (no GQA), 12 % tp_size == 0 for tp=2/4.
    # QK-norms (norm_q / norm_k / norm_added_q / norm_added_k) and the unaffine norm2 /
    # norm2_context stay replicated intentionally — they are per-head-dim ops.
    for _, block in transformer.transformer_blocks.named_children():
      shard_div_attr(block.attn, "heads", tp_mesh.size())
      layer_plan = {
        "attn.to_q": colwise_cls(),
        "attn.to_k": colwise_cls(),
        "attn.to_v": colwise_cls(),
        "attn.add_q_proj": colwise_cls(),
        "attn.add_k_proj": colwise_cls(),
        "attn.add_v_proj": colwise_cls(),
        "attn.to_out.0": rowwise_cls(),
        "attn.to_add_out": rowwise_cls(),
        # AdaLN modulation tables (chunk semantics need the full table)
        "norm1.linear": colwise_cls(output_layouts=replicate_cls()),
        "norm1_context.linear": colwise_cls(output_layouts=replicate_cls()),
        "ff.net.0.proj": colwise_cls(),
        "ff.net.2": rowwise_cls(),
        "ff_context.net.0.proj": colwise_cls(),
        "ff_context.net.2": rowwise_cls(),
      }
      parallelize_fn(
        module=block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_plan,
      )
      layer_plans.append(layer_plan)

    return transformer, layer_plans
