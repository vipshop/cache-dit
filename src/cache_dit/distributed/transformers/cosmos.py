"""Cosmos Predict2 transformer distributed parallelism planners (CP + TP).

`CosmosTransformer3DModel` is the backbone of both the Cosmos Predict video pipelines and
the Anima text-to-image modular pipeline. The block stack is uniform
(`transformer_blocks`, Pattern_2 blocks with `attn1` self-attention + `attn2` cross-attention
+ GELU `FeedForward`), and the per-block modulation (`embedded_timestep` / `temb`) is a
`[B, C]` broadcast, so TP shards every block with one layer plan.

CP uses a single-point split at `transformer_blocks.0`: the root forward receives 5D
latents (`padding_mask` is concatenated as extra channels inside `patch_embed`, it is not
an attention mask), so the token sequence only exists after patchification. `hidden_states`
and `encoder_hidden_states` are sharded on the first block and stay sharded (the Ulysses
all-to-all inside attention reassembles the full keys/values), the `rope` tables are
output-split per rank, and `proj_out` gathers the full sequence for the output head.
`CosmosAttnProcessor2_0` is patched once for both `attn1` and `attn2` so the dispatch
receives the CP config; the patch is inert while the processor carries no `_cp_config`
(`_dispatch_attention_fn` falls back to the plain backend), and the encoder `attention_mask`
of the Cosmos video path is not supported under CP (Anima never passes one).
"""

import functools
from typing import Dict, List, Optional, Tuple

import torch
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.transformer_cosmos import (
  CosmosAttnProcessor2_0,
  CosmosTransformer3DModel,
)
from torch.distributed.tensor.parallel import (
  ColwiseParallel,
  ParallelStyle,
  RowwiseParallel,
  parallelize_module,
)

from ...attention import _dispatch_attention_fn
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


@ContextParallelismPlannerRegister.register("CosmosTransformer3DModel")
class CosmosContextParallelismPlanner(ContextParallelismPlanner):
  """Context parallelism for `CosmosTransformer3DModel` (Cosmos Predict2 / Anima)."""

  def _apply(
    self,
    transformer: Optional[torch.nn.Module | ModelMixin] = None,
    parallelism_config=None,
    **kwargs,
  ) -> _ContextParallelModelPlan:
    assert isinstance(transformer, CosmosTransformer3DModel)
    if transformer.learnable_pos_embed is not None:
      raise NotImplementedError(
        "Cosmos context parallelism does not support `extra_pos_embed_type='learnable'` "
        "(Cosmos-Predict2 checkpoints): the learnable position embedding would need its "
        "own output-split hook.")
    if transformer.config.img_context_dim_in:
      raise NotImplementedError(
        "Cosmos context parallelism does not support image-context checkpoints: "
        "`attn2` then runs `CosmosAttnProcessor2_5`, which is not CP-patched.")
    if transformer.config.controlnet_block_every_n is not None:
      raise NotImplementedError(
        "Cosmos context parallelism does not support ControlNet checkpoints: the "
        "per-block control residuals are full-sequence tensors.")

    CosmosAttnProcessor2_0.__call__ = __patch__CosmosAttnProcessor2_0__call__
    if not hasattr(CosmosAttnProcessor2_0, "_cp_config"):
      CosmosAttnProcessor2_0._cp_config = None
    if not hasattr(CosmosAttnProcessor2_0, "_attention_backend"):
      CosmosAttnProcessor2_0._attention_backend = None

    _cp_plan = {
      # `rope` returns `(cos, sin)`, each `[S, D]` over the full token sequence: split both
      # outputs so every rank applies positions of its local chunk. `split_output=True`
      # skips the input split (the rope input is the 5D latent, not a sequence tensor).
      "rope": {
        0: _ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
        1: _ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
      },
      # Single-point split for the latents: root input is 5D, the token sequence starts at
      # the first block.
      "transformer_blocks.0": {
        "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
      },
      # The forward keeps the full text in a loop-invariant local variable and passes it to
      # every block, so each block re-splits `encoder_hidden_states` on entry; the cross
      # attention all-to-all then reassembles the full keys/values without duplicates.
      "transformer_blocks.*": {
        "encoder_hidden_states":
        _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
      },
      # Gather the full sequence for the output head (hidden_states is still local there).
      "proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
    }
    return _cp_plan


# Adapted from diffusers' Cosmos transformer implementation (processor `__call__`):
# https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_cosmos.py
# Two deltas from upstream: the attention dispatch goes through cache-dit's
# `_dispatch_attention_fn` so the Ulysses all-to-all receives the CP config, and the
# `torch.onnx.is_in_onnx_export()` branch is not kept (cache-dit does not target ONNX
# export). With no `_cp_config` attached the call is equivalent to the upstream dispatch.
@functools.wraps(CosmosAttnProcessor2_0.__call__)
def __patch__CosmosAttnProcessor2_0__call__(
  self,
  attn: torch.nn.Module,
  hidden_states: torch.Tensor,
  encoder_hidden_states: Optional[torch.Tensor] = None,
  attention_mask: Optional[torch.Tensor] = None,
  image_rotary_emb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
  # 1. QKV projections
  if encoder_hidden_states is None:
    encoder_hidden_states = hidden_states

  query = attn.to_q(hidden_states)
  key = attn.to_k(encoder_hidden_states)
  value = attn.to_v(encoder_hidden_states)

  query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
  key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
  value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

  # 2. QK normalization
  query = attn.norm_q(query)
  key = attn.norm_k(key)

  # 3. Apply RoPE
  if image_rotary_emb is not None:
    from diffusers.models.embeddings import apply_rotary_emb

    query = apply_rotary_emb(query, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)
    key = apply_rotary_emb(key, image_rotary_emb, use_real=True, use_real_unbind_dim=-2)

  # 4. Prepare for GQA
  query_idx = query.size(3)
  key_idx = key.size(3)
  value_idx = value.size(3)
  key = key.repeat_interleave(query_idx // key_idx, dim=3)
  value = value.repeat_interleave(query_idx // value_idx, dim=3)

  # 5. Attention
  hidden_states = _dispatch_attention_fn(
    query.transpose(1, 2),
    key.transpose(1, 2),
    value.transpose(1, 2),
    attn_mask=attention_mask,
    dropout_p=0.0,
    is_causal=False,
    backend=getattr(self, "_attention_backend", None),
    cp_config=getattr(self, "_cp_config", None),
  )
  hidden_states = hidden_states.flatten(2, 3).type_as(query)
  hidden_states = attn.to_out[0](hidden_states)
  hidden_states = attn.to_out[1](hidden_states)

  return hidden_states


@TensorParallelismPlannerRegister.register("CosmosTransformer3DModel")
class CosmosTensorParallelismPlanner(TensorParallelismPlanner):
  """Tensor parallelism for `CosmosTransformer3DModel` (Cosmos Predict2 / Anima)."""

  def _apply(
    self,
    transformer: torch.nn.Module,
    parallelism_config,
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
    tp_mesh,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    """Shard the attention and FFN projections of every block.

    :param transformer: Cosmos Predict2 transformer.
    :param tp_mesh: TP device mesh.
    :returns: The sharded transformer and the applied layer plans.
    """
    tp_size = tp_mesh.size()
    layer_plans = []

    for block in transformer.transformer_blocks:
      # The processor reshapes Q/K/V with `attn.heads`, which must follow the sharded
      # projection size (16 -> 8 for tp_size=2). Both attention layers need it: `attn2`
      # is an independent `Attention` instance (plain cross attention for Anima, whose
      # K/V come from the 1024-wide text stream).
      shard_div_attr(block.attn1, "heads", tp_size)
      shard_div_attr(block.attn2, "heads", tp_size)
      layer_plan = {
        "attn1.to_q": ColwiseParallel(),
        "attn1.to_k": ColwiseParallel(),
        "attn1.to_v": ColwiseParallel(),
        "attn1.to_out.0": RowwiseParallel(),
        "attn2.to_q": ColwiseParallel(),
        "attn2.to_k": ColwiseParallel(),
        "attn2.to_v": ColwiseParallel(),
        "attn2.to_out.0": RowwiseParallel(),
        # diffusers GELU `FeedForward`: `ff.net.0.proj` raises 2048 -> 8192, `ff.net.2`
        # projects back to 2048 (there is no gate/up pair in this FFN).
        "ff.net.0.proj": ColwiseParallel(),
        "ff.net.2": RowwiseParallel(),
      }
      parallelize_module(module=block, device_mesh=tp_mesh, parallelize_plan=layer_plan)
      layer_plans.append(layer_plan)

    return transformer, layer_plans
