"""Context-parallel and tensor-parallel planners for AnyFlowTransformer3DModel.

AnyFlow is a flow-map distilled Wan2.1 variant with dual-timestep conditioning.
The block forward matches Wan's Pattern_2: (hidden_states, encoder_hidden_states)
-> hidden_states only.  CFG is batch-concatenated (has_separate_cfg=False).

CP strategy: hybrid -- root-level split + rope offset patch + unpack patch.
Root-level split of hidden_states (dim=1, frame axis) and timestep (dim=1)
ensures that condition_embedder produces local-length temb matching the local
tokens.  The rope offset patch fixes temporal RoPE positions (each rank gets
globally-continuous positions instead of restarting from 0).  The unpack patch
corrects num_frames from the gathered latent.

TP strategy: standard MHA (no GQA) -- ColwiseParallel on each Q/K/V + RowwiseParallel
on to_out.0, plus ColwiseParallel/RowwiseParallel for FFN.  DistributedRMSNorm
on Q/K norms (compatible with diffusers RMSNorm using ``dim`` attribute).
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from diffusers.models.modeling_utils import ModelMixin
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor.parallel import (
  ColwiseParallel,
  ParallelStyle,
  RowwiseParallel,
  parallelize_module,
)

from ...attention import _dispatch_attention_fn
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

# ============================================================================
# Context Parallelism
# ============================================================================


def _patch_anyflow_rope_for_cp(transformer):
  """Patch rope.forward to compute RoPE on the GLOBAL frame grid per-rank.

  Under root-level CP, the CP hook framework splits ``hidden_states`` along
  the time axis BEFORE ``transformer.forward`` runs, so ``rope.forward`` is
  called with a LOCAL ``num_frames``.  Without intervention, every rank
  generates RoPE temporal positions 0, 1, 2, ... -- rank 1's tokens get
  wrong positions, which causes flickering at the chunk boundary after
  Ulysses all-to-all reorders the sequence.

  The fix uses ``dist.all_gather`` to learn every rank's local frame count,
  reconstructs the GLOBAL num_frames, calls the original rope on the full
  frame grid (so positions are globally correct), then slices the resulting
  frequency table by per-rank offset and length to give each rank its local
  slice (compatible with UAA's uneven splits).
  """
  orig_forward = transformer.rope.forward

  def patched_rope(self, layout_cfg, device):
    cp = getattr(transformer, "_cp_config", None)
    if cp is not None and getattr(cp, "_world_size", 1) > 1:
      rank = cp._rank
      ws = cp._world_size
      local_frames = layout_cfg["total_frames"]
      tokens_per_frame = layout_cfg["full_token_per_frame"]

      local_t = torch.tensor([local_frames], device=device, dtype=torch.long)
      all_frames = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(ws)]
      dist.all_gather(all_frames, local_t)
      global_frames = sum(int(f.item()) for f in all_frames)

      layout_cfg_full = dict(layout_cfg)
      layout_cfg_full["total_frames"] = global_frames
      result = orig_forward(layout_cfg_full, device)

      frame_offset = sum(int(f.item()) for f in all_frames[:rank])
      tok_offset = frame_offset * tokens_per_frame
      tok_length = local_frames * tokens_per_frame
      return {
        "query": result["query"][:, :, tok_offset:tok_offset + tok_length, :],
        "key": result["key"][:, :, tok_offset:tok_offset + tok_length, :],
      }
    return orig_forward(layout_cfg, device)

  transformer.rope.forward = patched_rope.__get__(transformer.rope)


def _patch_anyflow_attn_processor_for_cp(transformer):
  """Patch AnyFlowAttnProcessor.__call__ to use cache-dit's _dispatch_attention_fn.

  The original processor calls diffusers' ``dispatch_attention_fn`` with
  ``parallel_config=self._parallel_config``.  The CP runtime sets
  ``_parallel_config`` on each processor instance, but the diffusers dispatch
  path may not route the config to the cache-dit Ulysses all-to-all kernel
  (the active backend may be a plain SDPA backend that silently drops
  ``_parallel_config``).

  This patch replaces the dispatch call with cache-dit's own
  ``_dispatch_attention_fn``, which directly invokes the Ulysses all-to-all
  attention when ``cp_config`` is set -- mirroring how the Wan CP planner
  patches ``WanAttnProcessor.__call__``.

  Only ``AnyFlowAttnProcessor`` (self-attention) is patched: cross-attention
  (``AnyFlowCrossAttnProcessor``) operates on text tokens that are NOT split
  by CP, so it must run without all-to-all.
  """
  from diffusers.models.transformers.transformer_anyflow import (
    AnyFlowAttnProcessor, )

  def patched_self_attn_call(self,
                             attn,
                             hidden_states,
                             encoder_hidden_states=None,
                             attention_mask=None,
                             rotary_emb=None):
    if encoder_hidden_states is None:
      encoder_hidden_states = hidden_states

    query = attn.to_q(hidden_states)
    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)

    if attn.norm_q is not None:
      query = attn.norm_q(query)
    if attn.norm_k is not None:
      key = attn.norm_k(key)

    query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
    key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
    value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

    if rotary_emb is not None:
      from diffusers.models.transformers.transformer_anyflow import (
        apply_rotary_emb, )
      query = apply_rotary_emb(query, rotary_emb["query"])
      key = apply_rotary_emb(key, rotary_emb["key"])

    cp_config = getattr(self, "_cp_config", None)
    hidden_states = _dispatch_attention_fn(
      query.transpose(1, 2),
      key.transpose(1, 2),
      value.transpose(1, 2),
      attn_mask=attention_mask,
      dropout_p=0.0,
      is_causal=False,
      backend=self._attention_backend,
      cp_config=cp_config,
    )
    hidden_states = hidden_states.flatten(2, 3)
    hidden_states = hidden_states.type_as(query)
    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)
    return hidden_states

  AnyFlowAttnProcessor.__call__ = patched_self_attn_call


def _patch_anyflow_unpack_for_cp(transformer):
  """Patch _unpack_latent_sequence to derive num_frames from the latent.

  Under root-level CP, hidden_states is split along the time axis, so
  ``layout_cfg["total_frames"]`` holds a LOCAL frame count.  The output of
  ``proj_out`` is gathered back to the full sequence by the CP framework,
  but ``_unpack_latent_sequence`` is still called with the local
  ``num_frames`` -- producing a shape mismatch.

  This patch recomputes ``num_frames`` from the gathered latent tensor.
  """
  orig_unpack = transformer._unpack_latent_sequence

  def patched_unpack(latents, num_frames, height, width, patch_size):
    cp = getattr(transformer, "_cp_config", None)
    if cp is not None and getattr(cp, "_world_size", 1) > 1:
      _, num_patches, _ = latents.shape
      height_tiles = height // patch_size
      width_tiles = width // patch_size
      num_frames = num_patches // (height_tiles * width_tiles)
    return orig_unpack(latents, num_frames, height, width, patch_size)

  transformer._unpack_latent_sequence = patched_unpack


@ContextParallelismPlannerRegister.register("AnyFlowTransformer3D")
class AnyFlowContextParallelismPlanner(ContextParallelismPlanner):

  def _apply(
    self,
    transformer: Optional[torch.nn.Module | ModelMixin] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:
    _patch_anyflow_rope_for_cp(transformer)
    _patch_anyflow_attn_processor_for_cp(transformer)
    _patch_anyflow_unpack_for_cp(transformer)

    _cp_plan = {
      "": {
        "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=5, split_output=False),
        "timestep": _ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
        "r_timestep": _ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
      },
      "proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
    }
    return _cp_plan


# ============================================================================
# Tensor Parallelism
# ============================================================================


class DistributedRMSNorm(nn.Module):
  """RMSNorm sharded across the TP mesh.

  Each rank holds ``weight / tp_size`` channels; the all-reduce across the
  RMSNorm is avoided by squaring locally and relying on the downstream
  RowwiseParallel all-reduce to implicitly sum the partial contributions.
  Compatible with both ``torch.nn.RMSNorm`` and
  ``diffusers.models.normalization.RMSNorm``.
  """

  def __init__(
    self,
    tp_mesh: DeviceMesh,
    normalized_shape,
    eps: Optional[float],
    elementwise_affine: bool,
    weight: torch.nn.parameter.Parameter,
  ):
    super().__init__()
    self.tp_mesh = tp_mesh
    self.elementwise_affine = elementwise_affine
    self.normalized_shape = normalized_shape
    self.eps = eps
    if self.elementwise_affine:
      assert weight is not None
    self.weight = weight

  @classmethod
  def from_rmsnorm(cls, tp_mesh: DeviceMesh, rmsnorm):
    if hasattr(rmsnorm, "normalized_shape"):
      norm_shape = rmsnorm.normalized_shape
    elif hasattr(rmsnorm, "dim"):
      norm_shape = rmsnorm.dim
    else:
      norm_shape = rmsnorm.weight.shape

    if rmsnorm.weight is not None:
      tp_size = tp_mesh.get_group().size()
      tp_rank = tp_mesh.get_group().rank()
      weight = rmsnorm.weight.chunk(tp_size, dim=0)[tp_rank]
    else:
      weight = None

    return cls(
      tp_mesh=tp_mesh,
      normalized_shape=norm_shape,
      eps=getattr(rmsnorm, "eps", 1e-6),
      elementwise_affine=getattr(rmsnorm, "elementwise_affine", rmsnorm.weight is not None),
      weight=weight,
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.float()
    var = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(var + self.eps)
    if self.elementwise_affine:
      x_normed = x_normed * self.weight.to(device=x.device)
    return x_normed.to(orig_dtype)


def _patch_anyflow_far_causal_backend(transformer):
  """Preserve the ``flex`` backend on AnyFlow-FAR causal self-attention.

  ``AnyFlowCausalAttnProcessor`` requires the ``flex`` backend because its
  causal :class:`BlockMask` is only consumed by ``flex_attention``. cache-dit's
  TP dispatch calls ``transformer.set_attention_backend(<native>)`` *after*
  the planner returns, which would otherwise overwrite the causal processor's
  hard-coded ``flex`` default and raise at the first forward.

  This wraps ``set_attention_backend`` so that causal self-attention processors
  (``AnyFlowCausalAttnProcessor``) keep ``flex`` while cross-attention and any
  other processors still receive the requested backend. Only the FAR causal
  transformer is affected; the bidirectional AnyFlow path is untouched.
  """
  if "FAR" not in transformer.__class__.__name__:
    return
  from diffusers.models.transformers.transformer_anyflow_far import (
    AnyFlowCausalAttnProcessor, )

  def patched_set_backend(self, backend, *args, **kwargs):
    from diffusers.models.attention_processor import Attention, MochiAttention
    from diffusers.models.attention import AttentionModuleMixin

    attention_classes = (Attention, MochiAttention, AttentionModuleMixin)
    for module in self.modules():
      if not isinstance(module, attention_classes):
        continue
      processor = module.processor
      if processor is None or not hasattr(processor, "_attention_backend"):
        continue
      # Causal self-attention must stay on flex (BlockMask requirement);
      # all other processors accept the requested backend.
      if isinstance(processor, AnyFlowCausalAttnProcessor):
        processor._attention_backend = "flex"
      else:
        processor._attention_backend = backend

  transformer.set_attention_backend = patched_set_backend.__get__(transformer)


@TensorParallelismPlannerRegister.register("AnyFlow")
class AnyFlowTensorParallelismPlanner(TensorParallelismPlanner):

  def _apply(
    self,
    transformer: torch.nn.Module,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    tp_mesh = self.mesh(parallelism_config=parallelism_config)
    _patch_anyflow_far_causal_backend(transformer)
    transformer, layer_plans = self.parallelize_transformer(
      transformer=transformer,
      tp_mesh=tp_mesh,
    )
    return transformer, layer_plans

  def parallelize_transformer(
    self,
    transformer: nn.Module,
    tp_mesh: DeviceMesh,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:

    def prepare_block(block: nn.Module):
      tp_size = tp_mesh.size()
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
        "ffn.net.0.proj": ColwiseParallel(),
        "ffn.net.2": RowwiseParallel(),
      }
      parallelize_module(
        module=block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_plan,
      )
      block.attn1.norm_q = DistributedRMSNorm.from_rmsnorm(tp_mesh, block.attn1.norm_q)
      block.attn1.norm_k = DistributedRMSNorm.from_rmsnorm(tp_mesh, block.attn1.norm_k)
      block.attn2.norm_q = DistributedRMSNorm.from_rmsnorm(tp_mesh, block.attn2.norm_q)
      block.attn2.norm_k = DistributedRMSNorm.from_rmsnorm(tp_mesh, block.attn2.norm_k)
      return layer_plan

    layer_plans = []
    for _, block in transformer.blocks.named_children():
      layer_plans.append(prepare_block(block))
    # AnyFlow-FAR causal transformer allocates a cross-chunk KV cache in
    # ``AnyFlowFARPipeline.__call__`` sized by ``config.num_attention_heads``.
    # TP shards the heads on each rank (``shard_div_attr`` sets
    # ``block.attn1.heads = num_attention_heads // tp_size``), so the
    # per-rank KV cache must also be sized by the sharded head count -- the
    # causal processor writes ``key``/``value`` with the local head count
    # and would otherwise hit a shape mismatch. ``transformer.forward`` and
    # the attention processors read ``attn.heads`` (module attr, already
    # sharded), never ``config.num_attention_heads``, so patching the config
    # only affects the pipeline-side KV cache allocation. The bidirectional
    # AnyFlow path has no such cache and is left untouched.
    if "FAR" in transformer.__class__.__name__:
      transformer.config.num_attention_heads = (transformer.config.num_attention_heads //
                                                tp_mesh.size())
    return transformer, layer_plans
