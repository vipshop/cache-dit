"""Krea-2 (K2) single-stream MMDiT distributed parallelism planners (CP + TP)."""

import functools
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed import DeviceMesh
from torch.distributed.tensor.parallel import (
  ColwiseParallel,
  ParallelStyle,
  RowwiseParallel,
  parallelize_module,
)

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_krea2 import (
  Krea2Transformer2DModel,
  Krea2AttnProcessor,
)

from ...attention import _dispatch_attention_fn
from ...distributed.core import _ContextParallelModelPlan
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


def _patch_attention_processor_for_cp() -> None:
  """Monkey-patch ``Krea2AttnProcessor.__call__`` for context parallelism.

  Krea2 uses GQA with ``num_heads=48, num_kv_heads=12`` (4:1 ratio).  The
  templated Ulysses/UAA attention in cache-dit does not support ``enable_gqa``,
  so we repeat K/V heads to match Q heads before dispatching and pass
  ``enable_gqa=False``.

  .. important::

     This GQA→MHA repeat has a **significant secondary performance benefit**
     beyond CP compatibility.  The PyTorch SDPA backend selected for
     ``enable_gqa=True`` (48 Q heads, 12 KV heads, 128 head_dim, 4608 seq)
     falls back to a slow path (likely ``math`` or an inefficient
     ``mem_efficient`` kernel).  After the repeat, with ``enable_gqa=False``,
     SDPA selects the fast flash-attention / cuDNN MHA path, yielding a
     **~2.2× speedup on a single GPU** independently of any parallelism.

     Measured on NVIDIA L20 (Krea-2-Turbo, 8-step, 1024×1024, bf16):

     - ``enable_gqa=True``  (original): ~26.4 s
     - ``enable_gqa=False`` (MHA repeat): ~12.2 s

     This means the GQA repeat should be applied unconditionally (even without
     CP) for any model with a similar GQA head configuration.  The CP path
     happens to get this optimization "for free".

  This is a one-time class-level patch — it is safe to call multiple times.
  """
  if getattr(Krea2AttnProcessor, "_cp_patched", False):
    return

  _original_call = Krea2AttnProcessor.__call__

  @functools.wraps(_original_call)
  def _patched_call(
    self: Krea2AttnProcessor,
    attn,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    image_rotary_emb: Optional[tuple] = None,
  ) -> torch.Tensor:
    cp_config = getattr(self, "_cp_config", None)
    if cp_config is None:
      return _original_call(self, attn, hidden_states, attention_mask, image_rotary_emb)

    from diffusers.models.embeddings import apply_rotary_emb

    query = attn.to_q(hidden_states).unflatten(-1, (attn.num_heads, attn.head_dim))
    key = attn.to_k(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
    value = attn.to_v(hidden_states).unflatten(-1, (attn.num_kv_heads, attn.head_dim))
    gate = attn.to_gate(hidden_states)

    query = attn.norm_q(query)
    key = attn.norm_k(key)

    if image_rotary_emb is not None:
      query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
      key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

    # GQA: repeat K/V to match Q heads before UAA processing
    kv_heads = attn.num_kv_heads
    q_heads = attn.num_heads
    if kv_heads < q_heads:
      repeat_factor = q_heads // kv_heads
      key = key.repeat_interleave(repeat_factor, dim=2)
      value = value.repeat_interleave(repeat_factor, dim=2)

    hidden_states = _dispatch_attention_fn(
      query,
      key,
      value,
      attn_mask=attention_mask,
      enable_gqa=False,  # heads already matched after repeat
      cp_config=cp_config,
    )
    hidden_states = hidden_states.flatten(2, 3)
    hidden_states = hidden_states * torch.sigmoid(gate)
    return attn.to_out[0](hidden_states)

  Krea2AttnProcessor.__call__ = _patched_call
  Krea2AttnProcessor._cp_patched = True


def _patch_transformer_forward_for_cp(transformer: Krea2Transformer2DModel) -> None:
  """Monkey-patch ``Krea2Transformer2DModel.forward`` for context parallelism.

  Krea2 is a single-stream MMDiT: text and image tokens are concatenated into
  one sequence before the transformer block loop.  The text-fusion stage needs
  the full text sequence (its own attention blocks), so we cannot split at the
  transformer entry.  Instead we:

  1. Run preprocessing (time embedding, text fusion, projections, concat,
     rotary embedding) on the FULL sequence on every GPU.
  2. Split ``hidden_states`` along the sequence dimension (dim=1) and the
     ``image_rotary_emb`` tuple along the token dimension (dim=0) before the
     block loop.
  3. Keep ``temb_mod`` (shared per-timestep modulation) and
     ``attention_mask`` (Ulysses recovers the full sequence internally)
     un-split.
  4. Run the 28 transformer blocks on the local chunk.
  5. All-gather ``hidden_states`` after the block loop.
  6. Continue with post-processing (slice image tokens, final layer).

  The attention processor already calls ``dispatch_attention_fn`` with
  ``self._parallel_config``, and cache-dit's CP runtime automatically sets
  ``_parallel_config`` on every ``Krea2AttnProcessor`` during
  ``_enable_context_parallelism`` — no processor patch is needed.
  """
  _original_forward = transformer.forward

  @functools.wraps(_original_forward)
  def _cp_forward(
    self: Krea2Transformer2DModel,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    position_ids: torch.Tensor,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    attention_kwargs: Optional[Dict] = None,
    return_dict: bool = True,
  ):
    cp_config = getattr(self, "_cp_config", None)
    if cp_config is None:
      return _original_forward(
        hidden_states,
        encoder_hidden_states,
        timestep,
        position_ids,
        encoder_attention_mask,
        attention_kwargs,
        return_dict,
      )

    mesh: DeviceMesh = cp_config._flattened_mesh
    tp_size = mesh.size()
    rank = dist.get_rank(mesh.get_group())

    batch_size, image_seq_len, _ = hidden_states.shape
    text_seq_len = encoder_hidden_states.shape[1]

    # Time embedding (full, per-timestep)
    temb = self.time_embed(timestep, dtype=hidden_states.dtype)
    temb_mod = self.time_mod_proj(F.gelu(temb, approximate="tanh"))

    # Attention masks
    text_attention_mask = None
    attention_mask = None
    if encoder_attention_mask is not None:
      text_attention_mask = encoder_attention_mask[:, None, None, :]
      image_mask = encoder_attention_mask.new_ones((batch_size, image_seq_len))
      attention_mask = torch.cat([encoder_attention_mask, image_mask], dim=1)[:, None, None, :]

    # Text fusion + projection (full sequence)
    # Temporarily clear _cp_config on text_fusion attention processors so
    # Ulysses does not interfere with full-sequence attention inside the
    # layerwise / refiner blocks of Krea2TextFusion.
    _tf_cp_backup = {}
    for tf_mod in self.text_fusion.modules():
      tf_proc = getattr(tf_mod, "processor", None)
      if tf_proc is not None:
        _tf_cp_backup[tf_proc] = (
          getattr(tf_proc, "_cp_config", None),
          getattr(tf_proc, "_parallel_config", None),
        )
        tf_proc._cp_config = None
        tf_proc._parallel_config = None

    encoder_hidden_states = self.text_fusion(encoder_hidden_states,
                                             attention_mask=text_attention_mask)

    for tf_proc, (cp_cfg, par_cfg) in _tf_cp_backup.items():
      tf_proc._cp_config = cp_cfg
      tf_proc._parallel_config = par_cfg

    encoder_hidden_states = self.txt_in(encoder_hidden_states)

    # Image projection + concat
    hidden_states = self.img_in(hidden_states)
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    # Rotary embeddings (full sequence)
    image_rotary_emb = self.rotary_emb(position_ids)

    # ---- CP: split sequence-dependent tensors ----
    # hidden_states: (B, S_total, D) -> split dim=1
    hidden_states = hidden_states.tensor_split(tp_size, dim=1)[rank].contiguous()
    # image_rotary_emb: tuple of (S_total, D_rope) -> split dim=0
    cos_rot, sin_rot = image_rotary_emb
    cos_rot = cos_rot.tensor_split(tp_size, dim=0)[rank].contiguous()
    sin_rot = sin_rot.tensor_split(tp_size, dim=0)[rank].contiguous()
    image_rotary_emb = (cos_rot, sin_rot)
    # temb_mod: NOT split (shared per-timestep modulation)
    # attention_mask: NOT split (Ulysses recovers full sequence)

    # ---- Block loop on local chunk ----
    for block in self.transformer_blocks:
      if torch.is_grad_enabled() and self.gradient_checkpointing:
        hidden_states = self._gradient_checkpointing_func(
          block,
          hidden_states,
          temb_mod,
          image_rotary_emb,
          attention_mask,
        )
      else:
        hidden_states = block(hidden_states, temb_mod, image_rotary_emb, attention_mask)

    # ---- All-gather output ----
    total_seq = text_seq_len + image_seq_len
    # Compute per-rank chunk sizes for the full sequence (handles uneven splits)
    S_local = total_seq // tp_size
    remainder = total_seq % tp_size
    chunks = [
      torch.empty(
        batch_size,
        S_local + (1 if i < remainder else 0),
        *hidden_states.shape[2:],
        device=hidden_states.device,
        dtype=hidden_states.dtype,
      ) for i in range(tp_size)
    ]
    chunks[rank] = hidden_states
    dist.all_gather(chunks, hidden_states, group=mesh.get_group())
    hidden_states = torch.cat(chunks, dim=1)

    # ---- Post-processing ----
    hidden_states = hidden_states[:, text_seq_len:]
    output = self.final_layer(hidden_states, temb)

    if not return_dict:
      return (output, )
    return Transformer2DModelOutput(sample=output)

  transformer.forward = _cp_forward.__get__(transformer, type(transformer))


@ContextParallelismPlannerRegister.register("Krea2Transformer2DModel")
class Krea2ContextParallelismPlanner(ContextParallelismPlanner):
  """Context-parallel planner for Krea-2.

  Returns an empty CP plan ``{}`` — all split / all-gather is handled
  inside the monkey-patched forward.  See
  :func:`_patch_transformer_forward_for_cp` for the rationale.
  """

  def _apply(
    self,
    transformer: Optional[torch.nn.Module] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:
    if transformer is not None:
      _patch_attention_processor_for_cp()
      _patch_transformer_forward_for_cp(transformer)
    return {}


@TensorParallelismPlannerRegister.register("Krea2")
class Krea2TensorParallelismPlanner(TensorParallelismPlanner):
  """Tensor-parallel planner for Krea-2.

  Krea2 has MHA with GQA (48 Q heads, 12 KV heads, 4:1 ratio).  Both head
  counts are divisible by ``tp_size=2``, so we use standard ``ColwiseParallel``
  for Q/K/V/gate projections and ``RowwiseParallel`` for the output projection.

  Attention module uses ``num_heads`` and ``num_kv_heads`` (not ``heads``),
  so ``shard_div_attr`` targets those attribute names explicitly.

  The ``Krea2RMSNorm`` modules (``norm_q``, ``norm_k``, ``norm1``, ``norm2``)
  are elementwise and need no parallelization.

  The per-block ``scale_shift_table`` (shape ``(6, hidden_size)``) is not
  parallelized — it is combined with the shared ``temb_mod`` before the
  elementwise scale/shift/gate modulation.
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

  def parallelize_transformer(
    self,
    transformer: torch.nn.Module,
    tp_mesh: DeviceMesh,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    tp_size = tp_mesh.size()
    layer_plans: List[Dict[str, ParallelStyle]] = []
    for _, block in transformer.transformer_blocks.named_children():
      shard_div_attr(block.attn, "num_heads", tp_size)
      shard_div_attr(block.attn, "num_kv_heads", tp_size)
      layer_plan: Dict[str, ParallelStyle] = {
        "attn.to_q": ColwiseParallel(),
        "attn.to_k": ColwiseParallel(),
        "attn.to_v": ColwiseParallel(),
        "attn.to_gate": ColwiseParallel(),
        "attn.to_out.0": RowwiseParallel(),
        "ff.gate": ColwiseParallel(),
        "ff.up": ColwiseParallel(),
        "ff.down": RowwiseParallel(),
      }
      parallelize_module(
        module=block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_plan,
      )
      layer_plans.append(layer_plan)
    return transformer, layer_plans
