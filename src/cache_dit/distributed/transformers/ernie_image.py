"""Ernie-Image distributed parallelism planners (CP + TP)."""

import functools
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import DeviceMesh
from torch.distributed.tensor.parallel import (
  ColwiseParallel,
  ParallelStyle,
  RowwiseParallel,
  parallelize_module,
)

from diffusers.models.transformers.transformer_ernie_image import (
  ErnieImageTransformer2DModel,
  ErnieImageTransformer2DModelOutput,
)

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


def _patch_transformer_forward_for_cp(transformer: ErnieImageTransformer2DModel, ) -> None:
  """Monkey-patch ``ErnieImageTransformer2DModel.forward`` for context parallelism.

  **Why full monkey-patch instead of hook-based CP plan?**

  ErnieImage's block ``forward(x, rotary_pos_emb, temb, attention_mask)``
  receives ``temb`` as a **list of 6 tensors**, each shaped ``[S, B, D]``.
  The hook-based CP mechanism (``_ContextParallelInput``) resolves parameters
  by name via ``inspect.signature`` and applies ``tensor_split`` to the
  resolved value.  For a list-valued argument this would call
  ``tensor_split(list_obj, ...)`` — a ``TypeError``.  There is no "recurse
  into list elements" semantic in the current hook infrastructure.

  A hybrid approach (hook-split ``x`` + monkey-patch-split ``temb``) is also
  infeasible: if ``temb`` is pre-split in the monkey-patch, ``x`` must be
  pre-split as well, otherwise the hook on ``layers.0`` would double-split
  ``x``.  Once ``x``, ``temb``, and ``rotary_pos_emb`` are all handled in
  the monkey-patch, the hook plan becomes empty — which is exactly what we
  do here.

  In contrast, models like QwenImage (no shared temb) or HunyuanImage
  (``temb`` is ``[B, D]``, independent of sequence length) can use pure or
  hybrid hook-based CP.  ErnieImage is the first model where the shared
  per-sequence AdaLN modulation is both **list-valued** and
  **sequence-dependent**, requiring full monkey-patching.

  **Why Ring attention is not supported?**

  ErnieImage constructs an ``attention_mask`` (shape ``[B, 1, 1, S]``) to
  mask invalid text tokens.  The cache-dit Ring attention backend does not
  support ``attn_mask`` — it raises ``ValueError: attn_mask is not yet
  supported for native flash attention with lse.``  This is a backend
  limitation, not a CP-plan issue.  Only Ulysses (``--parallel ulysses``)
  works for models that require attention masks.  Per the cache-dit
  parallelism guide: *"Always prefer Ulysses over Ring — it is more mature,
  better tested."*
  """
  _original_forward = transformer.forward

  @functools.wraps(_original_forward)
  def _cp_forward(
    self: ErnieImageTransformer2DModel,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    text_bth: torch.Tensor,
    text_lens: torch.Tensor,
    return_dict: bool = True,
  ):
    cp_config = getattr(self, "_cp_config", None)
    if cp_config is None:
      return _original_forward(hidden_states, timestep, text_bth, text_lens, return_dict)

    mesh: DeviceMesh = cp_config._flattened_mesh
    tp_size = mesh.size()
    rank = dist.get_rank(mesh.get_group())

    device, dtype = hidden_states.device, hidden_states.dtype
    B, C, H, W = hidden_states.shape
    p, Hp, Wp = self.patch_size, H // self.patch_size, W // self.patch_size
    N_img = Hp * Wp

    img_sbh = self.x_embedder(hidden_states).transpose(0, 1).contiguous()
    if self.text_proj is not None and text_bth.numel() > 0:
      text_bth = self.text_proj(text_bth)
    Tmax = text_bth.shape[1]
    text_sbh = text_bth.transpose(0, 1).contiguous()

    x = torch.cat([img_sbh, text_sbh], dim=0)

    # Position IDs & RoPE (full sequence, computed identically on every rank)
    text_ids = (torch.cat(
      [
        torch.arange(Tmax, device=device, dtype=torch.float32).view(1, Tmax, 1).expand(B, -1, -1),
        torch.zeros((B, Tmax, 2), device=device),
      ],
      dim=-1,
    ) if Tmax > 0 else torch.zeros((B, 0, 3), device=device))
    grid_yx = torch.stack(
      torch.meshgrid(
        torch.arange(Hp, device=device, dtype=torch.float32),
        torch.arange(Wp, device=device, dtype=torch.float32),
        indexing="ij",
      ),
      dim=-1,
    ).reshape(-1, 2)
    image_ids = torch.cat(
      [
        text_lens.float().view(B, 1, 1).expand(-1, N_img, -1),
        grid_yx.view(1, N_img, 2).expand(B, -1, -1),
      ],
      dim=-1,
    )
    rotary_pos_emb = self.pos_embed(torch.cat([image_ids, text_ids], dim=1))

    valid_text = (torch.arange(Tmax, device=device).view(1, Tmax) < text_lens.view(B, 1)
                  if Tmax > 0 else torch.zeros((B, 0), device=device, dtype=torch.bool))
    attention_mask = torch.cat(
      [torch.ones((B, N_img), device=device, dtype=torch.bool), valid_text], dim=1)[:, None,
                                                                                    None, :]

    # AdaLN modulation (full sequence, then we will split temb)
    sample = self.time_proj(timestep)
    sample = sample.to(dtype=dtype)
    c = self.time_embedding(sample)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
      t.unsqueeze(0).expand(x.shape[0], -1, -1).contiguous()
      for t in self.adaLN_modulation(c).chunk(6, dim=-1)
    ]
    temb = [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]

    # ---- CP: split sequence-dependent tensors ----
    # x:              [S, B, D]       → split dim=0 (seq-first)
    # temb[i]:        [S, B, D]       → split dim=0 (seq-first)
    # rotary_pos_emb: [B, S, 1, H]    → split dim=1 (batch-first)
    # attention_mask: NOT split — Ulysses all-to-all internally recovers
    #   the full sequence, so each GPU needs the full [B, 1, 1, S_full] mask.
    x = x.tensor_split(tp_size, dim=0)[rank].contiguous()
    temb = [t.tensor_split(tp_size, dim=0)[rank].contiguous() for t in temb]
    rotary_pos_emb = rotary_pos_emb.tensor_split(tp_size, dim=1)[rank].contiguous()

    # ---- Layer loop on local chunk ----
    for layer in self.layers:
      if torch.is_grad_enabled() and self.gradient_checkpointing:
        x = self._gradient_checkpointing_func(layer,
                                              x,
                                              rotary_pos_emb,
                                              temb,
                                              attention_mask=attention_mask)
      else:
        x = layer(x, rotary_pos_emb, temb, attention_mask=attention_mask)

    # ---- All-gather output (handles uneven sizes via tensor_split) ----
    # Reconstruct the full-sequence split sizes to allocate correct buffers
    full_x_shape = list(x.shape)
    full_x_shape[0] = N_img + Tmax  # original S
    chunks = [
      torch.empty(
        full_x_shape[0] // tp_size + (1 if i < full_x_shape[0] % tp_size else 0),
        *full_x_shape[1:],
        device=x.device,
        dtype=x.dtype,
      ) for i in range(tp_size)
    ]
    chunks[rank] = x
    dist.all_gather(chunks, x, group=mesh.get_group())
    x = torch.cat(chunks, dim=0)

    # ---- Post-processing (full sequence) ----
    x = self.final_norm(x, c).type_as(x)
    patches = self.final_linear(x)[:N_img].transpose(0, 1).contiguous()
    output = (patches.view(B, Hp, Wp, p, p,
                           self.out_channels).permute(0, 5, 1, 3, 2, 4).contiguous().view(
                             B, self.out_channels, H, W))

    return (ErnieImageTransformer2DModelOutput(sample=output) if return_dict else (output, ))

  transformer.forward = _cp_forward.__get__(transformer, type(transformer))


@ContextParallelismPlannerRegister.register("ErnieImage")
class ErnieImageContextParallelismPlanner(ContextParallelismPlanner):
  """Context-parallel planner for Ernie-Image.

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
      _patch_transformer_forward_for_cp(transformer)
    return {}


@TensorParallelismPlannerRegister.register("ErnieImage")
class ErnieImageTensorParallelismPlanner(TensorParallelismPlanner):
  """Tensor-parallel planner for Ernie-Image.

  **Attention: standard Shard TP (MHA, same pattern as Flux2).**

  ErnieImage is MHA (``num_attention_heads=24``, ``head_dim=128``) with
  separate ``to_q/to_k/to_v`` projections, so heads divide evenly by
  ``tp_size``.  ``ColwiseParallel`` defaults to ``use_local_output=True``, so
  ``attn.to_q(x)`` returns a **plain local tensor** ``[B, S, inner_dim/tp]``
  (not a DTensor).  Calling ``shard_div_attr(self_attention, "heads", tp)``
  divides ``attn.heads`` (24 -> 12), so the processor's
  ``query.unflatten(-1, (attn.heads, -1))`` keeps ``head_dim = 128``
  intact.  ``apply_rotary_emb`` then slices ``x_in[..., :rot_dim]`` with
  ``rot_dim = freqs_cis.shape[-1] = 128``, an exact (non-clamped) slice, and
  the full ``freqs_cis`` is shared across all heads.  ``to_out.0`` uses
  ``RowwiseParallel`` which all-reduces the per-rank partial outputs.

  Only ``heads`` needs sharding (unlike Flux2's single block, which also
  shards ``inner_dim`` / ``mlp_hidden_dim``): ErnieImage has no fused
  QKV+MLP projection and its processor references only ``attn.heads``.

  **FFN: standard Shard TP.**

  ``mlp.gate_proj`` / ``mlp.up_proj`` use ``ColwiseParallel()`` (Shard(-1)),
  the elementwise ``up * gelu(gate)`` stays sharded, and ``mlp.linear_fc2``
  uses ``RowwiseParallel()`` which all-reduces back to the full hidden size.
  All ops are pointwise / diffusers ``RMSNorm`` — DTensor-safe, so no
  fused-kernel patching is needed (unlike BooguImage).

  The transformer-level shared AdaLN modulation is computed once outside the
  blocks and applied elementwise on the full hidden size, so it is left
  unparallelized.
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
    for _, block in transformer.layers.named_children():
      shard_div_attr(block.self_attention, "heads", tp_size)
      layer_plan: Dict[str, ParallelStyle] = {
        "self_attention.to_q": ColwiseParallel(),
        "self_attention.to_k": ColwiseParallel(),
        "self_attention.to_v": ColwiseParallel(),
        "self_attention.to_out.0": RowwiseParallel(),
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.linear_fc2": RowwiseParallel(),
      }
      parallelize_module(
        module=block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_plan,
      )
      layer_plans.append(layer_plan)
    return transformer, layer_plans
