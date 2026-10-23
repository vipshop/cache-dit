"""PRX / PRXPixel distributed parallelism planners (CP and TP)."""

import functools
from typing import Dict, List, Optional, Tuple

import torch
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.transformers.transformer_prx import (
  PRXAttnProcessor2_0,
  PRXTransformer2DModel,
  apply_rope,
)
from torch.distributed import DeviceMesh
from torch.distributed._tensor import Replicate
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
from ..config import ParallelismConfig
from .register import (
  ContextParallelismPlanner,
  ContextParallelismPlannerRegister,
  TensorParallelismPlanner,
  TensorParallelismPlannerRegister,
)

# Upstream `__call__` captured before the class-level CP patch is installed. Unarmed processors
# hand control back to it, so a cp-free run stays byte-identical to upstream.
_CP_PATCH_ORIGINALS: dict = {}


def _cp_is_armed(processor) -> bool:
  """Whether `processor` was armed by the CP runtime with a multi-rank `_cp_config`."""
  cp_config = getattr(processor, "_cp_config", None)
  return cp_config is not None and cp_config._world_size > 1


def _build_cp_joint_mask(
  processor,
  attention_mask: Optional[torch.Tensor],
  batch_size: int,
  local_txt: int,
  local_img: int,
  device: torch.device,
) -> Optional[torch.Tensor]:
  """Rebuild the joint text+image key mask in the global Ulysses all-to-all order.

  ``PRXAttnProcessor2_0`` concatenates the text K/V with the image K/V and masks the whole
  joint key sequence with ``cat([text_mask, ones(L_img)])``. Under CP the image tokens are
  sharded while the text mask still describes the global text sequence, and the Ulysses
  all-to-all lays the global sequence out as ``cat_r([text_r, img_r])``. The mask therefore
  has to be rebuilt in that order, otherwise its length no longer matches the global K (or,
  worse, it silently masks the wrong keys).

  :param processor: Attention processor carrying the active `_cp_config`.
  :param attention_mask: Text padding mask, `(B, L_txt_global)`.
  :param batch_size: Batch size of the local image tokens.
  :param local_txt: Local text token count on this rank (post CP split).
  :param local_img: Local image token count on this rank.
  :param device: Device of the local image tokens.
  :returns: A broadcastable mask `(B, 1, 1, L_txt_global + world * local_img)`.
  """
  if attention_mask is None:
    return None

  if attention_mask.dim() != 2:
    raise ValueError(f"Unsupported attention_mask shape: {attention_mask.shape}")

  # Upstream casts the mask to bool on the image device; a float mask would be read as an
  # additive bias by SDPA, so mixing it with `ones_img` would *reward* padded keys.
  attention_mask = attention_mask.to(device=device, dtype=torch.bool)

  world_size = processor._cp_config._world_size
  if attention_mask.shape[0] != batch_size:
    raise ValueError(
      f"CP joint mask batch {attention_mask.shape[0]} does not match the local image batch "
      f"{batch_size}.")
  # The text sequence is equipartitioned by `_EquipartitionSharder.shard` (`chunk`), so the
  # global mask must be exactly `world_size * local_txt` long and every chunk must be full size.
  if attention_mask.shape[-1] != local_txt * world_size:
    raise ValueError(f"CP joint mask expects a global text length of {local_txt * world_size}, got "
                     f"{attention_mask.shape[-1]}. The text split and the attention mask disagree.")

  chunks = attention_mask.chunk(world_size, dim=-1)
  if len(chunks) != world_size or any(chunk.shape[-1] != local_txt for chunk in chunks):
    raise ValueError(
      f"CP joint mask cannot be rebuilt from chunks of {local_txt} text tokens per rank "
      f"(world_size={world_size}, shapes={[tuple(c.shape) for c in chunks]}).")

  ones_img = torch.ones((batch_size, local_img), dtype=torch.bool, device=device)
  parts = []
  # The global sequence is the concatenation of each rank's `[text, image]` shard, in rank order.
  for text_chunk in chunks:
    parts.append(text_chunk)
    parts.append(ones_img)
  joint_mask = torch.cat(parts, dim=-1)

  return joint_mask[:, None, None, :]


@functools.wraps(PRXAttnProcessor2_0.__call__)
def __patch__PRXAttnProcessor2_0__call__(
  self: PRXAttnProcessor2_0,
  attn,
  hidden_states: torch.Tensor,
  encoder_hidden_states: Optional[torch.Tensor] = None,
  attention_mask: Optional[torch.Tensor] = None,
  image_rotary_emb: Optional[torch.Tensor] = None,
  **kwargs,
) -> torch.Tensor:
  original = _CP_PATCH_ORIGINALS.get(PRXAttnProcessor2_0)
  if original is not None and not _cp_is_armed(self):
    return original(
      self,
      attn,
      hidden_states,
      encoder_hidden_states=encoder_hidden_states,
      attention_mask=attention_mask,
      image_rotary_emb=image_rotary_emb,
      **kwargs,
    )

  if encoder_hidden_states is None:
    raise ValueError("PRXAttnProcessor2_0 requires 'encoder_hidden_states' containing text tokens.")

  img_qkv = attn.img_qkv_proj(hidden_states)
  batch_size, local_img, _ = img_qkv.shape
  img_qkv = img_qkv.reshape(batch_size, local_img, 3, attn.heads, attn.head_dim)
  img_qkv = img_qkv.permute(2, 0, 3, 1, 4)
  img_q, img_k, img_v = img_qkv[0], img_qkv[1], img_qkv[2]

  img_q = attn.norm_q(img_q)
  img_k = attn.norm_k(img_k)

  txt_kv = attn.txt_kv_proj(encoder_hidden_states)
  batch_size, local_txt, _ = txt_kv.shape
  txt_kv = txt_kv.reshape(batch_size, local_txt, 2, attn.heads, attn.head_dim)
  txt_kv = txt_kv.permute(2, 0, 3, 1, 4)
  txt_k, txt_v = txt_kv[0], txt_kv[1]
  txt_k = attn.norm_added_k(txt_k)

  if image_rotary_emb is not None:
    img_q = apply_rope(img_q, image_rotary_emb)
    img_k = apply_rope(img_k, image_rotary_emb)

  key = torch.cat((txt_k, img_k), dim=2)
  value = torch.cat((txt_v, img_v), dim=2)

  attn_mask_tensor = _build_cp_joint_mask(
    self,
    attention_mask,
    batch_size=batch_size,
    local_txt=local_txt,
    local_img=local_img,
    device=img_q.device,
  )

  query = img_q.transpose(1, 2)
  attn_output = dispatch_attention_fn(
    query,
    key.transpose(1, 2),
    value.transpose(1, 2),
    attn_mask=attn_mask_tensor,
    backend=self._attention_backend,
    parallel_config=self._parallel_config,
  )

  attn_output = attn_output.reshape(batch_size, local_img, attn.heads * attn.head_dim)
  attn_output = attn.to_out[0](attn_output)
  if len(attn.to_out) > 1:
    attn_output = attn.to_out[1](attn_output)
  return attn_output


def _install_cp_processor_patches() -> None:
  """Install the class-level attention patch the PRX CP path needs.

  Only the joint text+image key mask depends on the global all-to-all layout; every other
  operand (local hidden states, local text states, local RoPE) is already correct after the
  hook-based split. The patch is class-level, so it also reaches processors this runtime never
  armed (a pipeline holding a second attention tree); those delegate to the untouched upstream
  ``__call__``.
  """
  _CP_PATCH_ORIGINALS.setdefault(PRXAttnProcessor2_0, PRXAttnProcessor2_0.__call__)
  PRXAttnProcessor2_0.__call__ = __patch__PRXAttnProcessor2_0__call__  # type: ignore[assignment]


@ContextParallelismPlannerRegister.register("PRXTransformer2DModel")
class PrxContextParallelismPlanner(ContextParallelismPlanner):
  """Transformer-level CP for PRX / PRXPixel (Ulysses).

  PRX keeps ``txt`` and ``pe`` as loop-invariant locals, so splitting them at ``blocks.0``
  would only affect the first block. Instead the text sequence is split right after
  ``txt_in`` (``split_output=True``) and the image ``ids`` are split before ``pe_embedder``,
  which makes both local for every block. The 4D pixel latents are never split at the root:
  ``H/W`` feed ``get_image_ids``, ``resolution_embedder`` and ``seq2img``, so the 3D patch
  tokens are split at the first block and gathered at ``final_layer``.

  Ring is not supported: the pipeline always passes a text padding mask, and the native
  backend rejects ``attn_mask`` together with the LSE Ring requires. ``--ulysses-anything`` is
  rejected too, because the joint mask is rebuilt from equipartitioned text chunks.
  """

  def _apply(
    self,
    transformer: Optional[torch.nn.Module | PRXTransformer2DModel] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:
    assert transformer is not None, "Transformer must be provided."
    assert parallelism_config is not None, "ParallelismConfig must be provided."
    if parallelism_config.ring_size is not None and parallelism_config.ring_size > 1:
      raise NotImplementedError(
        "PRX / PRXPixel Context Parallelism does not support Ring attention: the pipeline "
        "always passes a text padding mask, and Ring needs `return_lse`, which the native "
        "backend rejects together with a mask. Use Ulysses (`--parallel ulysses`) instead.")
    if parallelism_config.ulysses_anything:
      raise NotImplementedError(
        "PRX / PRXPixel Context Parallelism does not support `--ulysses-anything`: the joint "
        "text+image key mask is rebuilt from equipartitioned text chunks. Use the default "
        "Ulysses partition.")
    assert isinstance(
      transformer,
      PRXTransformer2DModel), "Transformer must be an instance of PRXTransformer2DModel"

    _install_cp_processor_patches()

    _cp_plan: _ContextParallelModelPlan = {
      "txt_in": {
        0: _ContextParallelInput(split_dim=1, expected_dims=3, split_output=True),
      },
      "pe_embedder": {
        "ids": _ContextParallelInput(split_dim=1, expected_dims=3),
      },
      "blocks.0": {
        "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3),
      },
      "final_layer": _ContextParallelOutput(gather_dim=1, expected_dims=3),
    }
    return _cp_plan


@TensorParallelismPlannerRegister.register("PRXTransformer2DModel")
class PrxTensorParallelismPlanner(TensorParallelismPlanner):
  """TP planner for PRX / PRXPixel.

  ``img_qkv_proj`` packs ``[Q|K|V]`` and ``txt_kv_proj`` packs ``[K|V]``; the processor
  reshapes them as ``(3|2, B, heads, L, D)``. A plain ``ColwiseParallel()`` slices the fused
  output into contiguous blocks that cross Q/K/V and head boundaries, so the weights are
  sharded while the output is all-gathered back (``Replicate()``) and ``attn.heads`` stays
  complete -- hence ``shard_div_attr`` must NOT be called for this strategy. The output
  projection therefore needs ``RowwiseParallel(input_layouts=Replicate())``.

  The all-gathers make attention TP a **memory** optimization rather than a speed one: on two
  RTX PRO 5000 cards TP2 measured 18.8s vs 19.5s for a single card. Prefer Context Parallelism
  for latency, and use hybrid `ulysses_tp` only when a single card cannot hold the weights.
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
    transformer: PRXTransformer2DModel,
    tp_mesh: DeviceMesh,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    layer_plans: List[Dict[str, ParallelStyle]] = []
    for _, block in transformer.blocks.named_children():
      layer_plan: Dict[str, ParallelStyle] = {
        "attention.img_qkv_proj": ColwiseParallel(output_layouts=Replicate()),
        "attention.txt_kv_proj": ColwiseParallel(output_layouts=Replicate()),
        "attention.to_out.0": RowwiseParallel(input_layouts=Replicate()),
        "gate_proj": ColwiseParallel(),
        "up_proj": ColwiseParallel(),
        "down_proj": RowwiseParallel(),
        # `Modulation.lin` is chunked into 6 groups (attn/mlp shift-scale-gate), so the column
        # boundaries have to survive: shard the weight, replicate the output.
        "modulation.lin": ColwiseParallel(output_layouts=Replicate()),
      }
      parallelize_module(
        module=block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_plan,
      )
      layer_plans.append(layer_plan)

    final_layer_plan: Dict[str, ParallelStyle] = {
      "adaLN_modulation.1": ColwiseParallel(output_layouts=Replicate()),
    }
    parallelize_module(
      module=transformer.final_layer,
      device_mesh=tp_mesh,
      parallelize_plan=final_layer_plan,
    )
    layer_plans.append(final_layer_plan)
    return transformer, layer_plans
