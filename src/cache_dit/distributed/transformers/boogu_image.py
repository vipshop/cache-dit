"""Boogu-Image distributed parallelism planners (TP and CP)."""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed._tensor import Replicate
from torch.distributed.tensor.parallel import (
  ColwiseParallel,
  ParallelStyle,
  RowwiseParallel,
  parallelize_module,
)

from ...distributed.core import _ContextParallelModelPlan
from ...logger import init_logger
from ..config import ParallelismConfig
from ..utils import shard_div_attr  # noqa: F401
from .register import (
  ContextParallelismPlanner,
  ContextParallelismPlannerRegister,
  TensorParallelismPlanner,
  TensorParallelismPlannerRegister,
)

try:
  from boogu.models.attention_processor import (
    Attention,
    BooguImageAttnProcessor,
    BooguImageAttnProcessorFlash2Varlen,
  )
  from boogu.models.transformers import BooguImageTransformer2DModel
  BOOGU_IMAGE_AVAILABLE = True
except ImportError:
  # boogu-image not available
  BOOGU_IMAGE_AVAILABLE = False

logger = init_logger(__name__)


def _dtensor_safe_swiglu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  """Pure-PyTorch SwiGLU that is safe with DTensor sharded inputs.

  flash_attn's fused ``swiglu`` CUDA kernel does not understand DTensor
  shard placements and will crash with ``cudaErrorIllegalAddress`` when
  fed DTensor inputs from ColwiseParallel linear layers.

  This PyTorch-native implementation uses only ``F.silu`` and
  element-wise multiply, which are DTensor-compatible.
  """
  return torch.nn.functional.silu(x.float()).to(x.dtype) * y


def _patch_swiglu_for_tp(block: nn.Module) -> None:
  """Replace flash_attn fused swiglu with DTensor-safe PyTorch swiglu when TP is active.

  Only patches the FFN if it is currently using flash_attn's fused CUDA kernel. PyTorch-native
  swiglu is already DTensor-compatible and is left untouched.
  """
  for ffn_name in ("feed_forward", "img_feed_forward", "instruct_feed_forward"):
    ffn = getattr(block, ffn_name, None)
    if ffn is None:
      continue
    swiglu_fn = getattr(ffn, "swiglu", None)
    if swiglu_fn is None:
      continue
    # Only replace flash_attn fused kernel; PyTorch-native swiglu is safe.
    if hasattr(swiglu_fn, "__module__") and "flash_attn" in swiglu_fn.__module__:
      ffn.swiglu = _dtensor_safe_swiglu


def _patch_attention_processor_for_cp(transformer: nn.Module) -> None:
  """Monkey-patch attention processors to use ``_dispatch_attention_fn`` with UAA.

  Boogu-Image has ``num_kv_heads=7`` with GQA ratio 28:7 = 4:1.  UAA handles
  head-dimension padding internally, but after all-to-all the per-GPU GQA ratio
  becomes non-integer (14 Q heads / 4 KV heads = 3.5) because KV heads cannot
  be evenly divided.

  **Fix**: Do the GQA KV→Q repeat BEFORE calling ``_dispatch_attention_fn``,
  so that Q/K/V all have the same head count when UAA sees them.  After the
  repeat, Q/K/V have 28 heads each — divisible by 2, giving a clean 14:14 ratio
  per GPU.  We then pass ``enable_gqa=False`` since heads already match.
  """
  if not BOOGU_IMAGE_AVAILABLE:
    logger.warning("Boogu-Image not available; skipping attention processor patch.")
    return

  from ...attention import _dispatch_attention_fn

  # Patch the sdpa variant.
  _original_sdpa_call = BooguImageAttnProcessor.__call__

  def _patched_sdpa_call(
    self: BooguImageAttnProcessor,
    attn: Attention,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
    base_sequence_length: Optional[int] = None,
  ) -> torch.Tensor:
    cp_config = getattr(self, '_cp_config', None)
    if cp_config is None:
      return _original_sdpa_call(
        self,
        attn,
        hidden_states,
        encoder_hidden_states,
        attention_mask,
        image_rotary_emb,
        base_sequence_length,
      )

    # ---- Q/K/V projection + reshape (same as original) ----
    batch_size, sequence_length, _ = hidden_states.shape
    query = attn.to_q(hidden_states)  # type: torch.Tensor
    key = attn.to_k(encoder_hidden_states)  # type: torch.Tensor
    value = attn.to_v(encoder_hidden_states)  # type: torch.Tensor
    query_dim = query.shape[-1]
    inner_dim = key.shape[-1]
    head_dim = query_dim // attn.heads
    kv_heads = inner_dim // head_dim
    query = query.view(batch_size, -1, attn.heads, head_dim)
    key = key.view(batch_size, -1, kv_heads, head_dim)
    value = value.view(batch_size, -1, kv_heads, head_dim)
    if attn.norm_q is not None:
      query = attn.norm_q(query)
    if attn.norm_k is not None:
      key = attn.norm_k(key)

    # ---- RoPE (same as original) ----
    from boogu.models.embeddings import apply_rotary_emb as _boogu_apply_rotary_emb
    if image_rotary_emb is not None:
      query = _boogu_apply_rotary_emb(query, image_rotary_emb, use_real=False)
      key = _boogu_apply_rotary_emb(key, image_rotary_emb, use_real=False)

    # ---- GQA repeat BEFORE UAA ----
    # kv_heads=7, attn.heads=28.  UAA internally pads K/V heads from 7→8
    # but after all-to-all split the per-GPU ratio is 14:4 = non-integer.
    # We repeat K/V to match Q heads FIRST so all-to-all sees 28:28 (clean
    # 14:14 after split), then set enable_gqa=False.
    if kv_heads < attn.heads:
      repeat_factor = attn.heads // kv_heads
      key = key.repeat_interleave(repeat_factor, dim=2)
      value = value.repeat_interleave(repeat_factor, dim=2)

    # ---- Replace sdpa with UAA dispatch ----
    softmax_scale = attn.scale
    hidden_states = _dispatch_attention_fn(
      query,
      key,
      value,
      attn_mask=None,
      dropout_p=0.0,
      is_causal=False,
      scale=softmax_scale,
      enable_gqa=False,  # heads already 1:1 after manual repeat
      cp_config=cp_config,
    )
    hidden_states = hidden_states.flatten(-2)
    hidden_states = hidden_states.type_as(query)
    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)
    return hidden_states

  BooguImageAttnProcessor.__call__ = _patched_sdpa_call

  # Patch the flash-attn varlen variant similarly.
  if hasattr(BooguImageAttnProcessorFlash2Varlen, '__call__'):
    _original_varlen_call = BooguImageAttnProcessorFlash2Varlen.__call__

    def _patched_varlen_call(
      self: BooguImageAttnProcessorFlash2Varlen,
      attn: Attention,
      hidden_states: torch.Tensor,
      encoder_hidden_states: torch.Tensor,
      attention_mask: Optional[torch.Tensor] = None,
      image_rotary_emb: Optional[torch.Tensor] = None,
      base_sequence_length: Optional[int] = None,
    ) -> torch.Tensor:
      cp_config = getattr(self, '_cp_config', None)
      if cp_config is None:
        return _original_varlen_call(self, attn, hidden_states, encoder_hidden_states,
                                     attention_mask, image_rotary_emb, base_sequence_length)

      from boogu.models.embeddings import apply_rotary_emb as _boogu_apply_rotary_emb

      batch_size, sequence_length, _ = hidden_states.shape
      query = attn.to_q(hidden_states)  # type: torch.Tensor
      key = attn.to_k(encoder_hidden_states)  # type: torch.Tensor
      value = attn.to_v(encoder_hidden_states)  # type: torch.Tensor
      query_dim = query.shape[-1]
      inner_dim = key.shape[-1]
      head_dim = query_dim // attn.heads
      kv_heads = inner_dim // head_dim
      query = query.view(batch_size, -1, attn.heads, head_dim)
      key = key.view(batch_size, -1, kv_heads, head_dim)
      value = value.view(batch_size, -1, kv_heads, head_dim)
      if attn.norm_q is not None:
        query = attn.norm_q(query)
      if attn.norm_k is not None:
        key = attn.norm_k(key)
      if image_rotary_emb is not None:
        query = _boogu_apply_rotary_emb(query, image_rotary_emb, use_real=False)
        key = _boogu_apply_rotary_emb(key, image_rotary_emb, use_real=False)

      # NOTE: GQA repeat before UAA (same rationale as sdpa variant).
      if kv_heads < attn.heads:
        repeat_factor = attn.heads // kv_heads
        key = key.repeat_interleave(repeat_factor, dim=2)
        value = value.repeat_interleave(repeat_factor, dim=2)

      softmax_scale = attn.scale
      hidden_states = _dispatch_attention_fn(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=softmax_scale,
        enable_gqa=False,
        cp_config=cp_config,
      )
      hidden_states = hidden_states.flatten(-2)
      hidden_states = hidden_states.type_as(query)
      hidden_states = attn.to_out[0](hidden_states)
      hidden_states = attn.to_out[1](hidden_states)
      return hidden_states

    BooguImageAttnProcessorFlash2Varlen.__call__ = _patched_varlen_call


def _patch_transformer_forward_for_cp(transformer: BooguImageTransformer2DModel) -> None:
  """Monkey-patch ``transformer.forward`` for context parallelism.

  Boogu-Image does internal patching (flat_and_pad_to_seq) inside its forward,
  so top-level CP hooks cannot split the image-format input.

  This wrapper runs the full preprocessing and double-stream fusion on every
  GPU (those stages are cheap — only 2 double-stream layers), then splits the
  joint sequence via ``tensor_split`` before entering the single-stream loop.

  The attention processor monkey-patch (see ``_patch_attention_processor_for_cp``)
  handles the UAA all-to-all inside each attention layer, so each GPU computes
  attention over the FULL sequence for a SUBSET of heads.

  After single-stream, we ``all_gather`` the output and continue with
  ``norm_out`` / unpatchify.

  Use ``--ulysses-anything`` to enable UAA for uneven joint-sequence lengths.
  """
  _original_forward = transformer.forward

  def _cp_forward(
    self: BooguImageTransformer2DModel,
    hidden_states: Union[torch.Tensor, List[torch.Tensor]],
    timestep: torch.Tensor,
    instruction_hidden_states: torch.Tensor,
    freqs_cis: torch.Tensor,
    instruction_attention_mask: torch.Tensor,
    ref_image_hidden_states: Optional[List[List[torch.Tensor]]] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = False,
  ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    cp_config = getattr(self, '_cp_config', None)
    if cp_config is None:
      # No CP active — delegate to original forward.
      return _original_forward(
        hidden_states,
        timestep,
        instruction_hidden_states,
        freqs_cis,
        instruction_attention_mask,
        ref_image_hidden_states,
        attention_kwargs,
        return_dict,
      )

    mesh = cp_config._flattened_mesh
    tp_size = mesh.size()
    rank = dist.get_rank(mesh.get_group())

    # ---- Preprocessing (full sequence on every GPU) ----
    instruction_hidden_states = self.preprocess_instruction_hidden_states(
      instruction_hidden_states, self.instruction_feature_configs)
    temb, instruction_hidden_states = self.time_caption_embed(
      timestep,
      instruction_hidden_states,
      hidden_states[0].dtype,
    )

    batch_size = len(hidden_states)
    is_tensor = isinstance(hidden_states, torch.Tensor)
    if is_tensor:
      hidden_states = list(hidden_states)

    (
      hidden_states,
      ref_image_hidden_states,
      img_mask,
      ref_img_mask,
      l_effective_ref_img_len,
      l_effective_img_len,
      ref_img_sizes,
      img_sizes,
    ) = self.flat_and_pad_to_seq(hidden_states, ref_image_hidden_states)

    (
      context_rotary_emb,
      ref_img_rotary_emb,
      noise_rotary_emb,
      rotary_emb,
      encoder_seq_lengths,
      seq_lengths,
      combined_img_rotary_emb,
      combined_img_seq_lengths,
    ) = self.rope_embedder(freqs_cis, instruction_attention_mask, l_effective_ref_img_len,
                           l_effective_img_len, ref_img_sizes, img_sizes, hidden_states.device)

    for layer in self.context_refiner:
      instruction_hidden_states = layer(instruction_hidden_states, instruction_attention_mask,
                                        context_rotary_emb)

    combined_img_hidden_states = self.img_patch_embed_and_refine(
      hidden_states, ref_image_hidden_states, img_mask, ref_img_mask, noise_rotary_emb,
      ref_img_rotary_emb, l_effective_ref_img_len, l_effective_img_len, temb)

    # ---- Double-stream layers (full sequence, only 2 layers) ----
    # These layers are few and cheap — running them on the full sequence
    # avoids complicating the split/gather logic for joint cross-attention.
    instruct_hidden_states = instruction_hidden_states
    img_hidden_states = combined_img_hidden_states

    max_seq_len = max(seq_lengths)
    joint_attention_mask = hidden_states.new_zeros(batch_size, max_seq_len, dtype=torch.bool)
    for i, sl in enumerate(seq_lengths):
      joint_attention_mask[i, :sl] = True

    if self.num_double_stream_layers > 0:
      max_img_len = max(combined_img_seq_lengths)
      img_attention_mask = hidden_states.new_zeros(batch_size, max_img_len, dtype=torch.bool)
      for i, sl in enumerate(combined_img_seq_lengths):
        img_attention_mask[i, :sl] = True
      for layer in self.double_stream_layers:
        img_hidden_states, instruct_hidden_states = layer(img_hidden_states, instruct_hidden_states,
                                                          img_attention_mask, joint_attention_mask,
                                                          combined_img_rotary_emb, rotary_emb, temb,
                                                          encoder_seq_lengths, seq_lengths)

    # Fusion: concat instruction + image → joint_hidden_states [B, N, D].
    joint_hidden_states = torch.cat([instruct_hidden_states, img_hidden_states], dim=1)

    # ---- Split joint sequence for CP ----
    # tensor_split handles uneven sizes (UAA complement at the sequence level).
    local_h = joint_hidden_states.tensor_split(tp_size, dim=1)[rank].contiguous()
    local_mask = joint_attention_mask.tensor_split(tp_size, dim=1)[rank].contiguous()
    local_rope = rotary_emb.tensor_split(tp_size, dim=1)[rank].contiguous()

    # ---- Single-stream layers on local chunk ----
    h = local_h
    for layer in self.single_stream_layers:
      h = layer(h, local_mask, local_rope, temb)

    # ---- All-gather output (handles uneven sizes) ----
    chunks = [
      torch.empty_like(joint_hidden_states.tensor_split(tp_size, dim=1)[i]) for i in range(tp_size)
    ]
    chunks[rank] = h
    dist.all_gather(chunks, h, group=mesh.get_group())
    gathered_h = torch.cat(chunks, dim=1)

    # ---- Output projection (inline unpatchify) ----
    from einops import rearrange

    hidden_states = self.norm_out(gathered_h, temb)
    p = self.config.patch_size
    output = []
    for i, (img_size, img_len,
            seq_len) in enumerate(zip(img_sizes, l_effective_img_len, seq_lengths)):
      height, width = img_size
      img_tokens = hidden_states[i][seq_len - img_len:seq_len]
      img_output = rearrange(img_tokens,
                             "(h w) (p1 p2 c) -> c (h p1) (w p2)",
                             h=height // p,
                             w=width // p,
                             p1=p,
                             p2=p)
      output.append(img_output)

    if is_tensor:
      output = torch.stack(output, dim=0)
    if return_dict:
      return {"sample": output}
    return output

  transformer.forward = _cp_forward.__get__(transformer, type(transformer))


@ContextParallelismPlannerRegister.register("BooguImageTransformer")
class BooguImageContextParallelismPlanner(ContextParallelismPlanner):

  def _apply(
    self,
    transformer: Optional[torch.nn.Module] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:
    if not BOOGU_IMAGE_AVAILABLE:
      return {}

    # Boogu-Image receives image-format input with internal patching.
    # We patch both the forward (sequence split/gather) and the attention
    # processor (UAA all-to-all).  Use --ulysses-anything for UAA.
    if transformer is not None:
      _patch_attention_processor_for_cp(transformer)
      _patch_transformer_forward_for_cp(transformer)
    return {}


@TensorParallelismPlannerRegister.register("BooguImageTransformer")
class BooguImageTensorParallelismPlanner(TensorParallelismPlanner):

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

  @staticmethod
  def _single_stream_layer_plan(block: nn.Module, tp_mesh: DeviceMesh) -> Dict[str, ParallelStyle]:
    """TP plan for single-stream blocks (single_stream_layers, refiner blocks).

    Attention TP strategy for GQA models (``num_kv_heads=7`` indivisible by tp_size=2):

    * ``attn.to_q``: ColwiseParallel with ``output_layouts=Replicate()``.
      Each GPU computes half the Q features, then all-gather so the attention
      processor always sees a full (non-DTensor) Q tensor.  This avoids
      DTensor placement issues inside ``.view()`` / ``.transpose()``.
    * ``attn.to_k`` / ``attn.to_v``: **replicated** (no sharding).
    * ``attn.to_out.0``: RowwiseParallel with ``input_layouts=Replicate()``.
      Since the attention output is a regular (replicated) tensor, we tell
      RowwiseParallel to wrap it as Replicate, then redistribute to
      ``Shard(-1)`` for the partial output computation + all-reduce.

    ``attn.heads`` is NOT modified (stays at 28) because the attention
    processor sees the full Q dimension.
    """
    layer_plan: Dict[str, ParallelStyle] = {
      "attn.to_q": ColwiseParallel(output_layouts=Replicate()),
      "attn.to_out.0": RowwiseParallel(input_layouts=Replicate()),
      "feed_forward.linear_1": ColwiseParallel(),
      "feed_forward.linear_3": ColwiseParallel(),
      "feed_forward.linear_2": RowwiseParallel(),
    }
    if getattr(block.norm1, "linear", None) is not None:
      layer_plan["norm1.linear"] = ColwiseParallel(output_layouts=Replicate())
    return layer_plan

  @staticmethod
  def _double_stream_layer_plan(block: nn.Module, tp_mesh: DeviceMesh) -> Dict[str, ParallelStyle]:
    """TP plan for double-stream blocks.

    Two attention modules in each double-stream block:

    * ``img_instruct_attn``: joint cross-attention.  Q/K/V projections live
      in ``processor`` (``img_to_q``, ``instruct_to_q``, etc.).  Only the
      final ``to_out.0`` remains on the ``Attention`` module (``to_q`` /
      ``to_k`` / ``to_v`` are deleted at init).
    * ``img_self_attn``: standard self-attention within the image stream.

    Both use GQA (28:7), same strategy as single-stream: Q projections are
    ColwiseParallel with ``output_layouts=Replicate()``; K/V replicated;
    final out-proj uses RowwiseParallel with ``input_layouts=Replicate()``.
    Intermediate output projections (``img_out``, ``instruct_out``) are
    kept unsharded to avoid chained all-reduces.
    """
    layer_plan: Dict[str, ParallelStyle] = {
      # joint cross-attention: Q projections (in processor)
      "img_instruct_attn.processor.img_to_q": ColwiseParallel(output_layouts=Replicate()),
      "img_instruct_attn.processor.instruct_to_q": ColwiseParallel(output_layouts=Replicate()),
      # joint cross-attention: final output projection
      "img_instruct_attn.to_out.0": RowwiseParallel(input_layouts=Replicate()),
      # image self-attention
      "img_self_attn.to_q": ColwiseParallel(output_layouts=Replicate()),
      "img_self_attn.to_out.0": RowwiseParallel(input_layouts=Replicate()),
      # image FFN
      "img_feed_forward.linear_1": ColwiseParallel(),
      "img_feed_forward.linear_3": ColwiseParallel(),
      "img_feed_forward.linear_2": RowwiseParallel(),
      # instruction FFN
      "instruct_feed_forward.linear_1": ColwiseParallel(),
      "instruct_feed_forward.linear_3": ColwiseParallel(),
      "instruct_feed_forward.linear_2": RowwiseParallel(),
    }
    # image modulation
    for mod_name in ("img_norm1", "img_norm2", "img_norm3"):
      if getattr(getattr(block, mod_name, None), "linear", None) is not None:
        layer_plan[f"{mod_name}.linear"] = ColwiseParallel(output_layouts=Replicate())
    # instruction modulation
    for mod_name in ("instruct_norm1", "instruct_norm2"):
      if getattr(getattr(block, mod_name, None), "linear", None) is not None:
        layer_plan[f"{mod_name}.linear"] = ColwiseParallel(output_layouts=Replicate())
    return layer_plan

  def parallelize_transformer(
    self,
    transformer: nn.Module,
    tp_mesh: DeviceMesh,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    layer_plans: List[Dict[str, ParallelStyle]] = []

    # refiner blocks (same structure as single_stream)
    for block_list_name in ("noise_refiner", "ref_image_refiner", "context_refiner"):
      block_list = getattr(transformer, block_list_name, None)
      if block_list is None:
        continue
      for _, block in block_list.named_children():
        _patch_swiglu_for_tp(block)
        layer_plan = self._single_stream_layer_plan(block, tp_mesh)
        parallelize_module(module=block, device_mesh=tp_mesh, parallelize_plan=layer_plan)
        layer_plans.append(layer_plan)

    # double-stream layers (partial TP: img_self_attn + FFN + modulation)
    for _, block in transformer.double_stream_layers.named_children():
      _patch_swiglu_for_tp(block)
      layer_plan = self._double_stream_layer_plan(block, tp_mesh)
      parallelize_module(module=block, device_mesh=tp_mesh, parallelize_plan=layer_plan)
      layer_plans.append(layer_plan)

    # single-stream layers (full TP)
    for _, block in transformer.single_stream_layers.named_children():
      _patch_swiglu_for_tp(block)
      layer_plan = self._single_stream_layer_plan(block, tp_mesh)
      parallelize_module(module=block, device_mesh=tp_mesh, parallelize_plan=layer_plan)
      layer_plans.append(layer_plan)

    return transformer, layer_plans
