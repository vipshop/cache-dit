"""Boogu-Image distributed parallelism planners (TP and CP)."""

import functools
import math
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
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
  BOOGU_IMAGE_AVAILABLE = True
except ImportError:
  # boogu-image not available
  BOOGU_IMAGE_AVAILABLE = False

logger = init_logger(__name__)


def _boogu_attention_scale(
  attn: "Attention",
  sequence_length: int,
  base_sequence_length: Optional[int],
) -> float:
  if base_sequence_length is None:
    return attn.scale
  return math.sqrt(math.log(sequence_length, base_sequence_length)) * attn.scale


def _boogu_cp_gqa_dispatch(
  kv_heads: int,
  q_heads: int,
) -> tuple[Optional[str], bool]:
  if kv_heads >= q_heads:
    return None, False

  from ...attention import _AttnBackendRegistry

  active_backend, _ = _AttnBackendRegistry.get_active_backend()
  if active_backend.value == "flash_varlen":
    return "group_aligned_flash_varlen", True
  return "replicate_kv_sequence", False


def _dtensor_safe_swiglu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
  """Pure-PyTorch SwiGLU that is safe with DTensor sharded inputs.

  flash_attn's fused ``swiglu`` CUDA kernel does not understand DTensor
  shard placements and will crash with ``cudaErrorIllegalAddress`` when
  fed DTensor inputs from ColwiseParallel linear layers.

  This PyTorch-native implementation uses only ``F.silu`` and
  element-wise multiply, which are DTensor-compatible.
  """
  return torch.nn.functional.silu(x.float()).to(x.dtype) * y


def _patch_dtensor_unsafe_modules(block: nn.Module) -> None:
  """Replace flash_attn/triton fused kernels with DTensor-safe PyTorch versions when TP is active.

  Patches two categories:
  1. ``LuminaFeedForward.swiglu`` — flash_attn fused SwiGLU CUDA kernel.
  2. ``RMSNorm`` — triton fused layer norm (``boogu.ops.triton.layer_norm.RMSNorm``).
  """
  # 1. SwiGLU: replace flash_attn fused kernel with DTensor-safe PyTorch version.
  for ffn_name in ("feed_forward", "img_feed_forward", "instruct_feed_forward"):
    ffn = getattr(block, ffn_name, None)
    if ffn is None:
      continue
    swiglu_fn = getattr(ffn, "swiglu", None)
    if swiglu_fn is None:
      continue
    # flash_attn swiglu is a bound method (Function.apply); its __self__ is the
    # Function subclass whose __module__ starts with "flash_attn".
    fn_cls = getattr(swiglu_fn, "__self__", None)
    if fn_cls is not None and getattr(fn_cls, "__module__", "").startswith("flash_attn"):
      ffn.swiglu = _dtensor_safe_swiglu
      logger.debug("[TP] Patched %s.swiglu: flash_attn → DTensor-safe PyTorch.", ffn_name)

  # 2. RMSNorm: replace triton fused RMSNorm with torch.nn.RMSNorm.
  _triton_rmsnorm_cls = None
  try:
    from boogu.ops.triton.layer_norm import RMSNorm as TritonRMSNorm
    _triton_rmsnorm_cls = TritonRMSNorm
  except ImportError:
    pass

  if _triton_rmsnorm_cls is not None:
    for name, module in list(block.named_modules()):
      if isinstance(module, _triton_rmsnorm_cls):
        weight = module.weight
        new_norm = torch.nn.RMSNorm(
          weight.shape[0],
          eps=getattr(module, "eps", 1e-5),
          elementwise_affine=weight is not None,
          device=weight.device,
          dtype=weight.dtype,
        )
        # Copy the learned weight (triton RMSNorm always has affine weight).
        if weight is not None:
          new_norm.weight.data.copy_(weight.data)
        # Navigate to parent module via dotted path
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
          parent_path, leaf = parts
          parent = block.get_submodule(parent_path)
        else:
          parent, leaf = block, name
        setattr(parent, leaf, new_norm)
        logger.debug("[TP] Patched %s.RMSNorm: triton → torch.nn.RMSNorm.", name)


def _patch_attention_processor_for_cp(transformer: nn.Module) -> None:
  """Monkey-patch attention processors to use ``_dispatch_attention_fn`` with UAA.

  Boogu-Image has ``num_kv_heads=7`` with GQA ratio 28:7 = 4:1.  The CP path
  keeps K/V at 7 heads before communication and asks Ulysses to replicate K/V by
  sequence all-gather.  K/V are expanded to local Q heads only immediately before
  the backend attention call, so the backend still uses the MHA fast path with
  ``enable_gqa=False`` while avoiding pre-all-to-all K/V head expansion.
  """
  if not BOOGU_IMAGE_AVAILABLE:
    logger.warning("Boogu-Image not available; skipping attention processor patch.")
    return

  from ...attention import _dispatch_attention_fn

  for module in transformer.modules():
    processor = getattr(module, "processor", None)
    if processor is not None:
      processor._boogu_cp_eligible = False

  for layer in getattr(transformer, "single_stream_layers", []):
    processor = getattr(getattr(layer, "attn", None), "processor", None)
    if processor is not None:
      processor._boogu_cp_eligible = True

  if getattr(BooguImageAttnProcessor, "_cache_dit_boogu_cp_patched", False):
    return

  # Patch the sdpa variant.
  _original_sdpa_call = BooguImageAttnProcessor.__call__

  @functools.wraps(_original_sdpa_call)
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
    if cp_config is None or not getattr(self, "_boogu_cp_eligible", False):
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

    cp_gqa_strategy, enable_gqa = _boogu_cp_gqa_dispatch(kv_heads, attn.heads)
    dispatch_attn_mask = attention_mask if cp_gqa_strategy == "group_aligned_flash_varlen" else None

    # ---- Replace sdpa with UAA dispatch ----
    softmax_scale = _boogu_attention_scale(attn, sequence_length, base_sequence_length)
    hidden_states = _dispatch_attention_fn(
      query,
      key,
      value,
      attn_mask=dispatch_attn_mask,
      dropout_p=0.0,
      is_causal=False,
      scale=softmax_scale,
      enable_gqa=enable_gqa,
      cp_gqa_strategy=cp_gqa_strategy,
      cp_config=cp_config,
    )
    hidden_states = hidden_states.flatten(-2)
    hidden_states = hidden_states.type_as(query)
    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)
    return hidden_states

  BooguImageAttnProcessor.__call__ = _patched_sdpa_call
  BooguImageAttnProcessor._cache_dit_boogu_cp_patched = True

  # Patch the flash-attn varlen variant similarly.
  if hasattr(BooguImageAttnProcessorFlash2Varlen, '__call__'):
    _original_varlen_call = BooguImageAttnProcessorFlash2Varlen.__call__

    @functools.wraps(_original_varlen_call)
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
      if cp_config is None or not getattr(self, "_boogu_cp_eligible", False):
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

      cp_gqa_strategy, enable_gqa = _boogu_cp_gqa_dispatch(kv_heads, attn.heads)
      dispatch_attn_mask = attention_mask if cp_gqa_strategy == "group_aligned_flash_varlen" else None

      softmax_scale = _boogu_attention_scale(attn, sequence_length, base_sequence_length)
      hidden_states = _dispatch_attention_fn(
        query,
        key,
        value,
        attn_mask=dispatch_attn_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=softmax_scale,
        enable_gqa=enable_gqa,
        cp_gqa_strategy=cp_gqa_strategy,
        cp_config=cp_config,
      )
      hidden_states = hidden_states.flatten(-2)
      hidden_states = hidden_states.type_as(query)
      hidden_states = attn.to_out[0](hidden_states)
      hidden_states = attn.to_out[1](hidden_states)
      return hidden_states

    BooguImageAttnProcessorFlash2Varlen.__call__ = _patched_varlen_call
    BooguImageAttnProcessorFlash2Varlen._cache_dit_boogu_cp_patched = True


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

    if transformer is None:
      return {}

    _patch_attention_processor_for_cp(transformer)
    last_layer_index = len(transformer.single_stream_layers) - 1
    if last_layer_index < 0:
      return {}

    return {
      "single_stream_layers.0": {
        "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3),
      },
      "single_stream_layers.*": {
        "attention_mask": _ContextParallelInput(split_dim=1, expected_dims=2),
        "image_rotary_emb": _ContextParallelInput(split_dim=1, expected_dims=3),
      },
      f"single_stream_layers.{last_layer_index}":
      _ContextParallelOutput(
        gather_dim=1,
        expected_dims=3,
      ),
    }


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

    Only FFN and modulation layers are sharded.  Attention Q/K/V/out are
    kept **replicated** (not parallelized) because:

    1. ``num_kv_heads=7`` is not divisible by ``tp_size=2``.
    2. Even with the ``Replicate`` workaround, the all-gather cost on a
       long joint sequence (~262K tokens × 3360 dim × 2 bytes = 1.76 GB
       per layer) overwhelms any compute savings — FFN-only TP is
       empirically faster (108.9s vs 115.7s inference) and more accurate
       (PSNR 50.2 vs 49.9).
    """
    layer_plan: Dict[str, ParallelStyle] = {
      # "attn.to_q": ColwiseParallel(output_layouts=Replicate()),
      # "attn.to_out.0": RowwiseParallel(input_layouts=Replicate()),
      "feed_forward.linear_1": ColwiseParallel(),
      "feed_forward.linear_3": ColwiseParallel(),
      "feed_forward.linear_2": RowwiseParallel(),
    }
    if getattr(block.norm1, "linear", None) is not None:
      layer_plan["norm1.linear"] = ColwiseParallel(output_layouts=Replicate())
    return layer_plan

  @staticmethod
  def _double_stream_layer_plan(block: nn.Module, tp_mesh: DeviceMesh) -> Dict[str, ParallelStyle]:
    """TP plan for double-stream blocks (FFN + modulation only, no attention sharding).

    Attention Q/K/V/out projections are NOT sharded — see
    ``_single_stream_layer_plan`` docstring for rationale.
    """
    layer_plan: Dict[str, ParallelStyle] = {
      # joint cross-attention: Q projections (in processor)
      # NOTE: attention Q/out TP removed — all-gather on Q dominates communication
      # and makes TP slower than single GPU.  Only FFN + modulation are sharded.
      # "img_instruct_attn.processor.img_to_q": ColwiseParallel(output_layouts=Replicate()),
      # "img_instruct_attn.processor.instruct_to_q": ColwiseParallel(output_layouts=Replicate()),
      # "img_instruct_attn.to_out.0": RowwiseParallel(input_layouts=Replicate()),
      # "img_self_attn.to_q": ColwiseParallel(output_layouts=Replicate()),
      # "img_self_attn.to_out.0": RowwiseParallel(input_layouts=Replicate()),
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
        _patch_dtensor_unsafe_modules(block)
        layer_plan = self._single_stream_layer_plan(block, tp_mesh)
        parallelize_module(module=block, device_mesh=tp_mesh, parallelize_plan=layer_plan)
        layer_plans.append(layer_plan)

    # double-stream layers (partial TP: img_self_attn + FFN + modulation)
    for _, block in transformer.double_stream_layers.named_children():
      _patch_dtensor_unsafe_modules(block)
      layer_plan = self._double_stream_layer_plan(block, tp_mesh)
      parallelize_module(module=block, device_mesh=tp_mesh, parallelize_plan=layer_plan)
      layer_plans.append(layer_plan)

    # single-stream layers (full TP)
    for _, block in transformer.single_stream_layers.named_children():
      _patch_dtensor_unsafe_modules(block)
      layer_plan = self._single_stream_layer_plan(block, tp_mesh)
      parallelize_module(module=block, device_mesh=tp_mesh, parallelize_plan=layer_plan)
      layer_plans.append(layer_plan)

    return transformer, layer_plans
