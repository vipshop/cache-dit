import os
from typing import Optional

import torch

from .register import (
  _AttnBackend,
  _AttnBackendRegistry,
  _context_parallel_attention,
  _ContextParallelConfig,
)

try:
  from ffpa_attn import CUDABackend, ffpa_attn_func
except Exception:
  CUDABackend = None
  ffpa_attn_func = None

# FFPA CUDA kernels cover D in [64, 256] only with this flag; without it
# ffpa_attn_func silently falls back to SDPA (e.g. FLUX head_dim=128).
os.environ.setdefault("FFPA_CUDA_ALLOW_SMALL_D", "1")

_SM120_MAJOR = 12
# CUDABackend instances are immutable config objects; cache to avoid the
# per-call dataclass __post_init__ auto-resolve overhead.
_ffpa_backend_cache: dict[tuple, "CUDABackend"] = {}
# Global override for the hybrid fp16 early-rows count, set from the CLI
# (--ffpa-hybrid-n-early). When set it forces the hybrid mode on, including
# non-causal attention (where the default keeps hybrid off).
_ffpa_hybrid_n_early_override: Optional[int] = None
# Global override for the fp4 Hadamard Q/K pre-rotation, set from the CLI
# (--ffpa-fp4-hadamard). Off by default: it trades a small perf overhead
# for lower fp4 quantization noise on outlier-heavy activations.
_ffpa_fp4_hadamard_override: bool = False
# Global override for the fp8 Hadamard Q/K pre-rotation, set from the CLI
# (--ffpa-fp8-hadamard). Off by default: off-path stays free of any
# dependency on the hadamard-enabled CUDABackend fields.
_ffpa_fp8_hadamard_override: bool = False
# Global override for the fp4 PV MMA dtype ("fp8" = MXFP8, higher PV
# precision than NVFP4), set from the CLI (--ffpa-fp4-pv-mm-type). None
# keeps the ffpa-attn default (fp4).
_ffpa_fp4_pv_mm_type_override: Optional[str] = None
# Global override for the fp4 V-column-mean smoothing, set from the CLI
# (--ffpa-fp4-smooth-v).
_ffpa_fp4_smooth_v_override: bool = False
# Global overrides forcing the hybrid fp16 early-rows stage OFF, set from
# the CLI (--ffpa-fp8-no-hybrid / --ffpa-fp4-no-hybrid). Trades precision
# (causal early rows lose the fp16 attention-sink path) for speed.
_ffpa_fp8_no_hybrid_override: bool = False
_ffpa_fp4_no_hybrid_override: bool = False


def set_ffpa_hybrid_n_early(n_early: Optional[int]) -> None:
  """Override the FFPA hybrid fp16 early-rows count.

  :param n_early: Positive multiple of 128, or ``None`` to restore defaults.
  """
  global _ffpa_hybrid_n_early_override
  if n_early is not None and (n_early <= 0 or n_early % 128 != 0):
    raise ValueError(f"ffpa hybrid n_early must be a positive multiple of 128, got {n_early}.")
  _ffpa_hybrid_n_early_override = n_early


def set_ffpa_fp4_hadamard(enabled: bool) -> None:
  """Toggle the FFPA fp4 Hadamard Q/K pre-rotation.

  :param enabled: True to enable (requires the fp4 attention backend).
  """
  global _ffpa_fp4_hadamard_override
  _ffpa_fp4_hadamard_override = bool(enabled)


def set_ffpa_fp8_hadamard(enabled: bool) -> None:
  """Toggle the FFPA fp8 Hadamard Q/K pre-rotation.

  :param enabled: True to enable (requires an fp8 attention backend).
  """
  global _ffpa_fp8_hadamard_override
  _ffpa_fp8_hadamard_override = bool(enabled)


def set_ffpa_fp4_pv_mm_type(pv_mm_type: Optional[str]) -> None:
  """Override the FFPA fp4 PV MMA dtype.

  :param pv_mm_type: ``"fp4"`` (NVFP4), ``"fp8"`` (MXFP8, head_dim <= 192), or ``None`` for the default.
  """
  global _ffpa_fp4_pv_mm_type_override
  if pv_mm_type not in (None, "fp4", "fp8"):
    raise ValueError(f"ffpa fp4 pv_mm_type must be 'fp4' or 'fp8', got {pv_mm_type}.")
  _ffpa_fp4_pv_mm_type_override = pv_mm_type


def set_ffpa_fp4_smooth_v(enabled: bool) -> None:
  """Toggle the FFPA fp4 V-column-mean smoothing.

  :param enabled: True to enable (requires the fp4 attention backend).
  """
  global _ffpa_fp4_smooth_v_override
  _ffpa_fp4_smooth_v_override = bool(enabled)


def set_ffpa_no_hybrid(enable_fp8: Optional[bool] = None,
                       enable_fp4: Optional[bool] = None) -> None:
  """Force the FFPA hybrid fp16 early-rows stage off.

  :param enable_fp8: Force fp8 hybrid off (requires an fp8 attention backend).
  :param enable_fp4: Force fp4 hybrid off (requires the fp4 attention backend).
  """
  global _ffpa_fp8_no_hybrid_override, _ffpa_fp4_no_hybrid_override
  if enable_fp8 is not None:
    _ffpa_fp8_no_hybrid_override = bool(enable_fp8)
  if enable_fp4 is not None:
    _ffpa_fp4_no_hybrid_override = bool(enable_fp4)


def _require_sm120_cuda(query: torch.Tensor) -> None:
  if query.device.type != "cuda":
    raise RuntimeError(
      f"FFPA attention backends require CUDA tensors, got device {query.device.type}.")
  major, minor = torch.cuda.get_device_capability(query.device)
  if major != _SM120_MAJOR:
    raise RuntimeError(f"FFPA attention backends only support sm_120 GPUs, "
                       f"got capability sm_{major}{minor} on device {query.device}.")


def _is_geforce_5090_or_5080(device: torch.device) -> bool:
  name = torch.cuda.get_device_name(device)
  return "5090" in name or "5080" in name


def _build_ffpa_cuda_backend(
  device: torch.device,
  *,
  enable_fp8: bool = False,
  enable_fp4: bool = False,
  is_causal: bool = False,
  fp8_preset: Optional[str] = None,
) -> "CUDABackend":
  if enable_fp8 and enable_fp4:
    raise ValueError("enable_fp8 and enable_fp4 are mutually exclusive.")
  is_geforce_50x0 = _is_geforce_5090_or_5080(device)
  cache_key = (device.index, enable_fp8, enable_fp4, is_geforce_50x0, is_causal, fp8_preset,
               _ffpa_hybrid_n_early_override, _ffpa_fp4_hadamard_override,
               _ffpa_fp8_hadamard_override, _ffpa_fp4_pv_mm_type_override,
               _ffpa_fp4_smooth_v_override, _ffpa_fp8_no_hybrid_override,
               _ffpa_fp4_no_hybrid_override)
  backend = _ffpa_backend_cache.get(cache_key)
  if backend is not None:
    return backend

  kwargs = {}
  # Higher precision config: int8 QK MMA + fp16 PV acc + per_thread Q/K + per_channel V
  # Both quant methods support every fp8 head_dim including non-32-multiple. Hybrid fp16
  # keeps the precision of the early Q rows (attention sink), the rows most sensitive to
  # fp8/fp4 quantization noise. The know limitations:
  # 1. The per-block quantization must use PV acc type f32 to avoid overflow for large N.
  # 2. The hybrid mode is required for fp8 for better precision, but not required for fp4.
  #    Since fp4 already uses fine-grained per-group(16) quantization for better precision.
  #    We still keep improving the precision of fp8 and fp4 in the future. This hybrid mode
  #    will be removed once we have better precision for fp8/fp4.
  # 3. Currently, the best precision config for fp8 attention is always recommended (with
  #    negilible performance overhead, ~3% for 16K sequence length).
  # NOTE: The hybrid mode will be removed once we have better precision for fp8.
  n_early = 128 if is_causal else 256
  force_hybrid = False
  if _ffpa_hybrid_n_early_override is not None:
    # Explicit CLI value forces the hybrid mode on, even for non-causal
    # attention and the no-hybrid presets.
    n_early = _ffpa_hybrid_n_early_override
    force_hybrid = True
  if enable_fp8 and fp8_preset == "per_block":
    # Performance-first config: per_block Q/K/V + f32 PV acc (f32 acc is required
    # by per-block quantization to avoid overflow for large N), no hybrid.
    kwargs.update(
      fp8_qk_mm_type="int8",
      fp8_pv_acc_type="f32",
      fp8_q_quant_method="per_block",
      fp8_k_quant_method="per_block",
      fp8_v_quant_method="per_block",
      fp8_smooth_k=True,
      fp8_smooth_v=False,
      fp8_hybrid=force_hybrid,
      fp8_hybrid_n_early=n_early,
    )
  elif enable_fp8:
    kwargs.update(
      # Use QK INT8 for better precision, same as SageAttention
      fp8_qk_mm_type="int8",
      fp8_pv_acc_type="f16",
      fp8_q_quant_method="per_thread",
      fp8_k_quant_method="per_thread",
      fp8_v_quant_method="per_channel",
      fp8_smooth_k=True,
      fp8_smooth_v=True,
      # Hybrid keeps fp16 precision on the early Q rows (attention sink);
      # --ffpa-fp8-no-hybrid forces it off for speed at a precision cost.
      fp8_hybrid=((fp8_preset != "no_hybrid") or force_hybrid) and not _ffpa_fp8_no_hybrid_override,
      fp8_hybrid_n_early=n_early,
    )
  elif enable_fp4:
    kwargs.update(
      # FP4 alreay use fine-grained per-group(16) quantization for better precision,
      # the same as SageAttention3. So we don't need to use hybrid mode for FP4 for
      # non-causal attention senarios. But for causal attention, we still use hybrid
      # mode to keep the precision of the early Q rows (attention sink).
      # NOTE: This hybrid mode will be removed once we have better precision for fp4.
      fp4_hybrid=(is_causal or force_hybrid) and not _ffpa_fp4_no_hybrid_override,
      fp4_hybrid_n_early=n_early,
      fp4_hadamard=_ffpa_fp4_hadamard_override,
    )
  if enable_fp8 and _ffpa_fp8_hadamard_override:
    # Injected after both fp8 presets; off-path stays free of the kwarg so
    # older ffpa-attn builds (without the field) keep working.
    kwargs["fp8_hadamard"] = True
  if enable_fp4 and _ffpa_fp4_pv_mm_type_override is not None:
    kwargs["fp4_pv_mm_type"] = _ffpa_fp4_pv_mm_type_override
  if enable_fp4 and _ffpa_fp4_smooth_v_override:
    kwargs["fp4_smooth_v"] = True
  backend = CUDABackend(
    backward=False,
    enable_fp8=enable_fp8,
    enable_fp4=enable_fp4,
    **kwargs,
  )
  _ffpa_backend_cache[cache_key] = backend
  return backend


def _ffpa_attn_core(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  is_causal: bool,
  scale: Optional[float],
  enable_gqa: bool,
  backend: "CUDABackend",
  nhd_native: bool = False,
) -> torch.Tensor:
  # diffusers NHD [B, N, H, D] -> FFPA BHND [B, H, N, D]. The fp8 CUDA path
  # reads NHD gmem natively (Phase C), so a zero-copy permute view suffices;
  # fp16/fp4 kernels still require a BHND-packed copy (non-contiguous Q
  # silently corrupts them). Any non-packed input falls back to the copy.
  if nhd_native and query.is_contiguous() and key.is_contiguous() and value.is_contiguous():
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
  else:
    q = query.permute(0, 2, 1, 3).contiguous()
    k = key.permute(0, 2, 1, 3).contiguous()
    v = value.permute(0, 2, 1, 3).contiguous()
  out = ffpa_attn_func(
    q,
    k,
    v,
    is_causal=is_causal,
    scale=scale,
    enable_gqa=enable_gqa,
    forward_backend=backend,
  )
  return out.permute(0, 2, 1, 3)


def _make_ffpa_forward_op(backend: "CUDABackend", nhd_native: bool = False):

  def _ffpa_attention_forward_op(
    ctx: torch.autograd.function.FunctionCtx,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    _save_ctx: bool = True,
    _cp_config: Optional["_ContextParallelConfig"] = None,
  ):
    # attn_mask / dropout_p / return_lse are rejected in _ffpa_attention_impl
    # before the CP template runs; enable_gqa / is_causal are natively
    # supported by FFPA and forwarded as-is.
    return _ffpa_attn_core(query, key, value, is_causal, scale, enable_gqa, backend, nhd_native)

  return _ffpa_attention_forward_op


def _ffpa_attention_backward_op(
  ctx: torch.autograd.function.FunctionCtx,
  grad_out: torch.Tensor,
  *args,
):
  raise NotImplementedError("Backward pass is not implemented for FFPA attention.")


def _ffpa_attention_impl(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: Optional[torch.Tensor] = None,
  dropout_p: float = 0.0,
  is_causal: bool = False,
  scale: Optional[float] = None,
  enable_gqa: bool = False,
  return_lse: bool = False,
  cp_gqa_strategy: Optional[str] = None,
  _cp_config: Optional["_ContextParallelConfig"] = None,
  *,
  enable_fp8: bool = False,
  enable_fp4: bool = False,
  fp8_preset: Optional[str] = None,
) -> torch.Tensor:
  if attn_mask is not None:
    raise ValueError("`attn_mask` is not yet supported for FFPA attention.")
  if dropout_p > 0.0:
    raise ValueError("`dropout_p` is not yet supported for FFPA attention.")
  if return_lse:
    raise ValueError("`return_lse` is not supported for FFPA attention.")
  if ffpa_attn_func is None:
    raise RuntimeError(
      "FFPA attention backend is not available. Please install `ffpa-attn` to use it.")
  _require_sm120_cuda(query)

  backend = _build_ffpa_cuda_backend(query.device,
                                     enable_fp8=enable_fp8,
                                     enable_fp4=enable_fp4,
                                     is_causal=is_causal,
                                     fp8_preset=fp8_preset)
  if _cp_config is None:
    return _ffpa_attn_core(query,
                           key,
                           value,
                           is_causal,
                           scale,
                           enable_gqa,
                           backend,
                           nhd_native=enable_fp8)
  return _context_parallel_attention(
    query,
    key,
    value,
    None,
    0.0,
    is_causal,
    scale,
    enable_gqa,
    return_lse,
    cp_gqa_strategy,
    forward_op=_make_ffpa_forward_op(backend, enable_fp8),
    backward_op=_ffpa_attention_backward_op,
    _cp_config=_cp_config,
  )


@_AttnBackendRegistry.register(
  _AttnBackend.FFPA,
  constraints=[],
  supports_context_parallel=True,
)
def _ffpa_attention(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: Optional[torch.Tensor] = None,
  dropout_p: float = 0.0,
  is_causal: bool = False,
  scale: Optional[float] = None,
  enable_gqa: bool = False,
  return_lse: bool = False,
  cp_gqa_strategy: Optional[str] = None,
  _cp_config: Optional["_ContextParallelConfig"] = None,
) -> torch.Tensor:
  """FFPA CUDA backend (CUTE_TMA, fp16/bf16, forward-only, sm_120 only).

  :param query: ``[B, N, H, D]`` (diffusers / NHD convention).
  :param key: ``[B, N_kv, H, D]``.
  :param value: ``[B, N_kv, H, D]``.
  :param attn_mask: Not supported.
  :param dropout_p: Not supported.
  :param is_causal: Causal masking (FFPA tail-aligned semantics).
  :param scale: Pre-softmax scaling; defaults to ``1/sqrt(D)``.
  :param enable_gqa: GQA/MQA support (natively supported by FFPA).
  :param return_lse: Not supported.
  :returns: ``[B, N, H, D]`` attention output.
  """
  return _ffpa_attention_impl(
    query,
    key,
    value,
    attn_mask,
    dropout_p,
    is_causal,
    scale,
    enable_gqa,
    return_lse,
    cp_gqa_strategy,
    _cp_config,
  )


@_AttnBackendRegistry.register(
  _AttnBackend.FFPA_FP8,
  constraints=[],
  supports_context_parallel=True,
)
def _ffpa_fp8_attention(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: Optional[torch.Tensor] = None,
  dropout_p: float = 0.0,
  is_causal: bool = False,
  scale: Optional[float] = None,
  enable_gqa: bool = False,
  return_lse: bool = False,
  cp_gqa_strategy: Optional[str] = None,
  _cp_config: Optional["_ContextParallelConfig"] = None,
) -> torch.Tensor:
  """FFPA CUDA FP8 backend (CUTE_TMA_FP8, forward-only, sm_120 only).

  Inputs stay fp16/bf16; Q/K/V are fp8-quantized inside the kernel using the
  highest-precision config: int8 QK MMA, fp16 PV accumulation, Q/K quantized
  per_thread and V per_channel (supported for every fp8 head_dim including
  non-32-multiple D such as 120).

  :param query: ``[B, N, H, D]`` (diffusers / NHD convention).
  :param key: ``[B, N_kv, H, D]``.
  :param value: ``[B, N_kv, H, D]``.
  :returns: ``[B, N, H, D]`` attention output.
  """
  return _ffpa_attention_impl(
    query,
    key,
    value,
    attn_mask,
    dropout_p,
    is_causal,
    scale,
    enable_gqa,
    return_lse,
    cp_gqa_strategy,
    _cp_config,
    enable_fp8=True,
  )


@_AttnBackendRegistry.register(
  _AttnBackend.FFPA_FP4,
  constraints=[],
  supports_context_parallel=True,
)
def _ffpa_fp4_attention(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: Optional[torch.Tensor] = None,
  dropout_p: float = 0.0,
  is_causal: bool = False,
  scale: Optional[float] = None,
  enable_gqa: bool = False,
  return_lse: bool = False,
  cp_gqa_strategy: Optional[str] = None,
  _cp_config: Optional["_ContextParallelConfig"] = None,
) -> torch.Tensor:
  """FFPA CUDA NVFP4 backend (CUTE_TMA_FP4, forward-only, sm_120 only).

  Inputs stay fp16/bf16; Q/K/V are NVFP4-quantized inside the kernel.

  :param query: ``[B, N, H, D]`` (diffusers / NHD convention).
  :param key: ``[B, N_kv, H, D]``.
  :param value: ``[B, N_kv, H, D]``.
  :returns: ``[B, N, H, D]`` attention output.
  """
  return _ffpa_attention_impl(
    query,
    key,
    value,
    attn_mask,
    dropout_p,
    is_causal,
    scale,
    enable_gqa,
    return_lse,
    cp_gqa_strategy,
    _cp_config,
    enable_fp4=True,
  )


@_AttnBackendRegistry.register(
  _AttnBackend.FFPA_FP8_PER_BLOCK,
  constraints=[],
  supports_context_parallel=True,
)
def _ffpa_fp8_per_block_attention(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: Optional[torch.Tensor] = None,
  dropout_p: float = 0.0,
  is_causal: bool = False,
  scale: Optional[float] = None,
  enable_gqa: bool = False,
  return_lse: bool = False,
  cp_gqa_strategy: Optional[str] = None,
  _cp_config: Optional["_ContextParallelConfig"] = None,
) -> torch.Tensor:
  """FFPA CUDA FP8 per-block backend (CUTE_TMA_FP8, forward-only, sm_120 only).

  Performance-first config: int8 QK MMA, f32 PV accumulation, Q/K/V all
  quantized per_block, smooth K only, and no hybrid fp16 early rows. Fastest
  FFPA fp8 backend but the lowest precision.

  :param query: ``[B, N, H, D]`` (diffusers / NHD convention).
  :param key: ``[B, N_kv, H, D]``.
  :param value: ``[B, N_kv, H, D]``.
  :returns: ``[B, N, H, D]`` attention output.
  """
  return _ffpa_attention_impl(
    query,
    key,
    value,
    attn_mask,
    dropout_p,
    is_causal,
    scale,
    enable_gqa,
    return_lse,
    cp_gqa_strategy,
    _cp_config,
    enable_fp8=True,
    fp8_preset="per_block",
  )


@_AttnBackendRegistry.register(
  _AttnBackend.FFPA_FP8_NO_HYBRID,
  constraints=[],
  supports_context_parallel=True,
)
def _ffpa_fp8_no_hybrid_attention(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: Optional[torch.Tensor] = None,
  dropout_p: float = 0.0,
  is_causal: bool = False,
  scale: Optional[float] = None,
  enable_gqa: bool = False,
  return_lse: bool = False,
  cp_gqa_strategy: Optional[str] = None,
  _cp_config: Optional["_ContextParallelConfig"] = None,
) -> torch.Tensor:
  """FFPA CUDA FP8 backend without hybrid mode (forward-only, sm_120 only).

  Same quantization config as ``ffpa_fp8`` (int8 QK MMA, fp16 PV
  accumulation, Q/K per_thread, V per_channel, smooth K and V) but with the
  hybrid fp16 early-rows path forcibly disabled: slightly faster, slightly
  lower precision than ``ffpa_fp8``.

  :param query: ``[B, N, H, D]`` (diffusers / NHD convention).
  :param key: ``[B, N_kv, H, D]``.
  :param value: ``[B, N_kv, H, D]``.
  :returns: ``[B, N, H, D]`` attention output.
  """
  return _ffpa_attention_impl(
    query,
    key,
    value,
    attn_mask,
    dropout_p,
    is_causal,
    scale,
    enable_gqa,
    return_lse,
    cp_gqa_strategy,
    _cp_config,
    enable_fp8=True,
    fp8_preset="no_hybrid",
  )
