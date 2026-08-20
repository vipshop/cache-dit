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
) -> "CUDABackend":
  if enable_fp8 and enable_fp4:
    raise ValueError("enable_fp8 and enable_fp4 are mutually exclusive.")
  is_geforce_50x0 = _is_geforce_5090_or_5080(device)
  cache_key = (device.index, enable_fp8, enable_fp4, is_geforce_50x0, is_causal)
  backend = _ffpa_backend_cache.get(cache_key)
  if backend is not None:
    return backend

  kwargs = {}
  # RTX 5090/5080 use the fastest fp8 config: int8 QK MMA + fp16 PV acc.
  # Otherwise, use the default fp8 attn config: fp8 QK MMA + fp32 PV acc.
  # Both quant methods support every fp8 head_dim including non-32-multiple
  # Hybrid fp16 keeps the precision of the early Q rows (attention sink),
  # the rows most sensitive to fp8/fp4 quantization noise.
  n_early = 128 if is_causal else 256  # causal Q rows are less sensitive to quant noise
  if enable_fp8:
    kwargs.update(
      fp8_qk_mm_type="int8" if is_geforce_50x0 else "fp8",
      fp8_pv_acc_type="f16" if is_geforce_50x0 else "f32",
      fp8_q_quant_method="per_thread",
      fp8_k_quant_method="per_thread",
      fp8_v_quant_method="per_channel",
      fp8_smooth_k=True,
      fp8_smooth_v=True if is_geforce_50x0 else False,
      fp8_hybrid=True,
      fp8_hybrid_n_early=n_early,
    )
  elif enable_fp4:
    kwargs.update(
      fp4_hybrid=True,
      fp4_hybrid_n_early=n_early,
    )
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
) -> torch.Tensor:
  # diffusers NHD [B, N, H, D] -> FFPA BHND [B, H, N, D]. contiguous() is
  # mandatory: non-contiguous Q silently corrupts the fp8/fp4 kernels.
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


def _make_ffpa_forward_op(backend: "CUDABackend"):

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
    return _ffpa_attn_core(query, key, value, is_causal, scale, enable_gqa, backend)

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
                                     is_causal=is_causal)
  if _cp_config is None:
    return _ffpa_attn_core(query, key, value, is_causal, scale, enable_gqa, backend)
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
    forward_op=_make_ffpa_forward_op(backend),
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
