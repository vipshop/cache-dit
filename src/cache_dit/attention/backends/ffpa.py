import os
from typing import Optional

import torch

from ...logger import init_logger
from .register import (
  _AttnBackend,
  _AttnBackendRegistry,
  _context_parallel_attention,
  _ContextParallelConfig,
)

logger = init_logger(__name__)

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
# (--ffpa-hybrid-n-early). Only meaningful when hybrid is enabled.
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
# Global override for the fp8 V-column-mean smoothing, set from the CLI
# (--ffpa-fp8-smooth-v). Off by default: smooth_v can mask per-channel V
# regressions, so tests run the unsmoothed per-channel path.
_ffpa_fp8_smooth_v_override: bool = False
# Global override toggling the FFPA hybrid fp16 early-rows stage ON, set
# from the CLI (--ffpa-hybrid). Off by default for every backend, including
# causal attention; hybrid is opt-in only.
_ffpa_hybrid_override: bool = False


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


def set_ffpa_fp8_smooth_v(enabled: bool) -> None:
  """Toggle the FFPA fp8 V-column-mean smoothing.

  :param enabled: True to enable (requires an fp8 attention backend).
  """
  global _ffpa_fp8_smooth_v_override
  _ffpa_fp8_smooth_v_override = bool(enabled)


def set_ffpa_hybrid(enabled: bool) -> None:
  """Toggle the FFPA hybrid fp16 early-rows stage.

  :param enabled: True to enable the hybrid stage (requires an ffpa backend).
  """
  global _ffpa_hybrid_override
  _ffpa_hybrid_override = bool(enabled)


def _require_supported_cuda(query: torch.Tensor, *, enable_fp4: bool = False) -> tuple[int, int]:
  """Validate the FFPA device requirements and return the capability.

  Supported architectures mirror the ffpa-attn router:
  - fp16/bf16 (``ffpa``): sm_120 (CUTE_TMA) and sm_89 (Ada; the TMA /
    CUTE_TMA hints are sm_120-only, so Ada auto-resolves to the cp.async
    NATIVE family).
  - fp8 (``ffpa_fp8`` / ``ffpa_fp8_per_block``): sm_120 (CUTE_TMA_FP8) and
    sm_89 (TMA-free CUTE_FP8_SM_89).
  - fp4 (``ffpa_fp4``): sm_120 only (CUTE_TMA_FP4_SM_120).

  :param query: Query tensor; supplies the device to probe.
  :param enable_fp4: Whether the fp4 family is selected.
  :returns: ``(major, minor)`` device capability.
  """
  if query.device.type != "cuda":
    raise RuntimeError(
      f"FFPA attention backends require CUDA tensors, got device {query.device.type}.")
  major, minor = torch.cuda.get_device_capability(query.device)
  if enable_fp4:
    if major != _SM120_MAJOR:
      raise RuntimeError(f"FFPA fp4 attention backend only supports sm_120 GPUs, "
                         f"got capability sm_{major}{minor} on device {query.device}.")
    return major, minor
  if major != _SM120_MAJOR and not (major == 8 and minor >= 9):
    raise RuntimeError(f"FFPA attention backends support sm_120 or sm_89 (Ada) "
                       f"only; sm_90 / sm_100 are not supported. Got capability "
                       f"sm_{major}{minor} on device {query.device}.")
  return major, minor


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
               _ffpa_fp4_smooth_v_override, _ffpa_fp8_smooth_v_override, _ffpa_hybrid_override)
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
  if _ffpa_hybrid_n_early_override is not None:
    n_early = _ffpa_hybrid_n_early_override
  # Hybrid is opt-in only (--ffpa-hybrid); off by default for every backend,
  # including causal attention.
  force_hybrid = _ffpa_hybrid_override
  if enable_fp8 and fp8_preset == "per_block":
    # Performance-first config: per_block Q/K + f32 PV acc (f32 acc is required
    # by per-block quantization to avoid overflow for large N), no hybrid.
    # V uses per_channel: per_block V (one amax over a 128-row block)
    # collapses on outlier-heavy rows (FLUX text tokens), producing a
    # catastrophically wrong image at large N (PSNR ~12 at 2048). per_channel V
    # (amax per D column over all rows) keeps those rows intact. smooth_v is
    # decoupled (SageAttention2 treats it as an independent add-on) and off by
    # default: per_channel alone matches the ffpa-attn default and the
    # math-domain reference; enable via --ffpa-fp8-smooth-v.
    kwargs.update(
      # Uses QK INT8 for better precision, same as SageAttention2.
      fp8_qk_mm_type="int8",
      fp8_pv_acc_type="f32",
      fp8_q_quant_method="per_block",
      fp8_k_quant_method="per_block",
      fp8_v_quant_method="per_channel",
      fp8_smooth_k=True,
      fp8_hybrid=force_hybrid,
      fp8_hybrid_n_early=n_early,
    )
  elif enable_fp8:
    kwargs.update(
      # Uses QK INT8 for better precision, same as SageAttention2.
      fp8_qk_mm_type="int8",
      fp8_pv_acc_type="f16",
      fp8_q_quant_method="per_thread",
      fp8_k_quant_method="per_thread",
      fp8_v_quant_method="per_channel",
      fp8_smooth_k=True,
      # Hybrid keeps fp16 precision on the early Q rows (attention sink);
      # opt-in via --ffpa-hybrid, off by default.
      fp8_hybrid=force_hybrid,
      fp8_hybrid_n_early=n_early,
    )
  elif enable_fp4:
    kwargs.update(
      # FP4 already uses fine-grained per-group(16) quantization for better
      # precision, the same as SageAttention3. Hybrid is opt-in via --ffpa-hybrid
      # and off by default (including causal).
      # NOTE: This hybrid mode will be removed once we have better precision for fp4.
      fp4_hybrid=force_hybrid,
      fp4_hybrid_n_early=n_early,
      fp4_hadamard=_ffpa_fp4_hadamard_override,
    )
  if enable_fp8 and _ffpa_fp8_hadamard_override:
    # Injected after both fp8 presets; off-path stays free of the kwarg so
    # older ffpa-attn builds (without the field) keep working.
    kwargs["fp8_hadamard"] = True
  if enable_fp8 and _ffpa_fp8_smooth_v_override:
    # Requires per-channel V (both fp8 presets use it); off by default so
    # per-channel V regressions are not masked by the mean subtraction.
    kwargs["fp8_smooth_v"] = True
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


def _is_nhd_supported(
  nhd_out: bool,
  backend: "CUDABackend",
  headdim: int,
) -> bool:
  """Whether the caller-side NHD fast-path conditions hold.

  Only the per-family rules (hadamard, head_dim caps per quant family)
  live in CUDABackend.is_nhd_supported; hybrid no longer blocks NHD (its
  stage-1 writeback is a stride-generic slice copy). The caller owns the
  remaining gates: no attn_mask (the fp16 + bias route stores a
  BHND-packed O) and sm_120+ (the NHD O-store branch requires the
  CUTE_TMA family).

  :returns: True when the NHD fast path may run.
  """
  return (nhd_out and not torch.is_grad_enabled() and backend.is_nhd_supported(headdim))


def _normalize_attn_mask(attn_mask: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
  """Adapt a cache-dit attention mask to the FFPA/SDPA broadcast contract.

  The mask contract is documented, not inferred: ``True`` keeps an element
  (bool) or the value is an additive bias (float), and which positions or
  sequence shards are masked is the caller's responsibility. Only the
  unambiguous low-rank forms are expanded here: ``[Nkv]`` ->
  ``[1, 1, 1, Nkv]`` and ``[B, Nkv]`` -> ``[B, 1, 1, Nkv]``; 3-D / 4-D
  masks pass through untouched (ffpa-attn normalizes them).

  :param attn_mask: Mask whose last dimension is the key sequence length.
  :param key: Key tensor in NHD layout ``[B, Nkv, H_kv, D]``.
  :returns: Mask broadcastable to ``[B, Nh_q, Nq, Nkv]``.
  """
  if attn_mask.dim() == 1:
    attn_mask = attn_mask[None, None, None, :]
  elif attn_mask.dim() == 2:
    attn_mask = attn_mask[:, None, None, :]
  elif attn_mask.dim() not in (3, 4):
    raise ValueError(f"FFPA attention masks must be 1-D to 4-D, got {attn_mask.dim()}-D.")
  if attn_mask.shape[-1] != key.shape[-3]:
    raise ValueError(f"Attention mask must match the key's sequence length, "
                     f"got {attn_mask.shape[-1]} and {key.shape[-3]}.")
  return attn_mask


def _is_qkv_contiguous(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
) -> bool:
  """Whether the Q/K/V tensors are contiguous in memory.

  :returns: True if the three tensors are contiguous, False otherwise.
  """
  return (query.is_contiguous() and key.is_contiguous() and value.is_contiguous())


def _ffpa_attn_core(
  query: torch.Tensor,
  key: torch.Tensor,
  value: torch.Tensor,
  attn_mask: Optional[torch.Tensor],
  is_causal: bool,
  scale: Optional[float],
  enable_gqa: bool,
  backend: "CUDABackend",
  nhd_native: bool = False,
  nhd_out: bool = True,
) -> torch.Tensor:
  # diffusers NHD [B, N, H, D] -> FFPA BHND [B, H, N, D]. Every CUDA kernel
  # family (persist-D / split-D / M4N2 across fp8 / fp4 / fp16) reads NHD
  # gmem natively — packed-NHD and strided fused-QKV chunk views alike —
  # so the three tensors pass through zero-copy; layouts outside the NHD
  # contract fail loudly in the C++ layout gates instead of being silently
  # materialized.
  # A non-None mask also forces the HND route: a mask disables the ffpa
  # inference fast path (ffpa_attn_interface) and NHD is a fast-path-only
  # feature (the full meta/autograd chain below assumes BHND-packed
  # tensors, so an NHD call would fail or misread).
  if attn_mask is None and nhd_native and _is_nhd_supported(nhd_out, backend, query.size(-1)):
    # tensor_layout="NHD": the CUDA kernels read NHD inputs and write a
    # contiguous NHD output directly, skipping the input permute views and
    # the BHND->NHD output permute whose non-contiguous view forces a
    # strided copy in downstream consumers (diffusers flatten). Conditions
    # mirror the ffpa fast-path NHD gate; per-family head_dim caps keep
    # unsupported D on the graceful permute fallback below.
    # Stateful per call, like forward_backend.is_causal in the ffpa
    # fast path: the backend is a cached shared object, so every fallback
    # below restores the HND layout it passes.
    backend.tensor_layout = "NHD"
    try:
      return ffpa_attn_func(
        query,
        key,
        value,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
        forward_backend=backend,
      )
    except TypeError as exc:
      # `_ffpa_attn_forward`'s inference gates decline some shapes (short packed
      # sequences, for instance) and `ffpa_attn_func` then raises instead of
      # falling back, because NHD storage is only valid on the fast path. The HND
      # permute below is the same math with one extra transpose, so take it.
      if "requires the inference fast path" not in str(exc):
        raise
      logger.info(
        "FFPA declined the NHD fast path for this shape (seq_len=%s, head_dim=%s): "
        "falling back to the HND permute.", query.size(1), query.size(-1))
      backend.tensor_layout = "HND"
  if nhd_native and _is_qkv_contiguous(query, key, value):
    backend.tensor_layout = "HND"
    q = query.permute(0, 2, 1, 3)
    k = key.permute(0, 2, 1, 3)
    v = value.permute(0, 2, 1, 3)
  else:
    backend.tensor_layout = "HND"
    q = query.permute(0, 2, 1, 3).contiguous()
    k = key.permute(0, 2, 1, 3).contiguous()
    v = value.permute(0, 2, 1, 3).contiguous()
  out = ffpa_attn_func(
    q,
    k,
    v,
    attn_mask=attn_mask,
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
    # attn_mask never reaches this CP forward op: the templated-CP allowlist
    # rejects ffpa with a mask (its shard/order semantics are out of the
    # backend contract), and dropout_p / return_lse are rejected in
    # _ffpa_attention_impl before the template runs. enable_gqa / is_causal
    # are natively supported by FFPA and forwarded as-is. CP input shards
    # arrive NHD packed (send_q/k/v materialize them), and the reassembly
    # primitives (send_o) reshape/copy stride-agnostically — the same NHD
    # contract sage uses — so the forward op keeps the native NHD output
    # (nhd_out default).
    return _ffpa_attn_core(query,
                           key,
                           value,
                           None,
                           is_causal,
                           scale,
                           enable_gqa,
                           backend,
                           nhd_native=nhd_native)

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
  if dropout_p > 0.0:
    raise ValueError("`dropout_p` is not yet supported for FFPA attention.")
  if return_lse:
    raise ValueError("`return_lse` is not supported for FFPA attention.")
  if attn_mask is not None:
    if is_causal:
      raise ValueError("`attn_mask` and `is_causal=True` are mutually exclusive "
                       "for FFPA attention; pass a full mask instead.")
    attn_mask = _normalize_attn_mask(attn_mask, key)
  if ffpa_attn_func is None:
    raise RuntimeError(
      "FFPA attention backend is not available. Please install `ffpa-attn` to use it.")
  if query.dtype not in (torch.float16, torch.bfloat16):
    # The FFPA kernels only read fp16 / bf16. A pipeline-wide backend setting also reaches
    # components that run in float32 (MiniMax-H3's video VAE decoder does), and those are better
    # served by SDPA than by a hard failure at decode time.
    if _cp_config is not None:
      # Plain SDPA has no idea about the other ranks' shards: it would quietly return a
      # local-sequence result instead of the global one.
      raise ValueError("FFPA's SDPA fallback for non-fp16/bf16 inputs does not support context "
                       "parallelism; run the float32 component without a context-parallel plan.")
    sdpa_mask = attn_mask
    if sdpa_mask is not None and sdpa_mask.dim() == 3:
      # `[B, Nq, Nkv]` broadcast as `[1, B, Nq, Nkv]` would be wrong unless B == heads.
      sdpa_mask = sdpa_mask.unsqueeze(1)
    return torch.nn.functional.scaled_dot_product_attention(
      query.transpose(1, 2),
      key.transpose(1, 2),
      value.transpose(1, 2),
      attn_mask=sdpa_mask,
      dropout_p=dropout_p,
      is_causal=is_causal,
      scale=scale,
      enable_gqa=enable_gqa,
    ).transpose(1, 2)
  capability = _require_supported_cuda(query, enable_fp4=enable_fp4)
  # NHD fast path: sm_120 only (the NHD O-store branch lives in the sm_120
  # CUTE_TMA family; the `>=` mirrors the upstream `prop->major >= 12`
  # gate) and bias-free (see _ffpa_attn_core).
  nhd_native = capability[0] >= _SM120_MAJOR

  backend = _build_ffpa_cuda_backend(query.device,
                                     enable_fp8=enable_fp8,
                                     enable_fp4=enable_fp4,
                                     is_causal=is_causal,
                                     fp8_preset=fp8_preset)
  if _cp_config is None:
    return _ffpa_attn_core(query,
                           key,
                           value,
                           attn_mask,
                           is_causal,
                           scale,
                           enable_gqa,
                           backend,
                           nhd_native=nhd_native)
  # A non-None mask is rejected by the templated-CP allowlist (ffpa is not
  # a mask-enabled CP forward op): the mask shard/order semantics are out
  # of the backend contract, so this stays a loud error.
  return _context_parallel_attention(
    query,
    key,
    value,
    attn_mask,
    0.0,
    is_causal,
    scale,
    enable_gqa,
    return_lse,
    cp_gqa_strategy,
    forward_op=_make_ffpa_forward_op(backend, nhd_native),
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
  """FFPA CUDA backend (CUTE_TMA, fp16/bf16, forward-only, sm_120 / sm_89+).

  :param query: ``[B, N, H, D]`` (diffusers / NHD convention).
  :param key: ``[B, N_kv, H, D]``.
  :param value: ``[B, N_kv, H, D]``.
  :param attn_mask: Optional mask broadcastable to ``[B, H, N, N_kv]``
      (SDPA semantics: bool ``True`` keeps an element, float is an additive
      bias). Low-rank ``[N_kv]`` / ``[B, N_kv]`` forms are expanded to 4-D;
      3-D / 4-D masks pass through as-is. The mask itself is forwarded
      untouched: which positions or sequence shards are masked is the
      caller's contract — the backend never infers pad/sharding semantics.
      Mutually exclusive with ``is_causal=True``; not supported together
      with context parallelism.
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
  """FFPA CUDA FP8 backend (CUTE_TMA_FP8, forward-only, sm_120 / sm_89+).

  Inputs stay fp16/bf16; Q/K/V are fp8-quantized inside the kernel using the
  highest-precision config: int8 QK MMA, fp16 PV accumulation, Q/K quantized
  per_thread and V per_channel (supported for every fp8 head_dim including
  non-32-multiple D such as 120). On sm_89 the fp8 family runs the TMA-free
  persist-D kernel (f16 PV accumulator enforced by ffpa-attn).

  :param query: ``[B, N, H, D]`` (diffusers / NHD convention).
  :param key: ``[B, N_kv, H, D]``.
  :param value: ``[B, N_kv, H, D]``.
  :param attn_mask: Optional mask; same contract as the ``ffpa`` backend.
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

  Inputs stay fp16/bf16; Q/K/V are NVFP4-quantized inside the kernel. The
  fp4 family ships the CUTE_TMA_FP4_SM_120 kernels only, so this backend
  keeps the sm_120 gate (sm_89 is rejected before kernel dispatch).

  :param query: ``[B, N, H, D]`` (diffusers / NHD convention).
  :param key: ``[B, N_kv, H, D]``.
  :param value: ``[B, N_kv, H, D]``.
  :param attn_mask: Optional mask; same contract as the ``ffpa`` backend.
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
  """FFPA CUDA FP8 per-block backend (CUTE_TMA_FP8, forward-only, sm_120 / sm_89+).

  Performance-first config: int8 QK MMA, f32 PV accumulation, Q/K/V all
  quantized per_block, smooth K only, and no hybrid fp16 early rows. Fastest
  FFPA fp8 backend but the lowest precision. On sm_89 the f16 PV accumulator
  is enforced by ffpa-attn regardless of the preset.

  :param query: ``[B, N, H, D]`` (diffusers / NHD convention).
  :param key: ``[B, N_kv, H, D]``.
  :param value: ``[B, N_kv, H, D]``.
  :param attn_mask: Optional mask; same contract as the ``ffpa`` backend.
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
