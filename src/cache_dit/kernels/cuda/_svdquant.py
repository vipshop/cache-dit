import importlib
from types import ModuleType

import torch

_DTYPE_TO_ID = {
  torch.float16: 0,
  torch.bfloat16: 1,
  torch.float32: 2,
}
_ID_TO_DTYPE = {value: key for key, value in _DTYPE_TO_ID.items()}

_EXTENSION_MODULE_NAME = "cache_dit._C_svdquant"
_cached_extension_module: ModuleType | None = None
_cached_load_error: Exception | None = None


def _load_extension_module() -> ModuleType | None:
  global _cached_extension_module, _cached_load_error

  if _cached_extension_module is not None:
    return _cached_extension_module
  if _cached_load_error is not None:
    return None

  try:
    _cached_extension_module = importlib.import_module(_EXTENSION_MODULE_NAME)
  except Exception as exc:  # pragma: no cover - exercised in environments without the extension
    _cached_load_error = exc
    return None

  return _cached_extension_module


def svdq_extension_is_available() -> bool:
  """Return whether the optional SVDQ CUDA extension can be imported.

  :returns: `True` when the compiled SVDQ extension module is available.
  """

  return _load_extension_module() is not None


def svdq_get_load_error() -> Exception | None:
  """Return the cached import error from the optional SVDQ CUDA extension.

  :returns: The import error raised while loading the extension, or `None`
    when the extension loaded successfully.
  """

  _load_extension_module()
  return _cached_load_error


def _get_required_extension_module() -> ModuleType:
  """Return the loaded SVDQ extension module or raise a guided runtime error.

  :returns: The loaded Python extension module backing the SVDQ CUDA ops.
  :raises RuntimeError: If the optional extension is unavailable.
  """

  extension_module = _load_extension_module()
  if extension_module is None:
    error = svdq_get_load_error()
    raise RuntimeError(
      "The optional Cache-DiT SVDQuant CUDA extension is not available. Build it with "
      "`CACHE_DIT_BUILD_SVDQUANT=1 /workspace/dev/miniconda3/envs/cdit/bin/python -m pip install -e . --no-build-isolation` after `conda activate cdit`."
    ) from error
  return extension_module


def _get_required_ops_module() -> ModuleType:
  """Return the extension's `ops` submodule used by the Python wrappers.

  :returns: The `ops` submodule exported by the loaded SVDQ extension.
  :raises RuntimeError: If the extension does not expose the expected module.
  """

  ops_module = getattr(_get_required_extension_module(), "ops", None)
  if ops_module is None:
    raise RuntimeError(
      "The loaded Cache-DiT SVDQuant extension does not expose an `ops` submodule.")
  return ops_module


def _get_required_utils_module() -> ModuleType:
  """Return the extension's `utils` submodule used by advanced helpers.

  :returns: The `utils` submodule exported by the loaded SVDQ extension.
  :raises RuntimeError: If the extension does not expose the expected module.
  """

  utils_module = getattr(_get_required_extension_module(), "utils", None)
  if utils_module is None:
    raise RuntimeError(
      "The loaded Cache-DiT SVDQuant extension does not expose a `utils` submodule.")
  return utils_module


def _encode_svdq_output_dtype(output_dtype: torch.dtype) -> int:
  """Encode a public torch dtype into the integer id expected by the extension.

  :param output_dtype: Torch dtype requested by the Python wrapper caller.
  :returns: Integer dtype id understood by the compiled extension.
  """

  dtype_id = _DTYPE_TO_ID.get(output_dtype)
  if dtype_id is None:
    raise ValueError(f"Unsupported SVDQuant output dtype: {output_dtype}")
  return dtype_id


def _decode_svdq_output_dtype(dtype_id: int) -> torch.dtype:
  """Decode an extension dtype id back into a public torch dtype.

  :param dtype_id: Integer dtype id returned by the compiled extension.
  :returns: The corresponding public torch dtype.
  """

  output_dtype = _ID_TO_DTYPE.get(dtype_id)
  if output_dtype is None:
    raise ValueError(f"Unsupported SVDQuant output dtype id: {dtype_id}")
  return output_dtype


def _infer_svdq_output_dtype(
  out: torch.Tensor | None,
  lora_up: torch.Tensor | None,
  bias: torch.Tensor | None,
  wscales: torch.Tensor | None,
) -> torch.dtype | None:
  """Infer the dense output dtype from explicit buffers or auxiliary tensors.

  The runtime prefers the destination buffer dtype when one is supplied. If no
  explicit output exists, it falls back to LoRA, bias, or scale tensors whose
  dtype also constrains the final dense output.

  :param out: Optional caller-provided dense output buffer.
  :param lora_up: Optional LoRA up-projection tensor.
  :param bias: Optional dense bias tensor.
  :param wscales: Optional weight-scale tensor.
  :returns: The inferred output dtype, or `None` when no reliable dtype source
    is available.
  """

  if out is not None:
    return out.dtype
  if lora_up is not None:
    return lora_up.dtype
  if bias is not None:
    return bias.dtype
  if wscales is not None and wscales.dtype in (torch.float16, torch.bfloat16, torch.float32):
    return wscales.dtype
  return None


def _normalize_svdq_lora_scales(
  lora_scales: list[float] | None,
  lora_up: torch.Tensor | None,
) -> list[float]:
  """Normalize optional LoRA scaling factors to the ABI expected by the kernel.

  When callers omit explicit scales but provide `lora_up`, the kernel still
  expects one scale value per 16-rank group. This helper synthesizes a list of
  `1.0` values for those groups.

  :param lora_scales: Optional user-provided per-group LoRA scales.
  :param lora_up: Optional LoRA up-projection tensor used to infer the rank.
  :returns: A normalized per-group scale list ready for the extension call.
  """

  if lora_scales is not None:
    return lora_scales
  if lora_up is None:
    return []
  rank = lora_up.shape[1]
  return [1.0] * ((rank + 15) // 16)


def _call_svdq_quantize_w4a4_act_fuse_lora(
  input: torch.Tensor,
  output: torch.Tensor,
  oscales: torch.Tensor,
  lora_down: torch.Tensor | None,
  lora_act_out: torch.Tensor,
  smooth: torch.Tensor | None,
  fuse_glu: bool,
  fp4: bool,
) -> None:
  """Bridge the Python activation quantizer wrapper to the compiled extension.

  The CUDA kernel requires a non-zero LoRA rank even when the caller does not request LoRA fusion.
  For the rank-zero case this helper materializes a small dummy LoRA path so the ABI stays uniform
  while the effective result remains unchanged.

  :param input: Dense activation tensor to quantize.
  :param output: Destination buffer for packed activations.
  :param oscales: Destination buffer for activation scales.
  :param lora_down: Optional LoRA down-projection tensor.
  :param lora_act_out: Destination buffer for fused LoRA activations.
  :param smooth: Optional smoothing tensor applied before quantization.
  :param fuse_glu: Whether GLU fusion should be enabled in the kernel.
  :param fp4: Whether to use the FP4/NVFP4 quantization path.
  """

  ops_module = _get_required_ops_module()
  rank = 0 if lora_down is None else lora_down.shape[1]
  if rank == 0:
    dummy_rank = 16
    dummy_lora_down = torch.zeros(input.shape[1],
                                  dummy_rank,
                                  dtype=input.dtype,
                                  device=input.device)
    dummy_lora_act_out = torch.empty(output.shape[0],
                                     dummy_rank,
                                     dtype=torch.float32,
                                     device=input.device)
    ops_module.quantize_w4a4_act_fuse_lora(
      input,
      output,
      oscales,
      dummy_lora_down,
      dummy_lora_act_out,
      smooth,
      fuse_glu,
      fp4,
    )
    return

  ops_module.quantize_w4a4_act_fuse_lora(
    input,
    output,
    oscales,
    lora_down,
    lora_act_out,
    smooth,
    fuse_glu,
    fp4,
  )


def _call_svdq_quantize_w4a4_wgt(
  input: torch.Tensor,
  output: torch.Tensor,
  oscales: torch.Tensor,
) -> None:
  """Bridge the Python weight quantizer wrapper to the compiled extension.

  :param input: Dense floating-point weight tensor.
  :param output: Destination buffer for packed INT4 weights.
  :param oscales: Destination buffer for per-group weight scales.
  """

  _get_required_ops_module().quantize_w4a4_wgt(input, output, oscales)


def _call_svdq_gemm_w4a4(
  act: torch.Tensor,
  wgt: torch.Tensor,
  out: torch.Tensor,
  qout: torch.Tensor | None,
  ascales: torch.Tensor | None,
  wscales: torch.Tensor | None,
  oscales: torch.Tensor | None,
  poolout: torch.Tensor | None,
  lora_act_in: torch.Tensor | None,
  lora_up: torch.Tensor | None,
  lora_down: torch.Tensor | None,
  lora_act_out: torch.Tensor | None,
  norm_q: torch.Tensor | None,
  norm_k: torch.Tensor | None,
  rotary_emb: torch.Tensor | None,
  bias: torch.Tensor | None,
  smooth_factor: torch.Tensor | None,
  out_vk: torch.Tensor | None,
  out_linearattn: torch.Tensor | None,
  act_unsigned: bool,
  lora_scales: list[float] | None,
  fuse_silu: bool,
  fp4: bool,
  alpha: float,
  wcscales: torch.Tensor | None,
  out_q: torch.Tensor | None,
  out_k: torch.Tensor | None,
  out_v: torch.Tensor | None,
  attn_tokens: int,
) -> None:
  """Direct binding to the full SVDQ W4A4 CUDA GEMM ABI.

  :param act: Packed activation tensor `[M, K / 2]`.
  :param wgt: Packed quantized weight tensor `[N, K / 2]`.
  :param out: Dense output buffer `[M, N]`.
  :param qout: Optional packed quantized output buffer `[M, N / 2]` for the next layer.
  :param ascales: Activation scales `[K / G, M]`, where `G` is 64 for INT4 and 16 for FP4.
  :param wscales: Weight scales `[K / G, N]`, where `G` is 64 for INT4 and 16 for FP4.
  :param oscales: Optional output scales `[N / G, M]` for `qout`.
  :param poolout: Optional pooled output buffer used by specialized fused kernels.
  :param lora_act_in: Optional LoRA activation input `[M, R]`.
  :param lora_up: Optional LoRA up-projection weights `[N, R]`.
  :param lora_down: Optional LoRA down-projection weights `[N, R]` for the next fused layer.
  :param lora_act_out: Optional LoRA activation output buffer `[M, R]` for the next fused layer.
  :param norm_q: Optional query RMSNorm tensor `[HEAD_DIM]`.
  :param norm_k: Optional key RMSNorm tensor `[HEAD_DIM]`.
  :param rotary_emb: Optional packed rotary embeddings `[M, HEAD_DIM / 2, 2, 2]`.
  :param bias: Optional dense output bias `[N]`.
  :param smooth_factor: Optional smoothing factors `[N]` written for next-layer quantization.
  :param out_vk: Optional linear-attention VK output buffer.
  :param out_linearattn: Optional linear-attention output buffer.
  :param act_unsigned: Whether INT4 activations are stored as unsigned values.
  :param lora_scales: Optional per-16-rank LoRA scaling factors `[R / 16]`.
  :param fuse_silu: Whether to fuse SiLU inside supported kernel variants.
  :param fp4: Whether the packed tensors use FP4/NVFP4 instead of INT4.
  :param alpha: Per-tensor FP4 scaling factor.
  :param wcscales: Optional per-channel FP4 scales `[N]`.
  :param out_q: Optional packed attention-Q output buffer `[B, H, M, D]`.
  :param out_k: Optional packed attention-K output buffer `[B, H, M, D]`.
  :param out_v: Optional packed attention-V output buffer `[B, H, M, D]`.
  :param attn_tokens: Number of attention tokens for fused attention-style kernels.

  :returns: None. Results are written in-place to the provided output tensors.
  """
  ops_module = _get_required_ops_module()
  normalized_lora_scales = _normalize_svdq_lora_scales(lora_scales, lora_up)
  ops_module.gemm_w4a4(
    act,
    wgt,
    out,
    qout,
    ascales,
    wscales,
    oscales,
    poolout,
    lora_act_in,
    lora_up,
    lora_down,
    lora_act_out,
    norm_q,
    norm_k,
    rotary_emb,
    bias,
    smooth_factor,
    out_vk,
    out_linearattn,
    act_unsigned,
    normalized_lora_scales,
    fuse_silu,
    fp4,
    float(alpha),
    wcscales,
    out_q,
    out_k,
    out_v,
    attn_tokens,
  )


__all__ = [
  "_call_svdq_gemm_w4a4",
  "_call_svdq_quantize_w4a4_act_fuse_lora",
  "_call_svdq_quantize_w4a4_wgt",
  "_decode_svdq_output_dtype",
  "_encode_svdq_output_dtype",
  "_get_required_utils_module",
  "_infer_svdq_output_dtype",
  "_normalize_svdq_lora_scales",
  "svdq_get_load_error",
  "svdq_extension_is_available",
]
