"""Python-facing SVDQ operator entry points.

Next-stage runtime routing (design note, not implemented yet)
-------------------------------------------------------------
The current CuTe kernel is faster than native for short/medium sequences but still loses on
long sequences. The planned automatic policy is:

* ``M <= 4096``: ``cute`` for all ranks (2K/4K measured 3%-7% faster than native).
* ``M == 8192``: ``native`` for rank32 (cute is ~0.7% slower); ``cute`` for rank64/128
  (measured 0.9%-2.9% faster).
* ``M >= 16384``: ``native`` (cute is 1.3%-7.3% slower, worst at rank32).

The thresholds are kernel-only measurements at ``N=K=3072`` and must be revalidated with the
complete ``SVDQW4A4Linear`` path before becoming the default. Until then, callers select
``runtime_kernel`` explicitly. Do not put this policy in the CUDA host launcher: it belongs in
``SVDQW4A4Linear``/config, where the full token count and rank are known.
"""

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
      "The optional Cache-DiT SVDQuant CUDA extension is not available. Build the CUDA "
      "extensions with "
      "`CACHE_DIT_BUILD_CUDA=1 python -m pip install -e . --no-build-isolation` after "
      "`conda activate cdit`. The SVDQuant kernels need sm80+ and exclude sm90, so a build "
      "that printed a skipped-SM warning omits this extension: rerun it with "
      "CACHE_DIT_CUDA_ARCH_LIST set to a supported target.") from error
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


def _normalize_svdq_runtime_stage(stage: int | None, op_name: str) -> int:
  """Normalize and validate the runtime stage for an SVDQ kernel wrapper.

  :param stage: Optional requested pipeline stage count.
  :param op_name: Public operator name used in validation errors.
  :returns: A validated stage count.
  :raises ValueError: If the requested stage is outside the compiled range.
  """

  normalized_stage = 1 if stage is None else int(stage)
  if normalized_stage not in (1, 2, 3):
    raise ValueError(f"{op_name} stage must be one of 1, 2, or 3, got {normalized_stage}.")
  return normalized_stage


def _normalize_svdq_stage(stage: int | None) -> int:
  """Normalize and validate the native runtime stage of `svdq_gemm_w4a4`.

  Ada-class GPUs keep the plain native kernel register-bound, so the public API defaults to
  `stage=1` to preserve better occupancy. Higher stage counts are still accepted for explicit
  profiling or future experiments.

  :param stage: Optional requested pipeline stage count.
  :returns: A validated stage count.
  :raises ValueError: If the requested stage is outside the compiled range.
  """

  return _normalize_svdq_runtime_stage(stage, "svdq_gemm_w4a4")


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


def _normalize_svdq_runtime(runtime: str) -> str:
  """Validate the packed-GEMM runtime selector shared by the SVDQ operator entry points.

  :param runtime: Requested backend name.
  :returns: The validated backend name.
  :raises ValueError: If the requested backend is unknown.
  """

  if runtime not in ("native", "cute"):
    raise ValueError(f"runtime must be 'native' or 'cute', got {runtime!r}.")
  return runtime


def _normalize_svdq_lora_layout(lora_layout: str) -> str:
  """Validate how the LoRA-down term D is laid out for the packed CuTe GEMM.

  :param lora_layout: Requested layout name.
  :returns: The validated layout name.
  :raises ValueError: If the requested layout is unknown.
  """

  if lora_layout not in ("native", "row_major"):
    raise ValueError(f"lora_layout must be 'native' or 'row_major', got {lora_layout!r}.")
  return lora_layout


def _call_svdq_quantize_w4a4_native(
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
    ops_module.quantize_w4a4(
      input,
      output,
      oscales,
      dummy_lora_down,
      dummy_lora_act_out,
      smooth,
      fuse_glu,
      fp4,
      [],
      "native",
    )
    return

  ops_module.quantize_w4a4(
    input,
    output,
    oscales,
    lora_down,
    lora_act_out,
    smooth,
    fuse_glu,
    fp4,
    [],
    "native",
  )


def _call_svdq_gemm_w4a4_ext(
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
    "native",
    "native",
    1,
  )


def _call_svdq_gemm_w4a4_cute(
  act: torch.Tensor,
  wgt: torch.Tensor,
  out: torch.Tensor,
  ascales: torch.Tensor,
  wscales: torch.Tensor,
  lora_act_in: torch.Tensor | None,
  lora_up: torch.Tensor | None,
  bias: torch.Tensor | None,
  wcscales: torch.Tensor | None,
  lora_scales: list[float],
  alpha: float,
  lora_layout: str,
) -> None:
  """Bridge the NVFP4 CuTe GEMM ABI.

  :param act: Row-major packed activation `[M, K / 2]`, low nibble is even K.
  :param wgt: Row-major packed weight `[N, K / 2]`.
  :param out: BF16 output `[M, N]`.
  :param ascales: Activation scales in the CUTLASS 128x4 atom layout.
  :param wscales: Weight scales in CUTLASS 128x4 layout.
  :param lora_act_in: LoRA-down output D as `[M, R]`; the native packed FP32 fragment D for
    `lora_layout='native'`, row-major BF16 with the per-rank scales already folded in for
    `lora_layout='row_major'`.
  :param lora_up: Optional BF16 LoRA-up weight `[N, R]`.
  :param bias: Optional BF16 bias `[N]`.
  :param wcscales: Optional BF16 per-channel scales `[N]`.
  :param lora_scales: Per-16-rank FP32 scales, consumed only by `lora_layout='native'`.
  :param alpha: Scale applied only to the NVFP4 residual product.
  :param lora_layout: Layout of `lora_act_in`, either `'native'` or `'row_major'`.
  """
  ops = _get_required_ops_module()
  ops.gemm_w4a4(
    act,
    wgt,
    out,
    None,
    ascales,
    wscales,
    None,
    None,
    lora_act_in,
    lora_up,
    None,
    None,
    None,
    None,
    None,
    bias,
    None,
    None,
    None,
    False,
    lora_scales,
    False,
    True,
    float(alpha),
    wcscales,
    None,
    None,
    None,
    0,
    "cute",
    _normalize_svdq_lora_layout(lora_layout),
    1,
  )


def _call_svdq_quantize_w4a4(
  input: torch.Tensor,
  output: torch.Tensor,
  ascales: torch.Tensor,
  lora_down: torch.Tensor | None,
  lora_act: torch.Tensor,
  smooth: torch.Tensor | None,
  fuse_glu: bool,
  fp4: bool,
  lora_scales: list[float],
  lora_layout: str,
) -> None:
  """Bridge the unified native/CuTe activation producer through one raw op."""
  ops = _get_required_ops_module()
  ops.quantize_w4a4(input, output, ascales, lora_down, lora_act, smooth, fuse_glu, fp4, lora_scales,
                    lora_layout)


def _call_svdq_gemm_w4a4_native(
  act: torch.Tensor,
  wgt: torch.Tensor,
  out: torch.Tensor,
  ascales: torch.Tensor,
  wscales: torch.Tensor,
  lora_act_in: torch.Tensor | None,
  lora_up: torch.Tensor | None,
  bias: torch.Tensor | None,
  fp4: bool,
  alpha: float,
  wcscales: torch.Tensor | None,
  act_unsigned: bool,
  stage: int,
) -> None:
  """Call the plain native SVDQ GEMM entrypoint.

  On Ada-class GPUs this plain path is register-bound, and `stage=1` is the
  preferred runtime default because it keeps occupancy higher than deeper
  pipeline settings on the hot shapes.

  :param act: Packed activation tensor `[M, K / 2]`.
  :param wgt: Packed quantized weight tensor `[N, K / 2]`.
  :param out: Dense output buffer `[M, N]`.
  :param ascales: Activation scales `[K / 64, M]`.
  :param wscales: Weight scales `[K / 64, N]`.
  :param lora_act_in: Optional LoRA activation input `[M, R]`.
  :param lora_up: Optional LoRA up-projection weights `[N, R]`.
  :param bias: Optional dense output bias `[N]`.
  :param fp4: Whether the packed tensors use FP4/NVFP4 instead of INT4.
  :param alpha: Per-tensor scaling factor. The plain INT4 path requires `1.0`.
  :param wcscales: Optional per-channel post scales `[N]`.
  :param act_unsigned: Whether INT4 activations are stored as unsigned values.
  :param stage: Runtime pipeline stage count for the compiled plain native variants.
  """

  if fp4:
    raise NotImplementedError("svdq_gemm_w4a4 plain native runtime currently supports INT4 only.")

  normalized_stage = _normalize_svdq_stage(stage)
  _get_required_ops_module().gemm_w4a4(
    act,
    wgt,
    out,
    None,
    ascales,
    wscales,
    None,
    None,
    lora_act_in,
    lora_up,
    None,
    None,
    None,
    None,
    None,
    bias,
    None,
    None,
    None,
    act_unsigned,
    [],
    False,
    fp4,
    float(alpha),
    wcscales,
    None,
    None,
    None,
    0,
    "native",
    "native",
    normalized_stage,
  )


torch.library.define(
  "cache_dit_cuda_ops::svdq_quantize_w4a4",
  "(Tensor input, Tensor? lora_down, Tensor? smooth, str lora_layout, bool fuse_glu, bool fp4, float[] lora_scales, int pad_size) -> (Tensor output, Tensor oscales, Tensor lora_act)",
)


@torch.library.impl("cache_dit_cuda_ops::svdq_quantize_w4a4", "CUDA")
def _svdq_quantize_w4a4(
  input: torch.Tensor,
  lora_down: torch.Tensor | None,
  smooth: torch.Tensor | None,
  lora_layout: str,
  fuse_glu: bool,
  fp4: bool,
  lora_scales: list[float],
  pad_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  batch_size, channels = input.shape
  batch_size_pad = ((batch_size + pad_size - 1) // pad_size) * pad_size
  rank = 0 if lora_down is None else lora_down.shape[1]

  if _normalize_svdq_lora_layout(lora_layout) == "row_major":
    if smooth is None:
      raise ValueError("lora_layout='row_major' requires an explicit smooth tensor.")
    if batch_size_pad % 256 != 0:
      raise RuntimeError(
        "svdq_quantize_w4a4 requires the padded batch to be a multiple of 256 for the native "
        f"activation pack, got {batch_size_pad} (batch={batch_size}, pad_size={pad_size})")
    padded_input = input.contiguous()
    if batch_size_pad != batch_size:
      padded_input = torch.nn.functional.pad(padded_input, (0, 0, 0, batch_size_pad - batch_size))
    output = torch.empty((batch_size_pad, channels // 2), dtype=torch.uint8, device=input.device)
    oscales = torch.empty(batch_size_pad * channels // 16,
                          dtype=torch.float8_e4m3fn,
                          device=input.device)
    lora_act = torch.empty((batch_size_pad, rank), dtype=torch.bfloat16, device=input.device)
    _call_svdq_quantize_w4a4(padded_input, output, oscales, lora_down, lora_act, smooth, False,
                             True, lora_scales, "row_major")
    return output, oscales, lora_act

  output = torch.empty(batch_size_pad, channels // 2, dtype=torch.uint8, device=input.device)
  if fp4:
    if channels % 16 != 0:
      raise ValueError(f"Expected channels divisible by 16 for FP4 quantization, got {channels}.")
    oscales = torch.empty(channels // 16,
                          batch_size_pad,
                          dtype=torch.float8_e4m3fn,
                          device=input.device)
  else:
    if channels % 64 != 0:
      raise ValueError(f"Expected channels divisible by 64 for INT4 quantization, got {channels}.")
    oscales = torch.empty(channels // 64, batch_size_pad, dtype=input.dtype, device=input.device)
  lora_act_out = torch.empty(batch_size_pad, rank, dtype=torch.float32, device=input.device)

  _call_svdq_quantize_w4a4_native(
    input,
    output,
    oscales,
    lora_down,
    lora_act_out,
    smooth,
    fuse_glu,
    fp4,
  )
  return output, oscales, lora_act_out


@torch.library.register_fake("cache_dit_cuda_ops::svdq_quantize_w4a4")
def _fake_svdq_quantize_w4a4(
  input: torch.Tensor,
  lora_down: torch.Tensor | None,
  smooth: torch.Tensor | None,
  lora_layout: str,
  fuse_glu: bool,
  fp4: bool,
  lora_scales: list[float],
  pad_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  del smooth, fuse_glu, lora_scales
  batch_size, channels = input.shape
  batch_size_pad = ((batch_size + pad_size - 1) // pad_size) * pad_size
  rank = 0 if lora_down is None else lora_down.shape[1]
  output = input.new_empty((batch_size_pad, channels // 2), dtype=torch.uint8)
  if _normalize_svdq_lora_layout(lora_layout) == "row_major":
    oscales = input.new_empty((batch_size_pad * channels // 16, ), dtype=torch.float8_e4m3fn)
    lora_act = input.new_empty((batch_size_pad, rank), dtype=torch.bfloat16)
    return output, oscales, lora_act
  oscales = input.new_empty(
    (channels // (16 if fp4 else 64), batch_size_pad),
    dtype=torch.float8_e4m3fn if fp4 else input.dtype,
  )
  lora_act_out = input.new_empty((batch_size_pad, rank), dtype=torch.float32)
  return output, oscales, lora_act_out


torch.library.define(
  "cache_dit_cuda_ops::svdq_gemm_w4a4",
  "(Tensor act, Tensor wgt, Tensor ascales, Tensor wscales, Tensor? lora_act_in, Tensor? lora_up, Tensor? bias, Tensor? wcscales, str runtime, bool fp4, float alpha, bool act_unsigned, int out_dtype_id, int stage, float[] lora_scales, str lora_layout) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::svdq_gemm_w4a4", "CUDA")
def _svdq_gemm_w4a4(
  act: torch.Tensor,
  wgt: torch.Tensor,
  ascales: torch.Tensor,
  wscales: torch.Tensor,
  lora_act_in: torch.Tensor | None,
  lora_up: torch.Tensor | None,
  bias: torch.Tensor | None,
  wcscales: torch.Tensor | None,
  runtime: str,
  fp4: bool,
  alpha: float,
  act_unsigned: bool,
  out_dtype_id: int,
  stage: int,
  lora_scales: list[float],
  lora_layout: str,
) -> torch.Tensor:
  if _normalize_svdq_runtime(runtime) == "cute":
    output = torch.empty((act.shape[0], wgt.shape[0]), dtype=torch.bfloat16, device=act.device)
    _call_svdq_gemm_w4a4_cute(act, wgt, output, ascales, wscales, lora_act_in, lora_up, bias,
                              wcscales, lora_scales, alpha, lora_layout)
    return output

  if _normalize_svdq_lora_layout(lora_layout) != "native":
    raise ValueError("runtime='native' only supports lora_layout='native'.")

  output = torch.empty((act.shape[0], wgt.shape[0]),
                       dtype=_decode_svdq_output_dtype(out_dtype_id),
                       device=act.device)
  if not fp4:
    _call_svdq_gemm_w4a4_native(act, wgt, output, ascales, wscales, lora_act_in, lora_up, bias, fp4,
                                float(alpha), wcscales, act_unsigned, _normalize_svdq_stage(stage))
    return output

  # Keep positional placeholders aligned with csrc/kernels/svdq/ops.h::gemm_w4a4.
  # This compile-friendly wrapper only uses the plain linear path.
  _call_svdq_gemm_w4a4_ext(
    act,
    wgt,
    output,
    None,  # qout: no re-quantized output buffer
    ascales,
    wscales,
    None,  # oscales: no output activation scales requested
    None,  # poolout: no pooled output buffer
    lora_act_in,
    lora_up,
    None,  # lora_down: no fused LoRA-down branch here
    None,  # lora_act_out: no intermediate LoRA activation output
    None,  # norm_q: not running RMSNorm/Q attention fusion
    None,  # norm_k: not running RMSNorm/K attention fusion
    None,  # rotary_emb: not running RoPE fusion
    bias,
    None,  # smooth_factor: not producing next-layer smoothing factors
    None,  # out_vk: not writing linear-attention VK output
    None,  # out_linearattn: not writing linear-attention output
    act_unsigned,
    lora_scales,  # per-16-rank LoRA scales, already normalized by the public wrapper
    False,  # fuse_silu: not running SiLU fusion
    fp4,
    float(alpha),
    wcscales,
    None,  # out_q: not writing attention Q output
    None,  # out_k: not writing attention K output
    None,  # out_v: not writing attention V output
    0,
  )
  return output


@torch.library.register_fake("cache_dit_cuda_ops::svdq_gemm_w4a4")
def _fake_svdq_gemm_w4a4(
  act: torch.Tensor,
  wgt: torch.Tensor,
  ascales: torch.Tensor,
  wscales: torch.Tensor,
  lora_act_in: torch.Tensor | None,
  lora_up: torch.Tensor | None,
  bias: torch.Tensor | None,
  wcscales: torch.Tensor | None,
  runtime: str,
  fp4: bool,
  alpha: float,
  act_unsigned: bool,
  out_dtype_id: int,
  stage: int,
  lora_scales: list[float],
  lora_layout: str,
) -> torch.Tensor:
  del ascales, wscales, lora_act_in, lora_up, bias, wcscales, fp4, alpha, act_unsigned, lora_scales
  if _normalize_svdq_runtime(runtime) == "cute":
    _normalize_svdq_lora_layout(lora_layout)
    return act.new_empty((act.shape[0], wgt.shape[0]), dtype=torch.bfloat16)
  _normalize_svdq_stage(stage)
  return act.new_empty((act.shape[0], wgt.shape[0]), dtype=_decode_svdq_output_dtype(out_dtype_id))


torch.library.define(
  "cache_dit_cuda_ops::svdq_gemm_w4a4_ext",
  "(Tensor act, Tensor wgt, Tensor out, Tensor? qout, Tensor? ascales, Tensor? wscales, Tensor? oscales, Tensor? poolout, Tensor? lora_act_in, Tensor? lora_up, Tensor? lora_down, Tensor? lora_act_out, Tensor? norm_q, Tensor? norm_k, Tensor? rotary_emb, Tensor? bias, Tensor? smooth_factor, Tensor? out_vk, Tensor? out_linearattn, bool act_unsigned, float[] lora_scales, bool fuse_silu, bool fp4, float alpha, Tensor? wcscales, Tensor? out_q, Tensor? out_k, Tensor? out_v, int attn_tokens) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::svdq_gemm_w4a4_ext", "CUDA")
def _impl_svdq_gemm_w4a4_ext(
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
  lora_scales: list[float],
  fuse_silu: bool,
  fp4: bool,
  alpha: float,
  wcscales: torch.Tensor | None,
  out_q: torch.Tensor | None,
  out_k: torch.Tensor | None,
  out_v: torch.Tensor | None,
  attn_tokens: int,
) -> torch.Tensor:
  _call_svdq_gemm_w4a4_ext(
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
    lora_scales,
    fuse_silu,
    fp4,
    alpha,
    wcscales,
    out_q,
    out_k,
    out_v,
    attn_tokens,
  )
  return out


@torch.library.register_fake("cache_dit_cuda_ops::svdq_gemm_w4a4_ext")
def _fake_svdq_gemm_w4a4_ext(
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
  lora_scales: list[float],
  fuse_silu: bool,
  fp4: bool,
  alpha: float,
  wcscales: torch.Tensor | None,
  out_q: torch.Tensor | None,
  out_k: torch.Tensor | None,
  out_v: torch.Tensor | None,
  attn_tokens: int,
) -> torch.Tensor:
  del act, wgt, qout, ascales, wscales, oscales, poolout, lora_act_in, lora_up, lora_down
  del lora_act_out, norm_q, norm_k, rotary_emb, bias, smooth_factor, out_vk, out_linearattn
  del act_unsigned, lora_scales, fuse_silu, fp4, alpha, wcscales, out_q, out_k, out_v, attn_tokens
  return out


def svdq_quantize_w4a4(
  input: torch.Tensor,
  lora_down: torch.Tensor | None = None,
  smooth: torch.Tensor | None = None,
  lora_layout: str = "native",
  fp4: bool = True,
  fuse_glu: bool = False,
  lora_scales: list[float] | None = None,
  pad_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Quantize activations and fuse the LoRA-down reduction into one producer launch.

  `lora_layout` selects the D contract, and the following GEMM has to declare the same value:
  `'native'` emits the packed FP32 MMA-fragment D consumed by
  `svdq_gemm_w4a4(..., lora_layout='native')`, while `'row_major'` emits row-major BF16 D with the
  per-rank scales already folded in.

  :param input: Dense activations with shape `[tokens, in_features]`.
  :param lora_down: Optional LoRA down-projection weight `[in_features, R]`.
  :param smooth: Optional smoothing factors `[in_features]`; required by `'row_major'`.
  :param lora_layout: LoRA-down output layout, either `'native'` or `'row_major'`.
  :param fp4: Whether the packed activations use FP4/NVFP4 instead of INT4.
  :param fuse_glu: Whether GLU fusion should be enabled in the native kernel.
  :param lora_scales: Optional FP32 scale per 16-rank group, consumed by `'row_major'` and
    defaulting to ones.
  :param pad_size: Token padding multiple required by the packed activation layout.
  :returns: `(packed_activations, activation_scales, lora_down_output)`.
  :raises ValueError: If `lora_layout` is unknown, or if `'row_major'` is asked for an INT4 pack.
  """
  normalized_layout = _normalize_svdq_lora_layout(lora_layout)
  if normalized_layout == "row_major" and not fp4:
    raise ValueError("lora_layout='row_major' implements the NVFP4 producer only; pass fp4=True.")
  rank = 0 if lora_down is None else lora_down.shape[1]
  normalized_scales = lora_scales
  if normalized_scales is None:
    normalized_scales = [1.0] * max(1, rank // 16)
  return torch.ops.cache_dit_cuda_ops.svdq_quantize_w4a4(input, lora_down, smooth,
                                                         normalized_layout, fuse_glu, fp4,
                                                         normalized_scales, pad_size)


def svdq_gemm_w4a4(
  act: torch.Tensor,
  wgt: torch.Tensor,
  ascales: torch.Tensor,
  wscales: torch.Tensor,
  lora_act_in: torch.Tensor | None = None,
  lora_up: torch.Tensor | None = None,
  bias: torch.Tensor | None = None,
  wcscales: torch.Tensor | None = None,
  runtime: str = "native",
  lora_layout: str = "native",
  fp4: bool = True,
  alpha: float = 1.0,
  act_unsigned: bool = False,
  output_dtype: torch.dtype | None = None,
  stage: int = 1,
  lora_scales: list[float] | None = None,
) -> torch.Tensor:
  """Run the SVDQ W4A4 GEMM on the selected runtime with an optional LoRA-up correction.

  `runtime` picks the kernel family: `'native'` uses the plain INT4 kernel for INT4 and the
  ext kernel for NVFP4/fused paths, while `'cute'` uses the SM120 CuTe NVFP4 GEMM. `lora_layout` picks how the
  LoRA-down term is delivered and must match the producer that filled `lora_act_in`:

  Only three combinations are valid:
  `runtime='native'` + `lora_layout='native'` (packed FP32 fragment D),
  `runtime='cute'` + `lora_layout='native'` (packed FP32 fragment D),
  and `runtime='cute'` + `lora_layout='row_major'` (row-major BF16 D with the scales folded in).

  :param act: Packed activation tensor `[M, K / 2]`.
  :param wgt: Packed weight tensor `[N, K / 2]`.
  :param ascales: Activation scales, in the atom layout required by `runtime`.
  :param wscales: Weight scales, in the atom layout required by `runtime`.
  :param lora_act_in: Optional LoRA-down output `[M, R]`.
  :param lora_up: Optional BF16 LoRA-up weight `[N, R]`.
  :param bias: Optional BF16 bias `[N]`.
  :param wcscales: Optional BF16 per-channel scales `[N]`, applied only to the packed product.
  :param runtime: Backend selector, either `'native'` or `'cute'`.
  :param lora_layout: LoRA-down layout, either `'native'` or `'row_major'`; only the CuTe runtime
    honours `'row_major'`.
  :param fp4: Whether the packed tensors use FP4/NVFP4 instead of INT4.
  :param alpha: FP32 scale applied only to the packed product.
  :param act_unsigned: Whether INT4 activations are stored as unsigned values.
  :param output_dtype: Optional dense output dtype; the CuTe runtime always returns BF16.
  :param stage: Runtime pipeline stage count for the plain native INT4 kernel.
  :param lora_scales: Optional FP32 scale per 16-rank group, consumed by
    `lora_layout='native'` and defaulting to ones.
  :returns: Dense output `[M, N]`.
  :raises ValueError: If `runtime`/`lora_layout` is unknown, if `runtime='native'` is paired with
    `lora_layout='row_major'`, if the CuTe runtime is asked for INT4 or a non-BF16 output, or if
    per-rank `lora_scales` are passed for row-major D.
  """
  normalized_runtime = _normalize_svdq_runtime(runtime)
  normalized_layout = _normalize_svdq_lora_layout(lora_layout)
  if normalized_runtime == "native" and normalized_layout != "native":
    raise ValueError("runtime='native' only supports lora_layout='native'; use runtime='cute' for "
                     "row-major BF16 D.")
  if normalized_runtime == "cute":
    if not fp4:
      raise ValueError("runtime='cute' implements the NVFP4 path only; pass fp4=True.")
    if output_dtype is not None and output_dtype != torch.bfloat16:
      raise ValueError(f"runtime='cute' always produces BF16 output, got {output_dtype}.")
    if normalized_layout == "row_major" and lora_scales is not None:
      raise ValueError("lora_layout='row_major' expects the per-rank scales folded into D; drop "
                       "lora_scales.")
  if alpha is None:
    alpha = 1.0
  if output_dtype is None:
    # The CuTe runtime always writes BF16, so it needs no dtype hint from the auxiliary tensors.
    output_dtype = (torch.bfloat16 if normalized_runtime == "cute" else _infer_svdq_output_dtype(
      None, lora_up, bias, wscales))
    if output_dtype is None:
      raise ValueError("Unable to infer the output dtype for svdq_gemm_w4a4.")
  return torch.ops.cache_dit_cuda_ops.svdq_gemm_w4a4(
    act,
    wgt,
    ascales,
    wscales,
    lora_act_in,
    lora_up,
    bias,
    wcscales,
    normalized_runtime,
    fp4,
    float(alpha),
    act_unsigned,
    _encode_svdq_output_dtype(output_dtype),
    _normalize_svdq_stage(stage),
    _normalize_svdq_lora_scales(lora_scales, lora_up),
    normalized_layout,
  )


def _svdq_gemm_w4a4_ext(
  act: torch.Tensor,
  wgt: torch.Tensor,
  out: torch.Tensor | None = None,
  qout: torch.Tensor | None = None,
  ascales: torch.Tensor | None = None,
  wscales: torch.Tensor | None = None,
  oscales: torch.Tensor | None = None,
  poolout: torch.Tensor | None = None,
  lora_act_in: torch.Tensor | None = None,
  lora_up: torch.Tensor | None = None,
  lora_down: torch.Tensor | None = None,
  lora_act_out: torch.Tensor | None = None,
  norm_q: torch.Tensor | None = None,
  norm_k: torch.Tensor | None = None,
  rotary_emb: torch.Tensor | None = None,
  bias: torch.Tensor | None = None,
  smooth_factor: torch.Tensor | None = None,
  out_vk: torch.Tensor | None = None,
  out_linearattn: torch.Tensor | None = None,
  act_unsigned: bool = False,
  lora_scales: list[float] | None = None,
  fuse_silu: bool = False,
  fp4: bool = False,
  alpha: float | None = 1.0,
  wcscales: torch.Tensor | None = None,
  out_q: torch.Tensor | None = None,
  out_k: torch.Tensor | None = None,
  out_v: torch.Tensor | None = None,
  attn_tokens: int = 0,
) -> torch.Tensor:
  if alpha is None:
    alpha = 1.0
  if out is None:
    output_dtype = _infer_svdq_output_dtype(None, lora_up, bias, wscales)
    if output_dtype is None:
      raise ValueError("Unable to infer the output dtype for _svdq_gemm_w4a4_ext.")
    out = torch.empty((act.shape[0], wgt.shape[0]), dtype=output_dtype, device=act.device)
  return torch.ops.cache_dit_cuda_ops.svdq_gemm_w4a4_ext(
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
    _normalize_svdq_lora_scales(lora_scales, lora_up),
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
  "svdq_get_load_error",
  "svdq_extension_is_available",
  "svdq_gemm_w4a4",
  "svdq_quantize_w4a4",
]
