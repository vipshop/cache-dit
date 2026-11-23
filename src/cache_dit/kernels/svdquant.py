"""Python-facing SVDQ operator entry points.

Automatic runtime routing (design note, not implemented yet)
-----------------------------------------------------------
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

SVDQ-FP8 numerical contract
---------------------------
For a Linear with weight ``W[N,K]`` and smoothing ``s[K]``, the offline quantizer writes
``W*s = L1 @ L2 + R`` and stores the E4M3 residual with a per-output-channel scale. The fused
producer computes per-token activation scales, E4M3 codes and the BF16 LoRA-down term in one pass;
the GEMM accumulates the E4M3 residual product and BF16 LoRA-up product in the same FP32
accumulator. No standalone BF16 GEMM or second pass over the dense output is involved.
"""

import torch

from .cuda_ext import _get_required_ops_module

_DTYPE_TO_ID = {
  torch.float16: 0,
  torch.bfloat16: 1,
  torch.float32: 2,
}
_ID_TO_DTYPE = {value: key for key, value in _DTYPE_TO_ID.items()}


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
  gate: torch.Tensor | None,
  fuse_glu: bool,
  fp4: bool,
  gate_act: str,
  gate_first: bool,
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
  :param gate: Optional gating half for a fused SwiGLU; when omitted the second half of `input`
    supplies it.
  :param fuse_glu: Whether GLU fusion should be enabled in the kernel.
  :param fp4: Whether to use the FP4/NVFP4 quantization path.
  :param gate_act: Gate activation name forwarded to the extension.
  :param gate_first: Whether packed input stores gate before value.
  """

  ops_module = _get_required_ops_module()
  rank = 0 if lora_down is None else lora_down.shape[1]
  if rank == 0:
    dummy_rank = 16
    value_channels = input.shape[1]
    if fuse_glu and gate is None:
      value_channels //= 2
    dummy_lora_down = torch.zeros(value_channels,
                                  dummy_rank,
                                  dtype=input.dtype,
                                  device=input.device)
    dummy_lora_act_out = torch.empty(output.shape[0],
                                     dummy_rank,
                                     dtype=torch.float32,
                                     device=input.device)
    ops_module.svdq_quantize_w4a4(
      input,
      output,
      oscales,
      dummy_lora_down,
      dummy_lora_act_out,
      smooth,
      gate,
      fuse_glu,
      fp4,
      [],
      "native",
      gate_act,
      gate_first,
    )
    return

  ops_module.svdq_quantize_w4a4(
    input,
    output,
    oscales,
    lora_down,
    lora_act_out,
    smooth,
    gate,
    fuse_glu,
    fp4,
    [],
    "native",
    gate_act,
    gate_first,
  )


def _call_svdq_gemm_w4a4(
  act: torch.Tensor,
  wgt: torch.Tensor,
  out: torch.Tensor | None,
  qout: torch.Tensor | None,
  ascales: torch.Tensor | None,
  wscales: torch.Tensor | None,
  oscales: torch.Tensor | None,
  lora_act_in: torch.Tensor | None,
  lora_up: torch.Tensor | None,
  lora_down: torch.Tensor | None,
  lora_act_out: torch.Tensor | None,
  bias: torch.Tensor | None,
  smooth_factor: torch.Tensor | None,
  act_unsigned: bool,
  lora_scales: list[float] | None,
  fuse_silu: bool,
  fp4: bool,
  alpha: float,
  wcscales: torch.Tensor | None,
  runtime: str,
  lora_layout: str,
) -> None:
  """Bind the single SVDQ W4A4 CUDA GEMM ABI shared by the native and CuTe runtimes.

  Every argument is passed by keyword: the pybind entry has 21 positional parameters and a
  silent shift between two same-typed tensors (for example `ascales`/`wscales`) would still
  produce a valid-looking but wrong result.

  :param act: Packed activation tensor `[M, K / 2]`.
  :param wgt: Packed quantized weight tensor `[N, K / 2]`.
  :param out: Dense output buffer `[M, N]`, or None to skip the dense epilogue entirely (only
    valid on the native fused qout path).
  :param qout: Optional packed quantized output buffer `[M, N / 2]` for the next layer.
  :param ascales: Activation scales `[K / G, M]`, where `G` is 64 for INT4 and 16 for FP4.
  :param wscales: Weight scales `[K / G, N]`, where `G` is 64 for INT4 and 16 for FP4.
  :param oscales: Optional output scales `[N / G, M]` for `qout`.
  :param lora_act_in: Optional LoRA activation input `[M, R]`.
  :param lora_up: Optional LoRA up-projection weights `[N, R]`.
  :param lora_down: Optional LoRA down-projection weights `[N, R]` for the next fused layer.
  :param lora_act_out: Optional LoRA activation output buffer `[M, R]` for the next fused layer.
  :param bias: Optional dense output bias `[N]`.
  :param smooth_factor: Optional smoothing factors `[N]` written for next-layer quantization.
  :param act_unsigned: Whether INT4 activations are stored as unsigned values.
  :param lora_scales: Optional per-16-rank LoRA scaling factors `[R / 16]`.
  :param fuse_silu: Whether to fuse SiLU inside supported kernel variants.
  :param fp4: Whether the packed tensors use FP4/NVFP4 instead of INT4.
  :param alpha: Per-tensor FP4 scaling factor.
  :param wcscales: Optional per-channel FP4 scales `[N]`.
  :param runtime: Backend selector, either `'native'` or `'cute'`.
  :param lora_layout: LoRA-down layout, either `'native'` or `'row_major'`.

  :returns: None. Results are written in-place to the provided output tensors.
  """
  ops_module = _get_required_ops_module()
  ops_module.svdq_gemm_w4a4(
    act=act,
    wgt=wgt,
    out=out,
    qout=qout,
    ascales=ascales,
    wscales=wscales,
    oscales=oscales,
    lora_act_in=lora_act_in,
    lora_up=lora_up,
    lora_down=lora_down,
    lora_act_out=lora_act_out,
    bias=bias,
    smooth_factor=smooth_factor,
    act_unsigned=act_unsigned,
    lora_scales=_normalize_svdq_lora_scales(lora_scales, lora_up),
    fuse_silu=fuse_silu,
    fp4=fp4,
    alpha=float(alpha),
    wcscales=wcscales,
    runtime=runtime,
    lora_layout=lora_layout,
  )


def _call_svdq_quantize_w4a4(
  input: torch.Tensor,
  output: torch.Tensor,
  ascales: torch.Tensor,
  lora_down: torch.Tensor | None,
  lora_act: torch.Tensor,
  smooth: torch.Tensor | None,
  gate: torch.Tensor | None,
  fuse_glu: bool,
  fp4: bool,
  lora_scales: list[float],
  lora_layout: str,
  gate_act: str,
  gate_first: bool,
) -> None:
  """Bridge the unified native/CuTe activation producer through one raw op."""
  ops = _get_required_ops_module()
  ops.svdq_quantize_w4a4(
    input=input,
    output=output,
    ascales=ascales,
    lora_down=lora_down,
    lora_act=lora_act,
    smooth=smooth,
    gate=gate,
    fuse_glu=fuse_glu,
    fp4=fp4,
    lora_scales=lora_scales,
    lora_layout=lora_layout,
    gate_act=gate_act,
    gate_first=gate_first,
  )


torch.library.define(
  "cache_dit_cuda_ops::svdq_quantize_w4a4",
  "(Tensor input, Tensor? lora_down, Tensor? smooth, Tensor? gate, str lora_layout, bool fuse_glu, bool fp4, float[] lora_scales, int pad_size, str gate_act=\"silu\", bool gate_first=False) -> (Tensor output, Tensor oscales, Tensor lora_act)",
)


@torch.library.impl("cache_dit_cuda_ops::svdq_quantize_w4a4", "CUDA")
def _svdq_quantize_w4a4(
  input: torch.Tensor,
  lora_down: torch.Tensor | None,
  smooth: torch.Tensor | None,
  gate: torch.Tensor | None,
  lora_layout: str,
  fuse_glu: bool,
  fp4: bool,
  lora_scales: list[float],
  pad_size: int,
  gate_act: str = "silu",
  gate_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  batch_size, channels = input.shape
  batch_size_pad = ((batch_size + pad_size - 1) // pad_size) * pad_size
  rank = 0 if lora_down is None else lora_down.shape[1]
  # A gated activation halves the width: either the caller hands over a separate gate tensor of
  # the final width, or the second half of `input` supplies it.
  value_channels = channels
  if fuse_glu:
    if gate is None:
      if channels % 2 != 0:
        raise ValueError(f"fused GLU over a single tensor needs an even width, got {channels}.")
      value_channels = channels // 2
    else:
      if tuple(gate.shape) != (batch_size, channels):
        raise ValueError(f"fused GLU expects gate of shape {(batch_size, channels)}, got "
                         f"{tuple(gate.shape)}.")
      if not gate.is_contiguous():
        raise ValueError("fused GLU requires a contiguous gate tensor.")

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
      if gate is not None:
        # The CuTe producer walks every padded row, so a separate gate half has to be padded too.
        gate = torch.nn.functional.pad(gate, (0, 0, 0, batch_size_pad - batch_size))
    output = torch.empty((batch_size_pad, value_channels // 2),
                         dtype=torch.uint8,
                         device=input.device)
    oscales = torch.empty(batch_size_pad * value_channels // 16,
                          dtype=torch.float8_e4m3fn,
                          device=input.device)
    lora_act = torch.empty((batch_size_pad, rank), dtype=torch.bfloat16, device=input.device)
    _call_svdq_quantize_w4a4(padded_input, output, oscales, lora_down, lora_act, smooth, gate,
                             fuse_glu, True, lora_scales, "row_major", gate_act, gate_first)
    return output, oscales, lora_act

  output = torch.empty(batch_size_pad, value_channels // 2, dtype=torch.uint8, device=input.device)
  if fp4:
    if value_channels % 16 != 0:
      raise ValueError(f"Expected channels divisible by 16 for FP4 quantization, got "
                       f"{value_channels}.")
    oscales = torch.empty(value_channels // 16,
                          batch_size_pad,
                          dtype=torch.float8_e4m3fn,
                          device=input.device)
  else:
    if value_channels % 64 != 0:
      raise ValueError(f"Expected channels divisible by 64 for INT4 quantization, got "
                       f"{value_channels}.")
    oscales = torch.empty(value_channels // 64,
                          batch_size_pad,
                          dtype=input.dtype,
                          device=input.device)
  lora_act_out = torch.empty(batch_size_pad, rank, dtype=torch.float32, device=input.device)

  _call_svdq_quantize_w4a4_native(
    input,
    output,
    oscales,
    lora_down,
    lora_act_out,
    smooth,
    gate,
    fuse_glu,
    fp4,
    gate_act,
    gate_first,
  )
  return output, oscales, lora_act_out


@torch.library.register_fake("cache_dit_cuda_ops::svdq_quantize_w4a4")
def _fake_svdq_quantize_w4a4(
  input: torch.Tensor,
  lora_down: torch.Tensor | None,
  smooth: torch.Tensor | None,
  gate: torch.Tensor | None,
  lora_layout: str,
  fuse_glu: bool,
  fp4: bool,
  lora_scales: list[float],
  pad_size: int,
  gate_act: str = "silu",
  gate_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  del smooth, lora_scales, gate_act, gate_first
  batch_size, channels = input.shape
  batch_size_pad = ((batch_size + pad_size - 1) // pad_size) * pad_size
  rank = 0 if lora_down is None else lora_down.shape[1]
  value_channels = channels if (fuse_glu
                                and gate is not None) else channels // (2 if fuse_glu else 1)
  output = input.new_empty((batch_size_pad, value_channels // 2), dtype=torch.uint8)
  if _normalize_svdq_lora_layout(lora_layout) == "row_major":
    oscales = input.new_empty((batch_size_pad * value_channels // 16, ), dtype=torch.float8_e4m3fn)
    lora_act = input.new_empty((batch_size_pad, rank), dtype=torch.bfloat16)
    return output, oscales, lora_act
  oscales = input.new_empty(
    (value_channels // (16 if fp4 else 64), batch_size_pad),
    dtype=torch.float8_e4m3fn if fp4 else input.dtype,
  )
  lora_act_out = input.new_empty((batch_size_pad, rank), dtype=torch.float32)
  return output, oscales, lora_act_out


torch.library.define(
  "cache_dit_cuda_ops::svdq_gemm_w4a4",
  "(Tensor act, Tensor wgt, Tensor ascales, Tensor wscales, Tensor? qout, Tensor? oscales, "
  "Tensor? lora_act_in, Tensor? lora_up, Tensor? lora_down, Tensor? lora_act_out, "
  "Tensor? bias, Tensor? smooth_factor, Tensor? wcscales, bool act_unsigned, bool fuse_silu, "
  "bool fp4, float alpha, float[] lora_scales, str runtime, str lora_layout, int out_dtype_id, "
  "bool write_dense=True) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::svdq_gemm_w4a4", "CUDA")
def _svdq_gemm_w4a4(
  act: torch.Tensor,
  wgt: torch.Tensor,
  ascales: torch.Tensor,
  wscales: torch.Tensor,
  qout: torch.Tensor | None,
  oscales: torch.Tensor | None,
  lora_act_in: torch.Tensor | None,
  lora_up: torch.Tensor | None,
  lora_down: torch.Tensor | None,
  lora_act_out: torch.Tensor | None,
  bias: torch.Tensor | None,
  smooth_factor: torch.Tensor | None,
  wcscales: torch.Tensor | None,
  act_unsigned: bool,
  fuse_silu: bool,
  fp4: bool,
  alpha: float,
  lora_scales: list[float],
  runtime: str,
  lora_layout: str,
  out_dtype_id: int,
  write_dense: bool = True,
) -> torch.Tensor:
  if _normalize_svdq_runtime(runtime) == "cute":
    # The CuTe fused epilogue can skip the dense store, but only when it emits the packed
    # activation instead; without qout there would be nothing to return at all.
    if not write_dense:
      output = None
    else:
      output = torch.empty((act.shape[0], wgt.shape[0]), dtype=torch.bfloat16, device=act.device)
    _call_svdq_gemm_w4a4(
      act=act,
      wgt=wgt,
      out=output,
      qout=qout,
      ascales=ascales,
      wscales=wscales,
      oscales=oscales,
      lora_act_in=lora_act_in,
      lora_up=lora_up,
      lora_down=lora_down,
      lora_act_out=lora_act_out,
      bias=bias,
      smooth_factor=smooth_factor,
      act_unsigned=act_unsigned,
      lora_scales=lora_scales,
      fuse_silu=fuse_silu,
      fp4=fp4,
      alpha=alpha,
      wcscales=wcscales,
      runtime="cute",
      lora_layout=lora_layout,
    )
    if output is None:
      return act.new_empty((0, wgt.shape[0]))
    return output

  if _normalize_svdq_lora_layout(lora_layout) != "native":
    raise ValueError("runtime='native' only supports lora_layout='native'.")

  if not write_dense:
    # The kernel only skips the dense `EpilogueDefault` store when it takes the fused qout
    # branch, and that branch dereferences `smooth_factor` unconditionally. The matching C++
    # checks are `assert`s, which stay enabled in the nvcc translation units (`-DNDEBUG` only
    # reaches the host flags), so a bad combination aborts instead of failing gracefully.
    if qout is None or oscales is None:
      raise ValueError("write_dense=False requires the fused qout path: pass both qout and "
                       "oscales, otherwise the kernel has no output to write.")
    if smooth_factor is None:
      raise ValueError("write_dense=False requires smooth_factor; the fused qout epilogue "
                       "consumes it unconditionally.")
    if fp4 and all(t is None for t in (bias, lora_up, lora_down, wcscales)):
      raise ValueError("write_dense=False on the NVFP4 path needs one of "
                       "bias/lora_up/lora_down/wcscales: they are the only dtype sources left "
                       "once the dense output buffer is gone.")
    output = None
  else:
    output = torch.empty((act.shape[0], wgt.shape[0]),
                         dtype=_decode_svdq_output_dtype(out_dtype_id),
                         device=act.device)

  _call_svdq_gemm_w4a4(
    act=act,
    wgt=wgt,
    out=output,
    qout=qout,
    ascales=ascales,
    wscales=wscales,
    oscales=oscales,
    lora_act_in=lora_act_in,
    lora_up=lora_up,
    lora_down=lora_down,
    lora_act_out=lora_act_out,
    bias=bias,
    smooth_factor=smooth_factor,
    act_unsigned=act_unsigned,
    lora_scales=lora_scales,
    fuse_silu=fuse_silu,
    fp4=fp4,
    alpha=alpha,
    wcscales=wcscales,
    runtime="native",
    lora_layout="native",
  )
  if output is None:
    # Placeholder only: its dtype and shape carry no meaning beyond "the kernel produced no
    # dense output". `write_dense=False` callers must not read it.
    return act.new_empty((0, wgt.shape[0]))
  return output


@torch.library.register_fake("cache_dit_cuda_ops::svdq_gemm_w4a4")
def _fake_svdq_gemm_w4a4(
  act: torch.Tensor,
  wgt: torch.Tensor,
  ascales: torch.Tensor,
  wscales: torch.Tensor,
  qout: torch.Tensor | None,
  oscales: torch.Tensor | None,
  lora_act_in: torch.Tensor | None,
  lora_up: torch.Tensor | None,
  lora_down: torch.Tensor | None,
  lora_act_out: torch.Tensor | None,
  bias: torch.Tensor | None,
  smooth_factor: torch.Tensor | None,
  wcscales: torch.Tensor | None,
  act_unsigned: bool,
  fuse_silu: bool,
  fp4: bool,
  alpha: float,
  lora_scales: list[float],
  runtime: str,
  lora_layout: str,
  out_dtype_id: int,
  write_dense: bool = True,
) -> torch.Tensor:
  del ascales, wscales, qout, oscales, lora_act_in, lora_up, lora_down, lora_act_out, bias
  del smooth_factor, wcscales, act_unsigned, fuse_silu, fp4, alpha, lora_scales
  if _normalize_svdq_runtime(runtime) == "cute":
    _normalize_svdq_lora_layout(lora_layout)
    if not write_dense:
      return act.new_empty((0, wgt.shape[0]))
    return act.new_empty((act.shape[0], wgt.shape[0]), dtype=torch.bfloat16)
  if not write_dense:
    return act.new_empty((0, wgt.shape[0]))
  return act.new_empty((act.shape[0], wgt.shape[0]), dtype=_decode_svdq_output_dtype(out_dtype_id))


def svdq_quantize_w4a4(
  input: torch.Tensor,
  lora_down: torch.Tensor | None = None,
  smooth: torch.Tensor | None = None,
  lora_layout: str = "native",
  fp4: bool = True,
  fuse_glu: bool = False,
  gate: torch.Tensor | None = None,
  lora_scales: list[float] | None = None,
  pad_size: int = 256,
  gate_act: str = "silu",
  gate_first: bool = False,
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
  :param fuse_glu: Whether to gate the activations as `value * act(gate)` before quantizing.
  :param gate: Optional `[tokens, in_features]` gating half of a fused GLU. When omitted, one half
    of `input` supplies it and `input` must be twice as wide as the quantized output.
  :param lora_scales: Optional FP32 scale per 16-rank group, consumed by `'row_major'` and
    defaulting to ones.
  :param pad_size: Token padding multiple required by the packed activation layout.
  :param gate_act: Gate activation, one of `'silu'`, `'gelu_tanh'`, or exact `'gelu'`.
  :param gate_first: Interpret packed `input` as `[gate | value]` instead of `[value | gate]`.
    Only valid with `fuse_glu=True` and no separate `gate` tensor.
  :returns: `(packed_activations, activation_scales, lora_down_output)`.
  :raises ValueError: If `lora_layout` is unknown, or if `'row_major'` is asked for an INT4 pack.
  """
  if gate_act not in ("silu", "gelu_tanh", "gelu"):
    raise ValueError(f"gate_act must be 'silu', 'gelu_tanh', or 'gelu', got {gate_act!r}.")
  if gate_first and (not fuse_glu or gate is not None):
    raise ValueError("gate_first=True is only valid with fuse_glu=True and gate=None.")
  normalized_layout = _normalize_svdq_lora_layout(lora_layout)
  if normalized_layout == "row_major" and not fp4:
    raise ValueError("lora_layout='row_major' implements the NVFP4 producer only; pass fp4=True.")
  rank = 0 if lora_down is None else lora_down.shape[1]
  normalized_scales = lora_scales
  if normalized_scales is None:
    normalized_scales = [1.0] * max(1, rank // 16)
  return torch.ops.cache_dit_cuda_ops.svdq_quantize_w4a4(input, lora_down, smooth, gate,
                                                         normalized_layout, fuse_glu, fp4,
                                                         normalized_scales, pad_size, gate_act,
                                                         gate_first)


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
  lora_scales: list[float] | None = None,
  qout: torch.Tensor | None = None,
  oscales: torch.Tensor | None = None,
  lora_down: torch.Tensor | None = None,
  lora_act_out: torch.Tensor | None = None,
  smooth_factor: torch.Tensor | None = None,
  fuse_silu: bool = False,
  write_dense: bool = True,
) -> torch.Tensor:
  """Run the SVDQ W4A4 GEMM on the selected runtime with an optional LoRA-up correction.

  `runtime` picks the kernel family: `'native'` runs the single native W4A4 GEMM for both INT4 and
  NVFP4, `'cute'` runs the SM120 CuTe NVFP4 GEMM. `lora_layout` picks how the LoRA-down term is
  delivered and must match the producer that filled `lora_act_in`:

  Only three combinations are valid:
  `runtime='native'` + `lora_layout='native'` (packed FP32 fragment D),
  `runtime='cute'` + `lora_layout='native'` (packed FP32 fragment D),
  and `runtime='cute'` + `lora_layout='row_major'` (row-major BF16 D with the scales folded in).

  The trailing `qout`/`oscales`/`lora_down`/`lora_act_out`/`smooth_factor`/`fuse_silu` arguments
  drive the fused-MLP epilogue chain on both runtimes: the activated tile is quantized into the
  native 4-bit activation pack of the next layer and the next layer's LoRA-down term is reduced in
  the same kernel. `smooth_factor` is the next layer's row-major smoothing vector (not the packed
  native one), and `lora_down` is its row-major `[N, R_down]` down-projection.

  `write_dense=False` drops the dense output entirely: the kernel skips its store and only writes
  `qout`/`oscales`/`lora_act_out`, which is what a fused-MLP pass needs. It requires `qout`,
  `oscales` and `smooth_factor`, and the return value becomes a 0-element placeholder whose dtype
  and shape carry no meaning.

  `lora_act_out` follows `lora_layout`: the native layout receives the packed FP32 fragment D in
  place, while the row-major layout receives BF16 and is reduced through an internal FP32 scratch.

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
  :param lora_scales: Optional FP32 scale per 16-rank group, consumed by
    `lora_layout='native'` and defaulting to ones.
  :param qout: Optional packed quantized output buffer `[M, N / 2]` for the next layer.
  :param oscales: Optional output scales `[N / G, M]` written together with `qout`.
  :param lora_down: Optional LoRA down-projection weight `[N, R]` of the next fused layer.
  :param lora_act_out: Optional LoRA activation output buffer `[M, R]` of the next fused layer.
  :param smooth_factor: Optional smoothing factors `[N]` written for the next-layer quantizer.
  :param fuse_silu: Whether to fuse SiLU inside the native kernel.
  :param write_dense: Whether the fused qout path should also write the dense `[M, N]` activation.
    Pass `False` when the caller only consumes `qout`/`oscales`/`lora_act_out` (for example
    `fused_gelu_mlp`), which removes a full-size HBM store per call.
  :returns: Dense output `[M, N]`, or a 0-element placeholder when `write_dense=False`.
  :raises ValueError: If `runtime`/`lora_layout` is unknown, if `runtime='native'` is paired with
    `lora_layout='row_major'`, if the CuTe runtime is asked for INT4 or a non-BF16 output, if
    per-rank `lora_scales` are passed for row-major D, or if `write_dense=False` is requested
    without the native fused qout arguments.
  """
  normalized_runtime = _normalize_svdq_runtime(runtime)
  normalized_layout = _normalize_svdq_lora_layout(lora_layout)
  fused = qout is not None or fuse_silu
  if normalized_runtime == "native" and normalized_layout != "native":
    raise ValueError("runtime='native' only supports lora_layout='native'; use runtime='cute' for "
                     "row-major BF16 D.")
  if normalized_runtime == "cute":
    if not fp4:
      raise ValueError("runtime='cute' implements the NVFP4 path only; pass fp4=True.")
    if output_dtype is not None and output_dtype != torch.bfloat16:
      raise ValueError(f"runtime='cute' always produces BF16 output, got {output_dtype}.")
    if normalized_layout == "row_major" and lora_scales is not None and not fused:
      raise ValueError("lora_layout='row_major' expects the per-rank scales folded into D; drop "
                       "lora_scales unless the fused epilogue needs them to fold the next layer.")
  if fuse_silu and qout is not None:
    raise ValueError("fuse_silu is the dense-only activation; combine qout with the GELU epilogue "
                     "instead of asking for both.")
  if not write_dense:
    if qout is None:
      raise ValueError("write_dense=False requires qout; the fused epilogue is the only thing that "
                       "still produces an output once the dense store is skipped.")
    if oscales is None or smooth_factor is None:
      raise ValueError("write_dense=False requires qout, oscales and smooth_factor; without them "
                       "the kernel has no fused output to write.")
    if normalized_runtime == "native" and fp4 and all(
        t is None for t in (bias, lora_up, lora_down, wcscales)):
      raise ValueError("write_dense=False on the NVFP4 path needs one of "
                       "bias/lora_up/lora_down/wcscales: they are the only dtype sources left "
                       "once the dense output buffer is gone.")
  if alpha is None:
    alpha = 1.0
  if output_dtype is None:
    if not write_dense:
      # `out_dtype_id` only sizes the dense buffer, which is skipped here, so a neutral encoding
      # avoids failing the inference for rank-0/no-bias callers.
      output_dtype = torch.bfloat16
    else:
      # The CuTe runtime always writes BF16, so it needs no dtype hint from the auxiliary tensors.
      output_dtype = (torch.bfloat16 if normalized_runtime == "cute" else _infer_svdq_output_dtype(
        None, lora_up, bias, wscales))
      if output_dtype is None:
        raise ValueError("Unable to infer the output dtype for svdq_gemm_w4a4.")
  return torch.ops.cache_dit_cuda_ops.svdq_gemm_w4a4(
    act=act,
    wgt=wgt,
    ascales=ascales,
    wscales=wscales,
    qout=qout,
    oscales=oscales,
    lora_act_in=lora_act_in,
    lora_up=lora_up,
    lora_down=lora_down,
    lora_act_out=lora_act_out,
    bias=bias,
    smooth_factor=smooth_factor,
    wcscales=wcscales,
    act_unsigned=act_unsigned,
    fuse_silu=fuse_silu,
    fp4=fp4,
    alpha=float(alpha),
    lora_scales=_normalize_svdq_lora_scales(lora_scales, lora_up),
    runtime=normalized_runtime,
    lora_layout=normalized_layout,
    out_dtype_id=_encode_svdq_output_dtype(output_dtype),
    write_dense=write_dense,
  )


_SUPPORTED_W8A8_RANKS = (0, 32, 64, 128)


def _call_svdq_gemm_w8a8(act: torch.Tensor, wgt: torch.Tensor, out: torch.Tensor,
                         ascales: torch.Tensor, wscales: torch.Tensor,
                         lora_act_in: torch.Tensor | None, lora_up: torch.Tensor | None,
                         bias: torch.Tensor | None) -> None:
  """Invoke the pybind W8A8 GEMM with the caller-allocated output buffer."""
  ops = _get_required_ops_module()
  ops.svdq_gemm_w8a8(act=act,
                     wgt=wgt,
                     out=out,
                     ascales=ascales,
                     wscales=wscales,
                     lora_act_in=lora_act_in,
                     lora_up=lora_up,
                     bias=bias)


torch.library.define(
  "cache_dit_cuda_ops::svdq_gemm_w8a8",
  "(Tensor act, Tensor wgt, Tensor ascales, Tensor wscales, Tensor? lora_act_in, Tensor? lora_up, "
  "Tensor? bias) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::svdq_gemm_w8a8", "CUDA")
def _svdq_gemm_w8a8(act: torch.Tensor, wgt: torch.Tensor, ascales: torch.Tensor,
                    wscales: torch.Tensor, lora_act_in: torch.Tensor | None,
                    lora_up: torch.Tensor | None, bias: torch.Tensor | None) -> torch.Tensor:
  m, _ = act.shape
  n = wgt.shape[0]
  out = torch.empty((m, n), dtype=torch.bfloat16, device=act.device)
  _call_svdq_gemm_w8a8(act, wgt, out, ascales, wscales, lora_act_in, lora_up, bias)
  return out


@torch.library.register_fake("cache_dit_cuda_ops::svdq_gemm_w8a8")
def _fake_svdq_gemm_w8a8(act: torch.Tensor, wgt: torch.Tensor, ascales: torch.Tensor,
                         wscales: torch.Tensor, lora_act_in: torch.Tensor | None,
                         lora_up: torch.Tensor | None, bias: torch.Tensor | None) -> torch.Tensor:
  del ascales, wscales, lora_act_in, lora_up, bias
  return act.new_empty((act.shape[0], wgt.shape[0]), dtype=torch.bfloat16)


def svdq_gemm_w8a8(act: torch.Tensor,
                   wgt: torch.Tensor,
                   ascales: torch.Tensor,
                   wscales: torch.Tensor,
                   lora_act_in: torch.Tensor | None = None,
                   lora_up: torch.Tensor | None = None,
                   bias: torch.Tensor | None = None) -> torch.Tensor:
  """Compute the fused SVDQ-FP8 linear from E4M3 operands and a BF16 low-rank branch.

  :param act: E4M3 activation codes ``[M,K]``, contiguous.
  :param wgt: E4M3 residual codes ``[N,K]``, contiguous.
  :param ascales: FP32 per-token activation scales ``[M]``.
  :param wscales: FP32 per-output-channel residual scales ``[N]``.
  :param lora_act_in: Optional BF16 LoRA-down output ``[M,R]``.
  :param lora_up: Optional BF16 LoRA-up weight ``[N,R]``.
  :param bias: Optional BF16 bias ``[N]``.
  :returns: BF16 output ``[M,N]``.
  """
  if act.dtype != torch.float8_e4m3fn or wgt.dtype != torch.float8_e4m3fn:
    raise ValueError("svdq_gemm_w8a8 expects float8_e4m3fn activation and weight codes.")
  if act.dim() != 2 or wgt.dim() != 2:
    raise ValueError("svdq_gemm_w8a8 expects 2D activation and weight tensors.")
  if not act.is_contiguous() or not wgt.is_contiguous():
    raise ValueError("svdq_gemm_w8a8 expects contiguous activation and weight tensors.")
  m, k = act.shape
  n = wgt.shape[0]
  if wgt.shape[1] != k:
    raise ValueError("svdq_gemm_w8a8 inner dimensions do not match.")
  if m % 128 != 0 or n % 128 != 0 or k % 128 != 0:
    raise ValueError(f"svdq_gemm_w8a8 requires M/N/K multiples of 128, got {(m, n, k)}.")
  if ascales.dtype != torch.float32 or wscales.dtype != torch.float32:
    raise ValueError("svdq_gemm_w8a8 expects FP32 scales.")
  if ascales.numel() != m or wscales.numel() != n:
    raise ValueError("svdq_gemm_w8a8 scale shapes do not match the operands.")
  if (lora_act_in is None) != (lora_up is None):
    raise ValueError("svdq_gemm_w8a8 requires both LoRA tensors or neither.")
  if lora_up is not None:
    rank = lora_up.shape[1]
    if rank not in _SUPPORTED_W8A8_RANKS or rank == 0:
      raise ValueError(f"svdq_gemm_w8a8 supports rank 32/64/128, got {rank}.")
    if lora_act_in.dtype != torch.bfloat16 or lora_up.dtype != torch.bfloat16:
      raise ValueError("svdq_gemm_w8a8 expects BF16 LoRA tensors.")
    if tuple(lora_act_in.shape) != (m, rank) or tuple(lora_up.shape) != (n, rank):
      raise ValueError("svdq_gemm_w8a8 LoRA shapes do not match the operands.")
    if not lora_act_in.is_contiguous() or not lora_up.is_contiguous():
      raise ValueError("svdq_gemm_w8a8 expects contiguous LoRA tensors.")
  if bias is not None and (bias.dtype != torch.bfloat16 or bias.numel() != n):
    raise ValueError("svdq_gemm_w8a8 expects a BF16 bias of length N.")
  return torch.ops.cache_dit_cuda_ops.svdq_gemm_w8a8(act=act,
                                                     wgt=wgt,
                                                     ascales=ascales,
                                                     wscales=wscales,
                                                     lora_act_in=lora_act_in,
                                                     lora_up=lora_up,
                                                     bias=bias)


def _call_svdq_quantize_w8a8(input: torch.Tensor, output: torch.Tensor, ascales: torch.Tensor,
                             lora_down: torch.Tensor | None, lora_act: torch.Tensor | None,
                             smooth: torch.Tensor) -> None:
  """Invoke the pybind fused producer with caller-allocated outputs."""
  ops = _get_required_ops_module()
  ops.svdq_quantize_w8a8(input=input,
                         output=output,
                         ascales=ascales,
                         lora_down=lora_down,
                         lora_act=lora_act,
                         smooth=smooth)


torch.library.define(
  "cache_dit_cuda_ops::svdq_quantize_w8a8",
  "(Tensor input, Tensor smooth, Tensor? lora_down) -> (Tensor output, Tensor ascales, "
  "Tensor lora_act)",
)


def _w8a8_padded_rows(rows: int) -> int:
  return (rows + 127) // 128 * 128


@torch.library.impl("cache_dit_cuda_ops::svdq_quantize_w8a8", "CUDA")
def _svdq_quantize_w8a8(
    input: torch.Tensor, smooth: torch.Tensor,
    lora_down: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  rows, channels = input.shape
  rows_pad = _w8a8_padded_rows(rows)
  rank = 0 if lora_down is None else lora_down.shape[0]
  padded = input.contiguous()
  if rows_pad != rows:
    padded = torch.nn.functional.pad(padded, (0, 0, 0, rows_pad - rows))
  output = torch.empty((rows_pad, channels), dtype=torch.float8_e4m3fn, device=input.device)
  ascales = torch.empty(rows_pad, dtype=torch.float32, device=input.device)
  lora_act = torch.empty((rows_pad, rank), dtype=torch.bfloat16, device=input.device)
  _call_svdq_quantize_w8a8(padded, output, ascales, lora_down, lora_act if rank else None, smooth)
  return output, ascales, lora_act


@torch.library.register_fake("cache_dit_cuda_ops::svdq_quantize_w8a8")
def _fake_svdq_quantize_w8a8(
    input: torch.Tensor, smooth: torch.Tensor,
    lora_down: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  del smooth
  rows, channels = input.shape
  rows_pad = _w8a8_padded_rows(rows)
  rank = 0 if lora_down is None else lora_down.shape[0]
  return (input.new_empty((rows_pad, channels),
                          dtype=torch.float8_e4m3fn), input.new_empty(rows_pad,
                                                                      dtype=torch.float32),
          input.new_empty((rows_pad, rank), dtype=torch.bfloat16))


def svdq_quantize_w8a8(
    input: torch.Tensor,
    smooth: torch.Tensor,
    lora_down: torch.Tensor | None = None,
    pad_rows: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
  """Quantize an activation and compute the LoRA-down term in one fused kernel.

  :param input: BF16 activation ``[M,K]``.
  :param smooth: BF16 per-input-channel divisor ``[K]``.
  :param lora_down: Optional BF16 ``[R,K]`` down projection, already divided by ``smooth``.
  :param pad_rows: If `True`, return the 128-aligned padded outputs required by `svdq_gemm_w8a8`;
    otherwise slice rows back to M. Use `True` when composing producer and GEMM with ragged M.
  :returns: ``(qact e4m3, ascales fp32, lora_act bf16 or None)`` with M or padded rows.
  """
  if input.dtype != torch.bfloat16 or input.dim() != 2:
    raise ValueError("svdq_quantize_w8a8 expects a 2D BF16 activation.")
  if smooth.dtype != torch.bfloat16 or smooth.numel() != input.shape[1]:
    raise ValueError("svdq_quantize_w8a8 expects a BF16 smooth vector of length K.")
  if input.shape[1] % 64 != 0:
    raise ValueError(f"svdq_quantize_w8a8 requires K divisible by 64, got {input.shape[1]}.")
  if lora_down is not None:
    if lora_down.dtype != torch.bfloat16 or lora_down.dim() != 2:
      raise ValueError("svdq_quantize_w8a8 expects a 2D BF16 lora_down.")
    if lora_down.shape[0] not in _SUPPORTED_W8A8_RANKS or lora_down.shape[0] == 0:
      raise ValueError(f"svdq_quantize_w8a8 supports rank 32/64/128, got {lora_down.shape[0]}.")
    if lora_down.shape[1] != input.shape[1] or not lora_down.is_contiguous():
      raise ValueError("svdq_quantize_w8a8 expects a contiguous lora_down of shape [R,K].")
  rows = input.shape[0]
  if rows == 0:
    rank = 0 if lora_down is None else lora_down.shape[0]
    return (input.new_empty((0, input.shape[1]), dtype=torch.float8_e4m3fn),
            input.new_empty(0, dtype=torch.float32), input.new_empty(
              (0, rank), dtype=torch.bfloat16) if rank else None)
  output, ascales, lora_act = torch.ops.cache_dit_cuda_ops.svdq_quantize_w8a8(input=input,
                                                                              smooth=smooth,
                                                                              lora_down=lora_down)
  if pad_rows:
    return output, ascales, (lora_act if lora_down is not None else None)
  return output[:rows], ascales[:rows], (lora_act[:rows] if lora_down is not None else None)


def _call_svdq_gemm_w4a8(act: torch.Tensor, wgt: torch.Tensor, out: torch.Tensor,
                         ascales: torch.Tensor, wscales: torch.Tensor,
                         lora_act_in: torch.Tensor | None, lora_up: torch.Tensor | None,
                         bias: torch.Tensor | None) -> None:
  """Invoke the pybind W4A8 GEMM with a caller-allocated BF16 output."""
  ops = _get_required_ops_module()
  ops.svdq_gemm_w4a8(act=act,
                     wgt=wgt,
                     out=out,
                     ascales=ascales,
                     wscales=wscales,
                     lora_act_in=lora_act_in,
                     lora_up=lora_up,
                     bias=bias)


torch.library.define(
  "cache_dit_cuda_ops::svdq_gemm_w4a8",
  "(Tensor act, Tensor wgt, Tensor ascales, Tensor wscales, Tensor? lora_act_in, Tensor? lora_up, "
  "Tensor? bias) -> Tensor",
)


@torch.library.impl("cache_dit_cuda_ops::svdq_gemm_w4a8", "CUDA")
def _svdq_gemm_w4a8(act: torch.Tensor, wgt: torch.Tensor, ascales: torch.Tensor,
                    wscales: torch.Tensor, lora_act_in: torch.Tensor | None,
                    lora_up: torch.Tensor | None, bias: torch.Tensor | None) -> torch.Tensor:
  out = torch.empty((act.shape[0], wgt.shape[0]), dtype=torch.bfloat16, device=act.device)
  _call_svdq_gemm_w4a8(act, wgt, out, ascales, wscales, lora_act_in, lora_up, bias)
  return out


@torch.library.register_fake("cache_dit_cuda_ops::svdq_gemm_w4a8")
def _fake_svdq_gemm_w4a8(act: torch.Tensor, wgt: torch.Tensor, ascales: torch.Tensor,
                         wscales: torch.Tensor, lora_act_in: torch.Tensor | None,
                         lora_up: torch.Tensor | None, bias: torch.Tensor | None) -> torch.Tensor:
  del ascales, wscales, lora_act_in, lora_up, bias
  return act.new_empty((act.shape[0], wgt.shape[0]), dtype=torch.bfloat16)


def svdq_gemm_w4a8(act: torch.Tensor,
                   wgt: torch.Tensor,
                   ascales: torch.Tensor,
                   wscales: torch.Tensor,
                   lora_act_in: torch.Tensor | None = None,
                   lora_up: torch.Tensor | None = None,
                   bias: torch.Tensor | None = None) -> torch.Tensor:
  """Compute a fused W4A8 linear from E4M3 activation and packed E2M1 residual codes.

  :param act: E4M3 activation codes ``[M,K]``, contiguous.
  :param wgt: Packed E2M1 codes ``[N,K/2]`` uint8, even-k in the low nibble.
  :param ascales: FP32 per-token activation scales ``[M]``.
  :param wscales: FP32 per-output-channel residual scales ``[N]``.
  :param lora_act_in: Optional BF16 LoRA-down output ``[M,R]``.
  :param lora_up: Optional BF16 LoRA-up weight ``[N,R]``.
  :param bias: Optional BF16 bias ``[N]``.
  :returns: BF16 output ``[M,N]``.
  """
  if act.dtype != torch.float8_e4m3fn or wgt.dtype != torch.uint8:
    raise ValueError("svdq_gemm_w4a8 expects E4M3 activation and uint8 packed E2M1 weights.")
  if act.dim() != 2 or wgt.dim() != 2:
    raise ValueError("svdq_gemm_w4a8 expects 2D activation and packed weight tensors.")
  if not act.is_contiguous() or not wgt.is_contiguous():
    raise ValueError("svdq_gemm_w4a8 expects contiguous activation and weight tensors.")
  m, k = act.shape
  n = wgt.shape[0]
  if wgt.shape[1] * 2 != k:
    raise ValueError("svdq_gemm_w4a8 requires wgt.shape == [N,K/2].")
  if m % 128 != 0 or n % 128 != 0 or k % 128 != 0:
    raise ValueError(f"svdq_gemm_w4a8 requires M/N/K multiples of 128, got {(m, n, k)}.")
  if ascales.dtype != torch.float32 or wscales.dtype != torch.float32:
    raise ValueError("svdq_gemm_w4a8 expects FP32 scales.")
  if ascales.numel() != m or wscales.numel() != n:
    raise ValueError("svdq_gemm_w4a8 scale shapes do not match the operands.")
  if (lora_act_in is None) != (lora_up is None):
    raise ValueError("svdq_gemm_w4a8 requires both LoRA tensors or neither.")
  if lora_up is not None:
    rank = lora_up.shape[1]
    if rank not in _SUPPORTED_W8A8_RANKS or rank == 0:
      raise ValueError(f"svdq_gemm_w4a8 supports rank 32/64/128, got {rank}.")
    if lora_act_in.dtype != torch.bfloat16 or lora_up.dtype != torch.bfloat16:
      raise ValueError("svdq_gemm_w4a8 expects BF16 LoRA tensors.")
    if tuple(lora_act_in.shape) != (m, rank) or tuple(lora_up.shape) != (n, rank):
      raise ValueError("svdq_gemm_w4a8 LoRA shapes do not match the operands.")
    if not lora_act_in.is_contiguous() or not lora_up.is_contiguous():
      raise ValueError("svdq_gemm_w4a8 expects contiguous LoRA tensors.")
  if bias is not None and (bias.dtype != torch.bfloat16 or bias.numel() != n):
    raise ValueError("svdq_gemm_w4a8 expects a BF16 bias of length N.")
  return torch.ops.cache_dit_cuda_ops.svdq_gemm_w4a8(act=act,
                                                     wgt=wgt,
                                                     ascales=ascales,
                                                     wscales=wscales,
                                                     lora_act_in=lora_act_in,
                                                     lora_up=lora_up,
                                                     bias=bias)


def _call_svdq_quantize_w4a8(input: torch.Tensor, output: torch.Tensor, ascales: torch.Tensor,
                             lora_down: torch.Tensor | None, lora_act: torch.Tensor | None,
                             smooth: torch.Tensor) -> None:
  """Invoke the W4A8 alias of the shared FP8 activation and LoRA-down producer."""
  ops = _get_required_ops_module()
  ops.svdq_quantize_w4a8(input=input,
                         output=output,
                         ascales=ascales,
                         lora_down=lora_down,
                         lora_act=lora_act,
                         smooth=smooth)


torch.library.define(
  "cache_dit_cuda_ops::svdq_quantize_w4a8",
  "(Tensor input, Tensor smooth, Tensor? lora_down) -> (Tensor output, Tensor ascales, "
  "Tensor lora_act)",
)


@torch.library.impl("cache_dit_cuda_ops::svdq_quantize_w4a8", "CUDA")
def _svdq_quantize_w4a8(
    input: torch.Tensor, smooth: torch.Tensor,
    lora_down: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  rows, channels = input.shape
  rows_pad = _w8a8_padded_rows(rows)
  rank = 0 if lora_down is None else lora_down.shape[0]
  padded = input.contiguous()
  if rows_pad != rows:
    padded = torch.nn.functional.pad(padded, (0, 0, 0, rows_pad - rows))
  output = torch.empty((rows_pad, channels), dtype=torch.float8_e4m3fn, device=input.device)
  ascales = torch.empty(rows_pad, dtype=torch.float32, device=input.device)
  lora_act = torch.empty((rows_pad, rank), dtype=torch.bfloat16, device=input.device)
  _call_svdq_quantize_w4a8(padded, output, ascales, lora_down, lora_act if rank else None, smooth)
  return output, ascales, lora_act


@torch.library.register_fake("cache_dit_cuda_ops::svdq_quantize_w4a8")
def _fake_svdq_quantize_w4a8(
    input: torch.Tensor, smooth: torch.Tensor,
    lora_down: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  return _fake_svdq_quantize_w8a8(input, smooth, lora_down)


def svdq_quantize_w4a8(
    input: torch.Tensor,
    smooth: torch.Tensor,
    lora_down: torch.Tensor | None = None,
    pad_rows: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
  """Produce the W4A8 activation and LoRA-down tensors using the shared W8A8 producer.

  :param input: BF16 activation ``[M,K]``.
  :param smooth: BF16 per-input-channel divisor ``[K]``.
  :param lora_down: Optional BF16 ``[R,K]`` down projection, already divided by ``smooth``.
  :param pad_rows: Return 128-aligned outputs for the GEMM instead of slicing back to M.
  :returns: ``(qact e4m3, ascales fp32, lora_act bf16 or None)``.
  """
  if input.dtype != torch.bfloat16 or input.dim() != 2:
    raise ValueError("svdq_quantize_w4a8 expects a 2D BF16 activation.")
  if smooth.dtype != torch.bfloat16 or smooth.numel() != input.shape[1]:
    raise ValueError("svdq_quantize_w4a8 expects a BF16 smooth vector of length K.")
  if input.shape[1] % 64 != 0:
    raise ValueError(f"svdq_quantize_w4a8 requires K divisible by 64, got {input.shape[1]}.")
  if lora_down is not None:
    if lora_down.dtype != torch.bfloat16 or lora_down.dim() != 2:
      raise ValueError("svdq_quantize_w4a8 expects a 2D BF16 lora_down.")
    if lora_down.shape[0] not in _SUPPORTED_W8A8_RANKS or lora_down.shape[0] == 0:
      raise ValueError(f"svdq_quantize_w4a8 supports rank 32/64/128, got {lora_down.shape[0]}.")
    if lora_down.shape[1] != input.shape[1] or not lora_down.is_contiguous():
      raise ValueError("svdq_quantize_w4a8 expects a contiguous lora_down of shape [R,K].")
  rows = input.shape[0]
  if rows == 0:
    rank = 0 if lora_down is None else lora_down.shape[0]
    return (input.new_empty((0, input.shape[1]), dtype=torch.float8_e4m3fn),
            input.new_empty(0, dtype=torch.float32), input.new_empty(
              (0, rank), dtype=torch.bfloat16) if rank else None)
  output, ascales, lora_act = torch.ops.cache_dit_cuda_ops.svdq_quantize_w4a8(input=input,
                                                                              smooth=smooth,
                                                                              lora_down=lora_down)
  if pad_rows:
    return output, ascales, (lora_act if lora_down is not None else None)
  return output[:rows], ascales[:rows], (lora_act[:rows] if lora_down is not None else None)


def _wNa4_logical_k(wgt_lo: torch.Tensor, bits: int) -> int:
  return wgt_lo.shape[1] * 4 if bits == 3 else wgt_lo.shape[1] * 8 // bits


torch.library.define("cache_dit_cuda_ops::svdq_expand_wna4",
                     "(Tensor wgt_lo, Tensor? wgt_hi, int bits) -> Tensor")


@torch.library.impl("cache_dit_cuda_ops::svdq_expand_wna4", "CUDA")
def _svdq_expand_wna4(wgt_lo: torch.Tensor, wgt_hi: torch.Tensor | None, bits: int) -> torch.Tensor:
  k = _wNa4_logical_k(wgt_lo, bits)
  out = torch.empty((wgt_lo.shape[0], k // 2), dtype=torch.int8, device=wgt_lo.device)
  _get_required_ops_module().svdq_expand_wna4(wgt_lo=wgt_lo, wgt_hi=wgt_hi, out=out, bits=bits, K=k)
  return out


@torch.library.register_fake("cache_dit_cuda_ops::svdq_expand_wna4")
def _fake_svdq_expand_wna4(wgt_lo: torch.Tensor, wgt_hi: torch.Tensor | None,
                           bits: int) -> torch.Tensor:
  del wgt_hi
  k = _wNa4_logical_k(wgt_lo, bits)
  return wgt_lo.new_empty((wgt_lo.shape[0], k // 2), dtype=torch.int8)


def expand_wna4(wgt_lo: torch.Tensor, wgt_hi: torch.Tensor | None, bits: int) -> torch.Tensor:
  """Expand packed W1/W2/W3 weights into standard packed E2M1 bytes."""
  if bits not in (1, 2, 3) or (bits == 3 and wgt_hi is None):
    raise ValueError("W1/W2 need no high plane; W3 requires its sign plane.")
  if wgt_lo.dim() != 2 or wgt_lo.dtype != torch.uint8 or not wgt_lo.is_contiguous():
    raise ValueError("wgt_lo must be a contiguous 2D uint8 tensor.")
  if wgt_hi is not None and (wgt_hi.dim() != 2 or wgt_hi.dtype != torch.uint8):
    raise ValueError("wgt_hi must be a 2D uint8 tensor.")
  k = _wNa4_logical_k(wgt_lo, bits)
  if k % 8:
    raise ValueError("Packed W1/W2/W3 requires a multiple-of-8 K extent.")
  return torch.ops.cache_dit_cuda_ops.svdq_expand_wna4(wgt_lo=wgt_lo, wgt_hi=wgt_hi, bits=bits)


def svdq_gemm_wna4(act: torch.Tensor,
                   wgt_lo: torch.Tensor,
                   wgt_hi: torch.Tensor | None,
                   ascales: torch.Tensor,
                   wscales: torch.Tensor,
                   lora_act_in: torch.Tensor | None = None,
                   lora_up: torch.Tensor | None = None,
                   bias: torch.Tensor | None = None,
                   wcscales: torch.Tensor | None = None,
                   *,
                   alpha: float = 1.0,
                   bits: int = 2) -> torch.Tensor:
  """Run the packed W1/W2/W3 FP4 GEMM with on-chip weight expansion."""
  if bits not in (1, 2, 3):
    raise ValueError(f"bits must be 1, 2 or 3, got {bits}.")
  out = torch.empty((act.shape[0], wgt_lo.shape[0]), dtype=torch.bfloat16, device=act.device)
  _get_required_ops_module().svdq_gemm_wna4(act=act,
                                            wgt_lo=wgt_lo,
                                            wgt_hi=wgt_hi,
                                            out=out,
                                            ascales=ascales,
                                            wscales=wscales,
                                            lora_act_in=lora_act_in,
                                            lora_up=lora_up,
                                            bias=bias,
                                            wcscales=wcscales,
                                            alpha=alpha,
                                            K=act.shape[1] * 2,
                                            bits=bits)
  return out


__all__ = [
  "svdq_gemm_w4a4",
  "svdq_quantize_w4a4",
  "svdq_gemm_w8a8",
  "svdq_quantize_w8a8",
  "svdq_gemm_w4a8",
  "svdq_quantize_w4a8",
  "svdq_gemm_wna4",
]
