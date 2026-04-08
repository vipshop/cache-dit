import triton
import triton.language as tl


@triton.jit
def _fp8_comm_per_token_quant_kernel(
  y_ptr: tl.tensor,
  x_ptr: tl.tensor,
  H: int,
  eps: float,
  bit8_min: float,
  bit8_max: float,
  BLOCK: tl.constexpr,
):
  s_id = tl.program_id(0).to(tl.int64)
  y_ptr += s_id * (H + 2)
  y_s_ptr = y_ptr + H
  x_ptr += s_id * H

  _absmax = tl.full([BLOCK], value=eps, dtype=tl.float32)
  for h in range(0, H, BLOCK):
    cols = h + tl.arange(0, BLOCK).to(tl.int64)
    mask = cols < H
    x = tl.load(x_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_last").to(tl.float32)
    _absmax = tl.maximum(tl.abs(x), _absmax)

  _absmax = tl.max(_absmax)
  x_s = _absmax / bit8_max
  x_s_inv = 1.0 / x_s
  x_s = x_s.to(x_ptr.dtype.element_ty)

  y_s_ptr = y_s_ptr.to(tl.pointer_type(x_ptr.dtype.element_ty, 1))
  tl.store(y_s_ptr, x_s)

  for h in range(0, H, BLOCK):
    cols = h + tl.arange(0, BLOCK).to(tl.int64)
    mask = cols < H
    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x_q = tl.clamp(x * x_s_inv, bit8_min, bit8_max).to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + cols, x_q, mask=mask)


@triton.jit
def _fp8_comm_per_token_dequant_kernel(
  y_ptr: tl.tensor,
  x_ptr: tl.tensor,
  H: int,
  BLOCK: tl.constexpr,
):
  s_id = tl.program_id(0).to(tl.int64)
  y_ptr += s_id * H
  x_ptr += s_id * (H + 2)

  x_s_ptr = x_ptr + H
  x_s_ptr = x_s_ptr.to(tl.pointer_type(y_ptr.dtype.element_ty, 1))
  x_s = tl.load(x_s_ptr).to(tl.float32)

  for h in range(0, H, BLOCK):
    cols = h + tl.arange(0, BLOCK).to(tl.int64)
    mask = cols < H
    x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x = x * x_s
    tl.store(y_ptr + cols, x, mask=mask)


@triton.jit
def _fp8_comm_qkv_permute_quant_kernel(
  quant_x_ptr: tl.tensor,
  x_ptr: tl.tensor,
  qx_stride_b: int,
  qx_stride_n: int,
  x_stride_b: int,
  x_stride_s: int,
  x_stride_p: int,
  B: int,
  S: int,
  N: int,
  D: int,
  EPS: tl.constexpr,
  BLOCK_SIZE_N: tl.constexpr,
  BLOCK_SIZE_D: tl.constexpr,
):
  psb_id = tl.program_id(0).to(tl.int64)
  b_id = psb_id % B
  s_id = (psb_id // B) % S
  p_id = psb_id // (S * B)

  x_ptr += b_id * x_stride_b + s_id * x_stride_s + p_id * x_stride_p
  quant_x_ptr += psb_id * qx_stride_b
  scale_ptr = quant_x_ptr.to(tl.pointer_type(tl.float32, 1))

  n_offset = tl.arange(0, BLOCK_SIZE_N)[None, :]
  n_mask = n_offset < N
  d_offset = tl.arange(0, BLOCK_SIZE_D)[:, None]
  d_mask = d_offset < D
  mask = n_mask & d_mask

  quant_x_blk = quant_x_ptr + n_offset * qx_stride_n + d_offset
  scale_blk = scale_ptr + n_offset * (D // 4 + 1) + D // 4
  x_blk = x_ptr + n_offset * D + d_offset

  x = tl.load(x_blk, mask=mask, other=0.0).to(tl.float32)
  scale = tl.max(tl.abs(x), axis=0, keep_dims=True) / 448.0
  scale = tl.maximum(scale, EPS)
  quant_x = x / scale
  quant_x = tl.clamp(quant_x, -448.0, 448.0).to(tl.float8e4nv)

  tl.store(quant_x_blk, quant_x, mask=mask)
  tl.store(scale_blk, scale, mask=n_mask)


@triton.jit
def _fp8_comm_qkv_dequant_permute_kernel(
  x_ptr: tl.tensor,
  quant_x_ptr: tl.tensor,
  x_stride_s: int,
  qx_stride_s: int,
  qx_stride_b: int,
  qx_stride_n: int,
  B: int,
  S: int,
  N: int,
  D: int,
  BLOCK_SIZE_N: tl.constexpr,
  BLOCK_SIZE_D: tl.constexpr,
):
  bs_id = tl.program_id(0).to(tl.int64)
  b_id = bs_id % B
  s_id = bs_id // B

  quant_x_ptr += s_id * qx_stride_s + b_id * qx_stride_b
  scale_ptr = quant_x_ptr.to(tl.pointer_type(tl.float32, 1))
  x_ptr += bs_id * x_stride_s

  n_offset = tl.arange(0, BLOCK_SIZE_N)[None, :]
  n_mask = n_offset < N
  d_offset = tl.arange(0, BLOCK_SIZE_D)[:, None]
  d_mask = d_offset < D
  mask = n_mask & d_mask

  x_blk = x_ptr + n_offset * D + d_offset
  quant_x_blk = quant_x_ptr + n_offset * qx_stride_n + d_offset
  scale_blk = scale_ptr + n_offset * (D // 4 + 1) + D // 4

  qx = tl.load(quant_x_blk, mask=mask, other=0.0).to(tl.float32)
  scale = tl.load(scale_blk, mask=n_mask, other=0.0).to(tl.float32)

  tl.store(x_blk, qx * scale, mask=mask)
