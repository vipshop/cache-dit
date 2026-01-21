import torch
import triton
import triton.language as tl

__all__ = [
    "per_token_quant_fp8",
    "per_token_dequant_fp8",
    "qkv_permute_quant_fp8",
    "qkv_dequant_permute_fp8",
]


@triton.jit
def _per_token_quant_8bit(
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
def _per_token_dequant_8bit(
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


def per_token_quant_fp8(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16, f"expected bfloat16 but got {x.dtype}"
    dtype = torch.float8_e4m3fn
    finfo = torch.finfo(dtype)
    *shape, H = x.shape
    x = x.reshape(-1, H).contiguous()
    M, N = x.shape
    y = torch.empty((M, N + 2), dtype=dtype, device=x.device)

    BLOCK = max(min(8192, 65536 // x.element_size(), triton.next_power_of_2(N)), 128)
    num_warps = min(max(BLOCK // 256, 1), 8)

    with torch.cuda.device(x.device):
        _per_token_quant_8bit[(M,)](
            y,
            x,
            N,
            eps=1e-4,
            bit8_min=finfo.min,
            bit8_max=finfo.max,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
    return y.reshape(*shape, H + 2)


def per_token_dequant_fp8(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float8_e4m3fn, f"expected float8_e4m3fn but got {x.dtype}"
    *shape, H = x.shape
    x = x.reshape(-1, H).contiguous()
    M, N = x.shape
    N -= 2
    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)

    BLOCK = max(min(8192, 65536 // x.element_size(), triton.next_power_of_2(N)), 128)
    num_warps = min(max(BLOCK // 256, 1), 8)

    with torch.cuda.device(x.device):
        _per_token_dequant_8bit[(M,)](
            y,
            x,
            N,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
    return y.reshape(*shape, H - 2)


@triton.jit
def _qkv_permute_quant(
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
def _qkv_dequant_permute(
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


def qkv_permute_quant_fp8(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    B, S, P, N, D = x.shape

    quant_x = torch.empty((P, S, B, N, D + 4), dtype=torch.float8_e4m3fn, device=x.device)

    grid = (P * S * B,)

    with torch.cuda.device(x.device):
        _qkv_permute_quant[grid](
            quant_x,
            x,
            quant_x.stride(2),
            quant_x.stride(3),
            x.stride(0),
            x.stride(1),
            x.stride(2),
            B,
            S,
            N,
            D,
            eps,
            triton.next_power_of_2(N),
            triton.next_power_of_2(D),
        )

    return quant_x


def qkv_dequant_permute_fp8(
    quant_x: torch.Tensor, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    S, B, N, _D = quant_x.shape
    D = _D - 4
    x = torch.empty((B, S, N, D), dtype=dtype, device=quant_x.device)

    grid = (B * S,)

    with torch.cuda.device(x.device):
        _qkv_dequant_permute[grid](
            x,
            quant_x,
            x.stride(1),
            quant_x.stride(0),
            quant_x.stride(1),
            quant_x.stride(2),
            B,
            S,
            N,
            D,
            triton.next_power_of_2(N),
            triton.next_power_of_2(D),
        )

    return x
