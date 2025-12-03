import torch
import triton
import triton.language as tl

__all__ = ['per_token_quant_fp8_merge_scale', 'per_token_dequant_fp8']


@triton.jit
def _per_token_quant_8bit_merge_scale(
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
        x = tl.load(x_ptr + cols, mask=mask, other=0.0, eviction_policy='evict_last').to(tl.float32)
        _absmax = tl.maximum(tl.abs(x), _absmax)

    _absmax = tl.max(_absmax)
    x_s = _absmax / bit8_max
    x_s_inv = 1.0 / x_s

    x_s = x_s.to(x_ptr.dtype.element_ty)
    s16 = x_s.cast(tl.int16, bitcast=True)
    lo = (s16 & 0xFF).to(tl.int8)
    hi = ((s16 >> 8) & 0xFF).to(tl.int8)
    lo = lo.cast(y_ptr.dtype.element_ty, bitcast=True)
    hi = hi.cast(y_ptr.dtype.element_ty, bitcast=True)
    tl.store(y_s_ptr, lo)
    tl.store(y_s_ptr + 1, hi)

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
    lo = tl.load(x_s_ptr)[None]
    hi = tl.load(x_s_ptr + 1)[None]
    lo = lo.cast(tl.int8, bitcast=True).to(tl.int16)
    hi = hi.cast(tl.int8, bitcast=True).to(tl.int16)
    s16 = (lo & 0xFF) | ((hi & 0xFF) << 8)
    s16 = s16.cast(tl.bfloat16, bitcast=True)
    x_s = s16.to(tl.float32)

    for h in range(0, H, BLOCK):
        cols = h + tl.arange(0, BLOCK).to(tl.int64)
        mask = cols < H
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x = x * x_s
        tl.store(y_ptr + cols, x, mask=mask)


def per_token_quant_fp8_merge_scale(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16, f'expected bfloat16 but got {x.dtype}'
    dtype = torch.float8_e4m3fn
    finfo = torch.finfo(dtype)
    *shape, H = x.shape
    x = x.reshape(-1, H).contiguous()
    M, N = x.shape
    y = torch.empty((M, N + 2), dtype=dtype, device=x.device)

    BLOCK = max(min(8192, 65536 // x.element_size(), triton.next_power_of_2(N)), 128)
    num_warps = min(max(BLOCK // 256, 1), 8)

    with torch.cuda.device(x.device):
        _per_token_quant_8bit_merge_scale[(M,)](
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
    assert x.dtype == torch.float8_e4m3fn, f'expected float8_e4m3fn but got {x.dtype}'
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
