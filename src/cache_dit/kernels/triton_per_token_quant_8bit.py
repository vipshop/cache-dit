import torch
import triton
import triton.language as tl

__all__ = ['per_token_quant_fp8_merge_scale']


@triton.jit
def _per_token_quant_8bit_merge_scale(
        y_ptr: tl.tensor,
        x_ptr: tl.tensor,
        H: tl.int32,
        eps: tl.float32,
        bit8_min: tl.float32,
        bit8_max: tl.float32,
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
    u16 = tl.cast(x_s, tl.uint16, bitcast=True)
    b0 = (u16 & 0x00FF).to(tl.uint8)
    b1 = ((u16 >> 8) & 0x00FF).to(tl.uint8)
    tl.store(y_s_ptr, tl.cast(b0, y_ptr.dtype.element_ty, bitcast=True))
    tl.store(y_s_ptr + 1, tl.cast(b1, y_ptr.dtype.element_ty, bitcast=True))

    for h in range(0, H, BLOCK):
        cols = h + tl.arange(0, BLOCK).to(tl.int64)
        mask = cols < H
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x_q = tl.clamp(x * x_s_inv, bit8_min, bit8_max).to(y_ptr.dtype.element_ty)
        tl.store(y_ptr + cols, x_q, mask=mask)


def per_token_quant_fp8_merge_scale(x: torch.Tensor):
    finfo = torch.finfo(torch.float8_e4m3fn)
    shape = x.shape
    x = x.reshape(-1, shape[-1]).contiguous()
    M, N = x.shape
    y = torch.empty((M, N + 2), dtype=torch.float8_e4m3fn, device=x.device)

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
    return y
