import torch
import triton
from typing import Tuple


from .float8_comm import (
    _fp8_comm_per_token_dequant_kernel,
    _fp8_comm_per_token_quant_kernel,
    _fp8_comm_qkv_permute_quant_kernel,
    _fp8_comm_qkv_dequant_permute_kernel,
)
from .merge_attn_states import (
    _fused_merge_attn_states_kernel,
)

# Note: Use torch.library.define with the format "namespace::operator_name"
# Better compatibility with torch.compile if we register the triton kernels
# as custom operators in torch library.
# namespace scheme:
# i: cache-d[i]t
# fp8_ops: fp8 related ops, comm or gemm.
# attn_ops: attention related ops.

# FP8 related ops
torch.library.define(
    "_i_fp8_ops::fp8_comm_per_token_quant",
    "(Tensor x) -> Tensor",
)
torch.library.define(
    "_i_fp8_ops::fp8_comm_per_token_dequant",
    "(Tensor x) -> Tensor",
)
torch.library.define(
    "_i_fp8_ops::fp8_comm_qkv_permute_quant",
    "(Tensor x, float eps=1e-6) -> Tensor",
)
torch.library.define(
    "_i_fp8_ops::fp8_comm_qkv_permute_dequant",
    "(Tensor quant_x, ScalarType dtype=bfloat16) -> Tensor",
)


@torch.library.impl("_i_fp8_ops::fp8_comm_per_token_quant", "CUDA")
def _fp8_comm_per_token_quant_cuda(x: torch.Tensor) -> torch.Tensor:
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
        _fp8_comm_per_token_quant_kernel[(M,)](
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


@torch.library.impl("_i_fp8_ops::fp8_comm_per_token_dequant", "CUDA")
def _fp8_comm_per_token_dequant_cuda(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float8_e4m3fn, f"expected float8_e4m3fn but got {x.dtype}"
    *shape, H = x.shape
    x = x.reshape(-1, H).contiguous()
    M, N = x.shape
    N -= 2
    y = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)

    BLOCK = max(min(8192, 65536 // x.element_size(), triton.next_power_of_2(N)), 128)
    num_warps = min(max(BLOCK // 256, 1), 8)

    with torch.cuda.device(x.device):
        _fp8_comm_per_token_dequant_kernel[(M,)](
            y,
            x,
            N,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
    return y.reshape(*shape, H - 2)


@torch.library.impl("_i_fp8_ops::fp8_comm_qkv_permute_quant", "CUDA")
def _fp8_comm_qkv_permute_quant_cuda(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    B, S, P, N, D = x.shape

    quant_x = torch.empty((P, S, B, N, D + 4), dtype=torch.float8_e4m3fn, device=x.device)

    grid = (P * S * B,)

    with torch.cuda.device(x.device):
        _fp8_comm_qkv_permute_quant_kernel[grid](
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


@torch.library.impl("_i_fp8_ops::fp8_comm_qkv_permute_dequant", "CUDA")
def _fp8_comm_qkv_permute_dequant_cuda(
    quant_x: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    S, B, N, _D = quant_x.shape
    D = _D - 4
    x = torch.empty((B, S, N, D), dtype=dtype, device=quant_x.device)

    grid = (B * S,)

    with torch.cuda.device(x.device):
        _fp8_comm_qkv_dequant_permute_kernel[grid](
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


@torch.library.register_fake("_i_fp8_ops::fp8_comm_per_token_quant")
def _fp8_comm_per_token_quant_abstract(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.bfloat16
    *shape, H = x.shape
    return x.new_empty((*shape, H + 2), dtype=torch.float8_e4m3fn)


@torch.library.register_fake("_i_fp8_ops::fp8_comm_per_token_dequant")
def _fp8_comm_per_token_dequant_abstract(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float8_e4m3fn
    *shape, H = x.shape
    return x.new_empty((*shape, H - 2), dtype=torch.bfloat16)


@torch.library.register_fake("_i_fp8_ops::fp8_comm_qkv_permute_quant")
def _fp8_comm_qkv_permute_quant_abstract(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    B, S, P, N, D = x.shape
    return x.new_empty((P, S, B, N, D + 4), dtype=torch.float8_e4m3fn)


@torch.library.register_fake("_i_fp8_ops::fp8_comm_qkv_permute_dequant")
def _fp8_comm_qkv_permute_dequant_abstract(
    quant_x: torch.Tensor, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    S, B, N, _D = quant_x.shape
    D = _D - 4
    return quant_x.new_empty((B, S, N, D), dtype=dtype)


def fp8_comm_per_token_quant(x: torch.Tensor) -> torch.Tensor:
    return torch.ops._i_fp8_ops.fp8_comm_per_token_quant(x)


def fp8_comm_per_token_dequant(x: torch.Tensor) -> torch.Tensor:
    return torch.ops._i_fp8_ops.fp8_comm_per_token_dequant(x)


def fp8_comm_qkv_permute_quant(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.ops._i_fp8_ops.fp8_comm_qkv_permute_quant(x, eps)


def fp8_comm_qkv_permute_dequant(
    quant_x: torch.Tensor, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    return torch.ops._i_fp8_ops.fp8_comm_qkv_permute_dequant(quant_x, dtype)


# Attention related ops
torch.library.define(
    "_i_attn_ops::fused_merge_attn_states",
    "(Tensor prev_out, Tensor prev_lse, Tensor suff_out, Tensor suff_lse) -> (Tensor out, Tensor lse)",
)


@torch.library.impl("_i_attn_ops::fused_merge_attn_states", "CUDA")
def _fused_merge_attn_states_cuda(
    prev_out: torch.Tensor,  # [B, N, H, D]
    prev_lse: torch.Tensor,  # [B, N, H, 1]
    suff_out: torch.Tensor,
    suff_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, N, H, D = suff_out.shape  # Batch, Seq_len, Num_heads, Head_dim
    # Flatten the batch and sequence dimensions
    prev_out = prev_out.flatten(0, 1).contiguous()  # [B*N, H, D]
    suff_out = suff_out.flatten(0, 1).contiguous()  # [B*N, H, D]
    prev_lse = prev_lse.flatten(0, 1).squeeze(-1).contiguous()  # [B*N, H]
    suff_lse = suff_lse.flatten(0, 1).squeeze(-1).contiguous()  # [B*N, H]

    out = torch.empty_like(suff_out).contiguous()
    lse = torch.empty_like(suff_lse).contiguous()

    with torch.cuda.device(suff_out.device):
        _fused_merge_attn_states_kernel[(B * N, H)](
            out,
            lse,
            prev_out,
            prev_lse,
            suff_out,
            suff_lse,
            D,
            triton.next_power_of_2(D),
        )

    # Reshape back to original shape
    out = out.view(B, N, H, D)
    lse = lse.view(B, N, H, 1)
    return out, lse


@torch.library.register_fake("_i_attn_ops::fused_merge_attn_states")
def _fused_merge_attn_states_abstract(
    prev_out: torch.Tensor,  # [B, N, H, D]
    prev_lse: torch.Tensor,  # [B, N, H, 1]
    suff_out: torch.Tensor,
    suff_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, N, H, D = suff_out.shape

    # Exactly replicate the memory layout operations from the real impl
    # to ensure the abstract function has the same behavior as the real
    # function in terms of memory layout, which is important for torch
    # compile to generate correct code.
    suff_out = suff_out.flatten(0, 1).contiguous()
    suff_lse = suff_lse.flatten(0, 1).squeeze(-1).contiguous()
    out = suff_out.new_empty(suff_out.shape, dtype=suff_out.dtype)
    lse = suff_lse.new_empty(suff_lse.shape, dtype=suff_lse.dtype)
    out = out.view(B, N, H, D)
    lse = lse.view(B, N, H, 1)

    return out, lse


def fused_merge_attn_states(
    prev_out: torch.Tensor,  # [B, N, H, D]
    prev_lse: torch.Tensor,  # [B, N, H, 1]
    suff_out: torch.Tensor,
    suff_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops._i_attn_ops.fused_merge_attn_states(
        prev_out,
        prev_lse,
        suff_out,
        suff_lse,
    )


__all__ = [
    # FP8 related ops
    "fp8_comm_per_token_quant",
    "fp8_comm_per_token_dequant",
    "fp8_comm_qkv_permute_quant",
    "fp8_comm_qkv_permute_dequant",
    # Attention related ops
    "fused_merge_attn_states",
]
