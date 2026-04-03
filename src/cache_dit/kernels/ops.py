import torch
from typing import Callable, Tuple
from .backend import KernelBackend


_KERNEL_BE_FN = Callable[..., KernelBackend]
_TRITON_BE_FN = lambda: KernelBackend.TRITON
_CUDA_BE_FN = lambda: KernelBackend.CUDA
_ERROR_TEMPLATE = "kernel backend: {} is not supported now!"


def _ensure_backend_supported(backend: KernelBackend) -> None:
    if not KernelBackend.is_supported(backend):
        raise ValueError(_ERROR_TEMPLATE.format(backend))


# Ulysses FP8 communication related ops
def _fp8_comm_per_token_quant_impl(
    x: torch.Tensor,
    backend_fn: _KERNEL_BE_FN = _TRITON_BE_FN,
) -> torch.Tensor:
    backend = backend_fn()
    _ensure_backend_supported(backend)
    if backend == KernelBackend.TRITON:
        from .triton import fp8_comm_per_token_quant

        return fp8_comm_per_token_quant(x)
    else:
        raise ValueError(_ERROR_TEMPLATE.format(backend))


def _fp8_comm_per_token_dequant_impl(
    x: torch.Tensor,
    backend_fn: _KERNEL_BE_FN = _TRITON_BE_FN,
) -> torch.Tensor:
    backend = backend_fn()
    _ensure_backend_supported(backend)
    if backend == KernelBackend.TRITON:
        from .triton import fp8_comm_per_token_dequant

        return fp8_comm_per_token_dequant(x)
    else:
        raise ValueError(_ERROR_TEMPLATE.format(backend))


def _fp8_comm_qkv_permute_quant_impl(
    x: torch.Tensor,
    backend_fn: _KERNEL_BE_FN = _TRITON_BE_FN,
) -> torch.Tensor:
    backend = backend_fn()
    _ensure_backend_supported(backend)
    if backend == KernelBackend.TRITON:
        from .triton import fp8_comm_qkv_permute_quant

        return fp8_comm_qkv_permute_quant(x)
    else:
        raise ValueError(_ERROR_TEMPLATE.format(backend))


def _fp8_comm_qkv_permute_dequant_impl(
    quant_x: torch.Tensor,
    backend_fn: _KERNEL_BE_FN = _TRITON_BE_FN,
) -> torch.Tensor:
    backend = backend_fn()
    _ensure_backend_supported(backend)
    if backend == KernelBackend.TRITON:
        from .triton import fp8_comm_qkv_permute_dequant

        return fp8_comm_qkv_permute_dequant(quant_x)
    else:
        raise ValueError(_ERROR_TEMPLATE.format(backend))


# Attention related ops, e.g, for Ring Attention
def _fused_merge_attn_states_impl(
    prev_out: torch.Tensor,
    prev_lse: torch.Tensor,
    suff_out: torch.Tensor,
    suff_lse: torch.Tensor,
    backend_fn: _KERNEL_BE_FN = _TRITON_BE_FN,
) -> Tuple[torch.Tensor, torch.Tensor]:
    backend = backend_fn()
    _ensure_backend_supported(backend)
    if backend == KernelBackend.TRITON:
        from .triton import fused_merge_attn_states

        return fused_merge_attn_states(
            prev_out,
            prev_lse,
            suff_out,
            suff_lse,
        )
    else:
        raise ValueError(_ERROR_TEMPLATE.format(backend))


# SVDQuant related ops, with CUDA implementations by default.
def _svdq_extension_is_available_impl() -> bool:
    from .cuda import svdq_extension_is_available

    return svdq_extension_is_available()


def _svdq_get_load_error_impl() -> Exception | None:
    from .cuda import svdq_get_load_error

    return svdq_get_load_error()


def _svdq_gemm_w4a4_impl(
    act: torch.Tensor,
    wgt: torch.Tensor,
    ascales: torch.Tensor,
    wscales: torch.Tensor,
    lora_act_in: torch.Tensor | None = None,
    lora_up: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    fp4: bool = False,
    alpha: float | None = 1.0,
    wcscales: torch.Tensor | None = None,
    act_unsigned: bool = False,
    output_dtype: torch.dtype | None = None,
    backend_fn: _KERNEL_BE_FN = _CUDA_BE_FN,
) -> torch.Tensor:
    backend = backend_fn()
    _ensure_backend_supported(backend)
    if backend == KernelBackend.CUDA:
        from .cuda import svdq_gemm_w4a4

        return svdq_gemm_w4a4(
            act=act,
            wgt=wgt,
            ascales=ascales,
            wscales=wscales,
            lora_act_in=lora_act_in,
            lora_up=lora_up,
            bias=bias,
            fp4=fp4,
            alpha=alpha,
            wcscales=wcscales,
            act_unsigned=act_unsigned,
            output_dtype=output_dtype,
        )
    raise ValueError(_ERROR_TEMPLATE.format(backend))


def _svdq_gemm_w4a4_ext_impl(
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
    backend_fn: _KERNEL_BE_FN = _CUDA_BE_FN,
) -> torch.Tensor:
    backend = backend_fn()
    _ensure_backend_supported(backend)
    if backend == KernelBackend.CUDA:
        from .cuda import svdq_gemm_w4a4_ext

        return svdq_gemm_w4a4_ext(
            act=act,
            wgt=wgt,
            out=out,
            qout=qout,
            ascales=ascales,
            wscales=wscales,
            oscales=oscales,
            poolout=poolout,
            lora_act_in=lora_act_in,
            lora_up=lora_up,
            lora_down=lora_down,
            lora_act_out=lora_act_out,
            norm_q=norm_q,
            norm_k=norm_k,
            rotary_emb=rotary_emb,
            bias=bias,
            smooth_factor=smooth_factor,
            out_vk=out_vk,
            out_linearattn=out_linearattn,
            act_unsigned=act_unsigned,
            lora_scales=lora_scales,
            fuse_silu=fuse_silu,
            fp4=fp4,
            alpha=alpha,
            wcscales=wcscales,
            out_q=out_q,
            out_k=out_k,
            out_v=out_v,
            attn_tokens=attn_tokens,
        )
    raise ValueError(_ERROR_TEMPLATE.format(backend))


def _svdq_quantize_w4a4_act_fuse_lora_impl(
    input: torch.Tensor,
    lora_down: torch.Tensor | None = None,
    smooth: torch.Tensor | None = None,
    fuse_glu: bool = False,
    fp4: bool = False,
    pad_size: int = 256,
    backend_fn: _KERNEL_BE_FN = _CUDA_BE_FN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    backend = backend_fn()
    _ensure_backend_supported(backend)
    if backend == KernelBackend.CUDA:
        from .cuda import svdq_quantize_w4a4_act_fuse_lora

        return svdq_quantize_w4a4_act_fuse_lora(
            input=input,
            lora_down=lora_down,
            smooth=smooth,
            fuse_glu=fuse_glu,
            fp4=fp4,
            pad_size=pad_size,
        )
    raise ValueError(_ERROR_TEMPLATE.format(backend))


def _svdq_quantize_w4a4_wgt_impl(
    input: torch.Tensor,
    output: torch.Tensor | None = None,
    oscales: torch.Tensor | None = None,
    backend_fn: _KERNEL_BE_FN = _CUDA_BE_FN,
) -> tuple[torch.Tensor, torch.Tensor]:
    backend = backend_fn()
    _ensure_backend_supported(backend)
    if backend == KernelBackend.CUDA:
        from .cuda import svdq_quantize_w4a4_wgt

        return svdq_quantize_w4a4_wgt(input=input, output=output, oscales=oscales)
    raise ValueError(_ERROR_TEMPLATE.format(backend))


def _svdq_set_faster_i2f_mode_impl(
    mode: str,
    backend_fn: _KERNEL_BE_FN = _CUDA_BE_FN,
) -> None:
    backend = backend_fn()
    _ensure_backend_supported(backend)
    if backend == KernelBackend.CUDA:
        from .cuda import svdq_set_faster_i2f_mode

        svdq_set_faster_i2f_mode(mode)
        return
    raise ValueError(_ERROR_TEMPLATE.format(backend))


def _svdq_set_log_level_impl(
    level: str,
    backend_fn: _KERNEL_BE_FN = _CUDA_BE_FN,
) -> None:
    backend = backend_fn()
    _ensure_backend_supported(backend)
    if backend == KernelBackend.CUDA:
        from .cuda import svdq_set_log_level

        svdq_set_log_level(level)
        return
    raise ValueError(_ERROR_TEMPLATE.format(backend))


# Ulysses FP8 communication related ops
def fp8_comm_per_token_quant(x: torch.Tensor) -> torch.Tensor:
    """Quantize a floating-point tensor to FP8 per-token format.

    Args:
        x: Input floating-point tensor to be quantized.
    Returns:
        Quantized tensor in FP8 format, where the quantization is performed
        on a per-token quantization scheme suitable for communication purposes.
    """

    return _fp8_comm_per_token_quant_impl(x=x, backend_fn=_TRITON_BE_FN)


def fp8_comm_per_token_dequant(x: torch.Tensor) -> torch.Tensor:
    """Dequantize a FP8 tensor to floating-point format using per-token method.

    Args:
        x: Input FP8 tensor to be dequantized.
    Returns:
        Dequantized tensor in floating-point format, where the dequantization
        is performed on a per-token basis suitable for communication purposes.
    """
    return _fp8_comm_per_token_dequant_impl(x=x, backend_fn=_TRITON_BE_FN)


def fp8_comm_qkv_permute_quant(x: torch.Tensor) -> torch.Tensor:
    """Quantize a floating-point tensor to FP8 format with QKV permutation.

    Args:
        x: Input floating-point tensor to be quantized.
    Returns:
        Quantized tensor in FP8 format with QKV permutation, suitable for communication purposes.
    """
    return _fp8_comm_qkv_permute_quant_impl(x=x, backend_fn=_TRITON_BE_FN)


def fp8_comm_qkv_permute_dequant(quant_x: torch.Tensor) -> torch.Tensor:
    """Dequantize a FP8 tensor with QKV permutation to floating-point format.

    Args:
        quant_x: Input FP8 tensor with QKV permutation to be dequantized.
    Returns:
        Dequantized tensor in floating-point format, where the dequantization is performed
        on a per-token basis suitable for communication purposes.
    """
    return _fp8_comm_qkv_permute_dequant_impl(
        quant_x=quant_x,
        backend_fn=_TRITON_BE_FN,
    )


# Attention related ops, e.g, for Ring Attention
def fused_merge_attn_states(
    prev_out: torch.Tensor,
    prev_lse: torch.Tensor,
    suff_out: torch.Tensor,
    suff_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse the attention states of two consecutive attention states, e.g., Ring Attention.

    Args:
        prev_out: Previous output tensor.
        prev_lse: Previous log-sum-exp tensor.
        suff_out: Sufficient output tensor.
        suff_lse: Sufficient log-sum-exp tensor.
    Returns:
        Fused output and log-sum-exp tensors.
    """
    return _fused_merge_attn_states_impl(
        prev_out=prev_out,
        prev_lse=prev_lse,
        suff_out=suff_out,
        suff_lse=suff_lse,
        backend_fn=_TRITON_BE_FN,
    )


# SVDQuant related ops, with CUDA implementations by default.
def svdq_gemm_w4a4(
    act: torch.Tensor,
    wgt: torch.Tensor,
    ascales: torch.Tensor,
    wscales: torch.Tensor,
    lora_act_in: torch.Tensor | None = None,
    lora_up: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    fp4: bool = False,
    alpha: float | None = 1.0,
    wcscales: torch.Tensor | None = None,
    act_unsigned: bool = False,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """INT4/FP4 GEMM with optional fused LoRA and quantization support.

    Args:
        act: Packed activation tensor `[M, K / 2]`.
        wgt: Packed weight tensor `[N, K / 2]`.
        ascales: Activation scales `[K / 64, M]` for INT4 or `[K / 16, M]` for FP4.
        wscales: Weight scales `[K / 64, N]` or `[K / 16, N]` depending on precision.
        lora_act_in: Optional low-rank activation input `[M, R]` for fused LoRA paths.
        lora_up: Optional low-rank up projection `[N, R]` for fused LoRA paths.
        bias: Optional dense output bias `[N]`.
        fp4: Whether the input tensors are in FP4 (true) or INT4 (false) format.
        alpha: Weight scaling factor used by FP4 paths, typically set to 1.0 / max(abs(weight)).
        wcscales: Optional per-channel FP4 weight correction scales.
        act_unsigned: Whether activations are interpreted as unsigned quantized values, which can affect quantization behavior and output ranges.
        output_dtype: Optional dtype for the output tensor, which can be used to control the precision of the output in FP4 paths or to specify a different dtype for INT4 paths.
    Returns:
        The output tensor resulting from the quantized GEMM operation, with optional LoRA fusion applied. The dtype and device of the output tensor may depend on the input parameters and the specific kernel implementation
        used by the backend.
    """

    return _svdq_gemm_w4a4_impl(
        act=act,
        wgt=wgt,
        ascales=ascales,
        wscales=wscales,
        lora_act_in=lora_act_in,
        lora_up=lora_up,
        bias=bias,
        fp4=fp4,
        alpha=alpha,
        wcscales=wcscales,
        act_unsigned=act_unsigned,
        output_dtype=output_dtype,
        backend_fn=_CUDA_BE_FN,
    )


def svdq_gemm_w4a4_ext(
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
    """Fully-extended SVDQ GEMM with support for various fusion paths and quantization options.

    Args:
        act: Packed activation tensor `[M, K / 2]`.
        wgt: Packed weight tensor `[N, K / 2]`.
        out: Dense output tensor `[M, N]`.
        qout: Optional packed output buffer `[M, N / 2]` for re-quantized activations.
        ascales: Activation scales `[K / 64, M]` for INT4 or `[K / 16, M]` for FP4.
        wscales: Weight scales `[K / 64, N]` or `[K / 16, N]` depending on precision.
        oscales: Optional output activation scales for `qout`.
        poolout: Optional pooled output buffer for fused pooling paths.
        lora_act_in: Optional low-rank activation input `[M, R]`.
        lora_up: Optional low-rank up projection `[N, R]`.
        lora_down: Optional low-rank down projection used by fused LoRA-down paths.
        lora_act_out: Optional intermediate low-rank activation output buffer.
        norm_q: Optional RMSNorm/Q normalization tensor for fused attention paths.
        norm_k: Optional RMSNorm/K normalization tensor for fused attention paths.
        rotary_emb: Optional RoPE embedding tensor for fused attention paths.
        bias: Optional dense output bias `[N]`.
        smooth_factor: Optional next-layer smoothing factor written by fused quantization paths.
        out_vk: Optional linear-attention VK output buffer.
        out_linearattn: Optional linear-attention output buffer.
        act_unsigned: Whether activations are interpreted as unsigned quantized values.
        lora_scales: Per-16-rank LoRA scales used by the native fused LoRA implementation.
        fuse_silu: Whether to enable fused SiLU in advanced kernel variants.
        fp4: Whether the packed tensors use FP4 rather than INT4 layout.
        alpha: Weight scaling factor used by FP4 paths.
        wcscales: Optional per-channel FP4 weight correction scales.
        out_q: Optional packed attention-Q output buffer.
        out_k: Optional packed attention-K output buffer.
        out_v: Optional packed attention-V output buffer.
        attn_tokens: Token count for fused attention-style kernel variants.
    Returns:
        The output tensor, which may be the same as the `out` argument if provided.
    """
    return _svdq_gemm_w4a4_ext_impl(
        act=act,
        wgt=wgt,
        out=out,
        qout=qout,
        ascales=ascales,
        wscales=wscales,
        oscales=oscales,
        poolout=poolout,
        lora_act_in=lora_act_in,
        lora_up=lora_up,
        lora_down=lora_down,
        lora_act_out=lora_act_out,
        norm_q=norm_q,
        norm_k=norm_k,
        rotary_emb=rotary_emb,
        bias=bias,
        smooth_factor=smooth_factor,
        out_vk=out_vk,
        out_linearattn=out_linearattn,
        act_unsigned=act_unsigned,
        lora_scales=lora_scales,
        fuse_silu=fuse_silu,
        fp4=fp4,
        alpha=alpha,
        wcscales=wcscales,
        out_q=out_q,
        out_k=out_k,
        out_v=out_v,
        attn_tokens=attn_tokens,
        backend_fn=_CUDA_BE_FN,
    )


def svdq_quantize_w4a4_act_fuse_lora(
    input: torch.Tensor,
    lora_down: torch.Tensor | None = None,
    smooth: torch.Tensor | None = None,
    fuse_glu: bool = False,
    fp4: bool = False,
    pad_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize activations to INT4/FP4 format with optional fused LoRA support.

    Args:
        input: Input activation tensor `[M, K]`.
        lora_down: Optional low-rank down projection used by fused LoRA-down paths.
        smooth: Optional next-layer smoothing factor written by fused quantization paths.
        fuse_glu: Whether to enable fused GLU in advanced kernel variants.
        fp4: Whether the input tensors are in FP4 (true) or INT4 (false) format.
        pad_size: Padding size for the input tensor.
    Returns:
        A tuple containing the quantized activation tensor, the scale tensor, and the zero-point tensor.
    """

    return _svdq_quantize_w4a4_act_fuse_lora_impl(
        input=input,
        lora_down=lora_down,
        smooth=smooth,
        fuse_glu=fuse_glu,
        fp4=fp4,
        pad_size=pad_size,
        backend_fn=_CUDA_BE_FN,
    )


def svdq_quantize_w4a4_wgt(
    input: torch.Tensor,
    output: torch.Tensor | None = None,
    oscales: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weights to INT4/FP4 format with optional output buffer and scale tensor.

    Args:
        input: Input weight tensor `[N, K]`.
        output: Optional output tensor to store the quantized weights.
        oscales: Optional scale tensor for the output weights.
    Returns:
        A tuple containing the quantized weight tensor and the scale tensor.
    """
    return _svdq_quantize_w4a4_wgt_impl(
        input=input,
        output=output,
        oscales=oscales,
        backend_fn=_CUDA_BE_FN,
    )


def svdq_set_faster_i2f_mode(mode: str) -> None:
    _svdq_set_faster_i2f_mode_impl(mode=mode, backend_fn=_CUDA_BE_FN)


def svdq_set_log_level(level: str) -> None:
    _svdq_set_log_level_impl(level=level, backend_fn=_CUDA_BE_FN)


def svdq_extension_is_available() -> bool:
    return _svdq_extension_is_available_impl()


def svdq_get_load_error() -> Exception | None:
    return _svdq_get_load_error_impl()


__all__ = [
    # FP8 related ops
    "fp8_comm_per_token_quant",
    "fp8_comm_per_token_dequant",
    "fp8_comm_qkv_permute_quant",
    "fp8_comm_qkv_permute_dequant",
    # Attention related ops
    "fused_merge_attn_states",
    # SVDQuant related ops
    "svdq_get_load_error",
    "svdq_extension_is_available",
    "svdq_gemm_w4a4",
    "svdq_gemm_w4a4_ext",
    "svdq_quantize_w4a4_act_fuse_lora",
    "svdq_quantize_w4a4_wgt",
    "svdq_set_faster_i2f_mode",
    "svdq_set_log_level",
]
