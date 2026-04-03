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
    return _load_extension_module() is not None


def svdq_get_load_error() -> Exception | None:
    _load_extension_module()
    return _cached_load_error


def _get_required_extension_module() -> ModuleType:
    extension_module = _load_extension_module()
    if extension_module is None:
        error = svdq_get_load_error()
        raise RuntimeError(
            "The optional Cache-DiT SVDQuant CUDA extension is not available. Build it with "
            "`CACHE_DIT_BUILD_SVDQUANT=1 /workspace/dev/miniconda3/envs/cdit/bin/python -m pip install -e . --no-build-isolation` after `conda activate cdit`."
        ) from error
    return extension_module


def _get_required_ops_module() -> ModuleType:
    ops_module = getattr(_get_required_extension_module(), "ops", None)
    if ops_module is None:
        raise RuntimeError(
            "The loaded Cache-DiT SVDQuant extension does not expose an `ops` submodule."
        )
    return ops_module


def _get_required_utils_module() -> ModuleType:
    utils_module = getattr(_get_required_extension_module(), "utils", None)
    if utils_module is None:
        raise RuntimeError(
            "The loaded Cache-DiT SVDQuant extension does not expose a `utils` submodule."
        )
    return utils_module


def _encode_svdq_output_dtype(output_dtype: torch.dtype) -> int:
    dtype_id = _DTYPE_TO_ID.get(output_dtype)
    if dtype_id is None:
        raise ValueError(f"Unsupported SVDQuant output dtype: {output_dtype}")
    return dtype_id


def _decode_svdq_output_dtype(dtype_id: int) -> torch.dtype:
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
    ops_module = _get_required_ops_module()
    rank = 0 if lora_down is None else lora_down.shape[1]
    if rank == 0:
        dummy_rank = 16
        dummy_lora_down = torch.zeros(
            input.shape[1], dummy_rank, dtype=input.dtype, device=input.device
        )
        dummy_lora_act_out = torch.empty(
            output.shape[0], dummy_rank, dtype=torch.float32, device=input.device
        )
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

    Args:
        act: Packed activation tensor `[M, K / 2]`.
        wgt: Packed quantized weight tensor `[N, K / 2]`.
        out: Dense output buffer `[M, N]`.
        qout: Optional packed quantized output buffer `[M, N / 2]` for the next layer.
        ascales: Activation scales `[K / G, M]`, where `G` is 64 for INT4 and 16 for FP4.
        wscales: Weight scales `[K / G, N]`, where `G` is 64 for INT4 and 16 for FP4.
        oscales: Optional output scales `[N / G, M]` for `qout`.
        poolout: Optional pooled output buffer used by specialized fused kernels.
        lora_act_in: Optional LoRA activation input `[M, R]`.
        lora_up: Optional LoRA up-projection weights `[N, R]`.
        lora_down: Optional LoRA down-projection weights `[N, R]` for the next fused layer.
        lora_act_out: Optional LoRA activation output buffer `[M, R]` for the next fused layer.
        norm_q: Optional query RMSNorm tensor `[HEAD_DIM]`.
        norm_k: Optional key RMSNorm tensor `[HEAD_DIM]`.
        rotary_emb: Optional packed rotary embeddings `[M, HEAD_DIM / 2, 2, 2]`.
        bias: Optional dense output bias `[N]`.
        smooth_factor: Optional smoothing factors `[N]` written for next-layer quantization.
        out_vk: Optional linear-attention VK output buffer.
        out_linearattn: Optional linear-attention output buffer.
        act_unsigned: Whether INT4 activations are stored as unsigned values.
        lora_scales: Optional per-16-rank LoRA scaling factors `[R / 16]`.
        fuse_silu: Whether to fuse SiLU inside supported kernel variants.
        fp4: Whether the packed tensors use FP4/NVFP4 instead of INT4.
        alpha: Per-tensor FP4 scaling factor.
        wcscales: Optional per-channel FP4 scales `[N]`.
        out_q: Optional packed attention-Q output buffer `[B, H, M, D]`.
        out_k: Optional packed attention-K output buffer `[B, H, M, D]`.
        out_v: Optional packed attention-V output buffer `[B, H, M, D]`.
        attn_tokens: Number of attention tokens for fused attention-style kernels.

    Returns:
        None. Results are written in-place to the provided output tensors.
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
