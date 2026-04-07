from __future__ import annotations

import typing as tp

import torch
from torch import nn

from .linear import SVDQW4A4Linear
from .lowrank import decompose_lowrank_residual
from .packing import adapt_svdq_module_state_dict
from .packing import export_raw_svdq_w4a4_state_dict

CalibrationInputs = torch.Tensor | tp.Sequence[torch.Tensor]

__all__ = [
    "CalibrationInputs",
    "compute_smooth_scale",
    "quantize_linear_svdq_w4a4",
    "standardize_calibration_activations",
    "validate_svdq_linear_geometry",
]


def validate_svdq_linear_geometry(
    in_features: int,
    out_features: int,
    *,
    rank: int,
    precision: str = "int4",
) -> None:
    if precision != "int4":
        raise NotImplementedError("The minimal SVDQuant quantizer currently supports INT4 only.")
    if in_features % 64 != 0:
        raise ValueError(f"INT4 SVDQuant requires in_features divisible by 64, got {in_features}.")
    if in_features % 128 != 0:
        raise ValueError(
            f"The migrated W4A4 packer/runtime requires in_features divisible by 128, got {in_features}."
        )
    if out_features % 128 != 0:
        raise ValueError(
            f"The migrated W4A4 packer/runtime requires out_features divisible by 128, got {out_features}."
        )
    if rank < 0:
        raise ValueError(f"rank must be non-negative, got {rank}.")
    if rank != 0 and rank % 16 != 0:
        raise ValueError(
            f"The migrated W4A4 runtime requires rank 0 or a multiple of 16, got {rank}."
        )


def _normalize_dtype(torch_dtype: torch.dtype | None, device: torch.device | str) -> torch.dtype:
    if torch_dtype is not None:
        if torch_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"torch_dtype must be torch.float16 or torch.bfloat16, got {torch_dtype}."
            )
        return torch_dtype
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


@torch.inference_mode()
def standardize_calibration_activations(
    representative_activations: CalibrationInputs,
    *,
    in_features: int,
) -> list[torch.Tensor]:
    tensors = (
        (representative_activations,)
        if isinstance(representative_activations, torch.Tensor)
        else representative_activations
    )
    standardized: list[torch.Tensor] = []
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("representative_activations must be a tensor or a sequence of tensors.")
        if tensor.shape[-1] != in_features:
            raise ValueError(
                f"Expected representative activations with last dim {in_features}, got {tensor.shape[-1]}."
            )
        standardized.append(tensor.reshape(-1, in_features))
    if not standardized:
        raise ValueError("At least one representative activation tensor is required.")
    return standardized


def _repair_invalid_scale(scale: torch.Tensor) -> torch.Tensor:
    scale = scale.clone()
    scale[scale == 0] = 1
    scale[~torch.isfinite(scale)] = 1
    return scale


def _resolve_math_dtype(torch_dtype: torch.dtype, high_precision: bool) -> torch.dtype:
    return torch.float32 if high_precision else torch_dtype


@torch.inference_mode()
def compute_smooth_scale(
    activation_span: torch.Tensor,
    weight_span: torch.Tensor,
    *,
    alpha: float = 0.5,
    math_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
    beta = 1.0 - alpha
    smooth_scale = activation_span.to(dtype=math_dtype).pow(alpha)
    if beta > 0:
        smooth_scale = smooth_scale.div_(weight_span.to(dtype=math_dtype).pow(beta))
    smooth_scale = _repair_invalid_scale(smooth_scale)
    if output_dtype is not None:
        smooth_scale = smooth_scale.to(dtype=output_dtype)
    return smooth_scale


@torch.inference_mode()
def _compute_group_scales(
    weight: torch.Tensor,
    group_size: int = 64,
    *,
    math_dtype: torch.dtype = torch.float32,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(f"Expected in_features divisible by {group_size}, got {in_features}.")
    weight = weight.to(dtype=math_dtype).view(
        out_features, 1, in_features // group_size, group_size
    )
    scales = weight.abs().amax(dim=-1, keepdim=True).div_(7.0)
    scales = _repair_invalid_scale(scales)
    if output_dtype is not None:
        scales = scales.to(dtype=output_dtype)
    return scales


@torch.inference_mode()
def quantize_linear_svdq_w4a4(
    linear: nn.Linear,
    representative_activations: CalibrationInputs,
    *,
    rank: int = 32,
    alpha: float = 0.5,
    precision: str = "int4",
    act_unsigned: bool = False,
    torch_dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    return_state_dict: bool = False,
    high_precision: bool = False,
    fp32_fallback: bool = False,
    streaming: bool = True,
) -> SVDQW4A4Linear | dict[str, torch.Tensor]:
    if not isinstance(linear, nn.Linear):
        raise TypeError(f"Expected nn.Linear, got {type(linear)}.")

    device = torch.device(device or linear.weight.device)
    torch_dtype = _normalize_dtype(torch_dtype, device)
    math_dtype = _resolve_math_dtype(torch_dtype, high_precision)
    validate_svdq_linear_geometry(
        linear.in_features, linear.out_features, rank=rank, precision=precision
    )

    standardized_acts = standardize_calibration_activations(
        representative_activations,
        in_features=linear.in_features,
    )

    if streaming:
        activation_span: torch.Tensor | None = None
        for tensor in standardized_acts:
            device_tensor = tensor.to(device=device, dtype=torch_dtype)
            tensor_span = device_tensor.to(dtype=math_dtype).abs().amax(dim=0)
            if activation_span is None:
                activation_span = tensor_span
            else:
                activation_span = torch.maximum(activation_span, tensor_span)
            del device_tensor, tensor_span
        if activation_span is None:
            raise ValueError("At least one representative activation tensor is required.")
    else:
        eager_acts = [tensor.to(device=device, dtype=torch_dtype) for tensor in standardized_acts]
        activation_span = torch.stack(
            [tensor.to(dtype=math_dtype).abs().amax(dim=0) for tensor in eager_acts],
            dim=0,
        ).amax(dim=0)
        del eager_acts

    weight = linear.weight.detach().to(device=device, dtype=torch_dtype)
    weight_span = weight.to(dtype=math_dtype).abs().amax(dim=0)
    smooth_scale = compute_smooth_scale(
        activation_span,
        weight_span,
        alpha=alpha,
        math_dtype=math_dtype,
        output_dtype=torch_dtype,
    )

    smoothed_weight = weight.to(dtype=math_dtype) * smooth_scale.to(dtype=math_dtype).unsqueeze(0)
    lowrank_down, lowrank_up, residual = decompose_lowrank_residual(
        smoothed_weight,
        rank,
        output_dtype=torch_dtype,
        high_precision=high_precision,
        fp32_fallback=fp32_fallback,
    )
    scales = _compute_group_scales(
        residual,
        group_size=64,
        math_dtype=math_dtype,
        output_dtype=torch_dtype,
    )

    bias = (
        None if linear.bias is None else linear.bias.detach().to(device=device, dtype=torch_dtype)
    )
    raw_state_dict = export_raw_svdq_w4a4_state_dict(
        residual,
        scale=scales,
        bias=bias,
        smooth=smooth_scale,
        lora=None if rank == 0 else (lowrank_down, lowrank_up),
        float_point=False,
    )
    module_state_dict = adapt_svdq_module_state_dict(
        raw_state_dict,
        in_features=linear.in_features,
        out_features=linear.out_features,
        rank=rank,
        torch_dtype=torch_dtype,
        device=device,
        has_bias=linear.bias is not None,
    )

    if return_state_dict:
        return module_state_dict

    quantized = SVDQW4A4Linear.from_linear(
        linear,
        rank=rank,
        precision=precision,
        act_unsigned=act_unsigned,
        torch_dtype=torch_dtype,
        device=device,
    )
    incompatible = quantized.load_state_dict(module_state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected SVDQuant state_dict mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}."
        )
    return quantized
