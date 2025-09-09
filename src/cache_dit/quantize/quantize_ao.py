import gc
import time
import torch
from typing import Callable, Optional, List
from cache_dit.logger import init_logger

logger = init_logger(__name__)


def quantize_ao(
    module: torch.nn.Module,
    quant_type: str = "fp8_w8a8_dq",
    exclude_layers: List[str] = [
        "embedder",
        "embed",
    ],
    filter_fn: Optional[Callable] = None,
    # paramters for fp8 quantization
    per_row: bool = True,
    **kwargs,
) -> torch.nn.Module:
    # Apply FP8 DQ for module and skip any `embed` modules
    # by default to avoid non-trivial precision downgrade. Please
    # set `exclude_layers` as `[]` if you don't want this behavior.
    assert isinstance(module, torch.nn.Module)

    quant_type = quant_type.lower()
    assert quant_type in (
        "fp8_w8a8_dq",
        "fp8_w8a16_wo",
        "int8_w8a8_dq",
        "int8_w8a16_wo",
        "int4_w4a8_dq",
        "int4_w4a4_dq",
        "int4_w4a16_wo",
    ), f"{quant_type} is not supported for torchao backend now!"

    if "fp8" in quant_type:
        assert torch.cuda.get_device_capability() >= (
            8,
            9,
        ), "FP8 is not supported for current device."

    num_quant_linear = 0
    num_skip_linear = 0
    num_linear_layers = 0
    num_layers = 0

    # Ensure bfloat16 for per_row
    def _filter_fn(m: torch.nn.Module, name: str) -> bool:
        nonlocal num_quant_linear, num_skip_linear, num_linear_layers, num_layers
        num_layers += 1
        if isinstance(m, torch.nn.Linear):
            num_linear_layers += 1
            for exclude_name in exclude_layers:
                if exclude_name in name:
                    logger.info(
                        f"Skip Quantization: {name} -> "
                        f"pattern<{exclude_name}>"
                    )

                    num_skip_linear += 1
                    return False

            if (
                per_row
                and m.weight.dtype != torch.bfloat16
                and quant_type == "fp8_w8a8_dq"
            ):
                logger.info(
                    f"Skip Quantization: {name} -> "
                    f"pattern<dtype({m.weight.dtype})!=bfloat16>"
                )

                num_skip_linear += 1
                return False

            num_quant_linear += 1
            return True

        return False

    def _quantization_fn():
        try:
            if quant_type == "fp8_w8a8_dq":
                from torchao.quantization import (
                    float8_dynamic_activation_float8_weight,
                    PerTensor,
                    PerRow,
                )

                if per_row:  # Ensure bfloat16
                    module.to(torch.bfloat16)

                quantization_fn = float8_dynamic_activation_float8_weight(
                    weight_dtype=kwargs.get(
                        "weight_dtype",
                        torch.float8_e4m3fn,
                    ),
                    activation_dtype=kwargs.get(
                        "activation_dtype",
                        torch.float8_e4m3fn,
                    ),
                    granularity=(
                        ((PerRow(), PerRow()))
                        if per_row
                        else ((PerTensor(), PerTensor()))
                    ),
                )

            elif quant_type == "fp8_w8a16_wo":
                from torchao.quantization import float8_weight_only

                quantization_fn = float8_weight_only(
                    weight_dtype=kwargs.get(
                        "weight_dtype",
                        torch.float8_e4m3fn,
                    ),
                )

            elif quant_type == "int8_w8a8_dq":
                from torchao.quantization import (
                    int8_dynamic_activation_int8_weight,
                )

                quantization_fn = int8_dynamic_activation_int8_weight()

            elif quant_type == "int8_w8a16_wo":
                from torchao.quantization import int8_weight_only

                quantization_fn = int8_weight_only(
                    # group_size is None -> per_channel, else per group
                    group_size=kwargs.get("group_size", None),
                )

            elif quant_type == "int4_w4a8_dq":
                from torchao.quantization import (
                    int8_dynamic_activation_int4_weight,
                )

                quantization_fn = int8_dynamic_activation_int4_weight(
                    group_size=kwargs.get("group_size", 32),
                )

            elif quant_type == "int4_w4a4_dq":
                from torchao.quantization import (
                    int4_dynamic_activation_int4_weight,
                )

                quantization_fn = int4_dynamic_activation_int4_weight()

            elif quant_type == "int4_w4a16_wo":
                from torchao.quantization import int4_weight_only

                quantization_fn = int4_weight_only(
                    group_size=kwargs.get("group_size", 32),
                )

            else:
                raise ValueError(
                    f"quant_type: {quant_type} is not supported now!"
                )

        except ImportError as e:
            e.msg += (
                f"{quant_type} is not supported in torchao backend now! "
                "Please upgrade the torchao library."
            )
            raise e

        return quantization_fn

    from torchao.quantization import quantize_

    quantize_(
        module,
        _quantization_fn(),
        filter_fn=_filter_fn if filter_fn is None else filter_fn,
        device=kwargs.get("device", None),
    )

    force_empty_cache()

    logger.info(
        f"Quantized        Method: {quant_type:>5}\n"
        f"Quantized Linear Layers: {num_quant_linear:>5}\n"
        f"Skipped   Linear Layers: {num_skip_linear:>5}\n"
        f"Total     Linear Layers: {num_linear_layers:>5}\n"
        f"Total     (all)  Layers: {num_layers:>5}"
    )
    return module


def force_empty_cache():
    time.sleep(1)
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)
    gc.collect()
    torch.cuda.empty_cache()
