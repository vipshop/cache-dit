import gc
import time
import torch
from cache_dit.logger import init_logger

logger = init_logger(__name__)


def quantize_ao(
    transformer: torch.nn.Module,
    quant_type: str = "fp8_w8a8_dq",
    per_row: bool = True,
    exclude_layers: list[str] = ["embedder", "embed"],
    **kwargs,
) -> torch.nn.Module:
    # Apply FP8 DQ for Transformer and skip any `embed` modules
    # by default to avoid non-trivial precision downgrade. Please
    # set `exclude_layers` as `[]` if you don't want this behavior.
    quant_type = quant_type.lower()
    assert quant_type in (
        "fp8_w8a8_dq",
        "fp8_w8a16_wo",
        "int8_w8a8_dq",
        "int8_w8a16_wo",
        "int4_w4a8_dq",
        "int4_w4a4_dq",
        "int4_w4a16_wo",
    )
    if "fp8" in quant_type:
        assert torch.cuda.get_device_capability() >= (
            8,
            9,
        ), "FP8 is not supported for current device."

    from torchao.quantization import quantize_

    num_quant_linear = 0
    num_skip_linear = 0
    num_linear_layers = 0
    num_layers = 0

    # Ensure bfloat16 for per_row
    def filter_fn(m: torch.nn.Module, name: str) -> bool:
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

    def get_quantization_fn():
        try:
            if quant_type == "fp8_w8a8_dq":
                from torchao.quantization import (
                    float8_dynamic_activation_float8_weight,
                    PerTensor,
                    PerRow,
                )

                quantization_fn = float8_dynamic_activation_float8_weight(
                    granularity=(
                        ((PerRow(), PerRow()))
                        if per_row
                        else ((PerTensor(), PerTensor()))
                    )
                )

            elif quant_type == "fp8_w8a16_wo":
                from torchao.quantization import float8_weight_only

                quantization_fn = float8_weight_only()

            elif quant_type == "int8_w8a8_dq":
                from torchao.quantization import (
                    int8_dynamic_activation_int8_weight,
                )

                quantization_fn = int8_dynamic_activation_int8_weight()

            elif quant_type == "int8_w8a16_wo":
                from torchao.quantization import int8_weight_only

                quantization_fn = int8_weight_only()

            elif quant_type == "int4_w4a8_dq":
                from torchao.quantization import (
                    int8_dynamic_activation_int4_weight,
                )

                quantization_fn = int8_dynamic_activation_int4_weight()

            elif quant_type == "int4_w4a4_dq":
                from torchao.quantization import (
                    int4_dynamic_activation_int4_weight,
                )

                quantization_fn = int4_dynamic_activation_int4_weight()

            elif quant_type == "int4_w4a16_wo":
                from torchao.quantization import int4_weight_only

                quantization_fn = int4_weight_only()
            else:
                raise ValueError(
                    f"quant_type: {quant_type} is not supported now!"
                )
        except ImportError as e:
            e.msg += f"<quant_type: {quant_type} is not supported in torchao backend now!>"
            raise e

        return quantization_fn

    quantize_(transformer, get_quantization_fn(), filter_fn=filter_fn)
    force_empty_cache()

    logger.info(
        f"FP8 DQ  Linear Layers: {num_quant_linear:>5}\n"
        f"Skipped Linear Layers: {num_skip_linear:>5}\n"
        f"Total   Linear Layers: {num_linear_layers:>5}\n"
        f"Total   (all)  Layers: {num_layers:>5}"
    )
    return transformer


def force_empty_cache():
    time.sleep(1)
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)
    gc.collect()
    torch.cuda.empty_cache()
