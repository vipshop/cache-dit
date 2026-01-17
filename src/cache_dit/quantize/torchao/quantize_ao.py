import torch
import copy
from typing import Callable, Optional, List
from ...utils import maybe_empty_cache
from ...platforms import current_platform
from cache_dit.logger import init_logger

logger = init_logger(__name__)


def quantize_ao(
    module: torch.nn.Module,
    quant_type: str = "float8_weight_only",
    # Paramters for FP8 DQ quantization
    # Whether to quantize per row (True) or per tensor (False)
    per_row: bool = True,
    exclude_layers: List[str] = [
        "embedder",
        "embed",
    ],
    filter_fn: Optional[Callable] = None,
    **kwargs,
) -> torch.nn.Module:
    # Apply FP8 DQ for module and skip any `embed` modules
    # by default to avoid non-trivial precision downgrade. Please
    # set `exclude_layers` as `[]` if you don't want this behavior.
    assert isinstance(module, torch.nn.Module)
    assert (
        current_platform.is_accelerator_available() and current_platform.device_type == "cuda"
    ), "Quantization functionality with torchao backend is only supported on CUDA devices."
    try:
        import torchao  # noqa: F401
    except ImportError:
        raise ImportError(
            "Quantization functionality requires the 'quantization' extra dependencies. "
            "Install with: pip install cache-dit[quantization]"
        )

    alias_map = {
        "float8": "fp8_w8a8_dq",
        "float8_weight_only": "fp8_w8a16_wo",
        "float8_wo": "fp8_w8a16_wo",
        "int8": "int8_w8a8_dq",
        "int8_weight_only": "int8_w8a16_wo",
        "int8_wo": "int8_w8a16_wo",
        "int4": "int4_w4a8_dq",
        "int4_w4a4": "int4_w4a4_dq",
        "int4_weight_only": "int4_w4a16_wo",
        "int4_wo": "int4_w4a16_wo",
    }
    alias_map_rev = copy.deepcopy(alias_map)
    # remove duplicates *_wo in rev map
    for key in list(alias_map_rev.keys()):
        if key.endswith("_wo"):
            alias_map_rev.pop(key)
    alias_map_rev = {v: k for k, v in alias_map_rev.items()}

    if quant_type.lower() in alias_map:
        quant_type = alias_map[quant_type.lower()]

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
        assert current_platform.get_device_capability() >= (
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
                    logger.info(f"Skip Quantization: {name} -> " f"pattern<{exclude_name}>")

                    num_skip_linear += 1
                    return False

            if per_row and m.weight.dtype != torch.bfloat16 and quant_type == "fp8_w8a8_dq":
                logger.info(
                    f"Skip Quantization: {name} -> " f"pattern<dtype({m.weight.dtype})!=bfloat16>"
                )

                num_skip_linear += 1
                return False

            num_quant_linear += 1
            return True

        return False

    def _quant_config():
        try:
            if quant_type == "fp8_w8a8_dq":
                from torchao.quantization import (
                    Float8DynamicActivationFloat8WeightConfig,
                    PerTensor,
                    PerRow,
                )

                if per_row:  # Ensure bfloat16
                    module.to(torch.bfloat16)

                quant_config = Float8DynamicActivationFloat8WeightConfig(
                    weight_dtype=kwargs.get(
                        "weight_dtype",
                        torch.float8_e4m3fn,
                    ),
                    activation_dtype=kwargs.get(
                        "activation_dtype",
                        torch.float8_e4m3fn,
                    ),
                    granularity=(
                        ((PerRow(), PerRow())) if per_row else ((PerTensor(), PerTensor()))
                    ),
                )

            elif quant_type == "fp8_w8a16_wo":
                from torchao.quantization import Float8WeightOnlyConfig

                quant_config = Float8WeightOnlyConfig(
                    weight_dtype=kwargs.get(
                        "weight_dtype",
                        torch.float8_e4m3fn,
                    ),
                )

            elif quant_type == "int8_w8a8_dq":
                from torchao.quantization import (
                    Int8DynamicActivationInt8WeightConfig,
                )

                quant_config = Int8DynamicActivationInt8WeightConfig()

            elif quant_type == "int8_w8a16_wo":

                from torchao.quantization import Int8WeightOnlyConfig

                quant_config = Int8WeightOnlyConfig(
                    # group_size is None -> per_channel, else per group
                    group_size=kwargs.get("group_size", None),
                )

            elif quant_type == "int4_w4a8_dq":

                from torchao.quantization import (
                    Int8DynamicActivationInt4WeightConfig,
                )

                quant_config = Int8DynamicActivationInt4WeightConfig(
                    group_size=kwargs.get("group_size", 32),
                )

            elif quant_type == "int4_w4a4_dq":

                from torchao.quantization import (
                    Int4DynamicActivationInt4WeightConfig,
                )

                quant_config = Int4DynamicActivationInt4WeightConfig()

            elif quant_type == "int4_w4a16_wo":

                from torchao.quantization import Int4WeightOnlyConfig

                quant_config = Int4WeightOnlyConfig(
                    group_size=kwargs.get("group_size", 32),
                )

            else:
                raise ValueError(f"quant_type: {quant_type} is not supported now!")

        except ImportError as e:
            e.msg += (
                f"{quant_type} is not supported in torchao backend now! "
                "Please upgrade the torchao library."
            )
            raise e

        return quant_config

    from torchao.quantization import quantize_

    quantize_(
        module,
        _quant_config(),
        filter_fn=_filter_fn if filter_fn is None else filter_fn,
        device=kwargs.get("device", None),
    )

    maybe_empty_cache()

    if quant_type in alias_map_rev:
        quant_type = alias_map_rev[quant_type]

    logger.info(
        f"Quantized        Module: {module.__class__.__name__:>5}\n"
        f"Quantized        Method: {quant_type:>5}\n"
        f"Quantized Linear Layers: {num_quant_linear:>5}\n"
        f"Skipped   Linear Layers: {num_skip_linear:>5}\n"
        f"Total     Linear Layers: {num_linear_layers:>5}\n"
        f"Total     (all)  Layers: {num_layers:>5}"
    )

    module._quantize_type = quant_type
    module._is_quantized = True
    return module
