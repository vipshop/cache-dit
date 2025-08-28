import gc
import time
import torch
from cache_dit.logger import init_logger

logger = init_logger(__name__)


def quantize_fp8(
    transformer: torch.nn.Module,
    per_row: bool = True,
    exclude_layers: list[str] = ["embedder", "embed", "attn"],
) -> torch.nn.Module:
    assert torch.cuda.get_device_capability() >= (
        8,
        9,
    ), "FP8 is not supported for current device."
    from torchao.quantization import (
        float8_dynamic_activation_float8_weight,
        PerTensor,
        PerRow,
        quantize_,
    )

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

            if per_row and m.weight.dtype != torch.bfloat16:
                logger.info(
                    f"Skip Quantization: {name} -> "
                    f"pattern<dtype({m.weight.dtype})!=bfloat16>"
                )

                num_skip_linear += 1
                return False

            num_quant_linear += 1
            return True

        return False

    quantization_fn = float8_dynamic_activation_float8_weight(
        granularity=(
            ((PerRow(), PerRow())) if per_row else ((PerTensor(), PerTensor()))
        )
    )
    quantize_(transformer, quantization_fn, filter_fn=filter_fn)
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
