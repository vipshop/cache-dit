import gc
import time
import torch
import argparse


def GiB():
    if not torch.cuda.is_available():
        return 0

    try:
        total_memory_bytes = torch.cuda.get_device_properties(
            torch.cuda.current_device(),
        ).total_memory
        total_memory_gib = total_memory_bytes / (1024**3)
        return int(total_memory_gib)
    except Exception:
        return 0


def force_empty_cache():
    time.sleep(1)
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(1)
    gc.collect()
    torch.cuda.empty_cache()


def quantize_fp8(
    transformer: torch.nn.Module,
    per_row: bool = True,
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

    # Ensure bfloat16 for per_row
    def filter_fn(m: torch.nn.Module, name: str) -> bool:
        exclude_layers = ["embedder", "embed", "attn"]
        if isinstance(m, torch.nn.Linear):
            for exclude_name in exclude_layers:
                if exclude_name in name:
                    print(
                        f"Skip Quantization: {name} -> "
                        f"pattern<{exclude_name}>"
                    )
                    return False
            if per_row and m.weight.dtype != torch.bfloat16:
                print(
                    f"Skip Quantization: {name} -> "
                    f"pattern<dtype({m.weight.dtype})!=bfloat16>"
                )
                return False
            return True
        return False

    quantization_fn = float8_dynamic_activation_float8_weight(
        granularity=(
            ((PerRow(), PerRow())) if per_row else ((PerTensor(), PerTensor()))
        )
    )
    quantize_(transformer, quantization_fn, filter_fn=filter_fn)
    force_empty_cache()
    return transformer


def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", action="store_true", default=False)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--fp8", action="store_true", default=False)
    return parser.parse_args()
