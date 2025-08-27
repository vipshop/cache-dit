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
    transformer: torch.nn.Module, filter_fn=None
) -> torch.nn.Module:
    assert torch.cuda.get_device_capability() >= (
        8,
        9,
    ), "FP8 is not supported for current device."
    from torchao.quantization import (
        float8_dynamic_activation_float8_weight,
        PerTensor,
        quantize_,
    )

    quantization_fn = float8_dynamic_activation_float8_weight(
        granularity=((PerTensor(), PerTensor()))
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
