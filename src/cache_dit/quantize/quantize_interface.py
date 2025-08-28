import torch
from cache_dit.logger import init_logger

logger = init_logger(__name__)


def quantize(
    transformer: torch.nn.Module,
    quant_type: str = "fp8_dq",
    **kwargs,
) -> torch.nn.Module:
    if quant_type.lower() == "fp8_dq":
        from cache_dit.quantize.quantize_fp8 import quantize_fp8

        return quantize_fp8(transformer, **kwargs)
    else:
        raise ValueError(f"quant_type: {quant_type} is not supported now!")
