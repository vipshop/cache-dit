import torch
from typing import Callable, Optional, List
from cache_dit.logger import init_logger

logger = init_logger(__name__)


def quantize(
    module: torch.nn.Module,
    quant_type: str = "fp8_w8a8_dq",
    backend: str = "ao",
    exclude_layers: List[str] = [
        "embedder",
        "embed",
    ],
    filter_fn: Optional[Callable] = None,
    # only for fp8_w8a8_dq
    per_row: bool = True,
    **kwargs,
) -> torch.nn.Module:
    assert isinstance(module, torch.nn.Module)

    if backend.lower() in ("ao", "torchao"):
        from cache_dit.quantize.quantize_ao import quantize_ao

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

        return quantize_ao(
            module,
            quant_type=quant_type,
            per_row=per_row,
            exclude_layers=exclude_layers,
            filter_fn=filter_fn,
            **kwargs,
        )
    else:
        raise ValueError(f"backend: {backend} is not supported now!")
