import torch
from typing import Callable, Optional, List
from cache_dit.logger import init_logger
from .utils import normalize_quantize_type

logger = init_logger(__name__)


def quantize(
    module: torch.nn.Module,
    quant_type: Optional[str] = None,
    backend: str = "ao",
    # Specific parameters for torchao backend
    per_row: bool = True,
    exclude_layers: List[str] = [
        "embedder",
        "embed",
    ],
    filter_fn: Optional[Callable] = None,
    **kwargs,
) -> torch.nn.Module:
    assert isinstance(module, torch.nn.Module)

    if quant_type is None:
        quant_type = "float8_weight_only"
        logger.warning(f"quant_type is not specified, using default: {quant_type}")

    if backend.lower() in ("ao", "torchao"):
        from .torchao import quantize_ao

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
