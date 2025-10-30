import torch
from typing import Callable, Optional, List
from cache_dit.logger import init_logger

logger = init_logger(__name__)


def quantize(
    module: torch.nn.Module,
    quant_type: str = "float8_weight_only",
    backend: str = "ao",
    exclude_layers: List[str] = [
        "embedder",
        "embed",
    ],
    filter_fn: Optional[Callable] = None,
    **kwargs,
) -> torch.nn.Module:
    assert isinstance(module, torch.nn.Module)

    if backend.lower() in ("ao", "torchao"):
        from cache_dit.quantize.backends.torchao import quantize_ao

        return quantize_ao(
            module,
            quant_type=quant_type,
            per_row=kwargs.pop("per_row", True),
            exclude_layers=exclude_layers,
            filter_fn=filter_fn,
            **kwargs,
        )
    else:
        raise ValueError(f"backend: {backend} is not supported now!")
