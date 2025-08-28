import torch
from cache_dit.logger import init_logger

logger = init_logger(__name__)


def quantize(
    transformer: torch.nn.Module,
    quant_type: str = "fp8_w8a8_dq",
    backend: str = "ao",
    # only for fp8_w8a8_dq
    per_row: bool = True,
    exclude_layers: list[str] = [
        "embedder",
        "embed",
    ],
    **kwargs,
) -> torch.nn.Module:
    if backend.lower() in ("ao", "torchao"):
        from cache_dit.quantize.quantize_ao import quantize_ao

        return quantize_ao(
            transformer,
            quant_type=quant_type,
            per_row=per_row,
            exclude_layers=exclude_layers,
            **kwargs,
        )
    else:
        raise ValueError(f"backend: {backend} is not supported now!")
