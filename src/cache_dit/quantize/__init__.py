import torch
import copy
from typing import Callable, Optional, List
from cache_dit.logger import init_logger
from .utils import normalize_quantize_type
from .config import QuantizeConfig

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
    verbose: bool = False,
    quantize_config: Optional[QuantizeConfig] = None,
    **kwargs,
) -> torch.nn.Module:

    # Qwen-Image will generate nan after quantization with per_row quantization.
    # So, we disable per_row quantization for all layers in Qwen-Image for better
    # stability.
    _class_not_supported_per_row = [
        "QwenImageTransformer2DModel",
    ]

    def is_per_row_supported(m: torch.nn.Module) -> bool:
        return m.__class__.__name__ not in _class_not_supported_per_row

    if quantize_config is not None:
        # If quantize_config is provided, it will override the individual parameters
        backend = quantize_config.backend
        quant_type = quantize_config.quant_type
        per_row = quantize_config.per_row
        exclude_layers = quantize_config.exclude_layers
        filter_fn = quantize_config.filter_fn
        verbose = quantize_config.verbose

    per_row = per_row and is_per_row_supported(module)

    module = quantize_(
        module,
        quant_type=quant_type,
        backend=backend,
        per_row=per_row,
        exclude_layers=exclude_layers,
        filter_fn=filter_fn,
        verbose=verbose,
        **kwargs,
    )

    module._quantize_config = quantize_config or QuantizeConfig(
        quant_type=quant_type,
        backend=backend,
        per_row=per_row,
        exclude_layers=exclude_layers,
        filter_fn=filter_fn,
        verbose=verbose,
    )

    return module


def quantize_(
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
    verbose: bool = False,
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
            verbose=verbose,
            **kwargs,
        )
    else:
        raise ValueError(f"backend: {backend} is not supported now!")


def remove_quantization_stats(module: torch.nn.Module) -> torch.nn.Module:
    components_to_quantize = None
    if hasattr(module, "_quantize_config"):
        components_to_quantize = copy.deepcopy(
            module._quantize_config.components_to_quantize,
        )

    def _remove_quantization_stats(module: torch.nn.Module) -> None:
        if hasattr(module, "_quantize_config"):
            del module._quantize_config
        if hasattr(module, "_is_quantized"):
            del module._is_quantized
        if hasattr(module, "_quantize_type"):
            del module._quantize_type

    _remove_quantization_stats(module)

    # Only use 1 depth for the recursion of removing stats in sub modules.
    if components_to_quantize is not None:
        from ..utils import parse_extra_modules

        extra_modules = parse_extra_modules(module, components_to_quantize)
        for extra_module in extra_modules:
            _remove_quantization_stats(extra_module)
    return module
