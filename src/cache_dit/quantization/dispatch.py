import torch
import copy
from typing import Optional
from .backend import QuantizeBackend
from .config import QuantizeConfig
from ..logger import init_logger

logger = init_logger(__name__)


def quantize(
    module: torch.nn.Module,
    quantize_config: Optional[QuantizeConfig] = None,
    **kwargs,
) -> torch.nn.Module:
    # For backward compatibility, we still accept the old style of quantization arguments,
    # but they will be ignored if quantize_config is specified.
    if quantize_config is None:
        if kwargs:
            logger.warning(
                "The quantization arguments in kwargs will be deprecated, "
                "please use QuantizeConfig to specify quantization configurations."
            )
            quantize_config = QuantizeConfig.from_kwargs(**kwargs)
        else:
            raise ValueError("quantize_config should be specified for quantization.")
    else:
        if kwargs:
            logger.warning(
                "The quantization arguments in kwargs will be ignored since "
                "quantize_config is specified, please use QuantizeConfig to "
                "specify quantization configurations."
            )

    # Dispatch to different quantization backends according to the quantize_config.
    # Currently we only support torchao as the quantization backend, and we may
    # support more backends in the future.
    if quantize_config.backend == QuantizeBackend.TORCHAO:
        from .torchao import quantize_ao

        module = quantize_ao(module, quantize_config, **kwargs)
    elif quantize_config.backend == QuantizeBackend.CACHE_DIT:
        from .svdquant.ptq import quantize_svdq_ptq

        module = quantize_svdq_ptq(module, quantize_config)
    else:
        raise ValueError(f"backend: {quantize_config.backend} is not supported now!")

    return module


def load(
    module: torch.nn.Module,
    quantize_config_or_path: QuantizeConfig | str,
) -> torch.nn.Module:
    if isinstance(quantize_config_or_path, QuantizeConfig):
        backend = quantize_config_or_path.backend
    elif isinstance(quantize_config_or_path, str):
        backend = QuantizeBackend.CACHE_DIT
    else:
        raise TypeError(
            "quantize_config_or_path must be a QuantizeConfig or a serialized checkpoint path, "
            f"got {type(quantize_config_or_path)}."
        )

    if backend == QuantizeBackend.CACHE_DIT:
        from .svdquant.ptq import load_svdq

        return load_svdq(module, quantize_config_or_path)

    raise ValueError(f"backend: {backend} does not support module loading now.")


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
        if hasattr(module, "_exclude_layers"):
            del module._exclude_layers

    _remove_quantization_stats(module)

    # Only use 1 depth for the recursion of removing stats in sub modules.
    if components_to_quantize is not None:
        from ..utils import parse_extra_modules

        extra_modules = parse_extra_modules(module, components_to_quantize)
        for extra_module in extra_modules:
            _remove_quantization_stats(extra_module)
    return module
