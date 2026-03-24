import torch
import copy
import dataclasses
from functools import partial
from typing import Callable, Optional, List
from ...utils import maybe_empty_cache
from ...platforms import current_platform
from ...logger import init_logger

logger = init_logger(__name__)


def quantize_ao(
    module: torch.nn.Module,
    quant_type: str = "float8_weight_only",
    # Paramters for FP8 DQ quantization
    # Whether to quantize per row (True) or per tensor (False)
    per_row: bool = True,  # Deprecated, will be removed in future.
    exclude_layers: List[str] = [
        "embedder",
        "embed",
        "modulation",
        "norm",
        "mod",
    ],
    filter_fn: Optional[Callable] = None,
    verbose: bool = False,
    **kwargs,
) -> torch.nn.Module:
    from torchao.quantization import quantize_

    # Check if already quantized by checking the _is_quantized attribute.
    # This is to avoid redundant quantization which may cause performance
    # regression and other issues. If you want to quantize an already quantized.
    if not _check_if_module_can_quantized(module):
        return module

    quant_info = QuantizeInfo(
        quant_type=quant_type,
        per_row=per_row,
        exclude_layers=exclude_layers,
        verbose=verbose,
        kwargs=kwargs,
    )

    _normalize_quantize_info(module, quant_info)
    if quant_info.per_row:  # Ensure bfloat16
        module.to(torch.bfloat16)

    quantize_(
        module,
        _quant_config_impl(quant_info),
        filter_fn=(
            partial(
                _filter_fn_impl,
                quant_info=quant_info,
            )
            if filter_fn is None
            else filter_fn
        ),
        device=kwargs.get("device", None),
    )

    maybe_empty_cache()

    logger.info(
        f"Quantized        Module: {module.__class__.__name__:>5}\n"
        f"Quantized        Method: {quant_info.quant_type_rev:>5}\n"
        f"Quantized Linear Layers: {quant_info.num_quant_linear:>5}\n"
        f"Skipped   Linear Layers: {quant_info.num_skip_linear:>5}\n"
        f"Total     Linear Layers: {quant_info.num_linear_layers:>5}\n"
        f"Total     (all)  Layers: {quant_info.num_layers:>5}"
    )

    if verbose:
        logger.info(f"Skipped        Patterns: {quant_info.exclude_layers}")

    module._quantize_type = quant_type
    module._exclude_for_quantize = copy.deepcopy(quant_info.exclude_layers)
    module._is_quantized = True
    return module


@dataclasses.dataclass
class QuantizeInfo:
    quant_type: str = "fp8_w8a8_dq"
    quant_type_rev: str = "float8"
    per_row: bool = True
    exclude_layers: List[str] = dataclasses.field(default_factory=list)
    verbose: bool = False
    num_quant_linear: int = 0
    num_skip_linear: int = 0
    num_linear_layers: int = 0
    num_layers: int = 0
    kwargs: dict = dataclasses.field(default_factory=dict)


def _check_if_module_can_quantized(module: torch.nn.Module) -> bool:
    from ...utils import check_quantized

    if check_quantized(module):
        module_cls_name = module.__class__.__name__
        logger.warning(f"Module {module_cls_name} is already quantized, skipping. ")
        return False

    # Apply FP8 DQ for module and skip any `embed` modules
    # by default to avoid non-trivial precision downgrade. Please
    # set `exclude_layers` as `[]` if you don't want this behavior.
    assert isinstance(module, torch.nn.Module)
    assert (
        current_platform.is_accelerator_available() and current_platform.device_type == "cuda"
    ), "Quantization functionality with torchao backend is only supported on CUDA devices."
    try:
        import torchao  # noqa: F401
    except ImportError:
        raise ImportError(
            "Quantization functionality requires the 'quantization' extra dependencies. "
            "Install with: pip install cache-dit[quantization]"
        )

    return True


def _normalize_quantize_info(
    module: torch.nn.Module,
    quant_info: QuantizeInfo,
    **kwargs,
) -> QuantizeInfo:

    alias_map = {
        "float8": "fp8_w8a8_dq",
        "float8_blockwise": "fp8_blockwise",
        "float8_weight_only": "fp8_w8a16_wo",
        "float8_wo": "fp8_w8a16_wo",
        "int8": "int8_w8a8_dq",
        "int8_weight_only": "int8_w8a16_wo",
        "int8_wo": "int8_w8a16_wo",
        "int4": "int4_w4a8_dq",
        "int4_weight_only": "int4_w4a16_wo",
        "int4_wo": "int4_w4a16_wo",
    }
    alias_map_rev = copy.deepcopy(alias_map)
    # remove duplicates *_wo in rev map
    for key in list(alias_map_rev.keys()):
        if key.endswith("_wo"):
            alias_map_rev.pop(key)
    alias_map_rev = {v: k for k, v in alias_map_rev.items()}

    quant_type = quant_info.quant_type
    if quant_type.lower() in alias_map:
        quant_type = alias_map[quant_type.lower()]

    quant_type = quant_type.lower()
    assert quant_type in (
        "fp8_w8a8_dq",
        "fp8_w8a16_wo",
        "fp8_blockwise",
        "int8_w8a8_dq",
        "int8_w8a16_wo",
        "int4_w4a8_dq",
        "int4_w4a16_wo",
    ), f"{quant_type} is not supported for torchao backend now!"

    if "fp8" in quant_type:
        assert current_platform.get_device_capability() >= (
            8,
            9,
        ), "FP8 is not supported for current device."

    quant_info.quant_type = quant_type
    quant_info.quant_type_rev = alias_map_rev.get(quant_type, quant_type)

    if hasattr(module, "_exclude_for_quantize"):
        # Workaround for case: TP -> FP8 DQ per row, make torch._scaled_mm happy.
        # Avoid error: "RuntimeError: Expected b.stride(0) == 1 to be true, but got false"
        # use_local_tensor = True (default) in RowwiseParallel (TP) will cause the layout
        # of the linear weights changedly after '_dispatch_get_local_results_slow_path',
        # Why??? Need further investigation.
        if quant_info.quant_type == "fp8_w8a8_dq" and quant_info.per_row:
            exclude_layers = exclude_layers + module._exclude_for_quantize
            logger.info(
                f"Found extra excluding layers (TP) for {module.__class__.__name__}: "
                f"{module._exclude_for_quantize}"
            )
            quant_info.exclude_layers = copy.deepcopy(exclude_layers)

    return quant_info


def _quant_config_impl(quant_info: QuantizeInfo, **kwargs):
    try:
        if quant_info.quant_type == "fp8_w8a8_dq":
            from torchao.quantization import (
                Float8DynamicActivationFloat8WeightConfig,
                PerTensor,
                PerRow,
            )

            # if quant_info.per_row:  # Ensure bfloat16
            #     module.to(torch.bfloat16)

            quant_config = Float8DynamicActivationFloat8WeightConfig(
                weight_dtype=kwargs.get(
                    "weight_dtype",
                    torch.float8_e4m3fn,
                ),
                activation_dtype=kwargs.get(
                    "activation_dtype",
                    torch.float8_e4m3fn,
                ),
                granularity=(
                    ((PerRow(), PerRow())) if quant_info.per_row else ((PerTensor(), PerTensor()))
                ),
            )

        elif quant_info.quant_type == "fp8_blockwise":
            try:
                from torchao.quantization import (
                    Float8DynamicActivationFloat8WeightConfig,
                    PerBlock,
                )
            except ImportError:
                raise ImportError(
                    "Blockwise quantization is not supported in current version of torchao. "
                    "Please upgrade the torchao library to use this feature."
                )
            quant_config = Float8DynamicActivationFloat8WeightConfig(
                weight_dtype=kwargs.get(
                    "weight_dtype",
                    torch.float8_e4m3fn,
                ),
                activation_dtype=kwargs.get(
                    "activation_dtype",
                    torch.float8_e4m3fn,
                ),
                # Currently, torchao only supports blockwise FP8 quantization for linear
                # layers with weight tensors that are divisible by block size (128, 128).
                # We will check the block size of the weight tensor and skip quantization
                # if it's not supported. Only '_granularity_is_a_1_128_w_128_128' pattern
                # is supported now, we will add more patterns in the future once torchao
                # supports more blockwise FP8 quantization patterns.
                granularity=((PerBlock([1, 128]), PerBlock([128, 128]))),  # hardcode
            )

        elif quant_info.quant_type == "fp8_w8a16_wo":
            from torchao.quantization import Float8WeightOnlyConfig

            quant_config = Float8WeightOnlyConfig(
                weight_dtype=kwargs.get(
                    "weight_dtype",
                    torch.float8_e4m3fn,
                ),
            )

        elif quant_info.quant_type == "int8_w8a8_dq":
            from torchao.quantization import (
                Int8DynamicActivationInt8WeightConfig,
            )

            quant_config = Int8DynamicActivationInt8WeightConfig()

        elif quant_info.quant_type == "int8_w8a16_wo":

            from torchao.quantization import Int8WeightOnlyConfig

            quant_config = Int8WeightOnlyConfig(
                # group_size is None -> per_channel, else per group
                group_size=kwargs.get("group_size", None),
            )
        elif quant_info.quant_type == "int4_w4a16_wo":

            from torchao.quantization import Int4WeightOnlyConfig

            quant_config = Int4WeightOnlyConfig(
                group_size=kwargs.get("group_size", 32),
            )

        else:
            raise ValueError(f"quant_type: {quant_info.quant_type} is not supported now!")

    except ImportError as e:
        e.msg += (
            f"{quant_info.quant_type} is not supported in torchao backend now! "
            "Please consider to use another quantization type instead."
        )
        raise e

    return quant_config


def _filter_fn_impl(
    m: torch.nn.Module,
    name: str,
    quant_info: QuantizeInfo = QuantizeInfo(),
) -> bool:
    from torchao.float8.float8_linear import Float8Linear

    quant_info.num_layers += 1
    if isinstance(m, torch.nn.Linear) and not isinstance(m, Float8Linear):
        quant_info.num_linear_layers += 1

        for exclude_name in quant_info.exclude_layers:
            if exclude_name in name:
                if quant_info.verbose:
                    logger.info(f"Skip Quantization: {name} -> pattern<{exclude_name}>")

                quant_info.num_skip_linear += 1
                return False

        if (
            quant_info.per_row
            and m.weight.dtype != torch.bfloat16
            and quant_info.quant_type == "fp8_w8a8_dq"
        ):
            if quant_info.verbose:
                logger.info(
                    f"Skip Quantization: {name} -> pattern<dtype({m.weight.dtype})!=bfloat16>"
                )

            quant_info.num_skip_linear += 1
            return False

        # check blockwise fp8 support for linear layers, if not supported,
        # skip quantization for that layer.
        if quant_info.quant_type in [
            "fp8_blockwise",
        ] and not _check_if_linear_fp8_blockwise_can_support(m):
            weight_shape = tuple(m.weight.shape)
            if quant_info.verbose:
                logger.info(
                    f"Skip Quantization: {name} -> pattern<w{weight_shape} "
                    f"% block_size(128, 128) != 0>"
                )
            quant_info.num_skip_linear += 1
            return False

        if quant_info.quant_type in [
            "fp8_w8a8_dq",
            "fp8_blockwise",
        ] and not _check_if_linear_with_bias_fp8_can_support(m):
            if quant_info.verbose:
                logger.info(
                    f"Skip Quantization: {name} -> "
                    f"pattern<DTensor + bias is not supported for _scaled_mm>"
                )
            quant_info.num_skip_linear += 1
            return False

        quant_info.num_quant_linear += 1
        return True

    return False


def _check_if_linear_fp8_blockwise_can_support(module: torch.nn.Linear) -> bool:
    try:
        from torchao.quantization.utils import get_block_size
        from torchao.quantization import PerBlock
    except ImportError:
        return False

    weight_tensor = getattr(module, "weight", None)  # type: torch.Tensor
    if weight_tensor is None:
        return False

    # Currently, torchao only supports blockwise FP8 quantization for linear
    # layers with weight tensors that are divisible by block size (128, 128).
    # We will check the block size of the weight tensor and skip quantization
    # if it's not supported. Only '_granularity_is_a_1_128_w_128_128' pattern
    # is supported now, we will add more patterns in the future once torchao
    # supports more blockwise FP8 quantization patterns.
    weight_granularity = PerBlock([128, 128])  # hardcode
    try:
        block_size = get_block_size(weight_tensor.shape, weight_granularity)
        logger.debug(
            f"block_size: {block_size}, weight_granularity.block_size: "
            f"{weight_granularity.block_size}"
        )
        return block_size == weight_granularity.block_size
    except Exception as e:
        logger.debug(f"Failed to get block size for module {module}: {e}")
        return False


def _check_if_linear_with_bias_fp8_can_support(module: torch.nn.Linear) -> bool:
    # Avoid: AssertionError("_scaled_mm on DTensors doesn't support bias")
    # Check we are in distributed environment and the linear layer has bias,
    # and if the weight is DTensor, if all conditions are met, we will skip
    # quantization for that layer to avoid potential issues.
    if not torch.distributed.is_initialized():
        return True
    # If the linear layer doesn't have bias, we can quantize it without issues.
    if not hasattr(module, "bias") or module.bias is None:
        return True
    # For the case where the linear layer has bias, we need to check if the weight
    # or bias is DTensor. We only quantize the linear layer when both weight and
    # bias are not DTensor.
    from torch.distributed._tensor import DTensor

    weight_tensor = getattr(module, "weight", None)
    bias_tensor = getattr(module, "bias", None)
    if weight_tensor is None or bias_tensor is None:
        return False
    return not isinstance(weight_tensor, DTensor) and not isinstance(bias_tensor, DTensor)
