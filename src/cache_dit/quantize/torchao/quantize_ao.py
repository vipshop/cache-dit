import torch
import copy
import dataclasses
from functools import partial
from typing import Optional, List
from ..config import QuantizeConfig
from ...utils import maybe_empty_cache
from ...platforms import current_platform
from ...logger import init_logger

logger = init_logger(__name__)


def quantize_ao(
    module: torch.nn.Module,
    quantize_config: QuantizeConfig,
    **kwargs,
) -> torch.nn.Module:
    # Check if already quantized by checking the _is_quantized attribute.
    # This is to avoid redundant quantization which may cause performance
    # regression and other issues. If you want to quantize an already quantized.
    if not _check_if_module_can_quantized(module):
        return module

    quant_ctx = QuantizeAOContext.from_config(quantize_config, module, **kwargs)
    quant_ctx = quant_ctx.normalize(**kwargs)

    def _quantize_module(m: torch.nn.Module):
        from torchao.quantization import quantize_

        quantize_(
            m,
            _quant_config_impl(quant_ctx),
            filter_fn=(
                partial(_filter_fn_impl, quant_ctx=quant_ctx)
                if quantize_config.filter_fn is None
                else quantize_config.filter_fn
            ),
            device=kwargs.get("device", None),
        )

    # Regional quantization for transformer modules in Diffusers, users can
    # set regional_quantize to False to disable this behavior and quantize
    # the whole module directly. For models outside of diffusers, users can specify
    # the repeated blocks by setting repeated_blocks to a list of block names.
    if quant_ctx.regional_quantize:
        assert (
            quant_ctx.repeated_blocks is not None
        ), "repeated_blocks must be specified when regional_quantize is True."
        has_quantized_region = False
        # Reference: https://github.com/huggingface/diffusers/blob
        # /main/src/diffusers/models/modeling_utils.py#L1475
        for submod in module.modules():
            if submod.__class__.__name__ in quant_ctx.repeated_blocks:
                _quantize_module(submod)
                has_quantized_region = True
        if not has_quantized_region:
            raise ValueError(
                f"Regional quantization failed because {quant_ctx.repeated_blocks} "
                "classes are not found in the module."
            )
    else:
        _quantize_module(module)

    maybe_empty_cache()
    quant_ctx.summary()

    module._is_quantized = True
    module._quantize_type = quant_ctx.quant_type
    module._quantize_config = quantize_config
    module._exclude_layers = copy.deepcopy(quant_ctx.exclude_layers)

    return module


@dataclasses.dataclass
class QuantizeAOContext:
    module_ref: torch.nn.Module = None  # ref only
    # quantization config
    quant_type: str = "fp8_w8a8_dq"
    per_row: bool = True
    regional_quantize: bool = True
    repeated_blocks: Optional[List[str]] = None
    exclude_layers: List[str] = dataclasses.field(default_factory=list)
    verbose: bool = False
    # stats for summary
    quant_type_rev: str = "float8"
    num_quant_linear: int = 0
    num_skip_linear: int = 0
    num_linear_layers: int = 0
    num_layers: int = 0
    kwargs: dict = dataclasses.field(default_factory=dict)

    @staticmethod
    def from_config(
        quantize_config: QuantizeConfig,
        module: torch.nn.Module = None,
        **kwargs,
    ) -> "QuantizeAOContext":
        return QuantizeAOContext(
            module_ref=module,  # ref
            quant_type=quantize_config.quant_type,
            per_row=quantize_config.per_row,
            regional_quantize=quantize_config.regional_quantize,
            repeated_blocks=quantize_config.repeated_blocks,
            exclude_layers=quantize_config.exclude_layers,
            verbose=quantize_config.verbose,
            kwargs=copy.deepcopy(kwargs),
        )

    def summary(self):
        quantized_region = (
            f"{self.repeated_blocks}"
            if self.regional_quantize and self.repeated_blocks is not None
            else self.module_ref.__class__.__name__ if self.module_ref else "Module"
        )
        summary_strs = [
            f"Quantized        Method: {self.quant_type_rev}",
            f"Quantized        Region: {quantized_region}",
            f"Quantized Linear Layers: {self.num_quant_linear:<5}",
            f"Skipped   Linear Layers: {self.num_skip_linear:<5}",
            f"Total     Linear Layers: {self.num_linear_layers:<5}",
        ]
        if self.verbose:
            summary_strs.append(f"Skipped        Patterns: {self.exclude_layers}")
        max_len = max(max(len(s) for s in summary_strs), 0) + 2
        logger.info("-" * max_len)
        # extend strs with spaces for better formatting, the last char is '|'
        summary_strs = [s.ljust(max_len) + "|" for s in summary_strs]
        summary_str = "\n".join(summary_strs)
        logger.info(summary_str)
        logger.info("-" * max_len)

    def normalize(self, **kwargs) -> "QuantizeAOContext":
        # This function is used to normalize the quantization context, and it will be called
        # in the _normalize (staticmethod) function. We can do some normalization work here,
        # such as checking the quantization type and setting the quantization config for
        # different quantization types.

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

        quant_type = self.quant_type
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

        self.quant_type = quant_type
        self.quant_type_rev = alias_map_rev.get(quant_type, quant_type)

        prev_exclude_layers = copy.deepcopy(self.exclude_layers)
        if hasattr(self.module_ref, "_exclude_for_quantize"):
            # Workaround for case: TP -> FP8 DQ per row, make torch._scaled_mm happy.
            # Avoid error: "RuntimeError: Expected b.stride(0) == 1 to be true, but got false"
            # use_local_tensor = True (default) in RowwiseParallel (TP) will cause the layout
            # of the linear weights changedly after '_dispatch_get_local_results_slow_path',
            # Why??? Need further investigation.
            if self.quant_type == "fp8_w8a8_dq" and self.per_row:
                exclude_layers = prev_exclude_layers + self.module_ref._exclude_for_quantize
                logger.debug(
                    f"Found extra excluding layers for {self.module_ref.__class__.__name__}: "
                    f"{self.module_ref._exclude_for_quantize}"
                )
                self.exclude_layers = copy.deepcopy(exclude_layers)

        self.repeated_blocks = getattr(
            self.module_ref,
            "_repeated_blocks",
            self.repeated_blocks if self.repeated_blocks else None,
        )
        if self.repeated_blocks is None:
            # If the module doesn't have _repeated_blocks attribute and repeated_blocks
            # is not specified, we will set regional_quantize to False to avoid
            # potential issues.
            self.regional_quantize = False

        if self.per_row and self.module_ref is not None and self.quant_type == "fp8_w8a8_dq":
            # assert the dtype of module's is bfloat16
            for name, submod in self.module_ref.named_modules():
                if isinstance(submod, torch.nn.Linear):
                    assert submod.weight.dtype == torch.bfloat16, (
                        f"Per-row quantization is only supported for linear layers with "
                        f"weight dtype of bfloat16, but found dtype {submod.weight.dtype} "
                        f"in layer {name}."
                    )
        return self


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


def _quant_config_impl(quant_ctx: QuantizeAOContext, **kwargs):
    try:
        if quant_ctx.quant_type == "fp8_w8a8_dq":
            from torchao.quantization import (
                Float8DynamicActivationFloat8WeightConfig,
                PerTensor,
                PerRow,
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
                granularity=(
                    ((PerRow(), PerRow())) if quant_ctx.per_row else ((PerTensor(), PerTensor()))
                ),
            )

        elif quant_ctx.quant_type == "fp8_blockwise":
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

        elif quant_ctx.quant_type == "fp8_w8a16_wo":
            from torchao.quantization import Float8WeightOnlyConfig

            quant_config = Float8WeightOnlyConfig(
                weight_dtype=kwargs.get(
                    "weight_dtype",
                    torch.float8_e4m3fn,
                ),
            )

        elif quant_ctx.quant_type == "int8_w8a8_dq":
            from torchao.quantization import (
                Int8DynamicActivationInt8WeightConfig,
            )

            quant_config = Int8DynamicActivationInt8WeightConfig()

        elif quant_ctx.quant_type == "int8_w8a16_wo":

            from torchao.quantization import Int8WeightOnlyConfig

            quant_config = Int8WeightOnlyConfig(
                # group_size is None -> per_channel, else per group
                group_size=kwargs.get("group_size", None),
            )
        elif quant_ctx.quant_type == "int4_w4a16_wo":

            from torchao.quantization import Int4WeightOnlyConfig

            quant_config = Int4WeightOnlyConfig(
                group_size=kwargs.get("group_size", 32),
            )

        else:
            raise ValueError(f"quant_type: {quant_ctx.quant_type} is not supported now!")

    except ImportError as e:
        e.msg += (
            f"{quant_ctx.quant_type} is not supported in torchao backend now! "
            "Please consider to use another quantization type instead."
        )
        raise e

    return quant_config


def _filter_fn_impl(
    m: torch.nn.Module,
    name: str,
    quant_ctx: QuantizeAOContext = QuantizeAOContext(),
) -> bool:
    from torchao.float8.float8_linear import Float8Linear

    msg_template = "Skip Quantization: {name} -> pattern<{pattern}>"

    quant_ctx.num_layers += 1
    if isinstance(m, torch.nn.Linear) and not isinstance(m, Float8Linear):
        quant_ctx.num_linear_layers += 1

        for exclude_name in quant_ctx.exclude_layers:
            if exclude_name in name:
                if quant_ctx.verbose:
                    logger.info(
                        msg_template.format(
                            name=name,
                            pattern=exclude_name,
                        )
                    )

                quant_ctx.num_skip_linear += 1
                return False

        if (
            quant_ctx.per_row
            and m.weight.dtype != torch.bfloat16
            and quant_ctx.quant_type == "fp8_w8a8_dq"
        ):
            if quant_ctx.verbose:
                logger.info(
                    msg_template.format(
                        name=name,
                        pattern=f"dtype({m.weight.dtype})!=bfloat16",
                    )
                )

            quant_ctx.num_skip_linear += 1
            return False

        # check blockwise fp8 support for linear layers, if not supported,
        # skip quantization for that layer.
        if quant_ctx.quant_type in [
            "fp8_blockwise",
        ] and not _check_if_linear_fp8_blockwise_can_support(m):
            weight_shape = tuple(m.weight.shape)
            if quant_ctx.verbose:
                logger.info(
                    msg_template.format(
                        name=name,
                        pattern=f"w{weight_shape} % block_size(128, 128) != 0",
                    )
                )
            quant_ctx.num_skip_linear += 1
            return False

        if quant_ctx.quant_type in [
            "fp8_w8a8_dq",
            "fp8_blockwise",
        ] and not _check_if_linear_with_bias_fp8_can_support(m):
            if quant_ctx.verbose:
                logger.info(
                    msg_template.format(
                        name=name,
                        pattern="DTensor + bias is not supported for _scaled_mm",
                    )
                )
            quant_ctx.num_skip_linear += 1
            return False

        quant_ctx.num_quant_linear += 1
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
