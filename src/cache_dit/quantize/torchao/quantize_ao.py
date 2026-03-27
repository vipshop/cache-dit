import torch
import copy
import logging
import dataclasses
from functools import partial
from typing import Optional, List
from torchao.core.config import AOBaseConfig
from ..config import QuantizeConfig
from ...utils import maybe_empty_cache
from ...platforms import current_platform
from ...envs import ENV
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
    # Regional quantization for transformer modules in Diffusers, users can
    # set regional_quantize to False to disable this behavior and quantize
    # the whole module directly. For models outside of diffusers, users can specify
    # the repeated blocks by setting repeated_blocks to a list of block names.
    basic_ao_config = _get_torchao_config(
        quant_ctx.quant_type,
        per_row=quant_ctx.per_row,
        **quant_ctx.kwargs,
    )
    basic_filter_fn = partial(_basic_filter_fn, quant_ctx=quant_ctx)

    # Reference: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/modeling_utils.py#L1475
    if quant_ctx.regional_quantize:
        assert (
            quant_ctx.repeated_blocks is not None
        ), "repeated_blocks must be specified when regional_quantize is True."
        has_quantized_region = False
        # First, quantize non exclude layers with basic config, and skip layers that are
        # in fallback_layers if fallback is enabled, we will quantize those layers in
        # the second pass with fallback config.
        for submod in module.modules():
            if submod.__class__.__name__ in quant_ctx.repeated_blocks:
                _quantize_module(submod, basic_ao_config, basic_filter_fn)
                has_quantized_region = True
        # Second, quantize the fallback layers with fallback config if fallback is enabled and
        # the layers are not quantized in the first pass. Currently, only support float8 per-tensor
        # fallback for rowwise layers in TP, and the fallback config is set to per-tensor quantization.
        if quant_ctx.can_fallback():
            fallback_ao_config = _get_torchao_config(
                "fp8_w8a8_dq", per_row=False, **quant_ctx.kwargs
            )  # fallback to per-tensor quantization
            fallback_filter_fn = partial(_fallback_filter_fn, quant_ctx=quant_ctx)
            for submod in module.modules():
                if submod.__class__.__name__ in quant_ctx.repeated_blocks:
                    _quantize_module(submod, fallback_ao_config, fallback_filter_fn)
                    has_quantized_region = True
        if not has_quantized_region:
            raise ValueError(
                f"Regional quantization failed because {quant_ctx.repeated_blocks} "
                "classes are not found in the module."
            )
    else:
        _quantize_module(module, basic_ao_config, basic_filter_fn)

    maybe_empty_cache()
    quant_ctx.summary()

    module._is_quantized = True
    module._quantize_type = quant_ctx.quant_type
    module._quantize_config = quantize_config
    module._exclude_layers = copy.deepcopy(quant_ctx.exclude_layers)

    return module


def _quantize_module(
    m: torch.nn.Module,
    ao_config: AOBaseConfig,
    filter_fn: callable,
    **kwargs,
):
    from torchao.quantization import quantize_

    quantize_(
        m,
        ao_config,
        filter_fn=filter_fn,
        device=kwargs.get("device", None),
    )


@dataclasses.dataclass
class QuantizeAOContext:
    module_ref: torch.nn.Module = None  # ref only
    # quantization config
    quant_type: str = "fp8_w8a8_dq"
    per_row: bool = True
    regional_quantize: bool = True
    repeated_blocks: Optional[List[str]] = None
    exclude_layers: List[str] = dataclasses.field(default_factory=list)
    float8_per_tensor_fallback: bool = True
    verbose: bool = False
    # stats for summary
    quant_type_rev: str = "float8"
    num_basic_quant_linear: int = 0
    num_basic_skip_linear: int = 0
    num_fallback_quant_linear: int = 0
    num_fallback_skip_linear: int = 0
    num_linear_layers: int = 0
    num_layers: int = 0
    # record the full name of quantized layers for better summary and analysis,
    # the name is in the format of "module1.module2.linear"
    basic_quantized_layers: List[str] = dataclasses.field(default_factory=list)
    fallback_quantized_layers: List[str] = dataclasses.field(default_factory=list)
    skipped_reasons: List[str] = dataclasses.field(default_factory=list)
    alias_map: dict = dataclasses.field(
        default_factory=lambda: {
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
    )
    alias_map_rev: dict = dataclasses.field(
        default_factory=lambda: {
            "fp8_w8a8_dq": "float8",
            "fp8_blockwise": "float8_blockwise",
            "fp8_w8a16_wo": "float8_weight_only",
            "int8_w8a8_dq": "int8",
            "int8_w8a16_wo": "int8_weight_only",
            "int4_w4a8_dq": "int4",
            "int4_w4a16_wo": "int4_weight_only",
        }
    )
    # e.g, for rowwise TP -> FP8 per-row -> fallback -> FP8 per-tensor
    fallback_layers: List[str] = dataclasses.field(default_factory=list)  # all fallback layers
    # rowwise layers that may cause issue with FP8 per-row quantization,
    # recorded for better summary and analysis.
    rowwise_layers: List[str] = dataclasses.field(default_factory=list)
    # Extra kwargs for trival usage, e.g, weight_dtype and activation_dtype
    # for float8 quantization, etc.
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
            float8_per_tensor_fallback=quantize_config.float8_per_tensor_fallback,
            verbose=quantize_config.verbose,
            kwargs=copy.deepcopy(kwargs),
        )

    def summary(self):
        quantized_region = (
            f"{self.repeated_blocks}"
            if self.regional_quantize and self.repeated_blocks is not None
            else self.module_ref.__class__.__name__ if self.module_ref else "Module"
        )
        # Basic summary info.
        total_quant_linear = self.num_basic_quant_linear + self.num_fallback_quant_linear
        total_skip_linear = self.num_basic_skip_linear + self.num_fallback_skip_linear
        summary_strs = [
            f"Quantized                 Method: {self.quant_type_rev}",
            f"Quantized                 Region: {quantized_region}",
            f"Quantized    Basic Linear Layers: {self.num_basic_quant_linear:<5}",
            f"Quantized Fallback Linear Layers: {self.num_fallback_quant_linear:<5}",
            f"Total    Quantized Linear Layers: {total_quant_linear:<5}",
            f"Skipped      Basic Linear Layers: {self.num_basic_skip_linear:<5}",
            f"Skipped   Fallback Linear Layers: {self.num_fallback_skip_linear:<5}",
            f"Total      Skipped Linear Layers: {total_skip_linear:<5}",
            f"Total              Linear Layers: {self.num_linear_layers:<5}",
            f"Skipped                 Patterns: {self.exclude_layers}",
        ]
        if not self.verbose or not logger.isEnabledFor(logging.DEBUG):
            summary_strs.pop()  # remove skipped patterns in non-verbose mode
        max_len = max(max(len(s) for s in summary_strs), 0) + 2
        logger.info("-" * max_len)
        # extend strs with spaces for better formatting, the last char is '|'
        summary_strs = [s.ljust(max_len) + "|" for s in summary_strs]
        summary_str = "\n".join(summary_strs)
        logger.info(summary_str)
        logger.info("-" * max_len)

        # Detailed summary for skipped reasons, only log when verbose is True.
        if self.verbose and self.skipped_reasons:
            skipped_reasons_counter = {}
            for reason in self.skipped_reasons:
                skipped_reasons_counter[reason] = skipped_reasons_counter.get(reason, 0) + 1

            max_name_len = 0
            max_pattern_len = 0
            for reason, count in skipped_reasons_counter.items():
                name, pattern = reason.split("->")
                max_name_len = max(max_name_len, len(name.strip()))
                max_pattern_len = max(max_pattern_len, len(pattern.strip()))

            skipped_reasons_strs = []
            for reason, count in skipped_reasons_counter.items():
                name, pattern = reason.split("->")
                name_str = name.strip().ljust(max_name_len)
                pattern_str = pattern.strip().ljust(max_pattern_len)
                skipped_reasons_strs.append(f"{name_str}: {pattern_str}: {count:<4} layers")

            # update max_reason_len for the count info
            max_reason_len = max(max(len(s) for s in skipped_reasons_strs), 0) + 2
            logger.info("-" * max_reason_len)
            # extend strs with spaces for better formatting, the last char is '|'
            skipped_reasons_strs = [s.ljust(max_reason_len) + "|" for s in skipped_reasons_strs]
            skipped_reasons_str = "\n".join(skipped_reasons_strs)
            logger.info(skipped_reasons_str)
            logger.info("-" * max_reason_len)

    def normalize(self, **kwargs) -> "QuantizeAOContext":
        # This function is used to normalize the quantization context, and it will be called
        # in the _normalize (staticmethod) function. We can do some normalization work here,
        # such as checking the quantization type and setting the quantization config for
        # different quantization types.

        quant_type = self.quant_type
        if quant_type.lower() in self.alias_map:
            quant_type = self.alias_map[quant_type.lower()]

        quant_type = quant_type.lower()
        assert (
            quant_type in self.alias_map_rev
        ), f"{quant_type} is not supported for torchao backend now!"

        if "fp8" in quant_type:
            assert current_platform.get_device_capability() >= (
                8,
                9,
            ), "FP8 is not supported for current device."

        self.quant_type = quant_type
        self.quant_type_rev = self.alias_map_rev.get(quant_type, quant_type)

        # Preprocess exclude layers and fallback layers.
        self._maybe_fill_fallback_layers()

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

        if self.module_ref is not None and self.is_float8_dynamic_per_row():
            # assert the dtype of module's is bfloat16
            for name, submod in self.module_ref.named_modules():
                if isinstance(submod, torch.nn.Linear):
                    assert submod.weight.dtype == torch.bfloat16, (
                        f"Per-row quantization is only supported for linear layers with "
                        f"weight dtype of bfloat16, but found dtype {submod.weight.dtype} "
                        f"in layer {name}."
                    )
        return self

    def _maybe_fill_fallback_layers(self):
        exclude_layers = copy.deepcopy(self.exclude_layers)
        fallback_layers = copy.deepcopy(self.fallback_layers)
        # Case 0: TP + torchao FP8 per-row quantization.
        # Workaround for case: TP -> FP8 DQ per row, make torch._scaled_mm happy.
        # Avoid error: "RuntimeError: Expected b.stride(0) == 1 to be true, but got false"
        # RowwiseParallel (TP) will cause the layout of the linear weights changedly after
        # '_dispatch_get_local_results_slow_path', Why??? Need further investigation.
        rowwise_layers = copy.deepcopy(self.rowwise_layers)
        if self.module_ref is not None and self.is_float8_dynamic_per_row():
            if not ENV.CACHE_DIT_DISABLE_EXCLUDE_FOR_QUANTIZE_AFTER_TP:
                rowwise_layers = getattr(self.module_ref, "_rowwise_layers", [])
                if rowwise_layers:
                    if self.float8_per_tensor_fallback:
                        fallback_layers = fallback_layers + rowwise_layers
                        logger.info(f"Add fallback layers: {rowwise_layers}.")
                    else:
                        exclude_layers = exclude_layers + rowwise_layers
                        logger.info(f"Add exclude layers: {rowwise_layers}.")
        self.rowwise_layers = copy.deepcopy(rowwise_layers)
        # Case 1/2/3/...: Future cases ...
        # We may add more cases in the future where we need to automatically fill the
        # fallback layers based on the module's attributes or other conditions, so we
        # put this logic in a separate function for better maintainability and readability.
        self.exclude_layers = copy.deepcopy(exclude_layers)
        self.fallback_layers = copy.deepcopy(fallback_layers)

    def is_int8(self) -> bool:
        return "int8" in self.quant_type

    def is_int4(self) -> bool:
        return "int4" in self.quant_type

    def is_weight_only(self) -> bool:
        return "wo" in self.quant_type

    def is_float8(self) -> bool:
        return "fp8" in self.quant_type

    def is_float8_dynamic_per_row(self) -> bool:
        return self.quant_type == "fp8_w8a8_dq" and self.per_row

    def is_float8_blockwise(self) -> bool:
        return self.quant_type == "fp8_blockwise"

    def can_fallback(self) -> bool:
        # Currently, only support float8 per-tensor fallback for rowwise layers if
        # regional quantiztion is enabled. Not support fallback for int8/int4/weight-only
        # quantization for now, we may add more fallback options in the future.
        return (
            self.float8_per_tensor_fallback
            and (self.is_float8_dynamic_per_row() or self.is_float8_blockwise())
            and (not self.is_weight_only() and not self.is_int8() and not self.is_int4())
            and self.regional_quantize
            and bool(self.fallback_layers)
        )

    def is_fallback_layer(self, name: str) -> bool:
        if not self.can_fallback():
            return False
        fallback_layers = self.fallback_layers if self.fallback_layers else []
        for fallback_name in fallback_layers:
            if fallback_name in name:
                return True
        return False

    def is_exclude_layer(self, name: str) -> bool:
        for exclude_name in self.exclude_layers:
            if exclude_name in name:
                return True
        return False

    def is_basic_quantized_layer(self, name: str) -> bool:
        if name in self.basic_quantized_layers:
            return True
        return False

    def is_fallback_quantized_layer(self, name: str) -> bool:
        if name in self.fallback_quantized_layers:
            return True
        return False

    def is_quantized_layer(self, name: str) -> bool:
        return self.is_basic_quantized_layer(name) or self.is_fallback_quantized_layer(name)

    def is_rowwise_layer(self, name: str) -> bool:
        for rowwise_name in self.rowwise_layers:
            if rowwise_name in name:
                return True
        return False

    def get_exclude_name(self, name: str) -> Optional[str]:
        if self.is_rowwise_layer(name):
            return "Rowwise(Tensor Parallel)"
        for exclude_name in self.exclude_layers:
            if exclude_name in name:
                return exclude_name
        return None


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


def _get_torchao_config(quant_type: str, **kwargs) -> AOBaseConfig:
    per_row = kwargs.get("per_row", True)
    try:
        if quant_type == "fp8_w8a8_dq":
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
                granularity=(((PerRow(), PerRow())) if per_row else ((PerTensor(), PerTensor()))),
            )

        elif quant_type == "fp8_blockwise":
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

        elif quant_type == "fp8_w8a16_wo":
            from torchao.quantization import Float8WeightOnlyConfig

            quant_config = Float8WeightOnlyConfig(
                weight_dtype=kwargs.get(
                    "weight_dtype",
                    torch.float8_e4m3fn,
                ),
            )

        elif quant_type == "int8_w8a8_dq":
            from torchao.quantization import (
                Int8DynamicActivationInt8WeightConfig,
            )

            quant_config = Int8DynamicActivationInt8WeightConfig()

        elif quant_type == "int8_w8a16_wo":

            from torchao.quantization import Int8WeightOnlyConfig

            quant_config = Int8WeightOnlyConfig(
                # group_size is None -> per_channel, else per group
                group_size=kwargs.get("group_size", None),
            )
        elif quant_type == "int4_w4a16_wo":

            from torchao.quantization import Int4WeightOnlyConfig

            quant_config = Int4WeightOnlyConfig(
                group_size=kwargs.get("group_size", 32),
            )

        else:
            raise ValueError(f"quant_type: {quant_type} is not supported now!")

    except ImportError as e:
        e.msg += (
            f"{quant_type} is not supported in torchao backend now! "
            "Please consider to use another quantization type instead."
        )
        raise e

    return quant_config


def _basic_filter_fn(
    m: torch.nn.Module,
    name: str,
    quant_ctx: QuantizeAOContext = QuantizeAOContext(),
) -> bool:
    from torchao.float8.float8_linear import Float8Linear

    msg_template = "Skip: {name} -> pattern<{pattern}>"

    quant_ctx.num_layers += 1
    if isinstance(m, torch.nn.Linear) and not isinstance(m, Float8Linear):
        quant_ctx.num_linear_layers += 1

        # The fallback layers should be skipped in the basic filter function,
        # and they will be quantized in the fallback filter function.
        if quant_ctx.is_exclude_layer(name) or quant_ctx.is_fallback_layer(name):
            # Only record the skip reason for layers that are not in fallback layers,
            # if fallback is enabled, because we will quantize those layers in the
            # second pass with fallback config, and we only want to record the skip
            # reason here for layers that are skipped in basic filter fn, not the layers
            # that are skipped in fallback filter fn.
            if not quant_ctx.is_fallback_layer(name):
                if quant_ctx.verbose:
                    exclude_name = quant_ctx.get_exclude_name(name)
                    skip_reason = msg_template.format(name=name, pattern=exclude_name)
                    logger.debug(skip_reason)
                    quant_ctx.skipped_reasons.append(skip_reason)

                quant_ctx.num_basic_skip_linear += 1
            return False

        # Check for weight dtype for float8 per-row
        if (
            quant_ctx.per_row
            and m.weight.dtype != torch.bfloat16
            and quant_ctx.quant_type == "fp8_w8a8_dq"
        ):
            if quant_ctx.verbose:
                skip_reason = msg_template.format(
                    name=name,
                    pattern=f"dtype({m.weight.dtype})!=bfloat16",
                )
                logger.debug(skip_reason)
                quant_ctx.skipped_reasons.append(skip_reason)

            quant_ctx.num_basic_skip_linear += 1
            return False

        # check blockwise fp8 support for linear layers, if not supported,
        # skip quantization for that layer.
        if quant_ctx.quant_type in [
            "fp8_blockwise",
        ] and not _check_if_linear_fp8_blockwise_can_support(m):
            weight_shape = tuple(m.weight.shape)
            if quant_ctx.verbose:
                skip_reason = msg_template.format(
                    name=name,
                    pattern=f"w{weight_shape} % block_size(128, 128) != 0",
                )
                logger.debug(skip_reason)
                quant_ctx.skipped_reasons.append(skip_reason)
            quant_ctx.num_basic_skip_linear += 1
            return False

        if quant_ctx.quant_type in [
            "fp8_w8a8_dq",
            "fp8_blockwise",
        ] and not _check_if_linear_with_bias_fp8_can_support(m):
            if quant_ctx.verbose:
                skip_reason = msg_template.format(
                    name=name,
                    pattern="DTensor + bias is not supported for _scaled_mm",
                )
                logger.debug(skip_reason)
                quant_ctx.skipped_reasons.append(skip_reason)
            quant_ctx.num_basic_skip_linear += 1
            return False

        quant_ctx.num_basic_quant_linear += 1
        quant_ctx.basic_quantized_layers.append(name)
        return True

    return False


def _fallback_filter_fn(
    m: torch.nn.Module,
    name: str,
    quant_ctx: QuantizeAOContext = QuantizeAOContext(),
) -> bool:
    from torchao.float8.float8_linear import Float8Linear

    # Fallback to quant_type: fp8_w8a8_dq, per_row: False
    msg_template = "Fallback: {name} -> pattern<{pattern}>"

    # Some stats like num_layers and num_linear_layers will be counted in basic_filter_fn,
    # so here we only count the number of quantized and skipped layers for fallback filter fn.
    if isinstance(m, torch.nn.Linear) and not isinstance(m, Float8Linear):
        if not quant_ctx.is_fallback_layer(name):
            # Only record the skip reason for layers that are both not in fallback layers and
            # exclude layers, because the layers in exclude layers will be skipped in basic filter
            # fn, and we no longer want to record the skip reason for layers in exclude layers here.
            if not quant_ctx.is_exclude_layer(name) and not quant_ctx.is_quantized_layer(name):
                if quant_ctx.verbose:
                    skip_reason = msg_template.format(name=name, pattern="NOT in fallback layers")
                    logger.debug(skip_reason)
                    quant_ctx.skipped_reasons.append(skip_reason)
                quant_ctx.num_fallback_skip_linear += 1
            return False

        if not _check_if_linear_with_bias_fp8_can_support(m):
            if quant_ctx.verbose:
                skip_reason = msg_template.format(
                    name=name,
                    pattern="DTensor + bias is not supported for _scaled_mm",
                )
                logger.debug(skip_reason)
                quant_ctx.skipped_reasons.append(skip_reason)
            quant_ctx.num_fallback_skip_linear += 1
            return False

        quant_ctx.num_fallback_quant_linear += 1
        quant_ctx.fallback_quantized_layers.append(name)
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
