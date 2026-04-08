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

_ALLOWED_QUANTIZE_TYPES = [
  "float8_per_row",
  "float8_per_tensor",
  "float8_per_block",
  "float8_weight_only",
  "int8_per_row",
  "int8_per_tensor",
  "int8_weight_only",
  "int4_weight_only",
]

_PREFERRED_PRECISION_PLAN_ORDER = [
  "float8_per_tensor",  # prefer per_tensor for better compatibility (included fallback layers)
  "float8_per_row",
  "float8_per_block",
  "float8_weight_only",
  "int8_per_tensor",
  "int8_per_row",
  "int8_weight_only",
  "int4_weight_only",
]


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
  # Regional quantization for transformer modules in Diffusers, users can
  # set regional_quantize to False to disable this behavior and quantize
  # the whole module directly. For models outside of diffusers, users can specify
  # the repeated blocks by setting repeated_blocks to a list of block names.
  basic_ao_config = _get_torchao_config(quant_ctx.quant_type, **quant_ctx.kwargs)
  basic_filter_fn = partial(_basic_filter_fn, quant_ctx=quant_ctx, override_quant_type=None)

  # Reference: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/modeling_utils.py#L1475
  if quant_ctx.regional_quantize:
    assert (quant_ctx.repeated_blocks
            is not None), "repeated_blocks must be specified when regional_quantize is True."
    has_quantized_region = False

    if quant_ctx.precision_plan is not None:
      logger.debug(f"Precision plan is enabled with the following plan: {quant_ctx.precision_plan}")
      # If precision_plan is specified, we will quantize the layers according to the precision_plan.
      for quant_type in _PREFERRED_PRECISION_PLAN_ORDER:
        layer_names = quant_ctx.reverse_precision_plan.get(quant_type, [])
        if not layer_names:  # no layers for this quant_type, skip directly.
          continue
        # Update layers allowed to quantize for better summary and analysis.
        quant_ctx.layers_allowed_to_quantize = layer_names
        submod_ao_config = _get_torchao_config(quant_type, **quant_ctx.kwargs)
        submod_filter_fn = partial(
          _basic_filter_fn,
          quant_ctx=quant_ctx,
          override_quant_type=quant_type,
        )
        for submod in module.modules():
          if submod.__class__.__name__ in quant_ctx.repeated_blocks:
            _quantize_module(submod, submod_ao_config, submod_filter_fn)
            has_quantized_region = True
        quant_ctx.first_quantize_pass_applied = True

      # The default pass for layers that are not specified in the precision_plan, we will quantize
      # them with the basic config. DON'T forget to reset the layers_allowed_to_quantize to None
      # for the default pass.
      quant_ctx.precision_plan_pass_applied = True  # IMPORTANT!
      quant_ctx.layers_allowed_to_quantize = None
      for submod in module.modules():
        if submod.__class__.__name__ in quant_ctx.repeated_blocks:
          _quantize_module(submod, basic_ao_config, basic_filter_fn)
          has_quantized_region = True
    else:
      # First, quantize non exclude layers with basic config, and skip layers that are
      # in fallback_layers if fallback is enabled, we will quantize those layers in
      # the second pass with fallback config.
      for submod in module.modules():
        if submod.__class__.__name__ in quant_ctx.repeated_blocks:
          _quantize_module(submod, basic_ao_config, basic_filter_fn)
          has_quantized_region = True
      quant_ctx.first_quantize_pass_applied = True
      # Second, quantize the fallback layers with fallback config if fallback is enabled and
      # the layers are not quantized in the first pass. Currently, only support float8 per-tensor
      # fallback for rowwise layers in TP, and the fallback config is set to per-tensor quantization.
      if quant_ctx.required_fallback():
        fallback_ao_config = _get_torchao_config("float8_per_tensor", **quant_ctx.kwargs)
        fallback_filter_fn = partial(_fallback_filter_fn, quant_ctx=quant_ctx)
        for submod in module.modules():
          if submod.__class__.__name__ in quant_ctx.repeated_blocks:
            _quantize_module(submod, fallback_ao_config, fallback_filter_fn)
            has_quantized_region = True
    if not has_quantized_region:
      raise ValueError(f"Regional quantization failed because {quant_ctx.repeated_blocks} "
                       "classes are not found in the module.")
  else:
    _quantize_module(module, basic_ao_config, basic_filter_fn)

  maybe_empty_cache()
  quant_ctx.summary()

  module._is_quantized = True
  module._quantize_config = quantize_config
  module._quantize_type = quant_ctx.quant_type
  module._exclude_layers = quant_ctx.exclude_layers

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
  quant_type: str = "float8_per_row"
  regional_quantize: bool = True
  repeated_blocks: Optional[List[str]] = None
  exclude_layers: List[str] = dataclasses.field(default_factory=list)
  per_tensor_fallback: bool = True
  # e.g., {"block1": "float8_per_row", "block2": "float8_per_tensor"}
  precision_plan: Optional[dict] = None
  verbose: bool = False
  # stats for summary
  num_linear_layers: int = 0
  # e.g, for rowwise TP -> FP8 per-row -> fallback -> FP8 per-tensor
  fallback_layers: List[str] = dataclasses.field(default_factory=list)  # all fallback layers
  # rowwise layers that may cause issue with FP8 per-row quantization,
  # recorded for better summary and analysis.
  rowwise_layers: List[str] = dataclasses.field(default_factory=list)
  # Extra kwargs for trival usage, e.g, weight_dtype and activation_dtype
  # for float8 quantization, etc.
  kwargs: dict = dataclasses.field(default_factory=dict)
  # A dict that record quantized info: precision -> quantized layers.
  quantized_map: dict = dataclasses.field(
    default_factory=lambda: {quant_type: 0
                             for quant_type in _ALLOWED_QUANTIZE_TYPES}, )
  # A dict that record skipped layers info: precision -> skipped layers.
  skipped_map: dict[str, list[str]] = dataclasses.field(
    default_factory=lambda: {k: []
                             for k in _ALLOWED_QUANTIZE_TYPES})
  # Reverse precision plan to make it easier to use, the format is
  # {quant_type: [layer_name1, layer_name2, ...], ...}
  reverse_precision_plan: dict = dataclasses.field(default_factory=dict)
  # A list of layers that are allowed to be quantized, used for better summary
  # and analysis, it will be filled based on the exclude_layers and fallback_layers,
  # etc. The format is [layer_name1, layer_name2, ...]. Helpers for precision plan.
  # This list will be dynamically updated during the quantization process, and it
  # will be used to determine whether a layer is quantized or skipped, etc.
  layers_allowed_to_quantize: List[str] = dataclasses.field(default_factory=list)
  # whether the precision plan is applied, used for better summary and analysis.
  precision_plan_pass_applied: bool = False
  # To determine whether it's the first pass of quantization or the fallback pass, etc.
  first_quantize_pass_applied: bool = False

  def __post_init__(self):
    self.quant_type = self.quant_type.lower()
    assert self.quant_type in _ALLOWED_QUANTIZE_TYPES, (
      f"quant_type {self.quant_type} is not supported, allowed quantization "
      f"types: {_ALLOWED_QUANTIZE_TYPES}.")
    if self.is_float8():
      assert current_platform.get_device_capability() >= (
        8,
        9,
      ), "FP8 requires Ada or newer GPUs (>=sm89), but got " + str(
        current_platform.get_device_capability())

  @staticmethod
  def from_config(
    quantize_config: QuantizeConfig,
    module: torch.nn.Module,
    **kwargs,
  ) -> "QuantizeAOContext":
    return QuantizeAOContext(
      module_ref=module,  # ref
      quant_type=quantize_config.quant_type,
      regional_quantize=quantize_config.regional_quantize,
      repeated_blocks=quantize_config.repeated_blocks,
      exclude_layers=quantize_config.exclude_layers,
      precision_plan=quantize_config.precision_plan,
      per_tensor_fallback=quantize_config.per_tensor_fallback,
      verbose=quantize_config.verbose,
      kwargs=copy.deepcopy(kwargs),
    ).normalize(**kwargs)

  def summary(self):
    quantized_region = (f"{self.repeated_blocks}"
                        if self.regional_quantize and self.repeated_blocks is not None else
                        self.module_ref.__class__.__name__ if self.module_ref else "Module")
    # Basic summary info.
    all_quant = sum(self.quantized_map.values())
    all_skip = sum(len(v) for v in self.skipped_map.values())
    all_linear = self.num_linear_layers
    summary_strs = []
    mk = max(len(k) for k in self.quantized_map.keys())
    mc = max(len(str(v)) for v in self.quantized_map.values()) + 3
    quantized_map = {k: v for k, v in self.quantized_map.items() if v > 0}
    summary_strs.append(f"Quantized        Region: {quantized_region}")
    for q_type, c in quantized_map.items():
      sk = len(self.skipped_map.get(q_type, []))
      summary_strs.append(f"Quantized Linear Layers: {c:<{mc}} {q_type:<{mk}} {sk} (skipped)")
    summary_strs.append(f"Quantized Linear Layers: {all_quant:<{mc}} (total)")
    summary_strs.append(f"Skipped   Linear Layers: {all_skip:<{mc}} (total)")
    summary_strs.append(f"Linear           Layers: {all_linear:<{mc}} (total)")
    summary_strs.append(f"Skipped        Patterns: {self.exclude_layers}")

    if not self.verbose or not logger.isEnabledFor(logging.DEBUG):
      summary_strs.pop()  # remove skipped patterns in non-verbose mode
    ml = max(max(len(s) for s in summary_strs), 0) + 2
    logger.info("-" * ml)
    # extend strs with spaces for better formatting, the last char is '|'
    summary_strs = [s.ljust(ml) + "|" for s in summary_strs]
    summary_str = "\n".join(summary_strs)
    logger.info(summary_str)
    logger.info("-" * ml)

    # Detailed summary for skipped reasons, only log when verbose is True.
    skipped_reasons = []
    for quant_type, reasons in self.skipped_map.items():
      for reason in reasons:
        skipped_reasons.append(f"{quant_type}, {reason}")

    if self.verbose and skipped_reasons:
      skipped_reasons_counter = {}
      for reason in skipped_reasons:
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
    if self.precision_plan:
      for layer_name, quant_type in self.precision_plan.items():
        quant_type = quant_type.lower()
        assert quant_type in _ALLOWED_QUANTIZE_TYPES, (
          f"quant_type {quant_type} in precision_plan is not supported, allowed "
          f"quantization types: {_ALLOWED_QUANTIZE_TYPES}.")
        if quant_type not in self.reverse_precision_plan:
          self.reverse_precision_plan[quant_type] = []
        self.reverse_precision_plan[quant_type].append(layer_name)
      # remove emplty quant_type in reverse_precision_plan
      self.reverse_precision_plan = {k: v for k, v in self.reverse_precision_plan.items() if v}
      for quant_type in self.reverse_precision_plan:
        self.reverse_precision_plan[quant_type] = list(set(self.reverse_precision_plan[quant_type]))

    # Preprocess exclude layers and fallback layers.
    self._prepare_extra_layers_info()

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

    if self.module_ref is not None and self.is_float8_per_row():
      # assert the dtype of module's is bfloat16
      for name, submod in self.module_ref.named_modules():
        if isinstance(submod, torch.nn.Linear):
          assert submod.weight.dtype == torch.bfloat16, (
            f"Per-row quantization is only supported for linear layers with "
            f"weight dtype of bfloat16, but found dtype {submod.weight.dtype} "
            f"in layer {name}.")

    # Merged the fallback layers into the precision plan if precision_plan is specified,
    # to make sure those layers will be quantized with the fallback quantization type.
    if self.required_fallback() and self.fallback_layers:
      if self.precision_plan is not None:
        # Also add the fallback layers to the reverse_precision_plan.
        self.reverse_precision_plan["float8_per_tensor"] = (
          self.reverse_precision_plan.get("float8_per_tensor", []) + self.fallback_layers)
        self.reverse_precision_plan["float8_per_tensor"] = list(
          set(self.reverse_precision_plan["float8_per_tensor"]))
        # Fallback layers will merged into precision_plan if precision_plan is specified,
        # otherwise they will be added to the fallback_layers list directly, and they
        # will be quantized in the second pass with fallback config.
        # Clear fallback layers since they are merged into precision_plan now.
        # require_fallback() will always return False after this since fallback_layers
        # will be empty, so it won't cause issue for the logic that checks whether to
        # apply fallback quantization in the second pass.
        self.fallback_layers = []  # IMPORTANT!
        logger.info("precision_plan is specified, the fallback layers will be merged "
                    "into the float8_per_tensor plan.")

    return self

  def _prepare_extra_layers_info(self):
    exclude_layers = copy.deepcopy(self.exclude_layers)
    fallback_layers = copy.deepcopy(self.fallback_layers)
    # Case 0: TP + torchao FP8 per-row quantization.
    # Workaround for case: TP -> FP8 DQ per row, make torch._scaled_mm happy.
    # Avoid error: "RuntimeError: Expected b.stride(0) == 1 to be true, but got false"
    # RowwiseParallel (TP) will cause the layout of the linear weights changedly after
    # '_dispatch_get_local_results_slow_path', Why??? Need further investigation.
    rowwise_layers = copy.deepcopy(self.rowwise_layers)
    # The major quantization type may not be float8 per-row when precision_plan is specified,
    # we also enable the fallback for rowwise layers when precision_plan is specified since
    # some of the layers in the precision_plan may be quantized to float8 per-row, and we don't
    # want those layers to cause error due to the layout issue,.
    if self.module_ref is not None and (self.is_float8_per_row()
                                        or self.precision_plan is not None):
      if not ENV.CACHE_DIT_DISABLE_EXCLUDE_FOR_QUANTIZE_AFTER_TP:
        rowwise_layers = getattr(self.module_ref, "_rowwise_layers", [])
        if rowwise_layers:
          if self.per_tensor_fallback:
            fallback_layers = fallback_layers + rowwise_layers
            logger.debug(f"Add fallback layers: {rowwise_layers}.")
          else:
            exclude_layers = exclude_layers + rowwise_layers
            logger.debug(f"Add exclude layers: {rowwise_layers}.")
    self.rowwise_layers = copy.deepcopy(rowwise_layers)
    # Case 1/2/3/...: Future cases ...
    # We may add more cases in the future where we need to automatically fill the
    # fallback layers based on the module's attributes or other conditions, so we
    # put this logic in a separate function for better maintainability and readability.
    self.exclude_layers = copy.deepcopy(exclude_layers)
    self.fallback_layers = copy.deepcopy(fallback_layers)

  def is_int8(self) -> bool:
    return "int8" in self.quant_type

  def is_int8_per_row(self) -> bool:
    return self.quant_type == "int8_per_row"

  def is_int8_per_tensor(self) -> bool:
    return self.quant_type == "int8_per_tensor"

  def is_int8_weight_only(self) -> bool:
    return self.quant_type == "int8_weight_only"

  def is_int4(self) -> bool:
    return "int4" in self.quant_type

  def is_weight_only(self) -> bool:
    return "weight_only" in self.quant_type

  def is_float8(self) -> bool:
    return "float8" in self.quant_type

  def is_float8_per_row(self) -> bool:
    return self.quant_type == "float8_per_row"

  def is_float8_per_tensor(self) -> bool:
    return self.quant_type == "float8_per_tensor"

  def is_float8_per_block(self) -> bool:
    return self.quant_type == "float8_per_block"

  def is_float8_weight_only(self) -> bool:
    return self.quant_type == "float8_weight_only"

  def required_fallback(self) -> bool:
    # Currently, only support float8 per-tensor fallback for rowwise layers if
    # regional quantiztion is enabled. Not support fallback for int8/int4/weight-only
    # quantization for now, we may add more fallback options in the future.
    _required_fallback = (self.per_tensor_fallback
                          and (self.is_float8_per_row() or self.is_float8_per_tensor()
                               or self.is_float8_per_block()) and
                          (not self.is_weight_only() and not self.is_int8() and not self.is_int4())
                          and self.regional_quantize and bool(self.fallback_layers))
    return _required_fallback

  def is_fallback_layer(self, name: str) -> bool:
    if not self.required_fallback():
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

  def is_quantized_layer(self, m: torch.nn.Module) -> bool:
    return getattr(m, "_is_inner_quantized", False)

  def is_rowwise_layer(self, name: str) -> bool:
    for rowwise_name in self.rowwise_layers:
      if rowwise_name in name:
        return True
    return False

  def get_exclude_name(self, name: str) -> Optional[str]:
    if self.is_rowwise_layer(name):
      return "RowwiseParallel"
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
  assert (current_platform.is_accelerator_available() and current_platform.device_type == "cuda"
          ), "Quantization functionality with torchao backend is only supported on CUDA devices."
  try:
    import torchao  # noqa: F401
  except ImportError:
    raise ImportError("Quantization functionality requires the 'quantization' extra dependencies. "
                      "Install with: pip install cache-dit[quantization]")

  return True


def _get_torchao_config(quant_type: str, **kwargs) -> AOBaseConfig:
  try:
    if quant_type == "float8_per_row":
      from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
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
        granularity=(PerRow(), PerRow()),
      )
    elif quant_type == "float8_per_tensor":
      from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        PerTensor,
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
        granularity=((PerTensor(), PerTensor())),
      )

    elif quant_type == "float8_per_block":
      try:
        from torchao.quantization import (
          Float8DynamicActivationFloat8WeightConfig,
          PerBlock,
        )
      except ImportError:
        raise ImportError("Blockwise quantization is not supported in current version of torchao. "
                          "Please upgrade the torchao library to use this feature.")
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

    elif quant_type == "float8_weight_only":
      from torchao.quantization import Float8WeightOnlyConfig

      quant_config = Float8WeightOnlyConfig(weight_dtype=kwargs.get(
        "weight_dtype",
        torch.float8_e4m3fn,
      ), )

    elif quant_type == "int8_per_row":
      from torchao.quantization import (
        Int8DynamicActivationInt8WeightConfig,
        PerRow,
      )

      quant_config = Int8DynamicActivationInt8WeightConfig(granularity=(PerRow(), PerRow()), )
    elif quant_type == "int8_per_tensor":
      from torchao.quantization import (
        Int8DynamicActivationInt8WeightConfig,
        PerTensor,
      )

      quant_config = Int8DynamicActivationInt8WeightConfig(granularity=(PerTensor(), PerTensor()), )

    elif quant_type == "int8_weight_only":

      from torchao.quantization import Int8WeightOnlyConfig

      quant_config = Int8WeightOnlyConfig(
        # group_size is None -> per_channel, else per group
        group_size=kwargs.get("group_size", None), )
    elif quant_type == "int4_weight_only":

      from torchao.quantization import Int4WeightOnlyConfig

      quant_config = Int4WeightOnlyConfig(group_size=kwargs.get("group_size", 32), )

    else:
      raise ValueError(f"quant_type: {quant_type} is not supported now!")

  except ImportError as e:
    e.msg += (f"{quant_type} is not supported in torchao backend now! "
              "Please consider to use another quantization type instead. "
              f"Allowed quantization types: {_ALLOWED_QUANTIZE_TYPES}.")
    raise e

  return quant_config


def _basic_filter_fn(
    m: torch.nn.Module,
    name: str,
    quant_ctx: QuantizeAOContext = QuantizeAOContext(),
    override_quant_type: Optional[str] = None,  # For precision plan.
) -> bool:
  from torchao.float8.float8_linear import Float8Linear

  msg_template = "skip: {name} -> pattern<{pattern}>"
  curr_quant_type = (override_quant_type
                     if override_quant_type is not None else quant_ctx.quant_type)

  # for better code readability, although this function is not used in basic filter fn,
  # but it may be used in the future when we have more complex quantization logic and
  # need to check if a layer is already quantized or not.
  if quant_ctx.is_quantized_layer(m):
    return False

  def _skip_reason(pattern: str) -> str:
    skip_reason = msg_template.format(name=name, pattern=pattern)
    return skip_reason

  # If layers_allowed_to_quantize is not specified, it means all layers are allowed to
  # quantize by default.
  def _is_curr_plan_allow_to_quantize(name: str) -> bool:  # precision plan
    if quant_ctx.layers_allowed_to_quantize:
      for allow_name in quant_ctx.layers_allowed_to_quantize:
        if allow_name in name:
          return True
      return False
    return True

  if isinstance(m, torch.nn.Linear) and not isinstance(m, Float8Linear):
    # NOTE: We should only record this stats in the first quantize pass.
    if not quant_ctx.first_quantize_pass_applied:
      quant_ctx.num_linear_layers += 1

    # If precision_plan is specified, only quantize the layers that are specified in
    # the precision_plan, and skip the layers that are not in the precision_plan.
    if quant_ctx.precision_plan is not None and not quant_ctx.precision_plan_pass_applied:
      if not _is_curr_plan_allow_to_quantize(name):  # skip quantization for curr plan
        # DON'T record the skip reason for layers that are not in the current plan,
        # since the layers that are not in any plan will be skipped in all precison
        # plan passes, but still be quantized in the end with the basic config, and
        # we only want to record it in the basic quantization pass just 1 time.
        return False

    # The fallback layers should be skipped in the basic filter function,
    # and they will be quantized in the fallback filter function.
    if quant_ctx.is_exclude_layer(name) or quant_ctx.is_fallback_layer(name):
      # Only record the skip reason for layers that are not in fallback layers,
      # if fallback is enabled, because we will quantize those layers in the
      # second pass with fallback config, and we only want to record the skip
      # reason here for layers that are skipped in basic filter fn, not the layers
      # that are skipped in fallback filter fn.
      if not quant_ctx.is_fallback_layer(name):
        skip_reason = _skip_reason(quant_ctx.get_exclude_name(name))
        quant_ctx.skipped_map[curr_quant_type].append(skip_reason)
        logger.debug(skip_reason)
      return False

    # Check for weight dtype for float8 per-row
    if curr_quant_type == "float8_per_row" and m.weight.dtype != torch.bfloat16:
      skip_reason = _skip_reason(f"dtype({m.weight.dtype})!=bfloat16")
      quant_ctx.skipped_map[curr_quant_type].append(skip_reason)
      logger.debug(skip_reason)
      return False

    # check blockwise fp8 support for linear layers, if not supported,
    # skip quantization for that layer.
    if curr_quant_type in [
        "float8_per_block",
    ] and not _check_if_linear_fp8_blockwise_can_support(m):
      weight_shape = tuple(m.weight.shape)
      skip_reason = _skip_reason(f"w{weight_shape} % blocksize(128, 128) != 0")
      quant_ctx.skipped_map[curr_quant_type].append(skip_reason)
      logger.debug(skip_reason)
      return False

    if curr_quant_type in [
        "float8_per_row",
        "float8_per_tensor",
        "float8_per_block",
    ] and not _check_if_linear_with_bias_fp8_can_support(m):
      skip_reason = _skip_reason("_scaled_mm: DTensor + bias NOT supported")
      quant_ctx.skipped_map[curr_quant_type].append(skip_reason)
      logger.debug(skip_reason)
      return False

    # Set this attribute to avoid redundant quantization, which may cause
    # performance regression and other issues.
    m._is_inner_quantized = True  # type: ignore
    quant_ctx.quantized_map[curr_quant_type] += 1
    return True

  return False


def _fallback_filter_fn(
    m: torch.nn.Module,
    name: str,
    quant_ctx: QuantizeAOContext = QuantizeAOContext(),
) -> bool:
  from torchao.float8.float8_linear import Float8Linear

  # Fallback to quant_type: float8_per_tensor.
  msg_template = "fallback: {name} -> pattern<{pattern}>"
  fallback_quant_type = "float8_per_tensor"

  if quant_ctx.is_quantized_layer(m):
    return False

  def _fallback_reason(pattern: str) -> str:
    fallback_reason = msg_template.format(name=name, pattern=pattern)
    return fallback_reason

  # Some stats like num_layers and num_linear_layers will be counted in basic_filter_fn,
  # so here we only count the number of quantized and skipped layers for fallback filter fn.
  if isinstance(m, torch.nn.Linear) and not isinstance(m, Float8Linear):
    if not quant_ctx.is_fallback_layer(name):
      # Only record the skip reason for layers that are both not in fallback layers and
      # exclude layers, because the layers in exclude layers will be skipped in basic filter
      # fn, and we no longer want to record the skip reason for layers in exclude layers here.
      if not quant_ctx.is_exclude_layer(name) and not quant_ctx.is_quantized_layer(m):
        skip_reason = _fallback_reason("NOT in fallback layers")
        quant_ctx.skipped_map[fallback_quant_type].append(skip_reason)
        logger.debug(skip_reason)
      return False

    if not _check_if_linear_with_bias_fp8_can_support(m):
      skip_reason = _fallback_reason("_scaled_mm: DTensor + bias NOT supported")
      quant_ctx.skipped_map[fallback_quant_type].append(skip_reason)
      logger.debug(skip_reason)
      return False

    # Set this attribute to avoid redundant quantization, which may cause
    # performance regression and other issues.
    m._is_inner_quantized = True  # type: ignore
    quant_ctx.quantized_map[fallback_quant_type] += 1
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
    logger.debug(f"block_size: {block_size}, weight_granularity.block_size: "
                 f"{weight_granularity.block_size}")
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
