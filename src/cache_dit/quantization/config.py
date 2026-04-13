import os
import dataclasses
import re
from typing import Optional, Dict, Any, List, Union, Callable
from .backend import QuantizeBackend
from ..logger import init_logger

logger = init_logger(__name__)

_SVDQ_QUANT_TYPE_PATTERN = re.compile(r"^(svdq_int4)_r(\d+)(_dq)?$")
_SVDQ_CALIBRATE_PRECISIONS = ("low", "medium", "high")
_SVDQ_SMOOTH_STRATEGIES = ("activation", "identity", "weight", "weight_inv")
_SVDQ_KWARGS_DEFAULTS: dict[str, Any] = {
  # If streaming is set to True, Cache-DiT will quantize the model in a streaming manner,
  # which means it will quantize the model using the calibration samples one by one, and
  # update the quantization parameters after each sample. This can reduce the memory usage
  # during quantization, but it may cause longer quantization time. If set to False, we
  # will collect the quantization statistics for all calibration samples first, and then
  # compute the quantization parameters and quantize the model. The default value is True.
  "streaming": True,
  # Precision plan shared by calibration math and the low-rank decomposition path:
  # - low: keep calibration math in torch_dtype and use randomized svd_lowrank.
  # - medium: use float32 calibration math and full torch.linalg.svd.
  # - high: use float32 calibration math and float64 SVD.
  "calibrate_precision": "low",
  # Only valid when streaming is set to True. It specifies the number of samples after
  # which the activation buffers will be flushed and the quantization parameters will
  # be updated. This can help to reduce the memory usage during quantization, especially
  # for large models, by not keeping the activation buffers for all samples in memory at
  # the same time. The default value is 1, which means the activation buffers will be
  # flushed and the quantization parameters will be updated after each sample.
  "activation_buffer_flush_sample_count": 1,
  # Only valid when streaming is set to True. It specifies the total size in bytes of the
  # activation buffers that will trigger a flush and update of the quantization parameters.
  # This can help to reduce the memory usage during quantization by flushing the activation
  # buffers and updating the quantization parameters when the total size of the activation
  # buffers exceeds the specified limit. The default value is None, which means there is no
  # limit on the total size of the activation buffers, and they will only be flushed based
  # on the number of samples specified by activation_buffer_flush_sample_count.
  "activation_buffer_flush_cpu_bytes": None,
  # Smooth-factor strategy used by SVDQ quantization. PTQ currently supports
  # activation-derived smoothing only, while DQ intentionally fixes this to the
  # identity vector to preserve the v1 zero-calibration contract.
  "smooth_strategy": "activation",
}


def _parse_svdq_quant_type(quant_type: str) -> tuple[str, int, bool]:
  match = _SVDQ_QUANT_TYPE_PATTERN.fullmatch(quant_type)
  if match is None:
    raise ValueError("SVDQ currently supports quant_type in the form `svdq_int4_r{rank}` or "
                     f"`svdq_int4_r{{rank}}_dq`, got {quant_type!r}.")
  return match.group(1), int(match.group(2)), match.group(3) == "_dq"


def _resolve_svdq_bool_kwarg(key: str, value: Any) -> bool:
  if not isinstance(value, bool):
    raise TypeError(f"svdq_kwargs[{key!r}] must be a bool, got {type(value)}.")
  return value


def _resolve_svdq_calibrate_precision(key: str, value: Any) -> str:
  if not isinstance(value, str):
    raise TypeError(f"svdq_kwargs[{key!r}] must be a str, got {type(value)}.")
  normalized = value.lower()
  if normalized not in _SVDQ_CALIBRATE_PRECISIONS:
    raise ValueError(
      f"svdq_kwargs[{key!r}] must be one of {_SVDQ_CALIBRATE_PRECISIONS}, got {value!r}.")
  return normalized


def _resolve_svdq_smooth_strategy(key: str, value: Any) -> str:
  if not isinstance(value, str):
    raise TypeError(f"svdq_kwargs[{key!r}] must be a str, got {type(value)}.")
  normalized = value.lower()
  if normalized not in _SVDQ_SMOOTH_STRATEGIES:
    raise ValueError(
      f"svdq_kwargs[{key!r}] must be one of {_SVDQ_SMOOTH_STRATEGIES}, got {value!r}.")
  return normalized


def _resolve_svdq_positive_int_or_none(key: str, value: Any) -> int | None:
  if value is None:
    return None
  if isinstance(value, bool) or not isinstance(value, int):
    raise TypeError(f"svdq_kwargs[{key!r}] must be an int or None, got {type(value)}.")
  if value <= 0:
    raise ValueError(f"svdq_kwargs[{key!r}] must be a positive integer, got {value}.")
  return value


def _resolve_svdq_kwargs(svdq_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
  if svdq_kwargs is None:
    return dict(_SVDQ_KWARGS_DEFAULTS)
  if not isinstance(svdq_kwargs, dict):
    raise TypeError(f"svdq_kwargs must be a dict, got {type(svdq_kwargs)}.")

  unknown_keys = set(svdq_kwargs) - set(_SVDQ_KWARGS_DEFAULTS)
  if unknown_keys:
    raise ValueError("Unsupported SVDQ PTQ kwargs: "
                     f"{sorted(unknown_keys)}. Allowed keys: {sorted(_SVDQ_KWARGS_DEFAULTS)}.")

  resolved = dict(_SVDQ_KWARGS_DEFAULTS)
  validators = {
    "streaming": _resolve_svdq_bool_kwarg,
    "calibrate_precision": _resolve_svdq_calibrate_precision,
    "activation_buffer_flush_sample_count": _resolve_svdq_positive_int_or_none,
    "activation_buffer_flush_cpu_bytes": _resolve_svdq_positive_int_or_none,
    "smooth_strategy": _resolve_svdq_smooth_strategy,
  }
  for key, value in svdq_kwargs.items():
    resolved[key] = validators[key](key, value)
  return resolved


@dataclasses.dataclass
class QuantizeConfig:
  """Unified quantization configuration for cache-dit workflows.

  `QuantizeConfig` is the user-facing control surface for both TorchAO-backed online quantization
  flows and cache-dit's native SVDQ workflows. The top-level fields describe what to quantize,
  which backend to use, and how to scope quantization across components or repeated transformer
  blocks, while `svdq_kwargs` contains backend-specific controls for the SVDQ path.
  """

  # Quantization backend, only "ao" (torchao) is supported for now, more backends
  # will be supported in the future. The AUTO option will automatically select the
  # backend based on the hardware and quantization type, etc. Currently it will be
  # resolved to TORCHAO since it's the only supported backend for now.
  backend: str | QuantizeBackend = QuantizeBackend.AUTO
  # Quantization type, currently support "float8_weight_only" and "float8_per_row",
  # "float8_per_tensor", "float8_per_block", "int8_per_row", "int8_per_tensor",
  # "int8_weight_only", "int4_weight_only", etc.
  quant_type: str = "float8_per_row"
  # The layers specified in this variable will be excluded from quantization,
  # even if they are in the repeated blocks or not filtered out by filter_fn.
  # The format of the layer name should be the same as the name in the model's
  # state_dict, e.g, "transformer.blocks.0.attn.to_k.weight". This is useful
  # for cases when some specific layers cannot be quantized for some reasons,
  # e.g, they are already very small and quantization may cause significant
  # accuracy drop, or they are not supported to be quantized due to some
  # technical reasons, etc.
  exclude_layers: Optional[list] = dataclasses.field(default_factory=lambda: [
    "embedder",
    "embed",
    "modulation",
    "mod",
  ])
  # Quantize the _repeated_ blocks in the transformer (Diffusers).
  regional_quantize: bool = True  # name 'regional', vs regional compile.
  # For models outside of diffusers, users can specify the repeated blocks
  # by setting this variable to a list of block names.
  repeated_blocks: List[str] = dataclasses.field(default_factory=list)
  # A filter function to determine whether to quantize a specific module or not,
  # it will be called in the format of filter_fn(m: nn.Module, name: str) -> bool.
  # It should return True if the module needs to be quantized, otherwise False.
  # If filter_fn is specified, the exclude_layers will be ignored.
  filter_fn: Optional[Any] = None  # Usually not use.
  # components_to_quantize: (list[str] or dict[str, str], optional)
  # specify the components to quantize, if None, only the transformer
  # module will be quantized. e.g:
  # - List[str]: ['transformer', 'text_encoder'] quantize to 'quant_type'
  # - Dict[str, Dict[str, str]]: {
  #     'transformer': {'quant_type': 'float8_per_row'},
  #     'text_encoder': {'quant_type': 'float8_weight_only'}
  #   }.
  # The 'quant_type' will be ignored in this case, each module will quantized to
  # it's specified quantization type.
  components_to_quantize: Optional[Union[List[str], Dict[str, Dict[str, str]]]] = None
  # Whether to fallback to float8 quantization when float8 per-row or per-block
  # quantization is not supported for some layers. This is useful for cases when
  # tensor parallelism is applied, and some layers cannot be quantized to float8
  # per-row or per-block, e.g, layers applied RowwiseParallel may not support
  # float8 per-row quantization currently, _scaled_mm will raise memory layout
  # mismatch error when quantized to float8 per-row, setting this flag to True
  # will fallback to float8 per tensor quantization for those layers, instead of
  # raising error. (Only support for float8 quantization for now, int8 fallback
  # is not supported yet.)
  per_tensor_fallback: bool = True
  # Precision plan is a dict specifying the quantization type for each layer, it will
  # override the quant_type and components_to_quantize. The layers not contained in
  # the precision plan will be quantized according to the basic quant_type and
  # components_to_quantize. The format of the dict is
  # {
  #     'attn.to_q': 'float8_per_tensor',   # better performance
  #     'attn.to_k': 'float8_per_row',      # better accuracy
  #     'attn.to_v': 'float8_per_row',      # better accuracy
  #     'attn.to_out': 'float8_per_tensor', # better performance
  #     ...
  # }
  # The keys are the layer names, which should be the same as the name in the model's
  # state_dict, e.g, the layers that contain "to_q", "to_k", "to_v" in their names will
  # be quantized to different types according to the precision_plan. This is useful for
  # cases when users want to have more control over the quantization type of each layer,
  # and want to achieve better accuracy by using different quantization types for different
  # layers based on their sensitivity to quantization. Only valid when the regional quantize
  # is True, otherwise it will be ignored.
  precision_plan: Optional[Dict[str, str]] = None
  # Calibrate function for SVDQuant PTQ workflow, it will be called in the format of:
  # calibrate_fn(model) -> Any, users can run their calibration data through the model
  # in this function, and the calibration data will be collected by the observers
  # registered in the model (outsied the calibration function), this is useful for
  # SVDQuant PTQ workflow, where we need to collect the activation statistics to
  # compute the smooth scale for quantization. Currently this is only used for
  # SVDQuant PTQ workflow, and it will be ignored for other quantization workflows.
  # For example, in SVDQuant PTQ workflow, users can specify the calibrate_fn to run
  # the calibration dataset through the model:
  #   calibration_dataloader = DataLoader(calibration_dataset, batch_size=1)
  #   def calibrate_fn(**kwargs):
  #       pipe.eval()
  #       with torch.inference_mode():
  #           for batch in calibration_dataloader:
  #               pipe(batch)
  calibrate_fn: Optional[Callable[..., Any]] = None
  # The directory to serialize the quantization model, if needed, e.g, for SVDQuant PTQ workflow,
  # after the quantization parameters are computed, we can serialize the quantized model to
  # the specified path for later loading and inference. This is useful for scenarios when users
  # want to separate the quantization process and the inference process, or when the quantization
  # process is time-consuming and they want to save the quantized model for later use.
  # e.g, serialize_to="./FLUX.1-dev-svdq/", quant_type="svdq_int4_r32", after the quantization is
  # done, the quantized model will be serialized to:
  # "./FLUX.1-dev-svdq/svdq_int4_r32.safetensors".
  serialize_to: Optional[str] = None
  # Backend-specific kwargs for SVDQ PTQ. These settings affect not only the
  # calibration callback but also quantization math, serialization metadata,
  # and load compatibility, so they are grouped under a validated backend-
  # specific dict instead of being exposed as many top-level config fields.
  svdq_kwargs: Optional[Dict[str, Any]] = None
  # Whether to print detailed quantization information, such as the quantization
  # type of each layer, the reason for skipping quantization, etc. This is useful
  # for debugging and analysis.
  verbose: bool = False

  def __post_init__(self):
    raw_svdq_kwargs = dict(self.svdq_kwargs) if isinstance(self.svdq_kwargs,
                                                           dict) else self.svdq_kwargs
    if isinstance(self.quant_type, str):
      self.quant_type = self.quant_type.lower()
    # Resolve backend if it's in string format, and validate the backend.
    if isinstance(self.backend, str):
      self.backend = QuantizeBackend.from_str(self.backend)
    if self.backend == QuantizeBackend.AUTO:
      if self.quant_type.lower().startswith("svdq"):
        self.backend = QuantizeBackend.CACHE_DIT
      else:
        self.backend = QuantizeBackend.TORCHAO
    assert QuantizeBackend.is_supported(
      self.backend), f"Quantization backend {self.backend} is not supported in this environment."

    # Validate SVDQuant PTQ workflow configuration
    if self.is_svdq():
      _parse_svdq_quant_type(self.quant_type)
      if self.components_to_quantize is not None:
        raise ValueError("components_to_quantize is not supported for SVDQuant PTQ yet.")
      if self.precision_plan is not None:
        raise ValueError("precision_plan is not supported for SVDQuant PTQ yet.")
      if self.per_tensor_fallback is not True:
        raise ValueError("per_tensor_fallback is not supported for SVDQuant PTQ yet.")
      self.svdq_kwargs = _resolve_svdq_kwargs(self.svdq_kwargs)
      if self.is_svdq_dq():
        if self.calibrate_fn is not None:
          raise ValueError("calibrate_fn is not supported for SVDQuant dynamic quantization.")
        if self.serialize_to is not None:
          raise ValueError("serialize_to is not supported for SVDQuant dynamic quantization.")
        if isinstance(raw_svdq_kwargs, dict) and "smooth_strategy" in raw_svdq_kwargs:
          requested_smooth_strategy = str(raw_svdq_kwargs["smooth_strategy"]).lower()
          if requested_smooth_strategy not in {"identity", "weight", "weight_inv"}:
            raise ValueError(
              "SVDQuant dynamic quantization currently only supports "
              "svdq_kwargs['smooth_strategy'] in {'identity', 'weight', 'weight_inv'}.")
          self.svdq_kwargs["smooth_strategy"] = requested_smooth_strategy
        else:
          self.svdq_kwargs["smooth_strategy"] = "identity"
        self.svdq_kwargs["calibrate_precision"] = "low"
      else:
        if self.svdq_kwargs["smooth_strategy"] != "activation":
          raise ValueError(
            "SVDQuant PTQ currently only supports svdq_kwargs['smooth_strategy'] = 'activation'.")
        if self.calibrate_fn is None:
          raise ValueError("calibrate_fn must be set for SVDQuant PTQ workflow.")
        if self.serialize_to is None:
          raise ValueError("serialize_to must be set for SVDQuant PTQ workflow.")
        normalized_filename = f"{self.quant_type}.safetensors"
        if self.serialize_to.endswith(".safetensors"):
          if os.path.basename(self.serialize_to) != normalized_filename:
            raise ValueError(
              "serialize_to must be a directory path for SVDQuant PTQ, or an already "
              f"normalized file path ending with {normalized_filename!r}.")
          serialize_dir = os.path.dirname(self.serialize_to)
          if not serialize_dir:
            raise ValueError("serialize_to must include a parent directory.")
          os.makedirs(serialize_dir, exist_ok=True)
        else:
          os.makedirs(self.serialize_to, exist_ok=True)
          if not os.path.isdir(self.serialize_to):
            raise ValueError(f"serialize_to should be a directory path, got {self.serialize_to}.")
          self.serialize_to = os.path.join(self.serialize_to, normalized_filename)
    elif self.svdq_kwargs is not None:
      raise ValueError("svdq_kwargs is only valid when quant_type starts with 'svdq'.")

  def as_dict(self) -> Dict[str, Any]:
    """Return the configuration as a plain dictionary.

    :returns: A plain dictionary representation of this quantization config.
    """

    return dataclasses.asdict(self)

  def is_svdq(self) -> bool:
    """Return whether this config targets cache-dit's SVDQ workflow.

    :returns: `True` when `quant_type` selects an `svdq_*` workflow.
    """

    return isinstance(self.quant_type, str) and self.quant_type.startswith("svdq")

  def is_svdq_dq(self) -> bool:
    """Return whether this config targets SVDQ dynamic quantization.

    :returns: `True` when `quant_type` selects an `svdq_*_dq` workflow.
    """

    if not self.is_svdq():
      return False
    _, _, is_dynamic = _parse_svdq_quant_type(self.quant_type)
    return is_dynamic

  def is_svdq_ptq(self) -> bool:
    """Return whether this config targets SVDQ PTQ.

    :returns: `True` when `quant_type` selects an SVDQ PTQ workflow.
    """

    return self.is_svdq() and not self.is_svdq_dq()

  def get_svdq_rank(self) -> int:
    """Extract the low-rank value encoded in an SVDQ quant type.

    :returns: The low-rank value encoded in `quant_type`.
    """

    _, rank, _ = _parse_svdq_quant_type(self.quant_type)
    return rank

  def get_svdq_kwargs(self) -> Dict[str, Any]:
    """Return validated SVDQ-specific kwargs or an empty dict for non-SVDQ configs.

    :returns: The validated SVDQ kwargs for this config, or `{}` for non-SVDQ modes.
    """

    if not self.is_svdq():
      return {}
    return dict(self.svdq_kwargs or _SVDQ_KWARGS_DEFAULTS)

  def update(self, **kwargs) -> "QuantizeConfig":
    """Update non-`None` fields in place and re-run configuration validation.

    :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
    :returns: `self` after applying the updates.
    """

    for key, value in kwargs.items():
      if hasattr(self, key):
        if value is not None:
          setattr(self, key, value)
    self.__post_init__()
    return self

  def strify(self) -> str:
    """Build a compact human-readable summary of the quantization selection.

    :returns: A compact summary string suitable for logs and artifact names.
    """

    def _stringify_quant_type(quant_type: str) -> str:
      quant_type = quant_type.lower()
      if quant_type.startswith("svdq") and quant_type.endswith("_dq"):
        smooth_strategy = self.get_svdq_kwargs().get("smooth_strategy", "identity")
        if smooth_strategy != "identity":
          return f"{quant_type}_{smooth_strategy}"
      return quant_type

    if self.components_to_quantize is None or isinstance(self.components_to_quantize, list):
      return _stringify_quant_type(self.quant_type)
    else:
      quant_str = ""
      if isinstance(self.components_to_quantize, dict):
        for component, d in self.components_to_quantize.items():
          quant_str += f"<{component}:{_stringify_quant_type(d.get('quant_type', self.quant_type))}>"
      return quant_str

  def component_quant_types(self) -> Dict[str, str]:
    """Resolve the effective quantization type for each target component.

    :returns: A mapping from component names to their effective quantization type.
    """

    if self.components_to_quantize is None:
      return {"transformer": self.quant_type}
    elif isinstance(self.components_to_quantize, list):
      return {component: self.quant_type for component in self.components_to_quantize}
    elif isinstance(self.components_to_quantize, dict):
      return {
        component: d.get("quant_type", self.quant_type)
        for component, d in self.components_to_quantize.items()
      }
    else:
      raise ValueError("components_to_quantize should be either a list or a dict.")

  @classmethod
  def expand_configs(cls, config: "QuantizeConfig") -> List["QuantizeConfig"]:
    """Expand a multi-component config into one config per target component.

    :param config: Source config that may target multiple components.
    :returns: One derived config per component-specific quantization target.
    """

    # Transfer components_to_quantize to mutiple simple configs, each
    # with only 1 component to quantize, and the same quantization type.
    if config.components_to_quantize is None:
      return [config]

    if isinstance(config.components_to_quantize, list):
      return [
        dataclasses.replace(config, components_to_quantize=[component])
        for component in config.components_to_quantize
      ]

    if isinstance(config.components_to_quantize, dict):
      return [
        dataclasses.replace(
          config,
          backend=cfg.get("backend", config.backend),
          components_to_quantize=[component],
          quant_type=cfg.get("quant_type", config.quant_type),
          exclude_layers=cfg.get("exclude_layers", config.exclude_layers),
          regional_quantize=cfg.get("regional_quantize", config.regional_quantize),
          repeated_blocks=cfg.get("repeated_blocks", config.repeated_blocks),
          filter_fn=cfg.get("filter_fn", config.filter_fn),
          per_tensor_fallback=cfg.get("per_tensor_fallback", config.per_tensor_fallback),
          precision_plan=cfg.get("precision_plan", config.precision_plan),
          calibrate_fn=cfg.get("calibrate_fn", config.calibrate_fn),
          serialize_to=cfg.get("serialize_to", config.serialize_to),
          svdq_kwargs=cfg.get("svdq_kwargs", config.svdq_kwargs),
          verbose=cfg.get("verbose", config.verbose),
        ) for component, cfg in config.components_to_quantize.items()
      ]

    raise ValueError("components_to_quantize should be either a list or a dict.")

  @classmethod
  def from_kwargs(cls, **kwargs) -> "QuantizeConfig":
    """Build a config from legacy keyword arguments by dropping unknown keys.

    :param kwargs: Additional keyword arguments forwarded to the underlying implementation.
    :returns: A validated `QuantizeConfig` built from the supported keyword subset.
    """

    valid_kwargs = {}
    for key, value in kwargs.items():
      if key in cls.__dataclass_fields__:
        valid_kwargs[key] = value
    return cls(**valid_kwargs)
