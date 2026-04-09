from __future__ import annotations

import dataclasses
import inspect
import json
import os
from typing import Any

import torch
from torch import nn

from ...logger import init_logger
from ...utils import check_quantized
from ..config import QuantizeConfig
from ..config import _parse_svdq_quant_type
from ..config import _resolve_svdq_kwargs
from .linear import SVDQW4A4Linear
from .quantizer import _ActivationSpanAccumulator
from .quantizer import _normalize_dtype
from .quantizer import _quantize_linear_svdq_w4a4_from_activation_span
from .quantizer import _resolve_math_dtype
from .quantizer import validate_svdq_linear_geometry

logger = init_logger(__name__)

_SVDQ_METADATA_KEY = "cache_dit_svdq_ptq"
_SVDQ_FORMAT_VERSION = 2
_SVDQ_QUANT_CONFIG_FORMAT = "cache_dit_svdq_quant_config"
_SVDQ_QUANT_CONFIG_VERSION = 2
_SVDQ_QUANT_CONFIG_FILENAME = "quant_config.json"
_ROOT_LAYER_NAME = "__root__"


@dataclasses.dataclass(frozen=True)
class _ResolvedSVDQLoadSource:
  checkpoint_path: str
  quant_config_snapshot: dict[str, Any] | None = None


def _import_safetensors():
  try:
    from safetensors import safe_open
    from safetensors.torch import save_file
  except ImportError as exc:
    raise ImportError(
      "SVDQ PTQ serialization/loading requires `safetensors`. "
      "Install with `pip install cache-dit[quantization]` or `pip install safetensors`.") from exc
  return safe_open, save_file


def _serialize_layer_name(layer_name: str) -> str:
  return _ROOT_LAYER_NAME if layer_name == "" else layer_name


def _deserialize_layer_name(layer_name: str) -> str:
  return "" if layer_name == _ROOT_LAYER_NAME else layer_name


def _get_named_submodule(module: nn.Module, module_name: str) -> nn.Module:
  return module if module_name == "" else module.get_submodule(module_name)


def _get_parent_and_child(module: nn.Module, module_name: str) -> tuple[nn.Module, str]:
  if not module_name:
    raise ValueError("The root module does not have a parent/child path.")
  if "." not in module_name:
    return module, module_name
  parent_name, child_name = module_name.rsplit(".", 1)
  return module.get_submodule(parent_name), child_name


def _call_with_supported_kwargs(callback: Any, **kwargs) -> Any:
  try:
    signature = inspect.signature(callback)
  except (TypeError, ValueError):
    return callback(**kwargs)

  if any(parameter.kind == inspect.Parameter.VAR_KEYWORD
         for parameter in signature.parameters.values()):
    return callback(**kwargs)

  accepted_kwargs = {name: value for name, value in kwargs.items() if name in signature.parameters}
  return callback(**accepted_kwargs)


def _infer_module_execution_device(module: nn.Module) -> torch.device:
  for submodule in module.modules():
    hook = getattr(submodule, "_hf_hook", None)
    execution_device = getattr(hook, "execution_device", None)
    if execution_device is not None:
      return torch.device(execution_device)

  try:
    parameter = next(module.parameters())
    if parameter.device.type != "meta":
      return parameter.device
  except StopIteration:
    pass

  try:
    buffer = next(module.buffers())
    if buffer.device.type != "meta":
      return buffer.device
  except StopIteration:
    pass

  return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _log_boxed_summary(lines: list[str]) -> None:
  if not lines:
    return
  max_length = max(len(line) for line in lines) + 2
  logger.info("-" * max_length)
  logger.info("\n".join(line.ljust(max_length) + "|" for line in lines))
  logger.info("-" * max_length)


def _log_skipped_reasons(skipped_reasons: list[str], *, verbose: bool) -> None:
  if not verbose or not skipped_reasons:
    return

  skipped_reasons_counter: dict[str, int] = {}
  for reason in skipped_reasons:
    skipped_reasons_counter[reason] = skipped_reasons_counter.get(reason, 0) + 1

  rows: list[tuple[str, str, int]] = []
  name_width = 0
  reason_width = 0
  for entry, count in skipped_reasons_counter.items():
    layer_name, reason = entry.split(" -> ", 1)
    rows.append((layer_name, reason, count))
    name_width = max(name_width, len(layer_name))
    reason_width = max(reason_width, len(reason))

  detail_lines = [
    f"{layer_name.ljust(name_width)}: {reason.ljust(reason_width)}: {count:<4} layers"
    for layer_name, reason, count in rows
  ]
  _log_boxed_summary(detail_lines)


def _get_checkpoint_dir(checkpoint_path: str) -> str:
  return os.path.dirname(os.path.abspath(checkpoint_path))


def _get_quant_config_path(checkpoint_path: str) -> str:
  return os.path.join(_get_checkpoint_dir(checkpoint_path), _SVDQ_QUANT_CONFIG_FILENAME)


def _attach_quantization_metadata(
  module: nn.Module,
  *,
  quant_type: str,
  quantized_layer_names: list[str],
  exclude_layers: list[str],
  svdq_kwargs: dict[str, Any],
  checkpoint_path: str | None,
  quantize_config: QuantizeConfig | None = None,
) -> None:
  module._is_quantized = True
  module._quantize_type = quant_type
  module._exclude_layers = list(exclude_layers)
  module._svdq_quantized_layers = tuple(quantized_layer_names)
  module._svdq_kwargs = dict(svdq_kwargs)
  module._svdq_checkpoint_path = checkpoint_path
  if quantize_config is not None:
    module._quantize_config = quantize_config


@dataclasses.dataclass
class SVDQPTQContext:
  """Resolved plan for a module-level SVDQ PTQ run.

  The context snapshots the quantization scope, discovered candidate `nn.Linear` layers, skip
  reasons, and resolved SVDQ kwargs derived from the `QuantizeConfig`. It is shared between
  calibration, replacement, serialization, and summary logging.
  """

  root_module: nn.Module
  quantize_config: QuantizeConfig
  rank: int
  precision: str = "int4"
  regional_quantize: bool = True
  repeated_blocks: list[str] | None = None
  exclude_layers: list[str] = dataclasses.field(default_factory=list)
  filter_fn: Any = None
  svdq_kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
  verbose: bool = False
  linear_layer_names: list[str] = dataclasses.field(default_factory=list)
  candidate_layer_names: list[str] = dataclasses.field(default_factory=list)
  skipped_reasons: list[str] = dataclasses.field(default_factory=list)

  @property
  def quantized_region(self) -> str:
    if self.regional_quantize and self.repeated_blocks is not None:
      return str(self.repeated_blocks)
    return self.root_module.__class__.__name__

  @classmethod
  def from_config(cls, root_module: nn.Module, quantize_config: QuantizeConfig) -> "SVDQPTQContext":
    """Resolve a PTQ execution context from a root module and config.

    This gathers the repeated-block scope, exclusion rules, resolved SVDQ kwargs, and the final list
    of candidate linear layers that satisfy the migrated W4A4 geometry constraints.

    :param root_module: Root module that will be searched for quantizable linear layers.
    :param quantize_config: Structured quantization configuration.
    :returns: A resolved PTQ context describing scope, candidates, and SVDQ settings.
    """

    repeated_blocks = getattr(
      root_module,
      "_repeated_blocks",
      quantize_config.repeated_blocks if quantize_config.repeated_blocks else None,
    )
    regional_quantize = quantize_config.regional_quantize and repeated_blocks is not None
    context = cls(
      root_module=root_module,
      quantize_config=quantize_config,
      rank=quantize_config.get_svdq_rank(),
      regional_quantize=regional_quantize,
      repeated_blocks=list(repeated_blocks) if repeated_blocks is not None else None,
      exclude_layers=list(quantize_config.exclude_layers or []),
      filter_fn=quantize_config.filter_fn,
      svdq_kwargs=quantize_config.get_svdq_kwargs(),
      verbose=quantize_config.verbose,
    )
    context.candidate_layer_names = context._collect_candidate_layer_names()
    return context

  def _record_skip(self, layer_name: str, reason: str) -> None:
    self.skipped_reasons.append(f"{layer_name or _ROOT_LAYER_NAME} -> {reason}")

  def _is_excluded(self, layer_name: str) -> bool:
    for exclude_name in self.exclude_layers:
      if exclude_name in layer_name:
        return True
    return False

  def _iter_modules_in_scope(self):
    if not self.regional_quantize or not self.repeated_blocks:
      yield from self.root_module.named_modules()
      return

    region_found = False
    seen_names: set[str] = set()
    for block_name, block in self.root_module.named_modules():
      if not block_name or block.__class__.__name__ not in self.repeated_blocks:
        continue
      region_found = True
      for local_name, submodule in block.named_modules():
        full_name = block_name if local_name == "" else f"{block_name}.{local_name}"
        if full_name in seen_names:
          continue
        seen_names.add(full_name)
        yield full_name, submodule

    if not region_found:
      raise ValueError(
        f"Regional SVDQ PTQ failed because repeated block classes {self.repeated_blocks} "
        "were not found in the target module.")

  def _collect_candidate_layer_names(self) -> list[str]:
    candidate_layer_names: list[str] = []
    seen_names: set[str] = set()
    for layer_name, submodule in self._iter_modules_in_scope():
      if layer_name in seen_names or not isinstance(submodule, nn.Linear):
        continue
      seen_names.add(layer_name)
      self.linear_layer_names.append(layer_name)

      if self.filter_fn is not None:
        if not bool(
            _call_with_supported_kwargs(
              self.filter_fn,
              m=submodule,
              module=submodule,
              name=layer_name,
            )):
          self._record_skip(layer_name, "filtered by filter_fn")
          continue
      elif self._is_excluded(layer_name):
        self._record_skip(layer_name, "excluded by exclude_layers")
        continue

      try:
        validate_svdq_linear_geometry(
          submodule.in_features,
          submodule.out_features,
          rank=self.rank,
          precision=self.precision,
        )
      except (NotImplementedError, ValueError) as exc:
        self._record_skip(layer_name, str(exc))
        continue

      candidate_layer_names.append(layer_name)

    if not candidate_layer_names:
      raise ValueError("SVDQ PTQ found no quantizable Linear layers in the target module. "
                       f"Skipped layers: {self.skipped_reasons}.")
    return candidate_layer_names

  def quantize_summary_lines(
    self,
    *,
    observed_layer_names: list[str],
    quantized_layer_names: list[str],
    serialize_to: str,
  ) -> list[str]:
    skipped_total = max(len(self.linear_layer_names) - len(quantized_layer_names), 0)
    lines = [
      f"SVDQ PTQ         Region: {self.quantized_region}",
      f"SVDQ PTQ     Quant Type: {self.quantize_config.quant_type}",
      f"SVDQ PTQ           Rank: {self.rank}",
      f"Observed  Linear Layers: {len(observed_layer_names)} / {len(self.candidate_layer_names)}",
      f"Quantized Linear Layers: {len(quantized_layer_names)} / {len(self.candidate_layer_names)}",
      f"Skipped   Linear Layers: {skipped_total}",
      f"Linear           Layers: {len(self.linear_layer_names)}",
    ]
    if self.verbose:
      lines.append(f"SVDQ             Kwargs: {self.svdq_kwargs}")
      lines.append(f"Skipped        Patterns: {self.exclude_layers}")
      lines.append(f"Checkpoint         Path: {serialize_to}")
    return lines


class SVDQPTQCalibrator:
  """Collect per-layer activation spans for module-level SVDQ PTQ.

  The calibrator registers forward-pre-hooks on candidate float `nn.Linear`
  layers, reduces each observed activation batch to a per-channel span vector,
  and exposes the finalized spans to `quantize_svdq_ptq`.
  """

  def __init__(self, context: SVDQPTQContext) -> None:
    self.context = context
    self.activation_spans: dict[str, torch.Tensor] = {}
    self._handles: list[Any] = []
    self._accumulators: dict[str, _ActivationSpanAccumulator] = {}

  def _make_hook(self, layer_name: str):
    accumulator = self._accumulators[layer_name]

    def hook(module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
      if not args or not isinstance(args[0], torch.Tensor):
        raise TypeError(
          f"SVDQ PTQ expected tensor inputs for layer {layer_name or _ROOT_LAYER_NAME}.")
      if not isinstance(module, nn.Linear):
        raise TypeError(f"Expected nn.Linear during calibration, got {type(module)}.")

      activation = args[0]
      if activation.shape[-1] != module.in_features:
        raise ValueError(
          f"Expected activation last dim {module.in_features} for layer {layer_name}, "
          f"got {activation.shape[-1]}.")

      accumulator.add_tensor(activation.detach().reshape(-1, module.in_features))

    return hook

  def register(self) -> None:
    """Register calibration hooks on all candidate float linear layers."""

    calibrate_precision = self.context.svdq_kwargs["calibrate_precision"]
    flush_sample_count = self.context.svdq_kwargs["activation_buffer_flush_sample_count"]
    flush_cpu_bytes = self.context.svdq_kwargs["activation_buffer_flush_cpu_bytes"]

    for layer_name in self.context.candidate_layer_names:
      submodule = _get_named_submodule(self.context.root_module, layer_name)
      if not isinstance(submodule, nn.Linear):
        raise TypeError(
          f"Expected nn.Linear during calibration registration, got {type(submodule)}.")
      torch_dtype = _normalize_dtype(None, submodule.weight.device)
      math_dtype = _resolve_math_dtype(torch_dtype, calibrate_precision)
      self._accumulators[layer_name] = _ActivationSpanAccumulator(
        device=submodule.weight.device,
        torch_dtype=torch_dtype,
        math_dtype=math_dtype,
        flush_sample_count=flush_sample_count,
        flush_cpu_bytes=flush_cpu_bytes,
      )
      self._handles.append(submodule.register_forward_pre_hook(self._make_hook(layer_name)))

  def remove(self) -> None:
    """Remove calibration hooks and finalize any buffered activation spans."""

    for handle in self._handles:
      handle.remove()
    self._handles.clear()
    self.finalize()

  def finalize(self) -> None:
    """Materialize finalized activation spans for all observed layers."""

    for layer_name, accumulator in self._accumulators.items():
      if accumulator.has_observations:
        self.activation_spans[layer_name] = accumulator.finalize()


def _build_serialized_metadata(
  *,
  quant_type: str,
  rank: int,
  quantized_layer_names: list[str],
  svdq_kwargs: dict[str, Any],
) -> dict[str, str]:
  payload = {
    "format": "cache_dit_svdq_ptq",
    "version": _SVDQ_FORMAT_VERSION,
    "quant_type": quant_type,
    "rank": rank,
    "precision": "int4",
    "quantized_layer_names": [_serialize_layer_name(name) for name in quantized_layer_names],
    "svdq_kwargs": svdq_kwargs,
  }
  return {_SVDQ_METADATA_KEY: json.dumps(payload, sort_keys=True)}


def _build_quant_config_snapshot(
  *,
  checkpoint_path: str,
  quant_type: str,
  rank: int,
  exclude_layers: list[str],
  regional_quantize: bool,
  repeated_blocks: list[str] | None,
  verbose: bool,
  svdq_kwargs: dict[str, Any],
  backend: Any,
) -> dict[str, Any]:
  checkpoint_dir = _get_checkpoint_dir(checkpoint_path)
  return {
    "format": _SVDQ_QUANT_CONFIG_FORMAT,
    "version": _SVDQ_QUANT_CONFIG_VERSION,
    "backend": backend.value if hasattr(backend, "value") else str(backend),
    "quant_type": quant_type,
    "rank": rank,
    "checkpoint_path": os.path.relpath(checkpoint_path, checkpoint_dir),
    "exclude_layers": list(exclude_layers),
    "regional_quantize": regional_quantize,
    "repeated_blocks": list(repeated_blocks) if repeated_blocks is not None else None,
    "verbose": verbose,
    "svdq_kwargs": dict(svdq_kwargs),
  }


def _save_quant_config_snapshot(snapshot: dict[str, Any], *, checkpoint_path: str) -> str:
  quant_config_path = _get_quant_config_path(checkpoint_path)
  with open(quant_config_path, "w", encoding="utf-8") as file:
    json.dump(snapshot, file, indent=2, sort_keys=True)
    file.write("\n")
  return quant_config_path


def _save_quantized_module(
  module: nn.Module,
  *,
  serialize_to: str,
  quant_type: str,
  rank: int,
  quantized_layer_names: list[str],
  svdq_kwargs: dict[str, Any],
) -> None:
  _, save_file = _import_safetensors()
  tensors: dict[str, torch.Tensor] = {}
  for layer_name in quantized_layer_names:
    submodule = _get_named_submodule(module, layer_name)
    if not isinstance(submodule, SVDQW4A4Linear):
      raise TypeError(
        f"Expected SVDQW4A4Linear at {layer_name or _ROOT_LAYER_NAME}, got {type(submodule)}.")
    serialized_name = _serialize_layer_name(layer_name)
    for key, tensor in submodule.state_dict().items():
      tensors[f"{serialized_name}.{key}"] = tensor.detach().cpu().contiguous()

  save_file(
    tensors,
    serialize_to,
    metadata=_build_serialized_metadata(
      quant_type=quant_type,
      rank=rank,
      quantized_layer_names=quantized_layer_names,
      svdq_kwargs=svdq_kwargs,
    ),
  )


def _validate_quant_config_snapshot(path: str, payload: dict[str, Any]) -> dict[str, Any]:
  if payload.get("format") != _SVDQ_QUANT_CONFIG_FORMAT:
    raise ValueError(f"Invalid SVDQ quant_config format in {path}: {payload.get('format')!r}.")
  if payload.get("version") != _SVDQ_QUANT_CONFIG_VERSION:
    raise ValueError(f"Unsupported SVDQ quant_config version {payload.get('version')} in {path}.")

  quant_type = payload.get("quant_type")
  if not isinstance(quant_type, str):
    raise ValueError(f"SVDQ quant_config at {path} is missing a valid quant_type.")
  _parse_svdq_quant_type(quant_type)

  rank = payload.get("rank")
  if not isinstance(rank, int):
    raise ValueError(f"SVDQ quant_config at {path} is missing a valid integer rank.")

  checkpoint_path = payload.get("checkpoint_path")
  if not isinstance(checkpoint_path, str) or not checkpoint_path:
    raise ValueError(f"SVDQ quant_config at {path} is missing a valid checkpoint_path.")

  exclude_layers = payload.get("exclude_layers", [])
  if not isinstance(exclude_layers, list) or any(not isinstance(layer_name, str)
                                                 for layer_name in exclude_layers):
    raise ValueError(f"SVDQ quant_config at {path} has an invalid exclude_layers field.")

  repeated_blocks = payload.get("repeated_blocks")
  if repeated_blocks is not None and (not isinstance(repeated_blocks, list)
                                      or any(not isinstance(block_name, str)
                                             for block_name in repeated_blocks)):
    raise ValueError(f"SVDQ quant_config at {path} has an invalid repeated_blocks field.")

  regional_quantize = payload.get("regional_quantize", True)
  if not isinstance(regional_quantize, bool):
    raise ValueError(f"SVDQ quant_config at {path} has an invalid regional_quantize field.")

  verbose = payload.get("verbose", False)
  if not isinstance(verbose, bool):
    raise ValueError(f"SVDQ quant_config at {path} has an invalid verbose field.")

  svdq_kwargs = payload.get("svdq_kwargs", {})
  payload["svdq_kwargs"] = _resolve_svdq_kwargs(svdq_kwargs)
  payload["exclude_layers"] = exclude_layers
  payload["regional_quantize"] = regional_quantize
  payload["repeated_blocks"] = repeated_blocks
  payload["verbose"] = verbose
  return payload


def _load_quant_config_snapshot(path: str) -> dict[str, Any]:
  if not os.path.isfile(path):
    raise FileNotFoundError(f"SVDQ quant_config.json not found: {path}")

  with open(path, "r", encoding="utf-8") as file:
    try:
      payload = json.load(file)
    except json.JSONDecodeError as exc:
      raise ValueError(f"Invalid SVDQ quant_config JSON in {path}: {exc.msg}.") from exc

  if not isinstance(payload, dict):
    raise ValueError(f"SVDQ quant_config at {path} must contain a JSON object.")
  return _validate_quant_config_snapshot(path, payload)


def _maybe_load_quant_config_snapshot_for_checkpoint(checkpoint_path: str) -> dict[str, Any] | None:
  quant_config_path = _get_quant_config_path(checkpoint_path)
  if not os.path.isfile(quant_config_path):
    return None
  return _load_quant_config_snapshot(quant_config_path)


def _resolve_directory_load_source(directory_path: str) -> _ResolvedSVDQLoadSource:
  normalized_directory_path = os.path.abspath(directory_path)
  if not os.path.isdir(normalized_directory_path):
    raise FileNotFoundError(f"SVDQ PTQ checkpoint directory not found: {directory_path}")

  quant_config_path = os.path.join(normalized_directory_path, _SVDQ_QUANT_CONFIG_FILENAME)
  snapshot = _load_quant_config_snapshot(quant_config_path)
  checkpoint_path = snapshot["checkpoint_path"]
  resolved_checkpoint_path = (checkpoint_path
                              if os.path.isabs(checkpoint_path) else os.path.normpath(
                                os.path.join(normalized_directory_path, checkpoint_path)))
  if not os.path.isfile(resolved_checkpoint_path):
    raise FileNotFoundError("SVDQ PTQ checkpoint referenced by quant_config.json not found: "
                            f"{resolved_checkpoint_path}")
  return _ResolvedSVDQLoadSource(
    checkpoint_path=resolved_checkpoint_path,
    quant_config_snapshot=snapshot,
  )


def _resolve_load_path(quantize_config_or_path: QuantizeConfig | str, ) -> _ResolvedSVDQLoadSource:
  if isinstance(quantize_config_or_path, QuantizeConfig):
    if quantize_config_or_path.serialize_to is None:
      raise ValueError("serialize_to must be set when loading SVDQ PTQ from QuantizeConfig.")
    return _ResolvedSVDQLoadSource(
      checkpoint_path=os.path.abspath(quantize_config_or_path.serialize_to))
  if isinstance(quantize_config_or_path, str):
    if os.path.isdir(quantize_config_or_path):
      return _resolve_directory_load_source(quantize_config_or_path)
    checkpoint_path = os.path.abspath(quantize_config_or_path)
    return _ResolvedSVDQLoadSource(
      checkpoint_path=checkpoint_path,
      quant_config_snapshot=_maybe_load_quant_config_snapshot_for_checkpoint(checkpoint_path),
    )
  raise TypeError("quantize_config_or_path must be a QuantizeConfig or a safetensors file path, "
                  f"got {type(quantize_config_or_path)}.")


def _validate_serialized_payload(path: str, payload: dict[str, Any]) -> dict[str, Any]:
  if payload.get("format") != "cache_dit_svdq_ptq":
    raise ValueError(f"Invalid SVDQ PTQ checkpoint format in {path}: {payload.get('format')!r}.")
  if payload.get("version") != _SVDQ_FORMAT_VERSION:
    raise ValueError(f"Unsupported SVDQ PTQ checkpoint version {payload.get('version')} in {path}.")

  quant_type = payload.get("quant_type")
  if not isinstance(quant_type, str):
    raise ValueError(f"SVDQ PTQ checkpoint at {path} is missing a valid quant_type.")
  _parse_svdq_quant_type(quant_type)

  rank = payload.get("rank")
  if not isinstance(rank, int):
    raise ValueError(f"SVDQ PTQ checkpoint at {path} is missing a valid integer rank.")

  quantized_layer_names = payload.get("quantized_layer_names")
  if not isinstance(quantized_layer_names, list) or not quantized_layer_names:
    raise ValueError(
      f"SVDQ PTQ checkpoint at {path} must contain a non-empty quantized_layer_names list.")

  svdq_kwargs = payload.get("svdq_kwargs", {})
  payload["svdq_kwargs"] = _resolve_svdq_kwargs(svdq_kwargs)
  return payload


def _load_serialized_checkpoint(path: str) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
  if not os.path.isfile(path):
    raise FileNotFoundError(f"SVDQ PTQ checkpoint not found: {path}")

  safe_open, _ = _import_safetensors()
  with safe_open(path, framework="pt", device="cpu") as handle:
    metadata = handle.metadata() or {}
    payload_raw = metadata.get(_SVDQ_METADATA_KEY)
    if payload_raw is None:
      raise ValueError(f"SVDQ PTQ metadata key {_SVDQ_METADATA_KEY!r} not found in {path}.")
    try:
      payload = json.loads(payload_raw)
    except json.JSONDecodeError as exc:
      raise ValueError(f"Invalid SVDQ PTQ metadata JSON in {path}: {exc.msg}.") from exc
    tensors = {key: handle.get_tensor(key) for key in handle.keys()}

  return tensors, _validate_serialized_payload(path, payload)


def _log_quantize_summary(
  context: SVDQPTQContext,
  *,
  quantized_layer_names: list[str],
  serialize_to: str,
  observed_layer_names: list[str],
) -> None:
  _log_boxed_summary(
    context.quantize_summary_lines(
      observed_layer_names=observed_layer_names,
      quantized_layer_names=quantized_layer_names,
      serialize_to=serialize_to,
    ))
  _log_skipped_reasons(context.skipped_reasons, verbose=context.verbose)


def _log_load_summary(
  *,
  module: nn.Module,
  quant_type: str,
  rank: int,
  loaded_layer_names: list[str],
  checkpoint_path: str,
  svdq_kwargs: dict[str, Any],
  verbose: bool,
) -> None:
  lines = [
    f"SVDQ Load       Region: {module.__class__.__name__}",
    f"SVDQ Load   Quant Type: {quant_type}",
    f"SVDQ Load         Rank: {rank}",
    f"Loaded   Linear Layers: {len(loaded_layer_names)}",
  ]
  if verbose:
    lines.append(f"SVDQ             Kwargs: {svdq_kwargs}")
    lines.append(f"Loaded           Layers: {loaded_layer_names}")
    lines.append(f"Checkpoint         Path: {checkpoint_path}")
  _log_boxed_summary(lines)


def quantize_svdq_ptq(module: nn.Module, quantize_config: QuantizeConfig) -> nn.Module:
  """Run post-training SVDQ quantization over a module in place.

  :param module: Root module containing the float `nn.Linear` layers to quantize.
  :param quantize_config: SVDQ `QuantizeConfig` describing calibration, scope
  filters, serialization target, and SVDQ-specific kwargs.

  :returns: The input module with eligible `nn.Linear` submodules replaced by
  `SVDQW4A4Linear` instances. The return value may be a new module only when
  the root module itself is quantized.
  """

  if not isinstance(module, nn.Module):
    raise TypeError(f"Expected nn.Module, got {type(module)}.")
  if not quantize_config.is_svdq():
    raise ValueError(
      f"quantize_svdq_ptq only supports SVDQ quant types, got {quantize_config.quant_type}.")
  if check_quantized(module):
    logger.warning("Module %s is already quantized, skipping SVDQ PTQ.", module.__class__.__name__)
    return module

  context = SVDQPTQContext.from_config(module, quantize_config)
  calibrator = SVDQPTQCalibrator(context)
  quantize_device = _infer_module_execution_device(module)
  calibrator.register()

  was_training = module.training
  module.eval()
  try:
    with torch.inference_mode():
      _call_with_supported_kwargs(
        quantize_config.calibrate_fn,
        model=module,
        quantize_config=quantize_config,
        calibrator=calibrator,
        component_name=getattr(module, "_actual_module_name", module.__class__.__name__),
        svdq_kwargs=context.svdq_kwargs,
      )
  finally:
    calibrator.remove()
    module.train(was_training)

  quantized_layer_names: list[str] = []
  for layer_name in context.candidate_layer_names:
    activation_span = calibrator.activation_spans.get(layer_name)
    if activation_span is None:
      context._record_skip(layer_name, "not observed during calibration")
      continue

    float_module = _get_named_submodule(module, layer_name)
    if not isinstance(float_module, nn.Linear):
      raise TypeError(
        f"Expected nn.Linear at {layer_name or _ROOT_LAYER_NAME}, got {type(float_module)}.")

    quantized_module = _quantize_linear_svdq_w4a4_from_activation_span(
      float_module,
      activation_span,
      rank=context.rank,
      precision=context.precision,
      torch_dtype=float_module.weight.dtype,
      device=quantize_device,
      calibrate_precision=context.svdq_kwargs["calibrate_precision"],
    )

    if layer_name == "":
      module = quantized_module
    else:
      parent, child_name = _get_parent_and_child(module, layer_name)
      setattr(parent, child_name, quantized_module)
    quantized_layer_names.append(layer_name)

  if not quantized_layer_names:
    raise RuntimeError("SVDQ PTQ completed calibration but no layers were quantized. "
                       f"Skipped layers: {context.skipped_reasons}.")

  _attach_quantization_metadata(
    module,
    quant_type=quantize_config.quant_type,
    quantized_layer_names=quantized_layer_names,
    exclude_layers=context.exclude_layers,
    svdq_kwargs=context.svdq_kwargs,
    checkpoint_path=quantize_config.serialize_to,
    quantize_config=quantize_config,
  )
  _save_quantized_module(
    module,
    serialize_to=quantize_config.serialize_to,
    quant_type=quantize_config.quant_type,
    rank=context.rank,
    quantized_layer_names=quantized_layer_names,
    svdq_kwargs=context.svdq_kwargs,
  )
  _save_quant_config_snapshot(
    _build_quant_config_snapshot(
      checkpoint_path=quantize_config.serialize_to,
      quant_type=quantize_config.quant_type,
      rank=context.rank,
      exclude_layers=context.exclude_layers,
      regional_quantize=context.regional_quantize,
      repeated_blocks=context.repeated_blocks,
      verbose=context.verbose,
      svdq_kwargs=context.svdq_kwargs,
      backend=quantize_config.backend,
    ),
    checkpoint_path=quantize_config.serialize_to,
  )
  module.train(was_training)
  _log_quantize_summary(
    context,
    quantized_layer_names=quantized_layer_names,
    serialize_to=quantize_config.serialize_to,
    observed_layer_names=sorted(calibrator.activation_spans),
  )
  return module


def load_svdq(
  module: nn.Module,
  quantize_config_or_path: QuantizeConfig | str,
) -> nn.Module:
  """Load a serialized SVDQ PTQ checkpoint into a float module in place.

  :param module: Root module containing float `nn.Linear` layers that match the
  serialized checkpoint layout.
  :param quantize_config_or_path: Either a `QuantizeConfig` whose `serialize_to`
  points at an SVDQ checkpoint or a direct checkpoint path.

  :returns: The module with serialized layers replaced by `SVDQW4A4Linear` instances.
  The return value may be a new module if the root layer itself was
  serialized and loaded.
  """

  if not isinstance(module, nn.Module):
    raise TypeError(f"Expected nn.Module, got {type(module)}.")

  resolved_load_source = _resolve_load_path(quantize_config_or_path)
  checkpoint_path = resolved_load_source.checkpoint_path
  tensors, payload = _load_serialized_checkpoint(checkpoint_path)
  quant_type = payload["quant_type"]
  rank = int(payload["rank"])
  quantized_layer_names = [
    _deserialize_layer_name(name) for name in payload.get("quantized_layer_names", [])
  ]
  svdq_kwargs = dict(payload.get("svdq_kwargs", {}))
  quant_config_snapshot = resolved_load_source.quant_config_snapshot

  if quant_config_snapshot is not None and quant_config_snapshot["quant_type"] != quant_type:
    raise ValueError("The SVDQ checkpoint quant_type does not match quant_config.json: "
                     f"{quant_type} vs {quant_config_snapshot['quant_type']}.")

  if isinstance(quantize_config_or_path, QuantizeConfig):
    if not quantize_config_or_path.is_svdq():
      raise ValueError(
        "load_svdq only supports QuantizeConfig entries whose quant_type starts with 'svdq'.")
    if quantize_config_or_path.quant_type != quant_type:
      raise ValueError("The SVDQ checkpoint quant_type does not match QuantizeConfig: "
                       f"{quant_type} vs {quantize_config_or_path.quant_type}.")

  load_device = _infer_module_execution_device(module)
  was_training = module.training
  loaded_layer_names: list[str] = []
  for layer_name in quantized_layer_names:
    serialized_name = _serialize_layer_name(layer_name)
    layer_state_dict = {
      key.removeprefix(f"{serialized_name}."): tensor
      for key, tensor in tensors.items() if key.startswith(f"{serialized_name}.")
    }
    if not layer_state_dict:
      raise ValueError(f"No serialized tensors found for SVDQ layer {layer_name!r}.")

    float_module = _get_named_submodule(module, layer_name)
    if not isinstance(float_module, nn.Linear):
      raise TypeError(
        f"Expected a float nn.Linear at {layer_name or _ROOT_LAYER_NAME} when loading SVDQ PTQ, "
        f"got {type(float_module)}.")

    torch_dtype = layer_state_dict["smooth_factor"].dtype
    quantized_module = SVDQW4A4Linear.from_linear(
      float_module,
      rank=rank,
      precision="int4",
      torch_dtype=torch_dtype,
      device=load_device,
    )
    incompatible = quantized_module.load_state_dict(layer_state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
      raise RuntimeError(
        f"Unexpected SVDQ PTQ state_dict mismatch for {layer_name!r}: "
        f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}.")

    if layer_name == "":
      module = quantized_module
    else:
      parent, child_name = _get_parent_and_child(module, layer_name)
      setattr(parent, child_name, quantized_module)
    loaded_layer_names.append(layer_name)

  _attach_quantization_metadata(
    module,
    quant_type=quant_type,
    quantized_layer_names=loaded_layer_names,
    exclude_layers=(list(quantize_config_or_path.exclude_layers or []) if isinstance(
      quantize_config_or_path, QuantizeConfig) else list((quant_config_snapshot or {}).get(
        "exclude_layers", getattr(module, "_exclude_layers", [])))),
    svdq_kwargs=svdq_kwargs,
    checkpoint_path=checkpoint_path,
    quantize_config=(quantize_config_or_path
                     if isinstance(quantize_config_or_path, QuantizeConfig) else None),
  )
  module.train(was_training)
  _log_load_summary(
    module=module,
    quant_type=quant_type,
    rank=rank,
    loaded_layer_names=loaded_layer_names,
    checkpoint_path=checkpoint_path,
    svdq_kwargs=svdq_kwargs,
    verbose=isinstance(quantize_config_or_path, QuantizeConfig) and quantize_config_or_path.verbose,
  )
  return module


__all__ = ["load_svdq", "quantize_svdq_ptq", "SVDQPTQCalibrator"]
