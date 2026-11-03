r"""Opt-in per-layer accuracy observers for quantization analysis.

The observers answer one question: *which* linear layers does fp8 hurt? For
every quantized ``nn.Linear`` they keep the pre-quantization bf16 weight and,
on each forward, recompute the same GEMM in bf16 to get a reference output.
The statistics accumulated per layer are:

- ``rel_err``: ``||y_q - y_ref|| / ||y_ref||`` over all observed tokens,
- ``cos``: cosine similarity between the flattened outputs,
- ``out_amax``: max ``|y_ref|`` (are we in a large-magnitude regime?),
- ``in_outlier``: ``mean(amax(|x|) / mean(|x|))`` per token, which predicts how
  badly a per-token fp8 scale rounds the bulk of the row,
- ``w_rel_err``: relative L2 error of the *weight* alone
  (``||dequant(quant(W))||`` vs the original ``W``), measured once.

**Disabled by default.** The only switch is
:attr:`cache_dit.QuantizeConfig.observer`; it is ``None`` unless the caller
opts in::

    from cache_dit import QuantizeConfig, QuantizationObserverConfig

    config = QuantizeConfig(
      quant_type="float8_per_row",
      observer=QuantizationObserverConfig(
        enabled=True,
        json_path="./fp8_observers.json",
        max_steps=1,
        # Optional regex filter (re.search on the fully qualified name):
        # only these layers are observed and snapshotted.
        layer_name_patterns=[r"feed_forward", r"norm1\.linear$"],
      ),
    )

When the switch is off nothing in this module runs: no module walk, no bf16
weight copy, no hooks, so `quantize()` and the inference forward are unchanged
and free of overhead.

**Cost when enabled** (analysis runs only, never for benchmarks):
the observer keeps one bf16 copy of every observed linear weight (host RAM by
default, ``keep_refs_on_gpu=True`` moves it to the device) and adds a bf16 GEMM
per observed layer per forward, i.e. roughly 1.5-2x the transformer forward
time on double/single-stream DiT models, plus a few seconds of host-to-device
traffic when the references stay on the CPU.
"""

import atexit
import dataclasses
import json
import re
from typing import Dict, List, Optional

import torch

from ..logger import init_logger

logger = init_logger(__name__)


@dataclasses.dataclass
class QuantizationObserverConfig:
  """Switch and options for the per-layer quantization observers.

  :param enabled: Master switch; ``False`` (the default) makes the observer a
      no-op, including no module walk and no hooks.
  :param json_path: Where :func:`dump` writes the aggregated statistics;
      defaults to ``./quant_observer.json`` in the current working directory.
  :param max_steps: Number of transformer forward passes to observe before the
      JSON is refreshed; ``<=0`` only dumps at interpreter exit.
  :param keep_refs_on_gpu: Keep the bf16 reference weights on the device
      instead of streaming them per call (faster, but roughly doubles the
      resident weight memory of the observed model).
  :param layer_name_patterns: Python regular expressions matched against the
      fully qualified layer name with ``re.search`` semantics, so an unanchored
      fragment is enough (``r"feed_forward"``, ``r"single_stream_layers\\.(2[89]|3[01])\\."``).
      Only matching layers are observed, and the filter runs before the bf16
      snapshot so the others cost neither memory nor time. ``None`` or an empty
      list observes every quantized linear.
  """

  enabled: bool = False
  json_path: Optional[str] = None
  max_steps: int = 1
  keep_refs_on_gpu: bool = False
  layer_name_patterns: Optional[List[str]] = None
  _compiled_patterns: tuple = dataclasses.field(default=(), init=False, repr=False, compare=False)

  def __post_init__(self):
    # A single string is a common shorthand for a one-pattern filter.
    if isinstance(self.layer_name_patterns, str):
      self.layer_name_patterns = [self.layer_name_patterns]
    if self.layer_name_patterns:
      try:
        self._compiled_patterns = tuple(re.compile(pattern) for pattern in self.layer_name_patterns)
      except re.error as error:
        raise ValueError(f"invalid layer_name_patterns regex: {error}") from error

  def matches(self, name: str) -> bool:
    """Decide whether one layer should be observed.

    :param name: Fully qualified layer name, e.g.
        ``single_stream_layers.30.feed_forward.linear_3``.
    :returns: True when no filter is configured or any pattern matches.
    """
    if not self._compiled_patterns:
      return True
    return any(pattern.search(name) for pattern in self._compiled_patterns)


@dataclasses.dataclass
class _Bucket:
  """Statistics of one observed model."""

  model: str
  refs: Dict[str, tuple]
  stats: Dict[str, dict] = dataclasses.field(default_factory=dict)
  sentinel: str = ""
  steps: int = 0


_BUCKETS: List[_Bucket] = []
_CONFIG: Optional[QuantizationObserverConfig] = None
_HOOKS: List = []
_DUMPED = False


def is_enabled(config: Optional[QuantizationObserverConfig]) -> bool:
  """:param config: Observer config or None.

  :returns: True only when the caller explicitly opted in.
  """
  return bool(config is not None and config.enabled)


def configure(config: Optional[QuantizationObserverConfig]) -> None:
  """:param config: Active observer config for this process."""
  global _CONFIG
  _CONFIG = config


def _is_fp8(module) -> bool:
  """Detect a quantized linear, including torchao tensor-subclass weights.

  torchao keeps ``nn.Linear`` and swaps ``weight`` for a ``Float8Tensor``
  whose ``.dtype`` still reports the logical bf16 dtype, so the class name is
  the reliable signal here.

  :param module: Candidate module.
  :returns: True when the weight is stored in float8.
  """
  weight = getattr(module, "weight", None)
  if weight is None:
    return False
  if type(weight).__name__ == "Float8Tensor":
    return True
  return getattr(weight, "dtype", None) in (
    torch.float8_e4m3fn,
    torch.float8_e5m2,
  )


def capture_references(
  model: torch.nn.Module,
  config: Optional[QuantizationObserverConfig] = None,
) -> dict:
  """Snapshot bf16 weights of every linear layer before quantization.

  :param model: Module about to be quantized in place.
  :param config: Observer config; falls back to the process-wide one.
  :returns: Mapping ``fqn -> (weight_cpu, bias_cpu)``; empty when disabled.
  """
  cfg = config or _CONFIG
  if not is_enabled(cfg):
    return {}
  configure(cfg)
  refs = {}
  filtered = 0
  for name, module in model.named_modules():
    if not isinstance(module, torch.nn.Linear):
      continue
    if not cfg.matches(name):
      filtered += 1
      continue
    weight = getattr(module, "weight", None)
    if weight is None or weight.dtype != torch.bfloat16:
      continue
    bias = getattr(module, "bias", None)
    refs[name] = (
      weight.detach().to("cpu", copy=True),
      None if bias is None else bias.detach().to("cpu", copy=True),
    )
  if refs:
    logger.info(f"[observer] captured {len(refs)} bf16 linear references for "
                f"{model.__class__.__name__} ({filtered} filtered out by "
                f"layer_name_patterns={cfg.layer_name_patterns}, diag mode, "
                f"+1.5~2x forward cost on the observed layers)")
  return refs


def _accumulate(stats: dict, key: str, entry: dict, w_rel_err) -> None:
  row = stats.setdefault(
    key, {
      "n": 0,
      "sse": 0.0,
      "sref": 0.0,
      "dot": 0.0,
      "nyq": 0.0,
      "nyref": 0.0,
      "out_amax": 0.0,
      "in_outlier_sum": 0.0,
      "in_calls": 0,
      "w_rel_err": w_rel_err,
    })
  row["n"] += entry["n"]
  row["sse"] += entry["sse"]
  row["sref"] += entry["sref"]
  row["dot"] += entry["dot"]
  row["nyq"] += entry["nyq"]
  row["nyref"] += entry["nyref"]
  row["out_amax"] = max(row["out_amax"], entry["out_amax"])
  row["in_outlier_sum"] += entry["in_outlier"]
  row["in_calls"] += 1


def _make_hook(bucket: _Bucket, key: str, ref_weight, ref_bias, cfg):

  def hook(module, inputs, output):
    if key == bucket.sentinel and cfg.max_steps > 0:
      bucket.steps += 1
      if bucket.steps >= cfg.max_steps:
        dump()
    if not isinstance(output, torch.Tensor) or not isinstance(inputs[0], torch.Tensor):
      return None
    x = inputs[0].detach()
    # Attention / FFN linears see [B, S, D] and packed varlen inputs; both the
    # reference GEMM and the statistics work on any leading shape.
    weight = ref_weight
    if not cfg.keep_refs_on_gpu:
      weight = weight.to(x.device)
    bias = None if ref_bias is None else ref_bias.to(x.device)
    with torch.no_grad():
      y_ref = torch.nn.functional.linear(x.to(torch.bfloat16), weight, bias)
      y_out = output.detach().float()
      diff = y_out - y_ref.float()
      xf = x.float().reshape(-1, x.shape[-1])
      amax_tok = xf.abs().amax(dim=-1)
      mean_tok = xf.abs().mean(dim=-1).clamp_min(1e-8)
      _accumulate(
        bucket.stats, key, {
          "n": diff.numel(),
          "sse": float((diff * diff).sum()),
          "sref": float((y_ref.float() ** 2).sum()),
          "dot": float((y_out * y_ref.float()).sum()),
          "nyq": float((y_out ** 2).sum()),
          "nyref": float((y_ref.float() ** 2).sum()),
          "out_amax": float(y_ref.float().abs().max()),
          "in_outlier": float((amax_tok / mean_tok).mean()),
        }, None)
    return None

  return hook


def attach_observers(
  model: torch.nn.Module,
  refs: dict,
  config: Optional[QuantizationObserverConfig] = None,
) -> int:
  """Register a reference-recomputing forward hook on quantized linears.

  :param model: The already quantized module.
  :param refs: Output of :func:`capture_references`; empty disables the work.
  :param config: Observer config; falls back to the process-wide one.
  :returns: Number of hooks installed.
  """
  cfg = config or _CONFIG
  if not is_enabled(cfg) or not refs:
    return 0
  bucket = _Bucket(model=model.__class__.__name__, refs=refs)
  installed = 0
  for name, module in model.named_modules():
    if name not in refs or not _is_fp8(module):
      continue
    ref_w, ref_b = refs[name]
    if cfg.keep_refs_on_gpu:
      device = getattr(getattr(module, "weight", None), "device", None)
      device = device or torch.device("cuda")
      ref_w = ref_w.to(device)
      ref_b = None if ref_b is None else ref_b.to(device)
    _HOOKS.append(module.register_forward_hook(_make_hook(bucket, name, ref_w, ref_b, cfg)))
    bucket.sentinel = name
    installed += 1
    weight = module.weight
    w_rel = None
    with torch.no_grad():
      w_ref = refs[name][0].to(torch.float32)
      try:
        w_q = weight.dequantize().to(torch.float32).cpu()
        w_rel = float((w_q - w_ref).norm() / w_ref.norm().clamp_min(1e-12))
      except (RuntimeError, AttributeError):
        w_rel = None
    _accumulate(
      bucket.stats, name, {
        "n": 0,
        "sse": 0.0,
        "sref": 0.0,
        "dot": 0.0,
        "nyq": 0.0,
        "nyref": 0.0,
        "out_amax": 0.0,
        "in_outlier": 0.0,
      }, w_rel)
  _BUCKETS.append(bucket)
  logger.info(f"[observer] attached {installed} hooks on {model.__class__.__name__} "
              f"(diag mode, +1.5~2x forward cost)")
  return installed


def dump(path: Optional[str] = None) -> str:
  """Write the aggregated per-layer statistics as JSON.

  :param path: Target file; defaults to the config path, else
      ``./quant_observer.json``.
  :returns: Path actually written.
  """
  global _DUMPED
  target = (path or (getattr(_CONFIG, "json_path", None) or "./quant_observer.json"))
  rows = []
  for bucket in _BUCKETS:
    for name, row in bucket.stats.items():
      if row["sref"] <= 0 or row["n"] <= 0:
        continue
      denom = (row["nyq"] ** 0.5) * (row["nyref"] ** 0.5)
      rows.append({
        "model":
        bucket.model,
        "name":
        name,
        "rel_err": (row["sse"] / row["sref"]) ** 0.5,
        "cos":
        row["dot"] / denom if denom > 0 else 0.0,
        "out_amax":
        row["out_amax"],
        "in_outlier": (row["in_outlier_sum"] / row["in_calls"] if row["in_calls"] else 0.0),
        "w_rel_err":
        row["w_rel_err"],
        "n":
        row["n"],
      })
  rows.sort(key=lambda item: item["rel_err"], reverse=True)
  with open(target, "w", encoding="utf-8") as handle:
    json.dump(rows, handle, indent=2)
  _DUMPED = True
  logger.info(f"[observer] wrote {len(rows)} layer stats to {target}")
  return target


def _atexit_dump() -> None:
  if is_enabled(_CONFIG) and not _DUMPED and any(b.stats for b in _BUCKETS):
    dump()


atexit.register(_atexit_dump)
