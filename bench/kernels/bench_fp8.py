"""Operator-level benchmark for cache-dit and TorchAO FP8 quantize/GEMM paths."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from cache_dit.kernels import fp8_gemm
from cache_dit.kernels import fp8_quantize
from cache_dit.kernels import fp8_quantize_gemm
from cache_dit.kernels.fp8 import fp8_quantize_torch
from cache_dit.quantization.fp8.linear import QW8A8Linear

MODES = ("per_row", "per_tensor", "per_block")

DEFAULT_SHAPES = (
  (4096, 3072, 3072, "double_attn"),
  (4096, 12288, 3072, "double_ff_up"),
  (4096, 3072, 12288, "double_ff_down"),
  (4608, 3072, 3072, "single_attn"),
  (4608, 3072, 15360, "single_proj_out"),
  (4608, 12288, 3072, "single_mlp"),
)

SWEEP_SHAPES = (
  (128, 3072, 3072, "m128"),
  (512, 3072, 3072, "m512"),
  (1024, 3072, 3072, "m1024"),
  (2048, 3072, 3072, "m2048"),
  (4096, 3072, 3072, "m4096"),
  (4608, 3072, 3072, "m4608"),
  (8192, 3072, 3072, "m8192"),
  (4096, 12288, 3072, "ff_up"),
  (4096, 3072, 12288, "ff_down"),
  (4608, 12288, 3072, "mlp"),
  (4608, 3072, 15360, "proj_out"),
  (4096, 15360, 3072, "wide_out"),
  (4096, 3072, 15360, "wide_in"),
  (4032, 3072, 3072, "m_tail"),
  (4096, 3200, 3072, "n_tail"),
  (4096, 3072, 3328, "k_blocks"),
  (1024, 4096, 4096, "square1k"),
  (4096, 4096, 4096, "square4k"),
  (8192, 8192, 8192, "square8k"),
)


@dataclass(frozen=True)
class Shape:
  """One linear-layer shape in FLUX ``[tokens, out_features, in_features]`` form."""

  m: int
  n: int
  k: int
  name: str

  @property
  def label(self) -> str:
    """:returns: Compact shape label."""

    return f"{self.name}[{self.m}x{self.n}x{self.k}]"


def _parse_shapes(values: list[str] | None) -> list[Shape]:
  """Parse repeated ``MxNxK[:name]`` arguments, falling back to FLUX shapes.

  :param values: CLI shape strings.
  :returns: Parsed shapes.
  """

  if not values:
    return [Shape(*shape) for shape in DEFAULT_SHAPES]
  shapes = []
  for value in values:
    spec, _, name = value.partition(":")
    m, n, k = (int(part) for part in spec.lower().split("x"))
    shapes.append(Shape(m, n, k, name or "custom"))
  return shapes


def _metadata() -> dict:
  """Capture reproducible environment metadata.

  :returns: Benchmark metadata.
  """

  try:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
  except (OSError, subprocess.CalledProcessError):
    commit = "unknown"
  try:
    torchao_version = importlib.metadata.version("torchao")
  except importlib.metadata.PackageNotFoundError:
    torchao_version = "not-installed"
  return {
    "commit": commit,
    "gpu": torch.cuda.get_device_name(),
    "capability": list(torch.cuda.get_device_capability()),
    "torch": torch.__version__,
    "torchao": torchao_version,
    "triton_default_fp_fusion": os.environ.get("TRITON_DEFAULT_FP_FUSION", "unset"),
    "python": subprocess.check_output(["python", "--version"], text=True).strip(),
  }


def _time_cuda(fn, warmup: int, iters: int) -> dict:
  """Time a CUDA callable with paired CUDA events.

  :param fn: Zero-argument callable.
  :param warmup: Warmup invocations.
  :param iters: Measured invocations.
  :returns: Median/min/max milliseconds.
  """

  for _ in range(warmup):
    fn()
  torch.cuda.synchronize()
  samples = []
  for _ in range(iters):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    samples.append(start.elapsed_time(end))
  return {
    "median_ms": statistics.median(samples),
    "min_ms": min(samples),
    "max_ms": max(samples),
  }


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
  """Return relative L2 error.

  :param actual: Test tensor.
  :param expected: Reference tensor.
  :returns: Relative L2 error.
  """

  return float(
    (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12))


def _expand_scale(q: torch.Tensor, scale: torch.Tensor, mode: str, is_weight: bool) -> torch.Tensor:
  """Expand a scale table to the quantized tensor's shape.

  :param q: Quantized tensor.
  :param scale: Scale table.
  :param mode: FP8 granularity.
  :param is_weight: Whether ``q`` is the ``[N, K]`` weight.
  :returns: Dense FP32 scale tensor.
  """

  if mode == "per_row":
    return scale.reshape(q.shape[0], 1)
  if mode == "per_tensor":
    return scale.reshape(1, 1)
  if is_weight:
    expanded = scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
  else:
    expanded = scale.repeat_interleave(128, dim=1)
  return expanded[:q.shape[0], :q.shape[1]]


def _reference_gemm(qa: torch.Tensor, sa: torch.Tensor, qw: torch.Tensor, sw: torch.Tensor,
                    mode: str, bias: torch.Tensor | None) -> torch.Tensor:
  """Compute a float32 reference from dequantized operands.

  :param qa: Quantized activation.
  :param sa: Activation scale.
  :param qw: Quantized weight.
  :param sw: Weight scale.
  :param mode: FP8 granularity.
  :param bias: Optional BF16 bias.
  :returns: Reference output.
  """

  a = qa.float() * _expand_scale(qa, sa, mode, False)
  w = qw.float() * _expand_scale(qw, sw, mode, True)
  return F.linear(a, w, None if bias is None else bias.float())


def _torchao_api():
  """Import TorchAO's production FP8 primitives lazily.

  :returns: Tuple of Float8Tensor, granularity classes, KernelPreference and blockwise GEMM.
  """

  from torchao.kernel.blockwise_quantization import blockwise_fp8_gemm
  from torchao.quantization import PerBlock
  from torchao.quantization import PerRow
  from torchao.quantization import PerTensor
  from torchao.quantization.quantize_.common import KernelPreference
  from torchao.quantization.quantize_.workflows.float8.float8_tensor import Float8Tensor

  return (Float8Tensor, {
    "per_row": PerRow,
    "per_tensor": PerTensor,
    "per_block": PerBlock
  }, KernelPreference, blockwise_fp8_gemm)


def _torchao_granularity(mode: str, is_weight: bool, api) -> object:
  """Construct the production granularity object.

  :param mode: FP8 granularity.
  :param is_weight: Whether the tensor is a linear weight.
  :param api: Result of :func:`_torchao_api`.
  :returns: TorchAO granularity instance.
  """

  _, granularities, _, _ = api
  if mode == "per_block":
    return granularities[mode]([128, 128] if is_weight else [1, 128])
  return granularities[mode]()


def _torchao_quantize(x: torch.Tensor, mode: str, is_weight: bool, api) -> tuple:
  """Run TorchAO's production float8 quantization.

  :param x: BF16 tensor.
  :param mode: FP8 granularity.
  :param is_weight: Whether ``x`` is a ``[N, K]`` weight.
  :param api: Result of :func:`_torchao_api`.
  :returns: ``(qdata, scale)``.
  """

  float8_tensor, _, kernel_preference, _ = api
  granularity = _torchao_granularity(mode, is_weight, api)
  packed = float8_tensor.from_hp(x,
                                 granularity=granularity,
                                 kernel_preference=kernel_preference.TORCH)
  return packed.qdata, packed.scale


def _torchao_gemm(qa: torch.Tensor, sa: torch.Tensor, qw: torch.Tensor, sw: torch.Tensor, mode: str,
                  bias: torch.Tensor | None, api) -> torch.Tensor:
  """Call the GEMM kernel TorchAO uses in production for the given granularity.

  :param qa: Quantized activation.
  :param sa: Activation scale.
  :param qw: Quantized weight ``[N, K]``.
  :param sw: Weight scale.
  :param mode: FP8 granularity.
  :param bias: Optional BF16 bias.
  :param api: Result of :func:`_torchao_api`.
  :returns: BF16 output.
  """

  _, _, _, blockwise_fp8_gemm = api
  if mode == "per_block":
    out = blockwise_fp8_gemm(qa, sa, qw, sw, block_size=128)
    return out if bias is None else out + bias
  if mode == "per_row":
    sa = sa.reshape(-1, 1)
    sw = sw.reshape(1, -1)
  else:
    sa = sa.reshape(1)
    sw = sw.reshape(1)
  return torch._scaled_mm(qa, qw.t(), sa, sw, bias=bias, out_dtype=torch.bfloat16)


def _compiled_cache_linear(w: torch.Tensor, bias: torch.Tensor | None, mode: str):
  """Build a compiled cache-dit FP8 linear.

  :param w: BF16 weight ``[N, K]``.
  :param bias: Optional BF16 bias.
  :param mode: FP8 granularity.
  :returns: Compiled callable.
  """

  linear = torch.nn.Linear(w.shape[1],
                           w.shape[0],
                           bias=bias is not None,
                           device=w.device,
                           dtype=torch.bfloat16)
  with torch.no_grad():
    linear.weight.copy_(w)
    if bias is not None:
      linear.bias.copy_(bias)
  cache_module = QW8A8Linear.from_linear(linear, mode)
  return torch.compile(lambda x: cache_module(x), fullgraph=False)


def _compiled_torchao_linear(w: torch.Tensor, bias: torch.Tensor | None, mode: str, api):
  """Build a compiled TorchAO dynamic-activation FP8 linear.

  :param w: BF16 weight ``[N, K]``.
  :param bias: Optional BF16 bias.
  :param mode: FP8 granularity.
  :param api: Result of :func:`_torchao_api`.
  :returns: Compiled callable.
  """

  from torchao.float8.inference import Float8MMConfig

  float8_tensor, _, kernel_preference, _ = api
  act_gran = _torchao_granularity(mode, False, api)
  weight_gran = _torchao_granularity(mode, True, api)
  # Match the production TorchAO recipe and its compiled fake-tensor path.
  mm_config = Float8MMConfig(use_fast_accum=mode != "per_block")
  packed_weight = float8_tensor.from_hp(w,
                                        granularity=weight_gran,
                                        mm_config=mm_config,
                                        kernel_preference=kernel_preference.TORCH)

  def torchao_linear(x):
    packed_act = float8_tensor.from_hp(x,
                                       granularity=act_gran,
                                       mm_config=mm_config,
                                       kernel_preference=kernel_preference.TORCH)
    return F.linear(packed_act, packed_weight, bias)

  return torch.compile(torchao_linear, fullgraph=False)


def _bench_one(shape: Shape, mode: str, args, api) -> list[dict]:
  """Benchmark one shape and one granularity.

  :param shape: Linear shape.
  :param mode: FP8 granularity.
  :param args: CLI arguments.
  :param api: TorchAO API tuple.
  :returns: Result rows.
  """

  torch.manual_seed(args.seed)
  a = torch.randn(shape.m, shape.k, dtype=torch.bfloat16, device="cuda")
  w = torch.randn(shape.n, shape.k, dtype=torch.bfloat16, device="cuda")
  bias = torch.randn(shape.n, dtype=torch.bfloat16, device="cuda")
  qa, sa = fp8_quantize(a, mode)
  qw, sw = fp8_quantize(w, mode, is_weight=True)
  reference = _reference_gemm(qa, sa, qw, sw, mode, bias)
  rows = []

  def cache_chain(use_ws: bool) -> torch.Tensor:
    if mode == "per_row":
      return fp8_quantize_gemm(a, qw, sw, bias=bias, use_ws=use_ws)
    qa2, sa2 = fp8_quantize(a, mode)
    return fp8_gemm(qa2, qw, sa2, sw, bias=bias, mode=mode, use_ws=use_ws)

  def torchao_chain() -> torch.Tensor:
    qa2, sa2 = _torchao_quantize(a, mode, False, api)
    return _torchao_gemm(qa2, sa2, qw, sw, mode, bias, api)

  arms = {
    "cache_dit_quantize": lambda: fp8_quantize(a, mode),
    "torchao_quantize": lambda: _torchao_quantize(a, mode, False, api),
    "cache_dit_gemm_ws": lambda: fp8_gemm(qa, qw, sa, sw, bias=bias, mode=mode, use_ws=True),
    "cache_dit_gemm_nonws": lambda: fp8_gemm(qa, qw, sa, sw, bias=bias, mode=mode, use_ws=False),
    "torchao_gemm": lambda: _torchao_gemm(qa, sa, qw, sw, mode, bias, api),
    "cache_dit_chain_ws": lambda: cache_chain(True),
    "cache_dit_chain_nonws": lambda: cache_chain(False),
    "torchao_chain": torchao_chain,
  }
  if not args.skip_compiled:
    arms["cache_dit_torch_compile_quantize"] = torch.compile(lambda: fp8_quantize_torch(a, mode),
                                                             fullgraph=True)
  for arm, fn in arms.items():
    if arm.startswith("cache_dit") and "cache_dit" not in args.impls:
      continue
    if arm.startswith("torchao") and "torchao" not in args.impls:
      continue
    try:
      actual = fn()
      if isinstance(actual, tuple):
        actual = actual[0]
      timing = _time_cuda(fn, args.warmup, args.iters)
      impl = "cache_dit" if arm.startswith("cache_dit") else "torchao"
      row = {
        "shape": shape.label,
        "m": shape.m,
        "n": shape.n,
        "k": shape.k,
        "mode": mode,
        "impl": impl,
        "stage": arm.removeprefix(f"{impl}_"),
        "median_ms": timing["median_ms"],
        "min_ms": timing["min_ms"],
        "max_ms": timing["max_ms"],
      }
      if "gemm" in arm:
        row["rel_l2_vs_dequant"] = _relative_l2(actual, reference.to(torch.bfloat16))
        row["max_abs_err"] = float((actual.float() - reference.float()).abs().max())
      rows.append(row)
    except Exception as exc:
      impl = "cache_dit" if arm.startswith("cache_dit") else "torchao"
      rows.append({
        "shape": shape.label,
        "m": shape.m,
        "n": shape.n,
        "k": shape.k,
        "mode": mode,
        "impl": impl,
        "stage": arm.removeprefix(f"{impl}_"),
        "error": f"{type(exc).__name__}: {exc}",
      })

  if args.skip_compiled:
    return rows
  for impl in args.impls:
    try:
      if impl == "cache_dit":
        fn = _compiled_cache_linear(w, bias, mode)
      else:
        fn = _compiled_torchao_linear(w, bias, mode, api)
      out = fn(a)
      torch.cuda.synchronize()
      timing = _time_cuda(lambda: fn(a), args.warmup, args.iters)
      rows.append({
        "shape": shape.label,
        "m": shape.m,
        "n": shape.n,
        "k": shape.k,
        "mode": mode,
        "impl": impl,
        "stage": "compile_linear",
        "median_ms": timing["median_ms"],
        "min_ms": timing["min_ms"],
        "max_ms": timing["max_ms"],
        "rel_l2_vs_dequant": _relative_l2(out, reference.to(torch.bfloat16)),
        "max_abs_err": float((out.float() - reference.float()).abs().max()),
      })
    except Exception as exc:
      rows.append({
        "shape": shape.label,
        "m": shape.m,
        "n": shape.n,
        "k": shape.k,
        "mode": mode,
        "impl": impl,
        "stage": "compile_linear",
        "error": f"{type(exc).__name__}: {exc}",
      })
  return rows


def _print_table(rows: list[dict]) -> None:
  """Print a compact terminal table.

  :param rows: Benchmark rows.
  """

  print(f"{'shape':34s} {'mode':10s} {'impl':10s} {'stage':18s} "
        f"{'median_ms':>10s} {'min_ms':>10s} {'max_ms':>10s} {'rel_l2':>9s}")
  for row in rows:
    if "error" in row:
      error = row["error"].splitlines()[0][:160]
      print(f"{row['shape']:34s} {row['mode']:10s} {row['impl']:10s} "
            f"{row['stage']:18s} ERROR {error}")
      continue
    rel_l2 = row.get("rel_l2_vs_dequant")
    rel_text = "" if rel_l2 is None else f"{rel_l2:.3e}"
    print(f"{row['shape']:34s} {row['mode']:10s} {row['impl']:10s} "
          f"{row['stage']:18s} {row['median_ms']:10.4f} {row['min_ms']:10.4f} "
          f"{row['max_ms']:10.4f} {rel_text:>9s}")


def main() -> None:
  """Run the FP8 operator benchmark."""

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--shapes",
                      nargs="*",
                      default=None,
                      help="MxNxK[:name] entries; overrides --suite.")
  parser.add_argument("--suite",
                      choices=["flux", "sweep"],
                      default="flux",
                      help="flux: six production shapes; sweep: 16-shape matrix.")
  parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
  parser.add_argument("--impls",
                      nargs="+",
                      default=["cache_dit", "torchao"],
                      choices=["cache_dit", "torchao"])
  parser.add_argument("--warmup", type=int, default=10)
  parser.add_argument("--iters", type=int, default=50)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--skip-compiled", action="store_true")
  parser.add_argument("--output", type=Path, default=Path(".tmp/fp8_backend/bench_fp8.json"))
  args = parser.parse_args()

  if not torch.cuda.is_available():
    raise RuntimeError("bench_fp8.py requires CUDA.")
  torch.backends.cuda.matmul.allow_tf32 = False
  api = _torchao_api()
  shapes = (_parse_shapes(args.shapes) if args.shapes else [
    Shape(*shape) for shape in (SWEEP_SHAPES if args.suite == "sweep" else DEFAULT_SHAPES)
  ])
  rows = []
  for shape in shapes:
    for mode in args.modes:
      rows.extend(_bench_one(shape, mode, args, api))

  _print_table(rows)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  payload = {
    "metadata": _metadata(),
    "args": vars(args) | {
      "output": str(args.output)
    },
    "rows": rows
  }
  args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
  print(f"\nWrote {args.output}")


if __name__ == "__main__":
  main()
