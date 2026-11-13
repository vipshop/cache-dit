"""Operator-level benchmark for cache-dit and TorchAO NVFP4 quantize/GEMM paths."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from cache_dit.kernels import fp4_gemm
from cache_dit.kernels import fp4_quantize
from cache_dit.quantization.fp4.linear import NVFP4Linear

DEFAULT_SHAPES = (
  (4096, 3072, 3072, "double_attn"),
  (4096, 12288, 3072, "double_ff_up"),
  (4096, 3072, 12288, "double_ff_down"),
  (4608, 3072, 3072, "single_attn"),
  (4608, 3072, 15360, "single_proj_out"),
  (4608, 12288, 3072, "single_mlp"),
)


@dataclass(frozen=True)
class Shape:
  """One linear-layer shape in ``[tokens, out_features, in_features]`` form."""

  m: int
  n: int
  k: int
  name: str

  @property
  def label(self) -> str:
    """:returns: Compact shape label."""

    return f"{self.name}[{self.m}x{self.n}x{self.k}]"


def _parse_shapes(values: list[str] | None) -> list[Shape]:
  """Parse repeated ``MxNxK[:name]`` arguments.

  :param values: CLI shape strings.
  :returns: Parsed shapes, defaulting to six FLUX shapes.
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

  :returns: Version and device metadata.
  """

  try:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
  except (OSError, subprocess.CalledProcessError):
    commit = "unknown"
  versions = {}
  for package in ("torchao", "mslk"):
    try:
      versions[package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
      versions[package] = "not-installed"
  return {
    "commit": commit,
    "gpu": torch.cuda.get_device_name(),
    "capability": list(torch.cuda.get_device_capability()),
    "torch": torch.__version__,
    **versions,
    "python": sys.executable,
  }


def _time_cuda(fn, warmup: int, iters: int, rounds: int) -> dict:
  """Time a CUDA callable with paired events over multiple rounds.

  :param fn: Zero-argument callable.
  :param warmup: Initial warmup invocations.
  :param iters: Measured invocations per round.
  :param rounds: Number of measured rounds.
  :returns: Median of round medians and global min/max in milliseconds.
  """

  for _ in range(warmup):
    fn()
  torch.cuda.synchronize()
  round_medians = []
  all_samples = []
  for _ in range(rounds):
    samples = []
    for _ in range(iters):
      start = torch.cuda.Event(enable_timing=True)
      end = torch.cuda.Event(enable_timing=True)
      start.record()
      fn()
      end.record()
      end.synchronize()
      samples.append(start.elapsed_time(end))
    round_medians.append(statistics.median(samples))
    all_samples.extend(samples)
  return {
    "median_ms": statistics.median(round_medians),
    "min_ms": min(all_samples),
    "max_ms": max(all_samples),
    "round_medians_ms": round_medians,
  }


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
  """Return relative L2 error.

  :param actual: Test tensor.
  :param expected: Reference tensor.
  :returns: Relative L2 error.
  """

  return float(
    (actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12))


def _dequantize(qdata: torch.Tensor, scales: torch.Tensor,
                global_scales: torch.Tensor) -> torch.Tensor:
  """Decode FP4 data for a single-level GEMM reference.

  :param qdata: Packed FP4 tensor.
  :param scales: Canonical blocked scales.
  :param global_scales: Optional per-row global scales.
  :returns: Dequantized FP32 tensor.
  """

  from torchao.prototype.mx_formats.kernels import f4_unpacked_to_f32
  from torchao.prototype.mx_formats.kernels import unpack_uint4
  from torchao.prototype.mx_formats.utils import from_blocked

  rows, packed_cols = qdata.shape
  cols = packed_cols * 2
  values = f4_unpacked_to_f32(unpack_uint4(qdata.view(torch.uint8).contiguous())).view(rows, cols)
  scales = from_blocked(scales.flatten(), rows, cols // 16).view(rows, cols // 16).float()
  values = values * scales.repeat_interleave(16, dim=1)
  if global_scales.numel() != 0:
    values = values * global_scales[:, None]
  return values


def _torchao_api():
  """Import TorchAO's production NVFP4 primitives lazily.

  :returns: Production tensor class, kwargs and scale conversion helper.
  """

  from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
  from torchao.prototype.mx_formats.nvfp4_tensor import QuantizeTensorToNVFP4Kwargs
  from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale

  return NVFP4Tensor, QuantizeTensorToNVFP4Kwargs, per_tensor_amax_to_scale


def _torchao_weight(w: torch.Tensor, api):
  """Build the production dynamic-activation NVFP4 weight.

  :param w: BF16 weight ``[N, K]``.
  :param api: TorchAO API tuple.
  :returns: Packed NVFP4 weight.
  """

  tensor_type, kwargs_type, scale_fn = api
  act_kwargs = kwargs_type(
    use_dynamic_per_tensor_scale=True,
    use_triton_kernel=True,
    is_swizzled_scales=True,
  )
  weight = tensor_type.to_nvfp4(
    w,
    per_tensor_scale=scale_fn(torch.max(torch.abs(w))),
    act_per_tensor_scale=None,
    is_swizzled_scales=True,
    use_triton_kernel=False,
    act_quant_kwargs=act_kwargs,
  )
  weight.use_triton_kernel = True
  return weight


def _torchao_activation(x: torch.Tensor, api):
  """Quantize an activation with TorchAO's dynamic per-tensor recipe.

  :param x: BF16 activation.
  :param api: TorchAO API tuple.
  :returns: Packed NVFP4 activation.
  """

  tensor_type, _, scale_fn = api
  return tensor_type.to_nvfp4(
    x,
    per_tensor_scale=scale_fn(torch.max(torch.abs(x))),
    is_swizzled_scales=True,
    use_triton_kernel=True,
  )


def _compiled_native_linear(w: torch.Tensor, bias: torch.Tensor | None):
  """Build a compiled native NVFP4 linear.

  :param w: BF16 weight.
  :param bias: Optional BF16 bias.
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
  module = NVFP4Linear.from_linear(linear)
  return torch.compile(lambda x: module(x), fullgraph=False)


def _compiled_torchao_linear(weight, bias: torch.Tensor | None):
  """Build a compiled TorchAO NVFP4 linear.

  :param weight: Production packed NVFP4 weight.
  :param bias: Optional BF16 bias.
  :returns: Compiled callable.
  """

  return torch.compile(lambda x: F.linear(_torchao_activation(x, _TORCHAO_API), weight, bias),
                       fullgraph=False)


def _bench_one(shape: Shape, args, api) -> list[dict]:
  """Benchmark one shape.

  :param shape: Linear shape.
  :param args: CLI arguments.
  :param api: TorchAO API tuple.
  :returns: Result rows.
  """

  torch.manual_seed(args.seed)
  a = torch.randn(shape.m, shape.k, dtype=torch.bfloat16, device="cuda")
  w = torch.randn(shape.n, shape.k, dtype=torch.bfloat16, device="cuda")
  bias = torch.randn(shape.n, dtype=torch.bfloat16, device="cuda")
  bf16_reference = F.linear(a.float(), w.float(), bias.float())

  qa_single, sa_single, ga_single = fp4_quantize(a, level1=False)
  qw_single, sw_single, gw_single = fp4_quantize(w, level1=False)
  dequant_reference = F.linear(
    _dequantize(qa_single, sa_single, ga_single),
    _dequantize(qw_single, sw_single, gw_single),
    bias.float(),
  )
  qw, sw, gw = fp4_quantize(w)
  torchao_weight = _torchao_weight(w, api)
  rows = []

  def native_chain():
    qx, sx, gx = fp4_quantize(a)
    return fp4_gemm(qx, qw, sx, sw, gx, gw, bias=bias)

  def torchao_chain():
    return F.linear(_torchao_activation(a, api), torchao_weight, bias)

  arms = {
    "cache_dit_quantize":
    lambda: fp4_quantize(a),
    "torchao_quantize":
    lambda: _torchao_activation(a, api),
    "cache_dit_gemm_scaled_mm":
    lambda: fp4_gemm(qa_single, qw_single, sa_single, sw_single, ga_single, gw_single, bias=bias),
    "cache_dit_gemm_nonws":
    lambda: fp4_gemm(qa_single,
                     qw_single,
                     sa_single,
                     sw_single,
                     ga_single,
                     gw_single,
                     bias=bias,
                     use_ws=False,
                     use_native=True),
    "cache_dit_gemm_ws":
    lambda: fp4_gemm(qa_single,
                     qw_single,
                     sa_single,
                     sw_single,
                     ga_single,
                     gw_single,
                     bias=bias,
                     use_ws=True,
                     use_native=True),
    "torchao_gemm":
    lambda: torch._scaled_mm(
      qa_single, qw_single.t(), sa_single, sw_single, bias=bias, out_dtype=torch.bfloat16),
    "cache_dit_chain":
    native_chain,
    "torchao_chain":
    torchao_chain,
  }
  for arm, fn in arms.items():
    impl = "cache_dit" if arm.startswith("cache_dit") else "torchao"
    if impl not in args.impls:
      continue
    try:
      actual = fn()
      timing = _time_cuda(fn, args.warmup, args.iters, args.rounds)
      row = {
        "shape": shape.label,
        "m": shape.m,
        "n": shape.n,
        "k": shape.k,
        "impl": impl,
        "stage": arm.removeprefix(f"{impl}_"),
        **timing,
      }
      if "gemm" in arm:
        row["rel_l2_vs_dequant"] = _relative_l2(actual, dequant_reference.to(torch.bfloat16))
      if "chain" in arm:
        row["rel_l2_vs_bf16"] = _relative_l2(actual, bf16_reference.to(torch.bfloat16))
      rows.append(row)
    except Exception as exc:
      rows.append({
        "shape": shape.label,
        "m": shape.m,
        "n": shape.n,
        "k": shape.k,
        "impl": impl,
        "stage": arm.removeprefix(f"{impl}_"),
        "error": f"{type(exc).__name__}: {exc}",
      })

  if args.skip_compiled:
    return rows
  for impl in args.impls:
    try:
      if impl == "cache_dit":
        fn = _compiled_native_linear(w, bias)
      else:
        fn = _compiled_torchao_linear(torchao_weight, bias)
      actual = fn(a)
      torch.cuda.synchronize()
      timing = _time_cuda(lambda: fn(a), args.warmup, args.iters, args.rounds)
      rows.append({
        "shape":
        shape.label,
        "m":
        shape.m,
        "n":
        shape.n,
        "k":
        shape.k,
        "impl":
        impl,
        "stage":
        "compile_linear",
        **timing,
        "rel_l2_vs_bf16":
        _relative_l2(actual, bf16_reference.to(torch.bfloat16)),
      })
    except Exception as exc:
      rows.append({
        "shape": shape.label,
        "m": shape.m,
        "n": shape.n,
        "k": shape.k,
        "impl": impl,
        "stage": "compile_linear",
        "error": f"{type(exc).__name__}: {exc}",
      })
  return rows


def _print_table(rows: list[dict]) -> None:
  """Print a compact terminal table.

  :param rows: Benchmark rows.
  """

  print(f"{'shape':34s} {'impl':10s} {'stage':18s} {'median_ms':>10s} {'min_ms':>10s} "
        f"{'max_ms':>10s} {'rel_l2':>9s}")
  for row in rows:
    if "error" in row:
      error = row["error"].splitlines()[0][:160]
      print(f"{row['shape']:34s} {row['impl']:10s} {row['stage']:18s} ERROR {error}")
      continue
    rel_l2 = row.get("rel_l2_vs_dequant", row.get("rel_l2_vs_bf16"))
    rel_text = "" if rel_l2 is None else f"{rel_l2:.3e}"
    print(f"{row['shape']:34s} {row['impl']:10s} {row['stage']:18s} "
          f"{row['median_ms']:10.4f} {row['min_ms']:10.4f} {row['max_ms']:10.4f} {rel_text:>9s}")


def main() -> None:
  """Run the NVFP4 operator benchmark."""

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--shapes",
                      nargs="*",
                      default=None,
                      help="MxNxK[:name] entries; defaults to six FLUX shapes.")
  parser.add_argument("--impls",
                      nargs="+",
                      default=["cache_dit", "torchao"],
                      choices=["cache_dit", "torchao"])
  parser.add_argument("--warmup", type=int, default=10)
  parser.add_argument("--iters", type=int, default=50)
  parser.add_argument("--rounds", type=int, default=3)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--skip-compiled", action="store_true")
  parser.add_argument("--output", type=Path, default=Path(".tmp/native_nvfp4/bench/bench_fp4.json"))
  args = parser.parse_args()
  if not torch.cuda.is_available():
    raise RuntimeError("bench_fp4.py requires CUDA.")
  if torch.cuda.get_device_capability()[0] != 12:
    raise RuntimeError("bench_fp4.py requires sm120.")
  torch.backends.cuda.matmul.allow_tf32 = False

  global _TORCHAO_API
  _TORCHAO_API = _torchao_api()
  rows = []
  for shape in _parse_shapes(args.shapes):
    rows.extend(_bench_one(shape, args, _TORCHAO_API))
  _print_table(rows)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  payload = {
    "metadata": _metadata(),
    "args": vars(args) | {
      "output": str(args.output)
    },
    "rows": rows,
  }
  args.output.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
  print(f"\nWrote {args.output}")


if __name__ == "__main__":
  main()
