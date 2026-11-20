"""Operator-level benchmark for cache-dit's native MXFP4 weight-only path.

Covers the two things the backend actually costs:
  * the one-time weight packing (where the device choice matters, ~250-300x
    between CPU and an accelerator),
  * the steady-state dequant-then-GEMM forward (eager and under ``torch.compile``).

Usage::

  python3 -m bench.kernels.bench_mxfp4 --repeat 5
  python3 -m bench.kernels.bench_mxfp4 --shapes 4096,4096 12288,4096 --compile
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass

import torch

from cache_dit.quantization.mxfp4 import MXFP4Linear, mxfp4_is_available
from cache_dit.quantization.mxfp4.tensor import quantize_mxfp4_weight

DEFAULT_SHAPES = ((3072, 3072), (4096, 4096), (12288, 4096), (12288, 3072))


@dataclass(frozen=True)
class Shape:
  """One linear weight in ``[out_features, in_features]`` form.

  :param n: Output features.
  :param k: Input features (must be a multiple of 32).
  """

  n: int
  k: int

  @property
  def label(self) -> str:
    return f"{self.n}x{self.k}"


def _timeit(fn, repeat: int, warmup: int = 1) -> float:
  for _ in range(warmup):
    fn()
  if torch.cuda.is_available():
    torch.cuda.synchronize()
  samples = []
  for _ in range(repeat):
    start = time.perf_counter()
    fn()
    if torch.cuda.is_available():
      torch.cuda.synchronize()
    samples.append((time.perf_counter() - start) * 1e3)
  return statistics.median(samples)


def bench_pack(shape: Shape, repeat: int) -> dict:
  weight = torch.randn(shape.n, shape.k, dtype=torch.bfloat16)
  result = {"cpu_ms": _timeit(lambda: quantize_mxfp4_weight(weight), repeat)}
  if torch.cuda.is_available():
    accel = weight.cuda()
    result["accelerator_ms"] = _timeit(lambda: quantize_mxfp4_weight(accel), repeat)
    result["h2d_pack_ms"] = _timeit(lambda: quantize_mxfp4_weight(weight.cuda()), repeat)
    result["speedup"] = result["cpu_ms"] / max(result["accelerator_ms"], 1e-6)
  return result


def bench_forward(shape: Shape, tokens: int, repeat: int, use_compile: bool) -> dict:
  if not torch.cuda.is_available():
    return {}
  linear = torch.nn.Linear(shape.k, shape.n, bias=True, dtype=torch.bfloat16).cuda()
  layer = MXFP4Linear(linear).cuda()
  x = torch.randn(tokens, shape.k, device="cuda", dtype=torch.bfloat16)
  module = torch.compile(layer, dynamic=False) if use_compile else layer
  with torch.no_grad():
    ms = _timeit(lambda: module(x), repeat)
  return {"tokens": tokens, "compile": use_compile, "ms": ms}


def main(argv=None) -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--shapes",
                      nargs="*",
                      default=[f"{n},{k}" for n, k in DEFAULT_SHAPES],
                      help="Weight shapes as out,in (default: FLUX production shapes).")
  parser.add_argument("--tokens", type=int, default=4096)
  parser.add_argument("--repeat", type=int, default=5)
  parser.add_argument("--compile", action="store_true", help="Also time torch.compile.")
  parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
  args = parser.parse_args(argv)

  if not mxfp4_is_available():
    print("MXFP4 requires torch.float4_e2m1fn_x2 + torch.float8_e8m0fnu "
          "(PyTorch >= 2.6).",
          file=sys.stderr)
    return 1

  shapes = [Shape(*(int(part) for part in spec.split(","))) for spec in args.shapes]
  report = {
    "torch": torch.__version__,
    "cuda": torch.cuda.is_available(),
    "device": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    "pack": {},
    "forward": {},
  }
  for shape in shapes:
    report["pack"][shape.label] = bench_pack(shape, args.repeat)

  if torch.cuda.is_available():
    for shape in shapes:
      report["forward"][shape.label] = bench_forward(shape, args.tokens, args.repeat, False)
      if args.compile:
        report["forward"][shape.label + "_compile"] = bench_forward(shape, args.tokens, args.repeat,
                                                                    True)

  if args.json:
    print(json.dumps(report, indent=2))
    return 0

  print(f"torch={report['torch']} device={report['device']}")
  header = f"{'shape':>14s} {'cpu pack':>10s} {'accel pack':>11s} {'speedup':>8s}"
  print(header)
  for label, row in report["pack"].items():
    accel = f"{row['accelerator_ms']:.2f}" if "accelerator_ms" in row else "n/a"
    speedup = f"{row['speedup']:.0f}x" if "speedup" in row else "n/a"
    print(f"{label:>14s} {row['cpu_ms']:10.2f} {accel:>11s} {speedup:>8s}")
  if report["forward"]:
    print(f"\n{'shape':>14s} {'eager ms':>10s} {'compile ms':>11s}")
    for label, row in report["forward"].items():
      if label.endswith("_compile"):
        continue
      compiled = report["forward"].get(label + "_compile", {}).get("ms")
      compiled_text = f"{compiled:11.2f}" if compiled is not None else f"{'n/a':>11s}"
      print(f"{label:>14s} {row['ms']:10.2f} {compiled_text}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
