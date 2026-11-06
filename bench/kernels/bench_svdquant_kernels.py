#!/usr/bin/env python3
"""Compare SVDQuant kernels on identical shapes and rank settings.

The benchmark reports three timings per runtime. `quantize` is the activation quantizer plus
LoRA-down reduction (shared by both runtimes), `native`/`cute` are *kernel-only* GEMM timings
(each backend's pack is prepared once and only the GEMM is timed), and `native+e2e`/`cute+e2e`
measure one full `SVDQW4A4Linear.forward`, i.e. quantize + GEMM. `speedup_vs_native` compares
against the matching native baseline, so kernel rows use the kernel-only baseline and e2e rows use
the e2e baseline. FlashInfer activation quantization uses its own standard 128x4 pack; native/cute
use cache-dit's v1 activation pack.

Examples:

  # All installed backends on the default shapes at rank 32
  python bench/kernels/bench_svdquant_kernels.py

  # Selected shapes/ranks, local FlashInfer source tree
  python bench/kernels/bench_svdquant_kernels.py \
    --flashinfer-path /workspace/dev/vipshop/flashinfer \
    --shape 4096x3072x3072 --shape 16384x8192x8192 \
    --rank 32 --rank 64 --iterations 50

  # Only native and FlashInfer cutlass-sm120
  python bench/kernels/bench_svdquant_kernels.py \
    --impl native,flashinfer+cutlass-sm120

The FlashInfer kernel-only timings use its standard weight/activation pack. They are not
numerically interchangeable with cache-dit's v1 activation pack; `finite` is reported for
FlashInfer, while `rel_l2_vs_native` is only meaningful for cache-dit's cute kernel.

The FlashInfer implementations are optional. When FlashInfer cannot be imported, those rows are
reported as skipped and the cache-dit native/cute measurements still run to completion.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass

import torch

from cache_dit.quantization.svdquant.linear import SVDQW4A4Linear
from cache_dit.quantization.svdquant.packing import convert_svdq_nvfp4_pack_to_cute
from cache_dit.quantization.svdquant.quantizer import quantize_linear_svdq_w4a4

DEFAULT_SHAPES = (
  (4096, 3072, 3072),
  (8192, 3072, 3072),
  (4096, 6144, 3072),
  (8192, 6144, 3072),
  (16384, 8192, 8192),
)
DEFAULT_IMPLS = (
  "quantize",
  "native",
  "cute",
  "native+e2e",
  "cute+e2e",
  "flashinfer+cute-dsl",
  "flashinfer+cutlass-sm120",
)
FLASHINFER_RANKS = {
  "flashinfer+cute-dsl": {32, 64, 128},
  "flashinfer+cutlass-sm120": {32, 64},
}


@dataclass
class Case:
  x: torch.Tensor
  native_module: SVDQW4A4Linear
  cute_module: SVDQW4A4Linear
  qact_native: torch.Tensor
  a_sf_native: torch.Tensor
  lora_act_native: torch.Tensor
  qact_cute: torch.Tensor
  a_sf_cute: torch.Tensor
  lora_act_cute: torch.Tensor
  qweight_std: torch.Tensor
  wscales_std: torch.Tensor
  wscales_flash: torch.Tensor
  qact_flash: torch.Tensor
  a_sf_flash: torch.Tensor
  lora_d: torch.Tensor
  lora_l1: torch.Tensor
  lora_l1_flash: torch.Tensor
  bias: torch.Tensor | None
  wcscales: torch.Tensor | None
  smooth: torch.Tensor
  alpha_flash: torch.Tensor
  out_flash: torch.Tensor


def bench(fn, warmup: int, iterations: int) -> float:
  for _ in range(warmup):
    fn()
  torch.cuda.synchronize()
  samples = []
  for _ in range(iterations):
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    fn()
    end.record()
    end.synchronize()
    samples.append(begin.elapsed_time(end) * 1000.0)
  return statistics.median(samples)


def _load_flashinfer(flashinfer_path: str | None):
  if flashinfer_path:
    sys.path.insert(0, flashinfer_path)
  try:
    from flashinfer.gemm.gemm_svdquant import mm_nvfp4_svdquant, nvfp4_quantize_smooth
  except Exception as exc:
    raise RuntimeError(
      "FlashInfer is not importable. Install flashinfer-python or pass --flashinfer-path.") from exc
  return mm_nvfp4_svdquant, nvfp4_quantize_smooth


def prepare_case(m: int,
                 n: int,
                 k: int,
                 rank: int,
                 seed: int,
                 device: torch.device,
                 quantize_smooth=None) -> Case:
  torch.manual_seed(seed)
  x = torch.randn(m, k, dtype=torch.bfloat16, device=device)
  linear = torch.nn.Linear(k, n, bias=True).to(device=device, dtype=torch.bfloat16)
  state = quantize_linear_svdq_w4a4(
    linear,
    [x[:min(256, m)]],
    rank=rank,
    precision="nvfp4",
    device=device,
    torch_dtype=torch.bfloat16,
    return_state_dict=True,
  )
  qweight_std, wscales_std, proj_up_std, proj_down_std, bias, wcscales, smooth = (
    convert_svdq_nvfp4_pack_to_cute(
      state["qweight"],
      state["wscales"],
      state["proj_up"],
      state["proj_down"],
      state.get("bias"),
      state.get("wcscales"),
      state["smooth_factor"],
    ))

  native_module = SVDQW4A4Linear(k,
                                 n,
                                 rank=rank,
                                 precision="nvfp4",
                                 runtime_kernel="native",
                                 device=device,
                                 torch_dtype=torch.bfloat16).eval()
  cute_module = SVDQW4A4Linear(k,
                               n,
                               rank=rank,
                               precision="nvfp4",
                               runtime_kernel="cute",
                               device=device,
                               torch_dtype=torch.bfloat16).eval()
  native_module.load_state_dict(state)
  cute_module.load_state_dict(state)
  native_module.requires_grad_(False)
  cute_module.requires_grad_(False)

  qact_native, a_sf_native, lora_act_native = native_module.quantize(x)
  qact_cute, a_sf_cute, lora_act_cute = cute_module.quantize(x)
  qweight_std_u8 = qweight_std.view(torch.uint8).contiguous()
  wscales_std = wscales_std.contiguous()
  wscales_std_u8 = wscales_std.view(torch.uint8).flatten().contiguous()

  # FlashInfer's activation pack and alpha convention. This is only used for its own kernel.
  if quantize_smooth is not None:
    global_scale = ((448.0 * 6.0) / (x.float() * smooth.float().unsqueeze(0)).abs().max()).to(
      torch.float32)
    qact_flash, a_sf_flash = quantize_smooth(x, smooth, global_scale, backend="cute-dsl")
    alpha_scalar = 1.0 / float(global_scale.item())
    qact_flash = qact_flash.view(torch.uint8).contiguous()
    a_sf_flash = a_sf_flash.view(torch.uint8).flatten().contiguous()
  else:
    alpha_scalar = 1.0
    qact_flash = torch.empty(0, dtype=torch.uint8, device=device)
    a_sf_flash = torch.empty(0, dtype=torch.uint8, device=device)
  alpha_flash = torch.tensor([alpha_scalar], dtype=torch.float32, device=device)

  # Native emits an FP32 LoRA-down scratch; FlashInfer expects BF16 D/L1.
  lora_d = lora_act_native[:m].to(torch.bfloat16).contiguous()
  lora_l1 = proj_up_std.to(torch.bfloat16).contiguous()
  del proj_down_std
  return Case(
    x=x,
    native_module=native_module,
    cute_module=cute_module,
    qact_native=qact_native,
    a_sf_native=a_sf_native,
    lora_act_native=lora_act_native,
    qact_cute=qact_cute,
    a_sf_cute=a_sf_cute,
    lora_act_cute=lora_act_cute,
    qweight_std=qweight_std_u8,
    wscales_std=wscales_std,
    wscales_flash=wscales_std_u8,
    qact_flash=qact_flash,
    a_sf_flash=a_sf_flash,
    lora_d=lora_d,
    lora_l1=lora_l1,
    lora_l1_flash=(lora_l1 / alpha_scalar).contiguous(),
    bias=bias,
    wcscales=wcscales,
    smooth=smooth,
    alpha_flash=alpha_flash,
    out_flash=torch.empty(m, n, dtype=torch.bfloat16, device=device),
  )


def _native_call(case: Case):
  return case.native_module.forward_quant(case.qact_native, case.a_sf_native, case.lora_act_native)


def _cute_call(case: Case):
  return case.cute_module.forward_quant(case.qact_cute, case.a_sf_cute, case.lora_act_cute)


def _flashinfer_call(case: Case, mm_nvfp4_svdquant, backend: str):
  return mm_nvfp4_svdquant(
    case.qact_flash,
    case.qweight_std,
    case.a_sf_flash,
    case.wscales_flash,
    case.alpha_flash,
    case.lora_d,
    case.lora_l1_flash,
    None,
    case.out_flash,
    backend=backend,
  )


def _prepare_flash_case(case: Case, quantize_smooth) -> None:
  global_scale = ((448.0 * 6.0) /
                  (case.x.float() * case.smooth.float().unsqueeze(0)).abs().max()).to(torch.float32)
  qact, a_sf = quantize_smooth(case.x, case.smooth, global_scale, backend="cute-dsl")
  alpha_scalar = 1.0 / float(global_scale.item())
  case.qact_flash = qact.view(torch.uint8).contiguous()
  case.a_sf_flash = a_sf.view(torch.uint8).flatten().contiguous()
  case.alpha_flash = torch.tensor([alpha_scalar], dtype=torch.float32, device=case.x.device)
  case.lora_l1_flash = (case.lora_l1 / alpha_scalar).contiguous()


def run_case(m: int,
             n: int,
             k: int,
             rank: int,
             impls: list[str],
             seed: int,
             warmup: int,
             iterations: int,
             device: torch.device,
             flashinfer_path: str | None = None) -> list[dict]:
  # Keep FlashInfer import/JIT out of the native/cute timings: load it lazily on the first
  # FlashInfer impl, after cache-dit's kernels have already been measured.
  mm_nvfp4_svdquant = None
  quantize_smooth = None
  flashinfer_skip: str | None = None
  case = prepare_case(m, n, k, rank, seed, device, quantize_smooth=None)
  native_out = _native_call(case)
  native_us = bench(lambda: _native_call(case), warmup, iterations)
  # End-to-end rows additionally include the activation quantization and LoRA-down reduction,
  # which is what a real SVDQW4A4Linear forward pays on top of the GEMM.
  native_e2e_out = case.native_module(case.x)
  native_e2e_us = bench(lambda: case.native_module(case.x), warmup, iterations)
  rows = []

  for impl in impls:
    if impl == "quantize":
      # The activation quantizer plus LoRA-down reduction, shared by both runtimes. e2e rows are
      # this step followed by the GEMM.
      elapsed = bench(lambda: case.native_module.quantize(case.x), warmup, iterations)
      rows.append({
        "m": m,
        "n": n,
        "k": k,
        "rank": rank,
        "impl": impl,
        "median_us": elapsed,
        "speedup_vs_native": None,
        "rel_l2_vs_native": None,
        "finite": None,
        "skipped": None,
      })
      continue
    if impl == "native":
      call = lambda: _native_call(case)
      out = native_out
      rel = 0.0
      baseline_us = native_us
    elif impl == "cute":
      call = lambda: _cute_call(case)
      out = _cute_call(case)
      rel = ((out.float() - native_out.float()).norm() / native_out.float().norm()).item()
      baseline_us = native_us
    elif impl == "native+e2e":
      call = lambda: case.native_module(case.x)
      out = native_e2e_out
      rel = 0.0
      baseline_us = native_e2e_us
    elif impl == "cute+e2e":
      call = lambda: case.cute_module(case.x)
      out = call()
      rel = ((out.float() - native_e2e_out.float()).norm() / native_e2e_out.float().norm()).item()
      baseline_us = native_e2e_us
    elif impl in FLASHINFER_RANKS:
      if rank not in FLASHINFER_RANKS[impl]:
        rows.append({
          "m": m,
          "n": n,
          "k": k,
          "rank": rank,
          "impl": impl,
          "median_us": None,
          "speedup_vs_native": None,
          "rel_l2_vs_native": None,
          "finite": None,
          "skipped": f"unsupported rank {rank}",
        })
        continue
      if flashinfer_skip is not None:
        rows.append({
          "m": m,
          "n": n,
          "k": k,
          "rank": rank,
          "impl": impl,
          "median_us": None,
          "speedup_vs_native": None,
          "rel_l2_vs_native": None,
          "finite": None,
          "skipped": flashinfer_skip,
        })
        continue
      if mm_nvfp4_svdquant is None:
        try:
          mm_nvfp4_svdquant, quantize_smooth = _load_flashinfer(flashinfer_path)
        except Exception as exc:
          # FlashInfer is optional: report the impl as skipped so the cache-dit kernels still run
          # on hosts without it instead of aborting the whole sweep.
          flashinfer_skip = "flashinfer unavailable"
          print(f"[bench] skipping {impl}: {exc}", file=sys.stderr)
          rows.append({
            "m": m,
            "n": n,
            "k": k,
            "rank": rank,
            "impl": impl,
            "median_us": None,
            "speedup_vs_native": None,
            "rel_l2_vs_native": None,
            "finite": None,
            "skipped": flashinfer_skip,
          })
          continue
        _prepare_flash_case(case, quantize_smooth)
      backend = impl.split("+", 1)[1]
      call = lambda: _flashinfer_call(case, mm_nvfp4_svdquant, backend)
      out = call()
      rel = None
      baseline_us = native_us
    else:
      raise ValueError(f"unknown implementation {impl!r}")

    elapsed = bench(call, warmup, iterations)
    rows.append({
      "m": m,
      "n": n,
      "k": k,
      "rank": rank,
      "impl": impl,
      "median_us": elapsed,
      "speedup_vs_native": baseline_us / elapsed,
      "rel_l2_vs_native": rel,
      "finite": bool(torch.isfinite(out).all()),
      "skipped": None,
    })
  return rows


def parse_shapes(values: list[str] | None) -> list[tuple[int, int, int]]:
  if not values:
    return list(DEFAULT_SHAPES)
  shapes = []
  for value in values:
    parts = value.lower().split("x")
    if len(parts) != 3:
      raise argparse.ArgumentTypeError(f"shape must be MxNxK, got {value!r}")
    shapes.append(tuple(int(part) for part in parts))
  return shapes


def main() -> None:
  global _flashinfer_path
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--shape", action="append", help="MxNxK; repeat for multiple shapes")
  parser.add_argument("--rank", action="append", type=int, help="LoRA rank; repeat for multiple")
  parser.add_argument("--impl",
                      default=",".join(DEFAULT_IMPLS),
                      help="comma-separated implementations")
  parser.add_argument("--warmup", type=int, default=10)
  parser.add_argument("--iterations", type=int, default=50)
  parser.add_argument("--seed", type=int, default=20261120)
  parser.add_argument("--flashinfer-path",
                      default=None,
                      help="optional local FlashInfer source path")
  parser.add_argument("--json", action="store_true", help="emit JSON lines instead of a table")
  args = parser.parse_args()

  shapes = parse_shapes(args.shape)
  ranks = args.rank or [32]
  impls = [item.strip() for item in args.impl.split(",") if item.strip()]
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required")
  device = torch.device("cuda:0")

  rows = []
  for m, n, k in shapes:
    for rank in ranks:
      rows.extend(
        run_case(m, n, k, rank, impls, args.seed, args.warmup, args.iterations, device,
                 args.flashinfer_path))

  if args.json:
    for row in rows:
      print(json.dumps(row))
    return
  print("m,n,k,rank,impl,median_us,speedup_vs_native,rel_l2_vs_native,finite,skipped")
  for row in rows:
    print(f"{row['m']},{row['n']},{row['k']},{row['rank']},{row['impl']},"
          f"{row['median_us'] if row['median_us'] is not None else ''},"
          f"{row['speedup_vs_native'] if row['speedup_vs_native'] is not None else ''},"
          f"{row['rel_l2_vs_native'] if row['rel_l2_vs_native'] is not None else ''},"
          f"{row['finite'] if row['finite'] is not None else ''},"
          f"{row['skipped'] or ''}")


if __name__ == "__main__":
  main()
