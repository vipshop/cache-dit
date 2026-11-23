"""Benchmark packed W1/W2/W3 A4 against the same W4A4 producer and GEMM."""
import argparse
import statistics

import torch
from torch import nn

from cache_dit.kernels import expand_wna4
from cache_dit.kernels import svdq_gemm_w4a4
from cache_dit.kernels import svdq_gemm_wna4
from cache_dit.kernels import svdq_quantize_w4a4
from cache_dit.quantization.svdquant.wNa4 import _quantize_svdq_wNa4


def time_ms(fn, warmup: int, iters: int) -> float:
  for _ in range(warmup):
    fn()
  torch.cuda.synchronize()
  times = []
  for _ in range(iters):
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    times.append(start.elapsed_time(end))
  return statistics.median(times)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--m", type=int, default=4096)
  parser.add_argument("--n", type=int, default=12288)
  parser.add_argument("--k", type=int, default=3072)
  parser.add_argument("--rank", type=int, default=32)
  parser.add_argument("--warmup", type=int, default=10)
  parser.add_argument("--iters", type=int, default=30)
  args = parser.parse_args()
  torch.manual_seed(0)
  x = torch.randn(args.m, args.k, device="cuda", dtype=torch.bfloat16)
  linear = nn.Linear(args.k, args.n, bias=False, dtype=torch.bfloat16, device="cuda")
  print("| bits | qweight+sign B | scales B | producer ms | expand ms | GEMM ms | total ms | "
        "W4A4 total ms | ratio | on-chip GEMM ms | module total ms | module/W4A4 |")
  print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
  for bits in (1, 2, 3):
    module = _quantize_svdq_wNa4(linear, bits, rank=args.rank)
    qact, ascales, lora_act = svdq_quantize_w4a4(x,
                                                 module.proj_down,
                                                 module.smooth_factor,
                                                 lora_layout="row_major",
                                                 fp4=True,
                                                 pad_size=256)
    expanded = expand_wna4(module.qweight, module.qweight_hi, bits)
    wscales = module.wscales.view(torch.float8_e4m3fn)
    producer_ms = time_ms(
      lambda: svdq_quantize_w4a4(
        x, module.proj_down, module.smooth_factor, lora_layout="row_major", fp4=True, pad_size=256),
      args.warmup, args.iters)
    expand_ms = time_ms(lambda: expand_wna4(module.qweight, module.qweight_hi, bits), args.warmup,
                        args.iters)
    gemm_fn = lambda: svdq_gemm_w4a4(qact,
                                     expanded,
                                     ascales,
                                     wscales,
                                     lora_act_in=lora_act,
                                     lora_up=module.proj_up,
                                     wcscales=module.wcscales,
                                     lora_layout="row_major",
                                     runtime="cute")
    gemm_ms = time_ms(gemm_fn, args.warmup, args.iters)
    onchip_ms = time_ms(
      lambda: svdq_gemm_wna4(qact,
                             module.qweight,
                             module.qweight_hi,
                             ascales,
                             wscales,
                             lora_act_in=lora_act,
                             lora_up=module.proj_up,
                             wcscales=module.wcscales,
                             bits=bits), args.warmup, args.iters)
    total = producer_ms + expand_ms + gemm_ms
    baseline = producer_ms + gemm_ms
    module_ms = time_ms(lambda: module(x), args.warmup, args.iters)
    qbytes = module.qweight.numel() + (module.qweight_hi.numel() if bits == 3 else 0)
    sbytes = module.wscales.numel() + module.wcscales.numel() * module.wcscales.element_size()
    print(f"| {bits} | {qbytes} | {sbytes} | {producer_ms:.3f} | {expand_ms:.3f} "
          f"| {gemm_ms:.3f} | {total:.3f} | {baseline:.3f} | {total / baseline:.3f}x "
          f"| {onchip_ms:.3f} | {module_ms:.3f} | {module_ms / baseline:.3f}x |")


if __name__ == "__main__":
  main()
