"""Benchmark the fused SVDQ-FP8 producer + GEMM against plain FP8 per-row.

Example:
  python bench/kernels/bench_svdquant_fp8.py --ranks 32 64 128
"""

import argparse
import statistics

import torch

from cache_dit.kernels import fp8_gemm
from cache_dit.kernels import fp8_quantize
from cache_dit.kernels import svdq_gemm_w8a8
from cache_dit.kernels import svdq_quantize_w8a8
from cache_dit.kernels import svdq_gemm_w4a8
from cache_dit.kernels import svdq_quantize_w4a8
from cache_dit.quantization.svdquant.w4a8 import quantize_residual_fp4_per_row

DEFAULT_SHAPES = (
  (4096, 3072, 3072),
  (4096, 12288, 3072),
  (8192, 12288, 3072),
)


def time_ms(fn, warmup: int, iters: int) -> list[float]:
  """Return individual CUDA-event timings in milliseconds."""
  for _ in range(warmup):
    fn()
  torch.cuda.synchronize()
  samples = []
  for _ in range(iters):
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    samples.append(start.elapsed_time(end))
  return samples


def quantize_per_row(residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """Quantize a weight matrix to E4M3 with one scale per output channel."""
  scale = residual.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 448.0
  return (residual / scale).to(torch.float8_e4m3fn), scale.squeeze(1).contiguous()


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--ranks", type=int, nargs="+", default=[32, 64, 128])
  parser.add_argument("--shapes",
                      type=str,
                      nargs="+",
                      help="Shapes in MxNxK form; defaults to three FLUX projections.")
  parser.add_argument("--warmup", type=int, default=10)
  parser.add_argument("--iters", type=int, default=30)
  args = parser.parse_args()
  if any(rank not in (0, 32, 64, 128) for rank in args.ranks):
    raise ValueError("ranks must be selected from 0, 32, 64 and 128.")
  shapes = DEFAULT_SHAPES
  if args.shapes:
    shapes = tuple(tuple(int(v) for v in shape.lower().split("x")) for shape in args.shapes)

  torch.manual_seed(0)
  print(
    "| M | N | K | rank | FP8 total ms | W8A8 producer ms | W8A8 GEMM ms | W8A8 total ms "
    "| W8A8/FP8 | W4A8 producer ms | W4A8 GEMM ms | W4A8 total ms | W4A8/W8A8 | qB bytes ratio |")
  print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
  for m, n, k in shapes:
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / 8
    smooth = torch.rand(k, device="cuda", dtype=torch.bfloat16) + 0.5
    ws = w.float() * smooth.float()
    bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    xs = (x / smooth).contiguous()
    qa_plain, sa_plain = fp8_quantize(xs, mode="per_row")
    qb_plain, sb_plain = quantize_per_row(ws)
    plain_q = time_ms(lambda: fp8_quantize(xs, mode="per_row"), args.warmup, args.iters)
    plain_g = time_ms(
      lambda: fp8_gemm(qa_plain, qb_plain, sa_plain, sb_plain, bias=bias, mode="per_row"),
      args.warmup, args.iters)
    plain_total = statistics.median(plain_q) + statistics.median(plain_g)

    for rank in args.ranks:
      if rank:
        u, sv, vh = torch.svd_lowrank(ws, q=rank + 8, niter=2)
        l1 = u[:, :rank] * sv[:rank]
        l2 = vh[:, :rank].t().contiguous()
        residual = ws - l1 @ l2
        up = l1.bfloat16().contiguous()
        down = (l2.double() / smooth.double().unsqueeze(0)).bfloat16().contiguous()
      else:
        up = down = None
        residual = ws
      qb, sb = quantize_per_row(residual)
      producer_ms = statistics.median(
        time_ms(lambda: svdq_quantize_w8a8(x, smooth, down), args.warmup, args.iters))
      qa, sa, d = svdq_quantize_w8a8(x, smooth, down)
      gemm_ms = statistics.median(
        time_ms(lambda: svdq_gemm_w8a8(qa, qb, sa, sb, lora_act_in=d, lora_up=up, bias=bias),
                args.warmup, args.iters))
      total = producer_ms + gemm_ms
      qb4, sb4 = quantize_residual_fp4_per_row(residual)
      producer4_ms = statistics.median(
        time_ms(lambda: svdq_quantize_w4a8(x, smooth, down), args.warmup, args.iters))
      qa4, sa4, d4 = svdq_quantize_w4a8(x, smooth, down)
      gemm4_ms = statistics.median(
        time_ms(lambda: svdq_gemm_w4a8(qa4, qb4, sa4, sb4, lora_act_in=d4, lora_up=up, bias=bias),
                args.warmup, args.iters))
      total4 = producer4_ms + gemm4_ms
      print(f"| {m} | {n} | {k} | {rank} | {plain_total:.3f} | {producer_ms:.3f} "
            f"| {gemm_ms:.3f} | {total:.3f} | {total / plain_total:.3f}x | {producer4_ms:.3f} "
            f"| {gemm4_ms:.3f} | {total4:.3f} | {total4 / total:.3f}x | "
            f"{qb4.numel() / qb.numel():.3f}x |")


if __name__ == "__main__":
  main()
