from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from cache_dit.kernels import svdq_gemm_w4a4
from cache_dit.kernels import svdq_gemm_w4a4_v2
from cache_dit.kernels import svdq_quantize_w4a4_act_fuse_lora
from cache_dit.kernels import svdq_quantize_w4a4_wgt

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
  """Parse CLI arguments for the focused runtime benchmark harness.

  :returns: Parsed CLI arguments.
  """

  parser = argparse.ArgumentParser()
  parser.add_argument("--runtime-kernel", choices=("v1", "v2"), default="v2")
  parser.add_argument("--seq-len", type=int, default=4096)
  parser.add_argument("--embed-dim", type=int, default=4096)
  parser.add_argument("--rank", type=int, default=32)
  parser.add_argument("--stage", type=int, default=1)
  parser.add_argument("--block-m", type=int, choices=(64, 128, 256), default=128)
  parser.add_argument("--warmup", type=int, default=20)
  parser.add_argument("--iters", type=int, default=100)
  parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help="Optional JSON output path. Defaults to stdout-only when omitted.",
  )
  return parser.parse_args()


def runtime_dtype() -> torch.dtype:
  """Return the CUDA runtime dtype used by the benchmark.

  :returns: `torch.bfloat16` when supported, otherwise `torch.float16`.
  """

  return (torch.bfloat16
          if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16)


def benchmark(op, kwargs: dict[str, object], *, warmup: int, iters: int) -> dict[str, object]:
  """Benchmark a single CUDA op.

  :param op: Callable kernel wrapper to benchmark.
  :param kwargs: Keyword arguments forwarded to the callable.
  :param warmup: Warmup iteration count.
  :param iters: Timed iteration count.
  :returns: Dict containing latency, peak memory, and output tensor.
  """

  torch.cuda.synchronize()
  out = None
  for _ in range(warmup):
    out = op(**kwargs)
  torch.cuda.synchronize()
  torch.cuda.reset_peak_memory_stats()
  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
  start.record()
  for _ in range(iters):
    out = op(**kwargs)
  end.record()
  torch.cuda.synchronize()
  assert out is not None
  return {
    "latency_ms": start.elapsed_time(end) / iters,
    "peak_mem_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
    "output": out,
  }


def select_kernel(runtime_kernel: str):
  """Return the requested runtime kernel wrapper.

  :param runtime_kernel: Public runtime kernel name.
  :returns: Callable kernel wrapper.
  """

  if runtime_kernel == "v1":
    return svdq_gemm_w4a4
  return svdq_gemm_w4a4_v2


def maybe_set_block_env(runtime_kernel: str, block_m: int) -> tuple[str, str | None]:
  """Set the runtime BLOCK_M override for v2/v3 while preserving the previous value.

  :param runtime_kernel: Public runtime kernel name.
  :param block_m: Logical block-M override.
  :returns: Tuple of `(env_name, previous_value)`.
  """

  if runtime_kernel != "v2":
    return "", None
  env_name = "CACHE_DIT_SVDQ_V2_BLOCK_M"

  previous = os.environ.get(env_name)
  os.environ[env_name] = str(block_m)
  return env_name, previous


def restore_block_env(env_name: str, previous: str | None) -> None:
  """Restore the benchmark-time BLOCK_M override.

  :param env_name: Environment variable name.
  :param previous: Previous value before override.
  """

  if not env_name:
    return
  if previous is None:
    os.environ.pop(env_name, None)
  else:
    os.environ[env_name] = previous


def main() -> None:
  """Run a focused single-shape runtime benchmark and optionally export JSON.

  The harness is intended for Nsight Systems and Nsight Compute collection, so it only benchmarks
  one runtime kernel / shape combination per invocation.
  """

  args = parse_args()
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this benchmark.")

  device = torch.device("cuda")
  dtype = runtime_dtype()
  torch.manual_seed(0)

  activations = torch.randn(args.seq_len, args.embed_dim, device=device, dtype=dtype)
  smooth = torch.ones(args.embed_dim, device=device, dtype=dtype)
  lora_down = torch.randn(args.embed_dim, args.rank, device=device, dtype=dtype) * 0.01
  lora_up = torch.randn(args.embed_dim, args.rank, device=device, dtype=dtype) * 0.01
  bias = torch.randn(args.embed_dim, device=device, dtype=dtype) * 0.01

  qact, ascales, lora_act = svdq_quantize_w4a4_act_fuse_lora(
    input=activations,
    lora_down=lora_down,
    smooth=smooth,
    fp4=False,
    pad_size=256,
  )
  qweight, wscales = svdq_quantize_w4a4_wgt(
    torch.randn(args.embed_dim, args.embed_dim, device=device, dtype=dtype))

  kernel = select_kernel(args.runtime_kernel)
  kwargs: dict[str, object] = {
    "act": qact,
    "wgt": qweight,
    "ascales": ascales,
    "wscales": wscales,
    "lora_act_in": lora_act,
    "lora_up": lora_up,
    "bias": bias,
    "act_unsigned": False,
    "fp4": False,
    "alpha": 1.0,
    "output_dtype": dtype,
  }
  if args.runtime_kernel == "v2":
    kwargs["stage"] = args.stage

  env_name, previous = maybe_set_block_env(args.runtime_kernel, args.block_m)
  try:
    result = benchmark(kernel, kwargs, warmup=args.warmup, iters=args.iters)
  finally:
    restore_block_env(env_name, previous)

  summary = {
    "runtime_kernel": args.runtime_kernel,
    "shape": {
      "M": args.seq_len,
      "N": args.embed_dim,
      "K": args.embed_dim,
      "rank": args.rank,
    },
    "stage": args.stage if args.runtime_kernel == "v2" else None,
    "block_m": args.block_m if args.runtime_kernel == "v2" else None,
    "warmup": args.warmup,
    "iters": args.iters,
    "dtype": str(dtype),
    "latency_ms": result["latency_ms"],
    "peak_mem_mb": result["peak_mem_mb"],
  }

  if args.output is not None:
    output_path = args.output
    if not output_path.is_absolute():
      output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

  print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
