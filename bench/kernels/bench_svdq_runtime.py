from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch

from cache_dit.kernels import svdq_gemm_w4a4
from cache_dit.kernels import svdq_gemm_w4a4_v2
from cache_dit.kernels import svdq_quantize_w4a4_act_fuse_lora
from cache_dit.kernels import svdq_quantize_w4a4_wgt

REPO_ROOT = Path(__file__).resolve().parents[2]
BLOCK_MS = (64, 128, 256)
EXPLICIT_STAGES = (1, 2, 3)
SEQ_LENS = (256, 1024, 4096, 8192)
RANKS = (32, 128)
EMBED_DIM = 4096
DEFAULT_STAGE = 1


def parse_args() -> argparse.Namespace:
  """Parse command-line arguments for the runtime sweep benchmark.

  :returns: Parsed CLI arguments including the output prefix.
  """

  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--output-prefix",
    type=Path,
    default=REPO_ROOT / ".tmp/svdq_runtime_mrank_sweep_with_default_stage",
    help="Output prefix used for the generated JSON and CSV reports.",
  )
  return parser.parse_args()


def runtime_dtype() -> torch.dtype:
  """Return the CUDA runtime dtype used by the benchmark.

  :returns: `torch.bfloat16` when supported, otherwise `torch.float16`.
  """

  return (torch.bfloat16
          if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16)


def select_iters(seq_len: int, embed_dim: int) -> tuple[int, int]:
  """Choose warmup and iteration counts based on total GEMM work.

  :param seq_len: Sequence length used as the GEMM M dimension.
  :param embed_dim: Shared GEMM K/N dimension.
  :returns:`(warmup, iters)` tuned to keep large shapes tractable.
  """

  work_units = seq_len * embed_dim * embed_dim
  if work_units >= 250_000_000_000:
    return 3, 10
  if work_units >= 120_000_000_000:
    return 4, 12
  if work_units >= 50_000_000_000:
    return 5, 16
  if work_units >= 15_000_000_000:
    return 8, 24
  if work_units >= 5_000_000_000:
    return 10, 32
  return 12, 40


def time_cuda_call(fn):
  """Measure a single CUDA call with event timing.

  :param fn: Zero-argument callable executed once on CUDA.
  :returns: Tuple of `(elapsed_ms, result)`.
  """

  start = torch.cuda.Event(enable_timing=True)
  end = torch.cuda.Event(enable_timing=True)
  torch.cuda.synchronize()
  start.record()
  result = fn()
  end.record()
  torch.cuda.synchronize()
  return start.elapsed_time(end), result


def benchmark(op, kwargs, *, warmup: int, iters: int):
  """Benchmark a CUDA op and return latency, memory, and output.

  :param op: Callable kernel wrapper to benchmark.
  :param kwargs: Keyword arguments passed to the callable.
  :param warmup: Number of warmup iterations.
  :param iters: Number of timed iterations.
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


def write_csv(csv_path: Path, rows: list[dict[str, object]]) -> None:
  """Write the benchmark rows to CSV.

  :param csv_path: Destination CSV path.
  :param rows: Flat benchmark row dictionaries.
  """

  fieldnames = [
    "rank",
    "M",
    "N",
    "K",
    "block_m",
    "stage_mode",
    "requested_stage",
    "effective_stage",
    "warmup",
    "iters",
    "act_quant_ms",
    "wgt_quant_ms",
    "v1_latency_ms",
    "v2_latency_ms",
    "speedup_vs_v1",
    "v1_peak_mem_mb",
    "v2_peak_mem_mb",
    "max_abs_diff",
    "mean_abs_diff",
    "exact_match",
  ]
  with csv_path.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
      writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
  """Run the full runtime sweep and export JSON/CSV reports.

  The sweep includes both explicit `stage=1/2/3` calls and one additional
  default-call group where `svdq_gemm_w4a4_v2` is invoked without a `stage`
  keyword. This keeps the report aligned with the current Python default path.
  """

  args = parse_args()
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this benchmark.")

  json_path = args.output_prefix.with_suffix(".json")
  csv_path = args.output_prefix.with_suffix(".csv")
  json_path.parent.mkdir(parents=True, exist_ok=True)

  device = torch.device("cuda")
  dtype = runtime_dtype()
  torch.manual_seed(0)
  rows: list[dict[str, object]] = []
  default_stage_checks: list[dict[str, object]] = []

  with torch.inference_mode():
    for rank in RANKS:
      for seq_len in SEQ_LENS:
        warmup, iters = select_iters(seq_len, EMBED_DIM)
        activations = torch.randn(seq_len, EMBED_DIM, device=device, dtype=dtype)
        smooth = torch.ones(EMBED_DIM, device=device, dtype=dtype)
        lora_down = torch.randn(EMBED_DIM, rank, device=device, dtype=dtype) * 0.01
        lora_up = torch.randn(EMBED_DIM, rank, device=device, dtype=dtype) * 0.01
        bias = torch.randn(EMBED_DIM, device=device, dtype=dtype) * 0.01

        act_quant_ms, quantized = time_cuda_call(lambda: svdq_quantize_w4a4_act_fuse_lora(
          input=activations,
          lora_down=lora_down,
          smooth=smooth,
          fp4=False,
          pad_size=256,
        ))
        qact, ascales, lora_act = quantized
        wgt_quant_ms, quantized_wgt = time_cuda_call(lambda: svdq_quantize_w4a4_wgt(
          torch.randn(EMBED_DIM, EMBED_DIM, device=device, dtype=dtype)))
        qweight, wscales = quantized_wgt

        common_kwargs = {
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

        os.environ.pop("CACHE_DIT_SVDQ_V2_BLOCK_M", None)
        v1 = benchmark(svdq_gemm_w4a4, common_kwargs, warmup=warmup, iters=iters)

        for block_m in BLOCK_MS:
          os.environ["CACHE_DIT_SVDQ_V2_BLOCK_M"] = str(block_m)
          v2_results: dict[str, dict[str, object]] = {}

          default_v2 = benchmark(svdq_gemm_w4a4_v2, common_kwargs, warmup=warmup, iters=iters)
          default_diff = (default_v2["output"].float() - v1["output"].float()).abs()
          rows.append({
            "M": seq_len,
            "N": EMBED_DIM,
            "K": EMBED_DIM,
            "rank": rank,
            "dtype": str(dtype),
            "warmup": warmup,
            "iters": iters,
            "act_quant_ms": act_quant_ms,
            "wgt_quant_ms": wgt_quant_ms,
            "block_m": block_m,
            "stage_mode": "default",
            "requested_stage": None,
            "effective_stage": DEFAULT_STAGE,
            "v1_latency_ms": v1["latency_ms"],
            "v2_latency_ms": default_v2["latency_ms"],
            "speedup_vs_v1": v1["latency_ms"] / default_v2["latency_ms"],
            "v1_peak_mem_mb": v1["peak_mem_mb"],
            "v2_peak_mem_mb": default_v2["peak_mem_mb"],
            "max_abs_diff": default_diff.max().item(),
            "mean_abs_diff": default_diff.mean().item(),
            "exact_match": bool(torch.equal(default_v2["output"], v1["output"])),
          })
          v2_results["default"] = default_v2

          for stage in EXPLICIT_STAGES:
            v2_kwargs = dict(common_kwargs)
            v2_kwargs["stage"] = stage
            explicit_v2 = benchmark(svdq_gemm_w4a4_v2, v2_kwargs, warmup=warmup, iters=iters)
            explicit_diff = (explicit_v2["output"].float() - v1["output"].float()).abs()
            rows.append({
              "M": seq_len,
              "N": EMBED_DIM,
              "K": EMBED_DIM,
              "rank": rank,
              "dtype": str(dtype),
              "warmup": warmup,
              "iters": iters,
              "act_quant_ms": act_quant_ms,
              "wgt_quant_ms": wgt_quant_ms,
              "block_m": block_m,
              "stage_mode": "explicit",
              "requested_stage": stage,
              "effective_stage": stage,
              "v1_latency_ms": v1["latency_ms"],
              "v2_latency_ms": explicit_v2["latency_ms"],
              "speedup_vs_v1": v1["latency_ms"] / explicit_v2["latency_ms"],
              "v1_peak_mem_mb": v1["peak_mem_mb"],
              "v2_peak_mem_mb": explicit_v2["peak_mem_mb"],
              "max_abs_diff": explicit_diff.max().item(),
              "mean_abs_diff": explicit_diff.mean().item(),
              "exact_match": bool(torch.equal(explicit_v2["output"], v1["output"])),
            })
            v2_results[f"explicit_stage_{stage}"] = explicit_v2

          explicit_stage1 = v2_results["explicit_stage_1"]
          default_output = v2_results["default"]["output"]
          explicit_stage1_output = explicit_stage1["output"]
          default_vs_explicit = (default_output.float() - explicit_stage1_output.float()).abs()
          default_stage_checks.append({
            "rank":
            rank,
            "M":
            seq_len,
            "block_m":
            block_m,
            "default_effective_stage":
            DEFAULT_STAGE,
            "default_latency_ms":
            v2_results["default"]["latency_ms"],
            "explicit_stage1_latency_ms":
            explicit_stage1["latency_ms"],
            "latency_delta_ms":
            v2_results["default"]["latency_ms"] - explicit_stage1["latency_ms"],
            "latency_ratio_default_vs_explicit_stage1":
            (v2_results["default"]["latency_ms"] / explicit_stage1["latency_ms"]),
            "max_abs_diff_vs_explicit_stage1":
            default_vs_explicit.max().item(),
            "mean_abs_diff_vs_explicit_stage1":
            default_vs_explicit.mean().item(),
            "exact_match_vs_explicit_stage1":
            bool(torch.equal(default_output, explicit_stage1_output)),
          })

        del activations, smooth, lora_down, lora_up, bias
        del qact, ascales, lora_act, qweight, wscales
        del v1
        torch.cuda.empty_cache()

  rows.sort(key=lambda row: (
    row["rank"],
    row["M"],
    row["block_m"],
    row["stage_mode"],
    -1 if row["requested_stage"] is None else row["requested_stage"],
  ))
  default_stage_checks.sort(key=lambda row: (row["rank"], row["M"], row["block_m"]))
  write_csv(csv_path, rows)

  summary = {
    "device": str(device),
    "dtype": str(dtype),
    "default_stage": DEFAULT_STAGE,
    "rows": rows,
    "default_stage_checks": default_stage_checks,
    "artifacts": {
      "json": str(json_path),
      "csv": str(csv_path),
    },
  }
  json_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
  os.environ.pop("CACHE_DIT_SVDQ_V2_BLOCK_M", None)
  print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
