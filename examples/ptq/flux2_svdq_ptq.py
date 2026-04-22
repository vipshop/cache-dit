from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
  sys.path.insert(0, str(SRC_ROOT))

import cache_dit
from cache_dit.kernels import svdq_extension_is_available
from cache_dit.metrics.metrics import compute_psnr_file
from cache_dit.quantization import QuantizeConfig

DEFAULT_MODEL_SOURCE = os.getenv("FLUX_2_KLEIN_4B_DIR", "black-forest-labs/FLUX.2-klein-4B")
DEFAULT_PROMPTS_PATH = Path(__file__).resolve().parents[1] / "data" / "prompts" / "DrawBench200.txt"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "FLUX.2-klein-4B-svdq"
DEFAULT_BENCHMARK_RUNS = 5
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024
DEFAULT_INFERENCE_STEPS = 4
DEFAULT_SEED = 0
DEFAULT_SVDQ_KWARGS = {
  "streaming": True,
  "calibrate_precision": "low",
  "activation_buffer_flush_sample_count": 1,
  "activation_buffer_flush_cpu_bytes": None,
}
_CALIBRATE_PRECISION_CHOICES = ("low", "medium", "high")


@dataclass(frozen=True)
class StageBenchmark:
  """Summarize repeated inference measurements for one pipeline stage."""

  stage: str
  run_count: int
  avg_latency_s: float
  total_latency_s: float
  peak_memory_gb: float
  transformer_weight_cuda_gb: float


@dataclass
class StageResult:
  """Store benchmark outputs, image artifacts, and execution status for one stage."""

  stage: str
  benchmark: Optional[StageBenchmark] = None
  image_path: Optional[Path] = None
  status: str = "ok"
  warmup_latency_s: Optional[float] = None


def parse_args() -> argparse.Namespace:
  """Parse CLI arguments for the FLUX.2 SVDQ PTQ example."""

  parser = argparse.ArgumentParser(
    description=("Run an end-to-end SVDQ PTQ example for FLUX.2-klein-4B with DrawBench200 "
                 "calibration prompts, baseline/quantized/loaded/compiled comparisons, "
                 "and Markdown reporting."))
  parser.add_argument("--model-source",
                      default=DEFAULT_MODEL_SOURCE,
                      help="Model id or local path.")
  parser.add_argument(
    "--output-dir",
    default=str(DEFAULT_OUTPUT_DIR),
    help="Directory for serialized checkpoints, images, and reports.",
  )
  parser.add_argument(
    "--prompts-path",
    default=str(DEFAULT_PROMPTS_PATH),
    help="Path to DrawBench-style calibration prompts.",
  )
  parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Validation image height.")
  parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Validation image width.")
  parser.add_argument(
    "--num-inference-steps",
    type=int,
    default=DEFAULT_INFERENCE_STEPS,
    help="Inference steps for validation and latency benchmarking.",
  )
  parser.add_argument(
    "--calibration-height",
    type=int,
    default=None,
    help="Calibration image height. Defaults to --height.",
  )
  parser.add_argument(
    "--calibration-width",
    type=int,
    default=None,
    help="Calibration image width. Defaults to --width.",
  )
  parser.add_argument(
    "--calibration-steps",
    type=int,
    default=None,
    help="Inference steps used during calibration. Defaults to --num-inference-steps.",
  )
  parser.add_argument(
    "--calibration-limit",
    type=int,
    default=None,
    help="Optional prompt limit for faster local iteration. Defaults to all prompts.",
  )
  parser.add_argument(
    "--benchmark-runs",
    type=int,
    default=DEFAULT_BENCHMARK_RUNS,
    help="Repeated runs used to average latency per stage.",
  )
  parser.add_argument(
    "--rank",
    type=int,
    default=128,
    help="Low-rank SVDQ rank. Higher ranks for distillation models.",
  )
  parser.add_argument(
    "--calibrate-precision",
    choices=_CALIBRATE_PRECISION_CHOICES,
    default=DEFAULT_SVDQ_KWARGS["calibrate_precision"],
    help=("Precision policy for PTQ calibration and low-rank decomposition. "
          "Use low for randomized svd_lowrank, medium for float32 full SVD, "
          "and high for float64 full SVD."),
  )
  parser.add_argument("--seed",
                      type=int,
                      default=DEFAULT_SEED,
                      help="Seed for prompt and latent RNG.")
  parser.add_argument(
    "--skip-compile",
    action="store_true",
    default=False,
    help="Skip the torch.compile comparison stage.",
  )
  parser.add_argument(
    "--exclude-layers",
    nargs='+',
    default=None,
    help="Optional list of transformer layer names to exclude from quantization. ",
  )
  args = parser.parse_args()
  return args


def resolve_torch_dtype() -> torch.dtype:
  """Choose a runtime dtype that matches common FLUX.2 deployment settings."""

  if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    return torch.bfloat16
  return torch.float16


def ensure_runtime_requirements() -> None:
  """Fail early when the environment cannot run this CUDA-only SVDQ example."""

  if not torch.cuda.is_available():
    raise RuntimeError("This example requires a CUDA-capable device.")
  if not svdq_extension_is_available():
    raise RuntimeError(
      "The optional SVDQ extension is unavailable. Build/install cache-dit with SVDQ support first."
    )


def make_cpu_generator(seed: int) -> torch.Generator:
  """Create a deterministic CPU generator for diffusion sampling."""

  return torch.Generator(device="cpu").manual_seed(seed)


def set_global_seed(seed: int) -> None:
  """Set Python, NumPy, and Torch RNGs so calibration and testing stay reproducible."""

  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def load_calibration_prompts(prompts_path: Path, limit: Optional[int]) -> list[str]:
  """Load non-empty calibration prompts from DrawBench200 or a compatible text file."""

  if not prompts_path.is_file():
    raise FileNotFoundError(f"Calibration prompts file not found: {prompts_path}")
  prompts = [line.strip() for line in prompts_path.read_text(encoding="utf-8").splitlines()]
  prompts = [prompt for prompt in prompts if prompt]
  if not prompts:
    raise ValueError(f"No prompts found in calibration file: {prompts_path}")
  if limit is not None:
    if limit < 1:
      raise ValueError(f"--calibration-limit must be positive, got {limit}.")
    prompts = prompts[:limit]
  return prompts


def build_random_short_prompt() -> str:
  """Construct one deterministic short prompt for validation and visualization."""
  return "A cat sitting on a bench, digital art, by Cache-DiT x SVDQuant"


def cleanup_pipe(pipe) -> None:
  """Release GPU memory held by a pipeline between stages."""

  if pipe is None:
    return
  try:
    pipe.to("cpu")
  except Exception:
    pass
  del pipe
  if torch.cuda.is_available():
    torch.cuda.empty_cache()


def reset_cuda_peak_memory() -> None:
  """Reset CUDA allocator stats before timing one inference run."""

  if not torch.cuda.is_available():
    return
  torch.cuda.empty_cache()
  torch.cuda.reset_peak_memory_stats()
  torch.cuda.synchronize()


def unwrap_compiled_module(module: torch.nn.Module) -> torch.nn.Module:
  """Return the original module hidden behind `torch.compile` wrappers."""

  while hasattr(module, "_orig_mod"):
    module = module._orig_mod
  return module


def compute_module_cuda_tensor_bytes(module: torch.nn.Module) -> int:
  """Measure unique CUDA tensor storage used by one module's parameters and buffers."""

  module = unwrap_compiled_module(module)
  seen_storage_ptrs: set[int] = set()
  total_bytes = 0
  for tensor in list(module.parameters()) + list(module.buffers()):
    if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cuda":
      continue
    storage = tensor.untyped_storage()
    storage_ptr = storage.data_ptr()
    if storage_ptr == 0 or storage_ptr in seen_storage_ptrs:
      continue
    seen_storage_ptrs.add(storage_ptr)
    total_bytes += storage.nbytes()
  return total_bytes


def run_single_inference(
  pipe,
  *,
  prompt: str,
  height: int,
  width: int,
  num_inference_steps: int,
  seed: int,
):
  """Run one deterministic FLUX.2 inference and collect latency and peak memory."""

  reset_cuda_peak_memory()
  start_time = time.perf_counter()
  with torch.inference_mode():
    image = pipe(
      prompt=prompt,
      height=height,
      width=width,
      num_inference_steps=num_inference_steps,
      generator=make_cpu_generator(seed),
    ).images[0]
  if torch.cuda.is_available():
    torch.cuda.synchronize()
    peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
  else:
    peak_memory_gb = 0.0
  latency_s = time.perf_counter() - start_time
  return image, latency_s, peak_memory_gb


def benchmark_stage(
  pipe,
  *,
  stage: str,
  prompt: str,
  height: int,
  width: int,
  num_inference_steps: int,
  benchmark_runs: int,
  seed: int,
  image_path: Path,
) -> StageResult:
  """Benchmark repeated inference for one stage and persist the last image."""

  if benchmark_runs < 1:
    raise ValueError(f"benchmark_runs must be positive, got {benchmark_runs}.")

  latencies: list[float] = []
  peak_memory_gb = 0.0
  image = None
  for _ in range(benchmark_runs):
    image, latency_s, run_peak_memory_gb = run_single_inference(
      pipe,
      prompt=prompt,
      height=height,
      width=width,
      num_inference_steps=num_inference_steps,
      seed=seed,
    )
    latencies.append(latency_s)
    peak_memory_gb = max(peak_memory_gb, run_peak_memory_gb)

  assert image is not None
  image_path.parent.mkdir(parents=True, exist_ok=True)
  image.save(image_path)

  total_latency_s = sum(latencies)
  benchmark = StageBenchmark(
    stage=stage,
    run_count=benchmark_runs,
    avg_latency_s=total_latency_s / benchmark_runs,
    total_latency_s=total_latency_s,
    peak_memory_gb=peak_memory_gb,
    transformer_weight_cuda_gb=compute_module_cuda_tensor_bytes(pipe.transformer) / (1024 ** 3),
  )
  return StageResult(stage=stage, benchmark=benchmark, image_path=image_path)


def make_calibrate_fn(
  pipe,
  *,
  prompts: list[str],
  height: int,
  width: int,
  num_inference_steps: int,
  seed: int,
):
  """Build a PTQ calibration callback that drives DrawBench prompts through the pipeline."""

  def calibrate_fn(**_: object) -> None:
    # PTQ observers live inside the transformer; running the public pipeline keeps
    # prompt encoding and diffusion scheduling aligned with the real inference path.
    # Use the same fixed seed for every calibration request so the PTQ example is
    # directly reproducible across runs.
    with torch.inference_mode():
      for prompt in prompts:
        _ = pipe(
          prompt=prompt,
          height=height,
          width=width,
          num_inference_steps=num_inference_steps,
          generator=make_cpu_generator(seed),
        )

  return calibrate_fn


def compare_images(reference_path: Path, candidate_path: Path) -> tuple[float, int, float]:
  """Compute PSNR and absolute-difference summaries for one image pair."""

  image_a = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
  image_b = cv2.imread(str(candidate_path), cv2.IMREAD_COLOR)
  if image_a is None or image_b is None:
    raise FileNotFoundError(f"Missing comparison image pair: {reference_path}, {candidate_path}")
  abs_diff = cv2.absdiff(image_a, image_b)
  psnr = float(compute_psnr_file(str(reference_path), str(candidate_path)))
  return psnr, int(abs_diff.max()), float(abs_diff.mean())


def format_markdown_cell(value: object) -> str:
  """Escape Markdown table cell delimiters for generated reports."""

  return str(value).replace("|", "\\|").replace("\n", "<br>")


def format_markdown_table(
  title: str,
  headers: tuple[str, ...],
  rows: Iterable[tuple[object, ...]],
) -> str:
  """Build a GitHub-flavored Markdown table with a section title."""

  lines = [title, "", f"| {' | '.join(headers)} |", f"| {' | '.join(':---:' for _ in headers)} |"]
  for row in rows:
    row = tuple(row)
    if len(row) != len(headers):
      raise ValueError(f"Row has {len(row)} cells, expected {len(headers)}.")
    lines.append(f"| {' | '.join(format_markdown_cell(value) for value in row)} |")
  formated_table = "\n".join(lines) + "\n"
  print(formated_table)
  return formated_table


def relative_path(path: Optional[Path], root: Path) -> str:
  """Render one artifact path relative to the output directory when possible."""

  if path is None:
    return "n/a"
  try:
    return str(path.resolve().relative_to(root.resolve()))
  except ValueError:
    return str(path.resolve())


def format_float(value: Optional[float], digits: int = 4) -> str:
  """Format optional floats for Markdown tables."""

  if value is None:
    return "n/a"
  if math.isinf(value):
    return "inf"
  return f"{value:.{digits}f}"


def make_visual_comparison(
  *,
  output_path: Path,
  validation_prompt: str,
  stage_results: list[StageResult],
) -> Optional[Path]:
  """Create a side-by-side comparison canvas from available stage images."""

  panels: list[np.ndarray] = []
  panel_width = 320
  panel_height = 320
  panel_header_height = 92
  for result in stage_results:
    if result.image_path is None or result.benchmark is None:
      continue
    image = cv2.imread(str(result.image_path), cv2.IMREAD_COLOR)
    if image is None:
      continue
    resized = cv2.resize(image, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
    panel = np.full((panel_height + panel_header_height, panel_width, 3), 255, dtype=np.uint8)
    panel[panel_header_height:, :, :] = resized
    cv2.putText(panel, result.stage, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
    cv2.putText(
      panel,
      f"avg infer: {result.benchmark.avg_latency_s:.3f}s",
      (12, 50),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.54,
      (45, 45, 45),
      1,
    )
    cv2.putText(
      panel,
      f"tfmr cuda mem: {result.benchmark.transformer_weight_cuda_gb:.3f} GiB",
      (12, 75),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.54,
      (45, 45, 45),
      1,
    )
    panels.append(panel)

  if not panels:
    return None

  content = cv2.hconcat(panels)
  title_height = 72
  canvas = np.full((content.shape[0] + title_height, content.shape[1], 3), 255, dtype=np.uint8)
  canvas[title_height:, :, :] = content
  cv2.putText(
    canvas,
    "FLUX.2-klein-4B SVDQ PTQ comparison",
    (12, 28),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.82,
    (0, 0, 0),
    2,
  )
  subtitle = validation_prompt[:130] + ("..." if len(validation_prompt) > 130 else "")
  cv2.putText(
    canvas,
    subtitle,
    (12, 58),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.56,
    (40, 40, 40),
    1,
  )
  output_path.parent.mkdir(parents=True, exist_ok=True)
  cv2.imwrite(str(output_path), canvas)
  return output_path


def persist_report(
  *,
  output_path: Path,
  metadata_rows: list[tuple[object, ...]],
  stage_rows: list[tuple[object, ...]],
  psnr_rows: list[tuple[object, ...]],
  artifact_rows: list[tuple[object, ...]],
) -> Path:
  """Write one consolidated Markdown report for the PTQ run."""

  sections = [
    "# FLUX.2-klein-4B SVDQ PTQ example report",
    "",
    "This report compares one deterministic validation prompt across the baseline, "
    "memory-quantized, reloaded, and compiled quantized transformer stages.",
    "",
    format_markdown_table("## Run metadata", ("field", "value"), metadata_rows),
    format_markdown_table(
      "## Stage metrics",
      (
        "stage",
        "run_count",
        "avg_latency_s",
        "total_latency_s",
        "peak_memory_gb",
        "transformer_weight_cuda_gb",
        "status",
      ),
      stage_rows,
    ),
    format_markdown_table(
      "## PSNR comparisons",
      ("comparison", "psnr", "max_abs_diff", "mean_abs_diff", "status"),
      psnr_rows,
    ),
    format_markdown_table("## Artifacts", ("artifact", "path"), artifact_rows),
  ]
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text("\n".join(sections).rstrip() + "\n", encoding="utf-8")
  return output_path


def persist_summary_json(
  *,
  output_path: Path,
  metadata_rows: list[tuple[object, ...]],
  stage_order: list[StageResult],
  psnr_rows: list[tuple[object, ...]],
  artifact_rows: list[tuple[object, ...]],
) -> Path:
  """Write a machine-readable summary next to the Markdown report."""

  summary = {
    "metadata": {
      str(key): value
      for key, value in metadata_rows
    },
    "stages": {
      result.stage: {
        "status": result.status,
        "warmup_latency_s": result.warmup_latency_s,
        "image_path": str(result.image_path) if result.image_path is not None else None,
        "benchmark": None if result.benchmark is None else {
          "run_count": result.benchmark.run_count,
          "avg_latency_s": result.benchmark.avg_latency_s,
          "total_latency_s": result.benchmark.total_latency_s,
          "peak_memory_gb": result.benchmark.peak_memory_gb,
          "transformer_weight_cuda_gb": result.benchmark.transformer_weight_cuda_gb,
        },
      }
      for result in stage_order
    },
    "psnr_rows": [{
      "comparison": row[0],
      "psnr": row[1],
      "max_abs_diff": row[2],
      "mean_abs_diff": row[3],
      "status": row[4],
    } for row in psnr_rows],
    "artifacts": {
      str(name): path
      for name, path in artifact_rows
    },
  }
  output_path.parent.mkdir(parents=True, exist_ok=True)
  output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  return output_path


def load_flux2_pipe(model_source: str, torch_dtype: torch.dtype):
  """Load the FLUX.2-klein pipeline and move it onto CUDA for benchmarking."""

  from diffusers import Flux2KleinPipeline

  pipe = Flux2KleinPipeline.from_pretrained(model_source, torch_dtype=torch_dtype)
  if hasattr(pipe, "set_progress_bar_config"):
    pipe.set_progress_bar_config(disable=True)
  pipe.to("cuda")
  return pipe


def compile_pipe(pipe):
  """Compile the pipeline's transformer with torch.compile when available."""

  if not hasattr(torch, "compile"):
    raise RuntimeError("torch.compile is unavailable on this PyTorch version.")
  if hasattr(torch, "compiler"):
    torch.compiler.reset()
  cache_dit.set_compile_configs()
  torch.set_float32_matmul_precision("high")
  # Try compile repeated blocks if avalible, otherwise compile the whole transformer.
  if hasattr(pipe.transformer, "compile_repeated_blocks"):
    pipe.transformer.compile_repeated_blocks()
    print("Compiled repeated blocks in transformer.")
  else:
    pipe.transformer = torch.compile(pipe.transformer)
    print("Compiled the whole transformer.")
  return pipe


def run_example(args: argparse.Namespace) -> dict[str, Path | str | None]:
  """Execute the full baseline -> PTQ -> load -> compile comparison workflow."""

  ensure_runtime_requirements()
  set_global_seed(args.seed)

  output_dir = Path(args.output_dir).expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  images_dir = output_dir / "images"
  reports_dir = output_dir / "reports"
  checkpoint_dir = output_dir / "checkpoint"
  report_path = reports_dir / "report.md"
  summary_path = reports_dir / "summary.json"

  prompts_path = Path(args.prompts_path).expanduser().resolve()
  calibration_prompts = load_calibration_prompts(prompts_path, args.calibration_limit)
  validation_prompt = build_random_short_prompt()

  calibration_height = args.calibration_height or args.height
  calibration_width = args.calibration_width or args.width
  calibration_steps = args.calibration_steps or args.num_inference_steps
  torch_dtype = resolve_torch_dtype()
  quant_type = f"svdq_int4_r{args.rank}"

  stage_results: dict[str, StageResult] = {}
  checkpoint_path: Optional[Path] = None
  quant_config_path: Optional[Path] = None
  quantization_time_s: Optional[float] = None
  quantization_peak_memory_gb: Optional[float] = None

  pipe = None
  loaded_pipe = None
  try:
    pipe = load_flux2_pipe(args.model_source, torch_dtype)
    stage_results["baseline"] = benchmark_stage(
      pipe,
      stage="baseline",
      prompt=validation_prompt,
      height=args.height,
      width=args.width,
      num_inference_steps=args.num_inference_steps,
      benchmark_runs=args.benchmark_runs,
      seed=args.seed,
      image_path=images_dir / "baseline.png",
    )
    exclude_layers = QuantizeConfig(
    ).exclude_layers if args.exclude_layers is None else args.exclude_layers

    quantize_config = QuantizeConfig(
      quant_type=quant_type,
      calibrate_fn=make_calibrate_fn(
        pipe,
        prompts=calibration_prompts,
        height=calibration_height,
        width=calibration_width,
        num_inference_steps=calibration_steps,
        seed=args.seed,
      ),
      serialize_to=str(checkpoint_dir),
      exclude_layers=exclude_layers,
      svdq_kwargs={
        **DEFAULT_SVDQ_KWARGS,
        "calibrate_precision": args.calibrate_precision,
      },
    )

    reset_cuda_peak_memory()
    quantize_start = time.perf_counter()
    pipe.transformer = cache_dit.quantize(pipe.transformer, quantize_config)
    quantization_time_s = time.perf_counter() - quantize_start
    if torch.cuda.is_available():
      torch.cuda.synchronize()
      quantization_peak_memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    else:
      quantization_peak_memory_gb = 0.0
    checkpoint_path = Path(quantize_config.serialize_to).resolve()
    quant_config_path = checkpoint_path.parent / "quant_config.json"
    pipe.to("cuda")

    stage_results["memory_quantized"] = benchmark_stage(
      pipe,
      stage="memory_quantized",
      prompt=validation_prompt,
      height=args.height,
      width=args.width,
      num_inference_steps=args.num_inference_steps,
      benchmark_runs=args.benchmark_runs,
      seed=args.seed,
      image_path=images_dir / "memory_quantized.png",
    )

    cleanup_pipe(pipe)
    pipe = None

    loaded_pipe = load_flux2_pipe(args.model_source, torch_dtype)
    loaded_pipe.transformer = cache_dit.load(loaded_pipe.transformer, str(checkpoint_path.parent))
    loaded_pipe.to("cuda")

    stage_results["loaded_quantized"] = benchmark_stage(
      loaded_pipe,
      stage="loaded_quantized",
      prompt=validation_prompt,
      height=args.height,
      width=args.width,
      num_inference_steps=args.num_inference_steps,
      benchmark_runs=args.benchmark_runs,
      seed=args.seed,
      image_path=images_dir / "loaded_quantized.png",
    )

    if args.skip_compile:
      stage_results["compiled_quantized"] = StageResult(
        stage="compiled_quantized",
        status="skipped (--skip-compile)",
      )
    elif not hasattr(torch, "compile"):
      stage_results["compiled_quantized"] = StageResult(
        stage="compiled_quantized",
        status="skipped (torch.compile unavailable)",
      )
    else:
      try:
        # TODO: We should also compile the baseline model before apply quantization to get
        # a more apples-to-apples comparison in the future (to get more accurate quantized
        # model with compilation-aware optimizations)
        loaded_pipe = compile_pipe(loaded_pipe)
        _compiled_warmup_image, warmup_latency_s, _warmup_peak_memory_gb = run_single_inference(
          loaded_pipe,
          prompt=validation_prompt,
          height=args.height,
          width=args.width,
          num_inference_steps=args.num_inference_steps,
          seed=args.seed,
        )
        compiled_result = benchmark_stage(
          loaded_pipe,
          stage="compiled_quantized",
          prompt=validation_prompt,
          height=args.height,
          width=args.width,
          num_inference_steps=args.num_inference_steps,
          benchmark_runs=args.benchmark_runs,
          seed=args.seed,
          image_path=images_dir / "compiled_quantized.png",
        )
        compiled_result.warmup_latency_s = warmup_latency_s
        compiled_result.status = f"ok (warmup_latency_s={warmup_latency_s:.4f})"
        stage_results["compiled_quantized"] = compiled_result
      except Exception as exc:
        stage_results["compiled_quantized"] = StageResult(
          stage="compiled_quantized",
          status=f"skipped ({type(exc).__name__}: {str(exc).splitlines()[0][:120]})",
        )

  finally:
    cleanup_pipe(loaded_pipe)
    cleanup_pipe(pipe)

  stage_order = [
    stage_results["baseline"],
    stage_results["memory_quantized"],
    stage_results["loaded_quantized"],
    stage_results["compiled_quantized"],
  ]

  psnr_rows: list[tuple[object, ...]] = []
  baseline_image_path = stage_results["baseline"].image_path
  assert baseline_image_path is not None
  for result in stage_order[1:]:
    if result.image_path is None:
      psnr_rows.append((f"baseline vs {result.stage}", "n/a", "n/a", "n/a", result.status))
      continue
    psnr, max_abs_diff, mean_abs_diff = compare_images(baseline_image_path, result.image_path)
    psnr_rows.append((
      f"baseline vs {result.stage}",
      format_float(psnr, digits=4),
      max_abs_diff,
      format_float(mean_abs_diff, digits=4),
      result.status,
    ))

  comparison_grid_path = make_visual_comparison(
    output_path=images_dir / "comparison_grid.png",
    validation_prompt=validation_prompt,
    stage_results=stage_order,
  )

  stage_rows: list[tuple[object, ...]] = []
  for result in stage_order:
    benchmark = result.benchmark
    stage_rows.append((
      result.stage,
      benchmark.run_count if benchmark is not None else 0,
      format_float(benchmark.avg_latency_s if benchmark is not None else None),
      format_float(benchmark.total_latency_s if benchmark is not None else None),
      format_float(benchmark.peak_memory_gb if benchmark is not None else None),
      format_float(benchmark.transformer_weight_cuda_gb if benchmark is not None else None),
      result.status,
    ))

  checkpoint_size_gb = None
  if checkpoint_path is not None and checkpoint_path.is_file():
    checkpoint_size_gb = checkpoint_path.stat().st_size / (1024 ** 3)

  metadata_rows = [
    ("model_source", args.model_source),
    ("output_dir", str(output_dir)),
    ("quant_type", quant_type),
    ("calibrate_precision", args.calibrate_precision),
    ("torch_dtype", str(torch_dtype).replace("torch.", "")),
    ("prompts_path", str(prompts_path)),
    ("calibration_prompt_count", len(calibration_prompts)),
    ("validation_prompt", validation_prompt),
    ("height", args.height),
    ("width", args.width),
    ("num_inference_steps", args.num_inference_steps),
    ("calibration_height", calibration_height),
    ("calibration_width", calibration_width),
    ("calibration_steps", calibration_steps),
    ("benchmark_runs", args.benchmark_runs),
    ("seed", args.seed),
    ("seed_policy", "fixed seed shared by calibration and validation/testing"),
    ("quantization_time_s", format_float(quantization_time_s)),
    ("quantization_peak_memory_gb", format_float(quantization_peak_memory_gb)),
    ("checkpoint_size_gb", format_float(checkpoint_size_gb)),
  ]

  artifact_rows = [
    ("checkpoint", relative_path(checkpoint_path, output_dir)),
    ("quant_config_json", relative_path(quant_config_path, output_dir)),
    ("baseline_image", relative_path(stage_results["baseline"].image_path, output_dir)),
    (
      "memory_quantized_image",
      relative_path(stage_results["memory_quantized"].image_path, output_dir),
    ),
    (
      "loaded_quantized_image",
      relative_path(stage_results["loaded_quantized"].image_path, output_dir),
    ),
    (
      "compiled_quantized_image",
      relative_path(stage_results["compiled_quantized"].image_path, output_dir),
    ),
    ("comparison_grid", relative_path(comparison_grid_path, output_dir)),
    ("report", relative_path(report_path, output_dir)),
    ("summary_json", relative_path(summary_path, output_dir)),
  ]

  persist_report(
    output_path=report_path,
    metadata_rows=metadata_rows,
    stage_rows=stage_rows,
    psnr_rows=psnr_rows,
    artifact_rows=artifact_rows,
  )
  persist_summary_json(
    output_path=summary_path,
    metadata_rows=metadata_rows,
    stage_order=stage_order,
    psnr_rows=psnr_rows,
    artifact_rows=artifact_rows,
  )

  print(f"Calibration prompts: {len(calibration_prompts)}")
  print(f"Validation prompt: {validation_prompt}")
  print(f"Quantized checkpoint: {checkpoint_path}")
  print(f"Report: {report_path}")
  print(f"Summary JSON: {summary_path}")
  if comparison_grid_path is not None:
    print(f"Comparison grid: {comparison_grid_path}")

  return {
    "output_dir": output_dir,
    "checkpoint_path": checkpoint_path,
    "quant_config_path": quant_config_path,
    "report_path": report_path,
    "summary_path": summary_path,
    "comparison_grid_path": comparison_grid_path,
  }


def main() -> None:
  """Parse CLI args and run the FLUX.2 SVDQ PTQ example."""

  args = parse_args()
  run_example(args)


if __name__ == "__main__":
  main()
  # Command line examples:
  # python3 flux2_svdq_ptq.py --calibration-limit 100 --benchmark-runs 2
  # python3 flux2_svdq_ptq.py --calibration-limit 200 --benchmark-runs 2
