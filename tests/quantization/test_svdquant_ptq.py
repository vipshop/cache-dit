from __future__ import annotations

import copy
import gc
import json
import math
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

import cache_dit
from cache_dit.kernels import svdq_extension_is_available
from cache_dit.quantization import QuantizeConfig
from cache_dit.quantization.svdquant import SVDQW4A4Linear
from tests.quantization._svdq_test_utils import compute_accuracy_metrics
from tests.quantization._svdq_test_utils import format_markdown_table
from tests.quantization._svdq_test_utils import make_token_batch
from tests.quantization._svdq_test_utils import make_token_samples
from tests.quantization._svdq_test_utils import make_toy_model
from tests.quantization._svdq_test_utils import runtime_dtype

_ENABLE_FLUX2_SLOW_TEST = os.getenv("CACHE_DIT_SVDQ_TEST_FLUX2", "0").lower() == "1"
_ENABLE_FLUX2_COMPILE_TEST = os.getenv("CACHE_DIT_SVDQ_TEST_FLUX2_COMPILE", "0").lower() == "1"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_TMP_DIR = _REPO_ROOT / ".tmp"
_FLUX2_MODEL_SOURCE_ENVVAR = "FLUX_2_KLEIN_4B_DIR"
_FLUX2_DEFAULT_MODEL_SOURCE = "black-forest-labs/FLUX.2-klein-4B"
_FLUX2_CPU_SEED = 0
_SVDQ_METADATA_KEY_NAME = "cache_dit_svdq_ptq"
_SVDQ_QUANT_CONFIG_FILENAME = "quant_config.json"
_DEFAULT_SVDQ_KWARGS = {
  "streaming": True,
  "calibrate_precision": "low",
  "runtime_kernel": "v1",
  "activation_buffer_flush_sample_count": 1,
  "activation_buffer_flush_cpu_bytes": None,
  "smooth_strategy": "activation",
}
_FLUX2_NUM_INFERENCE_STEPS = 4
_FLUX2_VISUAL_SAMPLE_COUNT = 3
_FLUX2_REFERENCE_PSNR_THRESHOLD = 10.0
_FLUX2_PROMPTS = [
  "A cute cat sitting on the beach, watching the sunset.",
  "A cute dog sitting on the beach, watching the sunset.",
  "A cute fox sitting on the beach, watching the sunset.",
  "A cute rabbit sitting on the beach, watching the sunset.",
  "A cute panda sitting on the beach, watching the sunset.",
  "A cute deer sitting on the beach, watching the sunset.",
  "A cute otter sitting on the beach, watching the sunset.",
  "A cute koala sitting on the beach, watching the sunset.",
]
_FLUX2_FULL_PROMPTS = _FLUX2_PROMPTS[:4]

pytestmark = pytest.mark.skipif(
  not torch.cuda.is_available() or not svdq_extension_is_available(),
  reason="SVDQ PTQ tests require CUDA and the optional SVDQuant extension.",
)


@dataclass(frozen=True)
class Flux2ExecutionPlan:
  height: int
  width: int
  offload_mode: str

  @property
  def resolution(self) -> str:
    return f"{self.height}x{self.width}"


@dataclass(frozen=True)
class Flux2StageBenchmark:
  stage: str
  resolution: str
  offload_mode: str
  prompt_count: int
  avg_latency_s: float
  total_latency_s: float
  peak_memory_gb: float
  transformer_weight_cuda_gb: float


@dataclass(frozen=True)
class Flux2LatencyBenchmark:
  stage: str
  resolution: str
  offload_mode: str
  run_count: int
  avg_latency_s: float
  total_latency_s: float
  peak_memory_gb: float


def _make_flux2_cpu_generator() -> torch.Generator:
  return torch.Generator(device="cpu").manual_seed(_FLUX2_CPU_SEED)


def _resolve_flux2_model_source() -> str:
  source = os.getenv(_FLUX2_MODEL_SOURCE_ENVVAR, _FLUX2_DEFAULT_MODEL_SOURCE)
  candidate = Path(source).expanduser()
  if candidate.exists():
    return str(candidate.resolve())
  return source


def _resolve_positive_int_envvar(name: str, default: int) -> int:
  raw_value = os.getenv(name)
  if raw_value is None or raw_value == "":
    return default
  try:
    value = int(raw_value)
  except ValueError as exc:
    raise ValueError(f"{name} must be a positive integer, got {raw_value!r}.") from exc
  if value < 1:
    raise ValueError(f"{name} must be a positive integer, got {value}.")
  return value


def _resolve_flux2_compile_bench_runs() -> int:
  return _resolve_positive_int_envvar("CACHE_DIT_SVDQ_TEST_FLUX2_COMPILE_BENCH_RUNS", 5)


def _resolve_flux2_case_dir(case_name: str, plan: Flux2ExecutionPlan) -> Path:
  return (_REPO_TMP_DIR / "tests" / "svdq_flux2" / case_name /
          f"{plan.resolution}_{plan.offload_mode}")


def _resolve_saved_flux2_checkpoint_dir(case_name: str, plan: Flux2ExecutionPlan) -> Path:
  checkpoint_dir = _resolve_flux2_case_dir(case_name, plan) / "checkpoint"
  quant_config_path = checkpoint_dir / _SVDQ_QUANT_CONFIG_FILENAME
  if not quant_config_path.is_file():
    pytest.skip("Saved FLUX2 quantized checkpoint not found. Run "
                "test_svdq_ptq_flux2_klein_full_quantization_benchmark_report first.")

  checkpoint_snapshot = json.loads(quant_config_path.read_text(encoding="utf-8"))
  checkpoint_path = checkpoint_dir / str(checkpoint_snapshot.get("checkpoint_path", ""))
  if not checkpoint_path.is_file():
    pytest.skip(f"Saved FLUX2 checkpoint weights are missing: {checkpoint_path}")
  return checkpoint_dir


def _ptq_tolerance() -> tuple[float, float]:
  return 6e-2, 2e-2


def _make_calibrate_fn(samples: list[torch.Tensor]):

  def calibrate_fn(**kwargs) -> None:
    model = kwargs["model"]
    with torch.inference_mode():
      for sample in samples:
        model(sample)

  return calibrate_fn


def _make_ptq_config(
  serialize_dir: Path,
  calibrate_fn,
  **overrides: object,
) -> QuantizeConfig:
  kwargs = dict(_DEFAULT_SVDQ_KWARGS)
  kwargs.update(overrides.pop("svdq_kwargs", {}))
  return QuantizeConfig(
    quant_type="svdq_int4_r32",
    calibrate_fn=calibrate_fn,
    serialize_to=str(serialize_dir),
    svdq_kwargs=kwargs,
    **overrides,
  )


def _resolve_quant_checkpoint_dir(serialize_to: str | Path) -> Path:
  return Path(serialize_to).parent


def _resolve_quant_config_path(serialize_to: str | Path) -> Path:
  return _resolve_quant_checkpoint_dir(serialize_to) / _SVDQ_QUANT_CONFIG_FILENAME


def _load_quant_config_snapshot(serialize_to: str | Path) -> dict[str, object]:
  return json.loads(_resolve_quant_config_path(serialize_to).read_text(encoding="utf-8"))


def _iter_repeated_block_scope(module: torch.nn.Module):
  repeated_block_names = list(getattr(module, "_repeated_blocks", []))
  if not repeated_block_names:
    yield from module.named_modules()
    return

  seen_names: set[str] = set()
  for block_name, block in module.named_modules():
    if not block_name or block.__class__.__name__ not in repeated_block_names:
      continue
    for local_name, submodule in block.named_modules():
      full_name = block_name if local_name == "" else f"{block_name}.{local_name}"
      if full_name in seen_names:
        continue
      seen_names.add(full_name)
      yield full_name, submodule


def _summarize_linear_modules_in_scope(module: torch.nn.Module) -> dict[str, object]:
  float_linear_names: list[str] = []
  quantized_linear_names: list[str] = []
  for module_name, submodule in _iter_repeated_block_scope(module):
    if isinstance(submodule, SVDQW4A4Linear):
      quantized_linear_names.append(module_name)
    elif isinstance(submodule, torch.nn.Linear):
      float_linear_names.append(module_name)

  return {
    "float_linear_names": float_linear_names,
    "quantized_linear_names": quantized_linear_names,
    "float_linear_count": len(float_linear_names),
    "quantized_linear_count": len(quantized_linear_names),
  }


def _flux2_execution_plans() -> list[Flux2ExecutionPlan]:
  return [
    Flux2ExecutionPlan(height=1024, width=1024, offload_mode="cuda"),
    Flux2ExecutionPlan(height=512, width=512, offload_mode="cuda"),
    Flux2ExecutionPlan(height=512, width=512, offload_mode="model_cpu_offload"),
  ]


def _is_cuda_oom(exc: BaseException) -> bool:
  if isinstance(exc, torch.OutOfMemoryError):
    return True
  if not isinstance(exc, RuntimeError):
    return False
  message = str(exc).lower()
  return "out of memory" in message or "cuda error: out of memory" in message


def _cleanup_flux2_pipe(pipe) -> None:
  if pipe is None:
    return
  try:
    if hasattr(pipe, "remove_all_hooks"):
      pipe.remove_all_hooks()
  except Exception:
    pass
  try:
    pipe.to("cpu")
  except Exception:
    pass
  del pipe
  gc.collect()
  if torch.cuda.is_available():
    torch.cuda.empty_cache()


def _apply_flux2_execution_plan(pipe, plan: Flux2ExecutionPlan) -> None:
  if hasattr(pipe, "remove_all_hooks"):
    pipe.remove_all_hooks()

  if plan.offload_mode == "cuda":
    pipe.to("cuda")
    return

  if plan.offload_mode == "model_cpu_offload":
    try:
      pipe.enable_model_cpu_offload()
    except ImportError as exc:
      pytest.skip(f"Model CPU offload is unavailable in this environment: {exc}")
    return

  raise ValueError(f"Unsupported FLUX2 execution plan offload mode: {plan.offload_mode}.")


def _compute_module_cuda_tensor_bytes(module: torch.nn.Module) -> int:
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


def _format_gib(value_gib: float) -> str:
  return f"{value_gib:.3f} GiB"


def _run_flux2_single_inference_and_peak_memory_gib(
  pipe,
  *,
  prompt: str,
  plan: Flux2ExecutionPlan,
) -> tuple[object, float]:
  image, peak_memory_gib, _ = _run_flux2_single_inference_profile(
    pipe,
    prompt=prompt,
    plan=plan,
  )
  return image, peak_memory_gib


def _run_flux2_single_inference_profile(
  pipe,
  *,
  prompt: str,
  plan: Flux2ExecutionPlan,
) -> tuple[object, float, float]:
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

  start_time = time.perf_counter()
  with torch.inference_mode():
    image = pipe(
      prompt=prompt,
      height=plan.height,
      width=plan.width,
      num_inference_steps=_FLUX2_NUM_INFERENCE_STEPS,
      generator=_make_flux2_cpu_generator(),
    ).images[0]

  if torch.cuda.is_available():
    torch.cuda.synchronize()
    peak_memory_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
  else:
    peak_memory_gib = 0.0
  latency_s = time.perf_counter() - start_time
  return image, peak_memory_gib, latency_s


def _measure_flux2_latency_benchmark(
  pipe,
  *,
  prompt: str,
  plan: Flux2ExecutionPlan,
  stage: str,
  run_count: int,
) -> tuple[object, Flux2LatencyBenchmark]:
  if run_count < 1:
    raise ValueError(f"run_count must be positive, got {run_count}.")

  latencies: list[float] = []
  peak_memory_gb = 0.0
  image = None
  for _ in range(run_count):
    image, run_peak_memory_gb, latency_s = _run_flux2_single_inference_profile(
      pipe,
      prompt=prompt,
      plan=plan,
    )
    latencies.append(latency_s)
    peak_memory_gb = max(peak_memory_gb, run_peak_memory_gb)

  assert image is not None
  total_latency_s = sum(latencies)
  return image, Flux2LatencyBenchmark(
    stage=stage,
    resolution=plan.resolution,
    offload_mode=plan.offload_mode,
    run_count=run_count,
    avg_latency_s=total_latency_s / run_count,
    total_latency_s=total_latency_s,
    peak_memory_gb=peak_memory_gb,
  )


def _run_flux2_generation_stage(
  pipe,
  prompts: list[str],
  output_dir: Path,
  *,
  plan: Flux2ExecutionPlan,
  stage: str,
) -> tuple[Flux2StageBenchmark, list[Path]]:
  output_dir.mkdir(parents=True, exist_ok=True)
  latencies: list[float] = []
  peak_memory_bytes = 0
  image_paths: list[Path] = []
  transformer_weight_cuda_bytes = _compute_module_cuda_tensor_bytes(pipe.transformer)

  with torch.inference_mode():
    for index, prompt in enumerate(prompts):
      if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

      start_time = time.perf_counter()
      image = pipe(
        prompt=prompt,
        height=plan.height,
        width=plan.width,
        num_inference_steps=_FLUX2_NUM_INFERENCE_STEPS,
        generator=_make_flux2_cpu_generator(),
      ).images[0]
      if torch.cuda.is_available():
        torch.cuda.synchronize()

      latency_s = time.perf_counter() - start_time
      latencies.append(latency_s)
      if torch.cuda.is_available():
        peak_memory_bytes = max(peak_memory_bytes, torch.cuda.max_memory_allocated())

      image_path = output_dir / f"prompt_{index}.png"
      image.save(image_path)
      image_paths.append(image_path)

  benchmark = Flux2StageBenchmark(
    stage=stage,
    resolution=plan.resolution,
    offload_mode=plan.offload_mode,
    prompt_count=len(prompts),
    avg_latency_s=sum(latencies) / len(latencies),
    total_latency_s=sum(latencies),
    peak_memory_gb=peak_memory_bytes / (1024 ** 3),
    transformer_weight_cuda_gb=transformer_weight_cuda_bytes / (1024 ** 3),
  )
  return benchmark, image_paths


def _compare_image_dirs(
  dir_a: Path,
  dir_b: Path,
  prompts: list[str],
  *,
  label_a: str,
  label_b: str,
) -> tuple[list[tuple[object, ...]], dict[str, float]]:
  from cache_dit.metrics.metrics import compute_psnr_file

  rows: list[tuple[object, ...]] = []
  psnrs: list[float] = []
  max_abs_diffs: list[int] = []
  mean_abs_diffs: list[float] = []
  for index, _prompt in enumerate(prompts):
    path_a = dir_a / f"prompt_{index}.png"
    path_b = dir_b / f"prompt_{index}.png"
    image_a = cv2.imread(str(path_a), cv2.IMREAD_COLOR)
    image_b = cv2.imread(str(path_b), cv2.IMREAD_COLOR)
    if image_a is None or image_b is None:
      raise FileNotFoundError(f"Missing FLUX2 comparison image pair: {path_a}, {path_b}.")

    abs_diff = cv2.absdiff(image_a, image_b)
    max_abs_diff = int(abs_diff.max())
    mean_abs_diff = float(abs_diff.mean())
    psnr = float(compute_psnr_file(str(path_a), str(path_b)))

    psnrs.append(psnr)
    max_abs_diffs.append(max_abs_diff)
    mean_abs_diffs.append(mean_abs_diff)
    rows.append((
      index,
      f"{label_a} vs {label_b}",
      "inf" if math.isinf(psnr) else f"{psnr:.4f}",
      max_abs_diff,
      f"{mean_abs_diff:.4f}",
    ))

  finite_psnrs = [value for value in psnrs if math.isfinite(value)]
  avg_psnr = float("inf") if not finite_psnrs and psnrs else sum(finite_psnrs) / len(finite_psnrs)
  avg_psnr_str = "inf" if math.isinf(avg_psnr) else f"{avg_psnr:.4f}"
  max_abs_diff = max(max_abs_diffs) if max_abs_diffs else 0
  mean_abs_diff = sum(mean_abs_diffs) / len(mean_abs_diffs) if mean_abs_diffs else 0.0
  rows.append(
    ("avg", f"{label_a} vs {label_b}", avg_psnr_str, max_abs_diff, f"{mean_abs_diff:.4f}"))
  return rows, {
    "avg_psnr": avg_psnr,
    "max_abs_diff": float(max_abs_diff),
    "mean_abs_diff": mean_abs_diff,
  }


def _build_reference_psnr_rows(
  reference_dir: Path,
  memory_quantized_dir: Path,
  loaded_dir: Path,
  prompts: list[str],
) -> tuple[list[tuple[object, ...]], dict[str, float]]:
  from cache_dit.metrics.metrics import compute_psnr_file

  memory_psnrs: list[float] = []
  loaded_psnrs: list[float] = []
  rows: list[tuple[object, ...]] = []
  for index, prompt in enumerate(prompts):
    reference_path = reference_dir / f"prompt_{index}.png"
    memory_path = memory_quantized_dir / f"prompt_{index}.png"
    loaded_path = loaded_dir / f"prompt_{index}.png"
    memory_psnr = float(compute_psnr_file(str(reference_path), str(memory_path)))
    loaded_psnr = float(compute_psnr_file(str(reference_path), str(loaded_path)))
    memory_psnrs.append(memory_psnr)
    loaded_psnrs.append(loaded_psnr)
    rows.append((
      index,
      prompt,
      "inf" if math.isinf(memory_psnr) else f"{memory_psnr:.4f}",
      "inf" if math.isinf(loaded_psnr) else f"{loaded_psnr:.4f}",
    ))

  memory_avg = sum(memory_psnrs) / len(memory_psnrs)
  loaded_avg = sum(loaded_psnrs) / len(loaded_psnrs)
  rows.append(("avg", f"count={len(prompts)}", f"{memory_avg:.4f}", f"{loaded_avg:.4f}"))
  return rows, {"memory_avg_psnr": memory_avg, "loaded_avg_psnr": loaded_avg}


def _load_flux2_image_tensor(path: Path) -> torch.Tensor:
  image = cv2.imread(str(path), cv2.IMREAD_COLOR)
  if image is None:
    raise FileNotFoundError(f"Failed to load FLUX2 image: {path}.")
  return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).contiguous()


def _build_flux2_acc_metric_rows(
  reference_dir: Path,
  memory_quantized_dir: Path,
  loaded_dir: Path,
  prompts: list[str],
) -> tuple[list[tuple[object, ...]], dict[str, float]]:
  rows: list[tuple[object, ...]] = []
  memory_metrics = []
  loaded_metrics = []
  # float_reference: original full-precision output from the model
  # memory_quantized: output from the in-memory quantized model without serialization
  # loaded_quantized: output from the quantized model after saving and loading from disk
  for index, _prompt in enumerate(prompts):
    reference_tensor = _load_flux2_image_tensor(reference_dir / f"prompt_{index}.png")
    memory_tensor = _load_flux2_image_tensor(memory_quantized_dir / f"prompt_{index}.png")
    loaded_tensor = _load_flux2_image_tensor(loaded_dir / f"prompt_{index}.png")

    memory_metric = compute_accuracy_metrics(reference_tensor, memory_tensor)
    loaded_metric = compute_accuracy_metrics(reference_tensor, loaded_tensor)
    memory_metrics.append(memory_metric)
    loaded_metrics.append(loaded_metric)

    rows.append((
      index,
      "float_reference vs memory_quantized",
      f"{memory_metric.mae:.6f}",
      f"{memory_metric.rmse:.6f}",
      f"{memory_metric.max_abs:.6f}",
      f"{memory_metric.rel_l2:.6f}",
      f"{memory_metric.cosine:.6f}",
    ))
    rows.append((
      index,
      "float_reference vs loaded_quantized",
      f"{loaded_metric.mae:.6f}",
      f"{loaded_metric.rmse:.6f}",
      f"{loaded_metric.max_abs:.6f}",
      f"{loaded_metric.rel_l2:.6f}",
      f"{loaded_metric.cosine:.6f}",
    ))

  def _avg(metrics: list[object], name: str) -> float:
    return sum(getattr(metric, name) for metric in metrics) / len(metrics)

  rows.append((
    "avg",
    "float_reference vs memory_quantized",
    f"{_avg(memory_metrics, 'mae'):.6f}",
    f"{_avg(memory_metrics, 'rmse'):.6f}",
    f"{_avg(memory_metrics, 'max_abs'):.6f}",
    f"{_avg(memory_metrics, 'rel_l2'):.6f}",
    f"{_avg(memory_metrics, 'cosine'):.6f}",
  ))
  rows.append((
    "avg",
    "float_reference vs loaded_quantized",
    f"{_avg(loaded_metrics, 'mae'):.6f}",
    f"{_avg(loaded_metrics, 'rmse'):.6f}",
    f"{_avg(loaded_metrics, 'max_abs'):.6f}",
    f"{_avg(loaded_metrics, 'rel_l2'):.6f}",
    f"{_avg(loaded_metrics, 'cosine'):.6f}",
  ))
  return rows, {
    "memory_avg_rel_l2": _avg(memory_metrics, "rel_l2"),
    "loaded_avg_rel_l2": _avg(loaded_metrics, "rel_l2"),
    "memory_avg_cosine": _avg(memory_metrics, "cosine"),
    "loaded_avg_cosine": _avg(loaded_metrics, "cosine"),
  }


def _make_flux2_comparison_canvas(
  *,
  case_name: str,
  prompt_index: int,
  prompt: str,
  plan: Flux2ExecutionPlan,
  stage_benchmarks: tuple[Flux2StageBenchmark, Flux2StageBenchmark, Flux2StageBenchmark],
  reference_path: Path,
  memory_quantized_path: Path,
  loaded_path: Path,
) -> np.ndarray:
  images = [
    cv2.imread(str(reference_path), cv2.IMREAD_COLOR),
    cv2.imread(str(memory_quantized_path), cv2.IMREAD_COLOR),
    cv2.imread(str(loaded_path), cv2.IMREAD_COLOR),
  ]
  if any(image is None for image in images):
    raise FileNotFoundError(f"Failed to load FLUX2 comparison images for prompt_{prompt_index}: "
                            f"{reference_path}, {memory_quantized_path}, {loaded_path}.")

  panel_width = 384
  panel_height = 384
  panel_header_height = 84
  labels = ["float_reference", "memory_quantized", "loaded_quantized"]
  labeled_panels: list[np.ndarray] = []
  for image, label, benchmark in zip(images, labels, stage_benchmarks):
    resized = cv2.resize(image, (panel_width, panel_height), interpolation=cv2.INTER_AREA)
    panel = np.full(
      (panel_height + panel_header_height, panel_width, 3),
      255,
      dtype=np.uint8,
    )
    panel[panel_header_height:, :, :] = resized
    cv2.putText(
      panel,
      label,
      (12, 24),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.7,
      (0, 0, 0),
      2,
      cv2.LINE_AA,
    )
    cv2.putText(
      panel,
      f"avg infer: {benchmark.avg_latency_s:.3f}s",
      (12, 49),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.54,
      (45, 45, 45),
      1,
      cv2.LINE_AA,
    )
    cv2.putText(
      panel,
      f"tfmr cuda mem: {_format_gib(benchmark.transformer_weight_cuda_gb)}",
      (12, 72),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.54,
      (45, 45, 45),
      1,
      cv2.LINE_AA,
    )
    labeled_panels.append(panel)

  content = cv2.hconcat(labeled_panels)
  title_height = 72
  canvas = np.full(
    (content.shape[0] + title_height, content.shape[1], 3),
    255,
    dtype=np.uint8,
  )
  canvas[title_height:, :, :] = content
  title = f"{case_name} | prompt_{prompt_index} | {plan.resolution} | {plan.offload_mode}"
  subtitle = prompt[:110] + ("..." if len(prompt) > 110 else "")
  cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
  cv2.putText(canvas, subtitle, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1,
              cv2.LINE_AA)
  return canvas


def _save_flux2_visual_comparisons(
  *,
  output_dir: Path,
  case_name: str,
  prompts: list[str],
  plan: Flux2ExecutionPlan,
  stage_benchmarks: tuple[Flux2StageBenchmark, Flux2StageBenchmark, Flux2StageBenchmark],
  reference_dir: Path,
  memory_quantized_dir: Path,
  loaded_dir: Path,
) -> list[Path]:
  output_dir.mkdir(parents=True, exist_ok=True)

  saved_paths: list[Path] = []
  for index, prompt in enumerate(prompts[:_FLUX2_VISUAL_SAMPLE_COUNT]):
    canvas = _make_flux2_comparison_canvas(
      case_name=case_name,
      prompt_index=index,
      prompt=prompt,
      plan=plan,
      stage_benchmarks=stage_benchmarks,
      reference_path=reference_dir / f"prompt_{index}.png",
      memory_quantized_path=memory_quantized_dir / f"prompt_{index}.png",
      loaded_path=loaded_dir / f"prompt_{index}.png",
    )
    output_path = output_dir / f"prompt_{index}.png"
    cv2.imwrite(str(output_path), canvas)
    saved_paths.append(output_path)
  return saved_paths


def _partial_flux2_exclude_layers(transformer) -> list[str]:
  return ["single_transformer_blocks"] + [
    f"transformer_blocks.{index}" for index in range(2, len(transformer.transformer_blocks))
  ]


def _full_flux2_exclude_layers(_transformer) -> list[str]:
  return []


def _benchmark_rows_to_table_rows(rows: list[Flux2StageBenchmark]) -> list[tuple[object, ...]]:
  return [(
    row.stage,
    row.resolution,
    row.offload_mode,
    row.prompt_count,
    f"{row.avg_latency_s:.4f}",
    f"{row.total_latency_s:.4f}",
    f"{row.peak_memory_gb:.4f}",
    f"{row.transformer_weight_cuda_gb:.4f}",
  ) for row in rows]


def _format_speedup(speedup: float) -> str:
  return "infx" if not math.isfinite(speedup) else f"{speedup:.4f}x"


def _compile_benchmark_rows_to_table_rows(
  eager_benchmark: Flux2LatencyBenchmark,
  rows: list[Flux2LatencyBenchmark],
) -> list[tuple[object, ...]]:
  table_rows: list[tuple[object, ...]] = []
  for row in rows:
    if row.stage == eager_benchmark.stage:
      delta_vs_eager_s = 0.0
      speedup_vs_eager = 1.0
    else:
      delta_vs_eager_s = row.avg_latency_s - eager_benchmark.avg_latency_s
      speedup_vs_eager = (eager_benchmark.avg_latency_s /
                          row.avg_latency_s if row.avg_latency_s > 0.0 else float("inf"))
    table_rows.append((
      row.stage,
      row.resolution,
      row.offload_mode,
      row.run_count,
      f"{row.avg_latency_s:.4f}",
      f"{row.total_latency_s:.4f}",
      f"{row.peak_memory_gb:.4f}",
      f"{delta_vs_eager_s:.4f}",
      _format_speedup(speedup_vs_eager),
    ))
  return table_rows


def _persist_flux2_report(report_dir: Path, filename: str, content: str) -> Path:
  report_dir.mkdir(parents=True, exist_ok=True)
  report_path = report_dir / filename
  report_path.write_text(content + "\n", encoding="utf-8")
  return report_path


def _emit_flux2_report(
  *,
  report_dir: Path,
  filename: str,
  title: str,
  headers: tuple[str, ...],
  rows: list[tuple[object, ...]],
) -> Path:
  report = format_markdown_table(title, headers, rows)
  print(report)
  return _persist_flux2_report(report_dir, filename, report)


def _run_flux2_svdq_case(
  *,
  case_name: str,
  prompts: list[str],
  exclude_layers_factory,
  require_all_quantized: bool,
) -> dict[str, object]:
  from diffusers import Flux2KleinPipeline

  model_source = _resolve_flux2_model_source()

  plan_rows: list[tuple[object, ...]] = []
  last_oom: BaseException | None = None
  last_case_dir: Path | None = None
  for attempt, plan in enumerate(_flux2_execution_plans(), start=1):
    pipe = None
    loaded_pipe = None
    case_dir = _resolve_flux2_case_dir(case_name, plan)
    last_case_dir = case_dir
    shutil.rmtree(case_dir, ignore_errors=True)
    reference_dir = case_dir / "reference"
    memory_quantized_dir = case_dir / "memory_quantized"
    loaded_dir = case_dir / "loaded_quantized"
    try:
      pipe = Flux2KleinPipeline.from_pretrained(model_source, torch_dtype=torch.bfloat16)
      _apply_flux2_execution_plan(pipe, plan)
      reference_benchmark, _ = _run_flux2_generation_stage(
        pipe,
        prompts,
        reference_dir,
        plan=plan,
        stage="float_reference",
      )

      exclude_layers = exclude_layers_factory(pipe.transformer)
      config = QuantizeConfig(
        quant_type="svdq_int4_r32",
        calibrate_fn=lambda **_: None,
        serialize_to=str(case_dir / "checkpoint"),
        exclude_layers=exclude_layers,
        svdq_kwargs={
          "streaming": True,
          "calibrate_precision": "medium",
        },
      )

      pre_quant_summary = _summarize_linear_modules_in_scope(pipe.transformer)
      assert pre_quant_summary["float_linear_count"] > 0

      def calibrate_fn(**_: object) -> None:
        with torch.inference_mode():
          for index, prompt in enumerate(prompts):
            _ = pipe(
              prompt=prompt,
              height=plan.height,
              width=plan.width,
              num_inference_steps=_FLUX2_NUM_INFERENCE_STEPS,
              generator=_make_flux2_cpu_generator(),
            )

      config.calibrate_fn = calibrate_fn
      pipe.transformer = cache_dit.quantize(pipe.transformer, config)
      assert Path(config.serialize_to).is_file()
      checkpoint_dir = _resolve_quant_checkpoint_dir(config.serialize_to)
      quant_config_path = _resolve_quant_config_path(config.serialize_to)
      assert checkpoint_dir == case_dir / "checkpoint"
      assert quant_config_path.is_file()

      quantized_layer_names = tuple(getattr(pipe.transformer, "_svdq_quantized_layers", ()))
      assert quantized_layer_names
      post_quant_summary = _summarize_linear_modules_in_scope(pipe.transformer)
      assert post_quant_summary["quantized_linear_count"] == len(quantized_layer_names)
      if require_all_quantized:
        assert post_quant_summary["float_linear_count"] == 0
      else:
        assert len(quantized_layer_names) < pre_quant_summary["float_linear_count"]
        assert post_quant_summary["float_linear_count"] > 0

      _apply_flux2_execution_plan(pipe, plan)
      memory_benchmark, _ = _run_flux2_generation_stage(
        pipe,
        prompts,
        memory_quantized_dir,
        plan=plan,
        stage="memory_quantized",
      )

      _cleanup_flux2_pipe(pipe)
      pipe = None

      loaded_pipe = Flux2KleinPipeline.from_pretrained(model_source, torch_dtype=torch.bfloat16)
      _apply_flux2_execution_plan(loaded_pipe, plan)
      loaded_pipe.transformer = cache_dit.load(
        loaded_pipe.transformer,
        str(checkpoint_dir),
      )
      _apply_flux2_execution_plan(loaded_pipe, plan)
      assert getattr(loaded_pipe.transformer, "_is_quantized", False)
      assert (tuple(getattr(loaded_pipe.transformer, "_svdq_quantized_layers",
                            ())) == quantized_layer_names)

      loaded_benchmark, _ = _run_flux2_generation_stage(
        loaded_pipe,
        prompts,
        loaded_dir,
        plan=plan,
        stage="loaded_quantized",
      )

      reference_psnr_rows, reference_psnr_summary = _build_reference_psnr_rows(
        reference_dir,
        memory_quantized_dir,
        loaded_dir,
        prompts,
      )
      acc_metric_rows, acc_metric_summary = _build_flux2_acc_metric_rows(
        reference_dir,
        memory_quantized_dir,
        loaded_dir,
        prompts,
      )
      consistency_rows, consistency_summary = _compare_image_dirs(
        memory_quantized_dir,
        loaded_dir,
        prompts,
        label_a="memory_quantized",
        label_b="loaded_quantized",
      )

      assert reference_psnr_summary["loaded_avg_psnr"] > _FLUX2_REFERENCE_PSNR_THRESHOLD
      assert consistency_summary["avg_psnr"] > _FLUX2_REFERENCE_PSNR_THRESHOLD

      plan_rows.append((attempt, plan.resolution, plan.offload_mode, "selected"))
      benchmark_rows = [reference_benchmark, memory_benchmark, loaded_benchmark]
      reports_dir = case_dir / "reports"
      report_paths = {
        "execution_plan":
        _emit_flux2_report(
          report_dir=reports_dir,
          filename="execution_plan.md",
          title=f"SVDQ PTQ {case_name} execution plan",
          headers=("attempt", "resolution", "offload", "status"),
          rows=plan_rows,
        ),
        "benchmark":
        _emit_flux2_report(
          report_dir=reports_dir,
          filename="benchmark.md",
          title=f"SVDQ PTQ {case_name} benchmark",
          headers=(
            "stage",
            "resolution",
            "offload",
            "prompt_count",
            "avg_latency_s",
            "total_latency_s",
            "peak_memory_gb",
            "transformer_weight_cuda_gb",
          ),
          rows=_benchmark_rows_to_table_rows(benchmark_rows),
        ),
        "psnr":
        _emit_flux2_report(
          report_dir=reports_dir,
          filename="psnr.md",
          title=f"SVDQ PTQ {case_name} PSNR report",
          headers=("prompt_index", "prompt", "memory_quantized_psnr", "loaded_psnr"),
          rows=reference_psnr_rows,
        ),
        "consistency":
        _emit_flux2_report(
          report_dir=reports_dir,
          filename="consistency.md",
          title=f"SVDQ PTQ {case_name} memory-vs-loaded consistency",
          headers=("prompt_index", "comparison", "psnr", "max_abs_diff", "mean_abs_diff"),
          rows=consistency_rows,
        ),
        "acc_metrics":
        _emit_flux2_report(
          report_dir=reports_dir,
          filename="acc_metrics.md",
          title=f"SVDQ PTQ {case_name} acc metrics",
          headers=(
            "prompt_index",
            "comparison",
            "mae",
            "rmse",
            "max_abs",
            "rel_l2",
            "cosine",
          ),
          rows=acc_metric_rows,
        ),
      }

      visual_paths = _save_flux2_visual_comparisons(
        output_dir=case_dir / "visuals",
        case_name=case_name,
        prompts=prompts,
        plan=plan,
        stage_benchmarks=(reference_benchmark, memory_benchmark, loaded_benchmark),
        reference_dir=reference_dir,
        memory_quantized_dir=memory_quantized_dir,
        loaded_dir=loaded_dir,
      )
      return {
        "case_dir": case_dir,
        "plan": plan,
        "checkpoint_dir": checkpoint_dir,
        "quant_config_path": quant_config_path,
        "report_paths": report_paths,
        "visual_paths": visual_paths,
        "acc_metric_summary": acc_metric_summary,
        "reference_psnr_summary": reference_psnr_summary,
        "consistency_summary": consistency_summary,
      }
    except Exception as exc:
      if not _is_cuda_oom(exc):
        raise
      last_oom = exc
      plan_rows.append((
        attempt,
        plan.resolution,
        plan.offload_mode,
        f"oom -> {str(exc).splitlines()[0][:80]}",
      ))
    finally:
      _cleanup_flux2_pipe(loaded_pipe)
      _cleanup_flux2_pipe(pipe)

  execution_plan_report = format_markdown_table(
    f"SVDQ PTQ {case_name} execution plan",
    ("attempt", "resolution", "offload", "status"),
    plan_rows,
  )
  print(execution_plan_report)
  if last_case_dir is not None:
    _persist_flux2_report(last_case_dir / "reports", "execution_plan.md", execution_plan_report)
  if last_oom is not None:
    raise last_oom
  raise RuntimeError(f"FLUX2 SVDQ PTQ case {case_name} did not execute any plan.")


def test_svdq_ptq_public_api_flush_thresholds_preserve_quantized_outputs(tmp_path: Path) -> None:
  dtype = runtime_dtype()
  base_model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=101,
    device="cuda",
    dtype=dtype,
  )
  calibration_samples = make_token_samples(
    num_samples=4,
    batch_size=1,
    seq_len=16,
    width=128,
    seed=103,
    device="cuda",
    dtype=dtype,
  )

  baseline_config = _make_ptq_config(
    tmp_path / "baseline_public_api",
    _make_calibrate_fn(calibration_samples),
  )
  buffered_config = _make_ptq_config(
    tmp_path / "buffered_public_api",
    _make_calibrate_fn(calibration_samples),
    svdq_kwargs={
      "activation_buffer_flush_sample_count": 2,
      "activation_buffer_flush_cpu_bytes": 256,
    },
  )

  baseline_quantized = cache_dit.quantize(copy.deepcopy(base_model), baseline_config)
  buffered_quantized = cache_dit.quantize(copy.deepcopy(base_model), buffered_config)
  assert _resolve_quant_config_path(baseline_config.serialize_to).is_file()
  assert _resolve_quant_config_path(buffered_config.serialize_to).is_file()

  eval_inputs = make_token_batch(
    batch_size=2,
    seq_len=12,
    width=128,
    seed=107,
    device="cuda",
    dtype=dtype,
  )
  with torch.inference_mode():
    baseline_output = baseline_quantized(eval_inputs)
    buffered_output = buffered_quantized(eval_inputs)
    torch.cuda.synchronize()

  atol, rtol = _ptq_tolerance()
  torch.testing.assert_close(buffered_output, baseline_output, rtol=rtol, atol=atol)
  assert tuple(getattr(buffered_quantized, "_svdq_quantized_layers",
                       ())) == tuple(getattr(baseline_quantized, "_svdq_quantized_layers", ()))

  fresh_baseline_model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=101,
    device="cuda",
    dtype=dtype,
  )
  fresh_buffered_model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=101,
    device="cuda",
    dtype=dtype,
  )
  loaded_baseline = cache_dit.load(
    fresh_baseline_model,
    str(_resolve_quant_checkpoint_dir(baseline_config.serialize_to)),
  )
  loaded_buffered = cache_dit.load(
    fresh_buffered_model,
    buffered_config.serialize_to,
  )

  with torch.inference_mode():
    loaded_baseline_output = loaded_baseline(eval_inputs)
    loaded_buffered_output = loaded_buffered(eval_inputs)
    torch.cuda.synchronize()

  torch.testing.assert_close(loaded_baseline_output, baseline_output, rtol=rtol, atol=atol)
  torch.testing.assert_close(loaded_buffered_output, buffered_output, rtol=rtol, atol=atol)


def test_svdq_ptq_quantize_toy_model_replaces_linear_layers_and_serializes(
  tmp_path: Path, ) -> None:
  dtype = runtime_dtype()
  float_model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=7,
    device="cuda",
    dtype=dtype,
  )
  reference_model = copy.deepcopy(float_model).eval()
  calibration_samples = make_token_samples(
    num_samples=4,
    batch_size=1,
    seq_len=32,
    width=128,
    seed=19,
    device="cuda",
    dtype=dtype,
  )
  config = _make_ptq_config(tmp_path / "toy_ptq", _make_calibrate_fn(calibration_samples))

  quantized_model = cache_dit.quantize(float_model, config)

  assert isinstance(quantized_model.block.to_q, SVDQW4A4Linear)
  assert isinstance(quantized_model.block.to_k, SVDQW4A4Linear)
  assert isinstance(quantized_model.block.to_v, SVDQW4A4Linear)
  assert isinstance(quantized_model.block.to_out, SVDQW4A4Linear)
  assert getattr(quantized_model, "_is_quantized", False)
  assert Path(config.serialize_to).is_file()
  quant_config_snapshot = _load_quant_config_snapshot(config.serialize_to)
  assert quant_config_snapshot["quant_type"] == config.quant_type
  assert quant_config_snapshot["checkpoint_path"] == Path(config.serialize_to).name
  assert quant_config_snapshot["svdq_kwargs"] == config.get_svdq_kwargs()

  eval_inputs = make_token_batch(
    batch_size=2,
    seq_len=16,
    width=128,
    seed=23,
    device="cuda",
    dtype=dtype,
  )
  with torch.inference_mode():
    reference = reference_model(eval_inputs)
    candidate = quantized_model(eval_inputs)
    torch.cuda.synchronize()
  metrics = compute_accuracy_metrics(reference, candidate)
  assert metrics.rel_l2 < 0.1
  assert metrics.cosine > 0.99


def test_svdq_ptq_quantize_root_linear_and_load_roundtrip(tmp_path: Path) -> None:
  dtype = runtime_dtype()
  float_linear = torch.nn.Linear(128, 128, bias=True, device="cuda", dtype=dtype).eval()
  reference_linear = copy.deepcopy(float_linear).eval()
  calibration_samples = make_token_samples(
    num_samples=3,
    batch_size=1,
    seq_len=24,
    width=128,
    seed=53,
    device="cuda",
    dtype=dtype,
  )
  config = _make_ptq_config(
    tmp_path / "root_linear",
    _make_calibrate_fn(calibration_samples),
    svdq_kwargs={"runtime_kernel": "v2"},
  )

  quantized_linear = cache_dit.quantize(float_linear, config)
  assert isinstance(quantized_linear, SVDQW4A4Linear)
  assert quantized_linear.runtime_kernel == "v2"
  assert Path(config.serialize_to).is_file()

  eval_inputs = make_token_batch(
    batch_size=2,
    seq_len=12,
    width=128,
    seed=59,
    device="cuda",
    dtype=dtype,
  )
  fresh_linear = torch.nn.Linear(128, 128, bias=True, device="cuda", dtype=dtype).eval()
  fresh_linear.load_state_dict(reference_linear.state_dict())
  loaded_linear = cache_dit.load(
    fresh_linear,
    str(_resolve_quant_checkpoint_dir(config.serialize_to)),
  )
  assert isinstance(loaded_linear, SVDQW4A4Linear)
  assert loaded_linear.runtime_kernel == "v2"

  with torch.inference_mode():
    quantized_output = quantized_linear(eval_inputs)
    loaded_output = loaded_linear(eval_inputs)
    reference_output = reference_linear(eval_inputs)
    torch.cuda.synchronize()

  atol, rtol = _ptq_tolerance()
  torch.testing.assert_close(loaded_output, quantized_output, rtol=rtol, atol=atol)
  metrics = compute_accuracy_metrics(reference_output, quantized_output)
  assert metrics.rel_l2 < 0.12
  assert metrics.cosine > 0.99


def test_svdq_ptq_exclude_layers_keeps_non_target_linears_float(tmp_path: Path) -> None:
  dtype = runtime_dtype()
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=13,
    device="cuda",
    dtype=dtype,
  )
  calibration_samples = make_token_samples(
    num_samples=2,
    batch_size=1,
    seq_len=16,
    width=128,
    seed=31,
    device="cuda",
    dtype=dtype,
  )
  config = _make_ptq_config(
    tmp_path / "exclude_layers",
    _make_calibrate_fn(calibration_samples),
    exclude_layers=["to_k", "to_out"],
  )

  quantized_model = cache_dit.quantize(model, config)

  assert isinstance(quantized_model.block.to_q, SVDQW4A4Linear)
  assert isinstance(quantized_model.block.to_v, SVDQW4A4Linear)
  assert isinstance(quantized_model.block.to_k, torch.nn.Linear)
  assert isinstance(quantized_model.block.to_out, torch.nn.Linear)


def test_svdq_ptq_raises_when_calibration_observes_no_candidate_layers(tmp_path: Path) -> None:
  dtype = runtime_dtype()
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=71,
    device="cuda",
    dtype=dtype,
  )
  config = _make_ptq_config(tmp_path / "unobserved", lambda **_: None)

  with pytest.raises(RuntimeError, match="no layers were quantized"):
    cache_dit.quantize(model, config)


@pytest.mark.parametrize("load_arg_kind", ["checkpoint", "directory", "config"])
def test_svdq_ptq_load_roundtrip_restores_quantized_module(
  tmp_path: Path,
  load_arg_kind: str,
) -> None:
  dtype = runtime_dtype()
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=17,
    device="cuda",
    dtype=dtype,
  )
  calibration_samples = make_token_samples(
    num_samples=4,
    batch_size=1,
    seq_len=24,
    width=128,
    seed=37,
    device="cuda",
    dtype=dtype,
  )
  config = _make_ptq_config(tmp_path / "roundtrip", _make_calibrate_fn(calibration_samples))
  quantized_model = cache_dit.quantize(model, config)
  checkpoint_dir = _resolve_quant_checkpoint_dir(config.serialize_to)
  quant_config_snapshot = _load_quant_config_snapshot(config.serialize_to)
  assert quant_config_snapshot["checkpoint_path"] == Path(config.serialize_to).name
  assert quant_config_snapshot["exclude_layers"] == list(config.exclude_layers or [])

  fresh_model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=17,
    device="cuda",
    dtype=dtype,
  )
  if load_arg_kind == "config":
    load_arg = config
  elif load_arg_kind == "directory":
    load_arg = str(checkpoint_dir)
  else:
    load_arg = config.serialize_to
  loaded_model = cache_dit.load(fresh_model, load_arg)

  eval_inputs = make_token_batch(
    batch_size=2,
    seq_len=12,
    width=128,
    seed=41,
    device="cuda",
    dtype=dtype,
  )
  with torch.inference_mode():
    quantized_output = quantized_model(eval_inputs)
    loaded_output = loaded_model(eval_inputs)
    torch.cuda.synchronize()
  atol, rtol = _ptq_tolerance()
  torch.testing.assert_close(loaded_output, quantized_output, rtol=rtol, atol=atol)
  assert isinstance(loaded_model.block.to_q, SVDQW4A4Linear)
  assert isinstance(loaded_model.block.to_k, SVDQW4A4Linear)
  assert isinstance(loaded_model.block.to_v, SVDQW4A4Linear)
  assert isinstance(loaded_model.block.to_out, SVDQW4A4Linear)


def test_svdq_ptq_directory_load_requires_quant_config_json(tmp_path: Path) -> None:
  empty_checkpoint_dir = tmp_path / "missing_quant_config"
  empty_checkpoint_dir.mkdir(parents=True, exist_ok=True)

  with pytest.raises(FileNotFoundError, match="quant_config.json"):
    cache_dit.load(
      torch.nn.Linear(128, 128, device="cuda", dtype=runtime_dtype()),
      str(empty_checkpoint_dir),
    )


def test_svdq_ptq_config_validation_rejects_invalid_combinations(tmp_path: Path) -> None:
  with pytest.raises(ValueError, match="calibrate_fn"):
    QuantizeConfig(
      quant_type="svdq_int4_r32",
      serialize_to=str(tmp_path / "missing_calibrate"),
    )

  with pytest.raises(ValueError, match="serialize_to"):
    QuantizeConfig(
      quant_type="svdq_int4_r32",
      calibrate_fn=lambda **_: None,
    )

  with pytest.raises(ValueError, match="svdq_int4_r"):
    QuantizeConfig(
      quant_type="svdq_int8_r32",
      calibrate_fn=lambda **_: None,
      serialize_to=str(tmp_path / "unsupported_quant_type"),
    )

  with pytest.raises(ValueError, match="precision_plan"):
    QuantizeConfig(
      quant_type="svdq_int4_r32",
      calibrate_fn=lambda **_: None,
      serialize_to=str(tmp_path / "precision_plan"),
      precision_plan={"to_q": "float8_per_row"},
    )

  with pytest.raises(ValueError, match="components_to_quantize"):
    QuantizeConfig(
      quant_type="svdq_int4_r32",
      calibrate_fn=lambda **_: None,
      serialize_to=str(tmp_path / "components"),
      components_to_quantize=["transformer"],
    )

  with pytest.raises(ValueError, match="Unsupported SVDQ PTQ kwargs"):
    QuantizeConfig(
      quant_type="svdq_int4_r32",
      calibrate_fn=lambda **_: None,
      serialize_to=str(tmp_path / "unknown_kwargs"),
      svdq_kwargs={"unknown": True},
    )

  with pytest.raises(TypeError, match="must be a bool"):
    QuantizeConfig(
      quant_type="svdq_int4_r32",
      calibrate_fn=lambda **_: None,
      serialize_to=str(tmp_path / "bad_bool"),
      svdq_kwargs={"streaming": 1},
    )

  with pytest.raises(TypeError, match="must be an int or None"):
    QuantizeConfig(
      quant_type="svdq_int4_r32",
      calibrate_fn=lambda **_: None,
      serialize_to=str(tmp_path / "bad_sample_flush_type"),
      svdq_kwargs={"activation_buffer_flush_sample_count": "2"},
    )

  with pytest.raises(ValueError, match="positive integer"):
    QuantizeConfig(
      quant_type="svdq_int4_r32",
      calibrate_fn=lambda **_: None,
      serialize_to=str(tmp_path / "bad_sample_flush_value"),
      svdq_kwargs={"activation_buffer_flush_sample_count": 0},
    )

  with pytest.raises(TypeError, match="must be an int or None"):
    QuantizeConfig(
      quant_type="svdq_int4_r32",
      calibrate_fn=lambda **_: None,
      serialize_to=str(tmp_path / "bad_bytes_flush_type"),
      svdq_kwargs={"activation_buffer_flush_cpu_bytes": True},
    )

  with pytest.raises(TypeError, match="must be a str"):
    QuantizeConfig(
      quant_type="svdq_int4_r32",
      calibrate_fn=lambda **_: None,
      serialize_to=str(tmp_path / "bad_calibrate_precision_type"),
      svdq_kwargs={"calibrate_precision": 1},
    )

  with pytest.raises(ValueError, match="must be one of"):
    QuantizeConfig(
      quant_type="svdq_int4_r32",
      calibrate_fn=lambda **_: None,
      serialize_to=str(tmp_path / "bad_calibrate_precision_value"),
      svdq_kwargs={"calibrate_precision": "ultra"},
    )

  with pytest.raises(ValueError, match="smooth_strategy"):
    QuantizeConfig(
      quant_type="svdq_int4_r32",
      calibrate_fn=lambda **_: None,
      serialize_to=str(tmp_path / "bad_smooth_strategy"),
      svdq_kwargs={"smooth_strategy": "identity"},
    )

  with pytest.raises(ValueError, match="must be one of"):
    QuantizeConfig(
      quant_type="svdq_int4_r32",
      calibrate_fn=lambda **_: None,
      serialize_to=str(tmp_path / "bad_runtime_kernel_value"),
      svdq_kwargs={"runtime_kernel": "v3"},
    )

  with pytest.raises(ValueError, match="Unsupported SVDQ PTQ kwargs"):
    QuantizeConfig(
      quant_type="svdq_int4_r32",
      calibrate_fn=lambda **_: None,
      serialize_to=str(tmp_path / "legacy_precision_keys"),
      svdq_kwargs={"high_precision": True},
    )


def test_svdq_ptq_load_rejects_invalid_or_incomplete_metadata(tmp_path: Path) -> None:
  from safetensors.torch import save_file

  missing_metadata_path = tmp_path / "missing_metadata.safetensors"
  save_file({"dummy": torch.zeros(1)}, str(missing_metadata_path), metadata={})
  with pytest.raises(ValueError, match="metadata key"):
    cache_dit.load(
      torch.nn.Linear(128, 128, device="cuda", dtype=runtime_dtype()),
      str(missing_metadata_path),
    )

  malformed_metadata_path = tmp_path / "malformed_metadata.safetensors"
  save_file(
    {"dummy": torch.zeros(1)},
    str(malformed_metadata_path),
    metadata={_SVDQ_METADATA_KEY_NAME: "not-json"},
  )
  with pytest.raises(ValueError, match="Invalid SVDQ PTQ metadata JSON"):
    cache_dit.load(
      torch.nn.Linear(128, 128, device="cuda", dtype=runtime_dtype()),
      str(malformed_metadata_path),
    )

  legacy_metadata_path = tmp_path / "legacy_metadata.safetensors"
  save_file(
    {"dummy": torch.zeros(1)},
    str(legacy_metadata_path),
    metadata={
      _SVDQ_METADATA_KEY_NAME:
      '{"format":"cache_dit_svdq_ptq","version":1,"quant_type":"svdq_int4_r32","rank":32,"quantized_layer_names":["block.to_q"],"svdq_kwargs":{"streaming":true,"high_precision":false,"fp32_fallback":true}}'
    },
  )
  with pytest.raises(ValueError, match="Unsupported SVDQ PTQ checkpoint version 1"):
    cache_dit.load(
      torch.nn.Linear(128, 128, device="cuda", dtype=runtime_dtype()),
      str(legacy_metadata_path),
    )

  incomplete_metadata_path = tmp_path / "incomplete_metadata.safetensors"
  save_file(
    {"dummy": torch.zeros(1)},
    str(incomplete_metadata_path),
    metadata={
      _SVDQ_METADATA_KEY_NAME:
      '{"format":"cache_dit_svdq_ptq","version":2,"quant_type":"svdq_int4_r32","rank":32,"quantized_layer_names":["block.to_q"],"svdq_kwargs":{"streaming":true,"calibrate_precision":"medium"}}'
    },
  )
  with pytest.raises(ValueError, match="No serialized tensors found"):
    cache_dit.load(
      make_toy_model(embed_dim=128, num_heads=4, seed=73, device="cuda", dtype=runtime_dtype()),
      str(incomplete_metadata_path),
    )


def test_svdq_ptq_load_rejects_quant_type_mismatch(tmp_path: Path) -> None:
  dtype = runtime_dtype()
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=79,
    device="cuda",
    dtype=dtype,
  )
  calibration_samples = make_token_samples(
    num_samples=2,
    batch_size=1,
    seq_len=16,
    width=128,
    seed=83,
    device="cuda",
    dtype=dtype,
  )
  config = _make_ptq_config(tmp_path / "mismatch_src", _make_calibrate_fn(calibration_samples))
  cache_dit.quantize(model, config)

  mismatch_dir = tmp_path / "mismatch_dst"
  mismatch_dir.mkdir(parents=True, exist_ok=True)
  mismatch_path = mismatch_dir / "svdq_int4_r16.safetensors"
  shutil.copyfile(config.serialize_to, mismatch_path)
  mismatch_config = QuantizeConfig(
    quant_type="svdq_int4_r16",
    calibrate_fn=lambda **_: None,
    serialize_to=str(mismatch_path),
  )

  fresh_model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=79,
    device="cuda",
    dtype=dtype,
  )
  with pytest.raises(ValueError, match="does not match QuantizeConfig"):
    cache_dit.load(fresh_model, mismatch_config)


@pytest.mark.skipif(
  not _ENABLE_FLUX2_SLOW_TEST,
  reason="FLUX.2 SVDQ PTQ integration test requires CACHE_DIT_SVDQ_TEST_FLUX2=1.",
)
def test_svdq_ptq_flux2_klein_partial_quantization_psnr_report(tmp_path: Path) -> None:
  artifacts = _run_flux2_svdq_case(
    case_name="partial_blocks",
    prompts=_FLUX2_PROMPTS,
    exclude_layers_factory=_partial_flux2_exclude_layers,
    require_all_quantized=False,
  )
  assert artifacts["reference_psnr_summary"]["loaded_avg_psnr"] > _FLUX2_REFERENCE_PSNR_THRESHOLD
  assert artifacts["quant_config_path"].is_file()
  assert all(path.is_file() for path in artifacts["report_paths"].values())


@pytest.mark.skipif(
  not _ENABLE_FLUX2_SLOW_TEST,
  reason="FLUX.2 SVDQ PTQ integration test requires CACHE_DIT_SVDQ_TEST_FLUX2=1.",
)
def test_svdq_ptq_flux2_klein_full_quantization_benchmark_report(tmp_path: Path) -> None:
  artifacts = _run_flux2_svdq_case(
    case_name="full_blocks",
    prompts=_FLUX2_FULL_PROMPTS,
    exclude_layers_factory=_full_flux2_exclude_layers,
    require_all_quantized=True,
  )
  assert artifacts["reference_psnr_summary"]["loaded_avg_psnr"] > _FLUX2_REFERENCE_PSNR_THRESHOLD
  assert artifacts["quant_config_path"].is_file()
  assert all(path.is_file() for path in artifacts["report_paths"].values())


@pytest.mark.skipif(
  not _ENABLE_FLUX2_SLOW_TEST,
  reason="FLUX.2 SVDQ PTQ integration test requires CACHE_DIT_SVDQ_TEST_FLUX2=1.",
)
def test_svdq_ptq_flux2_klein_loaded_quantized_transformer_uses_less_cuda_memory() -> None:
  from diffusers import Flux2KleinPipeline

  model_source = _resolve_flux2_model_source()
  prompt = _FLUX2_PROMPTS[0]
  plan = Flux2ExecutionPlan(height=1024, width=1024, offload_mode="cuda")
  checkpoint_dir = _resolve_saved_flux2_checkpoint_dir("full_blocks", plan)

  float_pipe = None
  loaded_pipe = None
  try:
    float_pipe = Flux2KleinPipeline.from_pretrained(model_source, torch_dtype=torch.bfloat16)
    float_pipe.to("cuda")
    float_transformer_cuda_gib = _compute_module_cuda_tensor_bytes(float_pipe.transformer) / (1024
                                                                                              ** 3)
    assert float_transformer_cuda_gib > 0.0
    float_image, float_peak_memory_gib = _run_flux2_single_inference_and_peak_memory_gib(
      float_pipe,
      prompt=prompt,
      plan=plan,
    )
    assert float_peak_memory_gib > 0.0
    assert float_image.size == (plan.width, plan.height)
    _cleanup_flux2_pipe(float_pipe)
    float_pipe = None

    loaded_pipe = Flux2KleinPipeline.from_pretrained(model_source, torch_dtype=torch.bfloat16)
    loaded_pipe.transformer = cache_dit.load(
      loaded_pipe.transformer,
      str(checkpoint_dir),
    )
    loaded_pipe.to("cuda")
    quantized_transformer_cuda_gib = _compute_module_cuda_tensor_bytes(
      loaded_pipe.transformer) / (1024 ** 3)
    memory_ratio = quantized_transformer_cuda_gib / float_transformer_cuda_gib
    image, quantized_peak_memory_gib = _run_flux2_single_inference_and_peak_memory_gib(
      loaded_pipe,
      prompt=prompt,
      plan=plan,
    )
    assert quantized_peak_memory_gib > 0.0
    peak_memory_ratio = quantized_peak_memory_gib / float_peak_memory_gib

    print(
      "[SVDQ FLUX2 memory] "
      f"checkpoint_dir={checkpoint_dir}, "
      f"float_transformer_cuda_gib={float_transformer_cuda_gib:.4f}, "
      f"quantized_transformer_cuda_gib={quantized_transformer_cuda_gib:.4f}, "
      f"quantized_to_float_ratio={memory_ratio:.4f}, "
      f"float_peak_memory_gib={float_peak_memory_gib:.4f}, "
      f"quantized_peak_memory_gib={quantized_peak_memory_gib:.4f}, "
      f"quantized_peak_to_float_ratio={peak_memory_ratio:.4f}",
      flush=True,
    )

    assert quantized_transformer_cuda_gib > 0.0
    assert quantized_transformer_cuda_gib < float_transformer_cuda_gib
    assert memory_ratio < 0.5
    assert image.size == (plan.width, plan.height)
  finally:
    _cleanup_flux2_pipe(loaded_pipe)
    _cleanup_flux2_pipe(float_pipe)


@pytest.mark.skipif(
  not (_ENABLE_FLUX2_SLOW_TEST and _ENABLE_FLUX2_COMPILE_TEST),
  reason=("FLUX.2 SVDQ PTQ compile integration test requires "
          "CACHE_DIT_SVDQ_TEST_FLUX2=1 and CACHE_DIT_SVDQ_TEST_FLUX2_COMPILE=1."),
)
def test_svdq_ptq_flux2_klein_loaded_quantized_transformer_supports_torch_compile() -> None:
  from diffusers import Flux2KleinPipeline

  if not hasattr(torch, "compile"):
    pytest.skip("torch.compile is unavailable in this PyTorch build.")

  if hasattr(torch, "compiler"):
    torch.compiler.reset()

  model_source = _resolve_flux2_model_source()
  prompt = _FLUX2_PROMPTS[0]
  plan = Flux2ExecutionPlan(height=1024, width=1024, offload_mode="cuda")
  checkpoint_dir = _resolve_saved_flux2_checkpoint_dir("full_blocks", plan)
  benchmark_runs = _resolve_flux2_compile_bench_runs()

  loaded_pipe = None
  try:
    loaded_pipe = Flux2KleinPipeline.from_pretrained(model_source, torch_dtype=torch.bfloat16)
    loaded_pipe.transformer = cache_dit.load(
      loaded_pipe.transformer,
      str(checkpoint_dir),
    )
    assert getattr(loaded_pipe.transformer, "_is_quantized", False)

    loaded_pipe.to("cuda")

    eager_image, eager_benchmark = _measure_flux2_latency_benchmark(
      loaded_pipe,
      prompt=prompt,
      plan=plan,
      stage="eager",
      run_count=benchmark_runs,
    )
    assert eager_image.size == (plan.width, plan.height)
    assert eager_benchmark.peak_memory_gb > 0.0

    cache_dit.set_compile_configs()
    loaded_pipe.transformer = torch.compile(loaded_pipe.transformer)

    warmup_image, warmup_benchmark = _measure_flux2_latency_benchmark(
      loaded_pipe,
      prompt=prompt,
      plan=plan,
      stage="compiled_warmup",
      run_count=1,
    )
    assert warmup_image.size == (plan.width, plan.height)
    assert warmup_benchmark.peak_memory_gb > 0.0

    image, compiled_benchmark = _measure_flux2_latency_benchmark(
      loaded_pipe,
      prompt=prompt,
      plan=plan,
      stage="compiled_steady_state",
      run_count=benchmark_runs,
    )

    report_path = _emit_flux2_report(
      report_dir=checkpoint_dir.parent / "reports",
      filename="compile_latency.md",
      title=f"SVDQ PTQ FLUX2 compile latency benchmark ({plan.resolution}, {plan.offload_mode})",
      headers=(
        "stage",
        "resolution",
        "offload",
        "run_count",
        "avg_latency_s",
        "total_latency_s",
        "peak_memory_gb",
        "latency_delta_vs_eager_s",
        "speedup_vs_eager",
      ),
      rows=_compile_benchmark_rows_to_table_rows(
        eager_benchmark,
        [eager_benchmark, warmup_benchmark, compiled_benchmark],
      ),
    )

    compiled_speedup = (eager_benchmark.avg_latency_s / compiled_benchmark.avg_latency_s
                        if compiled_benchmark.avg_latency_s > 0.0 else float("inf"))
    compiled_delta_s = compiled_benchmark.avg_latency_s - eager_benchmark.avg_latency_s

    print(
      "[SVDQ FLUX2 compile] "
      f"checkpoint_dir={checkpoint_dir}, "
      f"benchmark_runs={benchmark_runs}, "
      f"eager_avg_latency_s={eager_benchmark.avg_latency_s:.4f}, "
      f"compiled_warmup_latency_s={warmup_benchmark.avg_latency_s:.4f}, "
      f"compiled_avg_latency_s={compiled_benchmark.avg_latency_s:.4f}, "
      f"compiled_latency_delta_vs_eager_s={compiled_delta_s:.4f}, "
      f"compiled_speedup_vs_eager={_format_speedup(compiled_speedup)}, "
      f"eager_peak_memory_gib={eager_benchmark.peak_memory_gb:.4f}, "
      f"warmup_peak_memory_gib={warmup_benchmark.peak_memory_gb:.4f}, "
      f"inference_peak_memory_gib={compiled_benchmark.peak_memory_gb:.4f}, "
      f"report_path={report_path}",
      flush=True,
    )

    assert compiled_benchmark.peak_memory_gb > 0.0
    assert report_path.is_file()
    assert image.size == (plan.width, plan.height)
  finally:
    _cleanup_flux2_pipe(loaded_pipe)
