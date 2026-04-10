from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from cache_dit.metrics import compute_psnr

_ENABLE_FLUX1_CP_SLOW_TEST = os.getenv("CACHE_DIT_TEST_FLUX1_CP", "0").lower() == "1"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_TMP_DIR = _REPO_ROOT / ".tmp"
_PYTHON_BIN = Path("/workspace/dev/miniconda3/envs/cdit/bin/python")
_TORCHRUN_BIN = Path("/workspace/dev/miniconda3/envs/cdit/bin/torchrun")
_DEFAULT_VISIBLE_DEVICES = os.getenv("CACHE_DIT_TEST_FLUX1_CP_CUDA_VISIBLE_DEVICES", "6,7")
_DEFAULT_MODEL_SOURCE = os.getenv("FLUX_DIR", "black-forest-labs/FLUX.1-dev")
_DEFAULT_PROMPT = os.getenv(
  "CACHE_DIT_TEST_FLUX1_CP_PROMPT",
  "A cat holding a sign that says hello world",
)
_DEFAULT_STEPS = int(os.getenv("CACHE_DIT_TEST_FLUX1_CP_STEPS", "28"))
_DEFAULT_SEED = int(os.getenv("CACHE_DIT_TEST_FLUX1_CP_SEED", "0"))
_DEFAULT_PSNR_THRESHOLD = float(os.getenv("CACHE_DIT_TEST_FLUX1_CP_PSNR_THRESHOLD", "20.0"))
_TEST_OUTPUT_DIR = _REPO_TMP_DIR / "tests" / "parallelism" / "flux1_context_parallel"
_SUMMARY_PATH = _TEST_OUTPUT_DIR / "summary.json"
_BASELINE_IMAGE_PATH = _TEST_OUTPUT_DIR / "baseline.png"
_RING_IMAGE_PATH = _TEST_OUTPUT_DIR / "ring2.png"
_ULYSSES_IMAGE_PATH = _TEST_OUTPUT_DIR / "ulysses2.png"

_INFERENCE_TIME_PATTERN = re.compile(r"Inference Time:\s*([0-9.]+)s")
_MEMORY_USAGE_PATTERN = re.compile(r"Memory Usage:\s*([0-9.]+)GiB")
_PEAK_MEMORY_PATTERN = re.compile(r"Peak GPU memory usage:\s*([0-9.]+)\s*GB")


@dataclass(frozen=True)
class FluxContextParallelRunResult:
  mode: str
  command: list[str]
  image_path: Path
  latency_s: float | None
  peak_memory_gib: float | None
  wall_time_s: float
  stdout: str


pytestmark = pytest.mark.skipif(
  not _ENABLE_FLUX1_CP_SLOW_TEST,
  reason="FLUX.1-dev context parallel slow test requires CACHE_DIT_TEST_FLUX1_CP=1.",
)


def _build_base_env() -> dict[str, str]:
  env = os.environ.copy()
  env["PYTHONPATH"] = str(_REPO_ROOT / "src")
  env["CUDA_VISIBLE_DEVICES"] = _DEFAULT_VISIBLE_DEVICES
  env.setdefault("FLUX_DIR", _DEFAULT_MODEL_SOURCE)
  return env


def _parse_latency(stdout: str) -> float | None:
  match = _INFERENCE_TIME_PATTERN.search(stdout)
  if match is None:
    return None
  return float(match.group(1))


def _parse_peak_memory(stdout: str) -> float | None:
  summary_match = _MEMORY_USAGE_PATTERN.search(stdout)
  if summary_match is not None:
    return float(summary_match.group(1))
  peak_match = _PEAK_MEMORY_PATTERN.search(stdout)
  if peak_match is not None:
    return float(peak_match.group(1))
  return None


def _run_generate_command(mode: str, command: list[str],
                          image_path: Path) -> FluxContextParallelRunResult:
  env = _build_base_env()
  image_path.parent.mkdir(parents=True, exist_ok=True)
  start_time = time.perf_counter()
  completed = subprocess.run(
    command,
    cwd=_REPO_ROOT,
    env=env,
    capture_output=True,
    text=True,
    check=True,
  )
  wall_time_s = time.perf_counter() - start_time

  if not image_path.is_file():
    raise AssertionError(f"Expected generated image at {image_path}, but it does not exist.")

  stdout = completed.stdout + completed.stderr
  return FluxContextParallelRunResult(
    mode=mode,
    command=command,
    image_path=image_path,
    latency_s=_parse_latency(stdout),
    peak_memory_gib=_parse_peak_memory(stdout),
    wall_time_s=wall_time_s,
    stdout=stdout,
  )


def _baseline_command() -> list[str]:
  return [
    str(_PYTHON_BIN),
    "-m",
    "cache_dit.generate",
    "flux",
    "--prompt",
    _DEFAULT_PROMPT,
    "--steps",
    str(_DEFAULT_STEPS),
    "--seed",
    str(_DEFAULT_SEED),
    "--track-memory",
    "--save-path",
    str(_BASELINE_IMAGE_PATH),
  ]


def _parallel_command(mode: str, save_path: Path) -> list[str]:
  return [
    str(_TORCHRUN_BIN),
    "--nproc_per_node=2",
    "-m",
    "cache_dit.generate",
    "flux",
    "--parallel",
    mode,
    "--prompt",
    _DEFAULT_PROMPT,
    "--steps",
    str(_DEFAULT_STEPS),
    "--seed",
    str(_DEFAULT_SEED),
    "--track-memory",
    "--save-path",
    str(save_path),
  ]


def _format_markdown_table(rows: list[dict[str, object]]) -> str:
  lines = [
    "| mode | latency_s | wall_time_s | peak_memory_gib | psnr_vs_baseline | image |",
    "| --- | ---: | ---: | ---: | ---: | --- |",
  ]
  for row in rows:
    latency = "N/A" if row["latency_s"] is None else f"{row['latency_s']:.2f}"
    memory = "N/A" if row["peak_memory_gib"] is None else f"{row['peak_memory_gib']:.2f}"
    psnr = "N/A" if row["psnr_vs_baseline"] is None else f"{row['psnr_vs_baseline']:.4f}"
    lines.append(
      f"| {row['mode']} | {latency} | {row['wall_time_s']:.2f} | {memory} | {psnr} | {row['image_path']} |"
    )
  return "\n".join(lines)


@pytest.mark.skipif(
  not torch.cuda.is_available(),
  reason="FLUX.1-dev context parallel slow test requires CUDA.",
)
def test_flux1_context_parallel_ring_and_ulysses_psnr() -> None:
  visible_devices = [
    device.strip() for device in _DEFAULT_VISIBLE_DEVICES.split(",") if device.strip()
  ]
  if len(visible_devices) < 2:
    pytest.skip("FLUX.1-dev context parallel slow test requires two visible devices.")

  if not _PYTHON_BIN.is_file() or not _TORCHRUN_BIN.is_file():
    pytest.skip("The configured cdit python/torchrun binaries are unavailable.")

  if _TEST_OUTPUT_DIR.exists():
    shutil.rmtree(_TEST_OUTPUT_DIR)
  _TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

  baseline_result = _run_generate_command(
    mode="baseline",
    command=_baseline_command(),
    image_path=_BASELINE_IMAGE_PATH,
  )
  ring_result = _run_generate_command(
    mode="ring",
    command=_parallel_command("ring", _RING_IMAGE_PATH),
    image_path=_RING_IMAGE_PATH,
  )
  ulysses_result = _run_generate_command(
    mode="ulysses",
    command=_parallel_command("ulysses", _ULYSSES_IMAGE_PATH),
    image_path=_ULYSSES_IMAGE_PATH,
  )

  ring_psnr, ring_count = compute_psnr(str(_BASELINE_IMAGE_PATH), str(_RING_IMAGE_PATH))
  ulysses_psnr, ulysses_count = compute_psnr(str(_BASELINE_IMAGE_PATH), str(_ULYSSES_IMAGE_PATH))

  assert ring_count == 1
  assert ulysses_count == 1
  assert ring_psnr is not None and ring_psnr >= _DEFAULT_PSNR_THRESHOLD
  assert ulysses_psnr is not None and ulysses_psnr >= _DEFAULT_PSNR_THRESHOLD

  rows = [
    {
      "mode": baseline_result.mode,
      "latency_s": baseline_result.latency_s,
      "wall_time_s": baseline_result.wall_time_s,
      "peak_memory_gib": baseline_result.peak_memory_gib,
      "psnr_vs_baseline": None,
      "image_path": str(baseline_result.image_path),
    },
    {
      "mode": ring_result.mode,
      "latency_s": ring_result.latency_s,
      "wall_time_s": ring_result.wall_time_s,
      "peak_memory_gib": ring_result.peak_memory_gib,
      "psnr_vs_baseline": ring_psnr,
      "image_path": str(ring_result.image_path),
    },
    {
      "mode": ulysses_result.mode,
      "latency_s": ulysses_result.latency_s,
      "wall_time_s": ulysses_result.wall_time_s,
      "peak_memory_gib": ulysses_result.peak_memory_gib,
      "psnr_vs_baseline": ulysses_psnr,
      "image_path": str(ulysses_result.image_path),
    },
  ]
  summary = {
    "prompt": _DEFAULT_PROMPT,
    "steps": _DEFAULT_STEPS,
    "seed": _DEFAULT_SEED,
    "cuda_visible_devices": _DEFAULT_VISIBLE_DEVICES,
    "psnr_threshold": _DEFAULT_PSNR_THRESHOLD,
    "rows": rows,
    "markdown_table": _format_markdown_table(rows),
  }
  _SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
  print(summary["markdown_table"])
