from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from diffusers import DiffusionPipeline

import cache_dit
from cache_dit import ParallelismConfig
from cache_dit.metrics import compute_psnr

ray = pytest.importorskip("ray")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_BIN = Path("/workspace/dev/miniconda3/envs/cdit/bin/python")
_DEFAULT_VISIBLE_DEVICES = os.getenv("CACHE_DIT_TEST_RAY_CUDA_VISIBLE_DEVICES", "6,7")
_ENABLE_FLUX_TEST = os.getenv("CACHE_DIT_TEST_RAY_FLUX", "0").lower() == "1"
_DEFAULT_MODEL_SOURCE = os.getenv("FLUX_DIR", "/workspace/dev/vipdev/hf_models/FLUX.1-dev")
_TEST_OUTPUT_DIR = _REPO_ROOT / ".tmp" / "tests" / "ray_wrapper"


class ToyPipeline(DiffusionPipeline):
  """Minimal DiffusionPipeline-shaped object with a transformer attribute."""

  def __init__(self, transformer: torch.nn.Module) -> None:
    super().__init__()
    self.register_modules(transformer=transformer)

  def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return self.transformer(hidden_states)


class ToyCompilableTransformer(torch.nn.Module):
  """Tiny transformer that exposes compile_repeated_blocks for Ray compile tests."""

  def __init__(self) -> None:
    super().__init__()
    self.linear = torch.nn.Linear(4, 4)
    self.repeated_blocks_compiled = False
    with torch.no_grad():
      self.linear.weight.copy_(torch.eye(4))
      self.linear.bias.zero_()

  def compile_repeated_blocks(self) -> None:
    """Mark repeated blocks as compiled without invoking torch.compile in unit tests."""

    self.repeated_blocks_compiled = True

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    marker = torch.tensor(float(self.repeated_blocks_compiled), device=hidden_states.device)
    return self.linear(hidden_states) + marker


@pytest.fixture(autouse=True)
def shutdown_ray_runtime():
  yield
  if ray.is_initialized():
    ray.shutdown()


def _toy_transformer() -> torch.nn.Sequential:
  linear = torch.nn.Linear(4, 4)
  with torch.no_grad():
    linear.weight.copy_(torch.eye(4))
    linear.bias.copy_(torch.arange(4, dtype=torch.float32))
  return torch.nn.Sequential(linear, torch.nn.Sigmoid())


def _toy_parallel_config() -> ParallelismConfig:
  return ParallelismConfig(
    ulysses_size=2,
    use_ray=True,
    ray_runtime_env={
      "env_vars": {
        "PYTHONPATH": f"{_REPO_ROOT / 'tests' / 'ray'}:{_REPO_ROOT / 'src'}",
      },
    },
    ray_worker_options={"num_gpus": 0},
    _ray_skip_native_parallelism=True,
  )


def test_parallelism_config_rejects_short_aliases() -> None:
  """ParallelismConfig should require canonical size field names."""

  with pytest.raises(TypeError):
    ParallelismConfig(ulysses=2)
  with pytest.raises(TypeError):
    ParallelismConfig(ring=2)
  with pytest.raises(TypeError):
    ParallelismConfig(tp=2)


def test_ray_wrapper_transformer_only_toy_model() -> None:
  hidden_states = torch.arange(8, dtype=torch.float32).reshape(2, 4)
  transformer = _toy_transformer()
  baseline = transformer(hidden_states)

  returned = cache_dit.enable_cache(transformer, parallelism_config=_toy_parallel_config())

  assert returned is transformer
  assert getattr(transformer, "_cache_dit_ray_enabled", False)
  original_dtype = next(transformer.parameters()).dtype
  assert transformer.to(torch.float64) is transformer
  assert next(transformer.parameters()).dtype == original_dtype
  result = transformer(hidden_states)
  torch.testing.assert_close(result, baseline)

  cache_dit.disable_cache(transformer)
  assert not hasattr(transformer, "_cache_dit_ray_enabled")
  transformer.to(torch.float64)
  assert next(transformer.parameters()).dtype == torch.float64
  transformer.to(original_dtype)
  torch.testing.assert_close(transformer(hidden_states), baseline)


def test_ray_wrapper_pipeline_level_toy_model() -> None:
  hidden_states = torch.arange(8, dtype=torch.float32).reshape(2, 4)
  pipe = ToyPipeline(_toy_transformer())
  baseline = pipe(hidden_states)

  returned = cache_dit.enable_cache(pipe, parallelism_config=_toy_parallel_config())

  assert returned is pipe
  assert getattr(pipe, "_cache_dit_ray_pipeline_enabled", False)
  result = pipe(hidden_states)
  torch.testing.assert_close(result, baseline)

  cache_dit.disable_cache(pipe)
  assert not hasattr(pipe, "_cache_dit_ray_pipeline_enabled")
  torch.testing.assert_close(pipe(hidden_states), baseline)


def test_ray_wrapper_compile_repeated_blocks_toy_model() -> None:
  hidden_states = torch.arange(8, dtype=torch.float32).reshape(2, 4)
  transformer = ToyCompilableTransformer()
  baseline = transformer(hidden_states)
  parallelism_config = _toy_parallel_config()
  parallelism_config.ray_use_compile = True

  cache_dit.enable_cache(transformer, parallelism_config=parallelism_config)

  result = transformer(hidden_states)
  torch.testing.assert_close(result, baseline + 1.0)

  cache_dit.disable_cache(transformer)


@pytest.mark.skipif(
  not _ENABLE_FLUX_TEST,
  reason="FLUX Ray wrapper test requires CACHE_DIT_TEST_RAY_FLUX=1.",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="FLUX Ray wrapper test requires CUDA.")
def test_ray_wrapper_flux_example_psnr() -> None:
  visible_devices = [
    device.strip() for device in _DEFAULT_VISIBLE_DEVICES.split(",") if device.strip()
  ]
  if len(visible_devices) < 2:
    pytest.skip("FLUX Ray wrapper test requires at least two visible CUDA devices.")
  if not _PYTHON_BIN.is_file():
    pytest.skip("The configured cdit python binary is unavailable.")

  if _TEST_OUTPUT_DIR.exists():
    shutil.rmtree(_TEST_OUTPUT_DIR)
  _TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

  baseline_path = _TEST_OUTPUT_DIR / "baseline.png"
  ray_path = _TEST_OUTPUT_DIR / "ray2.png"
  env = os.environ.copy()
  env["PYTHONPATH"] = str(_REPO_ROOT / "src")
  env["CUDA_VISIBLE_DEVICES"] = _DEFAULT_VISIBLE_DEVICES
  env["FLUX_DIR"] = _DEFAULT_MODEL_SOURCE

  subprocess.run(
    [
      str(_PYTHON_BIN),
      "examples/ray_wrapper_example.py",
      "--ulysses",
      "1",
      "--save-path",
      str(baseline_path),
    ],
    cwd=_REPO_ROOT,
    env=env,
    check=True,
  )
  subprocess.run(
    [
      str(_PYTHON_BIN),
      "examples/ray_wrapper_example.py",
      "--ulysses",
      "2",
      "--save-path",
      str(ray_path),
    ],
    cwd=_REPO_ROOT,
    env=env,
    check=True,
  )

  psnr, count = compute_psnr(str(baseline_path), str(ray_path))
  assert count == 1
  assert psnr is not None and psnr > 20.0
