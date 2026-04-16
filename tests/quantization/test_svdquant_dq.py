from __future__ import annotations

import copy
import gc
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import cache_dit
import cache_dit.quantization.svdquant.ptq as svdq_ptq
import cache_dit.quantization.svdquant.quantizer as svdq_quantizer
from cache_dit import BlockAdapter
from cache_dit.kernels import svdq_extension_is_available
from cache_dit.metrics import compute_psnr
from cache_dit._utils.utils import get_args
from cache_dit._utils.utils import maybe_apply_optimization
from cache_dit._utils.utils import maybe_compile_transformer
from cache_dit._utils.utils import maybe_finalize_deferred_svdq_pipe_move
from cache_dit._utils.utils import maybe_generic_module_offload
from cache_dit._utils.utils import maybe_quantize_transformer
from cache_dit._utils.utils import maybe_postprocess_args
from cache_dit.offload import get_layerwise_offload_handles
from cache_dit.quantization import QuantizeConfig
from cache_dit.quantization.svdquant import SVDQW4A4Linear
from tests.quantization._svdq_test_utils import TOY_ATTENTION_LINEAR_NAMES
from tests.quantization._svdq_test_utils import ToyTransformerBlock
from tests.quantization._svdq_test_utils import assert_rank_metric_trend
from tests.quantization._svdq_test_utils import compute_accuracy_metrics
from tests.quantization._svdq_test_utils import format_markdown_table
from tests.quantization._svdq_test_utils import make_token_batch
from tests.quantization._svdq_test_utils import make_toy_model
from tests.quantization._svdq_test_utils import runtime_dtype

_ENABLE_FLUX2_SLOW_TEST = os.getenv("CACHE_DIT_SVDQ_TEST_FLUX2", "0").lower() == "1"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPO_TMP_DIR = _REPO_ROOT / ".tmp"
_FLUX2_MODEL_SOURCE_ENVVAR = "FLUX_2_KLEIN_4B_DIR"
_FLUX2_DEFAULT_MODEL_SOURCE = "black-forest-labs/FLUX.2-klein-4B"
_FLUX2_CPU_SEED = 0
_FLUX2_NUM_INFERENCE_STEPS = 4
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
_FLUX2_FEW_SHOT_WARMUP_PROMPT = _FLUX2_PROMPTS[0]
_FLUX2_FEW_SHOT_BENCH_PROMPTS = _FLUX2_PROMPTS[1:5]

pytestmark = pytest.mark.skipif(
  not torch.cuda.is_available() or not svdq_extension_is_available(),
  reason="SVDQ DQ tests require CUDA and the optional SVDQuant extension.",
)


class ToyRepeatedBlocksModel(nn.Module):

  def __init__(self, embed_dim: int = 128, num_heads: int = 4) -> None:
    super().__init__()
    self.blocks = nn.ModuleList([
      ToyTransformerBlock(embed_dim=embed_dim, num_heads=num_heads),
      ToyTransformerBlock(embed_dim=embed_dim, num_heads=num_heads),
    ])
    self.head = nn.Linear(embed_dim, embed_dim, bias=True)
    self._repeated_blocks = ["ToyTransformerBlock"]

  def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    for block in self.blocks:
      hidden_states = block(hidden_states)
    return self.head(hidden_states)


def _make_dq_config(rank: int, **kwargs) -> QuantizeConfig:
  return QuantizeConfig(
    quant_type=f"svdq_int4_r{rank}_dq",
    **kwargs,
  )


def _make_normalized_transformer_adapter(transformer: nn.Module) -> BlockAdapter:
  adapter = BlockAdapter(transformer=[transformer], skip_post_init=True)
  adapter._is_normalized = True
  return adapter


def _assert_identity_smooth_factor(module: SVDQW4A4Linear) -> None:
  expected = torch.ones_like(module.smooth_factor)
  torch.testing.assert_close(module.smooth_factor.detach(), expected, rtol=0.0, atol=0.0)
  torch.testing.assert_close(module.smooth_factor_orig.detach(), expected, rtol=0.0, atol=0.0)


def _assert_weight_smooth_factor(module: SVDQW4A4Linear) -> None:
  smooth = module.smooth_factor.detach().float()
  smooth_orig = module.smooth_factor_orig.detach().float()
  assert torch.isfinite(smooth).all()
  assert torch.isfinite(smooth_orig).all()
  assert torch.all(smooth > 0)
  assert torch.all(smooth_orig > 0)
  torch.testing.assert_close(smooth, smooth_orig, rtol=0.0, atol=0.0)
  assert not torch.allclose(smooth, torch.ones_like(smooth), rtol=0.0, atol=1e-3)


def _assert_weight_inv_smooth_factor(module: SVDQW4A4Linear) -> None:
  smooth = module.smooth_factor.detach().float()
  smooth_orig = module.smooth_factor_orig.detach().float()
  assert torch.isfinite(smooth).all()
  assert torch.isfinite(smooth_orig).all()
  assert torch.all(smooth > 0)
  assert torch.all(smooth_orig > 0)
  torch.testing.assert_close(smooth, smooth_orig, rtol=0.0, atol=0.0)
  assert not torch.allclose(smooth, torch.ones_like(smooth), rtol=0.0, atol=1e-3)


def _capture_layer_activation_span_during_first_forward(
  model: nn.Module,
  layer: nn.Linear,
  inputs: torch.Tensor,
) -> torch.Tensor:
  captured: dict[str, torch.Tensor] = {}

  def hook(module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
    if not args or not isinstance(args[0], torch.Tensor):
      raise TypeError("Expected tensor inputs when capturing few-shot activation spans.")
    activation = args[0].detach().reshape(-1, module.in_features)
    captured["span"] = activation.abs().amax(dim=0).float()

  handle = layer.register_forward_pre_hook(hook)
  try:
    with torch.inference_mode():
      _ = model(inputs)
  finally:
    handle.remove()

  assert "span" in captured
  return captured["span"]


def _quartile_means(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  quartile = max(values.numel() // 4, 1)
  lower = values[:quartile].mean()
  middle = values[quartile:-quartile].mean() if values.numel() > quartile * 2 else values.mean()
  upper = values[-quartile:].mean()
  return lower, middle, upper


def _make_flux2_cpu_generator(seed: int) -> torch.Generator:
  return torch.Generator(device="cpu").manual_seed(seed)


def _resolve_flux2_model_source() -> str:
  source = os.getenv(_FLUX2_MODEL_SOURCE_ENVVAR, _FLUX2_DEFAULT_MODEL_SOURCE)
  candidate = Path(source).expanduser()
  if candidate.exists():
    return str(candidate.resolve())
  return source


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


def _run_flux2_case(
  *,
  prompts: list[str],
  output_dir: Path,
  quant_type: str | None = None,
  quantize_config: QuantizeConfig | None = None,
) -> dict[str, object]:
  from diffusers import Flux2KleinPipeline

  output_dir.mkdir(parents=True, exist_ok=True)
  model_source = _resolve_flux2_model_source()
  pipe = None
  rows: list[dict[str, object]] = []
  peak_memory_gib = 0.0
  quantize_time_s = 0.0
  try:
    pipe = Flux2KleinPipeline.from_pretrained(model_source, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    resolved_quantize_config = quantize_config
    if resolved_quantize_config is None and quant_type is not None:
      resolved_quantize_config = QuantizeConfig(quant_type=quant_type)
    if resolved_quantize_config is not None:
      quantize_start = time.perf_counter()
      pipe = cache_dit.enable_cache(
        pipe,
        quantize_config=resolved_quantize_config,
      )
      torch.cuda.synchronize()
      quantize_time_s = time.perf_counter() - quantize_start

    for prompt_index, prompt in enumerate(prompts):
      torch.cuda.reset_peak_memory_stats()
      start = time.perf_counter()
      image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=_FLUX2_NUM_INFERENCE_STEPS,
        guidance_scale=1.0,
        generator=_make_flux2_cpu_generator(_FLUX2_CPU_SEED + prompt_index),
      ).images[0]
      torch.cuda.synchronize()
      latency_s = time.perf_counter() - start
      peak_memory_gib = max(
        peak_memory_gib,
        torch.cuda.max_memory_allocated() / (1024 ** 3),
      )
      image_path = output_dir / f"prompt_{prompt_index:02d}.png"
      image.save(image_path)
      rows.append({
        "prompt_index": prompt_index,
        "prompt": prompt,
        "image_path": image_path,
        "latency_s": latency_s,
      })
    return {
      "rows": rows,
      "peak_memory_gib": peak_memory_gib,
      "quantize_time_s": quantize_time_s,
      "output_dir": output_dir,
    }
  finally:
    _cleanup_flux2_pipe(pipe)


def _run_flux2_few_shot_case(
  *,
  warmup_prompt: str,
  prompts: list[str],
  output_dir: Path,
  quantize_config: QuantizeConfig,
) -> dict[str, object]:
  from diffusers import Flux2KleinPipeline

  output_dir.mkdir(parents=True, exist_ok=True)
  model_source = _resolve_flux2_model_source()
  pipe = None
  rows: list[dict[str, object]] = []
  benchmark_peak_memory_gib = 0.0
  warmup_peak_memory_gib = 0.0
  warmup_latency_s = 0.0
  runtime_quantize_time_s = 0.0
  try:
    pipe = Flux2KleinPipeline.from_pretrained(model_source, torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    pipe = cache_dit.enable_cache(pipe, quantize_config=quantize_config)

    torch.cuda.reset_peak_memory_stats()
    warmup_start = time.perf_counter()
    _ = pipe(
      prompt=warmup_prompt,
      height=1024,
      width=1024,
      num_inference_steps=_FLUX2_NUM_INFERENCE_STEPS,
      guidance_scale=1.0,
      generator=_make_flux2_cpu_generator(_FLUX2_CPU_SEED),
    ).images[0]
    torch.cuda.synchronize()
    warmup_latency_s = time.perf_counter() - warmup_start
    warmup_peak_memory_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)

    transformer = pipe.transformer
    runtime_quantize_time_s = float(getattr(transformer, "_svdq_runtime_quantize_time_s", 0.0))
    assert runtime_quantize_time_s > 0.0
    assert not getattr(transformer, "_svdq_pending_quantization", False)

    for prompt_index, prompt in enumerate(prompts):
      torch.cuda.reset_peak_memory_stats()
      start = time.perf_counter()
      image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=_FLUX2_NUM_INFERENCE_STEPS,
        guidance_scale=1.0,
        generator=_make_flux2_cpu_generator(_FLUX2_CPU_SEED + prompt_index + 1),
      ).images[0]
      torch.cuda.synchronize()
      latency_s = time.perf_counter() - start
      benchmark_peak_memory_gib = max(
        benchmark_peak_memory_gib,
        torch.cuda.max_memory_allocated() / (1024 ** 3),
      )
      image_path = output_dir / f"prompt_{prompt_index:02d}.png"
      image.save(image_path)
      rows.append({
        "prompt_index": prompt_index,
        "prompt": prompt,
        "image_path": image_path,
        "latency_s": latency_s,
      })

    return {
      "rows": rows,
      "warmup_latency_s": warmup_latency_s,
      "warmup_peak_memory_gib": warmup_peak_memory_gib,
      "benchmark_peak_memory_gib": benchmark_peak_memory_gib,
      "runtime_quantize_time_s": runtime_quantize_time_s,
      "output_dir": output_dir,
    }
  finally:
    _cleanup_flux2_pipe(pipe)


def test_svdq_dq_config_validation_accepts_dynamic_without_ptq_args() -> None:
  config = QuantizeConfig(
    quant_type="svdq_int4_r32_dq",
    svdq_kwargs={
      "calibrate_precision": "high",
      "streaming": False
    },
  )

  assert config.is_svdq()
  assert config.is_svdq_dq()
  assert not config.is_svdq_ptq()
  assert config.get_svdq_rank() == 32
  assert config.get_svdq_kwargs()["calibrate_precision"] == "high"
  assert config.get_svdq_kwargs()["smooth_strategy"] == "identity"


def test_svdq_dq_config_validation_defaults_calibrate_precision_to_low() -> None:
  config = QuantizeConfig(quant_type="svdq_int4_r32_dq")

  assert config.get_svdq_kwargs()["calibrate_precision"] == "low"
  assert config.get_svdq_kwargs()["runtime_kernel"] == "v1"
  assert config.get_svdq_kwargs()["quantize_device"] == "auto"
  assert config.get_svdq_kwargs()["offload_quantized_layers_to_cpu"] is False
  assert config.get_svdq_kwargs()["layerwise_offload"] is False
  assert config.get_svdq_kwargs()["async_transfer"] is False
  assert config.get_svdq_kwargs()["transfer_buckets"] == 1
  assert config.get_svdq_kwargs()["max_copy_streams"] is None
  assert config.get_svdq_kwargs()["max_inflight_prefetch_bytes"] is None
  assert config.get_svdq_kwargs()["persistent_buckets"] == 0
  assert config.get_svdq_kwargs()["persistent_bins"] == 1
  assert config.get_svdq_kwargs()["defer_move_to_execution_device"] is False


@pytest.mark.parametrize("runtime_kernel", ["v2"])
def test_svdq_dq_config_validation_accepts_runtime_kernels(runtime_kernel: str) -> None:
  config = QuantizeConfig(
    quant_type="svdq_int4_r32_dq",
    svdq_kwargs={"runtime_kernel": runtime_kernel},
  )

  assert config.get_svdq_kwargs()["runtime_kernel"] == runtime_kernel


def test_svdq_dq_config_validation_accepts_explicit_identity_smooth_strategy() -> None:
  config = QuantizeConfig(
    quant_type="svdq_int4_r32_dq",
    svdq_kwargs={"smooth_strategy": "identity"},
  )

  assert config.get_svdq_kwargs()["smooth_strategy"] == "identity"


def test_svdq_dq_config_validation_accepts_weight_smooth_strategy() -> None:
  config = QuantizeConfig(
    quant_type="svdq_int4_r32_dq",
    svdq_kwargs={"smooth_strategy": "weight"},
  )

  assert config.get_svdq_kwargs()["smooth_strategy"] == "weight"


def test_svdq_dq_config_validation_accepts_weight_inv_smooth_strategy() -> None:
  config = QuantizeConfig(
    quant_type="svdq_int4_r32_dq",
    svdq_kwargs={"smooth_strategy": "weight_inv"},
  )

  assert config.get_svdq_kwargs()["smooth_strategy"] == "weight_inv"


def test_svdq_dq_config_validation_accepts_few_shot_smooth_strategy() -> None:
  config = QuantizeConfig(
    quant_type="svdq_int4_r32_dq",
    svdq_kwargs={"smooth_strategy": "few_shot"},
  )

  assert config.get_svdq_kwargs()["smooth_strategy"] == "few_shot"
  assert config.get_svdq_kwargs()["few_shot_steps"] == 1
  assert config.get_svdq_kwargs()["few_shot_relax_strategy"] == "auto"
  assert config.get_svdq_kwargs()["few_shot_auto_compile"] is False


def test_svdq_dq_config_validation_accepts_device_strategy_kwargs() -> None:
  config = QuantizeConfig(
    quant_type="svdq_int4_r32_dq",
    svdq_kwargs={
      "quantize_device": "cuda",
      "offload_quantized_layers_to_cpu": True,
      "layerwise_offload": True,
      "async_transfer": True,
      "transfer_buckets": 2,
      "max_copy_streams": 1,
      "max_inflight_prefetch_bytes": 4096,
      "persistent_buckets": 1,
      "persistent_bins": 2,
      "defer_move_to_execution_device": True,
    },
  )

  assert config.get_svdq_kwargs()["quantize_device"] == "cuda"
  assert config.get_svdq_kwargs()["offload_quantized_layers_to_cpu"] is True
  assert config.get_svdq_kwargs()["layerwise_offload"] is True
  assert config.get_svdq_kwargs()["async_transfer"] is True
  assert config.get_svdq_kwargs()["transfer_buckets"] == 2
  assert config.get_svdq_kwargs()["max_copy_streams"] == 1
  assert config.get_svdq_kwargs()["max_inflight_prefetch_bytes"] == 4096
  assert config.get_svdq_kwargs()["persistent_buckets"] == 1
  assert config.get_svdq_kwargs()["persistent_bins"] == 2
  assert config.get_svdq_kwargs()["defer_move_to_execution_device"] is True


def test_svdq_dq_config_validation_rejects_ptq_only_fields_and_load(tmp_path: Path) -> None:
  with pytest.raises(ValueError, match="calibrate_fn"):
    QuantizeConfig(
      quant_type="svdq_int4_r32_dq",
      calibrate_fn=lambda **_: None,
    )

  with pytest.raises(ValueError, match="serialize_to"):
    QuantizeConfig(
      quant_type="svdq_int4_r32_dq",
      serialize_to=str(tmp_path / "svdq_int4_r32_dq.safetensors"),
    )

  with pytest.raises(ValueError, match="smooth_strategy"):
    QuantizeConfig(
      quant_type="svdq_int4_r32_dq",
      svdq_kwargs={"smooth_strategy": "activation"},
    )

  config = QuantizeConfig(quant_type="svdq_int4_r32_dq")
  with pytest.raises(ValueError, match="does not support load"):
    cache_dit.load(
      nn.Linear(128, 128, device="cuda", dtype=runtime_dtype()),
      config,
    )


def test_svdq_dq_skips_calibrator_registration(monkeypatch) -> None:
  dtype = runtime_dtype()
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=11,
    device="cuda",
    dtype=dtype,
  )

  def _unexpected_register(*_args, **_kwargs):
    raise AssertionError("SVDQ PTQ calibrator should not run for SVDQ DQ.")

  monkeypatch.setattr(svdq_ptq.SVDQPTQCalibrator, "register", _unexpected_register)
  quantized_model = cache_dit.quantize(model, _make_dq_config(rank=32))

  assert isinstance(quantized_model.block.to_q, SVDQW4A4Linear)


def test_svdq_dq_cli_flags_map_to_quantize_type() -> None:
  parser = get_args(parse=False)

  args = maybe_postprocess_args(parser.parse_args(["--svdq-int4-r32-dq"]))
  assert args.quantize
  assert args.quantize_type == "svdq_int4_r32_dq"
  assert args.svdq_smooth_strategy == "identity"
  assert args.svdq_calibrate_precision == "low"
  assert args.svdq_runtime == "v1"
  assert args.svdq_quantize_device == "cuda"
  assert args.svdq_offload_quantized_layers_to_cpu is False
  assert args.svdq_defer_final_to_cuda is True

  args = maybe_postprocess_args(parser.parse_args(["--svdq-int4-r128-dq"]))
  assert args.quantize
  assert args.quantize_type == "svdq_int4_r128_dq"
  assert args.svdq_smooth_strategy == "identity"

  args = maybe_postprocess_args(parser.parse_args(["--quantize-type", "svdq_int4_r32_dq"]))
  assert args.quantize
  assert args.quantize_type == "svdq_int4_r32_dq"

  args = maybe_postprocess_args(
    parser.parse_args([
      "--quantize-type",
      "svdq_int4_r32_dq",
      "--svdq-smooth-strategy",
      "weight",
    ]))
  assert args.quantize
  assert args.quantize_type == "svdq_int4_r32_dq"
  assert args.svdq_smooth_strategy == "weight"

  args = maybe_postprocess_args(
    parser.parse_args([
      "--quantize-type",
      "svdq_int4_r32_dq",
      "--svdq-smooth-strategy",
      "few_shot",
    ]))
  assert args.quantize
  assert args.quantize_type == "svdq_int4_r32_dq"
  assert args.svdq_smooth_strategy == "few_shot"
  assert args.svdq_few_shot_steps == 1
  assert args.svdq_few_shot_relax_factor == 1.5
  assert args.svdq_few_shot_relax_top_ratio == 0.25
  assert args.svdq_few_shot_relax_strategy == "auto"

  args = maybe_postprocess_args(
    parser.parse_args([
      "--quantize-type",
      "svdq_int4_r32_dq",
      "--svdq-quantize-device",
      "auto",
      "--svdq-layerwise-offload",
      "--svdq-layerwise-async-transfer",
      "--svdq-layerwise-transfer-buckets",
      "2",
      "--svdq-layerwise-prefetch-limit",
      "--svdq-layerwise-max-copy-streams",
      "1",
      "--svdq-layerwise-max-inflight-prefetch-bytes",
      "4096",
      "--svdq-layerwise-persistent-buckets",
      "1",
      "--svdq-layerwise-persistent-bins",
      "2",
      "--svdq-keep-quantized-layers-on-device",
      "--svdq-no-defer-final-to-cuda",
    ]))
  assert args.svdq_quantize_device == "auto"
  assert args.svdq_layerwise_offload is True
  assert args.layerwise_async_transfer is True
  assert args.layerwise_transfer_buckets == 2
  assert args.layerwise_prefetch_limit is True
  assert args.layerwise_max_copy_streams == 1
  assert args.layerwise_max_inflight_prefetch_bytes == 4096
  assert args.layerwise_persistent_buckets == 1
  assert args.layerwise_persistent_bins == 2
  assert args.svdq_offload_quantized_layers_to_cpu is True
  assert args.svdq_defer_final_to_cuda is False


def test_generic_module_offload_cli_is_mutually_exclusive_with_diffusers_offload() -> None:
  parser = get_args(parse=False)

  with pytest.raises(SystemExit):
    parser.parse_args(["--cpu-offload", "--module-layerwise-cpu-offload"])

  with pytest.raises(SystemExit):
    parser.parse_args(["--sequential-cpu-offload", "--module-layerwise-cpu-offload"])

  args = maybe_postprocess_args(parser.parse_args(["--module-layerwise-cpu-offload"]))
  assert args.module_layerwise_cpu_offload is True

  args = maybe_postprocess_args(
    parser.parse_args([
      "--module-layerwise-cpu-offload",
      "--layerwise-async-transfer",
      "--layerwise-transfer-buckets",
      "2",
      "--layerwise-prefetch-limit",
      "--layerwise-max-copy-streams",
      "1",
      "--layerwise-max-inflight-prefetch-bytes",
      "4096",
      "--layerwise-persistent-buckets",
      "1",
      "--layerwise-persistent-bins",
      "2",
    ]))
  assert args.module_layerwise_cpu_offload is True
  assert args.layerwise_async_transfer is True
  assert args.layerwise_transfer_buckets == 2
  assert args.layerwise_prefetch_limit is True
  assert args.layerwise_max_copy_streams == 1
  assert args.layerwise_max_inflight_prefetch_bytes == 4096
  assert args.layerwise_persistent_buckets == 1
  assert args.layerwise_persistent_bins == 2

  args = maybe_postprocess_args(
    parser.parse_args([
      "--module-layerwise-cpu-offload",
      "--layerwise-async-transfer",
      "--layerwise-transfer-buckets",
      "2",
      "--layerwise-prefetch-limit",
      "--layerwise-max-copy-streams",
      "1",
      "--layerwise-max-inflight-prefetch-bytes",
      "4096",
      "--layerwise-persistent-buckets",
      "1",
      "--layerwise-persistent-bins",
      "2",
      "--layerwise-text-transfer-buckets",
      "5",
      "--no-layerwise-text-prefetch-limit",
      "--layerwise-text-max-copy-streams",
      "3",
      "--layerwise-text-max-inflight-prefetch-bytes",
      "2GiB",
      "--layerwise-text-persistent-buckets",
      "4",
      "--layerwise-text-persistent-bins",
      "3",
    ]))
  assert args.layerwise_text_async_transfer is None
  assert args.layerwise_text_transfer_buckets == 5
  assert args.layerwise_text_prefetch_limit is False
  assert args.layerwise_text_max_copy_streams == 3
  assert args.layerwise_text_max_inflight_prefetch_bytes == 2 * 1024 ** 3
  assert args.layerwise_text_persistent_buckets == 4
  assert args.layerwise_text_persistent_bins == 3


def test_layerwise_max_inflight_prefetch_bytes_cli_accepts_gib_suffix() -> None:
  parser = get_args(parse=False)

  args = maybe_postprocess_args(
    parser.parse_args([
      "--layerwise-max-inflight-prefetch-bytes",
      "4GiB",
    ]))
  assert args.layerwise_max_inflight_prefetch_bytes == 4 * 1024 ** 3

  args = maybe_postprocess_args(
    parser.parse_args([
      "--svdq-layerwise-max-inflight-prefetch-bytes",
      "0.5GiB",
    ]))
  assert args.layerwise_max_inflight_prefetch_bytes == 512 * 1024 ** 2

  args = maybe_postprocess_args(
    parser.parse_args([
      "--quantize-type",
      "svdq_int4_r32_dq",
      "--svdq-smooth-strategy",
      "few_shot",
      "--svdq-few-shot-relax-strategy",
      "fixed",
    ]))
  assert args.quantize
  assert args.quantize_type == "svdq_int4_r32_dq"
  assert args.svdq_smooth_strategy == "few_shot"
  assert args.svdq_few_shot_relax_strategy == "fixed"

  args = maybe_postprocess_args(
    parser.parse_args([
      "--quantize-type",
      "svdq_int4_r32_dq",
      "--svdq-smooth-strategy",
      "few_shot",
      "--svdq-few-shot-relax-strategy",
      "stable_auto",
    ]))
  assert args.quantize
  assert args.quantize_type == "svdq_int4_r32_dq"
  assert args.svdq_smooth_strategy == "few_shot"
  assert args.svdq_few_shot_relax_strategy == "stable_auto"

  args = maybe_postprocess_args(
    parser.parse_args([
      "--quantize-type",
      "svdq_int4_r32_dq",
      "--svdq-smooth-strategy",
      "weight_inv",
      "--svdq-calib",
      "high",
      "--svdq-runtime",
      "v2",
    ]))
  assert args.quantize
  assert args.quantize_type == "svdq_int4_r32_dq"
  assert args.svdq_smooth_strategy == "weight_inv"
  assert args.svdq_calibrate_precision == "high"
  assert args.svdq_runtime == "v2"

  args = maybe_postprocess_args(
    parser.parse_args([
      "--quantize-type",
      "svdq_int4_r32_dq",
      "--svdq-smooth-strategy",
      "few_shot",
      "--svdq-few-shot-steps",
      "3",
      "--svdq-few-shot-relax-factor",
      "2.5",
      "--svdq-few-shot-relax-top-ratio",
      "0.5",
      "--svdq-few-shot-relax-strategy",
      "auto",
      "--svdq-few-shot-compile",
    ]))
  assert args.quantize
  assert args.quantize_type == "svdq_int4_r32_dq"
  assert args.svdq_smooth_strategy == "few_shot"
  assert args.svdq_few_shot_steps == 3
  assert args.svdq_few_shot_relax_factor == 2.5
  assert args.svdq_few_shot_relax_top_ratio == 0.5
  assert args.svdq_few_shot_relax_strategy == "auto"
  assert args.svdq_few_shot_compile is True


def test_generic_module_offload_uses_text_specific_layerwise_overrides(
  monkeypatch: pytest.MonkeyPatch, ) -> None:
  args = SimpleNamespace(
    module_layerwise_cpu_offload=True,
    layerwise_async_transfer=True,
    layerwise_transfer_buckets=2,
    layerwise_prefetch_limit=True,
    layerwise_max_copy_streams=1,
    layerwise_max_inflight_prefetch_bytes=4096,
    layerwise_persistent_buckets=1,
    layerwise_persistent_bins=2,
    layerwise_text_async_transfer=None,
    layerwise_text_transfer_buckets=5,
    layerwise_text_prefetch_limit=False,
    layerwise_text_max_copy_streams=3,
    layerwise_text_max_inflight_prefetch_bytes=2 * 1024 ** 3,
    layerwise_text_persistent_buckets=4,
    layerwise_text_persistent_bins=3,
  )

  pipe = SimpleNamespace(
    transformer=nn.Linear(8, 8),
    text_encoder=nn.Linear(8, 8),
  )
  captured: dict[str, dict[str, object]] = {}

  monkeypatch.setattr("cache_dit._utils.utils.get_rank_device", lambda:
                      (0, torch.device("cuda", 0)))
  monkeypatch.setattr("cache_dit._utils.utils.remove_layerwise_offload", lambda _module: 0)
  monkeypatch.setattr("cache_dit._utils.utils._find_offload_related_hf_hook", lambda _module: None)

  def _capture_handle(module: nn.Module, **kwargs):
    module_name = next(name for name, candidate in vars(pipe).items() if candidate is module)
    captured[module_name] = kwargs
    return SimpleNamespace(remove=lambda **_kwargs: None)

  monkeypatch.setattr("cache_dit._utils.utils.layerwise_cpu_offload", _capture_handle)

  assert maybe_generic_module_offload(args, pipe) is True
  assert captured["transformer"]["async_transfer"] is True
  assert captured["transformer"]["transfer_buckets"] == 2
  assert captured["transformer"]["prefetch_limit"] is True
  assert captured["transformer"]["max_copy_streams"] == 1
  assert captured["transformer"]["max_inflight_prefetch_bytes"] == 4096
  assert captured["transformer"]["persistent_buckets"] == 1
  assert captured["transformer"]["persistent_bins"] == 2

  assert captured["text_encoder"]["async_transfer"] is True
  assert captured["text_encoder"]["transfer_buckets"] == 5
  assert captured["text_encoder"]["prefetch_limit"] is False
  assert captured["text_encoder"]["max_copy_streams"] == 3
  assert captured["text_encoder"]["max_inflight_prefetch_bytes"] == 2 * 1024 ** 3
  assert captured["text_encoder"]["persistent_buckets"] == 4
  assert captured["text_encoder"]["persistent_bins"] == 3


def test_svdq_dq_cli_smooth_strategy_is_applied_during_transformer_quantization() -> None:
  dtype = runtime_dtype()
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=53,
    device="cuda",
    dtype=dtype,
  )
  parser = get_args(parse=False)
  args = maybe_postprocess_args(
    parser.parse_args([
      "--quantize-type",
      "svdq_int4_r32_dq",
      "--svdq-smooth-strategy",
      "weight",
    ]))

  holder = SimpleNamespace(transformer=model)
  maybe_quantize_transformer(args, holder)

  assert holder.transformer._svdq_kwargs["calibrate_precision"] == "low"
  _assert_weight_smooth_factor(holder.transformer.block.to_q)
  _assert_weight_smooth_factor(holder.transformer.block.to_k)
  _assert_weight_smooth_factor(holder.transformer.block.to_v)
  _assert_weight_smooth_factor(holder.transformer.block.to_out)


def test_svdq_dq_cli_weight_inv_strategy_is_applied_during_transformer_quantization() -> None:
  dtype = runtime_dtype()
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=54,
    device="cuda",
    dtype=dtype,
  )
  parser = get_args(parse=False)
  args = maybe_postprocess_args(
    parser.parse_args([
      "--quantize-type",
      "svdq_int4_r32_dq",
      "--svdq-smooth-strategy",
      "weight_inv",
      "--svdq-calibrate-precision",
      "medium",
    ]))

  holder = SimpleNamespace(transformer=model)
  maybe_quantize_transformer(args, holder)

  assert holder.transformer._svdq_kwargs["calibrate_precision"] == "medium"
  _assert_weight_inv_smooth_factor(holder.transformer.block.to_q)
  _assert_weight_inv_smooth_factor(holder.transformer.block.to_k)
  _assert_weight_inv_smooth_factor(holder.transformer.block.to_v)
  _assert_weight_inv_smooth_factor(holder.transformer.block.to_out)


def test_svdq_dq_cli_quantization_runs_from_cpu_root_and_keeps_quantized_layers_on_device_by_default(
) -> None:
  dtype = runtime_dtype()
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=57,
    device="cpu",
    dtype=dtype,
  )
  parser = get_args(parse=False)
  args = maybe_postprocess_args(parser.parse_args([
    "--quantize-type",
    "svdq_int4_r32_dq",
  ]))

  holder = SimpleNamespace(transformer=model)
  maybe_quantize_transformer(args, holder)

  module = holder.transformer.block.to_q
  assert isinstance(module, SVDQW4A4Linear)
  assert module.qweight.device.type == "cuda"
  assert holder.transformer._svdq_kwargs["quantize_device"] == "cuda"
  assert holder.transformer._svdq_kwargs["offload_quantized_layers_to_cpu"] is False


def test_svdq_dq_cli_layerwise_offload_forces_quantized_layers_back_to_cpu() -> None:
  dtype = runtime_dtype()
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=58,
    device="cpu",
    dtype=dtype,
  )
  parser = get_args(parse=False)
  args = maybe_postprocess_args(
    parser.parse_args([
      "--quantize-type",
      "svdq_int4_r32_dq",
      "--svdq-layerwise-offload",
      "--svdq-keep-quantized-layers-on-device",
    ]))

  holder = SimpleNamespace(transformer=model)
  maybe_quantize_transformer(args, holder)

  module = holder.transformer.block.to_q
  assert isinstance(module, SVDQW4A4Linear)
  assert module.qweight.device.type == "cpu"
  assert holder.transformer._svdq_kwargs["quantize_device"] == "cuda"
  assert holder.transformer._svdq_kwargs["offload_quantized_layers_to_cpu"] is True


@pytest.mark.parametrize("runtime_kernel", ["v2"])
def test_svdq_dq_cli_runtime_kernel_is_applied_during_transformer_quantization(
  runtime_kernel: str, ) -> None:
  dtype = runtime_dtype()
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=55,
    device="cuda",
    dtype=dtype,
  )
  parser = get_args(parse=False)
  args = maybe_postprocess_args(
    parser.parse_args([
      "--quantize-type",
      "svdq_int4_r32_dq",
      "--svdq-runtime",
      runtime_kernel,
      "--svdq-keep-quantized-layers-on-device",
    ]))

  holder = SimpleNamespace(transformer=model)
  maybe_quantize_transformer(args, holder)

  assert holder.transformer._svdq_kwargs["runtime_kernel"] == runtime_kernel
  assert holder.transformer.block.to_q.runtime_kernel == runtime_kernel
  assert holder.transformer.block.to_k.runtime_kernel == runtime_kernel
  assert holder.transformer.block.to_v.runtime_kernel == runtime_kernel
  assert holder.transformer.block.to_out.runtime_kernel == runtime_kernel

  eval_inputs = make_token_batch(
    batch_size=2,
    seq_len=8,
    width=128,
    seed=89,
    device="cuda",
    dtype=dtype,
  )
  with torch.inference_mode():
    output = holder.transformer(eval_inputs)
    torch.cuda.synchronize()

  assert output.shape == eval_inputs.shape


def test_svdq_dq_toy_model_rank_trend_and_metadata() -> None:
  dtype = runtime_dtype()
  float_model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=17,
    device="cuda",
    dtype=dtype,
  )
  eval_inputs = make_token_batch(
    batch_size=2,
    seq_len=12,
    width=128,
    seed=29,
    device="cuda",
    dtype=dtype,
  )

  with torch.inference_mode():
    reference = float_model(eval_inputs)

  metrics_by_rank = {}
  for rank in (32, 128):
    quantized_model = cache_dit.quantize(copy.deepcopy(float_model), _make_dq_config(rank=rank))
    with torch.inference_mode():
      candidate = quantized_model(eval_inputs)
    metrics = compute_accuracy_metrics(reference, candidate)
    metrics_by_rank[rank] = metrics

    assert torch.isfinite(candidate).all()
    assert getattr(quantized_model, "_is_quantized", False)
    assert getattr(quantized_model, "_quantize_type", None) == f"svdq_int4_r{rank}_dq"
    assert getattr(quantized_model, "_svdq_checkpoint_path", "sentinel") is None
    assert isinstance(quantized_model.block.to_q, SVDQW4A4Linear)
    assert isinstance(quantized_model.block.to_k, SVDQW4A4Linear)
    assert isinstance(quantized_model.block.to_v, SVDQW4A4Linear)
    assert isinstance(quantized_model.block.to_out, SVDQW4A4Linear)

  assert_rank_metric_trend(
    metrics_by_rank,
    "rel_l2",
    ranks=(32, 128),
    atol=5e-4,
    rtol=0.25,
  )
  assert metrics_by_rank[32].cosine > 0.6
  assert metrics_by_rank[128].cosine > 0.7


def test_svdq_dq_materializes_identity_smooth_factors() -> None:
  dtype = runtime_dtype()
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=23,
    device="cuda",
    dtype=dtype,
  )

  quantized_model = cache_dit.quantize(model, _make_dq_config(rank=32))

  _assert_identity_smooth_factor(quantized_model.block.to_q)
  _assert_identity_smooth_factor(quantized_model.block.to_k)
  _assert_identity_smooth_factor(quantized_model.block.to_v)
  _assert_identity_smooth_factor(quantized_model.block.to_out)


def test_svdq_dq_weight_smooth_strategy_materializes_non_identity_smooth_factors() -> None:
  dtype = runtime_dtype()
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=41,
    device="cuda",
    dtype=dtype,
  )
  eval_inputs = make_token_batch(
    batch_size=2,
    seq_len=12,
    width=128,
    seed=43,
    device="cuda",
    dtype=dtype,
  )

  quantized_model = cache_dit.quantize(
    model,
    _make_dq_config(rank=32, svdq_kwargs={"smooth_strategy": "weight"}),
  )

  with torch.inference_mode():
    candidate = quantized_model(eval_inputs)

  assert torch.isfinite(candidate).all()
  _assert_weight_smooth_factor(quantized_model.block.to_q)
  _assert_weight_smooth_factor(quantized_model.block.to_k)
  _assert_weight_smooth_factor(quantized_model.block.to_v)
  _assert_weight_smooth_factor(quantized_model.block.to_out)


def test_svdq_dq_weight_inv_smooth_strategy_materializes_non_identity_smooth_factors() -> None:
  dtype = runtime_dtype()
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=42,
    device="cuda",
    dtype=dtype,
  )
  eval_inputs = make_token_batch(
    batch_size=2,
    seq_len=12,
    width=128,
    seed=44,
    device="cuda",
    dtype=dtype,
  )

  quantized_model = cache_dit.quantize(
    model,
    _make_dq_config(rank=32, svdq_kwargs={"smooth_strategy": "weight_inv"}),
  )

  with torch.inference_mode():
    candidate = quantized_model(eval_inputs)

  assert torch.isfinite(candidate).all()
  _assert_weight_inv_smooth_factor(quantized_model.block.to_q)
  _assert_weight_inv_smooth_factor(quantized_model.block.to_k)
  _assert_weight_inv_smooth_factor(quantized_model.block.to_v)
  _assert_weight_inv_smooth_factor(quantized_model.block.to_out)


def test_svdq_dq_few_shot_defers_quantization_until_trigger_step() -> None:
  dtype = runtime_dtype()
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=70,
    device="cuda",
    dtype=dtype,
  )
  eval_inputs = make_token_batch(
    batch_size=2,
    seq_len=12,
    width=128,
    seed=71,
    device="cuda",
    dtype=dtype,
  )

  quantized_model = cache_dit.quantize(
    model,
    _make_dq_config(
      rank=32,
      svdq_kwargs={
        "smooth_strategy": "few_shot",
        "few_shot_steps": 2,
      },
    ),
  )

  assert getattr(quantized_model, "_svdq_pending_quantization", False)
  assert isinstance(quantized_model.block.to_q, nn.Linear)

  with torch.inference_mode():
    first_output = quantized_model(eval_inputs)
  assert torch.isfinite(first_output).all()
  assert getattr(quantized_model, "_svdq_pending_quantization", False)
  assert isinstance(quantized_model.block.to_q, nn.Linear)

  with torch.inference_mode():
    second_output = quantized_model(eval_inputs)
  assert torch.isfinite(second_output).all()
  assert not getattr(quantized_model, "_svdq_pending_quantization", False)
  assert getattr(quantized_model, "_is_quantized", False)
  assert isinstance(quantized_model.block.to_q, SVDQW4A4Linear)
  assert isinstance(quantized_model.block.to_k, SVDQW4A4Linear)
  assert isinstance(quantized_model.block.to_v, SVDQW4A4Linear)
  assert isinstance(quantized_model.block.to_out, SVDQW4A4Linear)


def test_svdq_dq_few_shot_counts_root_forwards_cumulatively_across_runs() -> None:
  dtype = runtime_dtype()
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=170,
    device="cuda",
    dtype=dtype,
  )
  eval_inputs = make_token_batch(
    batch_size=2,
    seq_len=12,
    width=128,
    seed=171,
    device="cuda",
    dtype=dtype,
  )

  quantized_model = cache_dit.quantize(
    model,
    _make_dq_config(
      rank=32,
      svdq_kwargs={
        "smooth_strategy": "few_shot",
        "few_shot_steps": 4,
      },
    ),
  )

  def _run_pipeline_like_steps(step_count: int) -> torch.Tensor:
    output = eval_inputs
    with torch.inference_mode():
      for _ in range(step_count):
        output = quantized_model(eval_inputs)
    return output

  first_run_output = _run_pipeline_like_steps(2)
  assert torch.isfinite(first_run_output).all()
  assert getattr(quantized_model, "_svdq_pending_quantization", False)
  controller = getattr(quantized_model, "_svdq_few_shot_controller")
  assert controller.completed_forwards == 2
  assert isinstance(quantized_model.block.to_q, nn.Linear)

  second_run_output = _run_pipeline_like_steps(2)
  assert torch.isfinite(second_run_output).all()
  assert not getattr(quantized_model, "_svdq_pending_quantization", False)
  assert getattr(quantized_model, "_svdq_runtime_quantized_after_forwards", 0) == 4
  assert isinstance(quantized_model.block.to_q, SVDQW4A4Linear)


def test_svdq_dq_few_shot_cleanup_releases_controller_buffers() -> None:
  dtype = runtime_dtype()
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=172,
    device="cuda",
    dtype=dtype,
  )
  eval_inputs = make_token_batch(
    batch_size=2,
    seq_len=12,
    width=128,
    seed=173,
    device="cuda",
    dtype=dtype,
  )

  quantized_model = cache_dit.quantize(
    model,
    _make_dq_config(
      rank=32,
      svdq_kwargs={
        "smooth_strategy": "few_shot",
        "few_shot_steps": 1,
      },
    ),
  )
  controller = getattr(quantized_model, "_svdq_few_shot_controller")

  with torch.inference_mode():
    output = quantized_model(eval_inputs)

  assert torch.isfinite(output).all()
  assert not getattr(quantized_model, "_svdq_pending_quantization", False)
  assert not hasattr(quantized_model, "_svdq_few_shot_controller")
  assert controller._handles == []
  assert controller._accumulators == {}
  assert controller.activation_spans == {}


def test_svdq_dq_few_shot_cpu_root_collection_uses_layerwise_cuda_offload() -> None:
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=174,
    device="cpu",
    dtype=torch.float32,
  )
  eval_inputs = make_token_batch(
    batch_size=2,
    seq_len=12,
    width=128,
    seed=175,
    device="cpu",
    dtype=torch.float32,
  )

  quantized_model = cache_dit.quantize(
    model,
    _make_dq_config(
      rank=32,
      svdq_kwargs={
        "smooth_strategy": "few_shot",
        "few_shot_steps": 1,
        "quantize_device": "cuda",
        "offload_quantized_layers_to_cpu": True,
        "layerwise_offload": True,
        "async_transfer": True,
        "transfer_buckets": 2,
        "prefetch_limit": True,
        "max_copy_streams": 1,
        "max_inflight_prefetch_bytes": 4096,
        "persistent_buckets": 1,
        "persistent_bins": 2,
      },
    ),
  )
  pending_handles = get_layerwise_offload_handles(quantized_model)
  assert len(pending_handles) == 1
  assert pending_handles[0].async_transfer is True
  assert pending_handles[0].transfer_buckets == 2
  assert pending_handles[0].prefetch_limit is True
  assert pending_handles[0].max_copy_streams == 1
  assert pending_handles[0].max_inflight_prefetch_bytes == 4096
  assert pending_handles[0].effective_persistent_buckets == 1
  assert pending_handles[0].persistent_bins == 2
  assert pending_handles[0].effective_persistent_bins == 1
  assert "block.norm" in pending_handles[0].module_names
  for layer_name in TOY_ATTENTION_LINEAR_NAMES:
    assert layer_name in pending_handles[0].module_names
  observed_input_devices: list[str] = []
  observed_linear = quantized_model.block.to_q
  capture_handle = observed_linear.register_forward_pre_hook(
    lambda _module, args: observed_input_devices.append(args[0].device.type))

  try:
    with torch.inference_mode():
      output = quantized_model(eval_inputs)
      torch.cuda.synchronize()
  finally:
    capture_handle.remove()

  assert torch.isfinite(output).all()
  assert observed_input_devices == ["cuda"]
  assert not getattr(quantized_model, "_svdq_pending_quantization", False)
  assert isinstance(quantized_model.block.to_q, SVDQW4A4Linear)
  assert quantized_model.block.to_q.qweight.device.type == "cpu"


def test_svdq_dq_few_shot_materializes_relaxed_and_original_smooth_vectors() -> None:
  dtype = runtime_dtype()
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=72,
    device="cuda",
    dtype=dtype,
  )
  eval_inputs = make_token_batch(
    batch_size=2,
    seq_len=12,
    width=128,
    seed=73,
    device="cuda",
    dtype=dtype,
  )

  quantized_model = cache_dit.quantize(
    model,
    _make_dq_config(
      rank=32,
      svdq_kwargs={
        "smooth_strategy": "few_shot",
        "few_shot_steps": 1,
        "few_shot_relax_factor": 2.5,
        "few_shot_relax_top_ratio": 0.25,
        "few_shot_relax_strategy": "top",
      },
    ),
  )

  _ = _capture_layer_activation_span_during_first_forward(
    quantized_model,
    quantized_model.block.to_q,
    eval_inputs,
  )

  module = quantized_model.block.to_q
  assert isinstance(module, SVDQW4A4Linear)
  smooth = module.smooth_factor.detach().float()
  smooth_orig = module.smooth_factor_orig.detach().float()
  changed = ~torch.isclose(smooth, smooth_orig, rtol=0.0, atol=1e-4)
  assert changed.any()
  torch.testing.assert_close(
    smooth[~changed],
    smooth_orig[~changed],
    rtol=0.0,
    atol=0.0,
  )
  ratios = smooth[changed] / smooth_orig[changed]
  torch.testing.assert_close(
    ratios,
    torch.full_like(ratios, math.sqrt(2.5)),
    rtol=5e-3,
    atol=1e-2,
  )


def test_svdq_dq_few_shot_fixed_relax_strategy_keeps_original_smooth_vectors() -> None:
  dtype = runtime_dtype()
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=74,
    device="cuda",
    dtype=dtype,
  )
  eval_inputs = make_token_batch(
    batch_size=2,
    seq_len=12,
    width=128,
    seed=75,
    device="cuda",
    dtype=dtype,
  )

  quantized_model = cache_dit.quantize(
    model,
    _make_dq_config(
      rank=32,
      svdq_kwargs={
        "smooth_strategy": "few_shot",
        "few_shot_steps": 1,
        "few_shot_relax_factor": 4.0,
        "few_shot_relax_top_ratio": 0.05,
        "few_shot_relax_strategy": "fixed",
      },
    ),
  )

  _ = _capture_layer_activation_span_during_first_forward(
    quantized_model,
    quantized_model.block.to_q,
    eval_inputs,
  )

  module = quantized_model.block.to_q
  assert isinstance(module, SVDQW4A4Linear)
  torch.testing.assert_close(
    module.smooth_factor.detach().float(),
    module.smooth_factor_orig.detach().float(),
    rtol=0.0,
    atol=1e-4,
  )


def test_svdq_dq_few_shot_auto_relax_strategy_scales_monotonically() -> None:
  dtype = runtime_dtype()
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=76,
    device="cuda",
    dtype=dtype,
  )
  eval_inputs = make_token_batch(
    batch_size=2,
    seq_len=12,
    width=128,
    seed=77,
    device="cuda",
    dtype=dtype,
  )

  quantized_model = cache_dit.quantize(
    model,
    _make_dq_config(
      rank=32,
      svdq_kwargs={
        "smooth_strategy": "few_shot",
        "few_shot_steps": 1,
        "few_shot_relax_factor": 2.5,
        "few_shot_relax_top_ratio": 0.25,
        "few_shot_relax_strategy": "auto",
      },
    ),
  )

  activation_span = _capture_layer_activation_span_during_first_forward(
    quantized_model,
    quantized_model.block.to_q,
    eval_inputs,
  )

  module = quantized_model.block.to_q
  assert isinstance(module, SVDQW4A4Linear)
  smooth = module.smooth_factor.detach().float()
  smooth_orig = module.smooth_factor_orig.detach().float()
  ratios = smooth / smooth_orig
  sorted_indices = torch.argsort(activation_span)
  sorted_ratios = ratios[sorted_indices]
  lower_mean, _, upper_mean = _quartile_means(sorted_ratios)

  assert torch.all(sorted_ratios >= 1.0 - 1e-4)
  assert torch.all(sorted_ratios <= math.sqrt(2.5) + 2e-2)
  assert torch.any(sorted_ratios > 1.0 + 1e-2)
  assert torch.any((sorted_ratios > 1.0 + 1e-2) & (sorted_ratios < math.sqrt(2.5) - 1e-2))
  assert torch.any(
    torch.isclose(sorted_ratios, torch.full_like(sorted_ratios, math.sqrt(2.5)), atol=1e-2))
  assert upper_mean > lower_mean + 5e-2


@pytest.mark.parametrize("strategy", ["stable_auto", "power", "log", "rank"])
def test_svdq_dq_few_shot_extra_relax_strategies_scale_monotonically(strategy: str) -> None:
  dtype = runtime_dtype()
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=78,
    device="cuda",
    dtype=dtype,
  )
  eval_inputs = make_token_batch(
    batch_size=2,
    seq_len=12,
    width=128,
    seed=79,
    device="cuda",
    dtype=dtype,
  )

  quantized_model = cache_dit.quantize(
    model,
    _make_dq_config(
      rank=32,
      svdq_kwargs={
        "smooth_strategy": "few_shot",
        "few_shot_steps": 1,
        "few_shot_relax_factor": 2.5,
        "few_shot_relax_top_ratio": 0.25,
        "few_shot_relax_strategy": strategy,
      },
    ),
  )

  activation_span = _capture_layer_activation_span_during_first_forward(
    quantized_model,
    quantized_model.block.to_q,
    eval_inputs,
  )

  module = quantized_model.block.to_q
  assert isinstance(module, SVDQW4A4Linear)
  smooth = module.smooth_factor.detach().float()
  smooth_orig = module.smooth_factor_orig.detach().float()
  ratios = smooth / smooth_orig
  sorted_indices = torch.argsort(activation_span)
  sorted_ratios = ratios[sorted_indices]
  lower_mean, _, upper_mean = _quartile_means(sorted_ratios)

  assert torch.all(sorted_ratios >= 1.0 - 1e-4)
  assert torch.all(sorted_ratios <= math.sqrt(2.5) + 2e-2)
  assert upper_mean > lower_mean + 2e-2
  assert torch.any(sorted_ratios > 1.0 + 1e-2)
  assert torch.any(
    torch.isclose(sorted_ratios, torch.full_like(sorted_ratios, math.sqrt(2.5)), atol=1e-2))


@pytest.mark.parametrize("compile_args", [["--svdq-few-shot-compile"], ["--compile"]])
def test_svdq_dq_few_shot_deferred_compile_executes_once(
  monkeypatch: pytest.MonkeyPatch,
  compile_args: list[str],
) -> None:
  dtype = runtime_dtype()
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=74,
    device="cuda",
    dtype=dtype,
  )
  parser = get_args(parse=False)
  args = maybe_postprocess_args(
    parser.parse_args([
      "--quantize-type",
      "svdq_int4_r32_dq",
      "--svdq-smooth-strategy",
      "few_shot",
      "--svdq-few-shot-steps",
      "1",
      *compile_args,
    ]))
  holder = SimpleNamespace(transformer=model)
  compile_calls: list[str] = []

  def _fake_compile_transformer_module(_args, _pipe, transformer, name):
    compile_calls.append(name)
    return transformer

  monkeypatch.setattr("cache_dit._utils.utils._compile_transformer_module",
                      _fake_compile_transformer_module)

  maybe_quantize_transformer(args, holder)
  assert getattr(holder.transformer, "_svdq_pending_quantization", False)
  maybe_compile_transformer(args, holder)

  assert compile_calls == []
  assert len(getattr(holder.transformer, "_svdq_post_quantize_callbacks", [])) == 1

  eval_inputs = make_token_batch(
    batch_size=2,
    seq_len=12,
    width=128,
    seed=75,
    device="cuda",
    dtype=dtype,
  )
  with torch.inference_mode():
    output = holder.transformer(eval_inputs)
  assert torch.isfinite(output).all()
  assert compile_calls == ["transformer"]


def test_svdq_dq_few_shot_layerwise_offload_defers_full_pipeline_move_until_after_forward() -> None:
  dtype = runtime_dtype()
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=82,
    device="cpu",
    dtype=dtype,
  )
  parser = get_args(parse=False)
  args = maybe_postprocess_args(
    parser.parse_args([
      "--quantize-type",
      "svdq_int4_r32_dq",
      "--svdq-smooth-strategy",
      "few_shot",
      "--svdq-few-shot-steps",
      "1",
      "--svdq-layerwise-offload",
    ]))

  move_calls: list[torch.device] = []
  holder = SimpleNamespace(transformer=model)
  holder.to = lambda device: move_calls.append(torch.device(device))

  maybe_apply_optimization(args, holder)

  assert getattr(holder.transformer, "_svdq_pending_quantization", False)
  assert move_calls == []

  eval_inputs = make_token_batch(
    batch_size=2,
    seq_len=12,
    width=128,
    seed=83,
    device="cpu",
    dtype=dtype,
  )
  with torch.inference_mode():
    output = holder.transformer(eval_inputs)

  assert torch.isfinite(output).all()
  assert len(move_calls) == 0
  assert hasattr(holder, "_svdq_move_to_device_after_forward")
  assert hasattr(holder.transformer, "_svdq_runtime_layerwise_offload_handle")
  runtime_handle = holder.transformer._svdq_runtime_layerwise_offload_handle
  assert "block.norm" in runtime_handle.module_names
  for layer_name in TOY_ATTENTION_LINEAR_NAMES:
    assert layer_name in runtime_handle.module_names
  assert maybe_finalize_deferred_svdq_pipe_move(holder)
  assert len(move_calls) == 1
  assert move_calls[0].type == "cuda"
  assert not hasattr(holder, "_svdq_move_to_device_after_forward")
  assert not hasattr(holder.transformer, "_svdq_runtime_layerwise_offload_handle")


def test_svdq_dq_few_shot_falls_back_to_deferred_pipeline_move_on_failure() -> None:
  dtype = runtime_dtype()
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=84,
    device="cpu",
    dtype=dtype,
  )
  parser = get_args(parse=False)
  args = maybe_postprocess_args(
    parser.parse_args([
      "--quantize-type",
      "svdq_int4_r32_dq",
      "--svdq-smooth-strategy",
      "few_shot",
      "--svdq-few-shot-steps",
      "1",
      "--svdq-offload-quantized-layers-to-cpu",
    ]))

  move_calls: list[torch.device] = []
  move_attempts = 0
  holder = SimpleNamespace(transformer=model)

  holder.to = lambda _device: None

  maybe_apply_optimization(args, holder)

  def _move_pipe(device: torch.device | str) -> None:
    nonlocal move_attempts
    move_attempts += 1
    resolved_device = torch.device(device)
    if move_attempts == 1:
      raise RuntimeError("simulate immediate move failure")
    move_calls.append(resolved_device)

  holder.to = _move_pipe

  eval_inputs = make_token_batch(
    batch_size=2,
    seq_len=12,
    width=128,
    seed=85,
    device="cpu",
    dtype=dtype,
  )
  with torch.inference_mode():
    output = holder.transformer(eval_inputs)

  assert torch.isfinite(output).all()
  assert hasattr(holder, "_svdq_move_to_device_after_forward")
  assert move_calls == []

  assert maybe_finalize_deferred_svdq_pipe_move(holder)
  assert len(move_calls) == 1
  assert move_calls[0].type == "cuda"
  assert not hasattr(holder, "_svdq_move_to_device_after_forward")


def test_svdq_dq_few_shot_without_layerwise_offload_moves_pipe_to_cuda_eagerly() -> None:
  dtype = runtime_dtype()
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=86,
    device="cpu",
    dtype=dtype,
  )
  parser = get_args(parse=False)
  args = maybe_postprocess_args(
    parser.parse_args([
      "--quantize-type",
      "svdq_int4_r32_dq",
      "--svdq-smooth-strategy",
      "few_shot",
      "--svdq-few-shot-steps",
      "1",
    ]))

  move_calls: list[torch.device] = []
  holder = SimpleNamespace(transformer=model)
  holder.to = lambda device: move_calls.append(torch.device(device))

  maybe_apply_optimization(args, holder)

  assert getattr(holder.transformer, "_svdq_pending_quantization", False)
  assert len(move_calls) == 1
  assert move_calls[0].type == "cuda"

  eval_inputs = make_token_batch(
    batch_size=2,
    seq_len=12,
    width=128,
    seed=87,
    device="cpu",
    dtype=dtype,
  )
  with torch.inference_mode():
    output = holder.transformer(eval_inputs)

  assert torch.isfinite(output).all()
  assert len(move_calls) == 1
  assert not hasattr(holder, "_svdq_move_to_device_after_forward")


def test_generic_module_offload_applies_to_non_diffusers_transformer_holder() -> None:
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=84,
    device="cpu",
    dtype=torch.float32,
  )
  parser = get_args(parse=False)
  args = maybe_postprocess_args(
    parser.parse_args([
      "--module-layerwise-cpu-offload",
      "--layerwise-async-transfer",
      "--layerwise-transfer-buckets",
      "2",
    ]))

  move_calls: list[torch.device] = []
  holder = SimpleNamespace(transformer=model)
  holder.to = lambda device: move_calls.append(torch.device(device))

  maybe_apply_optimization(args, holder)

  assert not hasattr(holder, "_cache_dit_generic_offload_handles")
  handles = get_layerwise_offload_handles(holder.transformer)
  assert len(handles) == 1
  assert handles[0].async_transfer is True
  assert handles[0].transfer_buckets == 2
  assert move_calls == []
  assert all(parameter.device.type == "cpu" for parameter in holder.transformer.parameters())

  eval_inputs = make_token_batch(
    batch_size=2,
    seq_len=12,
    width=128,
    seed=85,
    device="cpu",
    dtype=torch.float32,
  )
  with torch.inference_mode():
    output = holder.transformer(eval_inputs)
    torch.cuda.synchronize()

  assert torch.isfinite(output).all()
  assert output.device.type == "cpu"


def test_svdq_dq_few_shot_auto_compile_falls_back_to_module_compile() -> None:
  dtype = runtime_dtype()
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=80,
    device="cuda",
    dtype=dtype,
  )
  compile_calls: list[str] = []

  def _fake_compile(*args, **kwargs) -> None:
    compile_calls.append("module.compile")

  model.compile = _fake_compile  # type: ignore[attr-defined]
  quantized_model = cache_dit.quantize(
    model,
    _make_dq_config(
      rank=32,
      svdq_kwargs={
        "smooth_strategy": "few_shot",
        "few_shot_steps": 1,
        "few_shot_auto_compile": True,
      },
    ),
  )

  eval_inputs = make_token_batch(
    batch_size=2,
    seq_len=12,
    width=128,
    seed=81,
    device="cuda",
    dtype=dtype,
  )
  with torch.inference_mode():
    output = quantized_model(eval_inputs)
  assert torch.isfinite(output).all()
  assert compile_calls == ["module.compile"]
  assert float(getattr(quantized_model, "_svdq_runtime_quantize_time_s", 0.0)) > 0.0


def test_svdq_dq_weight_and_weight_inv_bias_difficulty_in_opposite_directions() -> None:
  dtype = runtime_dtype()
  float_model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=61,
    device="cuda",
    dtype=dtype,
  )
  linear = float_model.block.to_q
  weight = linear.weight.detach().to(device=linear.weight.device, dtype=linear.weight.dtype)
  weight_smooth = svdq_quantizer._resolve_svdq_smooth_scale(
    weight,
    None,
    quant_mode="dq",
    smooth_strategy="weight",
    in_features=linear.in_features,
    alpha=0.5,
    device=linear.weight.device,
    torch_dtype=linear.weight.dtype,
    math_dtype=linear.weight.dtype,
  )
  weight_inv_smooth = svdq_quantizer._resolve_svdq_smooth_scale(
    weight,
    None,
    quant_mode="dq",
    smooth_strategy="weight_inv",
    in_features=linear.in_features,
    alpha=0.5,
    device=linear.weight.device,
    torch_dtype=linear.weight.dtype,
    math_dtype=linear.weight.dtype,
  )

  weight_span = linear.weight.detach().float().abs().amax(dim=0)
  max_idx = int(weight_span.argmax().item())
  min_idx = int(weight_span.argmin().item())
  assert weight_smooth[max_idx] <= weight_smooth[min_idx]
  assert weight_inv_smooth[max_idx] >= weight_inv_smooth[min_idx]
  assert not torch.allclose(weight_smooth.float(), weight_inv_smooth.float(), rtol=0.0, atol=1e-3)


def test_svdq_dq_respects_exclude_layers() -> None:
  dtype = runtime_dtype()
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=31,
    device="cuda",
    dtype=dtype,
  )
  quantized_model = cache_dit.quantize(
    model,
    _make_dq_config(rank=32, exclude_layers=["to_out"]),
  )

  assert isinstance(quantized_model.block.to_q, SVDQW4A4Linear)
  assert isinstance(quantized_model.block.to_k, SVDQW4A4Linear)
  assert isinstance(quantized_model.block.to_v, SVDQW4A4Linear)
  assert isinstance(quantized_model.block.to_out, nn.Linear)


def test_svdq_dq_respects_filter_fn() -> None:
  dtype = runtime_dtype()
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=37,
    device="cuda",
    dtype=dtype,
  )
  quantized_model = cache_dit.quantize(
    model,
    _make_dq_config(
      rank=32,
      filter_fn=lambda name: name.endswith("to_q") or name.endswith("to_v"),
    ),
  )

  assert isinstance(quantized_model.block.to_q, SVDQW4A4Linear)
  assert isinstance(quantized_model.block.to_k, nn.Linear)
  assert isinstance(quantized_model.block.to_v, SVDQW4A4Linear)
  assert isinstance(quantized_model.block.to_out, nn.Linear)


def test_svdq_dq_regional_quantize_respects_repeated_blocks() -> None:
  dtype = runtime_dtype()
  model = ToyRepeatedBlocksModel().to(device="cuda", dtype=dtype).eval()
  quantized_model = cache_dit.quantize(model, _make_dq_config(rank=32))

  assert isinstance(quantized_model.blocks[0].to_q, SVDQW4A4Linear)
  assert isinstance(quantized_model.blocks[0].to_k, SVDQW4A4Linear)
  assert isinstance(quantized_model.blocks[0].to_v, SVDQW4A4Linear)
  assert isinstance(quantized_model.blocks[0].to_out, SVDQW4A4Linear)
  assert isinstance(quantized_model.blocks[1].to_q, SVDQW4A4Linear)
  assert isinstance(quantized_model.blocks[1].to_k, SVDQW4A4Linear)
  assert isinstance(quantized_model.blocks[1].to_v, SVDQW4A4Linear)
  assert isinstance(quantized_model.blocks[1].to_out, SVDQW4A4Linear)
  assert isinstance(quantized_model.head, nn.Linear)


def test_svdq_dq_enable_cache_quantizes_block_adapter_transformer() -> None:
  dtype = runtime_dtype()
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=47,
    device="cuda",
    dtype=dtype,
  )
  adapter = _make_normalized_transformer_adapter(model)

  adapter = cache_dit.enable_cache(
    adapter,
    quantize_config=_make_dq_config(rank=32),
  )
  quantized_model = adapter.transformer[0]

  assert isinstance(quantized_model.block.to_q, SVDQW4A4Linear)
  assert isinstance(quantized_model.block.to_k, SVDQW4A4Linear)
  assert isinstance(quantized_model.block.to_v, SVDQW4A4Linear)
  assert isinstance(quantized_model.block.to_out, SVDQW4A4Linear)


@pytest.mark.skipif(
  not _ENABLE_FLUX2_SLOW_TEST,
  reason="FLUX.2 SVDQ DQ integration test requires CACHE_DIT_SVDQ_TEST_FLUX2=1.",
)
def test_svdq_dq_flux2_klein_psnr_markdown_report() -> None:
  output_root = _REPO_TMP_DIR / "tests" / "svdq_flux2_dq"
  baseline = _run_flux2_case(
    prompts=_FLUX2_PROMPTS,
    output_dir=output_root / "baseline",
    quant_type=None,
  )

  prompt_rows: list[tuple[object, ...]] = []
  summary_rows: list[tuple[object, ...]] = []
  baseline_avg_latency_s = sum(float(row["latency_s"])
                               for row in baseline["rows"]) / len(baseline["rows"])

  for quant_type in ("svdq_int4_r32_dq", "svdq_int4_r128_dq"):
    case = _run_flux2_case(
      prompts=_FLUX2_PROMPTS,
      output_dir=output_root / quant_type,
      quant_type=quant_type,
    )
    psnr_values: list[float] = []
    quantized_avg_latency_s = sum(float(row["latency_s"])
                                  for row in case["rows"]) / len(case["rows"])

    for baseline_row, quantized_row in zip(baseline["rows"], case["rows"]):
      psnr_value, image_count = compute_psnr(
        str(baseline_row["image_path"]),
        str(quantized_row["image_path"]),
      )
      assert image_count == 1
      psnr_values.append(float(psnr_value))
      speedup = (float(baseline_row["latency_s"]) / float(quantized_row["latency_s"])
                 if float(quantized_row["latency_s"]) > 0.0 else float("inf"))
      prompt_rows.append((
        quant_type,
        int(baseline_row["prompt_index"]),
        str(baseline_row["prompt"]),
        f"{psnr_value:.4f}",
        f"{float(baseline_row['latency_s']):.4f}",
        f"{float(quantized_row['latency_s']):.4f}",
        ("inf" if speedup == float("inf") else f"{speedup:.4f}x"),
      ))

    avg_psnr = sum(psnr_values) / len(psnr_values)
    speedup_vs_baseline = (baseline_avg_latency_s / quantized_avg_latency_s
                           if quantized_avg_latency_s > 0.0 else float("inf"))
    summary_rows.append((
      quant_type,
      f"{avg_psnr:.4f}",
      f"{baseline_avg_latency_s:.4f}",
      f"{quantized_avg_latency_s:.4f}",
      ("inf" if speedup_vs_baseline == float("inf") else f"{speedup_vs_baseline:.4f}x"),
      f"{float(baseline['peak_memory_gib']):.4f}",
      f"{float(case['peak_memory_gib']):.4f}",
      f"{float(case['quantize_time_s']):.4f}",
    ))
    assert avg_psnr > _FLUX2_REFERENCE_PSNR_THRESHOLD

  prompt_table = format_markdown_table(
    "SVDQ DQ FLUX.2-Klein-4B prompt-level report",
    (
      "quant_type",
      "prompt_index",
      "prompt",
      "psnr",
      "baseline_latency_s",
      "quantized_latency_s",
      "speedup_vs_baseline",
    ),
    prompt_rows,
  )
  summary_table = format_markdown_table(
    "SVDQ DQ FLUX.2-Klein-4B summary",
    (
      "quant_type",
      "avg_psnr",
      "baseline_avg_latency_s",
      "quantized_avg_latency_s",
      "speedup_vs_baseline",
      "baseline_peak_memory_gib",
      "quantized_peak_memory_gib",
      "quantize_time_s",
    ),
    summary_rows,
  )
  report_path = output_root / "flux2_klein_4b_dq_psnr.md"
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(f"{summary_table}\n{prompt_table}", encoding="utf-8")


@pytest.mark.skipif(
  not _ENABLE_FLUX2_SLOW_TEST,
  reason="FLUX.2 SVDQ DQ integration test requires CACHE_DIT_SVDQ_TEST_FLUX2=1.",
)
def test_svdq_dq_flux2_klein_few_shot_relax_strategy_benchmark_report() -> None:
  output_root = _REPO_TMP_DIR / "tests" / "svdq_flux2_dq_few_shot"
  baseline = _run_flux2_case(
    prompts=_FLUX2_FEW_SHOT_BENCH_PROMPTS,
    output_dir=output_root / "baseline",
    quant_type=None,
  )
  baseline_avg_latency_s = sum(float(row["latency_s"])
                               for row in baseline["rows"]) / len(baseline["rows"])

  prompt_rows: list[tuple[object, ...]] = []
  summary_rows: list[tuple[object, ...]] = []
  for relax_strategy in ("top", "auto"):
    quantize_config = QuantizeConfig(
      quant_type="svdq_int4_r128_dq",
      svdq_kwargs={
        "smooth_strategy": "few_shot",
        "few_shot_steps": 1,
        "few_shot_relax_factor": 2.0,
        "few_shot_relax_top_ratio": 0.25,
        "few_shot_relax_strategy": relax_strategy,
      },
    )
    case = _run_flux2_few_shot_case(
      warmup_prompt=_FLUX2_FEW_SHOT_WARMUP_PROMPT,
      prompts=_FLUX2_FEW_SHOT_BENCH_PROMPTS,
      output_dir=output_root / relax_strategy,
      quantize_config=quantize_config,
    )
    psnr_values: list[float] = []
    quantized_avg_latency_s = sum(float(row["latency_s"])
                                  for row in case["rows"]) / len(case["rows"])

    for baseline_row, quantized_row in zip(baseline["rows"], case["rows"]):
      psnr_value, image_count = compute_psnr(
        str(baseline_row["image_path"]),
        str(quantized_row["image_path"]),
      )
      assert image_count == 1
      psnr_values.append(float(psnr_value))
      speedup = (float(baseline_row["latency_s"]) / float(quantized_row["latency_s"])
                 if float(quantized_row["latency_s"]) > 0.0 else float("inf"))
      prompt_rows.append((
        relax_strategy,
        int(baseline_row["prompt_index"]),
        str(baseline_row["prompt"]),
        f"{psnr_value:.4f}",
        f"{float(baseline_row['latency_s']):.4f}",
        f"{float(quantized_row['latency_s']):.4f}",
        ("inf" if speedup == float("inf") else f"{speedup:.4f}x"),
      ))

    avg_psnr = sum(psnr_values) / len(psnr_values)
    speedup_vs_baseline = (baseline_avg_latency_s / quantized_avg_latency_s
                           if quantized_avg_latency_s > 0.0 else float("inf"))
    summary_rows.append((
      relax_strategy,
      f"{avg_psnr:.4f}",
      f"{float(case['runtime_quantize_time_s']):.4f}",
      f"{float(case['warmup_latency_s']):.4f}",
      f"{baseline_avg_latency_s:.4f}",
      f"{quantized_avg_latency_s:.4f}",
      ("inf" if speedup_vs_baseline == float("inf") else f"{speedup_vs_baseline:.4f}x"),
      f"{float(case['warmup_peak_memory_gib']):.4f}",
      f"{float(case['benchmark_peak_memory_gib']):.4f}",
    ))
    assert avg_psnr > _FLUX2_REFERENCE_PSNR_THRESHOLD

  prompt_table = format_markdown_table(
    "SVDQ few-shot FLUX.2-Klein-4B prompt-level report",
    (
      "relax_strategy",
      "prompt_index",
      "prompt",
      "psnr_vs_baseline",
      "baseline_latency_s",
      "quantized_latency_s",
      "speedup_vs_baseline",
    ),
    prompt_rows,
  )
  summary_table = format_markdown_table(
    "SVDQ few-shot FLUX.2-Klein-4B strategy summary",
    (
      "relax_strategy",
      "avg_psnr_vs_baseline",
      "runtime_quantize_time_s",
      "warmup_latency_s",
      "baseline_avg_latency_s",
      "quantized_avg_latency_s",
      "speedup_vs_baseline",
      "warmup_peak_memory_gib",
      "quantized_peak_memory_gib",
    ),
    summary_rows,
  )
  report_path = output_root / "flux2_klein_4b_few_shot_relax_strategies.md"
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text(f"{summary_table}\n{prompt_table}", encoding="utf-8")

  assert report_path.is_file()
