from __future__ import annotations

import copy
import gc
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
from cache_dit._utils.utils import maybe_quantize_transformer
from cache_dit._utils.utils import maybe_postprocess_args
from cache_dit.quantization import QuantizeConfig
from cache_dit.quantization.svdquant import SVDQW4A4Linear
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
    if quant_type is not None:
      quantize_start = time.perf_counter()
      pipe = cache_dit.enable_cache(
        pipe,
        quantize_config=QuantizeConfig(quant_type=quant_type),
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


@pytest.mark.parametrize("runtime_kernel", ["v2", "v3"])
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


@pytest.mark.parametrize("runtime_kernel", ["v2", "v3"])
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

  assert report_path.is_file()
