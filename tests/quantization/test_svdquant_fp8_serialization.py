from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

import cache_dit
from cache_dit import QuantizeConfig
from cache_dit.kernels import cuda_ext_is_available
from cache_dit.quantization.svdquant.wNa8 import SVDQW8A8Linear

requires_fp8_cuda = pytest.mark.skipif(
  not torch.cuda.is_available() or not cuda_ext_is_available()
  or torch.cuda.get_device_capability(0)[0] < 12,
  reason="SVDQ-FP8 execution requires the CUDA extension on sm120+",
)


def _make_model() -> nn.Module:
  torch.manual_seed(17)
  return nn.Sequential(nn.Linear(128, 256, dtype=torch.bfloat16, device="cuda"),
                       nn.Linear(256, 128, dtype=torch.bfloat16, device="cuda"))


def _dq_config(path: Path, rank: int = 32) -> QuantizeConfig:
  return QuantizeConfig(quant_type=f"svdq_fp8_r{rank}_dq",
                        serialize_to=str(path),
                        svdq_kwargs={"smooth_strategy": "weight"})


def _assert_same_weights(quantized: nn.Module, loaded: nn.Module) -> None:
  for quantized_layer, loaded_layer in zip(quantized, loaded, strict=True):
    assert isinstance(loaded_layer, SVDQW8A8Linear)
    quantized_state = quantized_layer.state_dict()
    loaded_state = loaded_layer.state_dict()
    assert quantized_state.keys() == loaded_state.keys()
    for key, value in quantized_state.items():
      assert torch.equal(value.cpu(), loaded_state[key].cpu()), key


@pytest.mark.parametrize("rank", [0, 32, 128])
def test_fp8_state_dict_safetensors_roundtrip(tmp_path: Path, rank: int) -> None:
  torch.manual_seed(0)
  module = SVDQW8A8Linear(130, 192, rank=rank, bias=True)
  with torch.no_grad():
    module.qweight.copy_(torch.randn(module.qweight.shape).to(torch.float8_e4m3fn))
    module.wscales.copy_(torch.rand(module.wscales.shape) + 0.25)
    module.smooth_factor.copy_(torch.rand(module.smooth_factor.shape).bfloat16() + 0.5)
    if rank:
      module.proj_down.copy_(torch.randn(module.proj_down.shape).bfloat16())
      module.proj_up.copy_(torch.randn(module.proj_up.shape).bfloat16())
    module.bias.copy_(torch.randn(module.bias.shape).bfloat16())

  path = tmp_path / "module.safetensors"
  save_file(module.state_dict(), str(path))
  with safe_open(str(path), framework="pt", device="cpu") as handle:
    state = {key: handle.get_tensor(key) for key in handle.keys()}
  restored = SVDQW8A8Linear(130, 192, rank=rank, bias=True)
  restored.load_state_dict(state, strict=True)
  for key, value in module.state_dict().items():
    assert torch.equal(value, restored.state_dict()[key]), key


def test_fp8_cpu_public_api_checkpoint_roundtrip(tmp_path: Path) -> None:
  model = nn.Sequential(nn.Linear(130, 192, bias=False, dtype=torch.bfloat16),
                        nn.Linear(192, 130, dtype=torch.bfloat16))
  fresh_model = copy.deepcopy(model)
  config = QuantizeConfig(quant_type="svdq_fp8_r32_dq",
                          serialize_to=str(tmp_path / "cpu"),
                          svdq_kwargs={
                            "smooth_strategy": "weight",
                            "quantize_device": "cpu"
                          })
  quantized = cache_dit.quantize(model, config)
  loaded = cache_dit.load(fresh_model, str(tmp_path / "cpu"))
  _assert_same_weights(quantized, loaded)
  assert loaded[0].bias is None
  assert loaded[1].bias is not None
  assert loaded[0].in_features == 130 and loaded[0].out_features == 192
  assert cache_dit.quantize(loaded, config) is loaded


def test_fp8_ptq_skips_unobserved_layers_in_checkpoint(tmp_path: Path) -> None:
  model = nn.Sequential(nn.Linear(64, 128, dtype=torch.bfloat16),
                        nn.Linear(128, 64, dtype=torch.bfloat16))
  fresh_model = copy.deepcopy(model)

  def calibrate_fn(module: nn.Module) -> None:
    module[0](torch.randn(2, 64, dtype=torch.bfloat16))

  config = QuantizeConfig(quant_type="svdq_fp8_r32",
                          calibrate_fn=calibrate_fn,
                          serialize_to=str(tmp_path / "partial"),
                          svdq_kwargs={"quantize_device": "cpu"})
  quantized = cache_dit.quantize(model, config)
  assert isinstance(quantized[0], SVDQW8A8Linear)
  assert isinstance(quantized[1], nn.Linear)
  assert quantized._svdq_quantized_layers == ("0", )

  loaded = cache_dit.load(fresh_model, str(tmp_path / "partial"))
  assert isinstance(loaded[0], SVDQW8A8Linear)
  assert isinstance(loaded[1], nn.Linear)
  assert loaded._svdq_quantized_layers == ("0", )
  for key, value in quantized[0].state_dict().items():
    assert torch.equal(value, loaded[0].state_dict()[key]), key


def test_fp8_quantize_rejects_non_fp8_svdq_config() -> None:
  from cache_dit.quantization.svdquant.wNa8 import _quantize_svdq_w8a8
  from cache_dit.quantization.svdquant.wNaN import _quantize_svdq_wNaN

  with pytest.raises(ValueError, match="requires an svdq_fp8"):
    _quantize_svdq_wNaN(
      nn.Linear(64, 64, dtype=torch.bfloat16),
      QuantizeConfig(quant_type="svdq_int4_r32_dq"),
      _quantize_svdq_w8a8,
      "fp8",
    )


def test_w8a8_rejects_int8_source_weights() -> None:
  from cache_dit.quantization.svdquant.wNa8 import _quantize_svdq_w8a8

  linear = nn.Linear(8, 8, dtype=torch.bfloat16)
  linear.weight = nn.Parameter(torch.ones((8, 8), dtype=torch.int8), requires_grad=False)
  with pytest.raises(ValueError, match="W8A8 does not support INT8"):
    _quantize_svdq_w8a8(linear)


def test_fp8_converter_rejects_non_bf16_before_loading(tmp_path: Path) -> None:
  from cache_dit.quantization.converter import entrypoint

  with pytest.raises(SystemExit):
    entrypoint([
      "--model-path", "unused", "--save-dir",
      str(tmp_path), "--quant-type", "svdq-fp8-r32-dq", "--torch-dtype", "float16"
    ])


@requires_fp8_cuda
@pytest.mark.parametrize("rank", [32, 64, 128])
@pytest.mark.parametrize("load_kind", ["checkpoint", "directory", "config"])
def test_fp8_dq_checkpoint_roundtrip(tmp_path: Path, rank: int, load_kind: str) -> None:
  model = _make_model()
  fresh_model = copy.deepcopy(model)
  config = _dq_config(tmp_path / f"r{rank}", rank)
  quantized = cache_dit.quantize(model, config)
  checkpoint_path = Path(config.serialize_to)
  assert checkpoint_path.name == f"svdq_fp8_r{rank}_dq.safetensors"
  assert checkpoint_path.is_file()
  assert (checkpoint_path.parent / "quant_config.json").is_file()

  with safe_open(str(checkpoint_path), framework="pt", device="cpu") as handle:
    payload = json.loads(handle.metadata()["cache_dit_svdq_ptq"])
  assert payload["precision"] == "fp8"
  assert payload["quant_type"] == config.quant_type
  assert payload["rank"] == rank
  assert payload["quantized_layer_names"] == ["0", "1"]

  load_arg = config if load_kind == "config" else str(checkpoint_path if load_kind ==
                                                      "checkpoint" else checkpoint_path.parent)
  loaded = cache_dit.load(fresh_model, load_arg)
  _assert_same_weights(quantized, loaded)
  assert loaded._is_quantized
  assert loaded._quantize_type == config.quant_type
  assert loaded._svdq_checkpoint_path == str(checkpoint_path)

  x = torch.randn(3, 19, 128, device="cuda", dtype=torch.bfloat16)
  with torch.inference_mode():
    expected = quantized(x)
    actual = loaded(x)
  torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


@requires_fp8_cuda
def test_fp8_ptq_checkpoint_roundtrip(tmp_path: Path) -> None:
  model = _make_model()
  fresh_model = copy.deepcopy(model)
  samples = [torch.randn(2, 8, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]

  def calibrate_fn(module: nn.Module) -> None:
    with torch.inference_mode():
      for sample in samples:
        module(sample)

  config = QuantizeConfig(quant_type="svdq_fp8_r64",
                          calibrate_fn=calibrate_fn,
                          serialize_to=str(tmp_path / "ptq"))
  quantized = cache_dit.quantize(model, config)
  loaded = cache_dit.load(fresh_model, str(tmp_path / "ptq"))
  _assert_same_weights(quantized, loaded)
  assert not torch.all(quantized[0].smooth_factor[:128] == 1)

  with torch.inference_mode():
    expected = quantized(samples[0])
    actual = loaded(samples[0])
  torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


@requires_fp8_cuda
def test_fp8_root_linear_checkpoint_roundtrip(tmp_path: Path) -> None:
  model = nn.Linear(128, 128, dtype=torch.bfloat16, device="cuda")
  fresh_model = copy.deepcopy(model)
  config = _dq_config(tmp_path / "root")
  quantized = cache_dit.quantize(model, config)
  loaded = cache_dit.load(fresh_model, config.serialize_to)
  assert isinstance(loaded, SVDQW8A8Linear)
  assert loaded._is_quantized
  for key, value in quantized.state_dict().items():
    assert torch.equal(value.cpu(), loaded.state_dict()[key].cpu()), key


@requires_fp8_cuda
def test_fp8_checkpoint_rejects_mismatched_precision(tmp_path: Path) -> None:
  config = _dq_config(tmp_path / "source")
  cache_dit.quantize(_make_model(), config)
  with safe_open(config.serialize_to, framework="pt", device="cpu") as handle:
    tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    metadata = handle.metadata()
  payload = json.loads(metadata["cache_dit_svdq_ptq"])
  payload["precision"] = "int4"
  metadata["cache_dit_svdq_ptq"] = json.dumps(payload)
  bad_path = tmp_path / "bad" / Path(config.serialize_to).name
  bad_path.parent.mkdir()
  save_file(tensors, str(bad_path), metadata=metadata)

  with pytest.raises(ValueError, match="expected 'fp8'"):
    cache_dit.load(_make_model(), str(bad_path))


@requires_fp8_cuda
def test_fp8_loaded_model_compiles_fullgraph(tmp_path: Path) -> None:
  config = _dq_config(tmp_path / "compile")
  cache_dit.quantize(_make_model(), config)
  loaded = cache_dit.load(_make_model(), str(tmp_path / "compile"))
  compiled = torch.compile(loaded, fullgraph=True)
  x = torch.randn(2, 16, 128, device="cuda", dtype=torch.bfloat16)
  with torch.inference_mode():
    expected = loaded(x)
    actual = compiled(x)
  torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
