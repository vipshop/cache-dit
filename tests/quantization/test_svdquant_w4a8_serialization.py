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
from cache_dit.quantization.svdquant.wNa8 import SVDQW4A8Linear

requires_w4a8_cuda = pytest.mark.skipif(
  not torch.cuda.is_available() or not cuda_ext_is_available()
  or torch.cuda.get_device_capability(0)[0] < 12,
  reason="SVDQ-W4A8 execution requires the CUDA extension on sm120+",
)


def _make_model() -> nn.Module:
  torch.manual_seed(17)
  return nn.Sequential(nn.Linear(128, 256, dtype=torch.bfloat16, device="cuda"),
                       nn.Linear(256, 128, dtype=torch.bfloat16, device="cuda"))


def _assert_same_weights(quantized: nn.Module, loaded: nn.Module) -> None:
  for quantized_layer, loaded_layer in zip(quantized, loaded, strict=True):
    assert isinstance(loaded_layer, SVDQW4A8Linear)
    for key, value in quantized_layer.state_dict().items():
      assert torch.equal(value.cpu(), loaded_layer.state_dict()[key].cpu()), key


@pytest.mark.parametrize("rank", [0, 32, 128])
def test_w4a8_state_dict_safetensors_roundtrip(tmp_path: Path, rank: int) -> None:
  torch.manual_seed(0)
  module = SVDQW4A8Linear(130, 192, rank=rank, bias=True)
  with torch.no_grad():
    module.qweight.copy_(torch.randint(0, 256, module.qweight.shape, dtype=torch.uint8))
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
  restored = SVDQW4A8Linear(130, 192, rank=rank, bias=True)
  restored.load_state_dict(state, strict=True)
  for key, value in module.state_dict().items():
    assert torch.equal(value, restored.state_dict()[key]), key


def test_w4a8_cpu_public_api_checkpoint_roundtrip(tmp_path: Path) -> None:
  model = nn.Sequential(nn.Linear(130, 192, bias=False, dtype=torch.bfloat16),
                        nn.Linear(192, 130, dtype=torch.bfloat16))
  fresh_model = copy.deepcopy(model)
  config = QuantizeConfig(quant_type="svdq_w4a8_r32_dq",
                          serialize_to=str(tmp_path / "cpu"),
                          svdq_kwargs={
                            "smooth_strategy": "weight",
                            "quantize_device": "cpu"
                          })
  quantized = cache_dit.quantize(model, config)
  loaded = cache_dit.load(fresh_model, str(tmp_path / "cpu"))
  _assert_same_weights(quantized, loaded)
  assert loaded[0].bias is None and loaded[1].bias is not None
  assert loaded[0].qweight.shape == (256, 128)


@requires_w4a8_cuda
@pytest.mark.parametrize("rank", [32, 64, 128])
@pytest.mark.parametrize("load_kind", ["checkpoint", "directory", "config"])
def test_w4a8_dq_checkpoint_roundtrip(tmp_path: Path, rank: int, load_kind: str) -> None:
  model = _make_model()
  fresh_model = copy.deepcopy(model)
  config = QuantizeConfig(quant_type=f"svdq_w4a8_r{rank}_dq", serialize_to=str(tmp_path / "dq"))
  quantized = cache_dit.quantize(model, config)
  checkpoint = Path(config.serialize_to)
  with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
    payload = json.loads(handle.metadata()["cache_dit_svdq_ptq"])
  assert payload["precision"] == "w4a8" and payload["rank"] == rank
  assert payload["quantized_layer_names"] == ["0", "1"]
  load_arg = config if load_kind == "config" else str(checkpoint if load_kind ==
                                                      "checkpoint" else checkpoint.parent)
  loaded = cache_dit.load(fresh_model, load_arg)
  _assert_same_weights(quantized, loaded)
  x = torch.randn(3, 19, 128, device="cuda", dtype=torch.bfloat16)
  with torch.inference_mode():
    expected = quantized(x)
    actual = loaded(x)
  torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


@requires_w4a8_cuda
def test_w4a8_ptq_checkpoint_roundtrip(tmp_path: Path) -> None:
  model = _make_model()
  fresh_model = copy.deepcopy(model)
  samples = [torch.randn(2, 8, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3)]

  def calibrate_fn(module: nn.Module) -> None:
    with torch.inference_mode():
      for sample in samples:
        module(sample)

  config = QuantizeConfig(quant_type="svdq_w4a8_r64",
                          calibrate_fn=calibrate_fn,
                          serialize_to=str(tmp_path / "ptq"))
  quantized = cache_dit.quantize(model, config)
  loaded = cache_dit.load(fresh_model, str(tmp_path / "ptq"))
  _assert_same_weights(quantized, loaded)
  assert not torch.all(quantized[0].smooth_factor[:128] == 1)


@requires_w4a8_cuda
def test_w4a8_root_linear_checkpoint_roundtrip(tmp_path: Path) -> None:
  model = nn.Linear(128, 128, dtype=torch.bfloat16, device="cuda")
  fresh_model = copy.deepcopy(model)
  config = QuantizeConfig(quant_type="svdq_w4a8_r32_dq", serialize_to=str(tmp_path / "root"))
  quantized = cache_dit.quantize(model, config)
  loaded = cache_dit.load(fresh_model, config.serialize_to)
  assert isinstance(loaded, SVDQW4A8Linear)
  for key, value in quantized.state_dict().items():
    assert torch.equal(value.cpu(), loaded.state_dict()[key].cpu()), key


@requires_w4a8_cuda
def test_w4a8_checkpoint_rejects_mismatched_precision(tmp_path: Path) -> None:
  config = QuantizeConfig(quant_type="svdq_w4a8_r32_dq", serialize_to=str(tmp_path / "source"))
  cache_dit.quantize(_make_model(), config)
  with safe_open(config.serialize_to, framework="pt", device="cpu") as handle:
    tensors = {key: handle.get_tensor(key) for key in handle.keys()}
    metadata = handle.metadata()
  payload = json.loads(metadata["cache_dit_svdq_ptq"])
  payload["precision"] = "fp8"
  metadata["cache_dit_svdq_ptq"] = json.dumps(payload)
  bad_path = tmp_path / "bad" / Path(config.serialize_to).name
  bad_path.parent.mkdir()
  save_file(tensors, str(bad_path), metadata=metadata)
  with pytest.raises(ValueError, match="expected 'w4a8'"):
    cache_dit.load(_make_model(), str(bad_path))
