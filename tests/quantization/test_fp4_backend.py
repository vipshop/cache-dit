"""Tests for cache-dit's native online NVFP4 backend and routing."""

import pytest
import torch
from torch import nn

from cache_dit.kernels import cuda_ext_is_available
from cache_dit.kernels import fp4_quantize
from cache_dit.quantization import QuantizeConfig
from cache_dit.quantization import load
from cache_dit.quantization import quantize
from cache_dit.quantization.backend import QuantizeBackend
from cache_dit.quantization.fp4.linear import NVFP4Linear

requires_native_fp4 = pytest.mark.skipif(
  not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12
  or not cuda_ext_is_available(),
  reason="cache-dit native FP4 requires sm120 and the compiled extension",
)


class _Block(nn.Module):
  """Tiny repeated block used by selection tests."""

  def __init__(self) -> None:
    super().__init__()
    self.proj = nn.Linear(256, 256, bias=True)
    self.bad_shape = nn.Linear(48, 24)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Run the supported projection.

    :param x: Input activation.
    :returns: Projected activation.
    """

    return self.proj(x)


class _Transformer(nn.Module):
  """Tiny transformer-shaped module with an excluded modulation branch."""

  def __init__(self) -> None:
    super().__init__()
    self._repeated_blocks = ["_Block"]
    self.blocks = nn.ModuleList([_Block(), _Block()])
    self.modulation = nn.Linear(256, 128)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Run every branch.

    :param x: Input activation.
    :returns: Output activation.
    """

    for block in self.blocks:
      x = block(x)
    return self.modulation(x)


class _MLPModel(nn.Module):

  def __init__(self, block: nn.Module) -> None:
    super().__init__()
    self.block = block

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.block(x)


class _SingleGeluBlock(nn.Module):

  def __init__(self) -> None:
    super().__init__()
    self.proj_mlp = nn.Linear(256, 1024)
    self.act_mlp = nn.GELU(approximate="tanh")
    self.proj_out = nn.Linear(1280, 256)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    mlp = self.act_mlp(self.proj_mlp(x))
    return self.proj_out(torch.cat((x, mlp), dim=-1))


class _DoubleGeluBlock(_SingleGeluBlock):

  def __init__(self) -> None:
    super().__init__()
    self.act_extra = nn.GELU(approximate="tanh")

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    mlp = self.act_extra(self.act_mlp(self.proj_mlp(x)))
    return self.proj_out(torch.cat((x, mlp), dim=-1))


class _SeparateSwigluBlock(nn.Module):

  def __init__(self, hidden: int = 1024) -> None:
    super().__init__()
    self.proj = nn.Linear(256, hidden)
    self.gate_layer = nn.Linear(256, hidden)
    self.out = nn.Linear(hidden, 256)
    self.activation_fn = nn.SiLU()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.out(self.activation_fn(self.gate_layer(x)) * self.proj(x))


def _fused_mlp_model(kind: str, hidden: int = 1024) -> _MLPModel:
  from diffusers.models.attention import FeedForward

  if kind == "gelu_mlp":
    block = FeedForward(dim=256, activation_fn="gelu-approximate")
  elif kind == "gelu_proj":
    block = _SingleGeluBlock()
  elif kind == "swiglu":
    block = _SeparateSwigluBlock(hidden)
  elif kind == "swiglu_half_split":
    block = FeedForward(dim=256, activation_fn="swiglu", inner_dim=hidden)
  else:
    raise ValueError(kind)
  return _MLPModel(block)


def _plain_state(module: nn.Module) -> dict:
  """Clone float weights before quantization.

  :param module: Float module.
  :returns: State dict copy.
  """

  return {key: value.detach().clone() for key, value in module.state_dict().items()}


@requires_native_fp4
def test_native_fp4_quantizes_only_supported_repeated_blocks() -> None:
  torch.manual_seed(0)
  model = _Transformer().cuda().to(torch.bfloat16).eval()
  out = quantize(
    model,
    QuantizeConfig(
      backend=QuantizeBackend.CACHE_DIT,
      quant_type="nvfp4",
      exclude_layers=["modulation"],
    ),
  )
  assert all(type(block.proj) is NVFP4Linear for block in out.blocks)
  assert all(type(block.bad_shape) is NVFP4Linear for block in out.blocks)
  assert isinstance(out.modulation, nn.Linear)
  assert out._quantize_type == "nvfp4"
  assert len(out._fp4_quantized_layers) == 4
  assert len(out._fp4_skipped_layers) == 0
  assert out._quantize_config.backend == QuantizeBackend.CACHE_DIT


@requires_native_fp4
def test_native_fp4_forward_tracks_bf16_reference() -> None:
  torch.manual_seed(1)
  model = _Transformer().cuda().to(torch.bfloat16).eval()
  reference = _Transformer().cuda().to(torch.bfloat16).eval()
  reference.load_state_dict(_plain_state(model))
  x = torch.randn(2, 256, dtype=torch.bfloat16, device="cuda")
  with torch.no_grad():
    expected = reference(x)
    out = quantize(
      model,
      QuantizeConfig(
        backend="cache_dit",
        quant_type="nvfp4",
        exclude_layers=["modulation"],
      ),
    )(x)
  diff = (out.float() - expected.float()).norm() / expected.float().norm()
  assert float(diff.detach()) < 0.35


@requires_native_fp4
def test_native_fp4_pads_and_depads_nonaligned_linear() -> None:
  torch.manual_seed(7)
  holder = nn.Module()
  holder.proj = nn.Linear(48, 24).cuda().to(torch.bfloat16)
  x = torch.randn(3, 48, dtype=torch.bfloat16, device="cuda")
  expected = holder.proj(x)
  quantize(holder, QuantizeConfig(backend="cache_dit", quant_type="nvfp4", regional_quantize=False))
  packed = holder.proj
  assert packed.qweight.shape == (128, 128)
  assert packed.dequantize_weight().shape == (24, 48)
  actual = packed(x)
  assert actual.shape == (3, 24)
  diff = (actual.float() - expected.float()).norm() / expected.float().norm()
  assert float(diff.detach()) < 0.35


@requires_native_fp4
def test_fp4_weight_dequantize_matches_quantized_linear() -> None:
  torch.manual_seed(2)
  holder = nn.Module()
  holder.proj = nn.Linear(256, 128).cuda().to(torch.bfloat16)
  quantize(holder, QuantizeConfig(backend="cache_dit", quant_type="nvfp4", regional_quantize=False))
  packed = holder.proj
  dense = packed.dequantize_weight()
  assert dense.shape == (128, 256)
  assert dense.dtype == torch.bfloat16
  x = torch.randn(3, 256, dtype=torch.bfloat16, device="cuda")
  qx, sx, gx = fp4_quantize(x)
  from cache_dit.quantization.fp4.linear import _dequantize_fp4

  activation = _dequantize_fp4(qx, sx, gx)
  expected = torch.nn.functional.linear(activation, dense.float(), packed.bias.float())
  diff = (packed(x) - expected).float().norm() / expected.float().norm()
  assert float(diff.detach()) < 5e-3


@requires_native_fp4
def test_native_fp4_compile_matches_eager() -> None:
  torch.manual_seed(3)
  holder = nn.Module()
  holder.proj = nn.Linear(256, 128).cuda().to(torch.bfloat16)
  quantize(holder, QuantizeConfig(backend="cache_dit", quant_type="nvfp4", regional_quantize=False))
  x = torch.randn(2, 4, 256, dtype=torch.bfloat16, device="cuda")
  expected = holder.proj(x)
  actual = torch.compile(holder.proj, fullgraph=True)(x)
  diff = (actual.float() - expected.float()).norm() / expected.float().norm()
  assert float(diff.detach()) < 5e-3


def test_backend_routing_and_validation() -> None:
  auto = QuantizeConfig(quant_type="nvfp4")
  assert auto.backend == QuantizeBackend.TORCHAO
  assert auto.strify() == "nvfp4"
  explicit = QuantizeConfig(backend="cache_dit", quant_type="nvfp4")
  assert explicit.strify() == "nvfp4_cache_dit"
  mixed = QuantizeConfig(
    quant_type="nvfp4",
    components_to_quantize={
      "transformer": {
        "quant_type": "nvfp4",
        "backend": "cache_dit",
      },
      "text_encoder": {
        "quant_type": "nvfp4",
        "backend": "torchao",
      },
    },
  )
  assert "<transformer:nvfp4_cache_dit>" in mixed.strify()
  assert "<text_encoder:nvfp4>" in mixed.strify()

  with pytest.raises(ValueError, match="three FP8 types and NVFP4"):
    QuantizeConfig(backend="cache_dit", quant_type="int8_per_row")
  with pytest.raises(ValueError, match="online-only"):
    QuantizeConfig(backend="cache_dit",
                   quant_type="nvfp4",
                   calibrate_fn=lambda **_: None,
                   serialize_to=".tmp/native_nvfp4/test-checkpoint")
  with pytest.raises(ValueError, match="three FP8 types and NVFP4"):
    QuantizeConfig(backend="cache_dit", quant_type="nvfp4_weight_only")


@requires_native_fp4
def test_native_fp4_rejects_a_selection_with_no_supported_linears() -> None:
  holder = nn.Module()
  holder.bad = nn.Linear(48, 24).cuda().float()
  with pytest.raises(ValueError, match="did not quantize any linear"):
    quantize(
      holder,
      QuantizeConfig(
        backend="cache_dit",
        quant_type="nvfp4",
        regional_quantize=False,
      ),
    )


@requires_native_fp4
def test_native_fp4_load_is_rejected() -> None:
  module = nn.Linear(256, 128).cuda().to(torch.bfloat16)
  config = QuantizeConfig(backend="cache_dit", quant_type="nvfp4")
  with pytest.raises(ValueError, match="online-only"):
    load(module, config)


@requires_native_fp4
@pytest.mark.parametrize("kind", ["gelu_mlp", "gelu_proj", "swiglu", "swiglu_half_split"])
def test_native_fp4_fused_mlp_matches_unfused_and_compiles(kind: str, monkeypatch) -> None:
  monkeypatch.setattr(torch._dynamo.config, "recompile_limit", 64)
  torch.manual_seed(51)
  model = _fused_mlp_model(kind).cuda().to(torch.bfloat16).eval()
  reference = _fused_mlp_model(kind).cuda().to(torch.bfloat16).eval()
  reference.load_state_dict(_plain_state(model))
  quantize(reference,
           QuantizeConfig(backend="cache_dit", quant_type="nvfp4", regional_quantize=False))
  quantize(
    model,
    QuantizeConfig(backend="cache_dit",
                   quant_type="nvfp4",
                   regional_quantize=False,
                   fp4_fused_mlp=True))
  expected_pass = "swiglu" if kind.startswith("swiglu") else kind
  assert model._fp4_fused_mlp_targets[expected_pass] == 1
  assert sum(model._fp4_fused_mlp_targets.values()) == 1

  x = torch.randn(2, 3, 256, dtype=torch.bfloat16, device="cuda")
  with torch.no_grad():
    expected = reference(x)
    actual = model(x)
  rel_l2 = ((actual.float() - expected.float()).norm() / expected.float().norm().clamp_min(1e-12))
  assert float(rel_l2) < 2e-3

  compiled = torch.compile(lambda value: model(value), fullgraph=True)
  with torch.no_grad():
    compiled_actual = compiled(x)
  compiled_rel_l2 = ((compiled_actual.float() - expected.float()).norm() /
                     expected.float().norm().clamp_min(1e-12))
  assert float(compiled_rel_l2) < 2e-3


@requires_native_fp4
@pytest.mark.parametrize("kind", ["swiglu", "swiglu_half_split"])
def test_native_fp4_fused_swiglu_pads_unaligned_hidden(kind: str, monkeypatch) -> None:
  monkeypatch.setattr(torch._dynamo.config, "recompile_limit", 64)
  torch.manual_seed(52)
  model = _fused_mlp_model(kind, hidden=960).cuda().to(torch.bfloat16).eval()
  reference = _fused_mlp_model(kind, hidden=960).cuda().to(torch.bfloat16).eval()
  reference.load_state_dict(_plain_state(model))
  quantize(reference,
           QuantizeConfig(backend="cache_dit", quant_type="nvfp4", regional_quantize=False))
  quantize(
    model,
    QuantizeConfig(backend="cache_dit",
                   quant_type="nvfp4",
                   regional_quantize=False,
                   fp4_fused_mlp=True))
  x = torch.randn(2, 3, 256, dtype=torch.bfloat16, device="cuda")
  with torch.no_grad():
    expected = reference(x)
    actual = model(x)
    compiled = torch.compile(lambda value: model(value), fullgraph=True)
    compiled_actual = compiled(x)
  assert torch.equal(actual, expected)
  assert torch.equal(compiled_actual, expected)


@requires_native_fp4
def test_native_fp4_fused_mlp_regional_and_idempotent() -> None:
  from cache_dit.quantization.fp4.passes import apply_fused_mlp_passes

  model = _fused_mlp_model("gelu_mlp").cuda().to(torch.bfloat16).eval()
  model._repeated_blocks = ["FeedForward"]
  quantize(model, QuantizeConfig(backend="cache_dit", quant_type="nvfp4", fp4_fused_mlp=True))
  assert model._fp4_fused_mlp_targets["gelu_mlp"] == 1
  original_forward = model.block.forward
  assert sum(apply_fused_mlp_passes(model).values()) == 0
  assert model.block.forward is original_forward
  x = torch.randn(2, 256, dtype=torch.bfloat16, device="cuda")
  assert model(x).shape == (2, 256)


@requires_native_fp4
def test_native_fp4_fused_mlp_structure_guards() -> None:
  from cache_dit.quantization.fp4.passes import apply_fused_mlp_passes

  config = QuantizeConfig(backend="cache_dit", quant_type="nvfp4", regional_quantize=False)
  double_gelu = _MLPModel(_DoubleGeluBlock()).cuda().to(torch.bfloat16).eval()
  quantize(double_gelu, config)
  assert sum(apply_fused_mlp_passes(double_gelu).values()) == 0

  dropout = _fused_mlp_model("gelu_mlp").cuda().to(torch.bfloat16).eval()
  dropout.block.net[1].p = 0.1
  quantize(dropout, config)
  assert sum(apply_fused_mlp_passes(dropout).values()) == 0

  training = _fused_mlp_model("gelu_mlp").cuda().to(torch.bfloat16).train()
  quantize(training, config)
  assert sum(apply_fused_mlp_passes(training).values()) == 0


@requires_native_fp4
def test_native_fp4_fused_mlp_rejects_zero_targets() -> None:
  model = _Transformer().cuda().to(torch.bfloat16).eval()
  with pytest.raises(ValueError, match="found no supported"):
    quantize(model, QuantizeConfig(backend="cache_dit", quant_type="nvfp4", fp4_fused_mlp=True))


def test_fp4_fused_mlp_config_strify_and_validation() -> None:
  config = QuantizeConfig(
    backend="cache_dit",
    quant_type="nvfp4",
    fp4_fused_mlp=True,
    components_to_quantize={"transformer": {
      "fp4_fused_mlp": True
    }},
  )
  assert config.strify() == "<transformer:nvfp4_cache_dit_fused_mlp>"
  expanded = QuantizeConfig.expand_configs(config)
  assert len(expanded) == 1 and expanded[0].fp4_fused_mlp is True
  with pytest.raises(ValueError, match="fp4_fused_mlp=True requires"):
    QuantizeConfig(backend="torchao", quant_type="nvfp4", fp4_fused_mlp=True)
  with pytest.raises(ValueError, match="fp4_fused_mlp=True requires"):
    QuantizeConfig(backend="cache_dit", quant_type="float8_per_row", fp4_fused_mlp=True)
