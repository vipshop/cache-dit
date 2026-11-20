"""Structural and numerical coverage for SVDQ fused-MLP passes."""

from __future__ import annotations

import types

import pytest
import torch
from torch import nn

from cache_dit.kernels import cuda_ext_is_available
from cache_dit.quantization.svdquant.dtensor import SVDQShardSpec
from cache_dit.quantization.svdquant.dtensor import SVDQW4A4ShardLinear
from cache_dit.quantization.svdquant.linear import SVDQW4A4Linear
from cache_dit.quantization.svdquant.passes import apply_passes
from cache_dit.quantization.svdquant.quantizer import quantize_linear_svdq_w4a4


def _svdq(in_features: int,
          out_features: int,
          *,
          bias: bool = False,
          runtime: str = "cute") -> SVDQW4A4Linear:
  return SVDQW4A4Linear(
    in_features,
    out_features,
    rank=32,
    bias=bias,
    precision="nvfp4",
    runtime_kernel=runtime,
    lora_layout="row_major" if runtime == "cute" else "native",
  )


class GELU(nn.Module):
  """Diffusers-shaped GELU wrapper with a projection child."""

  def __init__(self, proj: nn.Module, approximate: str) -> None:
    super().__init__()
    self.proj = proj
    self.approximate = approximate

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.gelu(self.proj(x), approximate=self.approximate)


class GEGLU(nn.Module):
  """Value-first packed GEGLU wrapper used by diffusers FeedForward."""

  def __init__(self, proj: nn.Module, approximate: str = "none") -> None:
    super().__init__()
    self.proj = proj
    self.approximate = approximate

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    value, gate = self.proj(x).chunk(2, dim=-1)
    return value * torch.nn.functional.gelu(gate, approximate=self.approximate)


class FeedForward(nn.Module):
  """Minimal diffusers FeedForward shape for pass detection."""

  def __init__(self,
               in_features: int,
               hidden: int,
               approximate: str,
               *,
               quantized: bool = True,
               runtime: str = "cute",
               activation: str = "gelu") -> None:
    super().__init__()
    fc1_out = 2 * hidden if activation == "geglu" else hidden
    fc1 = (_svdq(in_features, fc1_out, bias=True, runtime=runtime)
           if quantized else nn.Linear(in_features, fc1_out, bias=True))
    fc2 = (_svdq(hidden, in_features, bias=True, runtime=runtime)
           if quantized else nn.Linear(hidden, in_features, bias=True))
    act = GEGLU(fc1, approximate) if activation == "geglu" else GELU(fc1, approximate)
    self.net = nn.ModuleList([
      act,
      nn.Dropout(0.0),
      fc2,
    ])

  def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
    return self.net[2](self.net[0](hidden_states))


class SingleGeluBlock(nn.Module):

  def __init__(self,
               hidden: int = 512,
               approximate: str = "tanh",
               *,
               quantized: bool = True,
               runtime: str = "cute") -> None:
    super().__init__()
    self.proj_mlp = (_svdq(256, hidden, runtime=runtime)
                     if quantized else nn.Linear(256, hidden, bias=False))
    self.act_mlp = nn.GELU(approximate=approximate)
    self.proj_out = (_svdq(256 + hidden, 256, runtime=runtime)
                     if quantized else nn.Linear(256 + hidden, 256, bias=False))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.proj_out(torch.cat((x, self.act_mlp(self.proj_mlp(x))), dim=-1))


class DoubleGeluBlock(SingleGeluBlock):

  def __init__(self) -> None:
    super().__init__()
    self.act_extra = nn.GELU(approximate="tanh")


class LlamaStyleSwiGLU(nn.Module):
  """``w1/w3/w2`` gated MLP with functional SiLU but no activation attribute."""

  def __init__(self, hidden: int = 512, runtime: str = "cute", *, quantized: bool = True) -> None:
    super().__init__()
    if quantized:
      self.w1 = _svdq(256, hidden, runtime=runtime)
      self.w2 = _svdq(hidden, 256, runtime=runtime)
      self.w3 = _svdq(256, hidden, runtime=runtime)
    else:
      self.w1 = nn.Linear(256, hidden, bias=False)
      self.w2 = nn.Linear(hidden, 256, bias=False)
      self.w3 = nn.Linear(256, hidden, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))


class GateUpDownSwiGLU(nn.Module):
  """``gate_proj/up_proj/down_proj`` with functional SiLU attribute."""

  def __init__(self, hidden: int = 512, runtime: str = "cute", *, quantized: bool = True) -> None:
    super().__init__()
    self.activation_fn = torch.nn.functional.silu
    if quantized:
      self.gate_proj = _svdq(256, hidden, runtime=runtime)
      self.up_proj = _svdq(256, hidden, runtime=runtime)
      self.down_proj = _svdq(hidden, 256, runtime=runtime)
    else:
      self.gate_proj = nn.Linear(256, hidden, bias=False)
      self.up_proj = nn.Linear(256, hidden, bias=False)
      self.down_proj = nn.Linear(hidden, 256, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.down_proj(self.activation_fn(self.gate_proj(x)) * self.up_proj(x))


class Krea2StyleSwiGLU(nn.Module):
  """``gate/up/down`` projections with module SiLU."""

  def __init__(self, hidden: int = 512, runtime: str = "cute", *, quantized: bool = True) -> None:
    super().__init__()
    self.activation_fn = nn.SiLU()
    if quantized:
      self.gate = _svdq(256, hidden, runtime=runtime)
      self.up = _svdq(256, hidden, runtime=runtime)
      self.down = _svdq(hidden, 256, runtime=runtime)
    else:
      self.gate = nn.Linear(256, hidden, bias=False)
      self.up = nn.Linear(256, hidden, bias=False)
      self.down = nn.Linear(hidden, 256, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.down(self.activation_fn(self.gate(x)) * self.up(x))


class NonStandaloneGatedBlock(GateUpDownSwiGLU):
  """Gated projections inside a block that also owns attention."""

  def __init__(self) -> None:
    super().__init__()
    self.attn = nn.Linear(256, 256, bias=False)


class GegluMlp(GateUpDownSwiGLU):
  """Same projection names but GELU gating, not SwiGLU."""

  def __init__(self) -> None:
    super().__init__()
    self.activation_fn = None
    self.mlp_act = nn.GELU(approximate="tanh")

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.down_proj(self.up_proj(x) * self.mlp_act(self.gate_proj(x)))


class SequentialGeluMlp(nn.Module):

  def __init__(self, approximate: str = "tanh", *, quantized: bool = True) -> None:
    super().__init__()
    fc1 = _svdq(256, 512) if quantized else nn.Linear(256, 512, bias=False)
    fc2 = _svdq(512, 256) if quantized else nn.Linear(512, 256, bias=False)
    self.mlp = nn.Sequential(fc1, nn.GELU(approximate=approximate), nn.Dropout(0.0), fc2)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.mlp(x)


class AttributeGeluMlp(nn.Module):

  def __init__(self, approximate: str = "tanh", *, quantized: bool = True) -> None:
    super().__init__()
    self.fc1 = _svdq(256, 512) if quantized else nn.Linear(256, 512, bias=False)
    self.act = nn.GELU(approximate=approximate)
    self.drop1 = nn.Dropout(0.0)
    self.fc2 = _svdq(512, 256) if quantized else nn.Linear(512, 256, bias=False)
    self.drop2 = nn.Dropout(0.0)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class NonStandaloneGelu(AttributeGeluMlp):

  def __init__(self) -> None:
    super().__init__()
    self.attn = nn.Linear(256, 256, bias=False)


class SwiGLUBlock(nn.Module):

  def __init__(self, hidden: int = 512, runtime: str = "cute", *, quantized: bool = True) -> None:
    super().__init__()
    if quantized:
      self.proj = _svdq(256, hidden, runtime=runtime)
      self.gate_layer = _svdq(256, hidden, runtime=runtime)
      self.out = _svdq(hidden, 256, runtime=runtime)
    else:
      self.proj = nn.Linear(256, hidden, bias=False)
      self.gate_layer = nn.Linear(256, hidden, bias=False)
      self.out = nn.Linear(hidden, 256, bias=False)
    self.activation_fn = nn.SiLU()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.out(self.activation_fn(self.gate_layer(x)) * self.proj(x))


def _extra_forward(self, x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
  return x * scale


def test_gelu_mlp_pass_is_idempotent_and_preserves_state() -> None:
  block = FeedForward(256, 512, "tanh").eval()
  block.forward = types.MethodType(_extra_forward, block)
  state_keys = tuple(block.state_dict())
  original_forward = block.forward

  counts = apply_passes(block, ["fused_gelu_mlp"])
  assert counts == {"fused_gelu_mlp": 1}
  assert block._svdq_fused_mlp_targets == {
    "gelu_mlp": 1,
    "gelu_proj": 0,
    "swiglu": 0,
    "geglu_mlp": 0
  }
  assert block._svdq_fused_mlp_passes == counts
  assert block._svdq_fused_mlp_skipped == {"fused_gelu_mlp": {}}
  assert block._svdq_fused_mlp_pass == "fused_gelu_mlp"
  assert block._svdq_fused_mlp_original_forward == original_forward
  assert tuple(block.state_dict()) == state_keys

  patched_forward = block.forward
  x = torch.ones(2, 3, 256)
  assert torch.equal(block(x, scale=2.0), x * 2.0)

  assert apply_passes(block, ["fused_gelu_mlp"]) == {"fused_gelu_mlp": 0}
  assert block.forward is patched_forward
  assert block._svdq_fused_mlp_targets == {
    "gelu_mlp": 1,
    "gelu_proj": 0,
    "swiglu": 0,
    "geglu_mlp": 0
  }
  assert block._svdq_fused_mlp_passes == counts


def test_partial_pass_application_accumulates_target_counts() -> None:

  class _Composite(nn.Module):

    def __init__(self) -> None:
      super().__init__()
      self.ff = FeedForward(256, 512, "tanh")
      self.gated = SwiGLUBlock()

  block = _Composite().eval()
  assert apply_passes(block, ["fused_gelu_mlp"]) == {"fused_gelu_mlp": 1}
  assert apply_passes(block, ["fused_swiglu_mlp"]) == {"fused_swiglu_mlp": 1}
  assert block._svdq_fused_mlp_targets == {
    "gelu_mlp": 1,
    "gelu_proj": 0,
    "swiglu": 1,
    "geglu_mlp": 0
  }
  assert block._svdq_fused_mlp_passes == {"fused_gelu_mlp": 1, "fused_swiglu_mlp": 1}


def test_swiglu_alias_detection_and_standalone_guard() -> None:
  import cache_dit.quantization.svdquant.passes as passes

  cases = (
    (LlamaStyleSwiGLU, "w3", "w1"),
    (GateUpDownSwiGLU, "up_proj", "gate_proj"),
    (Krea2StyleSwiGLU, "up", "gate"),
  )
  for cls, up_name, gate_name in cases:
    block = cls().eval()
    targets = passes.get_pass("fused_swiglu_mlp")().detect(block)
    assert len(targets) == 1
    assert targets[0]["up"] is getattr(block, up_name)
    assert targets[0]["gate"] is getattr(block, gate_name)
    assert apply_passes(block, ["fused_swiglu_mlp"]) == {"fused_swiglu_mlp": 1}
    assert block._svdq_fused_mlp_pass == "fused_swiglu_mlp"

  non_standalone = NonStandaloneGatedBlock().eval()
  assert apply_passes(non_standalone, ["fused_swiglu_mlp"]) == {"fused_swiglu_mlp": 0}
  assert any("non_standalone_mlp" in reason
             for reason in non_standalone._svdq_fused_mlp_skipped["fused_swiglu_mlp"])

  geglu = GegluMlp().eval()
  assert apply_passes(geglu, ["fused_swiglu_mlp"]) == {"fused_swiglu_mlp": 0}
  assert any("non_silu_activation:mlp_act" in reason
             for reason in geglu._svdq_fused_mlp_skipped["fused_swiglu_mlp"])

  class _UnknownNames(nn.Module):

    def __init__(self) -> None:
      super().__init__()
      self.activation_fn = nn.SiLU()
      self.a = _svdq(256, 512)
      self.b = _svdq(256, 512)
      self.c = _svdq(512, 256)

  unknown = _UnknownNames().eval()
  assert apply_passes(unknown, ["fused_swiglu_mlp"]) == {"fused_swiglu_mlp": 0}
  assert any("unsupported_gated_names" in reason
             for reason in unknown._svdq_fused_mlp_skipped["fused_swiglu_mlp"])


def test_geglu_pass_distinguishes_exact_tanh_and_inference_guards() -> None:
  import cache_dit.quantization.svdquant.passes as passes

  for approximate, expected_act in (("none", "gelu"), ("tanh", "gelu_tanh")):
    block = FeedForward(256, 512, approximate, activation="geglu").eval()
    detected = passes.get_pass("fused_geglu_mlp")().detect(block)
    assert detected[0]["gate_act"] == expected_act
    assert detected[0]["gate_first"] is False
    assert apply_passes(block, ["fused_geglu_mlp"]) == {"fused_geglu_mlp": 1}
    assert block._svdq_fused_mlp_pass == "fused_geglu_mlp"
    assert block._svdq_fused_mlp_targets["geglu_mlp"] == 1
    assert block._svdq_fused_mlp_targets["swiglu"] == 0

  training = FeedForward(256, 512, "none", activation="geglu").train()
  assert apply_passes(training, ["fused_geglu_mlp"]) == {"fused_geglu_mlp": 0}
  assert any("training" in reason for reason in training._svdq_fused_mlp_skipped["fused_geglu_mlp"])

  dropout = FeedForward(256, 512, "none", activation="geglu").eval()
  dropout.net[1].p = 0.1
  assert apply_passes(dropout, ["fused_geglu_mlp"]) == {"fused_geglu_mlp": 0}
  assert any("active_dropout" in reason
             for reason in dropout._svdq_fused_mlp_skipped["fused_geglu_mlp"])


def test_flux_gate_first_requires_silu_and_standalone() -> None:
  from diffusers.models.transformers.transformer_flux2 import Flux2FeedForward
  import cache_dit.quantization.svdquant.passes as passes

  block = Flux2FeedForward(dim=256, inner_dim=512).eval()
  block.linear_in = _svdq(256, 1024)
  block.linear_out = _svdq(512, 256)
  block.act_fn.gate_fn = nn.GELU()
  assert passes.get_pass("fused_swiglu_mlp")().detect(block) == []

  block.act_fn.gate_fn = nn.SiLU()
  block.extra_norm = nn.LayerNorm(256)
  assert passes.get_pass("fused_swiglu_mlp")().detect(block) == []


def test_nested_gelu_detection_respects_tanh_and_standalone() -> None:
  sequential = SequentialGeluMlp().eval()
  assert apply_passes(sequential, ["fused_gelu_mlp"]) == {"fused_gelu_mlp": 1}
  assert sequential.mlp._svdq_fused_mlp_pass == "fused_gelu_mlp"

  exact_sequential = SequentialGeluMlp(approximate="none").eval()
  assert apply_passes(exact_sequential, ["fused_gelu_mlp"]) == {"fused_gelu_mlp": 0}
  assert any("exact_gelu" in reason
             for reason in exact_sequential._svdq_fused_mlp_skipped["fused_gelu_mlp"])

  attribute = AttributeGeluMlp().eval()
  assert apply_passes(attribute, ["fused_gelu_mlp"]) == {"fused_gelu_mlp": 1}
  assert attribute._svdq_fused_mlp_pass == "fused_gelu_mlp"

  exact_attribute = AttributeGeluMlp(approximate="none").eval()
  assert apply_passes(exact_attribute, ["fused_gelu_mlp"]) == {"fused_gelu_mlp": 0}
  assert any("exact_gelu" in reason
             for reason in exact_attribute._svdq_fused_mlp_skipped["fused_gelu_mlp"])

  non_standalone = NonStandaloneGelu().eval()
  assert apply_passes(non_standalone, ["fused_gelu_mlp"]) == {"fused_gelu_mlp": 0}
  assert any("non_standalone_mlp" in reason
             for reason in non_standalone._svdq_fused_mlp_skipped["fused_gelu_mlp"])


def test_gelu_mlp_pass_skips_exact_gelu_without_changing_forward(monkeypatch) -> None:
  import cache_dit.quantization.svdquant.passes as passes

  warnings = []
  monkeypatch.setattr(passes.logger, "warning",
                      lambda message, *args: warnings.append(message % args))
  block = FeedForward(256, 512, "none").eval()
  original_forward = block.forward

  counts = apply_passes(block, ["fused_gelu_mlp"])
  assert counts == {"fused_gelu_mlp": 0}
  assert block.forward == original_forward
  assert not hasattr(block, "_svdq_fused_mlp_pass")
  skipped = block._svdq_fused_mlp_skipped["fused_gelu_mlp"]
  assert any(reason.startswith("fused_gelu_mlp:exact_gelu") for reason in skipped)
  assert any("no supported MLP targets" in message for message in warnings)


def test_skip_warnings_are_aggregated_once(monkeypatch) -> None:
  import cache_dit.quantization.svdquant.passes as passes

  class _Mixed(nn.Module):

    def __init__(self) -> None:
      super().__init__()
      self.tanh = FeedForward(256, 512, "tanh")
      self.exact = FeedForward(256, 512, "none")

  warnings = []
  monkeypatch.setattr(passes.logger, "warning",
                      lambda message, *args: warnings.append(message % args))
  block = _Mixed().eval()
  assert apply_passes(block, ["fused_gelu_mlp"]) == {"fused_gelu_mlp": 1}
  assert len(warnings) == 1
  assert "skipped" in warnings[0] and "exact_gelu" in warnings[0]


def test_gelu_passes_skip_training_and_active_dropout() -> None:
  training = FeedForward(256, 512, "tanh").train()
  assert apply_passes(training, ["fused_gelu_mlp"]) == {"fused_gelu_mlp": 0}
  assert any("training" in reason for reason in training._svdq_fused_mlp_skipped["fused_gelu_mlp"])

  dropout = FeedForward(256, 512, "tanh").eval()
  dropout.net[1].p = 0.1
  assert apply_passes(dropout, ["fused_gelu_mlp"]) == {"fused_gelu_mlp": 0}
  assert any("active_dropout" in reason
             for reason in dropout._svdq_fused_mlp_skipped["fused_gelu_mlp"])

  double_gelu = FeedForward(256, 512, "tanh").eval()
  double_gelu.add_module("extra_gelu", nn.GELU(approximate="tanh"))
  assert apply_passes(double_gelu, ["fused_gelu_mlp"]) == {"fused_gelu_mlp": 0}
  assert any("ambiguous_gelu" in reason
             for reason in double_gelu._svdq_fused_mlp_skipped["fused_gelu_mlp"])

  projection = SingleGeluBlock().train()
  assert apply_passes(projection, ["fused_gelu_proj"]) == {"fused_gelu_proj": 0}
  projection.eval()
  projection.add_module("drop", nn.Dropout(0.1))
  assert apply_passes(projection, ["fused_gelu_proj"]) == {"fused_gelu_proj": 0}
  assert any("active_dropout" in reason
             for reason in projection._svdq_fused_mlp_skipped["fused_gelu_proj"])

  swiglu = SwiGLUBlock().eval()
  swiglu.add_module("drop", nn.Dropout(0.1))
  assert apply_passes(swiglu, ["fused_swiglu_mlp"]) == {"fused_swiglu_mlp": 0}
  assert any("active_dropout" in reason
             for reason in swiglu._svdq_fused_mlp_skipped["fused_swiglu_mlp"])


def test_gelu_projection_pass_requires_one_tanh_gelu_and_unique_pair() -> None:
  valid = SingleGeluBlock().eval()
  valid.proj_mlp.forward = types.MethodType(_extra_forward, valid.proj_mlp)
  counts = apply_passes(valid, ["fused_gelu_proj"])
  assert counts == {"fused_gelu_proj": 1}
  assert valid.proj_mlp._svdq_fused_mlp_pass == "fused_gelu_proj"
  x = torch.ones(2, 3, 256)
  assert torch.equal(
    valid.proj_mlp(x, scale=2.0),
    torch.nn.functional.gelu(x * 2.0, approximate="tanh"),
  )
  assert torch.equal(valid.act_mlp(x), x)
  assert apply_passes(valid, ["fused_gelu_proj"]) == {"fused_gelu_proj": 0}

  exact = SingleGeluBlock(approximate="none").eval()
  assert apply_passes(exact, ["fused_gelu_proj"]) == {"fused_gelu_proj": 0}
  assert any("exact_gelu" in reason for reason in exact._svdq_fused_mlp_skipped["fused_gelu_proj"])

  double = DoubleGeluBlock().eval()
  assert apply_passes(double, ["fused_gelu_proj"]) == {"fused_gelu_proj": 0}
  assert any("ambiguous_gelu" in reason
             for reason in double._svdq_fused_mlp_skipped["fused_gelu_proj"])


def test_fused_geometry_is_checked_before_patching() -> None:
  gelu = FeedForward(256, 640, "tanh").eval()
  assert apply_passes(gelu, ["fused_gelu_mlp"]) == {"fused_gelu_mlp": 0}
  assert any("not divisible by 256" in reason
             for reason in gelu._svdq_fused_mlp_skipped["fused_gelu_mlp"])

  projection = SingleGeluBlock(hidden=640).eval()
  assert apply_passes(projection, ["fused_gelu_proj"]) == {"fused_gelu_proj": 0}
  assert any("not divisible by 256" in reason
             for reason in projection._svdq_fused_mlp_skipped["fused_gelu_proj"])

  swiglu = SwiGLUBlock(hidden=640).eval()
  assert apply_passes(swiglu, ["fused_swiglu_mlp"]) == {"fused_swiglu_mlp": 1}
  misaligned_swiglu = SwiGLUBlock(hidden=1000).eval()
  assert apply_passes(misaligned_swiglu, ["fused_swiglu_mlp"]) == {"fused_swiglu_mlp": 0}
  assert any("not divisible by 64" in reason
             for reason in misaligned_swiglu._svdq_fused_mlp_skipped["fused_swiglu_mlp"])

  gate_mismatch = SwiGLUBlock(hidden=512).eval()
  gate_mismatch.gate_layer = _svdq(128, 512)
  assert apply_passes(gate_mismatch, ["fused_swiglu_mlp"]) == {"fused_swiglu_mlp": 0}
  assert any("gate K" in reason
             for reason in gate_mismatch._svdq_fused_mlp_skipped["fused_swiglu_mlp"])

  native_gelu = FeedForward(256, 640, "tanh", runtime="native").eval()
  assert apply_passes(native_gelu, ["fused_gelu_mlp"]) == {"fused_gelu_mlp": 1}
  native_projection = SingleGeluBlock(hidden=640, runtime="native").eval()
  assert apply_passes(native_projection, ["fused_gelu_proj"]) == {"fused_gelu_proj": 1}
  native_swiglu = SwiGLUBlock(hidden=640, runtime="native").eval()
  assert apply_passes(native_swiglu, ["fused_swiglu_mlp"]) == {"fused_swiglu_mlp": 1}

  geglu = FeedForward(256, 1000, "none", activation="geglu").eval()
  assert apply_passes(geglu, ["fused_geglu_mlp"]) == {"fused_geglu_mlp": 0}
  assert any("not divisible by 64" in reason
             for reason in geglu._svdq_fused_mlp_skipped["fused_geglu_mlp"])


def test_all_passes_skip_tp_shards_before_kernel_use() -> None:
  spec = SVDQShardSpec(
    mesh=object(),  # type: ignore[arg-type]
    placement=object(),  # type: ignore[arg-type]
    tp_size=2,
    tp_rank=0,
  )
  shard = SVDQW4A4ShardLinear(
    128,
    256,
    rank=32,
    tp_spec=spec,
    bias=False,
    precision="nvfp4",
    runtime_kernel="cute",
    lora_layout="row_major",
  )

  gelu = FeedForward(256, 512, "tanh").eval()
  gelu.net[0].proj = shard
  assert apply_passes(gelu, ["fused_gelu_mlp"]) == {"fused_gelu_mlp": 0}
  assert any("tp_sharded" in reason for reason in gelu._svdq_fused_mlp_skipped["fused_gelu_mlp"])

  projection = SingleGeluBlock().eval()
  projection.proj_mlp = shard
  assert apply_passes(projection, ["fused_gelu_proj"]) == {"fused_gelu_proj": 0}
  assert any("tp_sharded" in reason
             for reason in projection._svdq_fused_mlp_skipped["fused_gelu_proj"])

  swiglu = SwiGLUBlock().eval()
  swiglu.proj = shard
  assert apply_passes(swiglu, ["fused_swiglu_mlp"]) == {"fused_swiglu_mlp": 0}
  assert any("tp_sharded" in reason
             for reason in swiglu._svdq_fused_mlp_skipped["fused_swiglu_mlp"])

  geglu = FeedForward(256, 256, "none", activation="geglu").eval()
  geglu.net[0].proj = shard
  assert apply_passes(geglu, ["fused_geglu_mlp"]) == {"fused_geglu_mlp": 0}
  assert any("tp_sharded" in reason for reason in geglu._svdq_fused_mlp_skipped["fused_geglu_mlp"])


def test_swiglu_pass_skips_training_and_is_idempotent() -> None:
  training = SwiGLUBlock().train()
  assert apply_passes(training, ["fused_swiglu_mlp"]) == {"fused_swiglu_mlp": 0}
  assert any("training" in reason
             for reason in training._svdq_fused_mlp_skipped["fused_swiglu_mlp"])

  block = SwiGLUBlock().eval()
  block.forward = types.MethodType(_extra_forward, block)
  assert apply_passes(block, ["fused_swiglu_mlp"]) == {"fused_swiglu_mlp": 1}
  assert block._svdq_fused_mlp_pass == "fused_swiglu_mlp"
  x = torch.ones(2, 3, 256)
  assert torch.equal(block(x, scale=2.0), x * 2.0)
  assert apply_passes(block, ["fused_swiglu_mlp"]) == {"fused_swiglu_mlp": 0}


def _require_sm120() -> None:
  if not torch.cuda.is_available() or not cuda_ext_is_available():
    pytest.skip("SVDQ fused pass numerical tests require CUDA and the extension.")
  if torch.cuda.get_device_capability()[0] < 12:
    pytest.skip("SVDQ fused pass numerical tests require SM120.")


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
  return ((actual.float() - expected.float()).norm() /
          expected.float().norm().clamp_min(1e-12)).item()


def _quantize_projection(linear: nn.Linear,
                         calibration: torch.Tensor,
                         runtime: str = "native") -> SVDQW4A4Linear:
  state_dict = quantize_linear_svdq_w4a4(
    linear,
    [calibration.reshape(-1, linear.in_features)],
    rank=32,
    precision="nvfp4",
    device="cuda:0",
    torch_dtype=torch.bfloat16,
    return_state_dict=True,
  )
  layer = SVDQW4A4Linear(
    linear.in_features,
    linear.out_features,
    rank=32,
    bias=linear.bias is not None,
    precision="nvfp4",
    runtime_kernel=runtime,
    lora_layout="row_major" if runtime == "cute" else "native",
    torch_dtype=torch.bfloat16,
  ).cuda().eval()
  layer.load_state_dict(state_dict)
  return layer


def _capture_cuda_graph(fn, *, warmup: int = 2):
  side_stream = torch.cuda.Stream()
  side_stream.wait_stream(torch.cuda.current_stream())
  with torch.cuda.stream(side_stream):
    for _ in range(warmup):
      fn()
  torch.cuda.current_stream().wait_stream(side_stream)
  graph = torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):
    captured = fn()
  return graph, captured


def test_fused_gelu_mlp_pass_executes_roundtrips_and_captures_graph(monkeypatch) -> None:
  import cache_dit.quantization.svdquant.fused as fused_runtime

  _require_sm120()
  torch.manual_seed(20261130)
  block = FeedForward(256, 512, "tanh", quantized=False).cuda().to(torch.bfloat16).eval()
  x = torch.randn(1, 260, 256, dtype=torch.bfloat16, device="cuda:0")
  block.net[0].proj = _quantize_projection(block.net[0].proj, x)
  block.net[2] = _quantize_projection(
    block.net[2],
    torch.randn(1, 260, 512, dtype=torch.bfloat16, device="cuda:0"),
  )

  with torch.inference_mode():
    reference = block(x)

  original_fused = fused_runtime.fused_gelu_mlp
  fused_calls = []

  def _counted_fused(*args, **kwargs):
    fused_calls.append(1)
    return original_fused(*args, **kwargs)

  monkeypatch.setattr(fused_runtime, "fused_gelu_mlp", _counted_fused)
  assert apply_passes(block, ["fused_gelu_mlp"]) == {"fused_gelu_mlp": 1}
  with torch.inference_mode():
    fused = block(x)
  assert _relative_l2(fused, reference) < 5e-3
  cosine = torch.nn.functional.cosine_similarity(fused.flatten().float(),
                                                 reference.flatten().float(),
                                                 dim=0)
  assert cosine > 0.999
  assert fused_calls

  torch.compiler.reset()
  compiled = torch.compile(lambda value: block(value), fullgraph=False)
  with torch.inference_mode():
    compiled_out = compiled(x)
  assert _relative_l2(compiled_out, fused) < 5e-3

  with torch.inference_mode():
    graph, captured = _capture_cuda_graph(lambda: block(x))
    expected = block(x)
    graph.replay()
  assert _relative_l2(captured, expected) < 1e-3


def test_fused_gelu_proj_pass_matches_unfused_projection(monkeypatch) -> None:
  import cache_dit.quantization.svdquant.fused as fused_runtime

  _require_sm120()
  torch.manual_seed(20261131)
  block = SingleGeluBlock(quantized=False).cuda().to(torch.bfloat16).eval()
  x = torch.randn(1, 260, 256, dtype=torch.bfloat16, device="cuda:0")
  block.proj_mlp = _quantize_projection(block.proj_mlp, x)
  block.proj_out = _quantize_projection(
    block.proj_out,
    torch.randn(1, 260, 768, dtype=torch.bfloat16, device="cuda:0"),
  )

  with torch.inference_mode():
    reference = block(x)

  original_fused = fused_runtime.fused_gelu_proj
  fused_calls = []

  def _counted_fused(*args, **kwargs):
    fused_calls.append(1)
    return original_fused(*args, **kwargs)

  monkeypatch.setattr(fused_runtime, "fused_gelu_proj", _counted_fused)
  assert apply_passes(block, ["fused_gelu_proj"]) == {"fused_gelu_proj": 1}
  with torch.inference_mode():
    fused = block(x)
  assert _relative_l2(fused, reference) < 5e-3
  assert fused_calls


@pytest.mark.parametrize("runtime", ["native", "cute"])
def test_fused_swiglu_pass_accepts_producer_aligned_hidden(runtime: str, monkeypatch) -> None:
  import cache_dit.quantization.svdquant.fused as fused_runtime

  _require_sm120()
  torch.manual_seed(20261201)
  block = SwiGLUBlock(hidden=640, runtime=runtime, quantized=False).cuda().to(torch.bfloat16).eval()
  x = torch.randn(1, 260, 256, dtype=torch.bfloat16, device="cuda:0")
  block.proj = _quantize_projection(block.proj, x, runtime)
  block.gate_layer = _quantize_projection(block.gate_layer, x, runtime)
  block.out = _quantize_projection(
    block.out,
    torch.randn(1, 260, 640, dtype=torch.bfloat16, device="cuda:0"),
    runtime,
  )

  with torch.inference_mode():
    reference = block(x)

  original_fused = fused_runtime.fused_swiglu_mlp
  fused_calls = []

  def _counted_fused(*args, **kwargs):
    fused_calls.append(1)
    return original_fused(*args, **kwargs)

  monkeypatch.setattr(fused_runtime, "fused_swiglu_mlp", _counted_fused)
  assert apply_passes(block, ["fused_swiglu_mlp"]) == {"fused_swiglu_mlp": 1}
  with torch.inference_mode():
    fused = block(x)
  assert _relative_l2(fused, reference) < 5e-3
  assert fused_calls


@pytest.mark.parametrize("runtime", ["native", "cute"])
@pytest.mark.parametrize(
  "block_cls,up_name,gate_name,down_name",
  [
    (LlamaStyleSwiGLU, "w3", "w1", "w2"),
    (GateUpDownSwiGLU, "up_proj", "gate_proj", "down_proj"),
    (Krea2StyleSwiGLU, "up", "gate", "down"),
  ],
  ids=["w1-w3", "gate-up-proj", "gate-up"],
)
def test_gated_alias_pass_matches_unfused(block_cls, up_name: str, gate_name: str, down_name: str,
                                          runtime: str, monkeypatch) -> None:
  import cache_dit.quantization.svdquant.fused as fused_runtime

  _require_sm120()
  torch.manual_seed(20261202)
  block = block_cls(hidden=640, runtime=runtime, quantized=False).cuda().to(torch.bfloat16).eval()
  x = torch.randn(1, 260, 256, dtype=torch.bfloat16, device="cuda:0")
  setattr(block, up_name, _quantize_projection(getattr(block, up_name), x, runtime))
  setattr(block, gate_name, _quantize_projection(getattr(block, gate_name), x, runtime))
  setattr(
    block,
    down_name,
    _quantize_projection(
      getattr(block, down_name),
      torch.randn(1, 260, 640, dtype=torch.bfloat16, device="cuda:0"),
      runtime,
    ),
  )

  with torch.inference_mode():
    reference = block(x)

  original_fused = fused_runtime.fused_swiglu_mlp
  fused_calls = []

  def _counted_fused(*args, **kwargs):
    fused_calls.append(1)
    return original_fused(*args, **kwargs)

  monkeypatch.setattr(fused_runtime, "fused_swiglu_mlp", _counted_fused)
  assert apply_passes(block, ["fused_swiglu_mlp"]) == {"fused_swiglu_mlp": 1}
  with torch.inference_mode():
    fused = block(x)
  assert _relative_l2(fused, reference) < 5e-3
  cosine = torch.nn.functional.cosine_similarity(fused.flatten().float(),
                                                 reference.flatten().float(),
                                                 dim=0)
  assert cosine > 0.999
  assert fused_calls


@pytest.mark.parametrize("kind", ["sequential", "attributes"])
@pytest.mark.parametrize("runtime", ["native", "cute"])
def test_nested_gelu_pass_matches_unfused(kind: str, runtime: str, monkeypatch) -> None:
  import cache_dit.quantization.svdquant.fused as fused_runtime

  _require_sm120()
  torch.manual_seed(20261203)
  if kind == "sequential":
    block = SequentialGeluMlp(quantized=False).cuda().to(torch.bfloat16).eval()
    fc1 = block.mlp[0]
    fc2 = block.mlp[3]
  else:
    block = AttributeGeluMlp(quantized=False).cuda().to(torch.bfloat16).eval()
    fc1 = block.fc1
    fc2 = block.fc2
  x = torch.randn(1, 260, 256, dtype=torch.bfloat16, device="cuda:0")
  fc1 = _quantize_projection(fc1, x, runtime)
  fc2 = _quantize_projection(
    fc2,
    torch.randn(1, 260, 512, dtype=torch.bfloat16, device="cuda:0"),
    runtime,
  )
  if kind == "sequential":
    block.mlp[0] = fc1
    block.mlp[3] = fc2
  else:
    block.fc1 = fc1
    block.fc2 = fc2

  with torch.inference_mode():
    reference = block(x)

  original_fused = fused_runtime.fused_gelu_mlp
  fused_calls = []

  def _counted_fused(*args, **kwargs):
    fused_calls.append(1)
    return original_fused(*args, **kwargs)

  monkeypatch.setattr(fused_runtime, "fused_gelu_mlp", _counted_fused)
  assert apply_passes(block, ["fused_gelu_mlp"]) == {"fused_gelu_mlp": 1}
  with torch.inference_mode():
    fused = block(x)
  assert _relative_l2(fused, reference) < 5e-3
  cosine = torch.nn.functional.cosine_similarity(fused.flatten().float(),
                                                 reference.flatten().float(),
                                                 dim=0)
  assert cosine > 0.999
  assert fused_calls


@pytest.mark.parametrize("runtime", ["native", "cute"])
@pytest.mark.parametrize("approximate", ["none", "tanh"])
def test_geglu_pass_matches_unfused(approximate: str, runtime: str, monkeypatch) -> None:
  import cache_dit.quantization.svdquant.fused as fused_runtime

  _require_sm120()
  torch.manual_seed(20261204)
  block = FeedForward(256, 512, approximate, quantized=False,
                      activation="geglu").cuda().to(torch.bfloat16).eval()
  x = torch.randn(1, 260, 256, dtype=torch.bfloat16, device="cuda:0")
  block.net[0].proj = _quantize_projection(block.net[0].proj, x, runtime)
  block.net[2] = _quantize_projection(
    block.net[2],
    torch.randn(1, 260, 512, dtype=torch.bfloat16, device="cuda:0"),
    runtime,
  )

  with torch.inference_mode():
    reference = block(x)

  original_fused = fused_runtime.fused_swiglu_mlp
  fused_calls = []

  def _counted_fused(*args, **kwargs):
    fused_calls.append(1)
    assert kwargs["gate_act"] == ("gelu_tanh" if approximate == "tanh" else "gelu")
    assert kwargs["gate_first"] is False
    return original_fused(*args, **kwargs)

  monkeypatch.setattr(fused_runtime, "fused_swiglu_mlp", _counted_fused)
  assert apply_passes(block, ["fused_geglu_mlp"]) == {"fused_geglu_mlp": 1}
  with torch.inference_mode():
    fused = block(x)
  assert _relative_l2(fused, reference) < 5e-3
  cosine = torch.nn.functional.cosine_similarity(fused.flatten().float(),
                                                 reference.flatten().float(),
                                                 dim=0)
  assert cosine > 0.999
  assert fused_calls


@pytest.mark.parametrize("runtime", ["native", "cute"])
def test_diffusers_geglu_feedforward_pass_matches_unfused(runtime: str) -> None:
  from diffusers.models.attention import FeedForward as DiffusersFeedForward

  _require_sm120()
  torch.manual_seed(20261206)
  block = DiffusersFeedForward(dim=256, inner_dim=512,
                               activation_fn="geglu").cuda().to(torch.bfloat16).eval()
  x = torch.randn(1, 260, 256, dtype=torch.bfloat16, device="cuda:0")
  block.net[0].proj = _quantize_projection(block.net[0].proj, x, runtime)
  block.net[2] = _quantize_projection(
    block.net[2],
    torch.randn(1, 260, 512, dtype=torch.bfloat16, device="cuda:0"),
    runtime,
  )

  with torch.inference_mode():
    reference = block(x)
  assert apply_passes(block, ["fused_geglu_mlp"]) == {"fused_geglu_mlp": 1}
  with torch.inference_mode():
    fused = block(x)
  assert _relative_l2(fused, reference) < 5e-3
  cosine = torch.nn.functional.cosine_similarity(fused.flatten().float(),
                                                 reference.flatten().float(),
                                                 dim=0)
  assert cosine > 0.999


@pytest.mark.parametrize("runtime", ["native", "cute"])
@pytest.mark.parametrize("flux_version", [2, 3])
def test_flux_standalone_gate_first_pass_matches_unfused(flux_version: int, runtime: str,
                                                         monkeypatch) -> None:
  import cache_dit.quantization.svdquant.fused as fused_runtime
  import cache_dit.quantization.svdquant.passes as passes

  _require_sm120()
  torch.manual_seed(20261205)
  if flux_version == 2:
    from diffusers.models.transformers.transformer_flux2 import Flux2FeedForward

    block = Flux2FeedForward(dim=256, inner_dim=512).cuda().to(torch.bfloat16).eval()
  else:
    from diffusers.models.transformers.transformer_flux3 import Flux3FeedForward

    block = Flux3FeedForward(dim=256, inner_dim=512).cuda().to(torch.bfloat16).eval()
  x = torch.randn(1, 260, 256, dtype=torch.bfloat16, device="cuda:0")
  block.linear_in = _quantize_projection(block.linear_in, x, runtime)
  block.linear_out = _quantize_projection(
    block.linear_out,
    torch.randn(1, 260, 512, dtype=torch.bfloat16, device="cuda:0"),
    runtime,
  )

  with torch.inference_mode():
    reference = block(x)

  detected = passes.get_pass("fused_swiglu_mlp")().detect(block)
  assert len(detected) == 1
  assert detected[0]["gate_first"] is True
  assert detected[0]["gate"] is None

  original_fused = fused_runtime.fused_swiglu_mlp
  fused_calls = []

  def _counted_fused(*args, **kwargs):
    fused_calls.append(1)
    assert kwargs["gate_act"] == "silu"
    assert kwargs["gate_first"] is True
    return original_fused(*args, **kwargs)

  monkeypatch.setattr(fused_runtime, "fused_swiglu_mlp", _counted_fused)
  assert apply_passes(block, ["fused_swiglu_mlp"]) == {"fused_swiglu_mlp": 1}
  with torch.inference_mode():
    fused = block(x)
  assert _relative_l2(fused, reference) < 5e-3
  cosine = torch.nn.functional.cosine_similarity(fused.flatten().float(),
                                                 reference.flatten().float(),
                                                 dim=0)
  assert cosine > 0.999
  assert fused_calls
