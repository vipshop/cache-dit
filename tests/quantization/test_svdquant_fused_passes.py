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


class FeedForward(nn.Module):
  """Minimal diffusers FeedForward shape for pass detection."""

  def __init__(self,
               in_features: int,
               hidden: int,
               approximate: str,
               *,
               quantized: bool = True,
               runtime: str = "cute") -> None:
    super().__init__()
    fc1 = (_svdq(in_features, hidden, bias=True, runtime=runtime)
           if quantized else nn.Linear(in_features, hidden, bias=True))
    fc2 = (_svdq(hidden, in_features, bias=True, runtime=runtime)
           if quantized else nn.Linear(hidden, in_features, bias=True))
    self.net = nn.ModuleList([
      GELU(fc1, approximate),
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
  assert block._svdq_fused_mlp_targets == {"gelu_mlp": 1, "gelu_proj": 0, "swiglu": 0}
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
  assert block._svdq_fused_mlp_targets == {"gelu_mlp": 1, "gelu_proj": 0, "swiglu": 0}
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
  assert block._svdq_fused_mlp_targets == {"gelu_mlp": 1, "gelu_proj": 0, "swiglu": 1}
  assert block._svdq_fused_mlp_passes == {"fused_gelu_mlp": 1, "fused_swiglu_mlp": 1}


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
