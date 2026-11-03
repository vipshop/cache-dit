import pytest
import torch

from cache_dit.quantization import QuantizeConfig
from cache_dit.quantization import QuantizationObserverConfig
from cache_dit.quantization.observers import QuantizeObserver


class _Block(torch.nn.Module):

  def __init__(self, dim: int = 16):
    super().__init__()
    self.norm1 = torch.nn.Linear(dim, 4 * dim, bias=True)
    self.feed_forward = torch.nn.Sequential(
      torch.nn.Linear(dim, 2 * dim, bias=True),
      torch.nn.GELU(),
      torch.nn.Linear(2 * dim, dim, bias=True),
    )


class _TinyDiT(torch.nn.Module):

  def __init__(self, layers: int = 2, dim: int = 16):
    super().__init__()
    self.single_stream_layers = torch.nn.ModuleList([_Block(dim) for _ in range(layers)])

  def forward(self, x):
    for block in self.single_stream_layers:
      x = block.feed_forward(x) + block.norm1(x)
    return x


class _FakeSVDQLinear(torch.nn.Module):
  """Stand-in for the SVDQuant runtime layer (packed weights, no `weight`)."""

  def __init__(self, dim: int = 16, out_features: int = 64):
    super().__init__()
    self.qweight = torch.nn.Parameter(torch.zeros(dim, dim, dtype=torch.int8), requires_grad=False)
    self.dense = torch.nn.Parameter(torch.zeros(out_features, dim, dtype=torch.bfloat16),
                                    requires_grad=False)
    self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.bfloat16),
                                   requires_grad=False)

  def forward(self, x):
    return torch.nn.functional.linear(x, self.dense, self.bias)


def test_observer_is_off_by_default() -> None:
  config = QuantizeConfig(quant_type="float8_per_row")

  assert config.observer is None
  assert QuantizeObserver(config.observer).enabled is False


def test_observer_requires_explicit_enable() -> None:
  assert QuantizeObserver(QuantizationObserverConfig()).enabled is False
  assert QuantizeObserver(QuantizationObserverConfig(enabled=True)).enabled is True


def test_observer_accepts_dict_from_cli_callers() -> None:
  config = QuantizeConfig(
    quant_type="float8_per_row",
    observer={
      "enabled": True,
      "json_path": ".tmp/tests/observers.json",
      "layer_name_patterns": [r"feed_forward"],
    },
  )

  assert isinstance(config.observer, QuantizationObserverConfig)
  assert config.observer.enabled is True
  assert config.observer.matches("single_stream_layers.0.feed_forward.linear_1")
  assert not config.observer.matches("single_stream_layers.0.norm1.linear")


def test_observer_single_string_pattern_is_normalized() -> None:
  config = QuantizationObserverConfig(enabled=True, layer_name_patterns=r"norm1\.linear$")

  assert config.layer_name_patterns == [r"norm1\.linear$"]
  assert config.matches("single_stream_layers.3.norm1.linear")
  assert not config.matches("single_stream_layers.3.feed_forward.linear_1")


@pytest.mark.parametrize("pattern,expected", [
  (r"feed_forward", True),
  (r"feed_forward\.linear_1$", True),
  (r"^single_stream_layers\.1\.", True),
  (r"linear_[23]$", False),
  (r"double_stream", False),
  (r"feed_forward\.linear_3$", False),
])
def test_observer_regex_matching(pattern: str, expected: bool) -> None:
  config = QuantizationObserverConfig(enabled=True, layer_name_patterns=[pattern])

  assert config.matches("single_stream_layers.1.feed_forward.linear_1") is expected


def test_observer_matches_everything_without_patterns() -> None:
  config = QuantizationObserverConfig(enabled=True)

  assert config.matches("anything.at.all") is True
  assert QuantizationObserverConfig(enabled=True,
                                    layer_name_patterns=[]).matches("anything.at.all") is True


def test_observer_rejects_invalid_regex() -> None:
  with pytest.raises(ValueError, match="invalid layer_name_patterns regex"):
    QuantizationObserverConfig(enabled=True, layer_name_patterns=["[unclosed"])


def test_capture_references_skips_everything_when_disabled() -> None:
  model = _TinyDiT().to(torch.bfloat16)
  observer = QuantizeObserver(QuantizationObserverConfig())

  assert observer.capture_references(model) == {}
  assert QuantizeObserver(None).capture_references(model) == {}


def test_capture_references_honors_regex_filter() -> None:
  model = _TinyDiT().to(torch.bfloat16)
  observer = QuantizeObserver(
    QuantizationObserverConfig(enabled=True, layer_name_patterns=[r"feed_forward"]))

  refs = observer.capture_references(model)

  assert refs
  assert all("feed_forward" in name for name in refs)
  assert not any("norm1" in name for name in refs)
  weight, bias = refs["single_stream_layers.0.feed_forward.0"]
  assert weight.dtype == torch.bfloat16
  assert bias is not None


def test_capture_references_keeps_all_layers_without_filter() -> None:
  model = _TinyDiT().to(torch.bfloat16)
  observer = QuantizeObserver(QuantizationObserverConfig(enabled=True))

  refs = observer.capture_references(model)

  assert len(refs) == 2 * 3


def test_observer_attaches_to_svdq_runtime_layers() -> None:
  observer = QuantizeObserver(
    QuantizationObserverConfig(enabled=True,
                               max_steps=0,
                               json_path=".tmp/tests/observers_svdq.json"))
  model = _TinyDiT().to(torch.bfloat16)
  refs = observer.capture_references(model)

  # SVDQuant replaces the nn.Linear with a packed runtime layer in place.
  model.single_stream_layers[0].norm1 = _FakeSVDQLinear(16, 64)

  installed = observer.attach(model, refs)

  # Only the replaced (quantized) layer is observed; the remaining float
  # linears are skipped.
  assert installed == 1
  model.single_stream_layers[0].norm1(torch.randn(4, 16, dtype=torch.bfloat16))


def test_observer_skips_plain_float_linears() -> None:
  observer = QuantizeObserver(QuantizationObserverConfig(enabled=True, max_steps=0))
  model = _TinyDiT().to(torch.bfloat16)
  refs = observer.capture_references(model)

  # Nothing is quantized yet, so no hook must be installed.
  assert observer.attach(model, refs) == 0
