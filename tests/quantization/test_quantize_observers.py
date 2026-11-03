import pytest
import torch

from cache_dit.quantization import QuantizeConfig
from cache_dit.quantization import QuantizationObserverConfig
from cache_dit.quantization.observers import capture_references, is_enabled


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


def test_observer_is_off_by_default() -> None:
  config = QuantizeConfig(quant_type="float8_per_row")

  assert config.observer is None
  assert is_enabled(config.observer) is False


def test_observer_requires_explicit_enable() -> None:
  disabled = QuantizationObserverConfig()
  enabled = QuantizationObserverConfig(enabled=True)

  assert is_enabled(disabled) is False
  assert is_enabled(enabled) is True


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

  assert capture_references(model, None) == {}
  assert capture_references(model, QuantizationObserverConfig()) == {}


def test_capture_references_honors_regex_filter() -> None:
  model = _TinyDiT().to(torch.bfloat16)
  config = QuantizationObserverConfig(enabled=True, layer_name_patterns=[r"feed_forward"])

  refs = capture_references(model, config)

  assert refs
  assert all("feed_forward" in name for name in refs)
  assert not any("norm1" in name for name in refs)
  weight, bias = refs["single_stream_layers.0.feed_forward.0"]
  assert weight.dtype == torch.bfloat16
  assert bias is not None


def test_capture_references_keeps_all_layers_without_filter() -> None:
  model = _TinyDiT().to(torch.bfloat16)

  refs = capture_references(model, QuantizationObserverConfig(enabled=True))

  assert len(refs) == 2 * 3
