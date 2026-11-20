import pytest
import torch

from cache_dit.cli.utils import get_args, maybe_postprocess_args
from cache_dit.quantization import (MXFP4Linear, QuantizationObserverConfig, QuantizeBackend,
                                    QuantizeConfig, quantize)
from cache_dit.quantization.mxfp4 import mxfp4_is_available, resolve_mxfp4_quantize_device
from cache_dit.quantization.mxfp4.tensor import (_quant_codes, dequantize_mxfp4,
                                                 quantize_mxfp4_weight)

MXFP4_TYPE = "mxfp4_weight_only"


def _require_mxfp4_dtypes() -> None:
  if not mxfp4_is_available():
    pytest.skip("MXFP4 requires torch.float4_e2m1fn_x2 + torch.float8_e8m0fnu.")


def _require_mxfp4_device() -> None:
  _require_mxfp4_dtypes()
  if not torch.cuda.is_available():
    pytest.skip("MXFP4 tests require CUDA.")


def test_mxfp4_config_and_cli() -> None:
  config = QuantizeConfig(quant_type=MXFP4_TYPE)
  assert config.strify() == f"{MXFP4_TYPE}_cache_dit"
  assert config.backend.value == "CACHE_DIT"
  assert config.is_mxfp4()

  # MXFP4 is weight-only here, so there is no W4A4 ``mxfp4`` entry (unlike nvfp4).
  with pytest.raises(ValueError, match="MXFP4 is weight-only"):
    QuantizeConfig(quant_type="mxfp4")

  with pytest.raises(ValueError, match="mxfp4_block_size"):
    QuantizeConfig(quant_type=MXFP4_TYPE, mxfp4_block_size=64)

  parser = get_args(parse=False)
  for flag in ("--mxfp4", "--mxfp4-weight-only", "--mxfp4-wo"):
    args = maybe_postprocess_args(parser.parse_args([flag]))
    assert args.quantize is True
    assert args.quantize_type == MXFP4_TYPE

  args = maybe_postprocess_args(parser.parse_args(["--quantize-type", MXFP4_TYPE]))
  assert args.quantize is True
  assert args.quantize_type == MXFP4_TYPE

  # ``mxfp4`` is no longer a quant type, only a CLI alias.
  with pytest.raises(SystemExit):
    parser.parse_args(["--quantize-type", "mxfp4"])

  with pytest.raises(ValueError, match="mutually exclusive"):
    maybe_postprocess_args(parser.parse_args(["--mxfp4", "--nvfp4"]))


def test_mxfp4_config_rejects_bad_quantize_device() -> None:
  with pytest.raises(ValueError, match="mxfp4_quantize_device"):
    QuantizeConfig(quant_type=MXFP4_TYPE, mxfp4_quantize_device="xpu")
  # "auto"/"cpu"/"cuda" are all accepted; "cuda" is validated at packing time.
  QuantizeConfig(quant_type=MXFP4_TYPE, mxfp4_quantize_device="cpu")
  QuantizeConfig(quant_type=MXFP4_TYPE, mxfp4_quantize_device="auto")


def test_mxfp4_component_config_keeps_mxfp4_fields() -> None:
  config = QuantizeConfig(quant_type=MXFP4_TYPE,
                          components_to_quantize={
                            "transformer": {
                              "mxfp4_quantize_device": "cpu",
                              "mxfp4_block_size": 32
                            }
                          })
  expanded = QuantizeConfig.expand_configs(config)
  assert len(expanded) == 1
  assert expanded[0].mxfp4_quantize_device == "cpu"
  assert expanded[0].mxfp4_block_size == 32
  assert expanded[0].components_to_quantize == ["transformer"]


def test_mxfp4_backend_support_is_quant_type_aware(monkeypatch) -> None:
  """The CACHE_DIT gate must ask for the quant type, not just the backend."""
  import cache_dit.kernels as kernels

  monkeypatch.setattr(kernels, "cuda_ext_is_available", lambda: False)
  backend = QuantizeBackend.CACHE_DIT
  assert QuantizeBackend.is_supported(backend, MXFP4_TYPE) is mxfp4_is_available()
  # Native FP8/SVDQ still need the CUDA extension and must fail early.
  assert QuantizeBackend.is_supported(backend, "float8_per_row") is False
  assert QuantizeBackend.is_supported(backend, "svdq_int4_r32_dq") is False


def test_mxfp4_tie_breaking_rounds_to_nearest_even() -> None:
  """Values exactly on an E2M1 midpoint round to the even neighbouring code."""
  _require_mxfp4_dtypes()
  values = torch.tensor([[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]])
  # codes 0/2/2/4/4/6/6 -> 0.0 / 1.0 / 1.0 / 2.0 / 2.0 / 4.0 / 4.0
  assert _quant_codes(values).tolist() == [[0, 2, 2, 4, 4, 6, 6]]


def test_mxfp4_tie_breaking_on_negative_midpoints() -> None:
  _require_mxfp4_dtypes()
  values = torch.tensor([[-0.75, -1.75, -3.5]])
  # sign bit 0x08 plus the same even-rounded magnitudes.
  assert _quant_codes(values).tolist() == [[10, 12, 14]]


@pytest.mark.parametrize("in_features", [32, 64, 96, 160, 256])
def test_mxfp4_accepts_spec_legal_k(in_features: int) -> None:
  """Only block_size alignment is required; 32/96/160 are legal MX shapes."""
  _require_mxfp4_dtypes()
  torch.manual_seed(0)
  weight = torch.randn(64, in_features, dtype=torch.bfloat16)
  packed, scales = quantize_mxfp4_weight(weight)
  assert packed.shape == (64, in_features // 2)
  assert scales.shape == (64, in_features // 32)


@pytest.mark.parametrize("in_features", [16, 48, 100])
def test_mxfp4_rejects_unaligned_k(in_features: int) -> None:
  _require_mxfp4_dtypes()
  weight = torch.randn(64, in_features, dtype=torch.bfloat16)
  with pytest.raises(ValueError, match="multiple of block_size"):
    quantize_mxfp4_weight(weight)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_mxfp4_quant_dequant_roundtrip(dtype: torch.dtype) -> None:
  _require_mxfp4_device()
  torch.manual_seed(0)
  device = torch.device("cuda")
  weight = torch.randn(128, 256, device=device, dtype=dtype)

  packed, scales = quantize_mxfp4_weight(weight)
  assert packed.shape == (128, 128)  # 2 fp4 codes per byte along K
  assert scales.shape == (128, 8)  # one E8M0 scale per group of 32
  assert packed.dtype == torch.float4_e2m1fn_x2
  assert scales.dtype == torch.float8_e8m0fnu

  dequant = dequantize_mxfp4(packed, scales, 128, 256, dtype=torch.float32)
  rel_err = ((dequant - weight.float()).abs().mean() / weight.float().abs().mean()).item()
  # MXFP4 group quantization noise for N(0,1) weights is ~10-12% MAE-relative;
  # the bound mainly guards against packing/scale-order regressions.
  assert rel_err < 0.15, f"unexpected MXFP4 roundtrip error: {rel_err}"
  # Storage: 4-bit payload + 1/32-byte E8M0 scale -> ~0.28 bytes per element,
  # i.e. ~3.6x smaller than bf16 and ~7x smaller than fp32 weights.
  stored_bytes = packed.numel() + scales.numel()
  assert stored_bytes < weight.numel() * weight.element_size() / 2.5


def test_mxfp4_quantize_device_resolution() -> None:
  weight_cpu = torch.randn(64, 64, dtype=torch.bfloat16)

  assert resolve_mxfp4_quantize_device(weight_cpu, "cpu") == torch.device("cpu")

  auto = resolve_mxfp4_quantize_device(weight_cpu, "auto")
  assert auto == resolve_mxfp4_quantize_device(weight_cpu, None)
  expected = (torch.device("cuda", torch.cuda.current_device())
              if torch.cuda.is_available() else torch.device("cpu"))
  assert auto == expected

  with pytest.raises(ValueError, match="quantize_device"):
    resolve_mxfp4_quantize_device(weight_cpu, "xpu")

  if torch.cuda.is_available():
    # A weight already on the accelerator stays there under "auto".
    weight_cuda = weight_cpu.cuda()
    assert resolve_mxfp4_quantize_device(weight_cuda, "auto") == weight_cuda.device
  else:
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
      resolve_mxfp4_quantize_device(weight_cpu, "cuda")


@pytest.mark.parametrize("requested", ["auto", "cuda"])
def test_mxfp4_accelerator_pack_is_bit_identical(requested: str) -> None:
  """Packing on the accelerator must not change a single payload or scale bit."""
  _require_mxfp4_dtypes()
  if not torch.cuda.is_available():
    pytest.skip("cross-device packing parity needs an accelerator")
  torch.manual_seed(0)
  weight = torch.randn(128, 512, dtype=torch.bfloat16)

  device = resolve_mxfp4_quantize_device(weight, requested)
  assert device.type == "cuda"
  packed, scales = quantize_mxfp4_weight(weight, device=device)
  ref_packed, ref_scales = quantize_mxfp4_weight(weight, device=torch.device("cpu"))

  # Buffers always come back to the weight's device, not the packing device.
  assert packed.device == weight.device
  assert scales.device == weight.device
  assert torch.equal(packed.view(torch.uint8), ref_packed.view(torch.uint8))
  assert torch.equal(scales.view(torch.uint8), ref_scales.view(torch.uint8))


def test_mxfp4_quantize_device_reaches_packing(monkeypatch) -> None:
  """The config field must reach the packing math, not just be stored."""
  _require_mxfp4_dtypes()
  from cache_dit.quantization.mxfp4 import tensor as tensor_mod

  seen = []
  orig = tensor_mod.quantize_mxfp4_weight

  def spy(weight, block_size=tensor_mod.MXFP4_BLOCK_SIZE, device=None):
    seen.append(device)
    return orig(weight, block_size=block_size, device=device)

  monkeypatch.setattr(tensor_mod, "quantize_mxfp4_weight", spy)

  quantize(_ToyBlock(), QuantizeConfig(quant_type=MXFP4_TYPE, mxfp4_quantize_device="cpu"))
  assert seen == [torch.device("cpu")]

  seen.clear()
  quantize(_ToyBlock(), QuantizeConfig(quant_type=MXFP4_TYPE))
  expected = (torch.device("cuda", torch.cuda.current_device())
              if torch.cuda.is_available() else torch.device("cpu"))
  assert seen == [expected]


class _ToyBlock(torch.nn.Module):

  def __init__(self) -> None:
    super().__init__()
    self.proj = torch.nn.Linear(256, 32, dtype=torch.bfloat16)


@pytest.mark.parametrize("requested", ["auto", "cpu", "cuda"])
def test_mxfp4_quantizes_cpu_module_without_moving_it(requested: str) -> None:
  """Quantization runs before the pipeline is moved to the GPU: the module must
  stay on CPU (only the packing math is offloaded) and still run afterwards."""
  _require_mxfp4_dtypes()
  if requested == "cuda" and not torch.cuda.is_available():
    pytest.skip("mxfp4_quantize_device='cuda' needs an accelerator")
  torch.manual_seed(0)

  quantized = quantize(_ToyBlock(),
                       QuantizeConfig(quant_type=MXFP4_TYPE, mxfp4_quantize_device=requested))

  assert isinstance(quantized.proj, MXFP4Linear)
  assert quantized.proj.weight_packed.device.type == "cpu"
  assert quantized.proj.weight_scales.device.type == "cpu"

  x = torch.randn(2, 256, dtype=torch.bfloat16)
  with torch.no_grad():
    out = quantized.proj(x)
  assert out.shape == (2, 32)
  assert torch.isfinite(out).all()


def test_mxfp4_module_survives_dtype_cast() -> None:
  """``.half()``/``.to(dtype)`` must not try to cast the fp4 payload."""
  _require_mxfp4_dtypes()
  torch.manual_seed(0)
  quantized = quantize(_ToyBlock(),
                       QuantizeConfig(quant_type=MXFP4_TYPE, mxfp4_quantize_device="cpu"))
  half = quantized.proj.half()

  assert half.weight_packed.dtype == torch.float4_e2m1fn_x2
  assert half.weight_scales.dtype == torch.float8_e8m0fnu
  assert half.bias.dtype == torch.float16

  with torch.no_grad():
    out = half(torch.randn(2, 256, dtype=torch.float16))
  assert out.dtype == torch.float16
  assert torch.isfinite(out).all()


def test_mxfp4_module_survives_device_and_dtype_moves() -> None:
  _require_mxfp4_dtypes()
  if not torch.cuda.is_available():
    pytest.skip("device move needs an accelerator")
  quantized = quantize(_ToyBlock(),
                       QuantizeConfig(quant_type=MXFP4_TYPE, mxfp4_quantize_device="cpu"))
  moved = quantized.proj.to("cuda")

  assert moved.weight_packed.device.type == "cuda"
  assert moved.weight_scales.device.type == "cuda"
  with torch.no_grad():
    out = moved(torch.randn(2, 256, device="cuda", dtype=torch.bfloat16))
  assert out.dtype == torch.bfloat16


def test_mxfp4_honors_filter_fn() -> None:
  """``filter_fn`` must actually be consulted for every eligible layer."""
  _require_mxfp4_dtypes()

  class TwoLinears(torch.nn.Module):

    def __init__(self) -> None:
      super().__init__()
      self.keep = torch.nn.Linear(256, 32, dtype=torch.bfloat16)
      self.drop = torch.nn.Linear(256, 32, dtype=torch.bfloat16)

  seen = []

  def filter_fn(module, name):
    seen.append(name)
    return "drop" not in name

  quantized = quantize(
    TwoLinears(),
    QuantizeConfig(quant_type=MXFP4_TYPE, filter_fn=filter_fn, mxfp4_quantize_device="cpu"))
  assert isinstance(quantized.keep, MXFP4Linear)
  assert not isinstance(quantized.drop, MXFP4Linear)
  assert "drop" in seen
  assert quantized._mxfp4_skipped_layers == {"drop": "filter_fn"}


def test_mxfp4_exclude_layers_wins_over_filter_fn() -> None:
  """``exclude_layers`` is evaluated first, same order as the native FP8 path."""
  _require_mxfp4_dtypes()

  class TwoLinears(torch.nn.Module):

    def __init__(self) -> None:
      super().__init__()
      self.kept = torch.nn.Linear(256, 32, dtype=torch.bfloat16)
      self.excluded = torch.nn.Linear(256, 32, dtype=torch.bfloat16)

  seen = []

  def filter_fn(module, name):
    seen.append(name)
    return True

  quantized = quantize(
    TwoLinears(),
    QuantizeConfig(quant_type=MXFP4_TYPE,
                   exclude_layers=["excluded"],
                   filter_fn=filter_fn,
                   mxfp4_quantize_device="cpu"))
  assert quantized._mxfp4_skipped_layers == {"excluded": "exclude_layers"}
  # filter_fn is never consulted for an excluded layer.
  assert seen == ["kept"]


def test_mxfp4_regional_quantize_only_touches_repeated_blocks() -> None:
  """``regional_quantize`` (the default) must leave non-block linears alone."""
  _require_mxfp4_dtypes()

  class Block(torch.nn.Module):

    def __init__(self) -> None:
      super().__init__()
      self.proj = torch.nn.Linear(256, 32, dtype=torch.bfloat16)

  class Model(torch.nn.Module):
    _repeated_blocks = ["Block"]

    def __init__(self) -> None:
      super().__init__()
      self.blocks = torch.nn.ModuleList([Block() for _ in range(2)])
      self.head = torch.nn.Linear(256, 32, dtype=torch.bfloat16)

  quantized = quantize(Model(), QuantizeConfig(quant_type=MXFP4_TYPE, mxfp4_quantize_device="cpu"))
  assert isinstance(quantized.blocks[0].proj, MXFP4Linear)
  assert isinstance(quantized.blocks[1].proj, MXFP4Linear)
  assert not isinstance(quantized.head, MXFP4Linear)


def test_mxfp4_backend_rejects_other_quant_types() -> None:
  """Calling the MXFP4 entry with another quant type must not mislabel the module."""
  _require_mxfp4_dtypes()
  from cache_dit.quantization.mxfp4 import quantize_mxfp4

  config = QuantizeConfig(quant_type="nvfp4", backend="cache_dit")
  with pytest.raises(ValueError, match="does not support quant_type"):
    quantize_mxfp4(_ToyBlock(), config)


def test_mxfp4_all_skipped_raises() -> None:
  """A run that packs nothing must fail instead of returning a bf16 module."""
  _require_mxfp4_dtypes()

  class Unaligned(torch.nn.Module):

    def __init__(self) -> None:
      super().__init__()
      self.proj = torch.nn.Linear(48, 32, dtype=torch.bfloat16)

  with pytest.raises(ValueError, match="did not quantize any linear layer"):
    quantize(Unaligned(), QuantizeConfig(quant_type=MXFP4_TYPE, mxfp4_quantize_device="cpu"))


def test_mxfp4_attaches_observer(tmp_path) -> None:
  """``observer=...`` must install reference hooks, not be silently accepted."""
  _require_mxfp4_dtypes()
  torch.manual_seed(0)
  config = QuantizeConfig(
    quant_type=MXFP4_TYPE,
    mxfp4_quantize_device="cpu",
    observer=QuantizationObserverConfig(enabled=True,
                                        json_path=str(tmp_path / "observer.json"),
                                        max_steps=1),
  )
  quantized = quantize(_ToyBlock(), config)
  assert len(quantized.proj._forward_hooks) > 0

  with torch.no_grad():
    # The first pass registers the step and dumps nothing yet (no stats), the
    # second one has data to write out.
    quantized.proj(torch.randn(2, 256, dtype=torch.bfloat16))
    quantized.proj(torch.randn(2, 256, dtype=torch.bfloat16))
  # The observer records a weight-only error for the packed layer.
  rows = [row for row in _observer_rows() if row["name"] == "proj"]
  assert rows and rows[0]["w_rel_err"] is not None and rows[0]["w_rel_err"] > 0
  assert (tmp_path / "observer.json").exists()


def _observer_rows():
  from cache_dit.quantization.observers import _INSTANCES

  rows = []
  for observer in _INSTANCES:
    rows.extend(observer.rows())
  return rows


def test_mxfp4_observer_keep_refs_on_gpu_uses_the_layer_device(tmp_path) -> None:
  """``keep_refs_on_gpu`` must follow the layer's device, not a hardcoded cuda.

  A CPU-resident module would otherwise pair a cuda reference weight with CPU activations inside the
  hook.
  """
  _require_mxfp4_dtypes()
  torch.manual_seed(0)
  config = QuantizeConfig(
    quant_type=MXFP4_TYPE,
    mxfp4_quantize_device="cpu",
    observer=QuantizationObserverConfig(enabled=True,
                                        json_path=str(tmp_path / "observer.json"),
                                        max_steps=1,
                                        keep_refs_on_gpu=True),
  )
  quantized = quantize(_ToyBlock(), config)
  with torch.no_grad():
    out = quantized.proj(torch.randn(2, 256, dtype=torch.bfloat16))
  assert out.shape == (2, 32)
  assert torch.isfinite(out).all()


def test_mxfp4_quantize_linear_via_dispatch() -> None:
  _require_mxfp4_device()
  torch.manual_seed(0)

  class ToyBlock(torch.nn.Module):

    def __init__(self) -> None:
      super().__init__()
      self.proj = torch.nn.Linear(256, 256, dtype=torch.bfloat16)
      self.excluded = torch.nn.Linear(256, 64, dtype=torch.bfloat16)
      self.unaligned = torch.nn.Linear(48, 32, dtype=torch.bfloat16)  # 48 % 32 != 0

  class ToyModel(torch.nn.Module):

    def __init__(self) -> None:
      super().__init__()
      self.blocks = torch.nn.ModuleList([ToyBlock() for _ in range(2)])

    def forward(self, x):
      for block in self.blocks:
        x = block.proj(x)
      return x

  model = ToyModel().cuda()
  x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
  with torch.no_grad():
    ref = model(x)

  quantized = quantize(model, QuantizeConfig(quant_type=MXFP4_TYPE, exclude_layers=["excluded"]))
  assert isinstance(quantized.blocks[0].proj, MXFP4Linear)
  assert not isinstance(quantized.blocks[0].excluded, MXFP4Linear)
  assert not isinstance(quantized.blocks[0].unaligned, MXFP4Linear)
  assert quantized._mxfp4_skipped_layers == {
    "blocks.0.excluded": "exclude_layers",
    "blocks.0.unaligned": "in_features=48 is not a multiple of block_size=32 / even",
    "blocks.1.excluded": "exclude_layers",
    "blocks.1.unaligned": "in_features=48 is not a multiple of block_size=32 / even",
  }

  with torch.no_grad():
    out = quantized(x)
  rel_err = ((out.float() - ref.float()).abs().mean() / ref.float().abs().mean()).item()
  # Two chained MXFP4 linears compound per-layer ~12% noise; guard against
  # packing/scale-order bugs rather than quantization noise itself.
  assert rel_err < 0.30, f"unexpected MXFP4 end-to-end error: {rel_err}"
