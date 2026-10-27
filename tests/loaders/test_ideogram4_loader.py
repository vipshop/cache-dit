"""Regression tests for the Ideogram 4 loader.

`ideogram-ai/ideogram-4-fp8` stores both denoisers and the text encoder as FP8 weights with a fused
`attention.qkv`, an `attention.o` written where the module says `to_out.0`, and one scale per output
row. No `from_pretrained` route maps those onto the module parameters: the unmatched parameters stay
on the meta device and `pipe.to(device)` then fails. `Ideogram4Loader` patches
`Ideogram4Pipeline.from_pretrained` and hands the converted components over.

The failure modes pinned down here:

1. The conversion is driven by a marker in `text_encoder/config.json`; a checkpoint without it
   has to stay on the plain path, with the call forwarded exactly as it was made.
2. The patch belongs to the class, so it is idempotent by class, keeps a subclass a subclass, and
   can be removed without freezing a bound method into the class.
3. `dtype` / `torch_dtype`, scalar or per-component mapping, have to reach the rebuilt
   components, and a component the caller passed itself — an explicit `None` included — must not
   be replaced.
4. A `device_map` cannot place components the loader built out of band, so it is refused before a
   single weight is read.
5. The strict rebuild is the proof that no parameter is left on the meta device.

Usage:
  python -m pytest tests/loaders/test_ideogram4_loader.py -q
"""
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

try:
  from diffusers import DiffusionPipeline, Ideogram4Pipeline, Ideogram4Transformer2DModel
except ImportError:
  pytest.skip("diffusers without Ideogram4 support", allow_module_level=True)

from cache_dit.loaders import BaseLoader, Ideogram4Loader
from cache_dit.loaders import base as loaders_base
from cache_dit.loaders import ideogram4

# kwargs of the latest stand-in pipeline call
_SEEN = {}


class _RecordingIdeogram4Pipeline(Ideogram4Pipeline):
  """`Ideogram4Pipeline` whose own `from_pretrained` records the forwarded call.

  The recorder is defined on a subclass on purpose: a loader applied here keeps it as `original`,
  so no test has to monkeypatch `Ideogram4Pipeline.from_pretrained` globally, and the recorder sees
  the exact class the wrapper was called on.
  """

  @classmethod
  def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
    _SEEN.clear()
    _SEEN.update(cls=cls, path=pretrained_model_name_or_path, args=args, kwargs=kwargs)
    return SimpleNamespace(requested=cls)


class _DeeperRecordingIdeogram4Pipeline(_RecordingIdeogram4Pipeline):
  """Inherits the recorder and the patch, and must still see itself as the requested class."""


def _installed_loader(cls: type):
  """The loader that patched `cls`, read back through the marker it left in `cls.__dict__`."""
  return cls.__dict__.get(loaders_base.LOADER_PATCH_ATTR)


def _drop_installed_loader(cls: type) -> None:
  """Remove whatever patch a stray test may have left on `cls`."""
  loader = _installed_loader(cls)
  if loader is not None:
    loader.remove()


@pytest.fixture(autouse=True)
def _unpatched_pipeline_class():
  """A patch left on `Ideogram4Pipeline` would shadow the recorder these tests install."""
  _drop_installed_loader(Ideogram4Pipeline)
  _SEEN.clear()
  yield
  _SEEN.clear()
  _drop_installed_loader(Ideogram4Pipeline)


@pytest.fixture
def loader():
  """A loader applied to the recording subclass, removed again afterwards."""
  instance = Ideogram4Loader()
  instance.apply(_RecordingIdeogram4Pipeline)
  try:
    yield instance
  finally:
    instance.remove()
  assert _installed_loader(_RecordingIdeogram4Pipeline) is None


# FP8 checkpoint conversion. The fixtures below write the on-disk layout of the official
# `ideogram-ai/ideogram-4-fp8` release — a fused `attention.qkv` with one scale per output row, an
# `attention.o` written where the module says `to_out.0`, and bfloat16 norms without scales — at a
# size that fits in a test.

_FP8_DTYPE = torch.float8_e4m3fn
# e4m3 tops out at 448, so dividing the row maximum by it keeps every row in range.
_FP8_MAX = 448.0

_TINY_TRANSFORMER = {
  "in_channels": 4,
  "num_layers": 1,
  "attention_head_dim": 4,
  "num_attention_heads": 1,
  "intermediate_size": 8,
  "adaln_dim": 4,
  "llm_features_dim": 8,
}

_TINY_TEXT_ENCODER = {
  "vocab_size": 32,
  "hidden_size": 8,
  "intermediate_size": 16,
  "num_hidden_layers": 1,
  "num_attention_heads": 2,
  "num_key_value_heads": 2,
}


def _quantize(weight: torch.Tensor):
  """The weight as the checkpoint stores it: FP8 plus one scale per output row."""
  scale = weight.float().abs().amax(dim=-1).clamp_min(1e-6) / _FP8_MAX
  return (weight.float() / scale.unsqueeze(-1)).to(_FP8_DTYPE), scale


def _dequantize(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
  """The row-wise formula, written out here instead of reusing the converter under test."""
  return (weight.float() * scale.float().unsqueeze(-1)).to(torch.bfloat16)


def _write_transformer_checkpoint(root: Path, seed: int = 0):
  """One tiny denoiser in the FP8 layout, with the two attention renames applied."""
  torch.manual_seed(seed)
  module = Ideogram4Transformer2DModel(**_TINY_TRANSFORMER).to(torch.bfloat16)
  state = module.state_dict()

  projections = ("to_q", "to_k", "to_v")
  fused = torch.cat([state[f"layers.0.attention.{name}.weight"] for name in projections], dim=0)
  stored = {}
  stored["layers.0.attention.qkv.weight"], stored["layers.0.attention.qkv.weight_scale"] = (
    _quantize(fused))
  stored["layers.0.attention.o.weight"], stored["layers.0.attention.o.weight_scale"] = (_quantize(
    state["layers.0.attention.to_out.0.weight"]))
  for name, tensor in state.items():
    if any(f".attention.{projection}.weight" in name for projection in projections):
      continue
    if ".attention.to_out.0.weight" in name:
      continue
    stored[name] = tensor

  root.mkdir(parents=True, exist_ok=True)
  (root / "config.json").write_text(
    json.dumps({
      "_class_name": "Ideogram4Transformer2DModel",
      **_TINY_TRANSFORMER,
    }))
  save_file(stored, root / "diffusion_pytorch_model.safetensors")


def _write_text_encoder_checkpoint(root: Path):
  """One tiny `transformers` encoder, with `q_proj` FP8 and `k_proj` left in bfloat16."""
  from transformers import LlamaConfig, LlamaModel

  torch.manual_seed(0)
  config = LlamaConfig(**_TINY_TEXT_ENCODER)
  state = LlamaModel(config).to(torch.bfloat16).state_dict()
  stored = dict(state)
  stored["layers.0.self_attn.q_proj.weight"], stored["layers.0.self_attn.q_proj.weight_scale"] = (
    _quantize(state["layers.0.self_attn.q_proj.weight"]))

  root.mkdir(parents=True, exist_ok=True)
  (root / "config.json").write_text(
    json.dumps({
      **config.to_dict(),
      "ideogram_fp8_weight_only": True,
    }))
  save_file(stored, root / "model.safetensors")


def _fp8_checkpoint(root: Path) -> Path:
  """A checkpoint root holding both denoisers and the text encoder, marker included."""
  _write_transformer_checkpoint(root / "transformer", seed=0)
  _write_transformer_checkpoint(root / "unconditional_transformer", seed=1)
  _write_text_encoder_checkpoint(root / "text_encoder")
  return root


def _plain_checkpoint(root: Path) -> Path:
  """A checkpoint root without the FP8 marker, i.e. one that loads through the plain path."""
  root.mkdir(parents=True, exist_ok=True)
  (root / "text_encoder").mkdir(parents=True)
  (root / "text_encoder" / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))
  return root


# The patch: what `apply` installs, and what `remove` gives back.


def test_apply_patches_the_class_in_place_and_returns_it():
  instance = Ideogram4Loader()
  try:
    assert instance.apply(Ideogram4Pipeline) is Ideogram4Pipeline
    assert isinstance(Ideogram4Pipeline.__dict__["from_pretrained"], classmethod)
    assert instance.patched_class is Ideogram4Pipeline
    assert _installed_loader(Ideogram4Pipeline) is instance
  finally:
    instance.remove()
  assert "from_pretrained" not in Ideogram4Pipeline.__dict__
  assert _installed_loader(Ideogram4Pipeline) is None


def test_apply_accepts_an_instance_and_patches_its_class():
  """A caller holding a pipeline patches the class that built it, not the instance."""
  instance = Ideogram4Loader()
  uninitialized = object.__new__(Ideogram4Pipeline)
  try:
    assert instance.apply(uninitialized) is Ideogram4Pipeline
    assert instance.patched_class is Ideogram4Pipeline
  finally:
    instance.remove()


def test_repeated_apply_keeps_one_wrapper():
  """Idempotent by class, not by loader: a second instance must not stack a second conversion."""
  first = Ideogram4Loader()
  try:
    first.apply(Ideogram4Pipeline)
    installed = Ideogram4Pipeline.__dict__["from_pretrained"]
    assert Ideogram4Loader().apply(Ideogram4Pipeline) is Ideogram4Pipeline
    assert first.apply(Ideogram4Pipeline) is Ideogram4Pipeline
    assert Ideogram4Pipeline.__dict__["from_pretrained"] is installed
    assert installed.__func__ is first.wrapper
  finally:
    first.remove()


def test_a_subclass_call_stays_a_subclass_call(loader, tmp_path):
  """The wrapper re-binds the inherited classmethod to the class the call arrived on."""
  plain = _plain_checkpoint(tmp_path / "plain")

  result = _DeeperRecordingIdeogram4Pipeline.from_pretrained(str(plain))
  assert _SEEN["cls"] is _DeeperRecordingIdeogram4Pipeline
  assert result.requested is _DeeperRecordingIdeogram4Pipeline

  # The patched class itself is not special: it keeps receiving itself.
  result = _RecordingIdeogram4Pipeline.from_pretrained(str(plain))
  assert _SEEN["cls"] is _RecordingIdeogram4Pipeline
  assert result.requested is _RecordingIdeogram4Pipeline


def test_an_inherited_patch_is_not_wrapped_again(loader):
  """`Sub.from_pretrained` already runs the base class's wrapper; a second loader must not apply."""
  installed = _RecordingIdeogram4Pipeline.__dict__["from_pretrained"]

  again = Ideogram4Loader().apply(_DeeperRecordingIdeogram4Pipeline)

  assert again is _DeeperRecordingIdeogram4Pipeline
  assert "from_pretrained" not in _DeeperRecordingIdeogram4Pipeline.__dict__
  assert _RecordingIdeogram4Pipeline.__dict__["from_pretrained"] is installed


def test_remove_gives_the_inherited_classmethod_back():
  """Deleting the attribute is what restores inheritance; a stored bound method would not."""
  inherited = DiffusionPipeline.__dict__["from_pretrained"]
  instance = Ideogram4Loader()
  instance.apply(Ideogram4Pipeline)
  assert Ideogram4Pipeline.from_pretrained.__func__ is instance.wrapper

  instance.remove()

  assert "from_pretrained" not in Ideogram4Pipeline.__dict__
  assert Ideogram4Pipeline.from_pretrained.__func__ is inherited.__func__
  assert _installed_loader(Ideogram4Pipeline) is None
  # A second removal has nothing left to undo.
  instance.remove()
  assert "from_pretrained" not in Ideogram4Pipeline.__dict__


def test_remove_restores_a_class_that_had_its_own_from_pretrained():
  recorder = _RecordingIdeogram4Pipeline.__dict__["from_pretrained"]
  instance = Ideogram4Loader()
  instance.apply(_RecordingIdeogram4Pipeline)
  assert _RecordingIdeogram4Pipeline.__dict__["from_pretrained"].__func__ is instance.wrapper

  instance.remove()

  assert _RecordingIdeogram4Pipeline.__dict__["from_pretrained"].__func__ is recorder.__func__
  _RecordingIdeogram4Pipeline.from_pretrained("/no/such/path")
  assert _SEEN["path"] == "/no/such/path"


def test_remove_leaves_a_wrapper_it_does_not_own_alone():
  instance = Ideogram4Loader()
  instance.apply(Ideogram4Pipeline)

  def _foreign(cls, pretrained_model_name_or_path, **kwargs):
    return "foreign"

  try:
    setattr(Ideogram4Pipeline, "from_pretrained", classmethod(_foreign))
    instance.remove()
    assert Ideogram4Pipeline.from_pretrained.__func__ is _foreign
    # The marker this loader left is still dropped: it no longer describes the class.
    assert _installed_loader(Ideogram4Pipeline) is None
  finally:
    delattr(Ideogram4Pipeline, "from_pretrained")


def test_the_patched_signature_is_the_original_one():
  """`functools.wraps` keeps `from_pretrained` inspectable through the patch."""
  before = inspect.signature(Ideogram4Pipeline.from_pretrained)
  instance = Ideogram4Loader()
  try:
    instance.apply(Ideogram4Pipeline)
    patched = Ideogram4Pipeline.from_pretrained
    assert patched.__name__ == "from_pretrained"
    assert inspect.signature(patched) == before
  finally:
    instance.remove()
  assert inspect.signature(Ideogram4Pipeline.from_pretrained) == before


def test_the_seams_are_read_only(loader):
  assert loader.original is not None
  assert loader.wrapper is not None
  assert loader.patched_class is not None
  for name in ("original", "wrapper", "patched_class"):
    with pytest.raises(AttributeError):
      setattr(loader, name, None)


def test_the_base_loader_cannot_be_used_without_a_hook():
  with pytest.raises(TypeError):
    BaseLoader()


def test_the_loaders_stay_a_leaf_of_the_package():
  """`cache_dit.loaders` is exported by `cache_dit/__init__.py`, so it may not pull in the rest."""
  import ast

  import cache_dit.loaders.base as base_module
  import cache_dit.loaders.ideogram4 as ideogram4_module

  for module in (base_module, ideogram4_module):
    source = Path(module.__file__).read_text()
    assert "_utils" not in source, module.__name__
    imported = set()
    for node in ast.parse(source).body:
      if isinstance(node, ast.Import):
        imported |= {alias.name.split(".")[0] for alias in node.names}
      elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
        imported.add(node.module.split(".")[0])
    assert imported.isdisjoint({"diffusers", "transformers"}), module.__name__


# The conversion itself, helper by helper.


def test_fp8_marker_comes_from_the_text_encoder_config(tmp_path):
  assert ideogram4._ideogram4_is_fp8(str(_fp8_checkpoint(tmp_path / "fp8"))) is True

  # A bfloat16 re-upload carries neither the file nor the flag.
  plain = tmp_path / "plain"
  plain.mkdir()
  assert ideogram4._ideogram4_is_fp8(str(plain)) is False
  (plain / "text_encoder").mkdir()
  (plain / "text_encoder" / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))
  assert ideogram4._ideogram4_is_fp8(str(plain)) is False
  (plain / "text_encoder" / "config.json").write_text(
    json.dumps({"ideogram_fp8_weight_only": False}))
  assert ideogram4._ideogram4_is_fp8(str(plain)) is False

  # A mistyped local path stays local: it must not turn into a hub round trip.
  assert ideogram4._ideogram4_is_fp8(str(tmp_path / "missing")) is False


def test_a_missing_marker_is_not_an_error_but_a_broken_hub_is(monkeypatch):
  """Only `EntryNotFoundError` means "not FP8"; swallowing the rest would hide a dead network."""
  import huggingface_hub
  from huggingface_hub.errors import EntryNotFoundError

  def _missing(repo_id, filename, **kwargs):
    raise EntryNotFoundError("no such file")

  monkeypatch.setattr(huggingface_hub, "hf_hub_download", _missing)
  assert ideogram4._ideogram4_is_fp8("ideogram-ai/ideogram-4-fp8") is False

  def _offline(repo_id, filename, **kwargs):
    raise ConnectionError("the hub is unreachable")

  monkeypatch.setattr(huggingface_hub, "hf_hub_download", _offline)
  with pytest.raises(ConnectionError, match="unreachable"):
    ideogram4._ideogram4_is_fp8("ideogram-ai/ideogram-4-fp8")


def test_fused_qkv_weight_is_split_on_dim_zero():
  """`to_q`, `to_k` and `to_v` are the three thirds of the fused weight, each with its own scale."""
  weight = torch.arange(12, dtype=torch.float32).reshape(6, 2).to(_FP8_DTYPE)
  scale = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  converted = ideogram4._ideogram4_convert_tensor("layers.3.attention.qkv.weight", weight, scale,
                                                  torch.bfloat16)

  assert set(converted) == {
    "layers.3.attention.to_q.weight",
    "layers.3.attention.to_k.weight",
    "layers.3.attention.to_v.weight",
  }
  for index, projection in enumerate(("to_q", "to_k", "to_v")):
    expected = _dequantize(weight[2 * index:2 * index + 2], scale[2 * index:2 * index + 2])
    actual = converted[f"layers.3.attention.{projection}.weight"]
    assert actual.dtype is torch.bfloat16
    assert torch.equal(actual, expected)


def test_attention_output_projection_is_renamed_where_the_module_expects_it():
  weight = torch.ones(3, 2).to(_FP8_DTYPE)
  scale = torch.tensor([0.5, 1.0, 2.0])
  converted = ideogram4._ideogram4_convert_tensor("layers.0.attention.o.weight", weight, scale,
                                                  torch.bfloat16)

  assert set(converted) == {"layers.0.attention.to_out.0.weight"}
  assert torch.equal(converted["layers.0.attention.to_out.0.weight"], _dequantize(weight, scale))


def test_a_weight_without_a_scale_is_only_cast():
  """Norms and biases carry no scale and need no renaming."""
  weight = torch.randn(4, 4, dtype=torch.bfloat16)
  converted = ideogram4._ideogram4_convert_tensor("layers.0.attention_norm1.weight", weight, None,
                                                  torch.bfloat16)
  assert set(converted) == {"layers.0.attention_norm1.weight"}
  assert torch.equal(converted["layers.0.attention_norm1.weight"], weight)


def test_scale_tensors_never_reach_the_module(tmp_path):
  root = _fp8_checkpoint(tmp_path)
  converted = ideogram4._ideogram4_component_state_dict(str(root), "transformer",
                                                        "diffusion_pytorch_model", torch.bfloat16)

  assert converted, "the fixture checkpoint has to convert into something"
  assert not any(name.endswith("_scale") for name in converted)
  assert not any("qkv" in name for name in converted)
  assert not any(name.endswith(".attention.o.weight") for name in converted)
  assert "layers.0.attention.to_q.weight" in converted
  assert "layers.0.attention.to_out.0.weight" in converted


@pytest.mark.parametrize("subfolder", ["transformer", "unconditional_transformer"])
def test_fp8_denoiser_is_rebuilt_into_a_loadable_bf16_module(tmp_path, subfolder):
  """A strict load also proves the mapping covers every parameter, in both directions."""
  root = _fp8_checkpoint(tmp_path)
  loaded = ideogram4._ideogram4_bf16_transformer(str(root), subfolder, torch.bfloat16)

  # Nothing may be left on the meta device: that is what made `pipe.to(device)` fail.
  assert not any(tensor.is_meta for tensor in loaded.parameters())
  assert not any(tensor.is_meta for tensor in loaded.buffers())
  assert {parameter.dtype for parameter in loaded.parameters()} == {torch.bfloat16}
  assert loaded.training is False

  stored = load_file(root / subfolder / "diffusion_pytorch_model.safetensors")
  attention = loaded.layers[0].attention
  hidden = stored["layers.0.attention.qkv.weight"].shape[0] // 3
  for index, projection in enumerate(("to_q", "to_k", "to_v")):
    rows = slice(index * hidden, (index + 1) * hidden)
    expected = _dequantize(stored["layers.0.attention.qkv.weight"][rows],
                           stored["layers.0.attention.qkv.weight_scale"][rows])
    assert torch.equal(getattr(attention, projection).weight, expected)
  assert torch.equal(
    attention.to_out[0].weight,
    _dequantize(stored["layers.0.attention.o.weight"], stored["layers.0.attention.o.weight_scale"]))
  # An unquantized tensor is the checkpoint's own value, not a re-quantized one.
  assert torch.equal(loaded.layers[0].attention_norm1.weight,
                     stored["layers.0.attention_norm1.weight"])


def test_a_scale_stored_in_another_shard_is_still_paired(tmp_path):
  """Nothing guarantees a per-row scale lands in the shard of the weight it scales."""
  root = tmp_path / "transformer"
  root.mkdir()
  torch.manual_seed(0)
  weight = torch.randn(4, 2, dtype=torch.float32)
  quantized, scale = _quantize(weight)
  save_file({"layers.0.attention.o.weight": quantized}, root / "shard-1.safetensors")
  save_file({"layers.0.attention.o.weight_scale": scale}, root / "shard-2.safetensors")
  (root / "diffusion_pytorch_model.safetensors.index.json").write_text(
    json.dumps({
      "weight_map": {
        "layers.0.attention.o.weight": "shard-1.safetensors",
        "layers.0.attention.o.weight_scale": "shard-2.safetensors",
      }
    }))

  converted = ideogram4._ideogram4_component_state_dict(str(tmp_path), "transformer",
                                                        "diffusion_pytorch_model", torch.bfloat16)
  assert torch.equal(converted["layers.0.attention.to_out.0.weight"], _dequantize(quantized, scale))


def test_fp8_text_encoder_is_dequantized_the_same_way(tmp_path):
  root = _fp8_checkpoint(tmp_path)
  encoder = ideogram4._ideogram4_bf16_text_encoder(str(root), torch.bfloat16)

  assert not any(tensor.is_meta for tensor in encoder.parameters())
  assert {parameter.dtype for parameter in encoder.parameters()} == {torch.bfloat16}

  stored = load_file(root / "text_encoder" / "model.safetensors")
  assert torch.equal(
    encoder.layers[0].self_attn.q_proj.weight,
    _dequantize(stored["layers.0.self_attn.q_proj.weight"],
                stored["layers.0.self_attn.q_proj.weight_scale"]))
  # `k_proj` is stored unquantized, so it only has to change dtype.
  assert torch.equal(encoder.layers[0].self_attn.k_proj.weight,
                     stored["layers.0.self_attn.k_proj.weight"])


# The patch in action: what a `from_pretrained` call is forwarded with.


def test_a_non_fp8_checkpoint_passes_through_untouched(loader, tmp_path):
  plain = _plain_checkpoint(tmp_path / "plain")

  _RecordingIdeogram4Pipeline.from_pretrained(str(plain), dtype=torch.bfloat16)

  assert _SEEN["path"] == str(plain)
  # Literally the same call, with nothing added and nothing renamed.
  assert _SEEN["kwargs"] == {"dtype": torch.bfloat16}


def test_device_map_is_forwarded_for_a_non_fp8_checkpoint(loader, tmp_path):
  plain = _plain_checkpoint(tmp_path / "plain")

  _RecordingIdeogram4Pipeline.from_pretrained(str(plain), device_map="balanced")

  assert _SEEN["kwargs"] == {"device_map": "balanced"}


def test_an_fp8_checkpoint_injects_the_three_components(loader, tmp_path):
  root = _fp8_checkpoint(tmp_path)

  result = _RecordingIdeogram4Pipeline.from_pretrained(str(root))

  assert result.requested is _RecordingIdeogram4Pipeline
  forwarded = _SEEN["kwargs"]
  assert set(forwarded) == {
    "dtype",
    "transformer",
    "unconditional_transformer",
    "text_encoder",
  }
  assert forwarded["dtype"] is torch.bfloat16
  assert isinstance(forwarded["transformer"], Ideogram4Transformer2DModel)
  assert isinstance(forwarded["unconditional_transformer"], Ideogram4Transformer2DModel)
  assert type(forwarded["text_encoder"]).__name__ == "LlamaModel"
  # Both denoisers are separate objects, not one instance handed over twice.
  assert forwarded["transformer"] is not forwarded["unconditional_transformer"]


def test_the_injected_components_are_complete_and_meta_free(loader, tmp_path):
  """The strict rebuild is what proves every parameter is covered, so nothing stays on meta."""
  root = _fp8_checkpoint(tmp_path)

  _RecordingIdeogram4Pipeline.from_pretrained(str(root))

  for name in ("transformer", "unconditional_transformer", "text_encoder"):
    module = _SEEN["kwargs"][name]
    assert not any(tensor.is_meta for tensor in module.parameters()), name
    assert not any(tensor.is_meta for tensor in module.buffers()), name
    assert {parameter.dtype for parameter in module.parameters()} == {torch.bfloat16}, name
    assert module.training is False, name


def test_positional_arguments_are_forwarded(loader, tmp_path):
  plain = _plain_checkpoint(tmp_path / "plain")

  _RecordingIdeogram4Pipeline.from_pretrained(str(plain), "third-party-argument")

  assert list(_SEEN["args"]) == ["third-party-argument"]


# A per-component mapping, as `from_pretrained` accepts it. Named here because yapf rewrites an
# inline dict inside a `parametrize` list into one line per key.
_MAPPED_DTYPE = {"transformer": torch.float16, "default": torch.float32}
_MAPPED_TORCH_DTYPE = {"text_encoder": torch.float32}


@pytest.mark.parametrize(
  "dtype,torch_dtype,expected",
  [
    # `expected` is (transformer, unconditional_transformer, text_encoder).
    (None, None, (torch.bfloat16, torch.bfloat16, torch.bfloat16)),
    (torch.float16, None, (torch.float16, torch.float16, torch.float16)),
    (None, torch.float16, (torch.float16, torch.float16, torch.float16)),
    (_MAPPED_DTYPE, None, (torch.float16, torch.float32, torch.float32)),
    (None, _MAPPED_TORCH_DTYPE, (torch.bfloat16, torch.bfloat16, torch.float32)),
  ],
)
def test_dtype_is_resolved_like_diffusers(loader, tmp_path, dtype, torch_dtype, expected):
  root = _fp8_checkpoint(tmp_path)
  kwargs = {}
  if dtype is not None:
    kwargs["dtype"] = dtype
  if torch_dtype is not None:
    kwargs["torch_dtype"] = torch_dtype

  _RecordingIdeogram4Pipeline.from_pretrained(str(root), **kwargs)

  names = ("transformer", "unconditional_transformer", "text_encoder")
  for name, wanted in zip(names, expected):
    assert _SEEN["kwargs"][name].dtype is wanted, name
  if dtype is None and torch_dtype is None:
    assert _SEEN["kwargs"]["dtype"] is torch.bfloat16


def test_both_dtype_spellings_are_rejected_before_conversion(loader, tmp_path):
  root = _fp8_checkpoint(tmp_path)
  with pytest.raises(ValueError, match="not both"):
    _RecordingIdeogram4Pipeline.from_pretrained(
      str(root),
      dtype=torch.float16,
      torch_dtype=torch.float32,
    )


def test_explicit_components_are_never_replaced(loader, tmp_path):
  """A component the caller passed — `None` included — is a decision, not a gap to fill."""
  root = _fp8_checkpoint(tmp_path)
  given_transformer, given_text_encoder = object(), object()

  _RecordingIdeogram4Pipeline.from_pretrained(
    str(root),
    transformer=given_transformer,
    unconditional_transformer=None,
    text_encoder=given_text_encoder,
  )

  forwarded = _SEEN["kwargs"]
  assert forwarded["transformer"] is given_transformer
  assert forwarded["unconditional_transformer"] is None
  assert forwarded["text_encoder"] is given_text_encoder


def test_only_the_missing_components_are_injected(loader, tmp_path):
  root = _fp8_checkpoint(tmp_path)
  given_transformer = object()

  _RecordingIdeogram4Pipeline.from_pretrained(
    str(root),
    transformer=given_transformer,
    prompt_enhancer_head=None,
  )

  forwarded = _SEEN["kwargs"]
  assert forwarded["transformer"] is given_transformer
  assert isinstance(forwarded["unconditional_transformer"], Ideogram4Transformer2DModel)
  assert type(forwarded["text_encoder"]).__name__ == "LlamaModel"
  # Merged, not overwritten: the skipped optional component still reaches `from_pretrained`.
  assert forwarded["prompt_enhancer_head"] is None


@pytest.mark.parametrize("device_map", ["balanced", "cuda:0", True, {}])
def test_device_map_is_refused_before_any_weight_is_read(loader, tmp_path, monkeypatch, device_map):
  """Any `device_map` is refused: it would place the loaded components and strand the converted."""
  root = _fp8_checkpoint(tmp_path)

  def _unexpected(*args, **kwargs):
    raise AssertionError("no shard may be read for a call that is going to be refused")

  monkeypatch.setattr(ideogram4, "_ideogram4_component_state_dict", _unexpected)

  with pytest.raises(ValueError, match="device_map") as error:
    _RecordingIdeogram4Pipeline.from_pretrained(str(root), device_map=device_map)
  # The message speaks to the API, not to a CLI flag.
  assert "device-map-balance" not in str(error.value)
  assert _SEEN == {}


def test_hub_selectors_reach_the_marker_probe(loader, monkeypatch):
  """`revision` and friends have to select the snapshot the marker is read from."""
  import huggingface_hub
  from huggingface_hub.errors import EntryNotFoundError

  calls = {}

  def _fake_download(repo_id, filename, **kwargs):
    calls.update(repo_id=repo_id, filename=filename, kwargs=kwargs)
    raise EntryNotFoundError("no such file")

  monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)

  _RecordingIdeogram4Pipeline.from_pretrained(
    "some-org/ideogram-4-fp8",
    revision="refs/pr/1",
    token="hf_secret",
    local_files_only=True,
    cache_dir="/hub/cache",
  )

  assert calls["repo_id"] == "some-org/ideogram-4-fp8"
  assert calls["filename"] == "text_encoder/config.json"
  assert calls["kwargs"] == {
    "revision": "refs/pr/1",
    "token": "hf_secret",
    "local_files_only": True,
    "cache_dir": "/hub/cache",
  }
  # A repository whose marker is missing is not an FP8 checkpoint: the call goes out untouched.
  assert _SEEN["kwargs"]["revision"] == "refs/pr/1"
  assert "transformer" not in _SEEN["kwargs"]
