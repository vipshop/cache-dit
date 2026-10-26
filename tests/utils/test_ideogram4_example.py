"""Regression tests for the Ideogram 4 `generate` CLI example.

`Ideogram4Pipeline` is the first T2I example whose guidance is a per-step schedule instead of a
scalar, and `check_inputs` rejects a schedule whose length differs from `num_inference_steps`. The
failure modes pinned down here:

1. A schedule baked into `ExampleInputData` stays 48 entries long no matter what
   `--num-inference-steps` / `--warmup-num-inference-steps` ask for, and the run then dies in
   `check_inputs` after the weights are already on the device.
2. `ExampleInputData.data()` drops `None` but keeps `False`, so `prompt_upsampling=False` has to
   reach the pipeline while `guidance_scale` must not.
3. Setting both `guidance_scale` and `guidance_schedule` raises, so the scalar has to be removed
   when the schedule is filled in.
4. The default checkpoint is the official FP8 release, whose fused `qkv`, `attention.o` naming and
   per-row scales no `from_pretrained` route maps onto the module parameters: the unmatched
   parameters stay on the meta device and `pipe.to(device)` fails. The conversion is pinned down
   against tiny checkpoints written in the same on-disk layout, and a checkpoint without the
   `ideogram_fp8_weight_only` marker has to stay on the plain path.

Usage:
  python -m pytest tests/utils/test_ideogram4_example.py -q
"""
import argparse
import json
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

try:
  from diffusers import Ideogram4Pipeline, Ideogram4Transformer2DModel
except ImportError:
  pytest.skip("diffusers without Ideogram4 support", allow_module_level=True)

from cache_dit._utils import ideogram4_example as public_ideogram4_example
from cache_dit._utils import registers
from cache_dit._utils.examples import (
  _IDEOGRAM4_PROMPT,
  _IDEOGRAM4_STEPS,
  _Ideogram4Example,
  _Ideogram4InitConfig,
  _ideogram4_guidance_schedule,
  _ideogram4_bf16_text_encoder,
  _ideogram4_bf16_transformer,
  _ideogram4_component_state_dict,
  _ideogram4_convert_tensor,
  _ideogram4_is_fp8,
  ideogram4_example,
)
from cache_dit._utils.registers import ExampleInitConfig, ExampleRegister, ExampleType

# kwargs of the latest stand-in pipeline call
_SEEN = {}


@pytest.fixture(autouse=True)
def _cpu_distributed(monkeypatch):
  """`Example.__init__` asks the distributed helpers whether this run is parallel."""
  monkeypatch.setattr(registers, "maybe_init_distributed", lambda _args: (0, "cpu"))


def _factory_args(**overrides) -> argparse.Namespace:
  """Namespace with the attributes the example factory and `Example.__init__` read."""
  values = dict(
    model_path=None,
    num_inference_steps=None,
    parallel_type=None,
    config_path=None,
  )
  values.update(overrides)
  return argparse.Namespace(**values)


def _data_args(**overrides) -> argparse.Namespace:
  """Namespace with the attributes `ExampleInputData.data` reads."""
  values = dict(
    model_path=None,
    prompt=None,
    negative_prompt=None,
    skip_negative_prompt=False,
    height=None,
    width=None,
    num_inference_steps=None,
    num_frames=None,
    image_path=None,
    last_image_path=None,
    mask_image_path=None,
    input_height=None,
    input_width=None,
    generator_device=None,
    seed=None,
    extra_input_kwargs=None,
  )
  values.update(overrides)
  return argparse.Namespace(**values)


def _capture(**kwargs) -> SimpleNamespace:
  """Stand-in for the pipeline: records the kwargs of one call."""
  _SEEN.clear()
  _SEEN.update(kwargs)
  return SimpleNamespace(images=[None])


def test_example_is_registered_with_the_hf_model_id():
  assert "ideogram4" in ExampleRegister.list_examples()
  assert ExampleRegister.get_default("ideogram4") == "ideogram-ai/ideogram-4-fp8"


def test_public_reexport_resolves_to_the_example_factory():
  """`_safe_import` degrades to a stub on an ImportError, which would hide broken wiring."""
  assert public_ideogram4_example is ideogram4_example


def test_env_var_points_the_example_at_a_local_checkout(monkeypatch):
  monkeypatch.setenv("IDEOGRAM4_DIR", "/models/ideogram-v4")
  example = ideogram4_example(_factory_args())
  assert example.init_config.model_name_or_path == "/models/ideogram-v4"


def test_defaults_are_the_documented_ones():
  example = ideogram4_example(_factory_args())
  assert isinstance(example, _Ideogram4Example)
  assert example.init_config.task_type is ExampleType.T2I
  assert example.init_config.pipeline_class is Ideogram4Pipeline
  # The prompt upsampler rewrites the prompt, so the head is dropped instead of loaded.
  assert example.init_config.skip_components == ["prompt_enhancer_head"]
  # Asymmetric CFG: two transformers, one forward each, blended outside of both.
  assert example.init_config.extra_optimize_kwargs == {"enable_separate_cfg": False}

  input_data = example.input_data
  assert input_data.prompt == _IDEOGRAM4_PROMPT
  assert input_data.num_inference_steps == _IDEOGRAM4_STEPS
  assert (input_data.height, input_data.width) == (1024, 1024)
  assert input_data.guidance_scale is None


def test_default_prompt_is_the_official_structured_json_caption():
  prompt = json.loads(_IDEOGRAM4_PROMPT)
  assert list(prompt) == [
    "high_level_description",
    "style_description",
    "compositional_deconstruction",
  ]
  assert list(prompt["style_description"]) == ["aesthetics", "lighting", "photo", "medium"]
  composition = prompt["compositional_deconstruction"]
  assert list(composition) == ["background", "elements"]
  assert list(composition["elements"][0]) == ["type", "bbox", "desc"]
  assert list(composition["elements"][1]) == ["type", "bbox", "text", "desc"]
  assert composition["elements"][1]["text"] == "hello world"
  assert [element["type"] for element in composition["elements"]] == ["obj", "text"]
  assert [element["bbox"] for element in composition["elements"]] == [
    [120, 100, 920, 900],
    [500, 300, 680, 700],
  ]
  assert _IDEOGRAM4_PROMPT == json.dumps(
    prompt,
    separators=(",", ":"),
    ensure_ascii=False,
  )


def test_cli_step_override_reaches_the_example():
  example = ideogram4_example(_factory_args(num_inference_steps=8))
  assert example.input_data.num_inference_steps == 8


@pytest.mark.parametrize("steps", [48, 30, 8, 4, 3, 2, 1])
def test_schedule_length_always_matches_the_step_count(steps):
  assert len(_ideogram4_guidance_schedule(steps)) == steps


def test_schedule_keeps_the_polish_tail_bounded():
  assert _ideogram4_guidance_schedule(48) == [7.0] * 45 + [3.0] * 3
  assert _ideogram4_guidance_schedule(4) == [7.0, 3.0, 3.0, 3.0]
  assert _ideogram4_guidance_schedule(2) == [3.0, 3.0]
  assert _ideogram4_guidance_schedule(1) == [3.0]


def test_call_pipe_rebuilds_the_schedule_for_the_forwarded_steps():
  example = ideogram4_example(_factory_args())
  for steps in (48, 30, 2):
    example.call_pipe(_capture, {"prompt": "a cat", "num_inference_steps": steps})
    assert _SEEN["guidance_schedule"] == _ideogram4_guidance_schedule(steps)


def test_warmup_step_override_gets_its_own_schedule():
  """`run()` overwrites `num_inference_steps` for the warmup pass; the schedule has to follow."""
  example = ideogram4_example(_factory_args())
  example.call_pipe(_capture, {"prompt": "a cat", "num_inference_steps": 2})
  assert _SEEN["num_inference_steps"] == 2
  assert _SEEN["guidance_schedule"] == [3.0, 3.0]


def test_missing_step_count_falls_back_to_the_example_default():
  example = ideogram4_example(_factory_args(num_inference_steps=30))
  example.call_pipe(_capture, {"prompt": "a cat"})
  assert _SEEN["guidance_schedule"] == _ideogram4_guidance_schedule(30)


def test_call_pipe_preserves_explicit_guidance_scale():
  """Setting both the scalar and the schedule raises in `check_inputs`."""
  example = ideogram4_example(_factory_args())
  example.call_pipe(_capture, {"prompt": "a cat", "guidance_scale": 5.0})
  assert _SEEN["guidance_scale"] == 5.0
  assert _SEEN["guidance_schedule"] is None


def test_input_data_forwards_prompt_upsampling_but_no_guidance():
  data = ideogram4_example(_factory_args()).input_data.data(_data_args())
  assert data["prompt_upsampling"] is False
  # None values are filtered, so the mutually exclusive scalar never reaches the pipeline.
  assert "guidance_scale" not in data
  # The schedule is derived per call instead of being baked into the input dict.
  assert "guidance_schedule" not in data


def test_generated_schedules_pass_the_pipeline_input_check():
  input_data = ideogram4_example(_factory_args()).input_data
  # `check_inputs` only touches these two attributes before validating the schedule.
  stub = SimpleNamespace(vae_scale_factor=8, patch_size=2)

  def _check(steps: int, schedule) -> None:
    Ideogram4Pipeline.check_inputs(
      stub,
      prompt=input_data.prompt,
      height=input_data.height,
      width=input_data.width,
      num_inference_steps=steps,
      guidance_scale=None,
      guidance_schedule=schedule,
    )

  for steps in (input_data.num_inference_steps, 8, 2):
    _check(steps, _ideogram4_guidance_schedule(steps))
  # A 48-entry schedule forwarded to a shorter run is exactly what the override prevents.
  with pytest.raises(ValueError, match="must have length"):
    _check(8, _ideogram4_guidance_schedule(_IDEOGRAM4_STEPS))


def test_default_schedule_tracks_the_pipeline_default():
  """48 steps is not arbitrary: it is the length of the pipeline's own default schedule."""
  default = inspect.signature(Ideogram4Pipeline.__call__).parameters["guidance_schedule"].default
  assert list(default) == _ideogram4_guidance_schedule(_IDEOGRAM4_STEPS)


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


def _init_config(model_path: str) -> _Ideogram4InitConfig:
  return _Ideogram4InitConfig(
    task_type=ExampleType.T2I,
    model_name_or_path=model_path,
    pipeline_class=Ideogram4Pipeline,
    skip_components=["prompt_enhancer_head"],
  )


def _pipe_args(**overrides) -> argparse.Namespace:
  """Namespace with the attributes `ExampleInitConfig.get_pipe` reads."""
  values = dict(
    model_path=None,
    device_map_balance=False,
    quantize=False,
    quantize_text_encoder=False,
  )
  values.update(overrides)
  return argparse.Namespace(**values)


def test_fp8_marker_comes_from_the_text_encoder_config(tmp_path):
  assert _ideogram4_is_fp8(str(_fp8_checkpoint(tmp_path))) is True

  # A bfloat16 re-upload carries neither the file nor the flag.
  plain = tmp_path / "plain"
  (plain / "text_encoder").mkdir(parents=True)
  assert _ideogram4_is_fp8(str(plain)) is False
  (plain / "text_encoder" / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))
  assert _ideogram4_is_fp8(str(plain)) is False
  (plain / "text_encoder" / "config.json").write_text(
    json.dumps({"ideogram_fp8_weight_only": False}))
  assert _ideogram4_is_fp8(str(plain)) is False

  # A mistyped local path stays local: it must not turn into a hub round trip.
  assert _ideogram4_is_fp8(str(tmp_path / "missing")) is False


def test_fused_qkv_weight_is_split_on_dim_zero():
  """`to_q`, `to_k` and `to_v` are the three thirds of the fused weight, each with its own scale."""
  weight = torch.arange(12, dtype=torch.float32).reshape(6, 2).to(_FP8_DTYPE)
  scale = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  converted = _ideogram4_convert_tensor("layers.3.attention.qkv.weight", weight, scale,
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
  converted = _ideogram4_convert_tensor("layers.0.attention.o.weight", weight, scale,
                                        torch.bfloat16)

  assert set(converted) == {"layers.0.attention.to_out.0.weight"}
  assert torch.equal(converted["layers.0.attention.to_out.0.weight"], _dequantize(weight, scale))


def test_a_weight_without_a_scale_is_only_cast():
  """Norms and biases carry no scale and need no renaming."""
  weight = torch.randn(4, 4, dtype=torch.bfloat16)
  converted = _ideogram4_convert_tensor("layers.0.attention_norm1.weight", weight, None,
                                        torch.bfloat16)
  assert set(converted) == {"layers.0.attention_norm1.weight"}
  assert torch.equal(converted["layers.0.attention_norm1.weight"], weight)


def test_scale_tensors_never_reach_the_module(tmp_path):
  root = _fp8_checkpoint(tmp_path)
  converted = _ideogram4_component_state_dict(str(root), "transformer", "diffusion_pytorch_model",
                                              torch.bfloat16)

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
  loaded = _ideogram4_bf16_transformer(str(root), subfolder, torch.bfloat16)

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

  converted = _ideogram4_component_state_dict(str(tmp_path), "transformer",
                                              "diffusion_pytorch_model", torch.bfloat16)
  assert torch.equal(converted["layers.0.attention.to_out.0.weight"], _dequantize(quantized, scale))


def test_fp8_text_encoder_is_dequantized_the_same_way(tmp_path):
  root = _fp8_checkpoint(tmp_path)
  encoder = _ideogram4_bf16_text_encoder(str(root), torch.bfloat16)

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


def test_a_non_fp8_checkpoint_keeps_the_plain_loading_path(tmp_path):
  """The conversion must not run for a checkpoint without the marker."""
  plain = tmp_path / "plain"
  (plain / "text_encoder").mkdir(parents=True)
  (plain / "text_encoder" / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))

  components = _init_config(str(plain))._custom_components_kwargs()
  reference = ExampleInitConfig(
    task_type=ExampleType.T2I,
    model_name_or_path=str(plain),
    pipeline_class=Ideogram4Pipeline,
    skip_components=["prompt_enhancer_head"],
  )._custom_components_kwargs()
  assert components == reference
  # Only the skipped optional component, which `from_pretrained` must see as `None`.
  assert components == {"prompt_enhancer_head": None}


def test_fp8_checkpoint_components_are_injected(tmp_path):
  root = _fp8_checkpoint(tmp_path)
  components = _init_config(str(root))._custom_components_kwargs()

  assert isinstance(components["transformer"], Ideogram4Transformer2DModel)
  assert isinstance(components["unconditional_transformer"], Ideogram4Transformer2DModel)
  assert type(components["text_encoder"]).__name__ == "LlamaModel"
  # Both denoisers are separate objects, not one instance handed over twice.
  assert components["transformer"] is not components["unconditional_transformer"]
  # The skipped component still has to reach `from_pretrained` as `None`.
  assert components["prompt_enhancer_head"] is None


def test_device_map_balance_is_refused_before_anything_is_loaded(tmp_path):
  """`--device-map-balance` skips the `pipe.to` that would place the converted components."""
  root = _fp8_checkpoint(tmp_path)
  with pytest.raises(ValueError, match="device-map-balance"):
    _init_config(str(root)).get_pipe(_pipe_args(device_map_balance=True))


def test_device_map_balance_stays_available_for_a_non_fp8_checkpoint(tmp_path, monkeypatch):
  plain = tmp_path / "plain"
  (plain / "text_encoder").mkdir(parents=True)
  (plain / "text_encoder" / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}))

  seen = {}

  def _fake_from_pretrained(*args, **kwargs):
    seen.update(kwargs)
    return SimpleNamespace()

  monkeypatch.setattr(Ideogram4Pipeline, "from_pretrained", _fake_from_pretrained)
  _init_config(str(plain)).get_pipe(_pipe_args(device_map_balance=True))

  assert seen["device_map"] == "balanced"
  # The plain path passes no components of its own besides the skipped optional one.
  assert "transformer" not in seen
  assert seen["prompt_enhancer_head"] is None


def test_the_factory_uses_the_converting_init_config():
  example = ideogram4_example(_factory_args())
  assert isinstance(example.init_config, _Ideogram4InitConfig)
  assert isinstance(example.init_config, ExampleInitConfig)


def test_model_path_override_reaches_the_conversion(tmp_path):
  """`--model-path` has to feed the converter and not just `from_pretrained`'s own lookup."""
  root = _fp8_checkpoint(tmp_path)
  example = ideogram4_example(_factory_args(model_path=str(root)))

  assert example.init_config.model_name_or_path == str(root)
  components = example.init_config._custom_components_kwargs()
  assert isinstance(components["transformer"], Ideogram4Transformer2DModel)
  assert isinstance(components["text_encoder"], torch.nn.Module)
