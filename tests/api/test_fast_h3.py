"""Regression tests for the FastH3 integration (contract, registrations, loading decisions).

FastH3 is MiniMax-H3 with a distilled sigma ladder and a VSA compression gate per block. Three
things can silently degrade it back to the base model, so each is pinned here:

1. The `fastvideo_inference.json` contract must be read strictly: a grid-point / forward-count
   mismatch or a non-64-token tile recipe has to fail instead of producing a different schedule.
2. The ladder must be shifted per modality before it reaches the scheduler, because
   `MiniMaxH3Scheduler.set_timesteps(sigmas=...)` uses the values verbatim.
3. The class names are new, so adapter and planner lookup must resolve them; a miss is a
   `ValueError` at `--parallel` time or a silently uncached run.

Everything here is CPU-only.
"""
import json

import pytest
import torch

import cache_dit  # noqa: F401
import cache_dit.caching.block_adapters  # noqa: F401  (registers the adapters)
from cache_dit.caching.block_adapters.block_registers import BlockAdapterRegister
from cache_dit.distributed.transformers import planners as planners_module
from cache_dit.distributed.transformers.register import (
  ContextParallelismPlannerRegister,
  TensorParallelismPlannerRegister,
)

pytest.importorskip("diffusers")
from cache_dit.models.pipelines.fast_h3 import (  # noqa: E402
  FASTH3_INFERENCE_FILE, FastH3Contract, FastH3Transformer3DModel,
)


def _write_checkpoint(tmp_path, contract, video_shift=12.0, audio_shift=3.0, with_gates=True):
  (tmp_path / "scheduler").mkdir()
  (tmp_path / "audio_scheduler").mkdir()
  (tmp_path / "transformer").mkdir()
  (tmp_path / FASTH3_INFERENCE_FILE).write_text(json.dumps(contract))
  (tmp_path / "scheduler" / "scheduler_config.json").write_text(json.dumps({"shift": video_shift}))
  (tmp_path / "audio_scheduler" / "scheduler_config.json").write_text(
    json.dumps({"shift": audio_shift}))
  weights = {"proj_in.weight": "a"}
  if with_gates:
    weights["transformer_blocks.0.attn.to_gate_compress.weight"] = "b"
  index = {"metadata": {}, "weight_map": weights}
  (tmp_path / "transformer" / "diffusion_pytorch_model.safetensors.index.json").write_text(
    json.dumps(index))
  return tmp_path


BASE_CONTRACT = {
  "schema_version": "fasth3-inference-contract-v1",
  "task": "t2av",
  "transformer_forwards": 4,
  "num_inference_steps": 5,
  "dmd_denoising_steps": [999, 749, 500, 250],
  "guidance_scale": 1.0,
  "vsa_tile_size": 64,
  "vsa_sparsity": 0.9,
}


def test_contract_parses_a_release(tmp_path):
  root = _write_checkpoint(tmp_path, BASE_CONTRACT)
  contract = FastH3Contract.from_checkpoint(root)
  assert contract is not None
  assert contract.transformer_forwards == 4
  assert contract.num_inference_steps == 5
  assert contract.dmd_denoising_steps == (999, 749, 500, 250)
  assert contract.vsa_sparsity == 0.9
  assert contract.requires_vsa
  assert (contract.video_shift, contract.audio_shift) == (12.0, 3.0)


def test_contract_absent_for_a_base_checkpoint(tmp_path):
  assert FastH3Contract.from_checkpoint(tmp_path) is None


def test_contract_rejects_a_grid_point_mismatch(tmp_path):
  broken = dict(BASE_CONTRACT, num_inference_steps=6)
  root = _write_checkpoint(tmp_path, broken)
  with pytest.raises(ValueError, match="one more point"):
    FastH3Contract.from_checkpoint(root)


def test_contract_rejects_a_step_count_mismatch(tmp_path):
  broken = dict(BASE_CONTRACT, dmd_denoising_steps=[999, 749, 500])
  root = _write_checkpoint(tmp_path, broken)
  with pytest.raises(ValueError, match="inconsistent"):
    FastH3Contract.from_checkpoint(root)


def test_contract_rejects_a_non_64_token_tile(tmp_path):
  broken = dict(BASE_CONTRACT, vsa_tile_size=256)
  root = _write_checkpoint(tmp_path, broken)
  with pytest.raises(ValueError, match="tiles"):
    FastH3Contract.from_checkpoint(root)


def test_sigmas_apply_the_modality_shift_once(tmp_path):
  root = _write_checkpoint(tmp_path, BASE_CONTRACT, video_shift=12.0, audio_shift=3.0)
  contract = FastH3Contract.from_checkpoint(root)
  video = contract.sigmas(contract.video_shift)
  audio = contract.sigmas(contract.audio_shift)
  assert video.shape == (5, ) and audio.shape == (5, )
  assert float(video[-1]) == 0.0 and float(audio[-1]) == 0.0
  # The shift maps 1.0 -> 1.0 and compresses the interior; audio (3.0) must stay above video (12.0)
  # for the second rung.
  # A larger shift compresses the interior harder, so the video ladder sits above the audio one.
  assert float(audio[1]) <= float(video[1]) < 1.0
  rung = 0.749
  expected = 12.0 * rung / (1.0 + 11.0 * rung)
  assert abs(float(video[1]) - expected) < 1e-6


def test_adapter_and_planners_resolve_the_new_class_names():

  class FastH3ModularPipeline:  # stands in for the pipeline; lookup is by class-name prefix
    pass

  class FastH3Transformer3DModelStub:
    pass

  assert BlockAdapterRegister.is_supported(FastH3ModularPipeline())
  assert BlockAdapterRegister.is_supported(FastH3Transformer3DModelStub())
  planners_module._activate_cp_planners()
  planners_module._activate_tp_planners()
  cp = ContextParallelismPlannerRegister.get_planner("FastH3Transformer3DModel")
  tp = TensorParallelismPlannerRegister.get_planner("FastH3Transformer3DModel")
  assert cp.__name__ == "FastH3ContextParallelismPlanner"
  assert tp.__name__ == "FastH3TensorParallelismPlanner"


def test_transformer_builds_the_gate_before_loading():
  config = {
    "num_attention_heads": 4,
    "attention_head_dim": 8,
    "hidden_size": 32,
    "num_layers": 1,
    "num_refiner_layers": 1,
    "ffn_dim": 16,
  }
  model = FastH3Transformer3DModel.from_config(config, enable_vsa=True)
  attn = model.transformer_blocks[0].attn
  assert attn.to_gate_compress.weight.shape == (4 * 8, 32)
  assert attn._vsa_gate_active is None, "the activation state is decided after the weights load"
  model.refresh_vsa_gate_state()
  assert not attn._vsa_gate_active, "a zero-initialised gate must keep the branch off"

  with torch.no_grad():
    attn.to_gate_compress.weight.fill_(0.5)
  model.refresh_vsa_gate_state()
  assert attn._vsa_gate_active, "a trained gate must switch the branch on"

  dense = FastH3Transformer3DModel.from_config(config, enable_vsa=False)
  assert not hasattr(dense.transformer_blocks[0].attn, "to_gate_compress")


def test_load_components_swaps_in_the_vsa_transformer(tmp_path):
  """The pipeline must load the FastH3 transformer class, gates included, from a real directory."""
  from cache_dit.models.pipelines.fast_h3 import FastH3Contract, FastH3ModularPipeline

  config = {
    "num_attention_heads": 4,
    "attention_head_dim": 8,
    "hidden_size": 32,
    "num_layers": 2,
    "num_refiner_layers": 1,
    "ffn_dim": 16,
  }
  model = FastH3Transformer3DModel.from_config(config, enable_vsa=True)
  with torch.no_grad():
    for block in model.transformer_blocks:
      block.attn.to_gate_compress.weight.fill_(0.25)
  model.save_pretrained(tmp_path / "transformer")

  (tmp_path / "scheduler").mkdir()
  (tmp_path / "audio_scheduler").mkdir()
  (tmp_path / "scheduler" / "scheduler_config.json").write_text(json.dumps({"shift": 12.0}))
  (tmp_path / "audio_scheduler" / "scheduler_config.json").write_text(json.dumps({"shift": 3.0}))
  (tmp_path / FASTH3_INFERENCE_FILE).write_text(json.dumps(BASE_CONTRACT))

  def _spec(name, subfolder):
    return [
      "diffusers", name, {
        "type_hint": ["diffusers", name],
        "pretrained_model_name_or_path": str(tmp_path),
        "subfolder": subfolder,
        "variant": None,
        "revision": None,
      }
    ]

  index = {
    "_class_name": "MiniMaxH3ModularPipeline",
    "_blocks_class_name": "MiniMaxH3Blocks",
    "_diffusers_version": "0.42.0.dev0",
    "transformer": _spec("MiniMaxH3Transformer3DModel", "transformer"),
    "scheduler": _spec("MiniMaxH3Scheduler", "scheduler"),
    "audio_scheduler": _spec("MiniMaxH3Scheduler", "audio_scheduler"),
  }
  (tmp_path / "modular_model_index.json").write_text(json.dumps(index))

  pipe = FastH3ModularPipeline.from_pretrained(str(tmp_path))
  pipe.load_components(names=["transformer"])

  assert isinstance(pipe.transformer, FastH3Transformer3DModel)
  assert isinstance(pipe.contract, FastH3Contract)
  assert pipe.contract.requires_vsa
  for block in pipe.transformer.transformer_blocks:
    assert block.attn.to_gate_compress is not None
    assert block.attn._vsa_gate_active, "a loaded non-zero gate must switch the branch on"


def test_vsa_rejects_a_head_count_that_does_not_divide_the_ulysses_degree():
  """The Ulysses exchange splits heads, so a non-dividing degree must fail before any collective."""
  from types import SimpleNamespace

  from cache_dit.models.pipelines.fast_h3 import _ulysses_gather_sequence

  config = SimpleNamespace(ulysses_degree=5, _ulysses_mesh=None)
  q = torch.zeros(1, 8, 56, 128)
  with pytest.raises(ValueError, match="divide the Ulysses degree"):
    _ulysses_gather_sequence(q, q, q, None, 56, config)


def test_contract_shift_wins_over_a_stale_scheduler_config(tmp_path):
  """The contract is the distillation record; a disagreeing scheduler config must not win."""
  contract = dict(BASE_CONTRACT, video_scheduler_shift=10.0, audio_scheduler_shift=3.0)
  root = _write_checkpoint(tmp_path, contract, video_shift=12.0, audio_shift=3.0)
  parsed = FastH3Contract.from_checkpoint(root)
  assert parsed.video_shift == 10.0, "the contract shift must override the stale config"
  assert parsed.audio_shift == 3.0


def test_contract_falls_back_to_the_scheduler_config(tmp_path):
  """The 4-step preview ships no shift in its contract."""
  root = _write_checkpoint(tmp_path, BASE_CONTRACT, video_shift=12.0, audio_shift=3.0)
  parsed = FastH3Contract.from_checkpoint(root)
  assert (parsed.video_shift, parsed.audio_shift) == (12.0, 3.0)
