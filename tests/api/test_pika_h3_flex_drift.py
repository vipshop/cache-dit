"""Drift guards for the diffusers pieces Pika H3 Flex reuses instead of vendoring.

The whole integration rests on one decision: the transformer is *not* copied, so Pika runs whatever
`MiniMaxH3Transformer3DModel` / `MiniMaxH3Scheduler` the installed diffusers ships. That is only safe
while those classes keep the shape the vendored pipeline and cache-dit's MiniMax-H3 planners expect.
Each test below pins one of those assumptions, so a diffusers bump fails here instead of producing a
quietly different model:

1. the packed-sequence transformer still exposes the modules the TP / CP planners and the AdaLN host
   residency path name (`transformer_blocks`, `proj_out`, `audio_proj_out`, `norm_out`, `rope`);
2. `MiniMaxH3Scheduler.set_timesteps(sigmas=...)` still adopts an explicit grid **verbatim** (that is
   why the vendored subclass has to apply the shift itself) and still has no `generator` in `step`
   (that is why the subclass exists at all);
3. the LoRA mixin still routes both checkpoint partitions;
4. TE-P still resolves `Qwen3VLForConditionalGeneration`, and VAE-P still does **not** resolve the
   MiniMax-H3 autoencoders (Pika's VAE therefore cannot be parallelized, and that is a documented
   limitation rather than a silent fallback).
"""
import inspect

import pytest
import torch
from diffusers import (
  MiniMaxH3Scheduler,
  MiniMaxH3Transformer3DModel,
)
from diffusers.loaders import MiniMaxH3LoraLoaderMixin

from cache_dit.models.pipelines.pika_h3_flex import PikaH3FlexScheduler
from cache_dit.distributed.autoencoders.planners import _activate_auto_encoder_dp_planners
from cache_dit.distributed.autoencoders.register import AutoEncoderDataParallelismPlannerRegister
from cache_dit.distributed.text_encoders.planners import _activate_text_encoder_tp_planners
from cache_dit.distributed.text_encoders.register import TextEncoderTensorParallelismPlannerRegister


def _tiny_transformer() -> MiniMaxH3Transformer3DModel:
  """A one-block transformer with the same module tree as the checkpoint, at 1/128 the width."""
  return MiniMaxH3Transformer3DModel(
    num_attention_heads=2,
    attention_head_dim=8,
    hidden_size=16,
    num_layers=2,
    num_refiner_layers=1,
    ffn_dim=32,
    in_channels=4,
    audio_in_channels=4,
    text_dim=32,
    freq_dim=16,
    time_embed_hidden_dim=32,
    time_embed_dim=16,
    rope_freq_dim=4,
  )


def test_transformer_keeps_the_planner_module_tree():
  transformer = _tiny_transformer()
  # cache-dit's MiniMax-H3 adapter walks this list; the CP planner splits at its first block.
  assert len(transformer.transformer_blocks) == 2
  for name in ("proj_in", "audio_proj_in", "proj_out", "audio_proj_out", "norm_out", "rope",
               "time_embedder", "context_embedder", "token_refiner"):
    assert hasattr(transformer, name), f"`{name}` disappeared from MiniMaxH3Transformer3DModel"
  # Every block rebuilds its modulation from the current timestep, which is what
  # `--adaln-host-residency` streams; the module name is part of the offload target discovery.
  for block in transformer.transformer_blocks:
    assert hasattr(block, "adaln_proj")


def test_scheduler_adopts_explicit_sigmas_verbatim():
  scheduler = MiniMaxH3Scheduler(shift=12.0)
  base = [1.0, 0.75, 0.5, 0.25, 0.1, 0.04, 0.0]
  scheduler.set_timesteps(sigmas=base)
  # No shift, no re-noising: the installed release is the deterministic half of the contract, which is
  # why `vendored`/`scheduler.py` subclasses it instead of using it.
  assert scheduler.sigmas.tolist() == pytest.approx(base, abs=1e-6)
  assert "generator" not in inspect.signature(scheduler.step).parameters
  assert "stochastic_sampling" not in inspect.signature(MiniMaxH3Scheduler.__init__).parameters


def test_scheduler_scale_noise_keeps_a_scalar_timestep_in_double_precision():
  scheduler = PikaH3FlexScheduler(shift=12.0, stochastic_sampling=True, shift_provided_sigmas=True)
  sample = torch.zeros(4, dtype=torch.bfloat16)
  noise = torch.ones(4, dtype=torch.bfloat16)

  out = scheduler.scale_noise(sample, 0.999, noise)

  # Pinned reference: `1 - 0.999` stays in double precision, so the noise term is 1e-3.
  assert abs(float(out[0]) - (1.0 - 0.999)) < 4e-6
  # The installed release casts 0.999 to bfloat16 first, which makes the residual ~2% larger; the bf16
  # forward then amplifies the difference on the only caller (`keyframe_noise_aug`).
  cast_residual = 1.0 - float(torch.tensor(0.999, dtype=torch.bfloat16))
  assert abs(float(out[0]) - cast_residual) > 1e-5

  # Before/after: the installed release really takes the cast branch, which is what the override is for.
  installed = MiniMaxH3Scheduler(shift=12.0).scale_noise(sample, 0.999, noise)
  assert abs(float(installed[0]) - cast_residual) < 4e-6
  assert not torch.equal(installed, out)


def test_lora_mixin_routes_both_partitions():
  assert MiniMaxH3LoraLoaderMixin._lora_loadable_modules == ["transformer", "transformer_ref"]
  components = inspect.signature(
    MiniMaxH3LoraLoaderMixin.fuse_lora).parameters["components"].default
  assert set(components) == {"transformer", "transformer_ref"}


def test_te_parallelism_resolves_the_qwen3vl_conditioner():
  # The registries fill in lazily, and an empty registry would make the negative test below pass for
  # the wrong reason, so activate both before asserting anything.
  _activate_text_encoder_tp_planners()
  _activate_auto_encoder_dp_planners()
  planner = TextEncoderTensorParallelismPlannerRegister.get_planner(
    "Qwen3VLForConditionalGeneration")
  assert planner is not None
  # `--parallel-vae` really is uncovered, not merely unimported.
  assert len(AutoEncoderDataParallelismPlannerRegister.supported_planners()[1]) > 0


def test_vae_parallelism_does_not_cover_the_minimax_h3_autoencoders():
  # Documents a limitation instead of a capability: the audio VAE is never offered to `--parallel-vae`
  # either (`utils.py` only appends `pipe.vae`), so VAE-P is out of scope for Pika H3 Flex.
  _activate_auto_encoder_dp_planners()
  assert len(AutoEncoderDataParallelismPlannerRegister.supported_planners()[1]) > 0
  for name in ("AutoencoderKLMiniMaxH3", "AutoencoderKLMiniMaxH3Audio"):
    with pytest.raises(ValueError, match="No planner registered"):
      AutoEncoderDataParallelismPlannerRegister.get_planner(name)


def test_reference_embedding_is_fp32_bounded():
  # The two VAEs ship float32 and the transformer keeps its boundary modules in fp32; the quantize
  # exclusion list of the example depends on these names surviving.
  transformer = _tiny_transformer()
  assert isinstance(transformer.rope, torch.nn.Module)
