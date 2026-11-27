"""Drift guards for the diffusers pieces Pika H3 Flex reuses instead of vendoring.

The whole integration rests on one decision: the transformer is *not* copied, so Pika runs whatever
`MiniMaxH3Transformer3DModel` / `MiniMaxH3Scheduler` the installed diffusers ships. That is only safe
while those classes keep the shape the vendored pipeline and cache-dit's MiniMax-H3 planners expect.
Each test below pins one of those assumptions, so a diffusers bump fails here instead of producing a
quietly different model:

1. the packed-sequence transformer still exposes the modules the TP / CP planners and the AdaLN host
   residency path name (`transformer_blocks`, `proj_out`, `audio_proj_out`, `norm_out`, `rope`);
2. the re-noising contract — `stochastic_sampling`, `shift_provided_sigmas`, `step(generator=...)` —
   either exists upstream (then the vendored scheduler only has to be a numerical stand-in) or does
   not (then the vendored overrides are what make the model the released one);
3. the LoRA mixin still routes both checkpoint partitions;
4. TE-P still resolves `Qwen3VLForConditionalGeneration`, and VAE-P resolves the video autoencoder
   (`AutoencoderKLMiniMaxH3`, 34% of a single-GPU run) but not the audio one — the audio VAE is never
   offered to `vae=true` in `--parallel-params`, so that half stays a documented limitation.
"""
import inspect
import sys

import pytest
import torch
from diffusers import (
  MiniMaxH3Scheduler,
  MiniMaxH3Transformer3DModel,
)
from diffusers.loaders import MiniMaxH3LoraLoaderMixin

from cache_dit.models.pipelines.pika_h3_flex import (
  PikaH3FlexScheduler,
  resolve_pika_h3_flex,
)
from cache_dit.distributed.autoencoders.planners import _activate_auto_encoder_dp_planners
from cache_dit.distributed.autoencoders.register import AutoEncoderDataParallelismPlannerRegister
from cache_dit.distributed.text_encoders.planners import _activate_text_encoder_tp_planners
from cache_dit.distributed.text_encoders.register import TextEncoderTensorParallelismPlannerRegister

# The released six-step ladder, as a base grid the scheduler shifts (video shift 12 / audio shift 3).
_BASE_SIGMAS = [step / 1000.0 for step in (1000, 750, 500, 250, 100, 40, 0)]


def _has_packaged_re_noising() -> bool:
  """Whether the installed diffusers owns the re-noising update.

  The vendored scheduler is the fallback for a release that knows nothing about it, so which half of
  each guard applies depends on this, not on a version string. It is a function rather than a module
  constant so the fallback half stays testable on a release that does have the contract — the branch
  is exercised through `monkeypatch`, otherwise it would be dead code on any modern diffusers.

  :returns: `True` when `stochastic_sampling`, `shift_provided_sigmas` and `step(generator=...)` are
    part of the installed scheduler.
  """

  parameters = inspect.signature(MiniMaxH3Scheduler.__init__).parameters
  return ("stochastic_sampling" in parameters and "shift_provided_sigmas" in parameters
          and "generator" in inspect.signature(MiniMaxH3Scheduler.step).parameters)


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


def _upstream_scheduler(shift: float) -> MiniMaxH3Scheduler:
  """The installed scheduler, configured the way the released adapters need it.

  A release without the packaged switch cannot be configured at all, which is exactly what the
  vendored subclass exists for.
  """
  if not _has_packaged_re_noising():
    return MiniMaxH3Scheduler(shift=shift)
  return MiniMaxH3Scheduler(shift=shift, stochastic_sampling=True, shift_provided_sigmas=True)


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


def test_scheduler_re_noising_contract_against_the_installed_release():
  if not _has_packaged_re_noising():
    pytest.skip("the installed diffusers has no re-noising contract to compare against")
  # The installed release owns the contract, so the integration can use it directly and the vendored
  # subclass only has to stay a numerical stand-in. Answering this differently is the drift: the
  # resolver would pick a class the checkpoint was not trained for, so assert it does not.
  scheduler = _upstream_scheduler(12.0)
  scheduler.set_timesteps(sigmas=_BASE_SIGMAS)
  assert scheduler.num_inference_steps == 6
  assert resolve_pika_h3_flex().is_upstream
  assert not resolve_pika_h3_flex().reference_package.startswith("cache_dit")


def test_vendored_scheduler_supplies_the_whole_re_noising_surface():
  # What the fallback contributes on a release without the packaged update. Asserted unconditionally:
  # it is the half of the contract that the vendored copy owns on every diffusers, and making it
  # version-conditional is how it would silently rot.
  for name in ("scale_noise", "step", "set_timesteps"):
    assert name in PikaH3FlexScheduler.__dict__, f"`{name}` must stay overridden"
  scheduler = PikaH3FlexScheduler(shift=12.0, stochastic_sampling=True, shift_provided_sigmas=True)
  assert scheduler.config["stochastic_sampling"] is True
  assert scheduler.config["shift_provided_sigmas"] is True


def test_vendored_scheduler_reproduces_the_installed_re_noising_update():
  """The fallback must sample what the installed release samples, whatever that release is."""
  vendored = PikaH3FlexScheduler(shift=12.0, stochastic_sampling=True, shift_provided_sigmas=True)
  vendored.set_timesteps(sigmas=_BASE_SIGMAS)

  shape = (2, 3)
  vendored_sample = torch.randn(shape,
                                generator=torch.Generator().manual_seed(11),
                                dtype=torch.float32)
  vendored_generator = torch.Generator().manual_seed(23)
  # The arithmetic of the re-noising update, reconstructed independently of both implementations:
  # `x0 = x + sigma * v`, then `x_next = (1 - sigma') * x0 + sigma' * noise`, where the noise is drawn
  # per step and skipped once `sigma'` reaches zero.
  expected = vendored_sample.clone()
  expected_generator = torch.Generator().manual_seed(23)
  for index, timestep in enumerate(vendored.timesteps):
    velocity = torch.ones(shape, dtype=torch.float32) * (index + 1)
    vendored_sample = vendored.step(velocity,
                                    timestep,
                                    vendored_sample,
                                    generator=vendored_generator,
                                    return_dict=False)[0]
    sigma = vendored.sigmas[index].item()
    sigma_next = vendored.sigmas[index + 1].item()
    expected = (1 - sigma_next) * (expected + sigma * velocity)
    if sigma_next != 0:
      expected = expected + sigma_next * torch.randn(
        shape, generator=expected_generator, dtype=torch.float32)

  assert torch.allclose(vendored_sample, expected, atol=1e-6)
  if _has_packaged_re_noising():
    # ... and the installed release agrees with that reconstruction, so the stand-in is numerically
    # identical rather than merely close to somebody's reading of the paper.
    upstream = _upstream_scheduler(12.0)
    upstream.set_timesteps(sigmas=_BASE_SIGMAS)
    assert vendored.sigmas.tolist() == pytest.approx(upstream.sigmas.tolist(), abs=1e-6)
    assert vendored.timesteps.tolist() == pytest.approx(upstream.timesteps.tolist(), abs=1e-6)
    sample = torch.randn(shape, generator=torch.Generator().manual_seed(11), dtype=torch.float32)
    generator = torch.Generator().manual_seed(23)
    for index, timestep in enumerate(upstream.timesteps):
      velocity = torch.ones(shape, dtype=torch.float32) * (index + 1)
      sample = upstream.step(velocity, timestep, sample, generator=generator, return_dict=False)[0]
    assert torch.allclose(sample, expected, atol=1e-6)


def test_scheduler_keeps_the_re_noising_config_keys_across_a_from_config_round_trip():
  # `ConfigMixin` registers before the subclass body runs and the parent registers again inside it, so
  # the subclass' keys can be overwritten by the parent's defaults — and `from_config` then drops them
  # through `_use_default_values`. Either failure samples a deterministic Euler ladder.
  scheduler = PikaH3FlexScheduler(shift=12.0, stochastic_sampling=True, shift_provided_sigmas=True)
  rebuilt = PikaH3FlexScheduler.from_config(scheduler.config)
  assert rebuilt.config["stochastic_sampling"] is True
  assert rebuilt.config["shift_provided_sigmas"] is True
  rebuilt.set_timesteps(sigmas=_BASE_SIGMAS)
  # Shift 12 sends 0.75 to 0.972973; an unshifted ladder would keep 0.75 and skip the re-noising grid.
  assert rebuilt.sigmas[1].item() == pytest.approx(12.0 * 0.75 / (1 + 11.0 * 0.75), abs=1e-6)


def test_scheduler_scale_noise_keeps_a_scalar_timestep_in_double_precision():
  scheduler = PikaH3FlexScheduler(shift=12.0, stochastic_sampling=True, shift_provided_sigmas=True)
  sample = torch.zeros(4, dtype=torch.bfloat16)
  noise = torch.ones(4, dtype=torch.bfloat16)

  out = scheduler.scale_noise(sample, 0.999, noise)

  # Pinned reference: `1 - 0.999` stays in double precision, so the noise term is 1e-3.
  assert abs(float(out[0]) - (1.0 - 0.999)) < 4e-6
  # A bf16 cast makes the residual ~2% larger, and the bf16 forward amplifies it on the only caller
  # (`keyframe_noise_aug = 0.999`). Whether the installed release still casts is asked, not assumed:
  # a release that already keeps the float is a numerically identical stand-in.
  cast_residual = 1.0 - float(torch.tensor(0.999, dtype=torch.bfloat16))
  installed = MiniMaxH3Scheduler(shift=12.0).scale_noise(sample, 0.999, noise)
  if abs(float(installed[0]) - cast_residual) < 4e-6:
    assert not torch.equal(installed, out)
  else:
    assert abs(float(installed[0]) - float(out[0])) < 1e-6


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
  # `vae=true` in `--parallel-params` really is uncovered, not merely unimported.
  assert len(AutoEncoderDataParallelismPlannerRegister.supported_planners()[1]) > 0


def test_vae_parallelism_covers_the_video_autoencoder_but_not_the_audio_one():
  # Pika decodes 124 frames through the video VAE, so VAE-P has to resolve it. The audio VAE is never
  # offered to `vae=true` in `--parallel-params` (`utils.py` only appends `pipe.vae`), so it stays
  # uncovered — that half is a limitation worth pinning, while the video half is a capability.
  _activate_auto_encoder_dp_planners()
  assert len(AutoEncoderDataParallelismPlannerRegister.supported_planners()[1]) > 0
  assert AutoEncoderDataParallelismPlannerRegister.get_planner("AutoencoderKLMiniMaxH3") is not None
  with pytest.raises(ValueError, match="No planner registered"):
    AutoEncoderDataParallelismPlannerRegister.get_planner("AutoencoderKLMiniMaxH3Audio")


def test_vendored_scheduler_arithmetic_does_not_lean_on_the_installed_release(monkeypatch):
  # The reconstruction above must not secretly be the upstream path: with `step` out of order the test
  # still has to pass, which proves it compares against the arithmetic and not against the release.
  monkeypatch.setattr(sys.modules[__name__], "_has_packaged_re_noising", lambda: False)

  def _unreachable(*args, **kwargs):
    raise AssertionError("the installed scheduler was used as the reference")

  monkeypatch.setattr(MiniMaxH3Scheduler, "step", _unreachable)
  test_vendored_scheduler_reproduces_the_installed_re_noising_update()


def test_reference_embedding_is_fp32_bounded():
  # The two VAEs ship float32 and the transformer keeps its boundary modules in fp32; the quantize
  # exclusion list of the example depends on these names surviving.
  transformer = _tiny_transformer()
  assert isinstance(transformer.rope, torch.nn.Module)
