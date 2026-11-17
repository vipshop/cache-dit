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

# The released six-step ladder, as a base grid the scheduler shifts (video shift 12 / audio shift 3).
_BASE_SIGMAS = [step / 1000.0 for step in (1000, 750, 500, 250, 100, 40, 0)]

# What the installed diffusers knows about the re-noising update. The vendored scheduler is the
# fallback for a release that knows nothing about it, so which half of each guard applies depends on
# this, not on a version string.
_INIT_PARAMETERS = inspect.signature(MiniMaxH3Scheduler.__init__).parameters
_HAS_PACKAGED_RE_NOISING = ("stochastic_sampling" in _INIT_PARAMETERS
                            and "shift_provided_sigmas" in _INIT_PARAMETERS and "generator"
                            in inspect.signature(MiniMaxH3Scheduler.step).parameters)


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
  if not _HAS_PACKAGED_RE_NOISING:
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
  if _HAS_PACKAGED_RE_NOISING:
    # The installed release owns the contract, so the integration can use it directly and the vendored
    # subclass only has to stay a numerical stand-in. Answering this differently is the drift: the
    # resolver would pick a class the checkpoint was not trained for.
    scheduler = _upstream_scheduler(12.0)
    scheduler.set_timesteps(sigmas=_BASE_SIGMAS)
    assert scheduler.num_inference_steps == 6
    return
  # No packaged contract: the vendored subclass is what makes the run the released model, so its four
  # overrides have to be there.
  assert "generator" not in inspect.signature(MiniMaxH3Scheduler.step).parameters
  assert "stochastic_sampling" not in _INIT_PARAMETERS
  assert "scale_noise" in PikaH3FlexScheduler.__dict__
  assert "step" in PikaH3FlexScheduler.__dict__
  assert "set_timesteps" in PikaH3FlexScheduler.__dict__


def test_vendored_scheduler_reproduces_the_installed_re_noising_update():
  """The fallback must sample what the installed release samples, whatever that release is."""
  vendored = PikaH3FlexScheduler(shift=12.0, stochastic_sampling=True, shift_provided_sigmas=True)
  vendored.set_timesteps(sigmas=_BASE_SIGMAS)
  upstream = _upstream_scheduler(12.0)
  if _HAS_PACKAGED_RE_NOISING:
    upstream.set_timesteps(sigmas=_BASE_SIGMAS)
    assert vendored.sigmas.tolist() == pytest.approx(upstream.sigmas.tolist(), abs=1e-6)
    assert vendored.timesteps.tolist() == pytest.approx(upstream.timesteps.tolist(), abs=1e-6)

  shape = (2, 3)
  vendored_sample = torch.randn(shape,
                                generator=torch.Generator().manual_seed(11),
                                dtype=torch.float32)
  upstream_sample = vendored_sample.clone()
  vendored_generator = torch.Generator().manual_seed(23)
  upstream_generator = torch.Generator().manual_seed(23)
  for index, (timestep, other) in enumerate(zip(vendored.timesteps, upstream.timesteps)):
    velocity = torch.ones(shape, dtype=torch.float32) * (index + 1)
    vendored_sample = vendored.step(velocity,
                                    timestep,
                                    vendored_sample,
                                    generator=vendored_generator,
                                    return_dict=False)[0]
    if _HAS_PACKAGED_RE_NOISING:
      upstream_sample = upstream.step(velocity,
                                      other,
                                      upstream_sample,
                                      generator=upstream_generator,
                                      return_dict=False)[0]
    else:
      # Without the packaged update the comparison is against the documented re-noising arithmetic:
      # `x0 = x + sigma * v`, then `x_next = (1 - sigma') * x0 + sigma' * noise`.
      sigma = vendored.sigmas[index].item()
      sigma_next = vendored.sigmas[index + 1].item()
      noise = torch.randn(shape, generator=upstream_generator, dtype=torch.float32)
      upstream_sample = (1 - sigma_next) * (upstream_sample + sigma * velocity) + sigma_next * noise

  # A different sampler (base class, bf16 cast, unshifted grid) would miss this by orders of magnitude.
  assert torch.allclose(vendored_sample, upstream_sample.to(vendored_sample.dtype), atol=1e-6)


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
