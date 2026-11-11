"""Regression tests for the vendored Pika H3 Flex pipeline (no weights, no GPU).

Pika H3 Flex is the post-trained six-step MiniMax-H3, and the parts that decide whether a run is the
released model at all are all pure arithmetic:

1. **The re-noising update.** The installed `MiniMaxH3Scheduler` has no `stochastic_sampling`, and
   `ConfigMixin.extract_init_dict` drops config keys the signature does not name — a plain subclass
   would sample a deterministic Euler ladder, which is a different model, without raising.
2. **The packed row layout.** `[text | conditioning | audio | video]` with global row indices and a
   fp64 `(t, h, w)` grid. An off-by-one here does not crash; it conditions on the wrong rows.
3. **The per-row timestep plan.** One forward serves generated video, generated audio and the pinned
   conditioning rows at three different noise levels.
4. **The six-step ladder.** There is no `num_inference_steps`; the schedule is the adapter's.
"""
import argparse
import os
from types import SimpleNamespace

import pytest
import torch

from cache_dit.cli.examples import (
  _PIKA_H3_FLEX_CACHE_THRESHOLDS,
  _PIKA_H3_FLEX_STEPS,
  _pika_h3_flex_adapter_name,
  _pika_h3_flex_cache_config,
  _pika_h3_flex_init_hook,
)
from cache_dit.models.pipelines.pika_h3_flex import (
  PikaH3FlexBlocks,
  PikaH3FlexModularPipeline,
  PikaH3FlexScheduler,
)
from cache_dit.models.pipelines.pika_h3_flex.before_denoise import (
  PikaH3FlexPrepareLayoutStep,
  PikaH3FlexSetTimestepsStep,
)

_BASE_SIGMAS = [step / 1000.0 for step in (1000, 750, 500, 250, 100, 40, 0)]


def _rollout_reference(sigmas, num_steps, seed, shape=(2, 3)):
  """Hand-rolled re-noising rollout: `x0 = x + sigma * v`, then re-noise onto the next sigma."""
  generator = torch.Generator().manual_seed(seed)
  sample = torch.randn(shape, generator=generator, dtype=torch.float32)
  for index in range(num_steps):
    velocity = torch.ones(shape) * (index + 1)
    sigma, sigma_next = sigmas[index], sigmas[index + 1]
    denoised = sample + sigma * velocity
    noise = torch.randn(shape, generator=generator, dtype=torch.float32)
    sample = (1 - sigma_next) * denoised + sigma_next * noise
  return sample


def test_scheduler_keeps_the_re_noising_config_keys():
  scheduler = PikaH3FlexScheduler(
    shift=12.0,
    stochastic_sampling=True,
    shift_provided_sigmas=True,
  )
  assert scheduler.config["stochastic_sampling"] is True
  assert scheduler.config["shift_provided_sigmas"] is True


def test_scheduler_six_step_ladder_per_shift():
  expected = {
    12.0: [1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0],
    3.0: [1.0, 0.9, 0.75, 0.5, 0.25, 0.111111, 0.0],
  }
  for shift, sigmas in expected.items():
    scheduler = PikaH3FlexScheduler(
      shift=shift,
      stochastic_sampling=True,
      shift_provided_sigmas=True,
    )
    scheduler.set_timesteps(sigmas=_BASE_SIGMAS)
    assert scheduler.sigmas.tolist() == pytest.approx(sigmas, abs=1e-6)
    # The terminal sigma has no model evaluation: six forwards, not seven.
    assert scheduler.num_inference_steps == _PIKA_H3_FLEX_STEPS
    assert scheduler.timesteps.numel() == _PIKA_H3_FLEX_STEPS


def test_scheduler_step_matches_the_documented_re_noising_update():
  scheduler = PikaH3FlexScheduler(
    shift=12.0,
    stochastic_sampling=True,
    shift_provided_sigmas=True,
  )
  scheduler.set_timesteps(sigmas=_BASE_SIGMAS)
  generator = torch.Generator().manual_seed(7)
  sample = torch.randn(2, 3, generator=generator, dtype=torch.float32)
  for index, timestep in enumerate(scheduler.timesteps):
    sample = scheduler.step(
      torch.ones(2, 3) * (index + 1),
      timestep,
      sample,
      generator=generator,
      return_dict=False,
    )[0]
  reference = _rollout_reference(scheduler.sigmas.tolist(), len(scheduler.timesteps), 7)
  assert torch.equal(sample, reference)


def test_packed_layout_row_counts_and_indices():
  num_text, num_latent_frames, num_audio_latents = 7, 7, 200
  latent_height, latent_width = 28, 48
  rows_per_frame = (latent_height // 2) * (latent_width // 2)

  position_ids, token_tags, video_indices, audio_indices, text_indices, cond_video, cond_audio = (
    PikaH3FlexPrepareLayoutStep.build_packed_sequence(
      text_token_tags=torch.ones(num_text, dtype=torch.long),
      num_latent_frames=num_latent_frames,
      latent_height=latent_height,
      latent_width=latent_width,
      num_audio_latents=num_audio_latents,
      patch_size=(1, 2, 2),
      audio_channels=2,
      audio_tag=2,
      video_tag=0,
      keyframe_anchors=("first", ),
    ))

  sequence_length = num_text + rows_per_frame + num_audio_latents * 2 + (num_latent_frames *
                                                                         rows_per_frame)
  assert position_ids.shape == (sequence_length, 3)
  assert token_tags.shape == (sequence_length, )
  assert video_indices.numel() + audio_indices.numel() + text_indices.numel() == sequence_length
  assert cond_video == rows_per_frame
  assert cond_audio == 0
  # Text sits on the time axis at its row index, so the media clock starts where the text ends.
  assert position_ids[:num_text, 0].tolist() == [float(i) for i in range(num_text)]
  # Conditioning rows are pinned on the first latent frame; the generated video rows follow.
  assert position_ids[num_text:num_text + rows_per_frame, 0].unique().tolist() == [float(num_text)]
  assert token_tags[audio_indices].unique().tolist() == [2]
  assert token_tags[video_indices].unique().tolist() == [0]


def test_scheduler_contract_rejects_three_kinds_of_degradation():
  from diffusers import MiniMaxH3Scheduler

  from cache_dit.models.pipelines.pika_h3_flex.modular_pipeline import assert_scheduler_contract

  good = PikaH3FlexScheduler(shift=12.0, stochastic_sampling=True, shift_provided_sigmas=True)
  assert_scheduler_contract(good, "scheduler")

  # 1. the installed base class, even with the hidden config keys set to True.
  hidden = dict(MiniMaxH3Scheduler(shift=12.0).config)
  hidden.update(stochastic_sampling=True, shift_provided_sigmas=True)
  degraded = MiniMaxH3Scheduler.from_config(hidden)
  assert degraded.config["stochastic_sampling"] is True  # the config key proves nothing
  with pytest.raises(ValueError, match="not `PikaH3FlexScheduler`"):
    assert_scheduler_contract(degraded, "scheduler")

  # 2. a ladder trained at another shift.
  with pytest.raises(ValueError, match="shift=6"):
    assert_scheduler_contract(
      PikaH3FlexScheduler(shift=6.0, stochastic_sampling=True, shift_provided_sigmas=True),
      "scheduler",
    )

  # 3. the right class that still takes the grid verbatim (no shift, no re-noise).
  with pytest.raises(ValueError, match="verbatim"):
    assert_scheduler_contract(
      PikaH3FlexScheduler(shift=12.0, stochastic_sampling=True, shift_provided_sigmas=False),
      "scheduler",
    )


def test_reference_flags_decode_into_this_models_classes(tmp_path):
  # The vendored blocks reject a foreign reference class with an `isinstance` check that runs *after*
  # the 200 GiB checkpoint is resident, so the flags must build `PikaH3Flex*Reference` objects.
  from cache_dit.cli.examples import (
    _PIKA_H3_FLEX_REFERENCE_CLASSES,
    _PIKA_H3_FLEX_REFERENCE_PACKAGE,
  )
  from cache_dit.cli.registers import _references_from_args
  from cache_dit.models.pipelines.pika_h3_flex import PikaH3FlexReference
  from PIL import Image

  image_path = tmp_path / "reference.png"
  Image.new("RGB", (64, 48), (200, 30, 30)).save(image_path)
  args = SimpleNamespace(
    reference_image_path=[str(image_path)],
    reference_video_path=None,
    reference_audio_path=None,
  )

  references = _references_from_args(
    args,
    _PIKA_H3_FLEX_REFERENCE_PACKAGE,
    _PIKA_H3_FLEX_REFERENCE_CLASSES,
  )
  assert len(references) == 1
  assert isinstance(references[0], PikaH3FlexReference)
  assert references[0].kind == "image"

  # Before the fix the flags built the MiniMax-H3 class, which the `ref2va` validation rejects.
  default_references = _references_from_args(args)
  assert not isinstance(default_references[0], PikaH3FlexReference)


def test_row_timestep_plan_groups_the_three_noise_levels():
  video_indices = torch.tensor([2, 3, 4, 5])
  audio_indices = torch.tensor([6, 7])
  timesteps, inverse = PikaH3FlexSetTimestepsStep.build_row_timesteps(
    video_indices,
    audio_indices,
    num_condition_video_rows=1,
    num_condition_audio_rows=0,
    num_text_tokens=2,
    video_timestep=0.9,
    audio_timestep=0.5,
    condition_video_timestep=0.999,
    condition_audio_timestep=1.0,
  )
  # Sorted unique levels: audio 0.5, video 0.9, pinned conditioning 0.999.
  assert timesteps.tolist() == pytest.approx([0.5, 0.9, 0.999], abs=1e-6)
  # Rows: 2 text, 4 video (the first one the pinned conditioning row), 2 audio. Text rows never reach
  # an output head and inherit the video timestep.
  assert inverse.tolist() == [1, 1, 2, 1, 1, 1, 0, 0]


def test_workflow_prunes_once_and_refuses_a_second_prune():
  # `from_pretrained(workflow=...)` prunes inside `__init__`; cache-dit's CLI instead passes the
  # workflow to `load_components`. Passing it to both prunes a `SequentialPipelineBlocks`, which no
  # longer declares `_workflow_map` — that is the NotImplementedError pinned here.
  pipe = PikaH3FlexModularPipeline(blocks=PikaH3FlexBlocks(), workflow="ref2va")
  assert "transformer_ref" in {spec.name for spec in pipe.blocks.expected_components}
  assert "transformer" not in {spec.name for spec in pipe.blocks.expected_components}

  pruned = PikaH3FlexBlocks().get_workflow("ref2va")
  with pytest.raises(NotImplementedError, match="_workflow_map"):
    pruned.get_workflow("ref2va")


def test_pipeline_exposes_the_released_geometry():
  pipeline = PikaH3FlexModularPipeline(blocks=PikaH3FlexBlocks())
  assert pipeline.patch_size == (1, 2, 2)
  assert pipeline.canvas_multiple == 32
  assert pipeline.fps == 24
  assert pipeline.audio_channels == 2
  assert pipeline.audio_sampling_rate == 32000
  # The conditioner's 50th hidden state, not the final post-norm one.
  assert pipeline.text_encoder_layer == 50


def _cache_args(**overrides) -> argparse.Namespace:
  # The CLI defaults: Fn 1 / Bn 0 / mcc 3 / ws 8. The preset must pass Fn and Bn through untouched and
  # clamp only the warmup and the consecutive-cache run.
  values = dict(
    cache=True,
    residual_diff_threshold=None,
    max_warmup_steps=8,
    max_cached_steps=-1,
    max_continuous_cached_steps=3,
    Fn_compute_blocks=1,
    Bn_compute_blocks=0,
  )
  values.update(overrides)
  return SimpleNamespace(**values)


def test_cache_config_is_a_six_step_single_forward_ladder():
  for workflow in _PIKA_H3_FLEX_CACHE_THRESHOLDS:
    config = _pika_h3_flex_cache_config(_cache_args(max_warmup_steps=8), workflow)
    assert config is not None
    # `num_inference_steps` also arms the per-request refresh: the CLI-built config leaves it None.
    assert config.num_inference_steps == _PIKA_H3_FLEX_STEPS
    assert config.enable_separate_cfg is False
    assert config.Bn_compute_blocks == 0
    # The CLI default warmup (8) would eat the whole ladder and report PSNR=inf; clamp to one step,
    # which leaves the `compute, cache, compute, cache, ...` ladder reachable (3 of 6 forwards).
    assert config.max_warmup_steps == 1
    assert config.max_continuous_cached_steps == 1


def test_cache_config_passes_the_block_counts_through():
  config = _pika_h3_flex_cache_config(_cache_args(Fn_compute_blocks=2, Bn_compute_blocks=4), "t2va")
  # Hardcoding these would make the threshold sweep silently ignore `--Fn` / `--Bn`.
  assert config.Fn_compute_blocks == 2
  assert config.Bn_compute_blocks == 4


def test_cache_config_warns_but_does_not_raise_on_a_full_ladder_warmup(caplog):
  with caplog.at_level("WARNING"):
    config = _pika_h3_flex_cache_config(_cache_args(max_warmup_steps=6), "t2va")
  assert config.max_warmup_steps == 1
  assert any("never skip a step" in record.message % record.args for record in caplog.records
             if record.levelname == "WARNING")


def test_cache_config_honours_an_explicit_threshold_and_can_be_disabled():
  config = _pika_h3_flex_cache_config(
    _cache_args(residual_diff_threshold=0.07),
    "t2va",
  )
  assert config.residual_diff_threshold == pytest.approx(0.07)
  assert _pika_h3_flex_cache_config(_cache_args(cache=False), "t2va") is None


class _TrunklessPipe:
  """Pipeline stub whose `load_lora_weights` does nothing, like a mis-routed adapter."""

  _component_specs = {"transformer": None}

  def __init__(self):
    self.transformer = SimpleNamespace(modules=lambda: iter(()))

  def load_lora_weights(self, *args, **kwargs):
    return None


def test_init_hook_rejects_an_adapter_that_misses_the_trunk():
  # `get_pipe` returns from the modular branch before the shared LoRA block, so the hook is the only
  # thing that loads the adapter; a silent no-op there means the run samples the bare MiniMax-H3 trunk.
  hook = _pika_h3_flex_init_hook("/tmp/model", "fl2va/adapter.safetensors", "fl2va")
  with pytest.raises(RuntimeError, match="did not land"):
    hook(_TrunklessPipe())


def test_init_hook_requires_the_partition_of_its_workflow():
  hook = _pika_h3_flex_init_hook("/tmp/model", "ref2va/adapter.safetensors", "ref2va")
  with pytest.raises(ValueError, match="transformer_ref"):
    hook(_TrunklessPipe())


def test_adapter_guard_fails_fast_instead_of_sampling_the_bare_trunk(tmp_path):
  with pytest.raises(FileNotFoundError, match="adapter"):
    _pika_h3_flex_adapter_name(str(tmp_path), "fl2va")

  (tmp_path / "fl2va").mkdir()
  (tmp_path / "fl2va" / "adapter.safetensors").write_bytes(b"")
  (tmp_path / "ref2va").mkdir()
  (tmp_path / "ref2va" / "adapter.safetensors").write_bytes(b"")

  # `t2va` and `fl2va` share the `fl2va` partition, `ref2va` drives `transformer_ref/`.
  assert _pika_h3_flex_adapter_name(str(tmp_path), "t2va") == "fl2va/adapter.safetensors"
  assert _pika_h3_flex_adapter_name(str(tmp_path), "fl2va") == "fl2va/adapter.safetensors"
  assert _pika_h3_flex_adapter_name(str(tmp_path), "ref2va") == "ref2va/adapter.safetensors"
  assert os.path.exists(tmp_path / "fl2va" / "adapter.safetensors")
