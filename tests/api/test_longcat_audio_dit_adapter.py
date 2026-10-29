"""Regression tests for the LongCat-AudioDiT cache integration.

LongCat-AudioDiT is an audio-only diffusers pipeline: `AudioDiTBlock.forward` keeps
`hidden_states` first and returns a single tensor, while everything else it consumes
(`timestep_embed / cond / mask / cond_mask / rope / cond_rope / adaln_global_out`) is computed once
per step before the block loop. These CPU tests lock down the properties the integration relies on:

1. The adapter is registered under `"LongCatAudioDiT"` with `ForwardPattern.Pattern_3` and points at
   `transformer.blocks` (not `transformer_blocks`).
2. `has_separate_cfg` follows the pipeline's reachable `guidance_scale` default, because the
   pipeline runs the null forward only when guidance is on.
3. Collapsing the block list into the cached wrapper — which replays blocks with keyword arguments —
   is bit-exact against the unpatched transformer when no step is served from cache.

The CP rope helpers and the audio writer are covered here as well, since both are pure CPU logic.
"""
import argparse
import types

import numpy as np
import pytest
import torch

import cache_dit
from cache_dit import DBCacheConfig
from cache_dit.caching.block_adapters import BlockAdapterRegister, longcat_audio_dit_adapter
from cache_dit.caching.forward_pattern import ForwardPattern

diffusers = pytest.importorskip("diffusers")

if not hasattr(diffusers, "LongCatAudioDiTTransformer"):
  pytest.skip("LongCat-AudioDiT requires a newer diffusers", allow_module_level=True)

_LATENT_DIM = 4
_DIM = 32
_TEXT_DIM = 16
_TEXT_LEN = 5
_FRAMES = 7
_STEPS = 4


def _tiny_transformer():
  from diffusers import LongCatAudioDiTTransformer

  torch.manual_seed(0)
  transformer = LongCatAudioDiTTransformer(
    dit_dim=_DIM,
    dit_depth=3,
    dit_heads=2,
    dit_text_dim=_TEXT_DIM,
    latent_dim=_LATENT_DIM,
    ff_mult=2.0,
    qk_norm=True,
  ).eval()
  # `_initialize_weights` zeroes the output projections (they learn a zero-initialized delta), which
  # would make every cache comparison trivially exact.
  with torch.no_grad():
    transformer.proj_out.weight.normal_()
    transformer.norm_out.linear.weight.normal_()
    transformer.adaln_global_mlp.mlp[-1].weight.normal_()
  return transformer


def _inputs(seed: int = 0):
  generator = torch.Generator().manual_seed(seed)
  return dict(
    hidden_states=torch.randn(1, _FRAMES, _LATENT_DIM, generator=generator),
    encoder_hidden_states=torch.randn(1, _TEXT_LEN, _TEXT_DIM, generator=generator),
    encoder_attention_mask=torch.ones(1, _TEXT_LEN, dtype=torch.bool),
    timestep=torch.tensor([0.5]),
    attention_mask=torch.ones(1, _FRAMES, dtype=torch.bool),
    latent_cond=torch.zeros(1, _FRAMES, _LATENT_DIM),
  )


class _LongCatAudioLikePipe:
  """Minimal stand-in for `LongCatAudioDiTPipeline`.

  The real pipeline runs the conditional forward on every step and the null forward only when
  `guidance_scale > 1`, i.e. two transformer calls per step under the default configuration.
  """

  def __init__(self, transformer):
    self.transformer = transformer

  def __call__(self, steps, guidance_scale: float = 4.0):
    outputs = []
    for raw, text in steps:
      outputs.append(self.transformer(**raw).sample)
      if guidance_scale > 1.0:
        null_inputs = dict(raw)
        null_inputs["encoder_hidden_states"] = torch.zeros_like(text)
        null_inputs["encoder_attention_mask"] = torch.ones_like(raw["encoder_attention_mask"])
        outputs.append(self.transformer(**null_inputs).sample)
    return outputs


def test_adapter_is_registered_for_the_audio_dit():
  transformer = _tiny_transformer()
  pipe = types.SimpleNamespace(transformer=transformer)
  adapter = longcat_audio_dit_adapter(pipe)

  assert adapter.blocks is transformer.blocks
  assert adapter.forward_pattern is ForwardPattern.Pattern_3
  assert adapter.check_forward_pattern is True
  # The registry matches on class-name prefixes, so this model must resolve to the audio adapter
  # instead of the `LongCatImage` entry.
  resolved = BlockAdapterRegister.get_adapter(transformer)
  assert resolved.blocks is transformer.blocks


@pytest.mark.parametrize(("guidance_scale", "expected"), [(4.0, True), (1.0, False)])
def test_has_separate_cfg_follows_the_call_default(guidance_scale, expected):
  pipe = _LongCatAudioLikePipe(_tiny_transformer())

  def __call__(steps, guidance_scale: float = guidance_scale):
    return None

  pipe.__call__ = __call__
  assert longcat_audio_dit_adapter(pipe).has_separate_cfg is expected


def test_has_separate_cfg_defaults_to_guidance_enabled_without_a_call_default():
  pipe = types.SimpleNamespace(transformer=_tiny_transformer())
  assert longcat_audio_dit_adapter(pipe).has_separate_cfg is True


def test_cached_blocks_are_bit_exact_when_no_step_is_served_from_cache():
  """The collapsed keyword-called loop must reproduce the unpatched transformer output."""
  inputs = _inputs()
  steps = [(inputs, inputs["encoder_hidden_states"]) for _ in range(_STEPS)]

  reference = _tiny_transformer()
  with torch.no_grad():
    expected = _LongCatAudioLikePipe(reference)(steps)
  # Guard against the zero-initialized output layers sneaking back in, which would make every
  # comparison below trivially exact.
  assert expected[0].abs().max() > 0

  transformer = _tiny_transformer()
  pipe = _LongCatAudioLikePipe(transformer)
  adapter = longcat_audio_dit_adapter(pipe)
  cache_dit.enable_cache(
    adapter,
    cache_config=DBCacheConfig(
      Fn_compute_blocks=1,
      Bn_compute_blocks=0,
      max_warmup_steps=0,
      max_cached_steps=-1,
      # A zero threshold never reuses a step, so any dropped or reordered block kwarg shows up as a
      # numerical difference instead of being masked by a cache hit.
      residual_diff_threshold=0.0,
      num_inference_steps=_STEPS,
    ),
  )
  with torch.no_grad():
    actual = pipe(steps)

  assert len(actual) == len(expected)
  for cached, ref in zip(actual, expected):
    assert torch.equal(cached, ref)


def test_maskless_forward_is_a_no_op_for_an_all_true_mask():
  from cache_dit.distributed.transformers.longcat_audio_dit import _wrap_maskless_forward

  inputs = _inputs()
  reference = _tiny_transformer()
  with torch.no_grad():
    expected = reference(**inputs).sample

  transformer = _tiny_transformer()
  transformer.load_state_dict(reference.state_dict())
  _wrap_maskless_forward(transformer)
  with torch.no_grad():
    actual = transformer(**inputs).sample
  assert torch.equal(actual, expected)


def test_maskless_forward_rejects_a_padded_mask():
  from cache_dit.distributed.transformers.longcat_audio_dit import _wrap_maskless_forward

  transformer = _tiny_transformer()
  _wrap_maskless_forward(transformer)
  inputs = _inputs()
  inputs["attention_mask"] = torch.zeros_like(inputs["attention_mask"])
  with pytest.raises(NotImplementedError):
    transformer(**inputs)


def test_cp_rope_processors_preserve_the_unpatched_output():
  """With CP off, the full rotary table sliced at offset 0 must match the original embedding."""
  from cache_dit.distributed.transformers.longcat_audio_dit import _install_cp_rope

  inputs = _inputs()
  reference = _tiny_transformer()
  with torch.no_grad():
    expected = reference(**inputs).sample

  transformer = _tiny_transformer()
  transformer.load_state_dict(reference.state_dict())
  _install_cp_rope(transformer)
  with torch.no_grad():
    actual = transformer(**inputs).sample
  assert torch.equal(actual, expected)


def test_audio_output_is_written_as_a_wav(tmp_path):
  from cache_dit._utils.registers import ExampleOutputData

  sf = pytest.importorskip("soundfile")
  samples = 480
  waveform = torch.stack([torch.linspace(-0.5, 0.5, samples), torch.linspace(0.5, -0.5, samples)])
  output = ExampleOutputData(
    model_tag="longcat_audio_dit",
    strify_tag="unit",
    audio=waveform,
    audio_sampling_rate=24000,
  )
  save_path = str(tmp_path / "out.wav")
  output.save(argparse.Namespace(save_path=save_path, saved_fps=24))

  data, rate = sf.read(save_path)
  assert rate == 24000
  assert data.shape == (samples, 2)
  # `(channels, samples)` must be transposed, not written as interleaved garbage.
  np.testing.assert_allclose(data[:, 0], waveform[0].numpy(), atol=1e-3)


@pytest.mark.parametrize("as_tensor", [False, True])
def test_parse_audio_output_reads_one_request(as_tensor):
  from cache_dit._utils.registers import ExampleType, parse_audio_output

  audios = np.zeros((2, 1, 32), dtype=np.float32)
  audios[0, 0, :4] = 1.0
  if as_tensor:
    audios = torch.from_numpy(audios)
  pipe = types.SimpleNamespace(sample_rate=24000)
  parsed = parse_audio_output(types.SimpleNamespace(audios=audios), ExampleType.T2A, pipe=pipe)

  audio = parsed["audio"]
  assert tuple(audio.shape) == (1, 32)
  assert parsed["audio_sampling_rate"] == 24000
  assert float(np.asarray(audio).max()) == 1.0


def test_parse_audio_output_rejects_an_output_without_audios():
  from cache_dit._utils.registers import ExampleType, parse_audio_output

  with pytest.raises(ValueError):
    parse_audio_output(object(), ExampleType.T2A, pipe=types.SimpleNamespace(sample_rate=24000))
