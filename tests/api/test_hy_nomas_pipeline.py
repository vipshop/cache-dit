"""Pipeline-level contracts of the vendored HY-Nomas pipeline.

The segment loop has one failure mode that every metric hides: if the denoising steps call
``transformer.predict`` instead of ``transformer(...)``, the cached runtime never patches anything,
the run is bit-identical to a cached run's reference, and the only symptom is zero speedup.  These
tests drive ``_run_segment`` with a stub transformer and assert **which entry point** each step
uses,
that the K/V write happens exactly once and before the commit, and that the last segment writes
nothing.
"""
from types import SimpleNamespace

import pytest
import torch

from cache_dit.models.pipelines.hy_nomas.pipeline import NomasPipeline, NomasPipelineOutput
from cache_dit.models.pipelines.hy_nomas.sampling import NomasSegment, nomas_sigmas


class _StubConfig(SimpleNamespace):
  in_channels = 32
  num_layers = 2
  hidden_size = 32


class _StubTransformer:
  """Records the entry point of every call and returns zeros shaped like the latents."""

  def __init__(self):
    self.config = _StubConfig()
    self.parallel = SimpleNamespace(head_shards=1)
    self.calls = []

  def prepare_segment_context(self, latent, text_mask, cache, **kwargs):
    return ("context", latent.shape[2:])

  def __call__(self, latent, *args, **kwargs):
    self.calls.append("forward")
    return torch.zeros_like(latent)

  def predict(self, latent, *args, **kwargs):
    self.calls.append("predict")
    entries = [{"k": torch.zeros(1), "v": torch.zeros(1)}] * self.config.num_layers
    return torch.zeros_like(latent), entries

  def __call_never(self):  # pragma: no cover - documentation only
    raise AssertionError


class _StubCache:
  def __init__(self):
    self.commits = []

  def commit(self, entries, text_mask, video_tokens, frames, action_start=-1):
    self.commits.append((len(entries), video_tokens, frames, action_start))


def _pipeline():
  pipe = NomasPipeline.__new__(NomasPipeline)
  # DiffusionPipeline.__setattr__ consults `self.config` for existing attributes, and
  # `ConfigMixin.config` reads `_internal_dict`; a real instance gets both from `__init__`.
  pipe.__dict__["_internal_dict"] = {}
  pipe.transformer = _StubTransformer()
  pipe.vae = None
  pipe.text_encoder = None
  pipe.tokenizer = None
  pipe.scheduler = None
  pipe.text_len = 1280
  pipe.segment_latent_frames = 10
  pipe.flow_shift = 5.0
  pipe.cache_noise_sigma = 0.206897
  pipe.max_cache_gib = 32.0
  pipe._warmup_capped = False
  return pipe


def _condition(text_len=6, frames=2):
  return {
    "text_states": torch.zeros(1, 6, 16),
    "text_mask": (torch.arange(text_len)[None] < text_len - 1),
    "text_offsets": None,
    "video_windows": None,
    "action_start": 2,
  }


def test_denoise_uses_forward_and_kv_write_uses_predict():
  pipe = _pipeline()
  cache = _StubCache()
  segment = NomasSegment(0, 2, "caption")
  pipe._refresh_segment_cache = lambda: None
  pipe._run_segment(segment, _condition(), (2, 2), nomas_sigmas(8), cache, None, None,
                    torch.device("cpu"), torch.float32, cache_segment=True)
  # Eight Euler steps must go through `__call__` (the patched `forward`), then one K/V pass.
  assert pipe.transformer.calls.count("forward") == 8
  assert pipe.transformer.calls.count("predict") == 1
  assert pipe.transformer.calls[-1] == "predict"
  # frames * height * width of the latent grid = 2 * 2 * 2
  assert cache.commits == [(2, 8, 2, 2)]


def test_last_segment_skips_the_kv_write():
  pipe = _pipeline()
  cache = _StubCache()
  pipe._refresh_segment_cache = lambda: None
  pipe._run_segment(NomasSegment(0, 2, "caption"), _condition(), (2, 2), nomas_sigmas(4), cache,
                    None, None, torch.device("cpu"), torch.float32, cache_segment=False)
  assert "predict" not in pipe.transformer.calls
  assert cache.commits == []


def test_refresh_segment_cache_arms_the_hint_and_caps_warmup():
  pipe = _pipeline()

  class _Context:
    def __init__(self):
      self.cache_config = SimpleNamespace(max_warmup_steps=8, force_refresh_step_hint=None,
                                          force_refresh_step_policy="once")

    def get_current_transformer_step(self):
      return 3

  context = _Context()
  pipe._context_manager = SimpleNamespace(get_context=lambda: context)
  pipe._refresh_segment_cache()
  assert context.cache_config.force_refresh_step_hint == 4
  assert context.cache_config.max_warmup_steps == 1
  assert pipe._warmup_capped is True
  # Without a cache runtime the call is a no-op instead of an error.
  pipe._context_manager = None
  pipe._refresh_segment_cache()


def test_guidance_scale_must_be_one():
  pipe = _pipeline()
  with pytest.raises(ValueError, match="guidance_scale"):
    pipe(prompt="{}", guidance_scale=2.0)


def test_from_pretrained_rejects_device_map_and_drops_quantization_config():
  with pytest.raises(NotImplementedError, match="device_map"):
    NomasPipeline.from_pretrained("/nonexistent", device_map="balanced")
  # `quantization_config` is dropped with a warning; the missing weights then fail loudly.
  with pytest.raises(FileNotFoundError):
    NomasPipeline.from_pretrained("/nonexistent", quantization_config=object())


def test_plan_canonicalises_json_captions():
  pipe = _pipeline()
  caption = '{"segments": [{"visual_description_long": "x"}]}'
  plan = pipe._plan(caption, 40, None, None, "json")
  assert len(plan) == 1 and plan[0].start == 0 and plan[0].end == 10
  plain = pipe._plan("free text", 40, None, None, "plain")
  assert plain[0].caption == "free text"


def test_output_dataclass_carries_frames():
  tensor = torch.zeros(1, 3, 4, 8, 8)
  assert torch.equal(NomasPipelineOutput(frames=tensor).frames, tensor)
