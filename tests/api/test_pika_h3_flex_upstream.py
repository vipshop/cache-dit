"""Guards for the official-vs-vendored choice of the Pika H3 Flex pipeline.

The released model is `diffusers.PikaH3FlexModularPipeline`, and cache-dit keeps a copy of the pipeline
layer for a diffusers that predates it. The class names are identical, so nothing may branch on a name:
`resolve_pika_h3_flex()` decides on importability, and everything downstream — the CLI example, the
reference media builder, cache-dit's own `enable_cache` — consumes that one binding. These tests pin
that chain, plus the two facts it rests on:

1. the `PikaH3Flex` adapter prefix and the modular-pipeline allowlist already cover the official class;
2. `enable_cache` keeps `transformer.forward`'s signature, which the modular denoiser step introspects to
   build its kwargs (`inspect.signature(transformer.forward).parameters`).

Nothing here needs weights or a GPU: the transformer is a 2-block 16-wide build and the blocks are the
real ones.
"""
import inspect

import pytest
import torch

import cache_dit
import diffusers
from cache_dit.caching.block_adapters.block_registers import BlockAdapterRegister
from cache_dit.models.pipelines import pika_h3_flex as pika_h3_flex_models
from cache_dit.models.pipelines.pika_h3_flex import (
  PikaH3FlexBinding,
  PikaH3FlexBlocks,
  PikaH3FlexModularPipeline,
  resolve_pika_h3_flex,
  upstream_pika_h3_flex_available,
)

_REFERENCE_CLASS_NAMES = (
  "PikaH3FlexAudioReference",
  "PikaH3FlexImageReference",
  "PikaH3FlexReference",
  "PikaH3FlexVideoReference",
)


def _tiny_transformer() -> torch.nn.Module:
  """A transformer with the checkpoint's module tree at 1/128 the width."""
  from diffusers import MiniMaxH3Transformer3DModel

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


def test_the_installed_diffusers_ships_the_official_pipeline():
  # The premise of the whole lane. If this flips, the vendored copy is the only runnable one and every
  # number in the acceptance table belongs to the fallback instead of the released model.
  if not upstream_pika_h3_flex_available():
    pytest.skip("the installed diffusers predates modular_pipelines/pika_h3_flex")
  assert diffusers.PikaH3FlexModularPipeline.__module__ == (
    "diffusers.modular_pipelines.pika_h3_flex.modular_pipeline")
  assert diffusers.PikaH3FlexBlocks.__module__ == (
    "diffusers.modular_pipelines.pika_h3_flex.modular_blocks_pika_h3_flex")


def test_resolver_prefers_the_official_pipeline():
  if not upstream_pika_h3_flex_available():
    pytest.skip("the installed diffusers predates modular_pipelines/pika_h3_flex")
  binding = resolve_pika_h3_flex()
  assert isinstance(binding, PikaH3FlexBinding)
  assert binding.is_upstream is True
  assert binding.pipeline_class is diffusers.PikaH3FlexModularPipeline
  # The official blocks are resolved by name out of the `diffusers` namespace
  # (`ModularPipeline._blocks_class_name`), so the factory has to stay unset.
  assert binding.blocks_factory is None
  assert binding.reference_package == "diffusers.modular_pipelines.pika_h3_flex.references"


def test_resolver_falls_back_to_the_vendored_copy(monkeypatch):
  # The fallback is the `diffusers < 0.43` path. It is selected by importability, never by class name, so
  # this is the only way to reach it on a release that does ship the official pipeline.
  monkeypatch.setattr(pika_h3_flex_models, "upstream_pika_h3_flex_available", lambda: False)
  binding = resolve_pika_h3_flex()
  assert binding.is_upstream is False
  assert binding.pipeline_class is PikaH3FlexModularPipeline
  # The vendored block tree is passed in explicitly: it is not importable from `diffusers`, so a
  # `from_pretrained` blocks lookup would fail.
  assert binding.blocks_factory is PikaH3FlexBlocks
  assert binding.reference_package == "cache_dit.models.pipelines.pika_h3_flex"

  # Pipeline and blocks are a pair: mixing an official pipeline with the vendored blocks trips the
  # vendored `assert_scheduler_contract`, which is why the binding carries both.
  blocks = binding.blocks_factory()
  assert type(blocks).__name__ == "PikaH3FlexBlocks"


def test_reference_package_exposes_the_types_the_cli_builds():
  package = resolve_pika_h3_flex().reference_package
  module = __import__(package, fromlist=list(_REFERENCE_CLASS_NAMES))
  for name in _REFERENCE_CLASS_NAMES:
    assert hasattr(module, name), f"`{name}` disappeared from {package}"
  assert hasattr(module.PikaH3FlexImageReference, "from_file")


def test_adapter_register_covers_the_modular_pipeline_name():
  # `PikaH3Flex` is a prefix match against the *pipeline* class name, and the allowlist only decides
  # whether cache-dit warns. Both have to cover the official class, whose name is the vendored one.
  assert BlockAdapterRegister.is_supported("PikaH3FlexModularPipeline")
  name, factory = BlockAdapterRegister._match_adapter("PikaH3FlexModularPipeline")
  assert name == "PikaH3Flex"
  assert factory is not None
  assert "PikaH3Flex" in BlockAdapterRegister._verified_modular_pipelines


def test_cli_examples_follow_the_binding():
  from cache_dit.cli.registers import ExampleRegister
  from cache_dit.cli.utils import get_base_args, maybe_postprocess_args

  resolve_names = ("pika_h3_flex", "pika_h3_flex_fl2va", "pika_h3_flex_ref2va")
  vendored_names = ("pika_h3_flex_vendored", "pika_h3_flex_fl2va_vendored",
                    "pika_h3_flex_ref2va_vendored")
  registered = ExampleRegister.list_examples()
  for name in resolve_names + vendored_names:
    assert name in registered, f"`{name}` is not registered any more"
  # The vendored entry points must not share a name with the resolver ones: on a release that ships the
  # official pipeline the two would otherwise be the same entry point, and the fallback would be
  # unreachable (and untestable).
  assert not set(resolve_names) & set(vendored_names)

  # `parse=True` would read `sys.argv`, i.e. this test's own pytest flags.
  args = maybe_postprocess_args(get_base_args(parse=False).parse_args([]))
  cases = [(name, True) for name in resolve_names] + [(name, False) for name in vendored_names]
  for name, upstream_expected in cases:
    example = ExampleRegister.get_example(args, name)
    init_config = example.init_config
    assert init_config.modular_workflow == ("ref2va" if "ref2va" in name else
                                            "fl2va" if "fl2va" in name else "t2va")
    if not upstream_expected:
      assert init_config.pipeline_class.__module__.startswith("cache_dit.")
      assert init_config.modular_blocks_factory is PikaH3FlexBlocks
      assert example.input_data.reference_package == "cache_dit.models.pipelines.pika_h3_flex"
    elif upstream_pika_h3_flex_available():
      assert init_config.pipeline_class is diffusers.PikaH3FlexModularPipeline
      assert init_config.modular_blocks_factory is None
      assert example.input_data.reference_package == (
        "diffusers.modular_pipelines.pika_h3_flex.references")
    else:
      pytest.skip("the installed diffusers predates modular_pipelines/pika_h3_flex")


def test_enable_cache_on_the_official_pipeline_keeps_the_forward_signature():
  from diffusers import PikaH3FlexBlocks as UpstreamBlocks

  transformer = _tiny_transformer()
  before = list(inspect.signature(transformer.forward).parameters)
  original_forward = transformer.forward
  pipe = diffusers.PikaH3FlexModularPipeline(blocks=UpstreamBlocks())
  pipe.update_components(transformer=transformer)

  cache_dit.enable_cache(pipe)

  # The modular denoiser step builds its kwargs from this signature (`denoise.py`), so a patched forward
  # that collapses to `(*args, **kwargs)` drops `token_tags` / `position_ids` / the row indices and the
  # first denoise step raises a missing-argument TypeError.
  assert pipe.transformer.forward is not original_forward
  assert list(inspect.signature(pipe.transformer.forward).parameters) == before
  assert getattr(pipe.transformer, "_is_cached", False) is True
  # `summary` resolves a modular pipeline through the adapter, which is how `--cache-summary` prints.
  assert cache_dit.summary(pipe)


def test_enable_cache_on_the_vendored_copy_keeps_the_forward_signature():
  # The fallback is only a fallback if it runs: the vendored block tree is a `ModularPipeline` too, and
  # the same signature contract applies (its denoiser introspects `transformer.forward`).
  transformer = _tiny_transformer()
  before = list(inspect.signature(transformer.forward).parameters)
  pipe = PikaH3FlexModularPipeline(blocks=PikaH3FlexBlocks())
  pipe.update_components(transformer=transformer)

  cache_dit.enable_cache(pipe)

  assert list(inspect.signature(pipe.transformer.forward).parameters) == before
  assert getattr(pipe.transformer, "_is_cached", False) is True
  assert cache_dit.summary(pipe)
