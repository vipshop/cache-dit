"""Regression tests for the MAGI-2 Preview cache integration.

MAGI-2 Preview ships its own pipeline (`cache_dit.models.pipelines.magi2`) because diffusers has no
equivalent 114B audio-video MoE. These tests lock the adapter wiring (`block.layers` +
`Pattern_3` + non-separate CFG), the explicit stack assertions that replace the lenient
`_relaxed_assert` path, and the cached-run bookkeeping on a tiny configuration (no weights, CPU only).
"""
import copy

import pytest
import torch
from diffusers import DiffusionPipeline

import cache_dit
from cache_dit import DBCacheConfig
from cache_dit.caching.block_adapters import BlockAdapterRegister
from cache_dit.caching.forward_pattern import ForwardPattern
from cache_dit.models.pipelines.magi2.configuration import (
  Magi2MHCConfig,
  Magi2MoEConfig,
  Magi2PreviewConfig,
)
from cache_dit.models.pipelines.magi2.data_proxy import Magi2DataProxy, ModelInput
from cache_dit.models.pipelines.magi2.transformer import (
  Magi2PreviewTransformer,
  Magi2TransformerLayer,
)

NUM_LAYERS = 3
NUM_STEPS = 3
VIDEO_CHANNELS = 4
AUDIO_CHANNELS = 4
TEXT_CHANNELS = 8
LATENT_T = 2
LATENT_H = 4
LATENT_W = 4
AUDIO_TOKENS = 6
TEXT_TOKENS = 5
TIME_CHANNEL_DIM = 8


def tiny_config(**overrides) -> Magi2PreviewConfig:
  """Build a two-layer-scale config that exercises the dense, multimodal and MoE paths.

  :param overrides: Field overrides applied on top of the tiny defaults.
  :returns: A validated tiny config.
  """

  config = dict(
    num_layers=NUM_LAYERS,
    hidden_size=32,
    head_dim=8,
    num_query_groups=4,
    video_in_channels=VIDEO_CHANNELS,
    audio_in_channels=AUDIO_CHANNELS,
    text_in_channels=TEXT_CHANNELS,
    multimodal_layers=(0, ),
    params_dtype=torch.float32,
    mhc=Magi2MHCConfig(num_streams=2, sinkhorn_iterations=2),
    moe=Magi2MoEConfig(
      num_heads=2,
      num_experts=4,
      top_k=2,
      expert_intermediate_size=8,
      shared_expert_intermediate_size=8,
      modality_shared_expert_intermediate_size=8,
      layers=(1, ),
    ),
  )
  config.update(overrides)
  instance = Magi2PreviewConfig(**config)
  instance.validate()
  return instance


def tiny_transformer(seed: int = 0, **overrides) -> Magi2PreviewTransformer:
  """Build a tiny transformer with seeded weights.

  :param seed: Torch seed.
  :param overrides: Config overrides.
  :returns: The transformer in eval mode.
  """

  torch.manual_seed(seed)
  model = Magi2PreviewTransformer(tiny_config(**overrides)).eval()
  # The vendored modules allocate with `torch.empty` (the released weights are loaded on top), so a
  # test must fill every parameter itself: uninitialized memory makes runs order dependent.
  generator = torch.Generator().manual_seed(seed)
  with torch.no_grad():
    for parameter in model.parameters():
      parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)
  return model


def step_inputs(seed: int = 1):
  """Build one packed denoising-step input.

  :param seed: Torch seed.
  :returns: ``(packed_model_args, output_layout)``.
  """

  generator = torch.Generator().manual_seed(seed)
  batch = 2  # conditional + unconditional packed into one varlen sequence
  data = ModelInput(
    x_t=torch.randn(batch, VIDEO_CHANNELS, LATENT_T, LATENT_H, LATENT_W, generator=generator),
    audio_x_t=torch.randn(batch, AUDIO_TOKENS, AUDIO_CHANNELS, generator=generator),
    audio_feat_len=[AUDIO_TOKENS] * batch,
    txt_feat=torch.randn(batch, TEXT_TOKENS, TEXT_CHANNELS, generator=generator),
    txt_feat_len=[TEXT_TOKENS] * batch,
    t=torch.tensor([0.5] * batch),
    per_token_video_t=torch.full((batch, 1, LATENT_T, LATENT_H, LATENT_W), 0.5),
    per_token_audio_t=torch.full((batch, AUDIO_TOKENS, 1), 0.5),
  )
  packed = Magi2DataProxy(Magi2DataProxyConfigForTest()).process_input(data)
  return packed.model_args, packed.output_layout


def Magi2DataProxyConfigForTest():
  """Return the packing config used by the tests (small time channel).

  :returns: A data-proxy config.
  """

  from cache_dit.models.pipelines.magi2.data_proxy import Magi2PreviewDataProxyConfig

  return Magi2PreviewDataProxyConfig(time_channel_dim=TIME_CHANNEL_DIM)


class Magi2PreviewTestPipelineBase(DiffusionPipeline):
  """Minimal stand-in for `Magi2Pipeline`: registers `transformer` and drives the sampler loop."""

  def __init__(self, transformer: Magi2PreviewTransformer) -> None:
    super().__init__()
    self.register_modules(transformer=transformer)

  def __call__(self, model_args, num_steps: int = NUM_STEPS):
    """Run ``num_steps`` transformer calls and return the per-step outputs.

    :param model_args: Packed positional arguments of the transformer.
    :param num_steps: Number of denoising steps.
    :returns: One output tensor per step.
    """

    outputs = []
    with torch.no_grad():
      for _ in range(num_steps):
        outputs.append(self.transformer(*model_args))
    return outputs


def Magi2PreviewTestPipeline(transformer: Magi2PreviewTransformer):
  """Build a test pipeline bound to a fresh class.

  cache-dit patches `pipe.__class__.__call__` to open one cache context per inference, so two
  pipelines sharing a class would share the first test's context manager. A per-call subclass keeps
  the tests independent.

  :param transformer: Transformer to register.
  :returns: The test pipeline instance.
  """

  class _Pipeline(Magi2PreviewTestPipelineBase):
    pass

  return _Pipeline(transformer)


def test_prefix_is_the_unique_match():
  """`Magi2Preview` must be the only adapter prefix matching the transformer class."""

  prefixes = BlockAdapterRegister.supported_pipelines()[1]
  assert "Magi2Preview" in prefixes
  matches = [name for name in prefixes if "Magi2PreviewTransformer".startswith(name)]
  assert matches == ["Magi2Preview"], f"ambiguous adapter prefixes for MAGI-2: {matches}"


def test_adapter_wiring():
  """The adapter must point at the inner block stack with the documented contract."""

  from cache_dit.caching.block_adapters import magi2_preview_adapter

  transformer = tiny_transformer()
  pipe = Magi2PreviewTestPipeline(transformer)
  adapter = magi2_preview_adapter(pipe)

  assert adapter.transformer is transformer.block
  assert adapter.blocks is transformer.block.layers
  assert adapter.blocks_name == "layers"
  assert adapter.forward_pattern == ForwardPattern.Pattern_3
  assert adapter.check_forward_pattern is True
  # One forward per step already carries both CFG branches.
  assert adapter.has_separate_cfg is False
  # No loop-body state to keep alive, so no structural patch is installed.
  assert adapter.patch_functor is None


def test_registered_adapter_supports_transformer_only_usage():
  """The registry must resolve the adapter from a bare transformer as well."""

  transformer = tiny_transformer()
  adapter = BlockAdapterRegister.get_adapter(transformer)

  assert adapter is not None
  assert adapter.transformer is transformer.block
  assert adapter.blocks is transformer.block.layers


def test_forward_pattern_is_validated():
  """`check_forward_pattern=True` must accept the built-in layer signature."""

  transformer = tiny_transformer()
  adapter = BlockAdapterRegister.get_adapter(transformer)
  assert adapter.check_forward_pattern is True
  # A plain `torch.nn.Module` block would fail the check; ours declares the released signature.
  import inspect

  parameters = list(inspect.signature(Magi2TransformerLayer.forward).parameters)
  assert parameters[1] == "hidden_states"


def test_adapter_rejects_a_wrong_stack():
  """A transformer whose `block` is not the released container must fail fast."""

  from cache_dit.caching.block_adapters import magi2_preview_adapter

  class _WrongStack(torch.nn.Module):

    def __init__(self) -> None:
      super().__init__()
      self.layers = torch.nn.ModuleList([torch.nn.Identity()])

  transformer = tiny_transformer()
  transformer.block = _WrongStack()
  with pytest.raises(AssertionError, match="Magi2TransformerBlock"):
    magi2_preview_adapter(Magi2PreviewTestPipeline(transformer))


def test_repeated_blocks_are_declared():
  """Regional quantization and `compile_repeated_blocks` discover the stack from
  `_repeated_blocks`."""

  transformer = tiny_transformer()
  assert transformer._repeated_blocks == ["Magi2TransformerLayer"]
  assert isinstance(transformer.block.layers, torch.nn.ModuleList)
  assert len(transformer.block.layers) == NUM_LAYERS
  assert all(isinstance(layer, Magi2TransformerLayer) for layer in transformer.block.layers)


def test_cached_run_is_bit_exact_when_no_step_is_cached():
  """With caching disabled the block wrapper must be numerically transparent."""

  from cache_dit.caching.block_adapters import magi2_preview_adapter

  reference = tiny_transformer(seed=0)
  transformer = copy.deepcopy(reference)
  model_args, _ = step_inputs()

  pipe = Magi2PreviewTestPipeline(transformer)
  adapter = magi2_preview_adapter(pipe)
  cache_dit.enable_cache(
    adapter,
    cache_config=DBCacheConfig(
      num_inference_steps=NUM_STEPS,
      Fn_compute_blocks=1,
      Bn_compute_blocks=0,
      max_warmup_steps=0,
      max_cached_steps=0,  # caching disabled: the run must stay bit-exact
      residual_diff_threshold=0.0,
    ),
  )

  outputs = pipe(model_args)
  expected = Magi2PreviewTestPipeline(reference)(model_args)
  for step, (actual, reference_output) in enumerate(zip(outputs, expected)):
    torch.testing.assert_close(actual,
                               reference_output,
                               atol=0,
                               rtol=0,
                               msg=f"diverged at step {step}")


def test_cache_is_applied_and_skips_steps():
  """Enabling the cache must actually rewrite the stack and change the trajectory."""

  from cache_dit.caching.block_adapters import magi2_preview_adapter

  reference = tiny_transformer(seed=0)
  transformer = copy.deepcopy(reference)
  model_args, _ = step_inputs()

  pipe = Magi2PreviewTestPipeline(transformer)
  adapter = magi2_preview_adapter(pipe)
  cache_dit.enable_cache(
    adapter,
    cache_config=DBCacheConfig(
      num_inference_steps=NUM_STEPS,
      Fn_compute_blocks=1,
      Bn_compute_blocks=0,
      max_warmup_steps=0,
      max_cached_steps=1,
      residual_diff_threshold=10.0,  # everything after the first step is a cache hit
    ),
  )

  # cache-dit patches `block.forward` with a version that swaps `layers` for the cached stack during
  # the call, so the module list itself keeps its real layers; `_is_cached` marks the patched module.
  assert getattr(transformer.block, "_is_cached", False) is True

  outputs = pipe(model_args)
  expected = Magi2PreviewTestPipeline(reference)(model_args)
  assert torch.equal(outputs[0], expected[0]), "the first step must run the real stack"
  # The tiny fixture replays the same packed input every step, so only the steps the cache actually
  # answered differ; real acceptance runs feed evolving latents and measure PSNR/SSIM.
  assert any(not torch.equal(actual, reference_output) for actual, reference_output in zip(
    outputs[1:], expected[1:])), "the cache never answered a step"
