"""CPU tests for the vendored MAGI-2 Preview pipeline.

The released checkpoint is 228 GiB, so the pipeline contract is pinned with an injected tiny
transformer: the `from_pretrained` argument handling, the sampler wiring (one transformer call per
step, both CFG branches packed together) and the decode-free output path.
"""
import json

import pytest
import torch

import cache_dit
from cache_dit import DBCacheConfig
from cache_dit.caching.block_adapters import magi2_preview_adapter
from cache_dit.models.pipelines.magi2.configuration import (
  Magi2MHCConfig,
  Magi2MoEConfig,
  Magi2PreviewConfig,
)
from cache_dit.models.pipelines.magi2.pipeline import Magi2PreviewPipeline
from cache_dit.models.pipelines.magi2.transformer import Magi2PreviewTransformer

TEXT_TOKENS = 5
TEXT_DIM = 8
LATENT = (1, 4, 2, 4, 4)
AUDIO_LATENT = (1, 6, 4)


def tiny_transformer(seed: int = 0) -> Magi2PreviewTransformer:
  """Tiny transformer with every parameter filled deterministically."""

  config = Magi2PreviewConfig(
    num_layers=2,
    hidden_size=32,
    head_dim=8,
    num_query_groups=4,
    video_in_channels=4,
    audio_in_channels=4,
    text_in_channels=TEXT_DIM,
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
  config.validate()
  generator = torch.Generator().manual_seed(seed)
  model = Magi2PreviewTransformer(config).eval()
  with torch.no_grad():
    for parameter in model.parameters():
      parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)
  return model


def _checkpoint_root(tmp_path):
  """Create the minimal checkpoint tree `from_pretrained` validates."""

  preview = tmp_path / "preview"
  preview.mkdir()
  (preview / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
  return tmp_path


def _pipeline(tmp_path, transformer=None) -> Magi2PreviewPipeline:
  return Magi2PreviewPipeline.from_pretrained(
    _checkpoint_root(tmp_path),
    torch_dtype=torch.float32,
    transformer=transformer or tiny_transformer(),
    load_auxiliary=False,
  )


def test_from_pretrained_rejects_device_map(tmp_path):
  """`device_map` must fail fast: the CLI would otherwise skip `pipe.to(device)`."""

  with pytest.raises(NotImplementedError, match="device_map"):
    _pipeline_with_kwargs(tmp_path, device_map="balanced")


def _pipeline_with_kwargs(tmp_path, **kwargs) -> Magi2PreviewPipeline:
  return Magi2PreviewPipeline.from_pretrained(
    _checkpoint_root(tmp_path),
    transformer=tiny_transformer(),
    load_auxiliary=False,
    **kwargs,
  )


def test_from_pretrained_rejects_unknown_components(tmp_path):
  """Unknown kwargs must be rejected instead of silently ignored."""

  with pytest.raises(TypeError, match="unexpected MAGI-2 pipeline"):
    _pipeline_with_kwargs(tmp_path, controlnet=object())


def test_from_pretrained_requires_a_local_checkpoint(tmp_path):
  """A hub id is not accepted: the released repo has no diffusers layout."""

  with pytest.raises(ValueError, match="local checkpoint directory"):
    Magi2PreviewPipeline.from_pretrained(
      "sand-ai/MAGI-2-preview",
      transformer=tiny_transformer(),
      load_auxiliary=False,
    )


def test_missing_conditioner_is_reported(tmp_path):
  """A checkpoint without the released `text_encoder/` must name it instead of running blind."""

  with pytest.raises(FileNotFoundError, match="conditioner"):
    Magi2PreviewPipeline.from_pretrained(
      _checkpoint_root(tmp_path),
      transformer=tiny_transformer(),
    )


def test_pipeline_runs_the_transformer_once_per_step(tmp_path):
  """Both CFG branches share one transformer call per step, so the pack must stay inside it."""

  calls = []
  transformer = tiny_transformer()
  handle = transformer.register_forward_pre_hook(lambda module, args: calls.append(1))
  pipe = _pipeline(tmp_path, transformer)

  latents, audio = _sample(pipe, num_inference_steps=3)
  handle.remove()

  assert len(calls) == 3, f"expected one transformer call per step, got {len(calls)}"
  assert latents.shape == LATENT
  assert audio.shape == AUDIO_LATENT
  assert torch.isfinite(latents).all() and torch.isfinite(audio).all()


def _sample(pipe, num_inference_steps: int = 3):
  generator = torch.Generator().manual_seed(11)
  latent = torch.zeros(LATENT, dtype=torch.float32)
  audio_latent = torch.zeros(AUDIO_LATENT, dtype=torch.float32)
  txt = torch.randn(1, TEXT_TOKENS, TEXT_DIM, generator=generator)
  null = torch.randn(1, TEXT_TOKENS, TEXT_DIM, generator=generator)
  output = pipe(
    num_inference_steps=num_inference_steps,
    latent=latent,
    audio_latent=audio_latent,
    txt_feat=txt,
    null_txt_feat=null,
    latent_frames=LATENT[2],
    audio_frames=AUDIO_LATENT[1],
    decode=False,
  )
  return output["latent"], output["audio_latent"]


def test_seeded_run_is_reproducible(tmp_path):
  """Same seed and same inputs must reproduce the same latents."""

  pipe = _pipeline(tmp_path)
  first_latent, first_audio = _sample(pipe)
  second_latent, second_audio = _sample(pipe)
  torch.testing.assert_close(first_latent, second_latent, atol=0, rtol=0)
  torch.testing.assert_close(first_audio, second_audio, atol=0, rtol=0)


def test_decode_without_decoders_is_refused(tmp_path):
  """Asking for pixels without the released Wan2.2 VAE must fail loudly, not return latents."""

  pipe = _pipeline(tmp_path)
  generator = torch.Generator().manual_seed(11)
  txt = torch.randn(1, TEXT_TOKENS, TEXT_DIM, generator=generator)
  with pytest.raises(NotImplementedError, match="Wan2.2 VAE"):
    pipe(
      num_inference_steps=2,
      txt_feat=txt,
      null_txt_feat=txt.clone(),
      latent=torch.zeros(LATENT, dtype=torch.float32),
      audio_latent=torch.zeros(AUDIO_LATENT, dtype=torch.float32),
      latent_frames=LATENT[2],
      audio_frames=AUDIO_LATENT[1],
    )


def test_cache_wraps_the_layer_stack_only(tmp_path):
  """The adapter must leave the pre/post adapters outside the cached region."""

  transformer = tiny_transformer()
  pipe = _pipeline(tmp_path, transformer)
  adapter = magi2_preview_adapter(pipe)
  cache_dit.enable_cache(
    adapter,
    cache_config=DBCacheConfig(
      num_inference_steps=2,
      Fn_compute_blocks=1,
      Bn_compute_blocks=0,
      max_warmup_steps=0,
      max_cached_steps=1,
      residual_diff_threshold=100.0,
    ),
  )
  assert getattr(transformer.block, "_is_cached", False) is True
  assert not hasattr(transformer.pre_adapter, "_is_cached")
  assert not hasattr(transformer.post_adapter, "_is_cached")


def test_registry_entries_match_the_real_pipeline_class(tmp_path):
  """The adapter prefix, the planner key and the dispatch guard must all resolve for this model.

  Every one of these was broken at least once while the tests only exercised a hand-built adapter,
  so they are asserted through the real registries instead.
  """

  from diffusers.configuration_utils import ConfigMixin
  from diffusers.models.modeling_utils import ModelMixin

  from cache_dit.caching.block_adapters import BlockAdapterRegister
  from cache_dit.distributed.transformers.dispatch import (
    _ensure_cp_planners_activated,
    _ensure_tp_planners_activated,
    _parallelize_transformer_tensor,
  )
  from cache_dit.distributed.transformers.register import (
    ContextParallelismPlannerRegister,
    TensorParallelismPlannerRegister,
  )

  pipe = _pipeline(tmp_path)
  assert BlockAdapterRegister.is_supported(pipe)
  adapter = BlockAdapterRegister.get_adapter(pipe)
  assert adapter.blocks is pipe.transformer.block.layers
  assert BlockAdapterRegister.has_separate_cfg(pipe) is False

  assert isinstance(pipe.transformer.block, (ModelMixin, ConfigMixin))
  _ensure_cp_planners_activated()
  _ensure_tp_planners_activated()
  assert ContextParallelismPlannerRegister.get_planner(adapter.transformer) is not None
  assert TensorParallelismPlannerRegister.get_planner(adapter.transformer) is not None
  # The TP dispatch guards on `ModelMixin` before it asks the planner for a plan.
  assert _parallelize_transformer_tensor(adapter.transformer, None) is adapter.transformer


def test_image_conditioning_is_refused_explicitly(tmp_path):
  """`image` must fail with a named limit instead of a `TypeError` on the call signature."""

  pipe = _pipeline(tmp_path)
  with pytest.raises(NotImplementedError, match="Wan2.2 VAE encoder"):
    pipe(
      num_inference_steps=1,
      txt_feat=torch.zeros(1, TEXT_TOKENS, TEXT_DIM),
      null_txt_feat=torch.zeros(1, TEXT_TOKENS, TEXT_DIM),
      latent=torch.zeros(LATENT),
      audio_latent=torch.zeros(AUDIO_LATENT),
      latent_frames=LATENT[2],
      audio_frames=AUDIO_LATENT[1],
      decode=False,
      image="conditioning.png",
    )


def test_cli_offload_kwargs_use_the_declared_subtree_filter():
  """`--layerwise-offload` must stream whole layers, not fall back to leaf discovery.

  Leaf discovery would leave the MoE parameter blocks on the compute device, which is exactly the
  216 GiB OOM this model avoids by declaring its own filter.
  """

  from cache_dit._utils.utils import _declared_offload_owner, _layerwise_kwargs_for, get_args

  transformer = tiny_transformer()
  args = get_args(parse=False).parse_args([])
  kwargs = _layerwise_kwargs_for(
    args,
    target_name="transformer",
    module=transformer,
    onload_device=torch.device("cpu"),
    keep_vision_resident=False,
  )

  assert _declared_offload_owner(transformer) is transformer
  assert kwargs["module_filter"] is type(transformer).layerwise_offload_filter
  assert kwargs["module_filter"]("block.layers.0", transformer.block.layers[0])
  assert not kwargs["module_filter"]("pre_adapter", transformer.pre_adapter)


def test_cli_offload_keeps_the_non_streamed_parts_resident():
  """The adapters run every step, so the CLI path has to move them onto the compute device."""

  from cache_dit._utils.utils import _apply_declared_residency

  transformer = tiny_transformer()
  calls = []
  transformer.resident_to = lambda device: calls.append(device)

  _apply_declared_residency(transformer, torch.device("cpu"))

  assert calls == [torch.device("cpu")]


def test_resident_components_are_not_streamed(tmp_path):
  """The conditioner runs a hand-written kernel on a raw weight, so it must stay off the offload
  path.

  Unknown components also have to be skipped: an offloaded conditioner fails with a device mismatch
  only in the *synchronous* offload mode, which is why the declaration is asserted here.
  """

  from cache_dit._utils.utils import _resolve_generic_offload_targets

  pipe = _pipeline(tmp_path)
  pipe.text_encoder = torch.nn.Linear(2, 2)
  pipe.audio_decoder = torch.nn.Linear(2, 2)

  names = [name for name, _ in _resolve_generic_offload_targets(pipe)]

  assert "transformer" in names
  assert "text_encoder" not in names
  assert "audio_decoder" in names
