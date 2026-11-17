"""CPU-only guards for the P-H3-Image adapter.

P-H3-Image drives the **video** MiniMax-H3 transformer (`MiniMaxH3Transformer3DModel`), so its block
contract, CP plan and TP plan are the video ones. Two things are specific to it and are what this
file pins down:

1. **The pipeline prefix.** `PH3ImageModularPipeline` matches none of the registered prefixes, so
   `enable_cache` used to reject it. The `PH3Image` entry has to be registered *and* whitelisted as a
   verified modular pipeline — the whitelist only warns, so a missing entry would degrade silently.
2. **The forward signature.** A modular denoise step forwards only the kwargs that appear in
   `inspect.signature(transformer.forward)`, and P-H3-Image's step does exactly that
   (`p_h3_image/denoise.py:119-133`). A `(*args, **kwargs)` replacement would silently drop
   `audio_hidden_states` / `timestep_indices` / `token_tags` / `position_ids` / the three index
   tensors, and the first denoise step would raise inside the transformer.
"""

import inspect

import torch

import cache_dit
from cache_dit import DBCacheConfig
from cache_dit.caching.block_adapters.block_registers import BlockAdapterRegister

PH3_FORWARD_INPUTS = (
  "hidden_states",
  "audio_hidden_states",
  "encoder_hidden_states",
  "timestep",
  "timestep_indices",
  "token_tags",
  "position_ids",
  "video_indices",
  "audio_indices",
  "text_indices",
)


def _tiny_transformer():
  from diffusers import MiniMaxH3Transformer3DModel

  torch.manual_seed(0)
  return MiniMaxH3Transformer3DModel(
    num_attention_heads=4,
    attention_head_dim=8,
    hidden_size=32,
    num_layers=1,
    num_refiner_layers=1,
    ffn_dim=64,
    in_channels=4,
    audio_in_channels=4,
    patch_size=(1, 2, 2),
    text_dim=8,
    freq_dim=8,
    time_embed_hidden_dim=32,
    time_embed_dim=16,
    # `2 * 3 * rope_freq_dim` channels of the head are rotated, so it has to stay within `head_dim`.
    rope_freq_dim=1,
  )


def _packed_inputs(num_text=2, num_audio=2, num_video=4):
  """The `[text | audio | video]` packed sequence of one `text2image` step.

  :param num_text: Text rows.
  :param num_audio: Audio rows (channel-major, two per latent here).
  :param num_video: Image rows, `latent_height // 2 * latent_width // 2` for a `(1, 2, 2)` patch.
  :returns: Keyword arguments for `MiniMaxH3Transformer3DModel.forward`.
  """
  length = num_text + num_audio + num_video
  video_indices = torch.arange(num_text + num_audio, length)
  audio_indices = torch.arange(num_text, num_text + num_audio)
  text_indices = torch.arange(num_text)
  token_tags = torch.empty(length, dtype=torch.long)
  token_tags[text_indices] = 1
  token_tags[audio_indices] = 2
  token_tags[video_indices] = 0
  # Two noise levels in one sequence: the generated rows step down, the conditioning rows are pinned.
  timestep_indices = torch.zeros(length, dtype=torch.long)
  timestep_indices[video_indices[:2]] = 1
  return dict(
    hidden_states=torch.randn(1, num_video, 4 * 1 * 2 * 2),
    audio_hidden_states=torch.randn(1, num_audio, 4),
    encoder_hidden_states=torch.randn(1, num_text, 8),
    timestep=torch.tensor([0.5, 0.999]),
    timestep_indices=timestep_indices,
    token_tags=token_tags,
    position_ids=torch.zeros(length, 3),
    video_indices=video_indices,
    audio_indices=audio_indices,
    text_indices=text_indices,
  )


def test_pipeline_prefix_resolves_to_the_h3_family_adapter():
  """`PH3ImageModularPipeline` must hit the `PH3Image` entry, not "no adapter"."""
  assert BlockAdapterRegister.is_supported("PH3ImageModularPipeline")
  prefix, factory = BlockAdapterRegister._match_adapter("PH3ImageModularPipeline")
  assert prefix == "PH3Image"
  assert factory is not None
  # The video `MiniMaxH3` prefix does not match this class name, which is why the entry is needed.
  assert not "PH3ImageModularPipeline".startswith("MiniMaxH3")


def test_pipeline_is_whitelisted_as_a_verified_modular_pipeline():
  """The whitelist only warns, so a missing entry would pass every functional check."""
  assert "PH3Image" in BlockAdapterRegister._verified_modular_pipelines


def test_forward_signature_preserved_after_enable_cache():
  transformer = _tiny_transformer()
  signature_before = inspect.signature(transformer.forward)

  cache_dit.enable_cache(
    transformer,
    cache_config=DBCacheConfig(
      Fn_compute_blocks=1,
      Bn_compute_blocks=0,
      residual_diff_threshold=0.05,
    ),
  )

  signature_after = inspect.signature(transformer.forward)
  assert tuple(signature_after.parameters) == tuple(signature_before.parameters)
  for name in PH3_FORWARD_INPUTS:
    message = f"'{name}' disappeared from the patched transformer.forward signature"
    assert name in signature_after.parameters, message

  # The modular step forwards exactly these fields, all as keyword arguments.
  inputs = _packed_inputs()
  video_output, audio_output = transformer(**inputs, return_dict=False)
  assert video_output.shape == (1, 4, 4 * 1 * 2 * 2)
  assert audio_output.shape == (1, 2, 4)


def test_cp_and_tp_planners_cover_the_transformer_ph3_drives():
  """P-H3-Image gets CP / TP from the video planners: assert they resolve for its class."""
  from cache_dit.distributed.transformers.dispatch import (
    _ensure_cp_planners_activated,
    _ensure_tp_planners_activated,
  )
  from cache_dit.distributed.transformers.register import (
    ContextParallelismPlannerRegister,
    TensorParallelismPlannerRegister,
  )

  _ensure_cp_planners_activated()
  _ensure_tp_planners_activated()
  # Both `_ensure_*` helpers only activate when nothing is registered yet, so a test session that
  # loaded another family first would skip them; importing the module registers unconditionally and
  # keeps this assertion about the planner set rather than about the order files ran in.
  import cache_dit.distributed.transformers.minimax_h3  # noqa: F401

  assert "MiniMaxH3Transformer3DModel" in ContextParallelismPlannerRegister.supported_planners()[1]
  assert "MiniMaxH3Transformer3DModel" in TensorParallelismPlannerRegister.supported_planners()[1]
