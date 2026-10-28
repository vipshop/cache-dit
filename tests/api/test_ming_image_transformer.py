"""Regression tests for the Ming-Image transformer adapter.

Ming-Image reuses the Z-Image stack and only adds two input paths: reference latents appended on the
frame axis and direct conditions appended on the prompt sequence axis. These tests lock that wiring
with tiny configs and random weights only, so they stay CPU-fast:

* the two checkpoint variants keep the right padding-token semantics (`zero_masked` buffers vs
  learned parameters) without leaking `x_pad_token` / `cap_pad_token` into the zero-masked state
  dict;
* the adapter forward is bit-exact with a plain `ZImageTransformer2DModel` fed the same
  pre-concatenated inputs, which pins the concatenation order, the frame crop and the (unchanged)
  output sign;
* save/load roundtrips keep the config and stay strict.
"""

import pytest
import torch
from diffusers import ZImageTransformer2DModel

from cache_dit.models.pipelines.ming_image.transformer import (
  LEARNED_PADDING,
  MING_AXES_LENS,
  ZERO_MASKED_PADDING,
  MingImageTransformer2DModel,
)

CHANNELS = 4
DIM = 64
HEADS = 2
CAP_DIM = 8
HEIGHT = 8
WIDTH = 8

TINY_CONFIG = dict(
  all_patch_size=(2, ),
  all_f_patch_size=(1, ),
  in_channels=CHANNELS,
  dim=DIM,
  n_layers=2,
  n_refiner_layers=1,
  n_heads=HEADS,
  n_kv_heads=HEADS,
  cap_feat_dim=CAP_DIM,
  axes_dims=(8, 12, 12),
  axes_lens=(256, 64, 64),
)


def build_transformer(padding_mode=ZERO_MASKED_PADDING, multi_frame_output=False, **overrides):
  config = dict(TINY_CONFIG, **overrides)
  torch.manual_seed(0)
  return MingImageTransformer2DModel(
    **config,
    alignment_padding_mode=padding_mode,
    multi_frame_output=multi_frame_output,
  )


def build_inputs(batch_size=2,
                 frames=(1, 1),
                 ref_frames=(2, 1),
                 cap_lens=(5, 7),
                 direct_lens=(3, 2)):
  generator = torch.Generator().manual_seed(1)
  x = [
    torch.randn(CHANNELS, frames[i], HEIGHT, WIDTH, generator=generator) for i in range(batch_size)
  ]
  cap = [torch.randn(cap_lens[i], CAP_DIM, generator=generator) for i in range(batch_size)]
  ref = [
    torch.randn(CHANNELS, ref_frames[i], HEIGHT, WIDTH, generator=generator)
    for i in range(batch_size)
  ]
  direct = [torch.randn(direct_lens[i], DIM, generator=generator) for i in range(batch_size)]
  t = torch.linspace(0.2, 0.8, batch_size)
  return x, t, cap, ref, direct


def plain_zimage(**overrides):
  return ZImageTransformer2DModel(**dict(TINY_CONFIG, **overrides))


def test_zero_masked_pad_tokens_are_non_persistent_buffers():
  model = build_transformer(ZERO_MASKED_PADDING)

  assert model.config.alignment_padding_mode == ZERO_MASKED_PADDING
  assert not isinstance(model.x_pad_token, torch.nn.Parameter)
  assert not isinstance(model.cap_pad_token, torch.nn.Parameter)
  assert torch.count_nonzero(model.x_pad_token) == 0
  assert torch.count_nonzero(model.cap_pad_token) == 0

  buffer_names = {name for name, _ in model.named_buffers()}
  assert {"x_pad_token", "cap_pad_token"} <= buffer_names
  state_keys = set(model.state_dict().keys())
  assert "x_pad_token" not in state_keys
  assert "cap_pad_token" not in state_keys

  base_keys = set(plain_zimage().state_dict().keys())
  assert base_keys - state_keys == {"x_pad_token", "cap_pad_token"}
  assert state_keys <= base_keys


def test_learned_pad_tokens_are_parameters():
  model = build_transformer(LEARNED_PADDING, multi_frame_output=True)

  assert model.config.alignment_padding_mode == LEARNED_PADDING
  assert isinstance(model.x_pad_token, torch.nn.Parameter)
  assert isinstance(model.cap_pad_token, torch.nn.Parameter)

  state_keys = set(model.state_dict().keys())
  assert {"x_pad_token", "cap_pad_token"} <= state_keys
  assert state_keys == set(plain_zimage().state_dict().keys())


def test_default_axes_lens_matches_ming():
  config = {key: value for key, value in TINY_CONFIG.items() if key != "axes_lens"}
  model = MingImageTransformer2DModel(**config)
  assert list(model.config.axes_lens) == MING_AXES_LENS
  assert model.config.alignment_padding_mode == ZERO_MASKED_PADDING
  assert model.config.multi_frame_output is False
  assert model._repeated_blocks == ["ZImageTransformerBlock"]


def test_unsupported_config_is_rejected():
  with pytest.raises(ValueError, match="alignment padding mode"):
    build_transformer("silent_padding")
  with pytest.raises(TypeError, match="multi_frame_output"):
    build_transformer(ZERO_MASKED_PADDING, multi_frame_output="yes")


def copy_plain_weights(model, padding_mode):
  """Load the same weights into the adapter and into a plain Z-Image module.

  Zero-masked checkpoints have no padding-token weights, so the plain Z-Image padding parameters are
  zeroed to reproduce that behaviour; learned checkpoints keep the inherited parameters.
  """
  base = plain_zimage()
  state_dict = base.state_dict()
  with torch.no_grad():
    state_dict["x_pad_token"].zero_()
    state_dict["cap_pad_token"].zero_()
  if padding_mode == LEARNED_PADDING:
    model.load_state_dict(state_dict, strict=True)
  else:
    model.load_state_dict(
      {
        key: value
        for key, value in state_dict.items() if "pad_token" not in key
      }, strict=True)
    assert torch.count_nonzero(model.x_pad_token) == 0
    assert torch.count_nonzero(model.cap_pad_token) == 0
  return base


@pytest.mark.parametrize(
  "padding_mode,frames,ref_frames,cap_lens",
  [
    # Learned padding keeps the parent's rule (inner padding stays attendable), so it must stay
    # bit-exact with plain Z-Image even when the sequences carry inner padding rows.
    (LEARNED_PADDING, (1, 1), (2, 1), (5, 7)),
    (LEARNED_PADDING, (2, 2), (2, 2), (32, 32)),
    # Zero-masked additionally masks inner padding, which only coincides with the parent when there
    # is no inner padding: 4 frames x 4 x 4 patches and 32 prompt tokens are exactly SEQ_MULTI_OF
    # aligned, so every mask row is all-True.
    (ZERO_MASKED_PADDING, (2, 2), (2, 2), (32, 32)),
  ],
  ids=["learned-padded", "learned-aligned", "zero-masked-aligned"],
)
def test_reference_latents_match_plain_zimage_with_preconcatenated_frames(
    padding_mode, frames, ref_frames, cap_lens):
  model = build_transformer(padding_mode, multi_frame_output=frames[0] > 1)
  base = copy_plain_weights(model, padding_mode)

  x, t, cap, ref, _ = build_inputs(frames=frames, ref_frames=ref_frames, cap_lens=cap_lens)
  output = model(x, t, cap, ref_latents=ref)
  expected = base(
    [torch.cat([item, ref_item], dim=1) for item, ref_item in zip(x, ref)],
    t,
    cap,
  )

  assert len(output.sample) == len(x)
  for sample, reference, latent in zip(output.sample, expected.sample, x):
    assert sample.shape == latent.shape
    assert torch.equal(sample, reference[:, :latent.shape[1]])


@pytest.mark.parametrize(
  "direct_tokens,expected_offset",
  [
    (0, 257),  # prompt only: 256 is already SEQ_MULTI_OF aligned
    (1, 289),  # 257 -> 288
    (42, 321),  # text prompt: 256 + 42 = 298 -> 320
    (582, 865),  # reference image prompt: 256 + 582 = 838 -> 864
  ],
)
def test_prompt_length_sets_the_latent_rope_offset(direct_tokens, expected_offset):
  # The connector emits 256 prompt tokens and the direct condition is per prompt token
  # (`[B, S, dim]`), so the latent RoPE range starts at ceil((256 + S) / 32) * 32 + 1.
  model = build_transformer(ZERO_MASKED_PADDING, axes_lens=(1024, 64, 64))
  x = [torch.randn(CHANNELS, 1, HEIGHT, WIDTH)]
  prompt = torch.randn(256, CAP_DIM)
  cap_item = torch.cat([prompt, prompt[-1:].repeat(direct_tokens, 1)], dim=0)

  assert int(model.patchify_and_embed(x, [cap_item], 2, 1)[3][0][0, 0]) == expected_offset


def test_zero_masked_drops_inner_padding_from_the_attention_mask():
  model = build_transformer(ZERO_MASKED_PADDING)
  feats = [torch.randn(8, DIM), torch.randn(8, DIM)]
  pos_ids = [
    model.create_coordinate_grid((8, 1, 1), (1, 0, 0), torch.device("cpu")).flatten(0, 2)
    for _ in feats
  ]
  # Item 0 has 3 inner padding rows, item 1 has none. Equal lengths make the parent return None.
  inner_pad = [torch.tensor([False] * 5 + [True] * 3), torch.tensor([False] * 8)]

  for pad_token in (model.x_pad_token, model.cap_pad_token):
    _, _, attn_mask, item_seqlens, _ = model._prepare_sequence(feats, pos_ids, inner_pad, pad_token,
                                                               None, torch.device("cpu"))
    assert item_seqlens == [8, 8]
    assert attn_mask is not None
    assert attn_mask.tolist() == [[True] * 5 + [False] * 3, [True] * 8]

  learned = build_transformer(LEARNED_PADDING)
  learned_pad = [torch.tensor([False] * 5 + [True] * 3), torch.tensor([False] * 8)]
  _, _, learned_mask, _, _ = learned._prepare_sequence(feats, pos_ids, learned_pad,
                                                       learned.cap_pad_token, None,
                                                       torch.device("cpu"))
  assert learned_mask is None


def test_zero_masked_masking_only_matters_when_inner_padding_exists():
  masked = build_transformer(ZERO_MASKED_PADDING)
  # Same weights and zero padding embeddings, but the learned variant skips the inner-pad masking.
  unmasked = build_transformer(LEARNED_PADDING)
  with torch.no_grad():
    unmasked.x_pad_token.zero_()
    unmasked.cap_pad_token.zero_()

  aligned_x, aligned_t, aligned_cap, _, _ = build_inputs(frames=(2, 2), cap_lens=(32, 32))
  for left, right in zip(
      masked(aligned_x, aligned_t, aligned_cap).sample,
      unmasked(aligned_x, aligned_t, aligned_cap).sample,
  ):
    assert torch.equal(left, right)

  padded_x, padded_t, padded_cap, _, _ = build_inputs(frames=(1, 1), cap_lens=(5, 7))
  for left, right in zip(
      masked(padded_x, padded_t, padded_cap).sample,
      unmasked(padded_x, padded_t, padded_cap).sample,
  ):
    assert not torch.equal(left, right)


def test_empty_direct_condition_matches_no_direct_condition():
  # `MingImageConditioning` returns `(B, 0, dim)` when there is no direct condition, so an empty
  # `direct_condition` has to be a no-op rather than a degenerate row range.
  model = build_transformer(ZERO_MASKED_PADDING)
  x, t, cap, _, _ = build_inputs()

  empty = model(x, t, cap, direct_condition=torch.zeros(2, 0, DIM))
  plain = model(x, t, cap)

  for left, right in zip(empty.sample, plain.sample):
    assert torch.equal(left, right)


@pytest.mark.parametrize(
  "axes_lens,cap_lens,direct_lens",
  [
    ((256, 64, 64), (5, 7), (3, 2)),
    # Realistic scale: 256 prompt tokens plus a 42-token text prompt in the direct branch, i.e. two
    # prompt items of 298 rows that the parent pads to 320.
    ((2048, 64, 64), (256, 256), (42, 42)),
    # Same with a reference-image prompt: 256 + 582 = 838 rows, padded to 864.
    ((2048, 64, 64), (256, 256), (582, 582)),
  ],
  ids=["tiny", "text-prompt", "reference-image-prompt"],
)
def test_direct_condition_is_spliced_after_the_prompt_embedder(axes_lens, cap_lens, direct_lens):
  model = build_transformer(ZERO_MASKED_PADDING, axes_lens=axes_lens)
  x, t, cap, _, direct = build_inputs(cap_lens=cap_lens, direct_lens=direct_lens)

  spliced = model(x, t, cap, direct_condition=direct)

  # Reference: embed the prompts, append the hidden-space direct tokens, and feed the result back
  # through the same transformer with `cap_embedder` bypassed. Both paths must agree bit-exactly,
  # which pins the splice point and the prompt sequence length (and thus the RoPE range).
  embedded = model.cap_embedder(torch.cat(cap, dim=0)).split([len(item) for item in cap], dim=0)
  pre_embedded = [
    torch.cat([emb, direct_item], dim=0) for emb, direct_item in zip(embedded, direct)
  ]
  model.cap_embedder.forward = lambda tensor: tensor
  manual = model(x, t, pre_embedded)
  del model.cap_embedder.forward

  for sample, reference in zip(spliced.sample, manual.sample):
    assert torch.equal(sample, reference)

  # The direct tokens really take part in the denoising: dropping them changes the result.
  without_direct = model(x, t, cap)
  for with_direct, plain in zip(spliced.sample, without_direct.sample):
    assert with_direct.shape == plain.shape
    assert not torch.equal(with_direct, plain)


def test_accepts_the_connector_output_shapes():
  # `MingImageConditioning` emits cap_feats `(256, cap_feat_dim)` per sample and a hidden-space
  # direct condition of shape `(B, 1, dim)`; 256 prompt tokens also push the latent RoPE past its
  # first axis, so widen `axes_lens` like the real checkpoint does (`20480`).
  model = build_transformer(ZERO_MASKED_PADDING, axes_lens=(2048, 64, 64))
  x = [torch.randn(CHANNELS, 1, HEIGHT, WIDTH) for _ in range(2)]
  cap = [torch.randn(256, CAP_DIM) for _ in range(2)]
  direct = torch.randn(2, 42, DIM)
  t = torch.tensor([0.4, 0.6])

  output = model(x, t, cap, direct_condition=direct)
  assert [sample.shape for sample in output.sample] == [(CHANNELS, 1, HEIGHT, WIDTH)] * 2


def test_direct_condition_does_not_touch_the_prompt_embedding_weights():
  model = build_transformer(LEARNED_PADDING)
  x, t, cap, _, direct = build_inputs()
  before = {name: value.clone() for name, value in model.cap_embedder.state_dict().items()}

  model(x, t, cap, direct_condition=direct)

  for name, value in model.cap_embedder.state_dict().items():
    assert torch.equal(value, before[name])
  assert model.cap_embedder.direct_rows == []


@pytest.mark.parametrize("padding_mode", [ZERO_MASKED_PADDING, LEARNED_PADDING])
def test_state_dict_names_roundtrip_into_plain_zimage(padding_mode):
  model = build_transformer(padding_mode)
  base = copy_plain_weights(model, padding_mode)

  # A plain Z-Image module tree accepts the adapter's weights once the padding tokens exist.
  plain_state_dict = dict(model.state_dict())
  if padding_mode == ZERO_MASKED_PADDING:
    plain_state_dict["x_pad_token"] = base.x_pad_token
    plain_state_dict["cap_pad_token"] = base.cap_pad_token
  base.load_state_dict(plain_state_dict, strict=True)


def test_reference_frames_are_ignored_without_ref_latents():
  model = build_transformer(ZERO_MASKED_PADDING)
  x, t, cap, _, _ = build_inputs(ref_frames=(2, 2))

  without_ref = model(x, t, cap)
  with_ref = model(x, t, cap, ref_latents=[torch.randn_like(item) for item in x])

  for plain, conditioned in zip(without_ref.sample, with_ref.sample):
    assert plain.shape == conditioned.shape
    assert not torch.equal(plain, conditioned)


def test_multi_frame_output_keeps_generated_frames():
  x, t, cap, ref, direct = build_inputs(frames=(2, 2), ref_frames=(1, 1))

  per_frame = build_transformer(LEARNED_PADDING, multi_frame_output=True)
  assert [item.shape[1] for item in per_frame(x, t, cap, ref_latents=ref).sample] == [2, 2]

  single_frame = build_transformer(ZERO_MASKED_PADDING, multi_frame_output=False)
  assert [item.shape[1] for item in single_frame(x, t, cap, ref_latents=ref).sample] == [1, 1]

  output = per_frame(x, t, cap, ref_latents=ref, direct_condition=direct, return_dict=False)
  assert isinstance(output, tuple) and len(output) == 1
  assert [item.shape[1] for item in output[0]] == [2, 2]


def test_batched_tensor_inputs_match_list_inputs():
  model = build_transformer(ZERO_MASKED_PADDING)
  x, t, cap, ref, direct = build_inputs(ref_frames=(2, 2), cap_lens=(5, 5), direct_lens=(3, 3))

  from_lists = model(x, t, cap, ref_latents=ref, direct_condition=direct)
  from_tensors = model(
    x,
    t,
    cap,
    ref_latents=torch.stack(ref),
    direct_condition=torch.stack(direct),
  )

  for listed, stacked in zip(from_lists.sample, from_tensors.sample):
    assert torch.equal(listed, stacked)

  single_x, single_t, single_cap, single_ref, single_direct = build_inputs(batch_size=1,
                                                                           frames=(1, ),
                                                                           ref_frames=(1, ),
                                                                           cap_lens=(5, ),
                                                                           direct_lens=(3, ))
  single_out = model(
    single_x,
    single_t,
    single_cap,
    ref_latents=single_ref[0],
    direct_condition=single_direct[0],
  )
  assert single_out.sample[0].shape == (CHANNELS, 1, HEIGHT, WIDTH)


def test_invalid_inputs_raise():
  model = build_transformer(ZERO_MASKED_PADDING)
  x, t, cap, ref, direct = build_inputs()

  with pytest.raises(ValueError, match="batch size"):
    model(x, t, cap, ref_latents=ref[:1])
  with pytest.raises(ValueError, match="incompatible"):
    model(x, t, cap, ref_latents=[torch.randn(CHANNELS, 1, HEIGHT // 2, WIDTH) for _ in x])
  with pytest.raises(ValueError, match="hidden space"):
    model(x, t, cap, direct_condition=[torch.randn(2, CAP_DIM) for _ in x])
  with pytest.raises(ValueError, match="dims"):
    model(x, t, cap, direct_condition=[torch.randn(2, 2, CAP_DIM) for _ in x])
  with pytest.raises(ValueError, match="nested"):
    model([[item] for item in x], t, cap)
  with pytest.raises(ValueError, match="nested"):
    model(x, t, [[item] for item in cap])


@pytest.mark.parametrize("padding_mode", [ZERO_MASKED_PADDING, LEARNED_PADDING])
def test_save_and_load_roundtrip(tmp_path, padding_mode):
  model = build_transformer(padding_mode, multi_frame_output=True)
  model.save_pretrained(tmp_path)

  reloaded = MingImageTransformer2DModel.from_pretrained(tmp_path)
  assert set(reloaded.state_dict().keys()) == set(model.state_dict().keys())
  assert reloaded.config.alignment_padding_mode == padding_mode
  assert reloaded.config.multi_frame_output is True
  assert isinstance(reloaded.x_pad_token, torch.nn.Parameter) == (padding_mode == LEARNED_PADDING)

  x, t, cap, ref, direct = build_inputs()
  for sample, reference in zip(
      reloaded(x, t, cap, ref_latents=ref, direct_condition=direct).sample,
      model(x, t, cap, ref_latents=ref, direct_condition=direct).sample,
  ):
    assert torch.equal(sample, reference)


def test_regional_compile_keeps_direct_hook_outside_repeated_blocks():
  model = build_transformer(ZERO_MASKED_PADDING)
  cap_embedder = model.cap_embedder
  x, t, cap, _, direct = build_inputs()

  model.compile_repeated_blocks()
  sample = model(x, t, cap, direct_condition=direct).sample

  assert model.cap_embedder is cap_embedder
  assert len(sample) == len(x)
  assert all(torch.isfinite(item).all() for item in sample)
