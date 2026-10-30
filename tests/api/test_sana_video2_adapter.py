"""Regression tests for the SANA-Video 2.0 cache integration.

SANA-Video 2.0 ships a native (non-diffusers) checkpoint, so the transformer and pipeline live in
`cache_dit.models.pipelines.sana_video2`. Its block loop differs from every other model: the
Attention-Residual aggregation and the per-group flush run *inside* the block, and the depth buffers
survive across denoising steps. These tests pin down the adapter wiring, the group bookkeeping that
survives cache-dit's collapsed block list, and the buffer reuse that makes a cached step legal.

Usage:
  python -m pytest tests/api/test_sana_video2_adapter.py -q
"""
import pytest
import torch

from cache_dit.caching.block_adapters import BlockAdapterRegister
from cache_dit.caching.forward_pattern import ForwardPattern
from cache_dit.models.pipelines.sana_video2.transformer import (
  SanaVideo2Transformer3DModel,
  attend_buffer,
)
from cache_dit.models.pipelines.sana_video2.components import DepthRMSNorm

LATENT_CHANNELS = 4
HIDDEN = 192
DEPTH = 4
HEADS = 2
CAPTION_CHANNELS = 16
TEXT_TOKENS = 4
LINEAR_HEAD_DIM = 64
SOFTMAX_HEAD_DIM = 96
GROUP = 2


class _FakePipeline(torch.nn.Module):
  """Minimal stand-in for `SanaVideo2Pipeline`, only what the adapter reads."""

  def __init__(self, transformer):
    super().__init__()
    self.transformer = transformer


def _tiny_transformer(seed: int = 0,
                      depth: int = DEPTH,
                      group: int = GROUP,
                      hidden: int = HIDDEN) -> SanaVideo2Transformer3DModel:
  torch.manual_seed(seed)
  return SanaVideo2Transformer3DModel(
    in_channels=LATENT_CHANNELS,
    hidden_size=hidden,
    depth=depth,
    num_heads=HEADS,
    caption_channels=CAPTION_CHANNELS,
    model_max_length=TEXT_TOKENS,
    linear_head_dim=LINEAR_HEAD_DIM,
    softmax_head_dim=SOFTMAX_HEAD_DIM,
    attn_res_block_size=group,
    input_size=2,
  ).eval()


def _reference_forward(model, latents, timestep, text, mask):
  """Upstream `_forward_attn_res_inference`: the aggregation and the flush stay in the loop.

  `SanaVideo2Block.forward` moved both into the block so cache-dit can replay a single block call;
  this reference keeps the upstream shape (an `active_count` counter plus a `partial` carried by
  the loop) and reuses only the leaf sublayers, so comparing them proves the restructure is exact.
  """
  batch, _, frames, h_lat, w_lat = latents.shape
  x = model.x_embedder(latents.to(model.dtype))
  patch_f, patch_h, patch_w = model.patch_size
  f, h, w = frames // patch_f, h_lat // patch_h, w_lat // patch_w
  ropes = {
    "linear": model.rope_linear((f, h, w), latents.device),
    "softmax": model.rope_softmax((f, h, w), latents.device),
  }
  t = model.t_embedder(timestep.long().float().flatten())
  t0 = model.t_block(t).unflatten(0, timestep.shape)
  t = t.unflatten(0, timestep.shape)
  y = model.y_embedder(text.to(model.dtype)).reshape(batch, -1, model.hidden_size)
  if model.config.y_norm:
    y = model.attention_y_norm(y)
  rows = mask.reshape(batch, -1) if mask is not None else None
  groups = -(-model.depth // model.attn_res_block_size)
  values = x.new_empty(groups + 2, batch, x.shape[1], model.hidden_size)
  keys = torch.empty_like(values)
  key_norm = model.attn_res.key_norm
  values[0] = x
  keys[0] = key_norm(x.unsqueeze(0)).squeeze(0)
  partial, active = None, 1
  for block in model.blocks:
    hidden = attend_buffer(model.attn_res.attn_proj.weight, key_norm, values, keys, active, partial)
    delta = block.forward_attn_sublayer(hidden, y, t0, rows, ropes[block.attention_type], None)
    partial = delta if partial is None else partial + delta
    hidden = attend_buffer(model.attn_res.mlp_proj.weight, key_norm, values, keys, active, partial)
    partial = partial + block.forward_mlp_sublayer(hidden, t0)
    if (block.index + 1) % model.attn_res_block_size == 0 or block.index + 1 == model.depth:
      values[active] = partial
      keys[active] = key_norm(partial.unsqueeze(0)).squeeze(0)
      active += 1
      partial = None
  x = attend_buffer(model.attn_res.final_proj.weight, key_norm, values, keys, active, None)
  out = model.final_layer(x, t)
  return model._unpatchify(out, f, h, w)


def _inputs(seed: int = 1, frames: int = 2, size: int = 2):
  generator = torch.Generator().manual_seed(seed)
  latents = torch.randn(1, LATENT_CHANNELS, frames, size, size, generator=generator)
  text = torch.randn(1, TEXT_TOKENS, CAPTION_CHANNELS, generator=generator)
  mask = torch.ones(1, TEXT_TOKENS, dtype=torch.long)
  return latents, text, mask


def _run(transformer, latents, text, mask, timestep=None):
  if timestep is None:
    timestep = torch.full((latents.shape[0], ), 500.0)
  with torch.no_grad():
    return transformer(
      hidden_states=latents,
      timestep=timestep,
      encoder_hidden_states=text,
      encoder_attention_mask=mask,
      return_dict=False,
    )[0]


def test_adapter_wiring():
  transformer = _tiny_transformer()
  adapter = BlockAdapterRegister.get_adapter(transformer)
  assert adapter.blocks_name == "blocks"
  assert adapter.forward_pattern == ForwardPattern.Pattern_3
  assert adapter.has_separate_cfg is False
  assert adapter.check_forward_pattern is True
  assert isinstance(adapter.blocks[0], type(transformer.blocks[0]))


def test_repeated_blocks_declared():
  assert SanaVideo2Transformer3DModel._repeated_blocks == ["SanaVideo2Block"]


def test_group_boundaries_use_block_index():
  transformer = _tiny_transformer()
  pattern = [(block.is_group_start, block.is_group_end) for block in transformer.blocks]
  assert pattern == [(True, False), (False, True), (True, False), (False, True)]
  # `active_count` inside a block is derived from its static index, so collapsing the stack into a
  # single cached block (cache-dit's wrapper) cannot change it.
  assert [1 + block.index // GROUP for block in transformer.blocks] == [1, 1, 2, 2]


def test_buffers_survive_steps_without_leaking_state():
  transformer = _tiny_transformer()
  latents, text, mask = _inputs()
  first = _run(transformer, latents, text, mask)
  # A different input must not be influenced by the previous step's flushed groups.
  other_latents, other_text, other_mask = _inputs(seed=7)
  _run(transformer, other_latents, other_text, other_mask)
  again = _run(transformer, latents, text, mask)
  assert torch.equal(first, again)
  fresh = _tiny_transformer()
  fresh.load_state_dict(transformer.state_dict())
  assert torch.equal(first, _run(fresh, latents, text, mask))


def test_forward_is_deterministic_and_shape_preserving():
  transformer = _tiny_transformer()
  latents, text, mask = _inputs()
  out = _run(transformer, latents, text, mask)
  assert out.shape == latents.shape
  assert torch.equal(out, _run(transformer, latents, text, mask))


def test_attend_buffer_matches_explicit_aggregation():
  """The buffered aggregation must equal the autograd-free reference formulation."""
  torch.manual_seed(0)
  weight = torch.randn(1, 8)
  key_norm = DepthRMSNorm()
  values = torch.empty(3, 1, 2, 8)
  keys = torch.empty_like(values)
  base = torch.randn(1, 2, 8)
  partial = torch.randn(1, 2, 8)
  values[0] = base
  keys[0] = key_norm(base.unsqueeze(0)).squeeze(0)
  got = attend_buffer(weight, key_norm, values, keys, 1, partial)
  sources = torch.stack([base, partial], dim=0)
  logits = torch.einsum("d,nbtd->nbt", weight.squeeze(0), key_norm(sources))
  expected = torch.einsum("nbt,nbtd->btd", logits.softmax(0), sources)
  assert torch.allclose(got, expected, atol=1e-6)


@pytest.mark.parametrize("frames,size", [(2, 2), (3, 2)])
def test_geometry_agnostic(frames, size):
  transformer = _tiny_transformer()
  latents, text, mask = _inputs(frames=frames, size=size)
  assert _run(transformer, latents, text, mask).shape == latents.shape


@pytest.mark.parametrize("depth,group", [(4, 2), (5, 2), (6, 4), (3, 4), (8, 3)])
@pytest.mark.parametrize("batch", [1, 2])
def test_restructured_loop_matches_upstream_semantics(depth, group, batch):
  """The in-block aggregation equals the upstream counter-based loop, bit for bit.

  Covers the geometries that matter: an exact multiple of the group size, a tail group that ends
  early, `group > depth`, and a group size that does not divide the depth.
  """
  transformer = _tiny_transformer(depth=depth, group=group)
  latents, text, mask = _inputs(frames=2, size=2)
  latents = latents.expand(batch, -1, -1, -1, -1).contiguous()
  text = text.expand(batch, -1, -1).contiguous()
  mask = mask.expand(batch, -1).contiguous()
  timestep = torch.full((batch, ), 700.0)
  got = _run(transformer, latents, text, mask, timestep=timestep)
  with torch.no_grad():
    ref = _reference_forward(transformer, latents.clone(), timestep, text.clone(), mask.clone())
  assert torch.equal(got, ref), f"max abs diff {(got - ref).abs().max().item():.3e}"


def test_buffers_reallocate_when_the_batch_changes():
  transformer = _tiny_transformer()
  latents, text, mask = _inputs()
  _run(transformer, latents, text, mask)
  first_shape = tuple(transformer._attn_res_values.shape)
  doubled = latents.expand(2, -1, -1, -1, -1).contiguous()
  _run(transformer, doubled, text.expand(2, -1, -1).contiguous(), mask.expand(2, -1).contiguous())
  assert tuple(transformer._attn_res_values.shape)[1] == 2
  assert tuple(transformer._attn_res_values.shape)[0] == first_shape[0]
