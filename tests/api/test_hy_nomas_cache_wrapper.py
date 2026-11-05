"""Contract tests that only make sense against cache-dit's real block wrapper.

Three silent failure modes are locked here, because each one leaves the numbers looking healthy:

* a block that returns a single tensor keeps the *previous* ``encoder_hidden_states`` for every
  later block (``_process_block_outputs`` only unpacks non-tensor returns), so the text stream
  silently freezes — ``check_num_outputs=True`` is what catches it, and it is **off by default**;
* a block-skip wrapper cache-dit does not recognise must fail loudly on the K/V pass, not degrade
  into ``commit``'s ``TypeError``;
* the attention plan's mask must match an independently written dense reference, otherwise a wrong
  visibility row still produces plausible pixels.
"""
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from cache_dit.caching.block_adapters import BlockAdapter
from cache_dit.caching.cache_blocks.pattern_0_1_2 import CachedBlocks_Pattern_0_1_2
from cache_dit.caching.cache_contexts.cache_manager import CachedContextManager
from cache_dit.caching.forward_pattern import ForwardPattern
from cache_dit.models.pipelines.hy_nomas import NomasTransformer3DModel
from cache_dit.models.pipelines.hy_nomas.attention import NomasParallelInfo, ulysses_all_to_all
from cache_dit.models.pipelines.hy_nomas.kv_cache import NomasAttentionPlan

HIDDEN = 32
HEADS = 2
LAYERS = 2
TEXT_DIM = 16


def _tiny_model():
  torch.manual_seed(0)
  model = NomasTransformer3DModel(hidden_size=HIDDEN,
                                  num_attention_heads=HEADS,
                                  num_layers=LAYERS,
                                  text_states_dim=TEXT_DIM,
                                  rope_dim_list=(4, 6, 6),
                                  attention_backend="sdpa").eval().requires_grad_(False)
  with torch.no_grad():
    for name, param in model.named_parameters():
      if param.ndim == 1 and "norm" in name and name.endswith("weight"):
        param.fill_(1)
      else:
        param.normal_(0, 0.02)
  return model


def _inputs(model, cache, seed=2):
  generator = torch.Generator().manual_seed(seed)
  text_states = torch.randn(1, 6, TEXT_DIM, generator=generator)
  text_mask = (torch.arange(6)[None] < 5)
  latent = torch.randn(1, 32, 2, 2, 2, generator=generator)
  return latent, text_states, torch.tensor([500.0]), text_mask


class _OneOutputBlock(nn.Module):
  """A block whose annotation claims a single output; Pattern_0 must reject it."""

  def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor,
              ctx) -> torch.Tensor:
    return hidden_states


class _WrapperWithoutBlocks(nn.Module):
  """A block list element cache-dit cannot look through."""

  def forward(self, hidden_states, encoder_hidden_states, ctx):
    return hidden_states, encoder_hidden_states


def _wrapper(model, blocks=None):
  return CachedBlocks_Pattern_0_1_2(
    transformer_blocks=model.double_blocks if blocks is None else blocks,
    transformer=model,
    forward_pattern=ForwardPattern.Pattern_0,
    check_forward_pattern=True,
    check_num_outputs=True,
    cache_prefix="double_blocks",
    cache_context="hy_nomas_contract",
    context_manager=CachedContextManager(),
  )


def test_block_adapter_flags_default_to_off():
  # Documents the trap: a non-diffusers transformer silently gets no pattern validation unless the
  # adapter passes these explicitly (the CLI example / adapter will).
  adapter = BlockAdapter(pipe=None, transformer=None, blocks=None)
  assert adapter.check_num_outputs is False


def test_wrapper_rejects_single_output_block():
  model = _tiny_model()
  with pytest.raises(AssertionError, match="number of block's outputs"):
    _wrapper(model, blocks=[_OneOutputBlock()])


def test_real_wrapper_updates_both_streams():
  model = _tiny_model()
  cache = model.allocate_cache(64)
  latent, text_states, timestep, text_mask = _inputs(model, cache)
  wrapper = _wrapper(model)
  with torch.inference_mode():
    img = model.img_in(latent)
    txt = model.txt_in(text_states, timestep, text_mask)
    vec = model.time_in(timestep)
    context = model.prepare_segment_context(latent, text_mask, cache)
    from cache_dit.models.pipelines.hy_nomas.transformer import NomasBlockContext

    ctx = NomasBlockContext(vec=vec,
                            rope=context[1],
                            valid_length=cache.valid_length,
                            frame_offset=cache.frame_offset,
                            attention=context[0],
                            cache=cache,
                            camera_embedding=None,
                            camera_has_mask=None,
                            spatial_tokens=4,
                            token_offset=0,
                            collect_cache=False,
                            entries=None,
                            parallel=model.parallel)
    img_out, txt_out = wrapper(img, txt, ctx)
  # BOTH streams must move: a frozen text stream is the silent failure of a one-output block.
  assert img_out.shape == img.shape and txt_out.shape == txt.shape
  assert not torch.equal(img_out, img)
  assert not torch.equal(txt_out, txt)


def test_block_list_rejects_unrecognised_wrapper():
  model = _tiny_model()
  model.double_blocks = nn.ModuleList([_WrapperWithoutBlocks()])
  with pytest.raises(RuntimeError, match="no `transformer_blocks` list"):
    model._block_list(force_full_compute=True)


def test_block_list_rejects_short_block_list():
  model = _tiny_model()
  model.double_blocks = nn.ModuleList([_wrapper(model, blocks=[model.double_blocks[0]])])
  with pytest.raises(RuntimeError, match="expected 2"):
    model._block_list(force_full_compute=True)


def test_collect_reports_missing_entries():
  model = _tiny_model()
  cache = model.allocate_cache(64)
  latent, text_states, timestep, text_mask = _inputs(model, cache)
  # Simulate a replay that stops after one block: the collected entries must fail loudly.
  original = model._block_list
  model._block_list = lambda force_full_compute=False: [model.double_blocks[0]]
  try:
    with pytest.raises(RuntimeError, match="skipped"):
      with torch.inference_mode():
        model.predict(latent, timestep, text_states, text_mask, cache=cache, collect_cache=True)
  finally:
    model._block_list = original


def test_set_parallel_rejects_context_parallel():
  model = _tiny_model()
  with pytest.raises(NotImplementedError, match="not wired yet"):
    model.set_parallel(NomasParallelInfo(sp_size=2))


def test_set_parallel_owns_head_metadata():
  model = _tiny_model()
  model.set_parallel(NomasParallelInfo(tp_size=2))
  assert model.local_attention_heads == HEADS // 2
  assert model.double_blocks[0].heads == HEADS // 2
  assert model.double_blocks[0].attn_heads == HEADS // 2
  # The cache must follow the shard geometry: reverting `local_heads` to `num_attention_heads`
  # just fails here, and `head_dim` must not move with the shard count.
  cache = model.allocate_cache(64)
  assert cache.layers[0]["k"].shape[2] == HEADS // 2
  assert cache.layers[0]["k"].shape[3] == HIDDEN // HEADS
  with pytest.raises(ValueError, match="not divisible"):
    model.set_parallel(NomasParallelInfo(tp_size=3))


def test_ulysses_rejects_indivisible_heads():
  x = torch.zeros(1, 4, HEADS, 8)
  with pytest.raises(ValueError, match="not divisible by the context-parallel degree"):
    ulysses_all_to_all(x, NomasParallelInfo(sp_size=3, group=object()), 2, 1)


def _dense_reference(q, k, v, key_valid, query_valid):
  mask = key_valid[None, None, None, :].expand(1, 1, q.shape[1], -1).to(torch.bool)
  out = F.scaled_dot_product_attention(q.transpose(1, 2),
                                       k.transpose(1, 2),
                                       v.transpose(1, 2),
                                       attn_mask=mask)
  return out.transpose(1, 2) * query_valid[:, :, None, None]


@pytest.mark.parametrize("history_length", [0, 3])
def test_plan_matches_dense_reference(history_length):
  torch.manual_seed(3)
  frames, spatial = 2, 2
  text_len, valid = 5, 3
  video_tokens = frames * spatial
  text_mask = (torch.arange(text_len)[None] < valid)
  history_mask = torch.ones(1, history_length, dtype=torch.bool)
  plan = NomasAttentionPlan(text_mask, history_mask, frames, spatial, backend="sdpa")
  cache = None
  if history_length:
    cache = {
      "k": torch.randn(1, history_length, HEADS, HIDDEN // HEADS),
      "v": torch.randn(1, history_length, HEADS, HIDDEN // HEADS)
    }
  q = torch.randn(1, video_tokens + text_len, HEADS, HIDDEN // HEADS)
  k = torch.randn(1, video_tokens + text_len, HEADS, HIDDEN // HEADS)
  v = torch.randn(1, video_tokens + text_len, HEADS, HIDDEN // HEADS)
  got = plan(q, k, v, cache, history_length)
  if history_length:
    k_full = torch.cat((cache["k"], k), 1)
    v_full = torch.cat((cache["v"], v), 1)
  else:
    k_full, v_full = k, v
  key_valid = torch.cat((history_mask, plan.query_valid), 1)[0]
  expected = _dense_reference(q, k_full, v_full, key_valid, plan.query_valid)
  torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)


def test_flex_backend_matches_sdpa_on_dense_case():
  """The default backend is `flex`; without an action plan it is a plain causal-free softmax."""
  if not torch.cuda.is_available():
    pytest.skip("FlexAttention needs CUDA")
  torch.manual_seed(6)
  frames, spatial, text_len = 2, 2, 4
  text_mask = (torch.arange(text_len)[None] < 3).cuda()
  history_mask = torch.zeros(1, 0, dtype=torch.bool, device="cuda")
  tokens = frames * spatial + text_len
  q, k, v = (torch.randn(1, tokens, HEADS, HIDDEN // HEADS, device="cuda", dtype=torch.bfloat16)
             for _ in range(3))
  ref = NomasAttentionPlan(text_mask, history_mask, frames, spatial, backend="sdpa")(q, k, v, None,
                                                                                     0)
  out = NomasAttentionPlan(text_mask, history_mask, frames, spatial, backend="flex")(q, k, v, None,
                                                                                     0)
  torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


def test_masked_self_attention_ignores_padded_keys():
  from cache_dit.models.pipelines.hy_nomas.transformer import masked_self_attention

  torch.manual_seed(4)
  length, valid, heads, dim = 6, 4, 2, 8
  mask = (torch.arange(length)[None] < valid)
  q = torch.randn(1, length, heads, dim)
  k = torch.randn(1, length, heads, dim)
  v = torch.randn(1, length, heads, dim)
  base = masked_self_attention(q, k, v, mask)
  poisoned_k = k.clone()
  poisoned_v = v.clone()
  poisoned_k[:, valid:] = 1e4
  poisoned_v[:, valid:] = 1e4
  poisoned = masked_self_attention(q, poisoned_k, poisoned_v, mask)
  # Padded keys must not enter the softmax: the valid rows are bit-identical.
  assert torch.equal(base[:, :valid], poisoned[:, :valid])
  assert torch.equal(poisoned[:, valid:], torch.zeros_like(poisoned[:, valid:]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="layerwise offload needs CUDA")
def test_layerwise_offload_keeps_camera_module_resident():
  """The camera module is referenced, not registered: prove offload leaves it on the device."""
  from cache_dit.offload.layerwise import layerwise_offload, remove_layerwise_offload

  device = torch.device("cuda")
  model = _tiny_model().to(device)
  cache = model.allocate_cache(64)
  latent, text_states, timestep, text_mask = _inputs(model, cache)
  latent, text_states, timestep, text_mask = (latent.to(device), text_states.to(device),
                                              timestep.to(device), text_mask.to(device))
  # The camera branch only runs when an embedding is supplied, and that branch is exactly the
  # non-registered reference this test has to watch.
  camera = model.embed_camera(torch.randn(1, (2 - 1) * 4 + 1, 4, 4, device=device))
  with torch.inference_mode():
    reference = model(latent,
                      text_states,
                      timestep,
                      text_mask=text_mask,
                      cache=cache,
                      camera_embedding=camera)

  handle = layerwise_offload(
    model,
    module_filter=lambda name, module: NomasTransformer3DModel.layerwise_offload_filter(
      name, module),
    onload_device=device,
    offload_device="cpu",
    async_transfer=True,
  )
  try:
    model.resident_to(device)
    # ① the reference is shared, and the camera weights are on the compute device.
    assert model.double_blocks[0].action is model.action_modules[0]
    assert model.action_modules[0].img_attn_q.weight.device.type == "cuda"
    keys = set(model.state_dict())

    seen = []

    def _record(module, args):
      seen.append(module.img_attn_q.weight.device.type)

    hook = model.double_blocks[0].action.register_forward_pre_hook(_record)
    try:
      with torch.inference_mode():
        out = model(latent,
                    text_states,
                    timestep,
                    text_mask=text_mask,
                    cache=cache,
                    camera_embedding=camera)
    finally:
      hook.remove()
    # ② the state dict is untouched, ③ the camera module stays resident during the streamed
    # forward, ④ bf16 output matches the non-offloaded reference.
    assert set(model.state_dict()) == keys
    assert seen and set(seen) == {"cuda"}
    torch.testing.assert_close(out, reference, atol=1e-2, rtol=1e-2)
  finally:
    remove_layerwise_offload(model)
    if handle is not None:
      handle = None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flash-attention parity needs CUDA")
def test_fa3_backend_matches_sdpa():
  torch.manual_seed(5)
  frames, spatial, text_len = 2, 2, 4
  text_mask = (torch.arange(text_len)[None] < 3)
  history_mask = torch.zeros(1, 0, dtype=torch.bool)
  shapes = (frames * spatial + text_len, HEADS, HIDDEN // HEADS)
  q, k, v = (torch.randn(1, *shapes, device="cuda", dtype=torch.bfloat16) for _ in range(3))
  ref = NomasAttentionPlan(text_mask.cuda(), history_mask.cuda(), frames, spatial,
                           backend="sdpa")(q, k, v, None, 0)
  out = NomasAttentionPlan(text_mask.cuda(), history_mask.cuda(), frames, spatial,
                           backend="fa3")(q, k, v, None, 0)
  # Both are exact softmax over the same partition here (no action rows), so bf16 rounding only.
  cosine = F.cosine_similarity(ref.flatten(1).float(), out.flatten(1).float(), dim=-1)
  assert float(cosine.min()) > 0.999
