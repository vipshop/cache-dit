"""Unit tests for the Ming-Image MLLM condition extractor (tiny configs, CPU only)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from cache_dit.models.pipelines.ming_image.mllm import (
  MingImageMLLM,
  MingMllmInputs,
  MingMoeBlock,
  MingRouter,
  apply_3d_rotary_pos_emb,
  expand_image_placeholder,
  format_generation_prompt,
  load_ming_mllm_weights,
  ming_rope_position_ids,
)

HIDDEN = 32
HEAD_DIM = 16
NUM_LAYERS = 4
NUM_QUERY = 16
PATCH_ID = 60
START_ID = 61
END_ID = 62
EOS_ID = 59

#: Atomic token strings the stub tokenizer must resolve to the checkpoint's own ids.
_SPECIAL_TOKENS = (
  ("<imagePatch>", PATCH_ID),
  ("<image>", START_ID),
  ("</image>", END_ID),
  ("<|role_end|>", EOS_ID),
)


def tiny_config():
  llm_config = {
    "hidden_size": HIDDEN,
    "intermediate_size": 40,
    "moe_intermediate_size": 8,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "num_shared_experts": 1,
    "n_group": 2,
    "topk_group": 1,
    "routed_scaling_factor": 2.5,
    "num_hidden_layers": NUM_LAYERS,
    "first_k_dense_replace": 1,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": HEAD_DIM,
    "partial_rotary_factor": 0.5,
    "rope_theta": 600000.0,
    "rms_norm_eps": 1e-6,
    "hidden_act": "silu",
    "use_qkv_bias": False,
    "use_bias": False,
    "vocab_size": 64,
    "image_patch_token": PATCH_ID,
    "image_start_token": START_ID,
    "image_end_token": END_ID,
    "video_patch_token": 63,
    # sum(mrope_section) must stay head_dim * partial_rotary_factor / 2, as in the checkpoint.
    "rope_scaling": {
      "type": "video_rope",
      "mrope_section": [1, 2, 1]
    },
  }
  vision_config = {
    "hidden_size": 32,
    "num_heads": 4,
    "intermediate_size": 64,
    "depth": 2,
    "in_chans": 3,
    "out_hidden_size": 16,
    "patch_size": 14,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "hidden_act": "silu",
    "fullatt_block_indexes": [1],
    "window_size": 112,
  }
  return {"llm_config": llm_config, "vision_config": vision_config, "mlp_depth": 2}


def tiny_model(capture_layers=(1, 2, NUM_LAYERS)):
  torch.manual_seed(0)
  return MingImageMLLM(tiny_config(), img_gen_scales=(4, ), capture_layers=capture_layers).eval()


def test_te_p_plan_paths_resolve_on_the_real_tiny_tree(monkeypatch):
  from cache_dit.distributed.text_encoders.ming_image import (
    MingImageMLLMTensorParallelismPlanner, )

  model = tiny_model()
  captured = []

  def fake_parallelize_module(*, module, device_mesh, parallelize_plan):
    captured.append((module, parallelize_plan))

  monkeypatch.setattr("cache_dit.distributed.text_encoders.ming_image.parallelize_module",
                      fake_parallelize_module)
  monkeypatch.setattr("cache_dit.distributed.text_encoders.ming_image.maybe_empty_cache",
                      lambda: None)
  mesh = SimpleNamespace(get_group=lambda: SimpleNamespace(size=lambda: 2))

  MingImageMLLMTensorParallelismPlanner().parallelize_text_encoder(model, mesh)

  assert captured
  for block, plan in captured:
    for path in plan:
      block.get_submodule(path)


class StubTokenizer:
  """Character-level tokenizer stub that resolves the Ming structural tokens atomically."""

  eos_token = "<|role_end|>"

  def __call__(self, text, return_tensors=None, add_special_tokens=False):
    ids = []
    index = 0
    while index < len(text):
      for token, token_id in _SPECIAL_TOKENS:
        if text.startswith(token, index):
          ids.append(token_id)
          index += len(token)
          break
      else:
        ids.append(1 + ord(text[index]) % 50)
        index += 1
    return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class StubImageProcessor:
  """Image processor stub reporting a fixed ``(1, 4, 4)`` grid for any image."""

  def __call__(self, images=None, return_tensors=None):
    return {
      "pixel_values": torch.zeros(16, 3 * 2 * 14 * 14),
      "image_grid_thw": torch.tensor([[1, 4, 4]]),
    }


def attach_stubs(model):
  model.tokenizer = StubTokenizer()
  model.image_processor = StubImageProcessor()
  return model


def test_format_generation_prompt_matches_vendor_template():
  text = format_generation_prompt("draw a poster", "<|role_end|>")
  assert text.startswith("<role>SYSTEM</role>你是一个友好的AI助手。\n\ndetailed thinking off")
  assert text.endswith(
    "<|role_end|><role>HUMAN</role>draw a poster<|role_end|><role>ASSISTANT</role>")
  with_image = format_generation_prompt("edit", "<|role_end|>", has_reference_image=True)
  assert "<role>HUMAN</role><IMAGE>edit" in with_image
  custom = format_generation_prompt("x", "<eos>", system_prompt="be terse")
  assert "be terse\n\ndetailed thinking off" in custom


def test_expand_image_placeholder_counts_merged_patches():
  text = expand_image_placeholder("<IMAGE>hi", torch.tensor([1, 4, 4]), 2)
  assert text.count("<imagePatch>") == 4
  assert text.startswith("<image>" + "<imagePatch>" * 4 + "</image>\nhi")


def test_router_matches_reference_group_limited_topk():
  torch.manual_seed(3)
  router = MingRouter(
    hidden_size=8,
    num_experts=8,
    top_k=2,
    n_group=2,
    topk_group=1,
    routed_scaling_factor=2.5,
  )
  with torch.no_grad():
    router.expert_bias.uniform_(-0.1, 0.1)
  hidden = torch.randn(5, 8)

  index, weight = router(hidden)

  scores = torch.sigmoid(hidden.float() @ router.weight.float().T)
  biased = scores + router.expert_bias.float()
  group_scores = biased.view(5, 2, -1).topk(2, dim=-1)[0].sum(-1)
  keep = torch.zeros_like(group_scores).scatter_(1,
                                                 group_scores.topk(1, dim=-1, sorted=False)[1], 1)
  mask = keep.unsqueeze(-1).expand(5, 2, 4).reshape(5, -1)
  masked = biased.masked_fill(~mask.bool(), float("-inf"))
  _, reference_index = masked.topk(2, dim=-1, sorted=False)
  assert torch.equal(torch.sort(index, -1).values, torch.sort(reference_index, -1).values)

  reference_weight = scores.gather(1, reference_index)
  reference_weight = reference_weight / reference_weight.sum(-1, keepdim=True) * 2.5
  for row in range(5):
    expected = {
      int(reference_index[row, column]): float(reference_weight[row, column])
      for column in range(2)
    }
    for column in range(2):
      expert = int(index[row, column])
      assert expert in expected
      assert weight[row, column].item() == pytest.approx(expected[expert], rel=1e-5)


def test_router_keeps_top_k_experts_inside_selected_groups():
  torch.manual_seed(11)
  router = MingRouter(hidden_size=8,
                      num_experts=8,
                      top_k=2,
                      n_group=2,
                      topk_group=1,
                      routed_scaling_factor=1.0)
  index, weight = router(torch.randn(6, 8))
  assert index.dtype == torch.int64
  assert index.shape == (6, 2)
  assert torch.allclose(weight.sum(-1), torch.ones(6), atol=1e-4)
  for row in range(6):
    groups = {int(expert) // 4 for expert in index[row]}
    assert len(groups) == 1


def test_moe_block_routes_patch_tokens_through_image_gate():
  torch.manual_seed(5)
  block = MingMoeBlock(tiny_model().llm_config)
  hidden = torch.randn(1, 3, HIDDEN)
  mask = torch.tensor([[False, True, False]])

  output = block(hidden, mask)

  flat = hidden.reshape(-1, HIDDEN)
  text_index, text_weight = block.gate(flat)
  image_index, image_weight = block.image_gate(flat)
  index = torch.where(mask.reshape(-1, 1), image_index, text_index)
  weight = torch.where(mask.reshape(-1, 1), image_weight, text_weight)
  expected = []
  for token in range(3):
    accumulator = torch.zeros(HIDDEN)
    for slot in range(index.shape[1]):
      expert = block.experts[int(index[token, slot])]
      accumulator = accumulator + expert(flat[token:token + 1])[0] * weight[token, slot]
    expected.append(accumulator + block.shared_experts(flat[token:token + 1])[0])
  assert torch.allclose(output, torch.stack(expected).view(1, 3, HIDDEN), atol=1e-6)

  # A token without the patch mask ignores the image router entirely.
  with torch.no_grad():
    block.image_gate.weight.add_(1.0)
  assert torch.allclose(block(hidden, mask)[0, 0], output[0, 0], atol=1e-6)
  assert not torch.allclose(block(hidden, mask)[0, 1], output[0, 1], atol=1e-6)


def test_capture_layers_follow_hidden_states_semantics():
  model = attach_stubs(tiny_model(capture_layers=(1, NUM_LAYERS)))
  inputs = model.prepare_inputs("hello world")
  embeds = model.build_inputs_embeds(inputs)
  final, captured = model.forward(inputs)

  trunk = model.model.model
  cos, sin = trunk.rotary_emb(embeds, inputs.position_ids)
  manual = embeds
  for layer_idx, layer in enumerate(trunk.layers):
    if layer_idx == 1:
      assert torch.allclose(captured[1], manual, atol=1e-6)
    manual = layer(manual, cos, sin, None, inputs.image_mask)
  manual = trunk.norm(manual)
  assert torch.allclose(final, manual, atol=1e-6)
  assert torch.allclose(captured[NUM_LAYERS], final, atol=1e-6)


def test_query_block_injection_uses_image_features_then_learnable_tokens():
  model = attach_stubs(tiny_model())
  stub_features = torch.arange(4 * HIDDEN, dtype=torch.float32).view(4, HIDDEN) / 100.0
  model._image_features = lambda pixel_values, grid_thw: stub_features
  inputs = model.prepare_inputs("a prompt", reference_image=object())
  embeds = model.build_inputs_embeds(inputs)

  patch_positions = inputs.input_ids[0] == PATCH_ID
  assert int(patch_positions.sum()) == 4 + NUM_QUERY
  expected = torch.cat([stub_features, model.query_block()], dim=0)
  assert torch.allclose(embeds[0][patch_positions], expected, atol=1e-6)
  plain = model.model.model.word_embeddings(inputs.input_ids)[0]
  assert torch.allclose(embeds[0][~patch_positions], plain[~patch_positions], atol=1e-6)


def test_query_block_injection_rejects_patch_count_mismatch():
  model = attach_stubs(tiny_model())
  model._image_features = lambda pixel_values, grid_thw: torch.zeros(3, HIDDEN)
  inputs = model.prepare_inputs("a prompt", reference_image=object())
  with pytest.raises(ValueError, match="patch mismatch"):
    model.build_inputs_embeds(inputs)


def test_prepare_inputs_appends_query_suffix_and_positions():
  model = attach_stubs(tiny_model())
  inputs = model.prepare_inputs("abc")
  suffix = [START_ID] + [PATCH_ID] * NUM_QUERY + [END_ID]
  total = int(inputs.input_ids.shape[1])
  assert inputs.input_ids[0, -len(suffix):].tolist() == suffix
  assert inputs.prompt_length == total - len(suffix)
  assert inputs.query_start == total - 1 - NUM_QUERY

  positions = inputs.position_ids
  assert positions.shape == (3, 1, total)
  text = positions[:, 0, :inputs.prompt_length]
  assert torch.equal(text, torch.arange(inputs.prompt_length).expand(3, -1))
  # The query block behaves as a 1 x (2 * 16) pseudo image; the preceding <image> token keeps the
  # text position, so the block starts at prompt_length + 1.
  block = positions[:, 0, inputs.query_start:inputs.query_start + NUM_QUERY]
  block_start = inputs.prompt_length + 1
  assert torch.equal(block[0], torch.full((NUM_QUERY, ), float(block_start)))
  assert torch.equal(block[1], torch.full((NUM_QUERY, ), float(block_start)))
  expected_w = block_start + torch.arange(NUM_QUERY) - (NUM_QUERY - 1) // 2
  assert torch.equal(block[2], expected_w)
  assert int(inputs.image_mask.sum()) == NUM_QUERY
  assert int(inputs.image_mask[0, :inputs.prompt_length].sum()) == 0


def test_reference_image_positions_and_patch_mask():
  model = attach_stubs(tiny_model())
  inputs = model.prepare_inputs("abc", reference_image=object())
  assert inputs.image_grid_thw.tolist() == [[1, 4, 4]]
  assert int(inputs.image_mask.sum()) == 4 + NUM_QUERY

  first_patch = inputs.input_ids[0].tolist().index(PATCH_ID)
  text_len = first_patch  # the <image> token occupies first_patch - 1
  block = inputs.position_ids[:, 0, first_patch:first_patch + 4]
  assert torch.equal(block[0], torch.full((4, ), float(text_len)))
  assert block[1].tolist() == [text_len, text_len, text_len + 1, text_len + 1]
  assert block[2].tolist() == [text_len, text_len + 1, text_len, text_len + 1]
  # Text after the block resumes at max(t) + 1.
  assert int(inputs.position_ids[0, 0, first_patch + 4]) == text_len + 1


def test_rope_video_layout_keeps_tail_unrotated():
  torch.manual_seed(7)
  head_dim, rotary_dim = HEAD_DIM, HEAD_DIM // 2
  cos = torch.randn(3, 1, 3, head_dim)
  sin = torch.randn(3, 1, 3, head_dim)
  query = torch.randn(1, 2, 3, head_dim)
  key = torch.randn(1, 1, 3, head_dim)
  rotated_query, rotated_key = apply_3d_rotary_pos_emb(query,
                                                       key,
                                                       cos,
                                                       sin,
                                                       mrope_section=(1, 2, 1),
                                                       rope_type="video_rope")
  assert rotated_query.shape == query.shape
  assert rotated_key.shape == key.shape
  assert torch.allclose(rotated_query[..., rotary_dim:], query[..., rotary_dim:])
  assert not torch.allclose(rotated_query[..., :rotary_dim], query[..., :rotary_dim])


def test_extract_shapes_and_direct_feature_dimension():
  model = attach_stubs(tiny_model())
  inputs = model.prepare_inputs("a tiny prompt")
  conditions = model.extract("a tiny prompt")
  assert conditions.query_hidden_states.shape == (1, NUM_QUERY, HIDDEN)
  assert conditions.direct_hidden_states.shape == (1, inputs.prompt_length, 3 * HIDDEN)
  assert torch.isfinite(conditions.query_hidden_states).all()
  assert torch.isfinite(conditions.direct_hidden_states).all()
  # The direct condition excludes the appended query block.
  hidden_states, _ = model.forward(inputs)
  assert torch.allclose(
    conditions.query_hidden_states,
    hidden_states[:, inputs.query_start:inputs.query_start + NUM_QUERY],
    atol=1e-6,
  )


def test_direct_condition_channels_follow_capture_layer_order():
  model = attach_stubs(tiny_model(capture_layers=(1, 2, NUM_LAYERS)))
  inputs = model.prepare_inputs("abc")
  _, captured = model.forward(inputs)
  conditions = model.extract("abc")

  assert model.direct_hidden_size == 3 * HIDDEN
  assert conditions.direct_hidden_states.shape[-1] == model.direct_hidden_size
  prompt_length = inputs.prompt_length
  for slot, layer in enumerate((1, 2, NUM_LAYERS)):
    block = conditions.direct_hidden_states[..., slot * HIDDEN:(slot + 1) * HIDDEN]
    assert torch.equal(block, captured[layer][:, :prompt_length])

  # The channel order is the capture_layers order, not the ascending layer order.
  reordered = attach_stubs(tiny_model(capture_layers=(2, 1, NUM_LAYERS)))
  reordered_conditions = reordered.extract("abc")
  assert torch.equal(reordered_conditions.direct_hidden_states[..., :HIDDEN],
                     captured[2][:, :prompt_length])
  assert torch.equal(
    reordered_conditions.direct_hidden_states[..., HIDDEN:2 * HIDDEN],
    captured[1][:, :prompt_length],
  )


def test_encode_matches_extract():
  model = attach_stubs(tiny_model())
  query, direct = model.encode("prompt")
  conditions = model.extract("prompt")
  assert torch.equal(query, conditions.query_hidden_states)
  assert torch.equal(direct, conditions.direct_hidden_states)


def test_extract_with_reference_image_runs_vision_tower():
  model = attach_stubs(tiny_model())
  conditions = model.extract("prompt", reference_image=object())
  assert conditions.query_hidden_states.shape == (1, NUM_QUERY, HIDDEN)
  assert conditions.direct_hidden_states.shape[2] == 3 * HIDDEN
  assert torch.isfinite(conditions.query_hidden_states).all()


def _write_checkpoint(directory, state, extra=None):
  payload = {name: tensor.clone() for name, tensor in state.items()}
  payload.update(extra or {})
  directory.mkdir(parents=True, exist_ok=True)
  save_file(payload, str(directory / "model.safetensors"))


def test_weight_loading_accounts_for_every_checkpoint_key(tmp_path):
  source = tiny_model()
  state = source.state_dict()
  skipped_state = {
    "model.lm_head.weight": torch.zeros(4, HIDDEN),
    "model.model.layers.1.mlp.audio_gate.weight": torch.zeros(4, HIDDEN),
    "model.model.layers.1.mlp.audio_gate.expert_bias": torch.zeros(4),
  }
  _write_checkpoint(tmp_path, state, skipped_state)

  target = tiny_model()
  with torch.no_grad():
    for parameter in target.parameters():
      parameter.zero_()
  skipped = load_ming_mllm_weights(target, tmp_path)

  assert set(skipped) == set(skipped_state)
  for name, tensor in target.state_dict().items():
    if name.startswith("query_tokens_dict."):
      continue
    assert torch.equal(tensor, state[name]), name


def test_weight_loading_rejects_unknown_keys(tmp_path):
  _write_checkpoint(tmp_path, tiny_model().state_dict(), {"model.mystery.weight": torch.zeros(2)})
  with pytest.raises(KeyError, match="mystery"):
    load_ming_mllm_weights(tiny_model(), tmp_path)


def test_weight_loading_rejects_shape_mismatch(tmp_path):
  state = tiny_model().state_dict()
  state["model.model.layers.0.mlp.gate_proj.weight"] = torch.zeros(3, 3)
  _write_checkpoint(tmp_path, state)
  with pytest.raises(ValueError, match="shape mismatch"):
    load_ming_mllm_weights(tiny_model(), tmp_path)


def test_weight_loading_rejects_missing_parameters(tmp_path):
  state = {
    name: tensor
    for name, tensor in tiny_model().state_dict().items() if "layers.0.input_layernorm" not in name
  }
  _write_checkpoint(tmp_path, state)
  with pytest.raises(ValueError, match="missing"):
    load_ming_mllm_weights(tiny_model(), tmp_path)


def test_capture_layer_range_is_validated():
  with pytest.raises(ValueError, match="outside"):
    MingImageMLLM(tiny_config(), img_gen_scales=(4, ), capture_layers=(1, NUM_LAYERS + 1))


def test_rope_section_must_cover_the_partial_rotary_width():
  config = tiny_config()
  config["llm_config"]["rope_scaling"] = {"type": "video_rope", "mrope_section": [2, 4, 2]}
  with pytest.raises(ValueError, match="rotary dims"):
    MingImageMLLM(config, img_gen_scales=(4, ), capture_layers=(1, ))


def test_query_scale_contract_of_the_released_checkpoint():
  # ming/../mlp/config.json ships img_gen_scales=[16] -> a 256-token block and 258 appended ids.
  model = MingImageMLLM(tiny_config(), img_gen_scales=(16, ), capture_layers=(1, 2))
  # Cross-module contract: conditioning compares this tuple against its own copy, so it must stay a
  # plain immutable tuple of ints (JSON ints, never tensors).
  assert isinstance(model.capture_layers, tuple)
  assert all(isinstance(layer, int) for layer in model.capture_layers)
  assert isinstance(model.num_query_tokens, int)
  assert isinstance(model.direct_hidden_size, int)
  assert model.direct_hidden_size == len(model.capture_layers) * HIDDEN
  assert model.num_query_tokens == 256
  assert model.suffix_length == 258
  assert model.query_suffix_ids()[0] == START_ID
  assert model.query_suffix_ids()[-1] == END_ID
  assert model.query_block().shape == (256, HIDDEN)
  assert torch.allclose(model.query_block().norm(dim=-1), torch.ones(256), atol=1e-4)


def test_forward_without_patch_tokens_skips_multimodal_path():
  model = tiny_model()
  input_ids = torch.randint(1, 50, (1, 5), dtype=torch.long)
  inputs = MingMllmInputs(
    input_ids=input_ids,
    attention_mask=torch.ones_like(input_ids),
    position_ids=torch.arange(5).view(1, 1, -1).expand(3, 1, -1).clone(),
    image_mask=torch.zeros_like(input_ids, dtype=torch.bool),
    pixel_values=None,
    image_grid_thw=None,
    prompt_length=5,
    query_start=0,
  )
  final, captured = model.forward(inputs)
  assert final.shape == (1, 5, HIDDEN)
  assert set(captured) == {1, 2, NUM_LAYERS}
  assert model.build_inputs_embeds(inputs).shape == (1, 5, HIDDEN)


def test_module_exposes_dtype_and_device_for_diffusers_pipelines():
  model = tiny_model()
  assert model.dtype == torch.float32
  assert model.device == torch.device("cpu")
  model = model.to(dtype=torch.bfloat16)
  assert model.dtype == torch.bfloat16
  assert model.device == torch.device("cpu")


def test_padded_inputs_get_a_keep_mask():
  model = tiny_model()
  assert model._causal_keep_mask(torch.ones(1, 4, dtype=torch.long)) is None
  mask = model._causal_keep_mask(torch.tensor([[1, 1, 0, 0]]))
  assert mask.shape == (1, 1, 4, 4)
  assert not bool(mask[0, 0, :, 2:].any())


def test_rope_positions_are_monotonic_for_text_only_prompts():
  input_ids = torch.tensor([[5, 6, PATCH_ID, PATCH_ID, 7]])
  position_ids = ming_rope_position_ids(
    input_ids,
    None,
    image_token_id=PATCH_ID,
    image_start_token_id=START_ID,
    spatial_merge_size=2,
  )
  # No <image> start token, so every token is plain text.
  assert torch.equal(position_ids[0, 0], torch.arange(5))
  assert torch.equal(position_ids[0], position_ids[2])
