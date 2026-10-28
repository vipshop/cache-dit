"""CPU tests for the Ming-Image conditioning module.

The module carries three contracts that are easy to regress: the connector must run
*non-causally* over the query tokens, the eight ``mlp/`` tensors must map onto exact parameter
names, and ``cap_feats`` must leave through the connector's final RMSNorm. The tests rebuild a tiny
checkpoint on disk so the strict loader is exercised end to end, without touching the 6GB release.
"""
import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen2Config, Qwen2Model

from cache_dit.models.pipelines.ming_image.conditioning import MingImageConditioning

MLLM_HIDDEN = 16
CONNECTOR_HIDDEN = 32
GRID = 2
NUM_QUERY_TOKENS = GRID * GRID
CAP_DIM = 24
INNER_DIM = 20
DIRECT_DIM = MLLM_HIDDEN * 3
CONNECTOR_LAYERS = 2


def _connector_config() -> Qwen2Config:
  return Qwen2Config(
    vocab_size=128,
    hidden_size=CONNECTOR_HIDDEN,
    intermediate_size=64,
    num_hidden_layers=CONNECTOR_LAYERS,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=64,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
  )


def _mlp_config() -> dict:
  return {
    "use_identity_mlp": True,
    "use_vlm_directvlm_condition": True,
    "use_learnable_token_condition": True,
    "diffusion_c_input_dim": CAP_DIM,
    "text_encoder_norm": False,
    "connector_norm": False,
    "img_gen_scales": [GRID],
    "selected_hidden_states_layers": [3, 7, 11],
    "diffusion_inner_dim": INNER_DIM,
  }


def _mlp_tensors(seed: int = 0) -> dict:
  generator = torch.Generator().manual_seed(seed)

  def rand(*shape):
    return torch.randn(*shape, generator=generator)

  return {
    "proj_in.weight": rand(CONNECTOR_HIDDEN, MLLM_HIDDEN),
    "proj_in.bias": rand(CONNECTOR_HIDDEN),
    "proj_out.weight": rand(CAP_DIM, CONNECTOR_HIDDEN),
    "proj_out.bias": rand(CAP_DIM),
    f"query_tokens_dict.{GRID}x{GRID}": rand(NUM_QUERY_TOKENS, MLLM_HIDDEN),
    "proj_directvlm.0.weight": rand(DIRECT_DIM),
    "proj_directvlm.1.weight": rand(INNER_DIM, DIRECT_DIM),
    "proj_directvlm.1.bias": rand(INNER_DIM),
  }


def _connector_tensors(seed: int = 1) -> dict:
  torch.manual_seed(seed)
  model = Qwen2Model(_connector_config())
  return {
    f"model.{name}": torch.randn_like(value) * 0.05
    for name, value in model.state_dict().items()
  }


def _write_checkpoint(root, mlp_state=None, connector_state=None) -> dict:
  """Write a tiny Ming-Image layout: connector/, mlp/, config.json."""
  mlp_state = _mlp_tensors() if mlp_state is None else mlp_state
  connector_state = _connector_tensors() if connector_state is None else connector_state
  (root / "connector").mkdir(parents=True, exist_ok=True)
  (root / "mlp").mkdir(parents=True, exist_ok=True)
  _connector_config().save_pretrained(root / "connector")
  (root / "mlp" / "config.json").write_text(json.dumps(_mlp_config()))
  (root / "config.json").write_text(json.dumps({"llm_config": {"hidden_size": MLLM_HIDDEN}}))
  save_file(connector_state, str(root / "connector" / "model-00001-of-00001.safetensors"))
  save_file(mlp_state, str(root / "mlp" / "model.safetensors"))
  return {"mlp": mlp_state, "connector": connector_state}


def _tiny_module(**kwargs) -> MingImageConditioning:
  return MingImageConditioning.from_config(
    _connector_config(),
    _mlp_config(),
    hidden_size=MLLM_HIDDEN,
    **kwargs,
  )


def _inputs(batch: int = 2, tokens: int = NUM_QUERY_TOKENS, seed: int = 3):
  generator = torch.Generator().manual_seed(seed)
  query = torch.randn(batch, tokens, MLLM_HIDDEN, generator=generator)
  direct = torch.randn(batch, 1, DIRECT_DIM, generator=generator)
  return query, direct


def test_forward_returns_zimage_cap_feats_and_direct_condition():
  module = _tiny_module().eval()
  query, direct = _inputs()
  with torch.no_grad():
    cap_feats, direct_condition = module(query, direct)
  assert isinstance(cap_feats, list) and len(cap_feats) == 2
  assert cap_feats[0].shape == (NUM_QUERY_TOKENS, CAP_DIM)
  assert direct_condition.shape == (2, 1, INNER_DIM)
  assert module.num_query_tokens == NUM_QUERY_TOKENS


def test_connector_attends_to_later_query_tokens():
  module = _tiny_module().eval()
  query, _ = _inputs(batch=1)
  perturbed = query.clone()
  perturbed[0, -1] += 5.0
  with torch.no_grad():
    first, _ = module(query, _inputs(batch=1)[1])
    second, _ = module(perturbed, _inputs(batch=1)[1])
  # A causal connector would leave the first query token untouched.
  assert (first[0][0] - second[0][0]).abs().max() > 1e-4

  tokens = NUM_QUERY_TOKENS
  causal = torch.zeros(tokens, tokens)
  causal.masked_fill_(torch.ones(tokens, tokens, dtype=torch.bool).triu(1), -3e4)
  causal = causal[None, None]
  with torch.no_grad():
    x = module.proj_in(query)
    hidden = module.connector(inputs_embeds=x, attention_mask=causal, use_cache=False)
    hidden_perturbed = module.connector(inputs_embeds=module.proj_in(perturbed),
                                        attention_mask=causal,
                                        use_cache=False)
  delta = hidden.last_hidden_state[0, 0] - hidden_perturbed.last_hidden_state[0, 0]
  assert delta.abs().max() < 1e-5


def test_from_pretrained_maps_every_checkpoint_tensor(tmp_path):
  state = _write_checkpoint(tmp_path)
  module = MingImageConditioning.from_pretrained(tmp_path).eval()
  mlp, connector = state["mlp"], state["connector"]

  assert torch.equal(module.proj_in.weight, mlp["proj_in.weight"])
  assert torch.equal(module.proj_in.bias, mlp["proj_in.bias"])
  assert torch.equal(module.proj_out.weight, mlp["proj_out.weight"])
  assert torch.equal(module.query_tokens, mlp[f"query_tokens_dict.{GRID}x{GRID}"])
  assert torch.equal(module.direct_projector[0].weight, mlp["proj_directvlm.0.weight"])
  assert torch.equal(module.direct_projector[1].weight, mlp["proj_directvlm.1.weight"])
  assert torch.equal(module.direct_projector[1].bias, mlp["proj_directvlm.1.bias"])

  assert torch.equal(module.connector.norm.weight, connector["model.norm.weight"])
  last = CONNECTOR_LAYERS - 1
  assert torch.equal(module.connector.layers[last].self_attn.q_proj.weight,
                     connector[f"model.layers.{last}.self_attn.q_proj.weight"])
  assert torch.equal(module.connector.layers[0].mlp.down_proj.weight,
                     connector["model.layers.0.mlp.down_proj.weight"])
  # The connector never sees input ids, so the token table is not materialized.
  assert not hasattr(module.connector, "embed_tokens")
  assert module.query_token_grid == GRID and module.hidden_size == MLLM_HIDDEN


def test_direct_condition_matches_manual_projection(tmp_path):
  state = _write_checkpoint(tmp_path)
  module = MingImageConditioning.from_pretrained(tmp_path).eval()
  mlp = state["mlp"]
  _, direct = _inputs(batch=2)
  with torch.no_grad():
    _, direct_condition = module(_inputs(batch=2)[0], direct)
    weight, bias = mlp["proj_directvlm.0.weight"], mlp["proj_directvlm.1.bias"]
    normed = direct * torch.rsqrt(direct.square().mean(-1, keepdim=True) + 1e-5) * weight
    expected = normed @ mlp["proj_directvlm.1.weight"].T + bias
  assert torch.allclose(direct_condition, expected, atol=1e-5)


def test_direct_condition_covers_every_prompt_token(tmp_path):
  _write_checkpoint(tmp_path)
  module = MingImageConditioning.from_pretrained(tmp_path).eval()
  query, _ = _inputs(batch=2)
  generator = torch.Generator().manual_seed(11)
  direct = torch.randn(2, 3, DIRECT_DIM, generator=generator)
  with torch.no_grad():
    _, direct_condition = module(query, direct)
    rows = [module.direct_projector(direct[:, index]) for index in range(direct.shape[1])]
  # The direct rows are a prompt-length sequence, projected per token: no single-row broadcast.
  assert direct_condition.shape == (2, 3, INNER_DIM)
  for index, row in enumerate(rows):
    assert torch.allclose(direct_condition[:, index], row, atol=1e-6)
  assert not torch.allclose(direct_condition[:, 0], direct_condition[:, 1])


def test_cap_feats_leave_through_connector_final_norm(tmp_path):
  _write_checkpoint(tmp_path)
  module = MingImageConditioning.from_pretrained(tmp_path).eval()
  with torch.no_grad():
    module.connector.norm.weight.zero_()
    query, direct = _inputs(batch=1)
    cap_feats, _ = module(query, direct)
  # A zeroed final RMSNorm zeroes the connector output, leaving only proj_out's bias.
  expected = module.proj_out.bias.expand(NUM_QUERY_TOKENS, CAP_DIM)
  assert torch.allclose(cap_feats[0], expected, atol=1e-6)


def test_normalize_and_scale_flags_come_from_the_mlp_config():
  mlp_config = dict(_mlp_config(), connector_norm=True, text_encoder_norm=True)
  module = MingImageConditioning.from_config(_connector_config(),
                                             mlp_config,
                                             hidden_size=MLLM_HIDDEN).eval()
  query, direct = _inputs(batch=1)
  with torch.no_grad():
    cap_feats, _ = module(query, direct)
  norms = cap_feats[0].norm(dim=-1)
  assert torch.allclose(norms, torch.full_like(norms, 1000.0), rtol=1e-4)


def test_direct_channel_count_is_enforced():
  module = _tiny_module()
  assert module.direct_input_dim == DIRECT_DIM
  query, _ = _inputs(batch=1)
  wrong = torch.randn(1, 3, DIRECT_DIM - MLLM_HIDDEN)
  with pytest.raises(ValueError, match="channels"):
    module(query, wrong)
  with pytest.raises(ValueError, match="shape"):
    module(query, torch.randn(1, DIRECT_DIM))


def test_query_token_count_is_enforced():
  module = _tiny_module()
  query, direct = _inputs(tokens=NUM_QUERY_TOKENS + 1)
  with pytest.raises(ValueError, match="query-token hidden states"):
    module(query, direct)


def test_missing_mlp_tensor_raises(tmp_path):
  mlp = _mlp_tensors()
  mlp.pop("proj_out.bias")
  _write_checkpoint(tmp_path, mlp_state=mlp)
  with pytest.raises(ValueError, match="mlp weights must be exactly"):
    MingImageConditioning.from_pretrained(tmp_path)


def test_unexpected_mlp_tensor_raises(tmp_path):
  mlp = _mlp_tensors()
  mlp["norm.weight"] = torch.ones(CONNECTOR_HIDDEN)
  _write_checkpoint(tmp_path, mlp_state=mlp)
  with pytest.raises(ValueError, match="mlp weights must be exactly"):
    MingImageConditioning.from_pretrained(tmp_path)


def test_query_token_grid_key_is_derived_from_config(tmp_path):
  mlp = _mlp_tensors()
  mlp["query_tokens_dict.16x16"] = mlp.pop(f"query_tokens_dict.{GRID}x{GRID}")
  _write_checkpoint(tmp_path, mlp_state=mlp)
  with pytest.raises(ValueError, match="mlp weights must be exactly"):
    MingImageConditioning.from_pretrained(tmp_path)


def test_missing_connector_tensor_raises(tmp_path):
  connector = _connector_tensors()
  connector.pop("model.layers.1.self_attn.o_proj.weight")
  _write_checkpoint(tmp_path, connector_state=connector)
  with pytest.raises(ValueError, match="connector weights do not cover"):
    MingImageConditioning.from_pretrained(tmp_path)


def test_connector_requires_the_documented_layout(tmp_path):
  connector = _connector_tensors()
  connector.pop("model.embed_tokens.weight")
  _write_checkpoint(tmp_path, connector_state=connector)
  with pytest.raises(ValueError, match="layout key"):
    MingImageConditioning.from_pretrained(tmp_path)


def test_loaded_weights_are_cast_to_the_requested_dtype(tmp_path):
  _write_checkpoint(tmp_path)
  module = MingImageConditioning.from_pretrained(tmp_path, torch_dtype=torch.bfloat16)
  assert module.proj_in.weight.dtype == torch.bfloat16
  assert module.connector.norm.weight.dtype == torch.bfloat16
  assert module.direct_projector[1].weight.dtype == torch.bfloat16
  # A bf16 RoPE table would move every query-token position; transformers keeps it fp32 too.
  assert module.connector.rotary_emb.inv_freq.dtype == torch.float32


def test_to_dtype_keeps_the_float32_rope_tables(tmp_path):
  _write_checkpoint(tmp_path)
  module = MingImageConditioning.from_pretrained(tmp_path, torch_dtype=torch.bfloat16)
  module.to(device="cpu", dtype=torch.bfloat16)
  module.to(device="cpu", dtype=torch.float16)
  assert module.dtype == torch.float16
  # The table must be the pristine float32 one, not a float16 round trip of it.
  reference = _tiny_module()
  assert module.connector.rotary_emb.inv_freq.dtype == torch.float32
  assert torch.equal(module.connector.rotary_emb.inv_freq, reference.connector.rotary_emb.inv_freq)


def test_dtype_and_device_properties_follow_the_parameters(tmp_path):
  # `DiffusionPipeline.to` reads both on every component, so they must exist before any `.to`.
  module = _tiny_module()
  assert module.dtype == torch.float32
  assert module.device.type == "cpu"

  _write_checkpoint(tmp_path)
  loaded = MingImageConditioning.from_pretrained(tmp_path, torch_dtype=torch.bfloat16)
  assert loaded.dtype == torch.bfloat16
  assert loaded.device.type == "cpu"
  assert loaded.direct_projector[1].weight.dtype == loaded.dtype
  assert loaded.connector.norm.weight.dtype == loaded.dtype


def test_checkpoint_shards_are_merged(tmp_path):
  connector = _connector_tensors()
  _write_checkpoint(tmp_path, connector_state=connector)
  keys = sorted(connector)
  half = len(keys) // 2
  connector_dir = tmp_path / "connector"
  save_file({key: connector[key]
             for key in keys[:half]}, str(connector_dir / "model-00001-of-00001.safetensors"))
  save_file({key: connector[key]
             for key in keys[half:]}, str(connector_dir / "model-00002-of-00002.safetensors"))
  module = MingImageConditioning.from_pretrained(tmp_path)
  assert torch.equal(module.connector.norm.weight, connector["model.norm.weight"])
  assert torch.equal(module.connector.layers[0].self_attn.v_proj.bias,
                     connector["model.layers.0.self_attn.v_proj.bias"])
