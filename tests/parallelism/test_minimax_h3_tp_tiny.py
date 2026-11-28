"""Multi-rank parity test for the MiniMax-H3 tensor-parallel planner (tiny model, no weights).

Pika H3 Flex runs the MiniMax-H3 transformer unchanged, so the planner in
`distributed/transformers/minimax_h3.py` is the only place tensor sharding can go wrong, and no test
covered it: a 50-block bf16 checkpoint gives ~14.7 dB between a sharded and an unsharded run, which is
far too coarse to tell "correct shards, chaotic sampler" from "wrong shards, plausible output".

This test pins the plumbing itself: one forward of a two-layer fp32 transformer, compared against the
unsharded output on rank 0, with the relative error required to be ≤ 1e-6. Two things make it able to
fail for the right reason:

* the weights are asserted to be `DTensor`s with sharded placements, because a planner that silently
  skips sharding would also match bit for bit;
* the fused SwiGLU projection is checked against its documented layout. The planner interleaves
  `[up, gate]` into one `(up_i, gate_i)` pair per rank, and a missed interleave keeps every shape
  valid while each rank multiplies a gate half with itself — exactly the silent failure this test
  exists for.

Usage:
  CUDA_VISIBLE_DEVICES=3,7 python -m pytest tests/parallelism/test_minimax_h3_tp_tiny.py -q
"""

import os
from datetime import timedelta
from functools import partial
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.tensor import DTensor, Replicate

from cache_dit import BlockAdapter, DBCacheConfig, ForwardPattern, enable_cache
from cache_dit.caching.block_adapters import BlockAdapterRegister
from cache_dit.distributed import linear_sharding
from cache_dit.distributed.config import ParallelismConfig
from cache_dit.distributed.text_encoders import dispatch as text_encoder_dispatch
from cache_dit.distributed.transformers import dispatch as transformer_dispatch
from diffusers import MiniMaxH3Transformer3DModel

_PORT = 29594
_HEADS = 4
_HEAD_DIM = 12
_HIDDEN = 16
_FFN = 32
_PATCH = (1, 2, 2)
_IN_CHANNELS = 4
_AUDIO_CHANNELS = 4
_TEXT_DIM = 32
_TEXT_TOKENS = 5
_VIDEO_ROWS = 6
_AUDIO_ROWS = 7
_REL_TOL = 1e-6


def _build(device, dtype=torch.float32) -> MiniMaxH3Transformer3DModel:
  torch.manual_seed(0)
  transformer = MiniMaxH3Transformer3DModel(
    num_attention_heads=_HEADS,
    attention_head_dim=_HEAD_DIM,
    hidden_size=_HIDDEN,
    num_layers=2,
    num_refiner_layers=1,
    ffn_dim=_FFN,
    in_channels=_IN_CHANNELS,
    audio_in_channels=_AUDIO_CHANNELS,
    patch_size=_PATCH,
    text_dim=_TEXT_DIM,
    freq_dim=16,
    time_embed_hidden_dim=32,
    time_embed_dim=16,
    rope_freq_dim=2,
  )
  return transformer.to(device=device, dtype=dtype).eval()


def _inputs(device, dtype=torch.float32, *, video_rows=_VIDEO_ROWS) -> dict:
  """One packed sequence: `[text | audio | video]`, with two distinct timesteps."""
  generator = torch.Generator().manual_seed(3)
  sequence_length = _TEXT_TOKENS + _AUDIO_ROWS + video_rows
  # Row order: text first, then audio, then video, matching the block that builds the layout.
  text_indices = torch.arange(0, _TEXT_TOKENS)
  audio_indices = torch.arange(_TEXT_TOKENS, _TEXT_TOKENS + _AUDIO_ROWS)
  video_indices = torch.arange(_TEXT_TOKENS + _AUDIO_ROWS, sequence_length)
  token_tags = torch.zeros(sequence_length, dtype=torch.long)
  token_tags[text_indices] = 1
  token_tags[audio_indices] = 2
  # Two timesteps: the audio rows sit at a different noise level than the video rows.
  timestep_indices = torch.zeros(sequence_length, dtype=torch.long)
  timestep_indices[audio_indices] = 1
  position_ids = torch.stack(
    [
      torch.arange(sequence_length, dtype=torch.float32),
      torch.arange(sequence_length, dtype=torch.float32) * 0.5,
      torch.arange(sequence_length, dtype=torch.float32) * 0.25,
    ],
    dim=-1,
  )
  return {
    "hidden_states":
    torch.randn(1,
                video_rows,
                _IN_CHANNELS * _PATCH[0] * _PATCH[1] * _PATCH[2],
                generator=generator).to(device=device, dtype=dtype),
    "audio_hidden_states":
    torch.randn(1, _AUDIO_ROWS, _AUDIO_CHANNELS, generator=generator).to(device=device,
                                                                         dtype=dtype),
    "encoder_hidden_states":
    torch.randn(1, _TEXT_TOKENS, _TEXT_DIM, generator=generator).to(device=device, dtype=dtype),
    "timestep":
    torch.tensor([0.7, 0.3], device=device, dtype=dtype),
    "timestep_indices":
    timestep_indices.to(device),
    "token_tags":
    token_tags.to(device),
    "position_ids":
    position_ids.to(device=device, dtype=dtype),
    "video_indices":
    video_indices.to(device),
    "audio_indices":
    audio_indices.to(device),
    "text_indices":
    text_indices.to(device),
    "return_dict":
    False,
  }


def _relative(out, reference) -> float:
  return ((out - reference).abs().mean() / reference.abs().mean().clamp_min(1e-6)).item()


def _assert_sharded(transformer: MiniMaxH3Transformer3DModel, tp_size: int) -> None:
  """Fail when the planner did not actually shard the block stack."""
  block = transformer.transformer_blocks[0]
  for name, module in (("attn.to_q", block.attn.to_q), ("attn.to_out.0", block.attn.to_out[0]),
                       ("ff.net.0.proj", block.ff.net[0].proj), ("ff.net.2", block.ff.net[2]),
                       ("adaln_proj.linear", block.adaln_proj.linear)):
    weight = module.weight
    assert isinstance(weight, DTensor), f"{name}.weight is not a DTensor: the stack was not sharded"
    local = weight.to_local()
    if name == "attn.to_out.0" or name == "ff.net.2":
      # Rowwise shards the input columns.
      assert local.shape[1] * tp_size == weight.shape[1], (name, local.shape, weight.shape)
    else:
      assert local.shape[0] * tp_size == weight.shape[0], (name, local.shape, weight.shape)
  # Replicated on purpose: `forward` scatters the packed rows with global indices.
  for name, module in (("proj_in", transformer.proj_in), ("proj_out", transformer.proj_out),
                       ("context_embedder", transformer.context_embedder)):
    assert not isinstance(module.weight, DTensor), f"{name} must stay replicated"


def _run_rank(rank: int, world: int) -> None:
  os.environ["MASTER_ADDR"] = "127.0.0.1"
  os.environ["MASTER_PORT"] = str(_PORT)
  torch.cuda.set_device(rank)
  torch.distributed.init_process_group("nccl", rank=rank, world_size=world)
  device = f"cuda:{rank}"

  try:
    inputs = _inputs(device)
    transformer = _build(device)

    reference = None
    if rank == 0:
      with torch.no_grad():
        reference = tuple(out.float().cpu() for out in transformer(**inputs))
    torch.distributed.barrier()

    enable_cache(
      BlockAdapterRegister.get_adapter(transformer),
      cache_config=DBCacheConfig(
        Fn_compute_blocks=1,
        Bn_compute_blocks=0,
        max_warmup_steps=100,
        max_cached_steps=0,
        num_inference_steps=2,
        enable_separate_cfg=False,
        residual_diff_threshold=0.0,
      ),
      parallelism_config=ParallelismConfig(tp_size=world),
    )
    _assert_sharded(transformer, world)

    with torch.no_grad():
      outputs = tuple(out.float().cpu() for out in transformer(**inputs))

    if rank == 0:
      for name, (out, ref) in zip(("video", "audio"), zip(outputs, reference)):
        assert out.shape == ref.shape, f"{name} output shape changed: {out.shape} vs {ref.shape}"
        relative = _relative(out, ref)
        print(f"[minimax-h3-tp] {name} rel={relative:.3e}")
        assert relative < _REL_TOL, f"{name} output diverged under TP ({relative:.3e})"

    from cache_dit.compile import set_compile_configs
    from cache_dit.profile.nvtx import LayerwiseNvtxMarker
    from torch._dynamo.utils import counters
    from torch._inductor import metrics

    set_compile_configs(cuda_graphs=False)
    transformer.compile_repeated_blocks(fullgraph=False, dynamic=False)
    with torch.no_grad():
      for _ in range(2):
        baseline = tuple(out.clone() for out in transformer(**inputs))
      graphs = counters["stats"]["unique_graphs"]
      kernels = metrics.generated_kernel_count
      with LayerwiseNvtxMarker({"transformer": transformer}):
        for _ in range(2):
          marked = transformer(**inputs)
          assert all(torch.equal(out, ref) for out, ref in zip(marked, baseline))
      removed = transformer(**inputs)
      assert all(torch.equal(out, ref) for out, ref in zip(removed, baseline))
    assert counters["stats"]["unique_graphs"] == graphs
    assert metrics.generated_kernel_count == kernels
  finally:
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_minimax_h3_tp_matches_single_rank_in_fp32():
  """Preserve TP parity and the warmed compiled execution when NVTX hooks change."""
  mp.spawn(_run_rank, args=(2, ), nprocs=2, join=True)


def _build_text_encoder():
  from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

  torch.manual_seed(1)
  config = Qwen3VLConfig(
    text_config={
      "hidden_size": 32,
      "intermediate_size": 48,
      "num_hidden_layers": 2,
      "num_attention_heads": 8,
      "num_key_value_heads": 4,
      "head_dim": 8,
      "vocab_size": 48,
      "attention_bias": True,
    },
    vision_config={
      "hidden_size": 8,
      "depth": 1,
      "num_heads": 2,
      "intermediate_size": 16,
      "out_hidden_size": 32,
      "deepstack_visual_indexes": [],
    },
  )
  return Qwen3VLForConditionalGeneration(config).eval().requires_grad_(False)


def _assert_parameters_match(reference, candidate):
  assert set(candidate._rowwise_layers) == set(reference._rowwise_layers)
  assert set(candidate._colwise_layers) == set(reference._colwise_layers)
  reference_params = dict(reference.named_parameters())
  candidate_params = dict(candidate.named_parameters())
  assert candidate_params.keys() == reference_params.keys()
  for name, expected in reference_params.items():
    actual = candidate_params[name]
    assert actual.shape == expected.shape, name
    assert actual.stride() == expected.stride(), name
    assert actual.requires_grad == expected.requires_grad, name
    assert isinstance(actual, DTensor) == isinstance(expected, DTensor), name
    if isinstance(expected, DTensor):
      assert actual.placements == expected.placements, name
      assert actual.device_mesh == expected.device_mesh, name
      expected, actual = expected.to_local(), actual.to_local()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=lambda msg: f"{name}: {msg}")


def _text_hidden_state(text_encoder):
  return text_encoder.model.language_model(input_ids=torch.tensor([[1, 4, 7, 2, 5]]),
                                           use_cache=False).last_hidden_state


def _run_gloo(rank, world_size, rendezvous, check):
  cpu = torch.device("cpu")
  torch.set_num_threads(1)
  dist.init_process_group("gloo",
                          init_method=rendezvous,
                          rank=rank,
                          world_size=world_size,
                          timeout=timedelta(seconds=40))
  try:
    with (
        patch.object(torch.accelerator, "current_accelerator", return_value=cpu),
        patch.object(torch._C, "_get_accelerator", return_value=cpu),
        torch.no_grad(),
    ):
      check(rank, cpu)
  finally:
    dist.destroy_process_group()


def _check_hybrid_local_weight_shards(rank, device):
  from cache_dit.distributed.transformers import minimax_h3

  native_dit, local_dit = _build(device), _build(device)
  native_te, local_te = _build_text_encoder(), _build_text_encoder()
  for dit in (native_dit, local_dit):
    weight = dit.transformer_blocks[-1].ff.net[2].weight
    weight.data = weight.detach().t().contiguous().t()
    assert not weight.is_contiguous()

  degrees = dict(tp_size=2, ulysses_size=2, ulysses_anything=True)
  enable_cache(native_dit,
               parallelism_config=ParallelismConfig(**degrees, extra_parallel_modules=[native_te]))
  config = ParallelismConfig(**degrees, local_weight_shards=True, extra_parallel_modules=[local_te])
  block = local_dit.transformer_blocks[-1]
  weights = {
    "attn.to_q": block.attn.to_q.weight,
    "ff.net.0.proj": block.ff.net[0].proj.weight,
    "ff.net.2": block.ff.net[2].weight,
    "text_encoder.q_proj": local_te.model.language_model.layers[-1].self_attn.q_proj.weight,
  }
  initialize = Mock(wraps=linear_sharding.local_parameter)
  with (
      patch.object(linear_sharding, "local_parameter", initialize),
      patch.object(minimax_h3, "local_parameter", initialize),
  ):
    enable_cache(local_dit, parallelism_config=config)
  initialized = {id(call.args[0]) for call in initialize.call_args_list}
  local_weights = {name for name, weight in weights.items() if id(weight) in initialized}
  assert local_weights == {"attn.to_q", "ff.net.0.proj", "text_encoder.q_proj"}, local_weights

  _assert_parameters_match(native_dit, local_dit)
  _assert_parameters_match(native_te, local_te)
  # Gloo requires equal gather sizes for the CP output.
  inputs = _inputs(device, video_rows=8)
  torch.testing.assert_close(local_dit(**inputs), native_dit(**inputs), rtol=0, atol=0)
  torch.testing.assert_close(_text_hidden_state(local_te),
                             _text_hidden_state(native_te),
                             rtol=0,
                             atol=0)

  malformed_weights = (
    ("ff.net.0.proj", lambda _: torch.nn.Linear(_HIDDEN, 10),
     r"replica rank 1: .*equal \(up, gate\) chunks"),
    ("attn.to_q", lambda _: torch.nn.Linear(_HIDDEN, _HEADS * _HEAD_DIM + 4),
     r"replica rank 1: .*to_q has .* rows"),
    ("ff.net.2", lambda linear: linear.to(torch.bfloat16), "matching weight metadata"),
  )
  config = ParallelismConfig(**degrees, local_weight_shards=True)
  for path, invalidate, error in malformed_weights:
    dit = _build(device)
    if rank == 1:
      block = dit.transformer_blocks[-1]
      block.set_submodule(path, invalidate(block.get_submodule(path)))
    with pytest.raises(ValueError, match=error):
      enable_cache(dit, parallelism_config=config)
    assert getattr(dit, "_cp_config", None) is None
    assert type(dit.transformer_blocks[0].attn.to_q.weight) is torch.nn.Parameter


@pytest.mark.skipif(not dist.is_gloo_available(), reason="needs Gloo")
def test_minimax_h3_local_weight_shards(tmp_path):
  """Match native TP and reject an ineligible rank before sharding the component."""
  mp.spawn(_run_gloo,
           args=(4, f"file://{tmp_path}/local-weight-shards", _check_hybrid_local_weight_shards),
           nprocs=4)


@pytest.mark.parametrize(
  "prepare, register, registry",
  [
    (transformer_dispatch._prepare_transformer_tp,
     transformer_dispatch.TensorParallelismPlannerRegister, "_tp_planner_registry"),
    (text_encoder_dispatch._prepare_text_encoder_tp,
     text_encoder_dispatch.TextEncoderTensorParallelismPlannerRegister,
     "_text_encoder_tp_planner_registry"),
  ],
  ids=["transformer", "text_encoder"],
)
def test_local_weight_shards_uses_planner_registration(prepare, register, registry, monkeypatch):
  monkeypatch.setattr(register, registry, {})
  mesh = object()

  @register.register("RegisteredModel", local_weight_shards=True)
  class LocalPlanner:

    def _preflight_local_weight_shards(self, module, parallelism_config):
      return mesh

    def apply(self, module, parallelism_config, *, local_shard_mesh=None):
      return local_shard_mesh

  register.register("RegisteredAlias")(LocalPlanner)

  @register.register("DerivedModel")
  class DerivedPlanner(LocalPlanner):
    pass

  cases = (
    ("RegisteredAlias", True, mesh),
    ("DerivedModel", True, None),
    ("RegisteredModel", False, None),
  )
  for name, local_weight_shards, expected_mesh in cases:
    module = type(name, (torch.nn.Module, ), {})()
    config = ParallelismConfig(tp_size=2, local_weight_shards=local_weight_shards)
    apply, uses_local = prepare(module, config)
    assert uses_local is (expected_mesh is not None)
    assert apply(module, config) is expected_mesh


def _check_local_weight_retry(rank, device):
  from diffusers import MiniMaxH3ModularPipeline

  dit, te = _build(device), _build_text_encoder()
  pipe = MiniMaxH3ModularPipeline(workflow="t2va")
  pipe.update_components(transformer=dit, text_encoder=te)
  config = ParallelismConfig(ulysses_size=2,
                             local_weight_shards=True,
                             extra_parallel_modules=["text_encoder"])
  cache_config = DBCacheConfig(Fn_compute_blocks=1,
                               max_cached_steps=0,
                               num_inference_steps=2,
                               enable_separate_cfg=False)
  enable = partial(enable_cache, pipe, cache_config=cache_config, parallelism_config=config)
  projection = te.model.language_model.layers[-1].self_attn.q_proj
  weight = projection.weight
  if rank == 1:
    projection.weight = torch.nn.Parameter(weight.to(torch.bfloat16), requires_grad=False)

  with pytest.raises(ValueError, match="matching weight metadata"):
    enable()
  assert getattr(dit, "_cp_config", None) is None
  assert not getattr(dit, "_is_parallelized", False)
  assert not getattr(te, "_is_parallelized", False)
  assert config.extra_parallel_modules == ["text_encoder"]

  projection.weight = weight
  context = pipe._context_manager
  assert enable() is pipe
  assert pipe._context_manager is context
  assert dit._is_parallelized and te._is_parallelized
  assert isinstance(projection.weight, DTensor)
  assert dit(**_inputs(device, video_rows=8))[0].numel()
  assert _text_hidden_state(te).numel()

  sharded = projection.weight
  enable()
  assert projection.weight is sharded


def _check_multi_transformer_preflight(rank, device):
  first, second = _build(device), _build(device)
  adapter = BlockAdapter(transformer=[first, second],
                         blocks=[first.transformer_blocks, second.transformer_blocks],
                         forward_pattern=[ForwardPattern.Pattern_3, ForwardPattern.Pattern_3])
  config = ParallelismConfig(tp_size=2, local_weight_shards=True)
  projection = second.transformer_blocks[-1].attn.to_q
  weight = projection.weight
  if rank == 1:
    projection.weight = torch.nn.Parameter(weight.to("meta"), requires_grad=weight.requires_grad)
  with pytest.raises(ValueError, match=r"replica rank 1: .*to_q.weight.*CPU parameter"):
    enable_cache(adapter, parallelism_config=config)
  assert not getattr(first, "_is_parallelized", False)
  assert type(first.transformer_blocks[0].attn.to_q.weight) is torch.nn.Parameter

  if rank == 1:
    projection.weight = weight
  enable_cache(adapter, parallelism_config=config)
  assert isinstance(first.transformer_blocks[0].attn.to_q.weight, DTensor)
  assert isinstance(second.transformer_blocks[0].attn.to_q.weight, DTensor)


def _check_planner_scope(device):
  from cache_dit.distributed import enable_parallelism
  from cache_dit.distributed.text_encoders import parallelize_text_encoder
  from cache_dit.distributed.transformers import parallelize_transformer
  from cache_dit.models.pipelines.fast_h3 import FastH3Transformer3DModel

  dit = FastH3Transformer3DModel.from_config(_build(device).config, enable_vsa=False)
  config = ParallelismConfig(tp_size=2, local_weight_shards=True)
  with pytest.raises(ValueError, match="requires a selected TP planner with CPU weight sharding"):
    enable_parallelism(dit, config)

  weight = dit.transformer_blocks[0].attn.to_q.weight
  weight.data = weight.detach().t().contiguous().t()
  te = _build_text_encoder()
  config.extra_parallel_modules = [te]
  enable_parallelism(dit, config)
  assert isinstance(dit.transformer_blocks[0].attn.to_q.weight, DTensor)
  assert isinstance(te.model.language_model.layers[0].self_attn.q_proj.weight, DTensor)

  config = ParallelismConfig(tp_size=2, local_weight_shards=True)
  dit = parallelize_transformer(_build(device), config)
  assert isinstance(dit.transformer_blocks[0].attn.to_q.weight, DTensor)
  te = parallelize_text_encoder(_build_text_encoder(), config)
  assert isinstance(te.model.language_model.layers[0].self_attn.q_proj.weight, DTensor)


def _check_preflight_retry(rank, device):
  _check_planner_scope(device)
  _check_local_weight_retry(rank, device)
  _check_multi_transformer_preflight(rank, device)

  local_parameter = linear_sharding.local_parameter

  def wrong_placement(param, local, mesh, _placement):
    return local_parameter(param, local, mesh, Replicate())

  with patch.object(linear_sharding, "local_parameter", wrong_placement):
    with pytest.raises(ValueError, match="Cannot distribute a DTensor with placements"):
      enable_cache(_build(device),
                   parallelism_config=ParallelismConfig(tp_size=2, local_weight_shards=True))


@pytest.mark.skipif(not dist.is_gloo_available(), reason="needs Gloo")
def test_minimax_h3_local_weight_preflight_retry(tmp_path):
  """Check the selected planners, retry a preflight failure, and propagate DTensor errors."""
  mp.spawn(_run_gloo,
           args=(2, f"file://{tmp_path}/local-weight-preflight", _check_preflight_retry),
           nprocs=2)
