"""Regression tests for the MiniMax-H3 integration (cache, CP, TP).

Four silent failure modes are covered, all CPU-only:

1. `MiniMaxH3TensorParallelismPlanner.rearrange_ffn_weight` must keep the fused SwiGLU up/gate
   halves aligned per TP chunk. diffusers' `SwiGLU` puts the up projection **first**
   (`hidden_states, gate = chunk(2, -1)`), so a reorder written the other way round keeps every
   shape valid while multiplying the gate half with itself.
2. The AdaLN projection is sharded with `output_layouts=Replicate()` because the block reshapes its
   output with `view(-1, 6 * hidden).chunk(6)`. This test checks that the gathered partial outputs
   rebuild the full projection and that the reshape still works.
3. The CP plan must register `hidden_states` on the first block **before** `adaln_indices`: the
   cached path replays the first split hook of that block to re-derive the residual.
4. `enable_cache` must keep `transformer.forward`'s signature: a modular denoise step forwards only
   the kwargs that appear in `inspect.signature(forward)`.
"""
import inspect
import gzip
import importlib
import json
from contextlib import contextmanager
from functools import wraps
from types import SimpleNamespace

import pytest
import torch

import cache_dit
from cache_dit import DBCacheConfig
from cache_dit.distributed.config import ParallelismConfig
from cache_dit.distributed.transformers.minimax_h3 import (
  MiniMaxH3ContextParallelismPlanner,
  MiniMaxH3TensorParallelismPlanner,
)

H3_INPUTS = (
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


def _tiny_minimax_h3_transformer():
  from diffusers import MiniMaxH3Transformer3DModel

  return MiniMaxH3Transformer3DModel(
    num_attention_heads=4,
    attention_head_dim=8,
    hidden_size=32,
    num_layers=1,
    num_refiner_layers=1,
    ffn_dim=64,
    in_channels=2,
    audio_in_channels=2,
    patch_size=(1, 2, 2),
    text_dim=4,
    freq_dim=8,
    time_embed_hidden_dim=32,
    time_embed_dim=16,
    # 2 * 3 * rope_freq_dim rotated channels have to fit into `attention_head_dim`
    rope_freq_dim=1,
  )


class _FakeMesh:
  """Just enough of a `DeviceMesh` for the plan-construction guards."""

  def __init__(self, size: int):
    self._size = size

  def size(self) -> int:
    return self._size


def _tiny_packed_inputs(video_rows=4, audio_rows=2, text_rows=2):
  seq_len = video_rows + audio_rows + text_rows
  text_indices = torch.arange(0, text_rows, dtype=torch.long)
  audio_indices = torch.arange(text_rows, text_rows + audio_rows, dtype=torch.long)
  video_indices = torch.arange(text_rows + audio_rows, seq_len, dtype=torch.long)
  token_tags = torch.cat([
    torch.ones(text_rows, dtype=torch.long),
    torch.full((audio_rows, ), 2, dtype=torch.long),
    torch.zeros(video_rows, dtype=torch.long),
  ])
  return dict(
    hidden_states=torch.randn(1, video_rows, 8),
    audio_hidden_states=torch.randn(1, audio_rows, 2),
    encoder_hidden_states=torch.randn(1, text_rows, 4),
    timestep=torch.rand(1),
    timestep_indices=torch.zeros(seq_len, dtype=torch.long),
    token_tags=token_tags,
    position_ids=torch.zeros(seq_len, 3),
    video_indices=video_indices,
    audio_indices=audio_indices,
    text_indices=text_indices,
  )


def test_minimax_h3_ffn_weight_rearrange_keeps_swiglu_aligned():
  from diffusers.models.attention import FeedForward

  torch.manual_seed(0)
  dim, tp_size = 8, 2
  ff = FeedForward(dim, inner_dim=4 * dim, activation_fn="swiglu", bias=False)
  block = torch.nn.Module()
  block.ff = ff

  x = torch.randn(2, 3, dim)
  reference = ff(x)

  original_in_weight = ff.net[0].proj.weight.data.clone()
  original_out_weight = ff.net[2].weight.data.clone()

  planner = MiniMaxH3TensorParallelismPlanner()
  planner.rearrange_ffn_weight(block, tp_size)

  # simulate ColwiseParallel(ff.net[0].proj): a contiguous chunk of output rows per rank
  in_shards = ff.net[0].proj.weight.data.chunk(tp_size, dim=0)
  # simulate RowwiseParallel(ff.net[2]): a contiguous chunk of input columns per rank
  out_shards = original_out_weight.chunk(tp_size, dim=1)

  partial = []
  for in_shard, out_shard in zip(in_shards, out_shards):
    local = torch.nn.functional.linear(x, in_shard)
    # diffusers SwiGLU: `hidden_states, gate = chunk(2, -1)` then `hidden_states * silu(gate)`
    up, gate = local.chunk(2, dim=-1)
    partial.append(torch.nn.functional.linear(up * torch.nn.functional.silu(gate), out_shard))
  # the row-wise shards are all-reduced (summed) into the full output
  sharded = sum(partial)

  assert torch.allclose(
    sharded, reference,
    atol=1e-6), ("SwiGLU up/gate halves are misaligned after the TP weight rearrangement: "
                 f"max abs diff {(sharded - reference).abs().max().item():.6e}")

  # the rearrangement has to be a pure row permutation of the original weight
  assert torch.equal(
    ff.net[0].proj.weight.data.sort(dim=0).values,
    original_in_weight.sort(dim=0).values,
  )


def test_minimax_h3_adaln_output_rebuilds_from_replicated_shards():
  """`ColwiseParallel(output_layouts=Replicate())` all-gathers the partial outputs.

  The block then does `view(-1, 6 * hidden).chunk(6)`, so the gathered tensor must be the full
  `(num_timesteps * 3, 6 * hidden)` table.
  """
  from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3AdaLayerNormModulation

  torch.manual_seed(0)
  hidden, time_embed_dim, tp_size, num_timesteps = 12, 6, 2, 2
  modulation = MiniMaxH3AdaLayerNormModulation(time_embed_dim=time_embed_dim, hidden_size=hidden)
  temb = torch.randn(num_timesteps, time_embed_dim, dtype=torch.float32)

  reference = modulation(temb)
  assert len(reference) == 6
  assert reference[0].shape == (num_timesteps * 3, hidden)

  # colwise shard: each rank holds `96768 / tp_size` output rows of the same projection
  activation = torch.nn.functional.silu(temb).to(modulation.linear.weight.dtype)
  weight_shards = modulation.linear.weight.data.chunk(tp_size, dim=0)
  bias_shards = modulation.linear.bias.data.chunk(tp_size, dim=0)
  partial = [
    torch.nn.functional.linear(activation, w, b) for w, b in zip(weight_shards, bias_shards)
  ]
  # Replicate() output layout = all-gather along the last dim, i.e. the original row order
  gathered = torch.cat(partial, dim=-1)
  full = torch.nn.functional.linear(activation, modulation.linear.weight.data,
                                    modulation.linear.bias.data)
  assert torch.allclose(gathered, full, atol=1e-6)

  # the reshape the block relies on still works on the gathered tensor
  tab = gathered.view(-1, 6 * hidden).chunk(6, dim=-1)
  for got, want in zip(tab, reference):
    assert torch.allclose(got, want, atol=1e-6)


def test_minimax_h3_cp_plan_splits_hidden_state_before_adaln_indices():
  transformer = _tiny_minimax_h3_transformer()
  planner = MiniMaxH3ContextParallelismPlanner()
  parallelism_config = ParallelismConfig(ulysses_size=2, ulysses_anything=True)

  cp_plan = planner._apply(transformer, parallelism_config)

  keys = list(cp_plan)
  assert keys[0] == "rope"
  assert keys[1] == "transformer_blocks.0"
  assert keys.index("transformer_blocks.0") < keys.index("transformer_blocks.*"), (
    "the cached path replays the first split hook of `transformer_blocks[0]`, so its "
    "`hidden_states` plan has to be registered before the `adaln_indices` one")
  assert "hidden_states" in cp_plan["transformer_blocks.0"]

  # the rope module returns a (cos, sin) pair: both outputs are split
  for index in (0, 1):
    assert cp_plan["rope"][index].split_output is True

  # `forward` scatters the streams with global row indices, so its own inputs stay whole
  assert "" not in cp_plan, "the packed buffer is built from global indices: no root split"
  # these drive `forward`'s shape checks and the global scatter, and are never split
  for subplan in cp_plan.values():
    if isinstance(subplan, dict):
      assert not ({"position_ids", "token_tags"} & set(subplan))
  # `timestep_indices` is only split at the output norm, after `forward`'s validation
  assert "timestep_indices" in cp_plan["norm_out"]


def test_minimax_h3_cp_plan_rejects_indivisible_heads():
  transformer = _tiny_minimax_h3_transformer()  # 4 heads
  planner = MiniMaxH3ContextParallelismPlanner()
  # 4 heads / tp2 = 2 local heads, which do not divide by 3. A plain namespace stands in for
  # `ParallelismConfig`, whose constructor validates the process group of a real run.
  parallelism_config = SimpleNamespace(
    ulysses_size=3,
    ring_size=None,
    tp_size=2,
    ulysses_anything=False,
    ulysses_async=False,
  )
  with pytest.raises(ValueError, match="not divisible by ulysses_size"):
    planner._apply(transformer, parallelism_config)


def test_minimax_h3_forward_signature_preserved_after_enable_cache():
  transformer = _tiny_minimax_h3_transformer()
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
  for name in H3_INPUTS:
    assert name in signature_after.parameters, (
      f"'{name}' disappeared from the patched transformer.forward signature")

  kwargs = _tiny_packed_inputs()
  video_output, audio_output = transformer(**kwargs, return_dict=False)
  assert video_output.shape == (1, kwargs["video_indices"].numel(), 8)
  assert audio_output.shape == (1, kwargs["audio_indices"].numel(), 2)

  cache_dit.disable_cache(transformer)


def test_minimax_h3_tp_plan_rejects_fused_qkv():
  """`fuse_qkv_projections()` would leave the plan sharding projections that no longer exist."""
  transformer = _tiny_minimax_h3_transformer()
  transformer.transformer_blocks[0].attn.fused_projections = True
  planner = MiniMaxH3TensorParallelismPlanner()
  with pytest.raises(ValueError, match="fused_projections"):
    planner.parallelize_transformer(transformer, tp_mesh=_FakeMesh(2))


def test_minimax_h3_ffn_rearrange_is_guarded_against_a_second_call():
  from diffusers.models.attention import FeedForward

  torch.manual_seed(0)
  ff = FeedForward(8, inner_dim=32, activation_fn="swiglu", bias=False)
  block = torch.nn.Module()
  block.ff = ff
  planner = MiniMaxH3TensorParallelismPlanner()

  planner.rearrange_ffn_weight(block, tp_size=2)
  # the interleave is an involution: applying it twice would silently restore gate x gate
  with pytest.raises(RuntimeError, match="already rearranged"):
    planner.rearrange_ffn_weight(block, tp_size=2)


@pytest.mark.parametrize("workflow", ["t2va", "fl2va", "ref2va"])
@pytest.mark.parametrize("device", [
  "cpu",
  pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"))
])
def test_profiler_preserves_minimax_loop_and_captures_complete_steps(workflow, device, monkeypatch,
                                                                     tmp_path):
  from diffusers import MiniMaxH3Scheduler, MiniMaxH3ModularPipeline
  from cache_dit.utils import iter_transformer_modules
  from diffusers.modular_pipelines.modular_pipeline import PipelineState
  from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3SetTimestepsStep
  from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import MiniMaxH3Blocks
  from cache_dit.models.pipelines.minimax_h3 import MiniMaxH3DenoiseLoop, minimax_h3_blocks
  from cache_dit.profile.nvtx import nvtx_scope

  upstream = MiniMaxH3Blocks().get_workflow(workflow)
  pipe = MiniMaxH3ModularPipeline(blocks=minimax_h3_blocks(), workflow=workflow)
  selected = pipe.blocks
  assert list(selected.sub_blocks) == list(upstream.sub_blocks)
  for name, block in upstream.sub_blocks.items():
    actual = selected.sub_blocks[name]
    assert (actual.inputs, actual.intermediate_outputs, actual.outputs,
            actual.expected_components) == (block.inputs, block.intermediate_outputs, block.outputs,
                                            block.expected_components)
    if name == "denoise.denoise":
      assert isinstance(actual, MiniMaxH3DenoiseLoop)
      assert list(actual.sub_blocks) == list(block.sub_blocks)
      for key, child in block.sub_blocks.items():
        assert inspect.unwrap(actual.sub_blocks[key].__call__) is inspect.unwrap(child.__call__)
        assert actual.sub_blocks[key].expected_components == child.expected_components
    else:
      assert isinstance(actual, type(block))
      assert inspect.unwrap(actual.__call__) is inspect.unwrap(block.__call__)
  root_name = "transformer_ref" if workflow == "ref2va" else "transformer"
  assert selected.sub_blocks["denoise.denoise"].sub_blocks["denoiser"].transformer_name == root_name
  component_names = {spec.name for spec in selected.expected_components}
  assert ("transformer_ref" in component_names) is (workflow == "ref2va")
  assert ("transformer" in component_names) is (workflow != "ref2va")

  profiler_module = importlib.import_module("cache_dit.profile.profiler")
  nvtx_module = importlib.import_module("cache_dit.profile.nvtx")
  monkeypatch.setattr(nvtx_module, "current_platform", SimpleNamespace(device_type="cuda"))
  stage_stack = []

  def push(name):
    stage_stack.append(name)

  monkeypatch.setattr(torch.cuda.nvtx, "range_push", push)
  monkeypatch.setattr(torch.cuda.nvtx, "range_pop", stage_stack.pop)
  if device == "cpu":
    # The collector is real; only accelerator validation and synchronization are faked on CPU.
    monkeypatch.setattr(
      profiler_module, "current_platform",
      SimpleNamespace(device_type="cuda",
                      is_accelerator_available=lambda: True,
                      synchronize=lambda device: None))
  torch.manual_seed(31)
  transformer = _tiny_minimax_h3_transformer().to(device).eval()
  inputs = {name: tensor.to(device) for name, tensor in _tiny_packed_inputs().items()}
  initial_rng = torch.get_rng_state()
  initial_cuda_rng = torch.cuda.get_rng_state() if device == "cuda" else None
  condition_video = int(workflow != "t2va")
  condition_audio = int(workflow == "ref2va")
  outputs = []

  for mode in ("original", "disabled", "all", "0,2"):
    torch.set_rng_state(initial_rng)
    if device == "cuda":
      torch.cuda.set_rng_state(initial_cuda_rng)
    counts = dict.fromkeys(("predict", "video_update", "audio_update", "progress"), 0)
    context = cache_dit.ProfilerContext(enabled=mode in ("all", "0,2"),
                                        ranges=mode if mode in ("all", "0,2") else "all",
                                        activities=["CPU", "GPU"] if device == "cuda" else ["CPU"],
                                        devices=[device],
                                        output_dir=tmp_path / mode,
                                        with_stack=False)

    def trace_call(original, label):

      @wraps(original)
      def call(*args, **kwargs):
        i = counts[label]
        counts[label] += 1
        if mode != "original":
          expected_stage = ["denoise_loop", f"denoise_step {i}"]
          if label != "progress":
            expected_stage.append("predict" if label == "predict" else "scheduler_step")
          assert stage_stack == expected_stage
        with torch.profiler.record_function(f"h3_{label}_{i}"):
          return original(*args, **kwargs)

      return call

    components = SimpleNamespace(scheduler=MiniMaxH3Scheduler(shift=12),
                                 audio_scheduler=MiniMaxH3Scheduler(shift=3),
                                 _execution_device=torch.device(device),
                                 keyframe_noise_aug=0.999)
    setattr(components, root_name, transformer)
    assert dict(iter_transformer_modules(components)) == {root_name: transformer}
    components.scheduler.step = trace_call(components.scheduler.step, "video_update")
    components.audio_scheduler.step = trace_call(components.audio_scheduler.step, "audio_update")
    original_forward = transformer.forward
    transformer.forward = trace_call(original_forward, "predict")
    state = PipelineState(values=dict(latents=inputs["hidden_states"][0].clone(),
                                      audio_latents=inputs["audio_hidden_states"][0].clone(),
                                      prompt_embeds=inputs["encoder_hidden_states"],
                                      num_inference_steps=5,
                                      num_condition_video_rows=condition_video,
                                      num_condition_audio_rows=condition_audio))
    for name in ("token_tags", "position_ids", "video_indices", "audio_indices", "text_indices"):
      state.set(name, inputs[name], kwargs_type="denoiser_input_fields")
    MiniMaxH3SetTimestepsStep()(components, state)
    loop = (upstream if mode == "original" else selected).sub_blocks["denoise.denoise"]

    @contextmanager
    def progress(total):
      assert total == 4
      yield SimpleNamespace(update=trace_call(lambda: None, "progress"))

    loop.progress_bar = progress
    try:
      with nvtx_scope(mode != "original"), context:
        with torch.profiler.record_function("h3_encode"):
          torch.ones(1, device=device) + 1
        returned_components, returned_state = loop(components, state)
        with torch.profiler.record_function("h3_decode"):
          torch.ones(1, device=device) * 2
    finally:
      transformer.forward = original_forward
    assert returned_components is components and returned_state is state
    assert not stage_stack
    assert all(count == 4 for count in counts.values())
    assert components.scheduler.step_index == components.audio_scheduler.step_index == 4
    assert torch.equal(
      state.get("latents")[:condition_video], inputs["hidden_states"][0, :condition_video])
    assert torch.equal(
      state.get("audio_latents")[:condition_audio],
      inputs["audio_hidden_states"][0, :condition_audio])
    assert torch.equal(torch.get_rng_state(), initial_rng)
    if device == "cuda":
      assert torch.equal(torch.cuda.get_rng_state(), initial_cuda_rng)
    outputs.append((state.get("latents").clone(), state.get("audio_latents").clone()))
    for result in context.results:
      with gzip.open(result.trace_path, "rt") as trace_file:
        trace = json.load(trace_file)["traceEvents"]
      names = {event["name"] for event in trace if event["name"].startswith("h3_")}
      selected_steps = range(4) if mode == "all" else range(result.actual_range[0],
                                                            result.actual_range[1] + 1)
      expected = {f"h3_{label}_{i}" for i in selected_steps for label in counts}
      if mode == "all":
        expected.update(("h3_encode", "h3_decode"))
      assert names == expected
      if device == "cuda":
        assert any(event.get("cat") == "kernel" for event in trace)
  assert all(
    torch.equal(video, outputs[0][0]) and torch.equal(audio, outputs[0][1])
    for video, audio in outputs[1:])
