"""Two-rank tensor-parallel parity test for the Qwen3-VL text encoder (tiny model, no weights).

Ideogram 4 consumes a complete `Qwen3VLModel` as its text encoder: `encode_prompt` calls
`Ideogram4Pipeline._get_text_encoder_hidden_states`, which walks `text_encoder.language_model`
layer by layer and taps a fixed set of decoder layers. TP therefore has to shard the *nested*
text stack while keeping the outer wrapper intact, and reproduce the unsharded stack up to
all-reduce rounding. A plan that parallelizes the wrong object (e.g. the `Qwen3VLTextModel`
that the `Qwen3VLModel` parent owns), or that only shards the first layer, still yields
plausible-looking conditioning, so every tapped layer is compared against a TP-free reference.

Usage:
  CUDA_VISIBLE_DEVICES=4,5 python -m pytest tests/parallelism/test_qwen3_vl_model_tp_tiny.py -q
"""

import os

import pytest
import torch
import torch.multiprocessing as mp

try:
  from transformers import Qwen3VLConfig, Qwen3VLModel
  from diffusers import Ideogram4Pipeline
  from diffusers.pipelines.ideogram4.pipeline_ideogram4 import (
    QWEN3_VL_ACTIVATION_LAYERS, )
except ImportError:
  pytest.skip("diffusers without Ideogram-4 support", allow_module_level=True)

# Same tiny Qwen3-VL shape as `tests/pipelines/ideogram4/test_pipeline_ideogram4.py`: the text
# encoder must be deep enough to expose the last tapped layer, and every sharded dim (heads,
# kv heads, intermediate) has to stay divisible by the TP degree.
_TEXT_HIDDEN_SIZE = 8
_NUM_TEXT_LAYERS = max(QWEN3_VL_ACTIVATION_LAYERS) + 1
_VOCAB_SIZE = 151936
_HEADS = 4
_KV_HEADS = 2
_HEAD_DIM = 8
_INTERMEDIATE = 16

_BATCH = 2
_SEQ = 12
_TEXT_LENGTHS = (5, _SEQ)  # left-padded, so the causal mask has a real padding region

_DTYPES = ("float32", "bfloat16")
# fp32 only reorders an 8-term dot product inside the RowwiseParallel all-reduce. bf16 also
# rounds each partial sum to bf16 before the all-reduce, and the residual stream carries that
# rounding through all 36 layers, so its bound follows ~sqrt(depth) * bf16 eps: the measured
# drift grows smoothly from 4e-4 (layer 0) to 2.5e-2 (layer 30), which is accumulation, not a
# mis-shard. Both bounds stay far below the O(1) relative error a wrong shard produces, and the
# fp32 run pins the algebra at 1e-5 so a wrong plan cannot hide behind the bf16 slack.
_REL_TOL = {"float32": 1e-5, "bfloat16": 5e-2}
_PORT = 29617


def _build_model(device: str, dtype: torch.dtype) -> Qwen3VLModel:
  torch.manual_seed(0)
  model = Qwen3VLModel(
    Qwen3VLConfig(
      text_config={
        "hidden_size": _TEXT_HIDDEN_SIZE,
        "num_hidden_layers": _NUM_TEXT_LAYERS,
        "num_attention_heads": _HEADS,
        "num_key_value_heads": _KV_HEADS,
        "intermediate_size": _INTERMEDIATE,
        "head_dim": _HEAD_DIM,
        "vocab_size": _VOCAB_SIZE,
        "max_position_embeddings": 256,
        "rope_theta": 10_000.0,
      },
      vision_config={
        "hidden_size": 8,
        "depth": 2,
        "num_heads": 2,
        "intermediate_size": 16,
        "out_hidden_size": _TEXT_HIDDEN_SIZE,
        "patch_size": 14,
      },
    ))
  return model.to(device=device, dtype=dtype).eval()


def _language_inputs(device: str) -> dict:
  """Left-padded text-only inputs, matching the layout `encode_prompt` feeds the encoder."""
  generator = torch.Generator().manual_seed(1)
  token_ids = torch.zeros(_BATCH, _SEQ, dtype=torch.long)
  attention_mask = torch.zeros(_BATCH, _SEQ, dtype=torch.long)
  text_position_ids = torch.zeros(_BATCH, _SEQ, dtype=torch.long)
  for batch, length in enumerate(_TEXT_LENGTHS):
    offset = _SEQ - length
    token_ids[batch, offset:] = torch.randint(0, _VOCAB_SIZE, (length, ), generator=generator)
    attention_mask[batch, offset:] = 1
    text_position_ids[batch, offset:] = torch.arange(length)
  return {
    "token_ids": token_ids.to(device),
    "attention_mask": attention_mask.to(device),
    "text_position_ids": text_position_ids.to(device),
  }


def _tapped_hidden_states(text_encoder, inputs) -> list[torch.Tensor]:
  """Run the production language-stack path and return the tapped layers in fp32 on CPU."""
  with torch.no_grad():
    tapped = Ideogram4Pipeline._get_text_encoder_hidden_states(
      text_encoder,
      inputs["token_ids"],
      inputs["attention_mask"],
      inputs["text_position_ids"],
    )
  return [hidden.float().cpu() for hidden in tapped]


def _last_hidden_state(text_encoder, inputs) -> torch.Tensor:
  with torch.no_grad():
    out = text_encoder(
      input_ids=inputs["token_ids"],
      attention_mask=inputs["attention_mask"],
    )
  return out.last_hidden_state.float().cpu()


def _rel_l2(got: torch.Tensor, ref: torch.Tensor) -> float:
  return ((got - ref).norm() / ref.norm().clamp_min(1e-12)).item()


def _local_shape(param: torch.Tensor) -> tuple:
  # A sharded parameter is a DTensor: `.shape` stays global, `.to_local()` is the rank's shard.
  return tuple(param.to_local().shape) if hasattr(param, "to_local") else tuple(param.shape)


def _check_planner_contract(rank: int, world: int, device: str) -> None:
  """Assert the planner shards the nested text stack and hands back the original wrapper."""
  from cache_dit.distributed import ParallelismConfig
  from cache_dit.distributed.text_encoders.qwen3 import (
    Qwen3VLModelTensorParallelismPlanner, )
  from cache_dit.distributed.text_encoders.register import (
    TextEncoderTensorParallelismPlannerRegister, )

  torch.manual_seed(0)
  model = _build_model(device, torch.float32)
  language_model = model.language_model

  planner_cls = TextEncoderTensorParallelismPlannerRegister.get_planner(model)
  assert planner_cls is Qwen3VLModelTensorParallelismPlanner, (
    f"Qwen3VLModel resolved to {planner_cls.__name__}")

  parallelized, layer_plans = planner_cls()._apply(
    text_encoder=model,
    parallelism_config=ParallelismConfig(tp_size=world),
  )
  # The wrapper must survive: callers keep using `pipe.text_encoder` and rely on
  # `.language_model`, and `TextEncoderParallelismPlanner.apply` records the plans on it.
  assert parallelized is model, "planner replaced the Qwen3VLModel wrapper"
  assert isinstance(parallelized, Qwen3VLModel)
  assert parallelized.language_model is language_model, "planner swapped the nested text stack"

  # One plan per decoder layer, each sharding the 7 projections of the block.
  assert len(layer_plans) == _NUM_TEXT_LAYERS, (
    f"expected {_NUM_TEXT_LAYERS} layer plans, got {len(layer_plans)}")
  expected = {
    "self_attn.q_proj": "ColwiseParallel",
    "self_attn.k_proj": "ColwiseParallel",
    "self_attn.v_proj": "ColwiseParallel",
    "self_attn.o_proj": "RowwiseParallel",
    "mlp.gate_proj": "ColwiseParallel",
    "mlp.up_proj": "ColwiseParallel",
    "mlp.down_proj": "RowwiseParallel",
  }
  for layer_plan in layer_plans:
    got = {name: style.__class__.__name__ for name, style in layer_plan.items()}
    assert got == expected, f"unexpected layer plan: {got}"

  # Every layer, not just the first one, must actually hold sharded weights.
  q_local = (_HEADS * _HEAD_DIM // world, _TEXT_HIDDEN_SIZE)
  kv_local = (_KV_HEADS * _HEAD_DIM // world, _TEXT_HIDDEN_SIZE)
  down_local = (_TEXT_HIDDEN_SIZE, _INTERMEDIATE // world)
  for index, block in enumerate(language_model.layers):
    assert _local_shape(block.self_attn.q_proj.weight) == q_local, f"layer {index} q_proj"
    assert _local_shape(block.self_attn.k_proj.weight) == kv_local, f"layer {index} k_proj"
    assert _local_shape(block.self_attn.v_proj.weight) == kv_local, f"layer {index} v_proj"
    assert _local_shape(block.self_attn.o_proj.weight) == (_TEXT_HIDDEN_SIZE, _HEADS * _HEAD_DIM //
                                                           world), f"layer {index} o_proj"
    assert _local_shape(
      block.mlp.gate_proj.weight) == (_INTERMEDIATE // world,
                                      _TEXT_HIDDEN_SIZE), f"layer {index} gate_proj"
    assert _local_shape(block.mlp.down_proj.weight) == down_local, f"layer {index} down_proj"
    # `.shape` of a DTensor stays global, so the shard above is not a re-shaped copy.
    assert tuple(block.self_attn.q_proj.weight.shape) == (
      _HEADS * _HEAD_DIM, _TEXT_HIDDEN_SIZE), f"layer {index} q_proj global shape"
  # The embedding stays replicated: vocabulary sharding would need a gather before the taps.
  assert not hasattr(language_model.embed_tokens.weight, "to_local")

  if rank == 0:
    print(f"[qwen3vl-te-tp] planner={planner_cls.__name__} layer_plans={len(layer_plans)} "
          f"plan_keys={len(expected)} sharded_layers={len(language_model.layers)}")


def _run_rank(rank: int, world: int, dtype_name: str) -> None:
  os.environ["MASTER_ADDR"] = "127.0.0.1"
  os.environ["MASTER_PORT"] = str(_PORT)
  torch.cuda.set_device(rank)
  torch.distributed.init_process_group("nccl", rank=rank, world_size=world)
  device = f"cuda:{rank}"

  try:
    from cache_dit.distributed import ParallelismConfig
    from cache_dit.distributed.text_encoders import parallelize_text_encoder
    from cache_dit.utils import check_text_encoder

    dtype = getattr(torch, dtype_name)
    _check_planner_contract(rank, world, device)

    inputs = _language_inputs(device)
    model = _build_model(device, dtype)
    language_model = model.language_model
    # `enable_parallelism(transformer, cfg(extra_parallel_modules=[text_encoder]))` dispatches to
    # this very entry for any transformers module, so the path under test is the pipeline's.
    assert check_text_encoder(model) is True

    # TP-free reference, computed on rank 0 only.
    if rank == 0:
      reference = [_tapped_hidden_states(model, inputs), _last_hidden_state(model, inputs)]
    torch.distributed.barrier()

    parallelized = parallelize_text_encoder(model, ParallelismConfig(tp_size=world))
    assert parallelized is model, "parallelize_text_encoder returned a different module"
    assert type(parallelized) is Qwen3VLModel
    assert parallelized.language_model is language_model
    assert getattr(
      parallelized, "_is_parallelized",
      False) is True, ("_is_parallelized was not set, so the flag lives on a different object")
    # `record_plans` runs on whatever the planner returns: if the plan had been recorded on the
    # nested text stack these attributes would be missing from the wrapper.
    assert set(parallelized._colwise_layers) == {
      "self_attn.q_proj",
      "self_attn.k_proj",
      "self_attn.v_proj",
      "mlp.gate_proj",
      "mlp.up_proj",
    }, parallelized._colwise_layers
    assert set(parallelized._rowwise_layers) == {"self_attn.o_proj",
                                                 "mlp.down_proj"}, parallelized._rowwise_layers

    got = [_tapped_hidden_states(parallelized, inputs), _last_hidden_state(parallelized, inputs)]

    got_flat = got[0] + [got[1]]
    if rank == 0:
      ref_flat = reference[0] + [reference[1]]
    else:
      ref_flat = [torch.empty_like(tensor) for tensor in got_flat]
    # The reference is broadcast from rank 0 so every rank also shows it converged to the same
    # full (replicated) output instead of two different half-correct shards. NCCL needs the
    # staging buffers on the GPU.
    for ref, mine in zip(ref_flat, got_flat):
      assert ref.shape == mine.shape, f"shape changed: {tuple(mine.shape)} vs {tuple(ref.shape)}"
    for ref in ref_flat:
      staged = ref.to(device)
      torch.distributed.broadcast(staged, src=0)
      ref.copy_(staged)

    names = [f"layer{index}" for index in QWEN3_VL_ACTIVATION_LAYERS] + ["last_hidden_state"]
    worst = 0.0
    for name, got_tensor, ref_tensor in zip(names, got_flat, ref_flat):
      rel = _rel_l2(got_tensor, ref_tensor)
      worst = max(worst, rel)
      max_abs = (got_tensor - ref_tensor).abs().max().item()
      if rank == 0:
        print(f"[qwen3vl-te-tp|{dtype_name}] {name} rel_l2={rel:.3e} max_abs={max_abs:.3e} "
              f"shape={tuple(got_tensor.shape)}")
      assert rel < _REL_TOL[dtype_name], f"{name} diverged (rel_l2={rel:.3e})"
    if rank == 0:
      print(f"[qwen3vl-te-tp|{dtype_name}] worst rel_l2={worst:.3e} "
            f"tol={_REL_TOL[dtype_name]:.1e} tensors={len(got_flat)}")
  finally:
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("dtype_name", _DTYPES)
def test_qwen3_vl_model_tp_matches_single_rank(dtype_name):
  mp.spawn(_run_rank, args=(2, dtype_name), nprocs=2, join=True)
