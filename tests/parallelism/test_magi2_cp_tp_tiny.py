"""Multi-rank parity test for the MAGI-2 Preview CP and TP plumbing (tiny model, no weights).

Two properties are pinned here that no single-rank test can reach:

* **CP**: the model shards the packed sequence itself and the packed attention performs the Ulysses
  head/sequence exchange with uneven `split_sizes`. The packed batch is built with **different text
  lengths per sample**, so the total token count is odd and the split is genuinely uneven.
* **TP**: the grouped-linear column/row sharding plus the MoE head-axis sharding must reproduce the
  replicated model after the collection all-reduces.

Usage:
  python -m pytest tests/parallelism/test_magi2_cp_tp_tiny.py -q
"""
import pytest
import torch
import torch.multiprocessing as mp

from cache_dit.models.pipelines.magi2.configuration import (
  Magi2MHCConfig,
  Magi2MoEConfig,
  Magi2PreviewConfig,
)
from cache_dit.models.pipelines.magi2.attention import VarlenHandler
from cache_dit.models.pipelines.magi2.data_proxy import Magi2DataProxy, ModelInput
from cache_dit.models.pipelines.magi2.parallel import reset_magi2_groups, set_magi2_groups
from cache_dit.models.pipelines.magi2.transformer import Magi2PreviewTransformer

_PORT = 29611
# Ulysses only reorders tokens and heads, so CP should reproduce the single-rank result to fp32 noise.
_REL_TOL_CP = 1e-5
_REL_TOL_CP_BF16 = 5e-3
# TP shards the projections; the all-reduce changes the accumulation order of every row-parallel matmul.
_REL_TOL_TP = 1e-5
_REL_TOL_TP_BF16 = 2e-2

_HIDDEN = 64
_HEAD_DIM = 8
_NUM_LAYERS = 2
_LATENT = (1, 4, 2, 4, 4)
_AUDIO_TOKENS = 6
_LATENT_FRAMES = 2


def _config(dtype) -> Magi2PreviewConfig:
  config = Magi2PreviewConfig(
    num_layers=_NUM_LAYERS,
    hidden_size=_HIDDEN,
    head_dim=_HEAD_DIM,
    num_query_groups=8,
    video_in_channels=4,
    audio_in_channels=4,
    text_in_channels=8,
    multimodal_layers=(0, ),
    params_dtype=dtype,
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
  return config


def _build(dtype) -> Magi2PreviewTransformer:
  """Build the tiny transformer with deterministic weights (identical on every rank)."""

  generator = torch.Generator().manual_seed(0)
  model = Magi2PreviewTransformer(_config(dtype)).eval()
  with torch.no_grad():
    for parameter in model.parameters():
      parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)
  return model


def _packed_inputs(dtype):
  """Packed inputs whose two samples have different text lengths (odd total token count)."""

  generator = torch.Generator().manual_seed(3)
  data = ModelInput(
    x_t=torch.randn(2, 4, _LATENT_FRAMES, 4, 4, generator=generator),
    audio_x_t=torch.randn(2, _AUDIO_TOKENS, 4, generator=generator),
    audio_feat_len=[_AUDIO_TOKENS, _AUDIO_TOKENS],
    txt_feat=torch.randn(2, 5, 8, generator=generator),
    txt_feat_len=[5, 2],
    t=torch.tensor([0.4, 0.4]),
    per_token_video_t=torch.full((2, 1, _LATENT_FRAMES, 4, 4), 0.4),
    per_token_audio_t=torch.full((2, _AUDIO_TOKENS, 1), 0.4),
  )
  packed = Magi2DataProxy().process_input(data)
  args = tuple(
    value.to(dtype=dtype) if torch.is_tensor(value) and value.is_floating_point() else value
    for value in packed.model_args)
  return args


def _move_args(model_args, device):
  """Move a packed input tuple (including the varlen bounds) to ``device``."""

  from dataclasses import replace

  moved = []
  for value in model_args:
    if torch.is_tensor(value):
      moved.append(value.to(device))
    elif isinstance(value, VarlenHandler):
      moved.append(
        replace(
          value,
          cu_seqlens_q=value.cu_seqlens_q.to(device),
          cu_seqlens_k=value.cu_seqlens_k.to(device),
        ))
    else:
      moved.append(value)
  return tuple(moved)


def _reference(model, model_args):
  """Full-sequence (world-size 1) output, computed with no group installed."""

  reset_magi2_groups()
  with torch.no_grad():
    return model(*model_args)


def _relative_error(actual, expected) -> float:
  denominator = expected.abs().max().clamp_min(1e-6)
  return float((actual - expected).abs().max() / denominator)


def _run_rank(rank: int, world: int, mode: str, dtype, tolerance: float) -> None:
  import torch.distributed as dist

  torch.cuda.set_device(rank)
  dist.init_process_group("nccl",
                          init_method=f"tcp://localhost:{_PORT}",
                          rank=rank,
                          world_size=world)
  try:
    model = _build(dtype).to(device=f"cuda:{rank}")
    model_args = _move_args(_packed_inputs(dtype), f"cuda:{rank}")

    reference = _reference(model, model_args)
    group = dist.new_group(list(range(world)))

    if mode == "cp":
      set_magi2_groups(ulysses_group=group)
      with torch.no_grad():
        actual = model(*model_args)
      assert isinstance(actual, torch.Tensor)
      assert actual.shape == reference.shape, f"{actual.shape} != {reference.shape}"
    else:
      set_magi2_groups(tp_group=group)
      model.shard_for_tp(
        __import__("cache_dit.models.pipelines.magi2.parallel",
                   fromlist=["get_magi2_tp_group"]).get_magi2_tp_group())
      with torch.no_grad():
        actual = model(*model_args)
      assert actual.shape == reference.shape, f"{actual.shape} != {reference.shape}"

    error = _relative_error(actual, reference)
    error_tensor = torch.tensor([error], device=f"cuda:{rank}")
    dist.all_reduce(error_tensor, op=dist.ReduceOp.MAX)
    worst = float(error_tensor.item())
    assert worst <= tolerance, f"{mode} rel error {worst:.3e} exceeds {tolerance:.3e} on rank {rank}"
    if rank == 0:
      print(f"[magi2 tiny {mode}] rel_error={worst:.3e} tol={tolerance:.3e}")
  finally:
    dist.destroy_process_group()


def _require_two_gpus() -> None:
  if torch.cuda.device_count() < 2:
    pytest.skip("MAGI-2 CP/TP tiny parity needs two GPUs")


@pytest.mark.parametrize("mode", ["cp", "tp"])
def test_magi2_parallel_matches_single_rank(mode):
  """fp32 parity: the parallel layouts must reproduce the replicated model tightly."""

  _require_two_gpus()
  tolerance = _REL_TOL_CP if mode == "cp" else _REL_TOL_TP
  mp.spawn(_run_rank, args=(2, mode, torch.float32, tolerance), nprocs=2, join=True)


@pytest.mark.parametrize("mode", ["cp", "tp"])
def test_magi2_parallel_is_stable_in_bfloat16(mode):
  """Bf16 shares the release dtype; the tolerance only has to separate drift from misalignment."""

  _require_two_gpus()
  tolerance = _REL_TOL_CP_BF16 if mode == "cp" else _REL_TOL_TP_BF16
  mp.spawn(_run_rank, args=(2, mode, torch.bfloat16, tolerance), nprocs=2, join=True)
