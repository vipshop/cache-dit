"""Multi-rank parity test for the MiniMax-H3-Image CP / TP planners (tiny model, no weights).

MiniMax-H3-Image is MiniMax-H3 without its audio stream, and one structural difference decides the
plans: the checkpoint is distilled with **one output head per sampling step** (`pdd_proj_out`, a
`ModuleList` of six `Linear`), applied after the block stack. A plan that keeps the video model's
`proj_out` gather therefore gathers nothing and hands every rank a partial sequence, with no shape
error anywhere — the failure this test exists for.

The test pins, on a two-layer fp32 transformer compared against the unsharded output on rank 0:

* the block stack is really sharded (weights are `DTensor`s with sharded placements), because a
  planner that silently skips sharding would also match bit for bit;
* the fused SwiGLU projection keeps its documented `(up_i, gate_i)` layout, since a missed
  interleave keeps every shape valid while each rank multiplies a gate half with itself;
* the CP plan gathers at the per-step head and returns the full image-row output.

Usage:
  CUDA_VISIBLE_DEVICES=0,1 python -m pytest tests/parallelism/test_minimax_h3_image_cp_tp_tiny.py -q
"""

import os

import pytest
import torch
import torch.multiprocessing as mp

from cache_dit import DBCacheConfig
from cache_dit.caching.block_adapters import BlockAdapterRegister
from cache_dit.distributed.config import ParallelismConfig
from diffusers import MiniMaxH3ImageTransformer2DModel

_PORT = 29596
_HEADS = 4
_HEAD_DIM = 12
_HIDDEN = 16
_FFN = 32
_PATCH = (1, 2, 2)
_IN_CHANNELS = 4
_TEXT_DIM = 32
_TEXT_TOKENS = 5
_IMAGE_ROWS = 6
_REL_TOL = 1e-6


def _build(device, dtype=torch.float32) -> MiniMaxH3ImageTransformer2DModel:
  torch.manual_seed(0)
  transformer = MiniMaxH3ImageTransformer2DModel(
    num_attention_heads=_HEADS,
    attention_head_dim=_HEAD_DIM,
    hidden_size=_HIDDEN,
    num_layers=2,
    num_refiner_layers=1,
    ffn_dim=_FFN,
    in_channels=_IN_CHANNELS,
    patch_size=_PATCH,
    text_dim=_TEXT_DIM,
    freq_dim=16,
    time_embed_hidden_dim=32,
    time_embed_dim=16,
    rope_freq_dim=2,
  )
  return transformer.to(device=device, dtype=dtype).eval()


def _inputs(device, dtype=torch.float32) -> dict:
  """One packed sequence: `[text | image rows]`, with two distinct timesteps."""
  generator = torch.Generator().manual_seed(3)
  sequence_length = _TEXT_TOKENS + _IMAGE_ROWS
  text_indices = torch.arange(0, _TEXT_TOKENS)
  image_indices = torch.arange(_TEXT_TOKENS, sequence_length)
  token_tags = torch.zeros(sequence_length, dtype=torch.long)
  token_tags[text_indices] = 1
  # The generated rows share one noise level; the text rows sit at a second one.
  timestep_indices = torch.zeros(sequence_length, dtype=torch.long)
  timestep_indices[text_indices] = 1
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
                _IMAGE_ROWS,
                _IN_CHANNELS * _PATCH[0] * _PATCH[1] * _PATCH[2],
                generator=generator).to(device=device, dtype=dtype),
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
    "image_indices":
    image_indices.to(device),
    "text_indices":
    text_indices.to(device),
    "pdd_step_index":
    0,
    "return_dict":
    False,
  }


def _relative(out, reference) -> float:
  return ((out - reference).abs().mean() / reference.abs().mean().clamp_min(1e-6)).item()


def _assert_sharded(transformer: MiniMaxH3ImageTransformer2DModel, tp_size: int) -> None:
  """Fail when the planner did not actually shard the block stack."""
  from torch.distributed.tensor import DTensor

  block = transformer.transformer_blocks[0]
  for name, module in (("attn.to_q", block.attn.to_q), ("attn.to_out.0", block.attn.to_out[0]),
                       ("ff.net.0.proj", block.ff.net[0].proj), ("ff.net.2", block.ff.net[2]),
                       ("adaln_proj.linear", block.adaln_proj.linear)):
    weight = module.weight
    assert isinstance(weight, DTensor), f"{name}.weight is not a DTensor: the stack was not sharded"
    local = weight.to_local()
    if name in ("attn.to_out.0", "ff.net.2"):
      # Rowwise shards the input columns.
      assert local.shape[1] * tp_size == weight.shape[1], (name, local.shape, weight.shape)
    else:
      assert local.shape[0] * tp_size == weight.shape[0], (name, local.shape, weight.shape)
  # Replicated on purpose: `forward` scatters the packed rows with global indices, and the residual
  # stream stays full-width, so `norm_out` and the per-step heads read a complete tensor.
  for name, module in (("proj_in", transformer.proj_in), ("context_embedder",
                                                          transformer.context_embedder),
                       ("pdd_proj_out.0", transformer.pdd_proj_out[0])):
    assert not isinstance(module.weight, DTensor), f"{name} must stay replicated"


def _run_rank(rank: int, world: int, mode: str) -> None:
  os.environ["MASTER_ADDR"] = "127.0.0.1"
  os.environ["MASTER_PORT"] = str(_PORT + {"tp": 0, "cp": 1, "cp_tp": 2}[mode])
  torch.cuda.set_device(rank)
  torch.distributed.init_process_group("nccl", rank=rank, world_size=world)
  device = f"cuda:{rank}"

  try:
    import cache_dit

    inputs = _inputs(device)
    transformer = _build(device)

    # Every sampling step runs its own output head, and the CP gather is registered on the whole
    # `pdd_proj_out` ModuleList: one head would still pass if the wildcard only hooked the first one.
    steps = list(range(transformer.config.num_pdd_heads))
    reference = None
    if rank == 0:
      with torch.no_grad():
        reference = [
          transformer(**{
            **inputs, "pdd_step_index": step
          })[0].float().cpu() for step in steps
        ]
    torch.distributed.barrier()

    if mode == "tp":
      parallel_config = ParallelismConfig(tp_size=world)
    elif mode == "cp_tp":
      # 2x2 mesh: the sequence split and the weight split have to be independent.
      parallel_config = ParallelismConfig(ulysses_size=2, tp_size=2, ulysses_anything=True)
    else:
      parallel_config = ParallelismConfig(ulysses_size=world, ulysses_anything=True)
    cache_dit.enable_cache(
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
      parallelism_config=parallel_config,
    )
    if mode in ("tp", "cp_tp"):
      _assert_sharded(transformer, 2 if mode == "cp_tp" else world)

    with torch.no_grad():
      outputs = [
        transformer(**{
          **inputs, "pdd_step_index": step
        })[0].float().cpu() for step in steps
      ]

    if rank == 0:
      for step, (output, expected) in enumerate(zip(outputs, reference)):
        assert output.shape == expected.shape, (step, output.shape, expected.shape)
        relative = _relative(output, expected)
        print(f"[minimax-h3-image-{mode}] head {step} rel={relative:.3e}")
        assert relative < _REL_TOL, f"head {step} diverged under {mode} ({relative:.3e})"
  finally:
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_minimax_h3_image_tp_matches_single_rank_in_fp32():
  mp.spawn(_run_rank, args=(2, "tp"), nprocs=2, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_minimax_h3_image_cp_matches_single_rank_in_fp32():
  mp.spawn(_run_rank, args=(2, "cp"), nprocs=2, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs four GPUs")
def test_minimax_h3_image_tp2_cp2_hybrid_matches_single_rank_in_fp32():
  mp.spawn(_run_rank, args=(4, "cp_tp"), nprocs=4, join=True)
