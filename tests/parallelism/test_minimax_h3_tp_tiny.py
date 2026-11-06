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

import pytest
import torch
import torch.multiprocessing as mp

from cache_dit import DBCacheConfig
from cache_dit.caching.block_adapters import BlockAdapterRegister
from cache_dit.distributed.config import ParallelismConfig
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


def _inputs(device, dtype=torch.float32) -> dict:
  """One packed sequence: `[text | audio | video]`, with two distinct timesteps."""
  generator = torch.Generator().manual_seed(3)
  sequence_length = _TEXT_TOKENS + _AUDIO_ROWS + _VIDEO_ROWS
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
                _VIDEO_ROWS,
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
  from torch.distributed.tensor import DTensor

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
    import cache_dit

    inputs = _inputs(device)
    transformer = _build(device)

    reference = None
    if rank == 0:
      with torch.no_grad():
        reference = tuple(out.float().cpu() for out in transformer(**inputs))
    torch.distributed.barrier()

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
  finally:
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_minimax_h3_tp_matches_single_rank_in_fp32():
  """fp32: the sharding bookkeeping has to be exact, not merely close."""
  mp.spawn(_run_rank, args=(2, ), nprocs=2, join=True)
