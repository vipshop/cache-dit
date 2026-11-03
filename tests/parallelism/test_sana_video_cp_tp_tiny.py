"""Multi-rank parity test for the SANA-Video CP and TP plumbing (tiny model, no weights).

The CP plan shards the **frame axis** of the 5D latent grid, so every shape the block derives
(`post_patch_num_frames/height/width`, the `unflatten` in the feed-forward, the unpatchify) is local
on each rank. Two patches have to make that sharded computation equal to the single-rank one:

* the linear self-attention replaces its two per-token sums with a global SUM all-reduce;
* `conv_temp` exchanges one halo frame per side, so the 3-tap temporal convolution sees the same
  neighbours at a chunk boundary as it does in the full-sequence run.

A wrong RoPE window or a missing halo still returns correctly shaped tensors, so this test compares
against a cp-free reference computed on rank 0. CPU unit tests cover the plan strings; only this test
proves the numbers.

Usage:
  CUDA_VISIBLE_DEVICES=6,7 python -m pytest tests/parallelism/test_sana_video_cp_tp_tiny.py -q
"""

import os

import pytest
import torch
import torch.multiprocessing as mp

_IN_CHANNELS = 4
_HEADS = 4
_HEAD_DIM = 8
_LAYERS = 2
# Two latent frames so the frame axis splits evenly across two ranks.
_FRAMES = 2
_LATENT_H = 8
_LATENT_W = 8
_TEXT_TOKENS = 5
_CAPTION_CHANNELS = 8
# CP reproduces the single-rank result up to the all-reduce accumulation order; TP additionally
# reorders every projection's reduction.
_REL_TOL_CP = 4e-3
_REL_TOL_TP = 8e-3
_REL_TOL_FP32 = 1e-5
_PORT = 29583


def _build(device, dtype=torch.bfloat16):
  from diffusers import SanaVideoTransformer3DModel

  torch.manual_seed(0)
  transformer = SanaVideoTransformer3DModel(
    in_channels=_IN_CHANNELS,
    out_channels=_IN_CHANNELS,
    num_attention_heads=_HEADS,
    attention_head_dim=_HEAD_DIM,
    num_layers=_LAYERS,
    num_cross_attention_heads=_HEADS,
    cross_attention_head_dim=_HEAD_DIM,
    cross_attention_dim=_HEADS * _HEAD_DIM,
    caption_channels=_CAPTION_CHANNELS,
    mlp_ratio=2.0,
    sample_size=_LATENT_H,
    patch_size=(1, 2, 2),
  )
  return transformer.to(device=device, dtype=dtype).eval()


def _inputs(dtype=torch.bfloat16, frames: int = _FRAMES, i2v: bool = False):
  generator = torch.Generator().manual_seed(3)
  # I2V passes a per-frame 5D timestep that flattens to the same token grid as `hidden_states`; T2V
  # passes a 1D `[B]` one, which the `expected_dims=5` guard leaves alone.
  # The 5D grid is already patchified (the tiny model patches (1, 2, 2)), so it matches the token
  # count of `hidden_states` after the transformer's own patchify.
  timestep = torch.zeros(1, 1, frames, _LATENT_H // 2, _LATENT_W //
                         2, dtype=dtype) if i2v else torch.zeros(1, dtype=dtype)
  return {
    "hidden_states":
    torch.randn(1, _IN_CHANNELS, frames, _LATENT_H, _LATENT_W, generator=generator).to(dtype),
    "encoder_hidden_states":
    torch.randn(1, _TEXT_TOKENS, _CAPTION_CHANNELS, generator=generator).to(dtype),
    "timestep":
    timestep,
    "return_dict":
    False,
  }


def _to_device(inputs, device):
  return {
    key: (value.to(device) if torch.is_tensor(value) else value)
    for key, value in inputs.items()
  }


def _run_rank(rank: int,
              world: int,
              mode: str,
              dtype=torch.bfloat16,
              tolerance_override=None,
              frames: int = _FRAMES,
              uaa: bool = False,
              i2v: bool = False) -> None:
  os.environ["MASTER_ADDR"] = "127.0.0.1"
  os.environ["MASTER_PORT"] = str(_PORT)
  torch.cuda.set_device(rank)
  torch.distributed.init_process_group("nccl", rank=rank, world_size=world)
  device = f"cuda:{rank}"

  try:
    import cache_dit
    from cache_dit import DBCacheConfig
    from cache_dit.caching.block_adapters import BlockAdapterRegister
    from cache_dit.distributed.config import ParallelismConfig

    device_inputs = _to_device(_inputs(dtype, frames, i2v), device)

    transformer = _build(device, dtype)
    reference = None
    if rank == 0:
      with torch.no_grad():
        reference = transformer(**device_inputs)[0].float().cpu()
    torch.distributed.barrier()

    if mode == "cp":
      parallelism_config = ParallelismConfig(ulysses_size=world, ulysses_anything=uaa)
    elif mode == "cp_tp":
      # 2x2 mesh: the frame split and the attention shard have to be independent.
      parallelism_config = ParallelismConfig(ulysses_size=2, tp_size=2)
    else:
      parallelism_config = ParallelismConfig(tp_size=world)

    cache_dit.enable_cache(
      BlockAdapterRegister.get_adapter(transformer),
      cache_config=DBCacheConfig(
        Fn_compute_blocks=1,
        Bn_compute_blocks=0,
        max_warmup_steps=100,
        max_cached_steps=0,
        num_inference_steps=8,
        enable_separate_cfg=False,
        residual_diff_threshold=0.0,
      ),
      parallelism_config=parallelism_config,
    )

    with torch.no_grad():
      out = transformer(**device_inputs)[0].float().cpu()

    if rank == 0:
      relative = (out - reference).abs().mean() / reference.abs().mean().clamp_min(1e-6)
      if tolerance_override is not None:
        tolerance = tolerance_override
      else:
        tolerance = _REL_TOL_CP if mode == "cp" else _REL_TOL_TP
      print(f"[sana-video-{mode}] rel={relative.item():.6f} shape={tuple(out.shape)}")
      assert out.shape == reference.shape, f"{mode} shape changed: {out.shape}"
      assert relative.item() < tolerance, f"{mode} diverged (rel={relative.item():.6f})"
  finally:
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("mode", ["cp", "tp"])
def test_sana_video_parallel_matches_single_rank(mode):
  mp.spawn(_run_rank, args=(2, mode), nprocs=2, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("mode", ["cp", "tp"])
def test_sana_video_fp32_is_tight(mode):
  """In fp32 the sharding bookkeeping must be exact, i.e. no split or all-reduce misalignment."""

  mp.spawn(_run_rank, args=(2, mode, torch.float32, _REL_TOL_FP32), nprocs=2, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_sana_video_odd_frame_count_needs_ulysses_anything():
  """3 latent frames split 2/1: the halo and the RoPE window must follow the uneven shards."""

  mp.spawn(_run_rank, args=(2, "cp", torch.float32, _REL_TOL_FP32, 3, True), nprocs=2, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("mode", ["cp", "tp"])
def test_sana_video_i2v_timestep_is_sharded(mode):
  """I2V's per-frame timestep must follow the frame split, or the first block raises."""

  mp.spawn(_run_rank,
           args=(2, mode, torch.float32, _REL_TOL_FP32, _FRAMES, False, True),
           nprocs=2,
           join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs four GPUs for a 2x2 mesh")
def test_sana_video_cp_tp_hybrid_matches_single_rank():
  """CP and TP at once: the frame split and the attention shard must compose."""

  mp.spawn(_run_rank, args=(4, "cp_tp", torch.float32, _REL_TOL_FP32), nprocs=4, join=True)
