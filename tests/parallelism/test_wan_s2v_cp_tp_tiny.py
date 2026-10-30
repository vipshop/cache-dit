"""Multi-rank parity test for the Wan2.2-S2V CP and TP plumbing (tiny model, no weights).

What makes S2V different from the other Wan variants, and what this test covers:

- The sequence is `[video tokens | reference-image tokens | motion tokens]`, and the modulation
  boundary and the audio injection both depend on where the video region ends. A CP shard is a
  plain contiguous chunk, so rank 1 of 2 starts in the middle of the video region here; the audio
  injection has to rebuild whole frames from the gather before it attends.
- The RoPE table is split at the first block, and `inject_motion` concatenates the motion-token
  rotary embedding before the blocks run, so a wrong split point shows up as a coherent but
  different clip.
- The audio injector is a cross-attention stack outside `blocks`; a TP plan that forgets it leaves
  the residual computed on replicated weights, which still returns the right shape.

A wrong split or a missing gather produces tensors of the right shape, so the check compares
against a single-rank reference computed on rank 0.

Usage:
  CUDA_VISIBLE_DEVICES=2,3 python -m pytest tests/parallelism/test_wan_s2v_cp_tp_tiny.py -q
"""

import os

import pytest
import torch
import torch.multiprocessing as mp

from cache_dit.models.pipelines.wan_s2v.transformer import WanS2VTransformer3DModel

_GRID = 8
_LATENT_CHANNELS = 16
_VIDEO_FRAMES = 5
_LATENT_MOTION_FRAMES = 19
_AUDIO_FRAMES = 20  # gives exactly `_VIDEO_FRAMES` audio frames after the causal encoder
_TEXT_DIM = 32
_TEXT_TOKENS = 8
_SEQ_LEN = 136  # 80 video + 16 reference + 40 motion tokens, divisible by 2
_REL_TOL_CP = 2e-3
_REL_TOL_TP = 6e-3
_PORT = 29631


def _build(device, dtype=torch.bfloat16) -> WanS2VTransformer3DModel:
  torch.manual_seed(0)
  transformer = WanS2VTransformer3DModel(
    patch_size=(1, 2, 2),
    num_attention_heads=4,
    attention_head_dim=16,
    in_channels=_LATENT_CHANNELS,
    out_channels=_LATENT_CHANNELS,
    text_dim=_TEXT_DIM,
    freq_dim=32,
    audio_dim=1024,
    audio_inject_layers=(0, 2, 4, 6),
    enable_adain=True,
    adain_mode="attn_norm",
    pose_dim=16,
    ffn_dim=128,
    num_layers=8,
    num_weighted_avg_layers=25,
    cross_attn_norm=True,
    qk_norm="rms_norm_across_heads",
    eps=1e-6,
    added_kv_proj_dim=None,
    rope_max_seq_len=1024,
    enable_framepack=True,
    framepack_drop_mode="padd",
    add_last_motion=True,
    zero_timestep=True,
  )
  return transformer.to(device=device, dtype=dtype).eval()


def _inputs(dtype=torch.bfloat16) -> dict:
  generator = torch.Generator().manual_seed(3)
  return dict(
    hidden_states=torch.randn(1, _LATENT_CHANNELS, _VIDEO_FRAMES, _GRID, _GRID,
                              generator=generator).to(dtype),
    timestep=torch.tensor([750.0]),
    encoder_hidden_states=torch.randn(1, _TEXT_TOKENS, _TEXT_DIM, generator=generator).to(dtype),
    motion_latents=torch.randn(1,
                               _LATENT_CHANNELS,
                               _LATENT_MOTION_FRAMES,
                               _GRID,
                               _GRID,
                               generator=generator).to(dtype),
    audio_embeds=torch.randn(1, 25, 1024, _AUDIO_FRAMES, generator=generator).to(dtype),
    image_latents=torch.randn(1, _LATENT_CHANNELS, 1, _GRID, _GRID, generator=generator).to(dtype),
    pose_latents=torch.zeros(1, 16, _VIDEO_FRAMES, _GRID, _GRID).to(dtype),
    motion_frames=[73, _LATENT_MOTION_FRAMES],
    drop_motion_frames=False,
  )


def _to_device(inputs: dict, device) -> dict:
  moved = {}
  for key, value in inputs.items():
    moved[key] = value.to(device) if torch.is_tensor(value) else value
  return moved


def _run_rank(rank, world, mode, dtype=torch.bfloat16, tolerance_override=None):
  os.environ["MASTER_ADDR"] = "127.0.0.1"
  os.environ["MASTER_PORT"] = str(_PORT)
  torch.cuda.set_device(rank)
  device = torch.device("cuda", rank)
  torch.distributed.init_process_group("nccl", rank=rank, world_size=world)

  try:
    import cache_dit
    from cache_dit import DBCacheConfig
    from cache_dit.caching.block_adapters import BlockAdapterRegister
    from cache_dit.distributed.config import ParallelismConfig

    inputs = _inputs(dtype)
    transformer = _build(device, dtype)

    reference = None
    if rank == 0:
      with torch.no_grad():
        reference = transformer(**_to_device(inputs, device), return_dict=False)[0].float().cpu()
    torch.distributed.barrier()

    parallelism_config = (ParallelismConfig(
      ulysses_size=world) if mode == "cp" else ParallelismConfig(tp_size=world))
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
      out = transformer(**_to_device(inputs, device), return_dict=False)[0].float().cpu()

    if rank == 0:
      relative = (out - reference).abs().mean() / reference.abs().mean().clamp_min(1e-6)
      tolerance = tolerance_override or (_REL_TOL_CP if mode == "cp" else _REL_TOL_TP)
      print(f"[wan-s2v-{mode}] rel={relative.item():.6f} shape={tuple(out.shape)}")
      assert out.shape == reference.shape, f"{mode} shape changed: {out.shape}"
      assert relative.item() < tolerance, f"{mode} diverged (rel={relative.item():.6f})"
  finally:
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("mode", ["cp", "tp"])
def test_wan_s2v_parallel_matches_single_rank(mode):
  mp.spawn(_run_rank, args=(2, mode), nprocs=2, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_wan_s2v_cp_frame_misaligned_shard():
  """fp32 tightens the CP tolerance: the audio gather must not lose or reorder a token."""
  mp.spawn(_run_rank, args=(2, "cp", torch.float32, 1e-5), nprocs=2, join=True)
