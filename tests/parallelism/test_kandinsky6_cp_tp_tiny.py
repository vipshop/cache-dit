"""Multi-rank parity test for the Kandinsky 6.0 CP and TP plumbing (tiny model, no weights).

Kandinsky 6.0 denoises a video and an audio stream through one fused block stack. Context
parallelism shards the video stream only, gathers it back on the last block's output (the audio
stream stays replicated), and rebuilds the video KV for the audio->video cross-attention. A wrong
gather still yields tensors of the right shape, so this test compares both streams against a cp-free
reference computed on rank 0.

Caching is disabled here (the warmup covers the whole run, so no step can be served from cache):
this test isolates the parallel paths. The `Bn_compute_blocks >= 1` requirement that the CP gather
implies is covered separately in `tests/api/test_kandinsky6_adapter.py`.

Usage:
  CUDA_VISIBLE_DEVICES=5,6 python -m pytest tests/parallelism/test_kandinsky6_cp_tp_tiny.py -q
"""

import os

import pytest
import torch
import torch.multiprocessing as mp

_TEXT_LEN = 5
_FRAMES = 2
_LATENT_H = 4
_LATENT_W = 4
_VISUAL_IN = 4
_AUDIO_IN = 4
_AUDIO_LEN = 3
_MODEL_DIM = 32
_AUDIO_DIM = 16
_REL_TOL = 2e-2  # bf16 all-reduce order noise on a 2-layer random model
_PORT = 29582


def _build(device):
  from diffusers import Kandinsky6Transformer3DModel

  torch.manual_seed(0)
  transformer = Kandinsky6Transformer3DModel(
    in_visual_dim=_VISUAL_IN,
    out_visual_dim=_VISUAL_IN,
    in_text_dim=8,
    in_text_dim2=8,
    time_dim=8,
    patch_size=(1, 2, 2),
    model_dim=_MODEL_DIM,
    ff_dim=2 * _MODEL_DIM,
    num_text_blocks=1,
    num_visual_blocks=2,
    axes_dims=(2, 2, 4),
    visual_cond=False,
    in_audio_dim=_AUDIO_IN,
    out_audio_dim=_AUDIO_IN,
    model_dim_a=_AUDIO_DIM,
    time_dim_a=8,
    ff_dim_a=2 * _AUDIO_DIM,
    axes_dims_a=(2, 2, 4),
    ca_rope=True,
    cross_gates=True,
    fix_modulation=True,
    text_token_padding=True,
  )
  return transformer.to(device=device, dtype=torch.bfloat16).eval()


def _inputs():
  generator = torch.Generator().manual_seed(3)
  return {
    "hidden_states":
    torch.randn(1, _FRAMES, _LATENT_H, _LATENT_W, _VISUAL_IN,
                generator=generator).to(torch.bfloat16),
    "audio_hidden_states":
    torch.randn(1, _AUDIO_LEN, _AUDIO_IN, generator=generator).to(torch.bfloat16),
    "encoder_hidden_states":
    torch.randn(1, _TEXT_LEN, 8, generator=generator).to(torch.bfloat16),
    "pooled_projections":
    torch.randn(1, 8, generator=generator).to(torch.bfloat16),
    "timestep":
    torch.tensor([0.5]),
  }


def _to_device(inputs, device):
  return {
    key: (value.to(device) if torch.is_tensor(value) else value)
    for key, value in inputs.items()
  }


def _run_rank(rank: int, world: int, mode: str) -> None:
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

    inputs = _inputs()
    transformer = _build(device)

    reference = None
    if rank == 0:
      with torch.no_grad():
        video, audio = transformer(**_to_device(inputs, device), return_dict=False)
        reference = (video.float().cpu(), audio.float().cpu())
    torch.distributed.barrier()

    if mode == "cp":
      parallelism_config = ParallelismConfig(ulysses_size=world)
    else:
      parallelism_config = ParallelismConfig(tp_size=world)

    cache_dit.enable_cache(
      BlockAdapterRegister.get_adapter(transformer),
      cache_config=DBCacheConfig(
        Fn_compute_blocks=1,
        # The CP gather lives on the last block's output: keep a tail block in every path.
        Bn_compute_blocks=1,
        max_warmup_steps=100,
        max_cached_steps=0,
        num_inference_steps=8,
        enable_separate_cfg=False,
        residual_diff_threshold=0.0,
      ),
      parallelism_config=parallelism_config,
    )

    # Guard against a no-op parallelization: a silently unsharded transformer would also "match".
    assert getattr(transformer, "_is_parallelized", False), f"{mode} left the transformer unsharded"

    with torch.no_grad():
      video, audio = transformer(**_to_device(inputs, device), return_dict=False)
    video, audio = video.float().cpu(), audio.float().cpu()

    if rank == 0:
      for name, out, ref in (("video", video, reference[0]), ("audio", audio, reference[1])):
        relative = (out - ref).abs().mean() / ref.abs().mean().clamp_min(1e-6)
        print(f"[kandinsky6-{mode}-{name}] rel={relative.item():.6f} shape={tuple(out.shape)}")
        assert out.shape == ref.shape, f"{mode} {name} shape changed: {out.shape}"
        assert relative.item() < _REL_TOL, f"{mode} {name} diverged (rel={relative.item():.6f})"
  finally:
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("mode", ["cp", "tp"])
def test_kandinsky6_parallel_matches_single_rank(mode):
  mp.spawn(_run_rank, args=(2, mode), nprocs=2, join=True)
