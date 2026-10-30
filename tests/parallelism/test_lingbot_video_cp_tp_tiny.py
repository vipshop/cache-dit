"""Multi-rank parity test for the LingBot-Video CP and TP plumbing (tiny model, no weights).

LingBot-Video runs one joint `[video, text]` sequence per sample. Context parallelism splits
`hidden_states` at the first block, splits the RoPE table, the token-wise modulation and the timestep
conditioning through their `cp_*` markers, gathers the projected velocity at `cp_out`, and only then
slices the video tokens. A wrong RoPE partition or a gather that runs after the slice still returns
tensors of the right shape, so this test compares against a cp-free reference computed on rank 0.

Usage:
  CUDA_VISIBLE_DEVICES=4,5 python -m pytest tests/parallelism/test_lingbot_video_cp_tp_tiny.py -q
"""

import os

import pytest
import torch
import torch.multiprocessing as mp

from cache_dit.models.pipelines.lingbot_video import LingBotVideoTransformer3DModel

_HEIGHT = 4
_WIDTH = 4
_LATENT_CHANNELS = 16
_TEXT_DIM = 32
_TEXT_TOKENS = 16  # H*W/4 + text = 20, divisible by 2 for plain Ulysses
_TEXT_TOKENS_ODD = 15  # odd joint length, only `--ulysses-anything` can split it
# Ulysses only reorders the sequence, so CP must reproduce the single-rank result almost exactly.
# TP shards the projections, which changes the fp32/bf16 accumulation order of the all-reduce, so it
# carries a wider (still tiny) tolerance.
_REL_TOL_CP = 2e-3
_REL_TOL_TP = 6e-3
_PORT = 29581


def _build(device, dtype=torch.bfloat16):
  torch.manual_seed(0)
  transformer = LingBotVideoTransformer3DModel(
    patch_size=(1, 2, 2),
    in_channels=_LATENT_CHANNELS,
    out_channels=_LATENT_CHANNELS,
    hidden_size=64,
    num_attention_heads=4,
    depth=2,
    intermediate_size=128,
    text_dim=_TEXT_DIM,
    freq_dim=16,
    axes_dims=(4, 6, 6),
    axes_lens=(16, 8, 8),
  )
  return transformer.to(device=device, dtype=dtype).eval()


def _inputs(text_tokens: int = _TEXT_TOKENS, dtype=torch.bfloat16):
  generator = torch.Generator().manual_seed(3)
  return {
    "hidden_states":
    torch.randn(1, _LATENT_CHANNELS, 1, _HEIGHT, _WIDTH, generator=generator).to(dtype),
    "timestep":
    torch.tensor([750.0]),
    "encoder_hidden_states":
    torch.randn(1, text_tokens, _TEXT_DIM, generator=generator).to(dtype),
    "encoder_attention_mask":
    torch.ones(1, text_tokens, dtype=torch.long),
  }


def _to_device(inputs, device):
  return {
    key: (value.to(device) if torch.is_tensor(value) else value)
    for key, value in inputs.items()
  }


def _run_rank(
  rank: int,
  world: int,
  mode: str,
  text_tokens: int,
  uaa: bool,
  dtype=torch.bfloat16,
  tolerance_override=None,
) -> None:
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

    inputs = _inputs(text_tokens, dtype)
    transformer = _build(device, dtype)

    reference = None
    if rank == 0:
      with torch.no_grad():
        reference = transformer(**_to_device(inputs, device), return_dict=False)[0].float().cpu()
    torch.distributed.barrier()

    if mode == "cp":
      parallelism_config = ParallelismConfig(ulysses_size=world, ulysses_anything=uaa)
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
      out = transformer(**_to_device(inputs, device), return_dict=False)[0].float().cpu()

    if rank == 0:
      relative = (out - reference).abs().mean() / reference.abs().mean().clamp_min(1e-6)
      tolerance = tolerance_override or (_REL_TOL_CP if mode == "cp" else _REL_TOL_TP)
      print(f"[lingbot-{mode}-uaa{int(uaa)}] rel={relative.item():.6f} shape={tuple(out.shape)}")
      assert out.shape == reference.shape, f"{mode} shape changed: {out.shape}"
      assert relative.item() < tolerance, f"{mode} diverged (rel={relative.item():.6f})"
  finally:
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("mode", ["cp", "tp"])
def test_lingbot_video_parallel_matches_single_rank(mode):
  mp.spawn(_run_rank, args=(2, mode, _TEXT_TOKENS, False), nprocs=2, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("mode", ["cp", "tp"])
def test_lingbot_video_odd_joint_length(mode):
  # An odd joint length needs `--ulysses-anything` for CP; TP never splits the sequence.
  mp.spawn(_run_rank, args=(2, mode, _TEXT_TOKENS_ODD, mode == "cp"), nprocs=2, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("mode", ["cp", "tp"])
def test_lingbot_video_fp32_is_tight(mode):
  """In fp32 the sharding bookkeeping must be exact, i.e. no all-reduce or split misalignment."""
  mp.spawn(
    _run_rank,
    args=(2, mode, _TEXT_TOKENS, False, torch.float32, 1e-5),
    nprocs=2,
    join=True,
  )
