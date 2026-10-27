"""Multi-rank parity test for the MOSS-Image CP and TP plumbing (tiny model, no weights).

MOSS-Image runs one joint `[image, text]` sequence per sample, so context parallelism splits
`hidden_states` at the first block, splits the shared RoPE table through the `rope_split` marker and
gathers the joint sequence at `sequence_gather`. A wrong RoPE partition, or a gather that runs after
the image slice, still returns tensors of the right shape, so this test compares against a cp-free
reference computed on rank 0.

Usage:
  CUDA_VISIBLE_DEVICES=4,5 python -m pytest tests/parallelism/test_moss_image_cp_tp_tiny.py -q
"""

import os

import pytest
import torch
import torch.multiprocessing as mp

from cache_dit.models.pipelines.moss_image import MOSSImageTransformer

_HEIGHT = 4
_WIDTH = 4
_LATENT_CHANNELS = 8
_TEXT_DIM = 16
_TEXT_TOKENS = 16  # H*W + text = 32, divisible by 2 for plain Ulysses
_TEXT_TOKENS_ODD = 15  # odd joint length, only `--ulysses-anything` can split it
_REL_TOL = 2e-3
_PORT = 29571


def _build(device, text_tokens: int = _TEXT_TOKENS):
  torch.manual_seed(0)
  transformer = MOSSImageTransformer(
    dim=32,
    ffn_dim=64,
    num_heads=4,
    num_layers=2,
    text_dim=_TEXT_DIM,
    freq_dim=16,
    latent_channels=_LATENT_CHANNELS,
  )
  return transformer.to(device=device, dtype=torch.bfloat16).eval()


def _inputs(text_tokens: int = _TEXT_TOKENS):
  generator = torch.Generator().manual_seed(3)
  return {
    "hidden_states":
    torch.randn(1, _LATENT_CHANNELS, _HEIGHT, _WIDTH, generator=generator).to(torch.bfloat16),
    "timestep":
    torch.tensor([0.5]),
    "encoder_hidden_states":
    torch.randn(1, text_tokens, _TEXT_DIM, generator=generator).to(torch.bfloat16),
    "encoder_attention_mask":
    torch.ones(1, text_tokens, dtype=torch.long),
  }


def _to_device(inputs, device):
  return {
    key: (value.to(device) if torch.is_tensor(value) else value)
    for key, value in inputs.items()
  }


def _run_rank(rank: int, world: int, mode: str, text_tokens: int, uaa: bool) -> None:
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

    inputs = _inputs(text_tokens)
    transformer = _build(device, text_tokens)

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
      print(f"[moss-{mode}-uaa{int(uaa)}] rel={relative.item():.6f} shape={tuple(out.shape)}")
      assert out.shape == reference.shape, f"{mode} shape changed: {out.shape}"
      assert relative.item() < _REL_TOL, f"{mode} diverged (rel={relative.item():.6f})"
  finally:
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("mode", ["cp", "tp"])
def test_moss_image_parallel_matches_single_rank(mode):
  mp.spawn(_run_rank, args=(2, mode, _TEXT_TOKENS, False), nprocs=2, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("mode", ["cp", "tp"])
def test_moss_image_odd_joint_length(mode):
  # An odd joint length needs `--ulysses-anything` for CP; TP never splits the sequence.
  mp.spawn(_run_rank, args=(2, mode, _TEXT_TOKENS_ODD, mode == "cp"), nprocs=2, join=True)
