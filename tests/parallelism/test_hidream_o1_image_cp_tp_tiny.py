"""Multi-rank parity test for the HiDream-O1-Image CP and TP plumbing (tiny model, no weights).

HiDream-O1-Image keeps text, the timestep token and the image patches in one joint sequence, so
context parallelism splits `hidden_states` at the first block, splits the packed mRoPE table through
the `rope_split` marker and gathers the joint sequence at `sequence_gather`, while the dense
causal/full attention mask is passed through untouched. A wrong RoPE partition, a split mask or a
gather that runs after the output head still returns tensors of the right shape, so this test
compares against a cp-free/tp-free reference computed on rank 0.

Usage:
  CUDA_VISIBLE_DEVICES=0,1 python -m pytest tests/parallelism/test_hidream_o1_image_cp_tp_tiny.py -q
"""

import os

import pytest
import torch
import torch.multiprocessing as mp

from cache_dit.models.pipelines.hidream_o1_image import HiDreamO1ImageTransformer2DModel
from cache_dit.models.pipelines.hidream_o1_image.pipeline import (
  build_hidream_o1_attention_mask, )

_IMAGE_LEN = 4  # 2x2 patches of a 64x64 image
_TEXT_TOKENS_EVEN = 6  # joint length 10, divisible by 2 for plain Ulysses
_TEXT_TOKENS_ODD = 7  # joint length 11, only `--ulysses-anything` can split it
_TMS_TOKEN_ID = 100
# CP is bit-exact in fp32 (measured 0.0) and stays under 2e-3 in bf16. TP runs the same sharded
# matmuls through several bf16 all-reduces on this deliberately tiny random-init model, which lands
# around 5e-3; the fp32 TP parity is 1.7e-7, so the tolerance only absorbs bf16 rounding.
_REL_TOL_CP = 2e-3
_REL_TOL_TP = 1e-2
_PORT = 29591


def _build(device):
  torch.manual_seed(0)
  transformer = HiDreamO1ImageTransformer2DModel(
    hidden_size=64,
    intermediate_size=128,
    num_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    vocab_size=128,
    patch_size=32,
    in_channels=3,
    tms_token_id=_TMS_TOKEN_ID,
  )
  return transformer.to(device=device, dtype=torch.bfloat16).eval()


def _inputs(text_tokens: int):
  generator = torch.Generator().manual_seed(3)
  seq = text_tokens + _IMAGE_LEN
  input_ids = torch.arange(1, text_tokens + 1).unsqueeze(0)
  input_ids[0, -1] = _TMS_TOKEN_ID

  token_types = torch.zeros((1, seq), dtype=torch.long)
  token_types[0, text_tokens - 1:text_tokens + _IMAGE_LEN] = 1
  token_types[0, text_tokens - 1] = 3

  grid = torch.arange(_IMAGE_LEN).view(2, 2).reshape(-1)
  position_ids = torch.zeros(3, 1, seq, dtype=torch.long)
  position_ids[:, 0, :text_tokens] = torch.arange(text_tokens)
  position_ids[:, 0, text_tokens:] = torch.stack([grid, grid, grid]) + 4096

  return {
    "input_ids": input_ids,
    "position_ids": position_ids,
    "vinputs": torch.randn(1, _IMAGE_LEN, 3 * 32 * 32, generator=generator),
    "timestep": torch.tensor([0.7]),
    "attention_mask": build_hidream_o1_attention_mask(token_types, torch.bfloat16),
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
    transformer = _build(device)

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
      tolerance = _REL_TOL_CP if mode == "cp" else _REL_TOL_TP
      print(f"[hidream-o1-{mode}-uaa{int(uaa)}] rel={relative.item():.6f} shape={tuple(out.shape)}")
      assert out.shape == reference.shape, f"{mode} shape changed: {out.shape}"
      assert relative.item() < tolerance, f"{mode} diverged (rel={relative.item():.6f})"
  finally:
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("mode", ["cp", "tp"])
def test_hidream_o1_image_parallel_matches_single_rank(mode):
  mp.spawn(_run_rank, args=(2, mode, _TEXT_TOKENS_EVEN, False), nprocs=2, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("mode", ["cp", "tp"])
def test_hidream_o1_image_odd_joint_length(mode):
  # An odd joint length needs `--ulysses-anything` for CP; TP never splits the sequence.
  mp.spawn(_run_rank, args=(2, mode, _TEXT_TOKENS_ODD, mode == "cp"), nprocs=2, join=True)
