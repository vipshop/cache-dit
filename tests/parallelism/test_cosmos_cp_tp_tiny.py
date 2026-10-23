"""Multi-rank parity test for the Cosmos Predict2 (Anima) CP and TP plumbing (tiny model).

`CosmosTransformer3DModel` receives 5D latents and only builds the token sequence inside
`patch_embed`, so the CP plan splits `hidden_states` + `encoder_hidden_states` at
`transformer_blocks.0`, output-splits the `rope` tables, and gathers on `proj_out`. The
`CosmosAttnProcessor2_0` patch must forward the CP config into the attention dispatch or
every rank attends to its local chunk only — a coherent-but-wrong output that this test
catches against a cp-free reference on rank 0.

Usage:
  CUDA_VISIBLE_DEVICES=4,5 python -m pytest tests/parallelism/test_cosmos_cp_tp_tiny.py -q
"""

import os

import pytest
import torch
import torch.multiprocessing as mp

try:
  from diffusers.models.transformers.transformer_cosmos import (  # noqa: F401
    CosmosTransformer3DModel, )
except ImportError:
  pytest.skip("diffusers without Cosmos support", allow_module_level=True)

_TEXT_TOKENS = 8
_REL_TOL = 2e-2  # bf16 all-reduce order noise on a 2-layer random model
_PORT = 29573


def _build(device):
  torch.manual_seed(0)
  transformer = CosmosTransformer3DModel(
    in_channels=16,
    out_channels=16,
    num_attention_heads=2,
    attention_head_dim=32,
    num_layers=2,
    mlp_ratio=2.0,
    text_embed_dim=64,
    adaln_lora_dim=32,
    max_size=(8, 32, 32),
    rope_scale=(1.0, 1.0, 1.0),
    extra_pos_embed_type=None,  # the Anima checkpoint ships `null`
  )
  return transformer.to(device=device, dtype=torch.bfloat16).eval()


def _inputs():
  generator = torch.Generator().manual_seed(3)
  return {
    "hidden_states": torch.randn(1, 16, 1, 32, 32, generator=generator).to(torch.bfloat16),
    "timestep": torch.tensor([0.5]),
    "encoder_hidden_states": torch.randn(1, _TEXT_TOKENS, 64,
                                         generator=generator).to(torch.bfloat16),
    # Anima t2i passes an all-zero pixel-space padding mask, concatenated as a channel;
    # keep it in the model dtype or `torch.cat` promotes the latents to fp32.
    "padding_mask": torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16),
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
        reference = [
          t.float().cpu() for t in transformer(**_to_device(inputs, device), return_dict=False)
        ]
    torch.distributed.barrier()

    if mode == "cp":
      parallelism_config = ParallelismConfig(ulysses_size=world)
    else:
      parallelism_config = ParallelismConfig(tp_size=world)

    # Cache is disabled (warmup covers the whole run): this test isolates the parallel paths.
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
      out = [t.float().cpu() for t in transformer(**_to_device(inputs, device), return_dict=False)]

    if rank == 0:
      got, ref = out[0], reference[0]
      relative = (got - ref).abs().mean() / ref.abs().mean().clamp_min(1e-6)
      print(f"[cosmos-{mode}] rel={relative.item():.6f} shape={tuple(got.shape)}")
      assert got.shape == ref.shape, f"{mode} shape changed: {got.shape} vs {ref.shape}"
      assert relative.item() < _REL_TOL, f"{mode} diverged (rel={relative.item():.6f})"
  finally:
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("mode", ["cp", "tp"])
def test_cosmos_parallel_matches_single_rank(mode):
  mp.spawn(_run_rank, args=(2, mode), nprocs=2, join=True)
