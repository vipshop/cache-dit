"""Multi-rank parity test for the PRX / PRXPixel CP and TP plumbing (tiny model, no weights).

PRX keeps `txt` and `pe` as loop-invariant locals, so a CP plan that splits them at
`blocks.0` would only affect the first block. The plan therefore splits the text sequence
after `txt_in`, splits the image ids before `pe_embedder`, and splits the 3D patch tokens at
the first block. The attention processor also rebuilds the joint text+image key mask in the
Ulysses all-to-all order; getting that order wrong still produces a coherent image, so this
test compares against a cp-free reference on rank 0 with a padding mask present.

Usage:
  CUDA_VISIBLE_DEVICES=4,5 python -m pytest tests/parallelism/test_prx_cp_tp_tiny.py -q
"""

import os

import pytest
import torch
import torch.multiprocessing as mp

try:
  from diffusers.models.transformers.transformer_prx import PRXTransformer2DModel  # noqa: F401
except ImportError:
  pytest.skip("diffusers without PRX support", allow_module_level=True)

_TEXT_TOKENS = 16
_TEXT_PAD = 6
_CONTEXT_DIM = 64
_REL_TOL = 2e-2  # bf16 all-reduce order noise on a 2-layer random model
_PORT = 29561


def _build(device):
  torch.manual_seed(0)
  transformer = PRXTransformer2DModel(
    in_channels=3,
    patch_size=16,
    context_in_dim=_CONTEXT_DIM,
    hidden_size=256,
    mlp_ratio=2.0,
    num_heads=2,
    depth=2,
    axes_dim=[64, 64],
    bottleneck_size=96,
    resolution_embeds=True,
  )
  return transformer.to(device=device, dtype=torch.bfloat16).eval()


def _inputs(use_mask: bool = True):
  generator = torch.Generator().manual_seed(3)
  hidden_states = torch.randn(1, 3, 64, 64, generator=generator)
  encoder_hidden_states = torch.randn(1, _TEXT_TOKENS, _CONTEXT_DIM, generator=generator)
  inputs = {
    "hidden_states": hidden_states.to(torch.bfloat16),
    "timestep": torch.tensor([0.5]),
    "encoder_hidden_states": encoder_hidden_states.to(torch.bfloat16),
  }
  if use_mask:
    attention_mask = torch.ones(1, _TEXT_TOKENS, dtype=torch.bool)
    attention_mask[:, _TEXT_TOKENS - _TEXT_PAD:] = False
    inputs["attention_mask"] = attention_mask
  return inputs


def _to_device(inputs, device):
  return {
    key: (value.to(device) if torch.is_tensor(value) else value)
    for key, value in inputs.items()
  }


def _run_rank(rank: int, world: int, mode: str, use_mask: bool = True) -> None:
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
      print(f"[prx-{mode}] rel={relative.item():.6f} shape={tuple(got.shape)}")
      assert got.shape == ref.shape, f"{mode} shape changed: {got.shape} vs {ref.shape}"
      assert relative.item() < _REL_TOL, f"{mode} diverged (rel={relative.item():.6f})"
  finally:
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("use_mask", [True, False])
@pytest.mark.parametrize("mode", ["cp", "tp"])
def test_prx_parallel_matches_single_rank(mode, use_mask):
  mp.spawn(_run_rank, args=(2, mode, use_mask), nprocs=2, join=True)
