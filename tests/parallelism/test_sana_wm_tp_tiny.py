"""Multi-rank parity test for the SANA-WM tensor-parallel plumbing (tiny model, no weights).

The SANA-WM plan shards the attention weights and keeps every activation full width (the fused
`qkv` reshape, the global Q/K RMSNorm and the depthwise temporal convolution all need full
channels). A plan that silently shards an activation still returns tensors of the right shape, so
this test compares against a single-rank reference computed on rank 0, on a model that exercises
both the GDN block (layer 0) and the softmax block (layer 1), with camera conditioning enabled.

Context parallelism is not covered: the planner refuses it by design (see
`tests/api/test_sana_wm_adapter.py`).

Usage:
  CUDA_VISIBLE_DEVICES=2,3 python -m pytest tests/parallelism/test_sana_wm_tp_tiny.py -q
"""

import os

import pytest
import torch
import torch.multiprocessing as mp

_HIDDEN = 32
_HEADS = 2
_HEAD_DIM = 16
_FRAMES = 2
_LATENT_H = 4
_LATENT_W = 4
_IN_CHANNELS = 4
_TEXT_LEN = 5
_CAPTION_CHANNELS = 8
# bf16 all-reduce order noise on a 2-layer random model.
_REL_TOL = 2e-2
_PORT = 29597


def _build(device):
  from diffusers import SanaWMTransformer3DModel

  torch.manual_seed(0)
  transformer = SanaWMTransformer3DModel(
    in_channels=_IN_CHANNELS,
    num_layers=2,
    hidden_size=_HIDDEN,
    num_attention_heads=_HEADS,
    patch_size=(1, 1, 1),
    softmax_every_n=2,
    linear_head_dim=_HEAD_DIM,
    caption_channels=_CAPTION_CHANNELS,
  )
  return transformer.to(device=device, dtype=torch.bfloat16).eval()


def _inputs():
  generator = torch.Generator().manual_seed(3)
  return {
    "hidden_states":
    torch.randn(1, _IN_CHANNELS, _FRAMES, _LATENT_H, _LATENT_W,
                generator=generator).to(torch.bfloat16),
    "timestep":
    torch.zeros(1, 1, _FRAMES),
    "encoder_hidden_states":
    torch.randn(1, 1, _TEXT_LEN, _CAPTION_CHANNELS, generator=generator).to(torch.bfloat16),
    "mask":
    torch.ones(1, _TEXT_LEN, dtype=torch.int16),
    "data_info": {
      "img_hw": torch.tensor([[64.0, 64.0]])
    },
    # Exercise the UCPE camera branch: its projections are part of the TP plan.
    "camera_conditions":
    torch.zeros(1, _FRAMES, 20, dtype=torch.bfloat16),
  }


def _to_device(inputs, device):
  return {
    key: (value.to(device) if torch.is_tensor(value) else value)
    for key, value in inputs.items()
  }


def _run_rank(rank: int, world: int) -> None:
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
        reference = transformer(**_to_device(inputs, device), return_dict=False)[0].float().cpu()
    torch.distributed.barrier()

    cache_dit.enable_cache(
      BlockAdapterRegister.get_adapter(transformer),
      cache_config=DBCacheConfig(
        Fn_compute_blocks=1,
        Bn_compute_blocks=1,
        max_warmup_steps=100,
        max_cached_steps=0,
        num_inference_steps=8,
        enable_separate_cfg=False,
        residual_diff_threshold=0.0,
      ),
      parallelism_config=ParallelismConfig(tp_size=world),
    )

    # Guard against a no-op parallelization: an unsharded transformer would also "match".
    assert getattr(transformer, "_is_parallelized", False), "tp left the transformer unsharded"

    with torch.no_grad():
      out = transformer(**_to_device(inputs, device), return_dict=False)[0].float().cpu()

    if rank == 0:
      relative = (out - reference).abs().mean() / reference.abs().mean().clamp_min(1e-6)
      print(f"[sana-wm-tp] rel={relative.item():.6f} shape={tuple(out.shape)}")
      assert out.shape == reference.shape, f"tp changed the output shape: {out.shape}"
      assert relative.item() < _REL_TOL, f"tp diverged (rel={relative.item():.6f})"
  finally:
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_sana_wm_tp_matches_single_rank():
  mp.spawn(_run_rank, args=(2, ), nprocs=2, join=True)
