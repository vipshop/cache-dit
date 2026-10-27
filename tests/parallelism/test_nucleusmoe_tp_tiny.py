"""Multi-rank parity test for the NucleusMoE-Image tensor-parallel plumbing (tiny model).

NucleusMoE-Image keeps one `transformer_blocks` stack whose attention concatenates the image and
the text K/V, and whose FFN is a SwiGLU `FeedForward` in the dense blocks and a
shared-expert + routed-expert MoE in the rest. The TP plan shards the attention, both SwiGLU
projections and the shared expert, and keeps the routed expert tables replicated; a wrong column
grid (image vs text K/V, or the fused `(up, gate)` interleave) still produces tensors of the right
shape, so only a numeric comparison against a single-rank reference catches it.

Context parallelism is not offered for this model (the expert-choice capacity is derived from the
sequence length), so this test covers TP only.

The fixture runs the MoE with a capacity that covers the whole sequence: with a smaller capacity the
top-C selection has a hard boundary, and ordinary bf16 noise flips selected tokens, which would
measure quantisation noise instead of the sharding.

Usage:
  CUDA_VISIBLE_DEVICES=5,6 python -m pytest tests/parallelism/test_nucleusmoe_tp_tiny.py -q
"""

import os

import pytest
import torch
import torch.multiprocessing as mp

_TEXT_LEN = 8
_PAD_TOKENS = 2
_FRAMES = 1
_LATENT_H = 4
_LATENT_W = 4
_IN_CHANNELS = 4
_DIM = 128
_REL_TOL = 2e-2  # bf16 all-reduce order noise on a 4-layer random model
_PORT = 29581


def _build(device):
  from diffusers import NucleusMoEImageTransformer2DModel

  torch.manual_seed(0)
  transformer = NucleusMoEImageTransformer2DModel(
    patch_size=2,
    in_channels=_IN_CHANNELS,
    out_channels=_IN_CHANNELS,
    num_layers=4,
    attention_head_dim=8,
    num_attention_heads=_DIM // 8,
    num_key_value_heads=2,
    joint_attention_dim=32,
    axes_dims_rope=(2, 2, 4),
    moe_enabled=True,
    dense_moe_strategy="leave_first_block_dense",
    num_experts=4,
    moe_intermediate_dim=32,
    # `capacity = ceil(cf * S / num_experts)`; 4.0 x 16 tokens / 4 experts covers the sequence, so
    # no token sits on the top-C boundary and bf16 noise cannot flip the routing.
    capacity_factors=4.0,
    use_grouped_mm=False,
  )
  # The routed expert tables are `torch.empty` upstream (they always come from a checkpoint).
  with torch.no_grad():
    for block in transformer.transformer_blocks:
      if block.moe_enabled:
        block.img_mlp.experts.gate_up_proj.normal_(0.0, 0.2)
        block.img_mlp.experts.down_proj.normal_(0.0, 0.2)
  return transformer.to(device=device, dtype=torch.bfloat16).eval()


def _inputs():
  generator = torch.Generator().manual_seed(3)
  mask = torch.ones(1, _TEXT_LEN, dtype=torch.bool)
  mask[:, -_PAD_TOKENS:] = False
  return {
    "hidden_states":
    torch.randn(1, _FRAMES * _LATENT_H * _LATENT_W, _IN_CHANNELS,
                generator=generator).to(torch.bfloat16),
    "img_shapes": [(_FRAMES, _LATENT_H, _LATENT_W)],
    "encoder_hidden_states":
    torch.randn(1, _TEXT_LEN, 32, generator=generator).to(torch.bfloat16),
    "encoder_hidden_states_mask":
    mask,
    "timestep":
    torch.tensor([0.5]),
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
        Bn_compute_blocks=0,
        max_warmup_steps=100,
        max_cached_steps=0,
        num_inference_steps=8,
        enable_separate_cfg=False,
        residual_diff_threshold=0.0,
      ),
      parallelism_config=ParallelismConfig(tp_size=world),
    )

    # Guard against a no-op parallelization: a silently unsharded transformer would also "match".
    from torch.distributed._tensor import DTensor

    assert isinstance(transformer.transformer_blocks[0].attn.to_q.weight,
                      DTensor), ("the TP plan did not shard `to_q`")

    with torch.no_grad():
      out = transformer(**_to_device(inputs, device), return_dict=False)[0].float().cpu()

    if rank == 0:
      relative = (out - reference).abs().mean() / reference.abs().mean().clamp_min(1e-6)
      print(f"[nucleus-tp] rel={relative.item():.6f} shape={tuple(out.shape)}")
      assert out.shape == reference.shape, f"tp shape changed: {out.shape}"
      assert relative.item() < _REL_TOL, f"tp diverged (rel={relative.item():.6f})"
  finally:
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_nucleus_moe_tensor_parallel_matches_single_rank():
  mp.spawn(_run_rank, args=(2, ), nprocs=2, join=True)
