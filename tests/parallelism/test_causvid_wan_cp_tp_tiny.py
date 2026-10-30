"""Multi-rank parity test for the CausVid-Wan CP and TP plumbing (tiny model, no weights).

The causal attention is special in one way that a shape check cannot catch: the KV cache is
addressed by global token coordinates and partitioned **by head**, so a rank must write only its own
head slice while every rank keeps the same slot range. This test therefore replays two chunks (the
second one attends to the committed cache of the first) and compares the chunk outputs *and* the
cache tensors against a cp-free reference computed on rank 0.

Usage:
  CUDA_VISIBLE_DEVICES=4,5 python -m pytest tests/parallelism/test_causvid_wan_cp_tp_tiny.py -q
"""

import os

import pytest
import torch
import torch.multiprocessing as mp

from cache_dit.models.pipelines.causvid.transformer import CausVidTransformer3DModel

_CHUNK_FRAMES = 2
_LATENT_CHANNELS = 16
_HEIGHT = 8
_WIDTH = 8
_TEXT_DIM = 32
_TEXT_TOKENS = 6
# Ulysses only reorders the sequence, so CP must reproduce the single-rank result almost exactly.
# TP shards the projections, which changes the fp32/bf16 accumulation order of the all-reduce.
_REL_TOL_CP = 2e-3
# bf16 all-reduce on a 2-layer model: measured 6.2e-3 on the KV tensors, the fp32 case pins the
# bookkeeping exactly (1e-5), so this only has to separate noise from a real misalignment.
_REL_TOL_TP = 1.2e-2
_PORT = 29593
_LATENT_SHAPE = (1, _LATENT_CHANNELS, _CHUNK_FRAMES, _HEIGHT, _WIDTH)


def _build(device, dtype=torch.bfloat16) -> CausVidTransformer3DModel:
  torch.manual_seed(0)
  transformer = CausVidTransformer3DModel(
    patch_size=(1, 2, 2),
    num_attention_heads=4,
    attention_head_dim=8,
    in_channels=_LATENT_CHANNELS,
    out_channels=_LATENT_CHANNELS,
    text_dim=_TEXT_DIM,
    freq_dim=32,
    ffn_dim=64,
    num_layers=2,
    cross_attn_norm=False,
    qk_norm="rms_norm_across_heads",
    eps=1e-6,
    rope_max_seq_len=64,
  )
  return transformer.to(device=device, dtype=dtype).eval()


def _inputs(dtype=torch.bfloat16, device="cpu"):
  generator = torch.Generator().manual_seed(3)
  chunks = [
    torch.randn(_LATENT_SHAPE, generator=generator).to(device=device, dtype=dtype),
    torch.randn(_LATENT_SHAPE, generator=generator).to(device=device, dtype=dtype),
  ]
  context = torch.randn(1, _TEXT_TOKENS, _TEXT_DIM, generator=generator).to(device=device,
                                                                            dtype=dtype)
  return chunks, context


def _run_sequence(transformer, chunks, context):
  """Two denoising forwards plus the clean commit of chunk 0, i.e. a real cross-chunk KV reuse."""
  frame_seq_len = transformer.frame_seq_len(_LATENT_SHAPE)
  chunk_tokens = _CHUNK_FRAMES * frame_seq_len
  kv_cache = transformer.init_kv_cache(batch_size=1,
                                       latent_shape=_LATENT_SHAPE,
                                       num_frames=_CHUNK_FRAMES * len(chunks))
  outputs = []
  for index, chunk in enumerate(chunks):
    current_start = index * chunk_tokens
    current_end = current_start + chunk_tokens
    outputs.append(
      transformer(
        chunk,
        torch.tensor([1000.0], device=chunk.device),
        context,
        kv_cache=kv_cache,
        current_start=current_start,
        current_end=current_end,
      ))
    transformer(
      outputs[-1],
      torch.tensor([0.0], device=chunk.device),
      context,
      kv_cache=kv_cache,
      current_start=current_start,
      current_end=current_end,
      force_full_compute=True,
    )
  return outputs, kv_cache, chunk_tokens


def _relative(out, reference) -> float:
  return ((out - reference).abs().mean() / reference.abs().mean().clamp_min(1e-6)).item()


def _run_rank(rank: int, world: int, mode: str, dtype, tolerance_override) -> None:
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

    chunks, context = _inputs(dtype, device)
    transformer = _build(device, dtype)

    reference = None
    if rank == 0:
      with torch.no_grad():
        ref_outputs, ref_cache, chunk_tokens = _run_sequence(transformer, chunks, context)
      reference = (
        [out.float().cpu() for out in ref_outputs],
        [cache.k[:, :2 * chunk_tokens].float().cpu() for cache in ref_cache],
        [cache.v[:, :2 * chunk_tokens].float().cpu() for cache in ref_cache],
      )
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
      outputs, kv_cache, chunk_tokens = _run_sequence(transformer, chunks, context)

    if rank == 0:
      tolerance = tolerance_override or (_REL_TOL_CP if mode == "cp" else _REL_TOL_TP)
      ref_outputs, ref_k, ref_v = reference
      for index, (out, ref) in enumerate(zip(outputs, ref_outputs)):
        out = out.float().cpu()
        assert out.shape == ref.shape, f"{mode} chunk {index} shape changed: {out.shape}"
        relative = _relative(out, ref)
        print(f"[causvid-{mode}] chunk{index} rel={relative:.6f}")
        assert relative < tolerance, f"{mode} chunk {index} diverged (rel={relative:.6f})"

      for layer, cache in enumerate(kv_cache):
        # The cache is partitioned by head: rank 0 owns the first `cache.heads` heads, both after the
        # TP sharding and after the CP all-to-all, so the reference is sliced the same way.
        local_heads = cache.heads
        key = cache.k[:, :chunk_tokens].float().cpu()
        value = cache.v[:, :chunk_tokens].float().cpu()
        key_rel = _relative(key, ref_k[layer][:, :chunk_tokens, :local_heads])
        value_rel = _relative(value, ref_v[layer][:, :chunk_tokens, :local_heads])
        print(f"[causvid-{mode}] kv layer{layer} k_rel={key_rel:.6f} v_rel={value_rel:.6f}")
        assert key_rel < tolerance, f"{mode} K diverged at layer {layer} (rel={key_rel:.6f})"
        assert value_rel < tolerance, f"{mode} V diverged at layer {layer} (rel={value_rel:.6f})"
  finally:
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("mode", ["cp", "tp"])
def test_causvid_wan_parallel_matches_single_rank(mode):
  mp.spawn(_run_rank, args=(2, mode, torch.bfloat16, None), nprocs=2, join=True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("mode", ["cp", "tp"])
def test_causvid_wan_parallel_is_tight_in_fp32(mode):
  """In fp32 the sharding bookkeeping must be exact, i.e. no split or head-offset misalignment."""
  mp.spawn(_run_rank, args=(2, mode, torch.float32, 1e-5), nprocs=2, join=True)
