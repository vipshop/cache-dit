"""Multi-rank regression test: Ulysses GQA strategies vs single-rank reference.

Covers the world=4 silent-corruption bug in ``gather_replicated_kv_for_local_q``
(28Q/7KV GQA: local Q heads 28/4 == 7 == global KV heads skipped the per-window
GQA head selection and mismatched KV heads on every rank).

Each rank holds its sequence shard (tensor_split semantics) with full heads and
dispatches attention through cache-dit's Ulysses path (replicate and
group-aligned strategies); the output shard must match a cp-free full-sequence
flash MHA reference.

Usage (from cache-dit/, defaults to the 4-GPU regression case; WORLD=2 for
the historically-correct smaller mesh):
  CUDA_VISIBLE_DEVICES=4,5,6,7 python tests/parallelism/test_ulysses_gqa_multirank.py
  CUDA_VISIBLE_DEVICES=4,5 WORLD=2 python tests/parallelism/test_ulysses_gqa_multirank.py
"""

import os

import torch
import torch.multiprocessing as mp

_ATOL = 3e-2
_H_Q, _H_KV, _D = 28, 7, 120
_SEQ_LENS = [2944, 2945, 6120]  # even, odd (UAA), single-image latent


def _run_rank(rank: int, world: int, port: int) -> None:
  os.environ["MASTER_ADDR"] = "127.0.0.1"
  os.environ["MASTER_PORT"] = str(port)
  torch.cuda.set_device(rank)
  torch.distributed.init_process_group("nccl", rank=rank, world_size=world)
  device = f"cuda:{rank}"

  from cache_dit.attention import (
    _dispatch_attention_fn,
    _maybe_register_custom_attn_backends,
    set_attn_backend,
  )
  from cache_dit.distributed.core import _ContextParallelConfig

  # Register cache-dit's own backends first: without this the flash_varlen
  # name resolves to the diffusers proxy, which drops cp_gqa_strategy.
  _maybe_register_custom_attn_backends()
  set_attn_backend("flash_varlen")

  mesh = torch.distributed.device_mesh.init_device_mesh("cuda",
                                                        mesh_shape=(1, world),
                                                        mesh_dim_names=("ring", "ulysses"))
  cfg = _ContextParallelConfig(ring_degree=1, ulysses_degree=world, ulysses_anything=True)
  cfg.setup(rank, world, torch.device(device), mesh=mesh)

  from flash_attn import flash_attn_varlen_func

  torch.manual_seed(7)
  kw = dict(device=device, dtype=torch.bfloat16)
  for seq_len in _SEQ_LENS:
    q = torch.randn(1, seq_len, _H_Q, _D, **kw) * 0.5
    k = torch.randn(1, seq_len, _H_KV, _D, **kw) * 0.5
    v = torch.randn(1, seq_len, _H_KV, _D, **kw) * 0.5
    cu = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    ref = flash_attn_varlen_func(
      q.reshape(seq_len, _H_Q, _D),
      k.repeat_interleave(_H_Q // _H_KV, dim=2).reshape(seq_len, _H_Q, _D),
      v.repeat_interleave(_H_Q // _H_KV, dim=2).reshape(seq_len, _H_Q, _D),
      cu_seqlens_q=cu,
      cu_seqlens_k=cu,
      max_seqlen_q=seq_len,
      max_seqlen_k=seq_len,
      dropout_p=0.0,
    ).view(1, seq_len, _H_Q, _D)

    chunks = torch.tensor_split(q, world, dim=1)
    start = sum(c.shape[1] for c in chunks[:rank])
    q_l = chunks[rank].contiguous()
    k_l = torch.tensor_split(k, world, dim=1)[rank].contiguous()
    v_l = torch.tensor_split(v, world, dim=1)[rank].contiguous()

    for strategy in ("replicate_kv_sequence", "group_aligned_flash_varlen"):
      with torch.no_grad():
        out = _dispatch_attention_fn(
          q_l,
          k_l,
          v_l,
          attn_mask=None,
          is_causal=False,
          scale=None,
          enable_gqa=True,
          cp_gqa_strategy=strategy,
          backend="flash_varlen",
          cp_config=cfg,
        )
      max_diff = (out.float() - ref[:, start:start + q_l.shape[1]].float()).abs().max().item()
      if max_diff >= _ATOL:
        raise RuntimeError(f"world={world} S={seq_len} rank={rank} {strategy}: "
                           f"max diff {max_diff:.3e} >= {_ATOL}")
  torch.distributed.destroy_process_group()


def main() -> None:
  world = int(os.environ.get("WORLD", "4"))
  if torch.cuda.device_count() < world:
    print(f"SKIP: need {world} GPUs, visible {torch.cuda.device_count()}")
    return
  mp.spawn(_run_rank, args=(world, 29617 + os.getpid() % 1000), nprocs=world)
  print(f"PASS: world={world} GQA Ulysses strategies match single-rank reference")


if __name__ == "__main__":
  main()
