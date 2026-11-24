import math
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from cache_dit.attention import set_sol_attn_config
from cache_dit.attention.backends import sol_attn as sol_backend
from cache_dit.attention.backends.register import _dispatch_attention_fn
from cache_dit.distributed.parallel import _ContextParallelConfig


def _free_port() -> int:
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    return sock.getsockname()[1]


def _pool_64(x: torch.Tensor) -> torch.Tensor:
  batch, tokens, heads, dim = x.shape
  blocks = (tokens + 63) // 64
  padded = F.pad(x, (0, 0, 0, 0, 0, blocks * 64 - tokens)).view(batch, blocks, 64, heads, dim)
  lengths = [min(64, tokens - block * 64) for block in range(blocks)]
  return torch.stack([
    padded[:, block, :length].float().mean(dim=1).to(x.dtype)
    for block, length in enumerate(lengths)
  ],
                     dim=1)


def _route_mask(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
  batch, tokens, heads, dim = q.shape
  q_blocks = _pool_64(q)
  k_blocks = _pool_64(k)
  blocks = k_blocks.shape[1]
  k_bh = k_blocks.permute(0, 2, 1, 3)
  q_bh = q_blocks.permute(0, 2, 1, 3)
  mean = k_bh.float().mean(dim=2)
  variance_k = (k_bh.float().square().mean(dim=2) - mean.square()).clamp_min(0.0)
  raw_mean = (q_bh.float() * mean[:, :, None, :]).sum(dim=-1)
  variance = (q_bh.float().square() * variance_k[:, :, None, :]).sum(dim=-1)
  log2_scale = dim ** -0.5 * math.log2(math.e)
  threshold = raw_mean * log2_scale + (variance * log2_scale ** 2 + 1e-6).sqrt()
  scores = torch.einsum("bihd,bjhd->bhij", q_blocks.float(), k_blocks.float()) * log2_scale
  exact = scores > threshold[..., None]
  query_blocks = q_blocks.shape[1]
  local = (torch.arange(query_blocks, device=q.device)[:, None] -
           torch.arange(blocks, device=q.device)[None, :]).abs() <= 1
  return exact | local[None, None, :, :]


def _relative_l2(output: torch.Tensor, reference: torch.Tensor) -> float:
  return ((output.float() - reference.float()).norm() /
          reference.float().norm().clamp_min(1e-12)).item()


def _run_ulysses(rank: int, world_size: int, port: int) -> None:
  os.environ["MASTER_ADDR"] = "127.0.0.1"
  os.environ["MASTER_PORT"] = str(port)
  torch.cuda.set_device(rank)
  dist.init_process_group("nccl", rank=rank, world_size=world_size)
  try:
    set_sol_attn_config(tau=1.0, thresh_type="diag", sink_tokens=0)
    mesh = dist.device_mesh.init_device_mesh(
      "cuda",
      (1, world_size),
      mesh_dim_names=("ring", "ulysses"),
    )
    cp_config = _ContextParallelConfig(ring_degree=1, ulysses_degree=world_size)
    cp_config._ulysses_mesh = mesh["ulysses"]

    torch.manual_seed(20261124)
    q = torch.randn(1, 256, world_size, 128, dtype=torch.bfloat16).cuda(rank)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    local = slice(rank * q.shape[1] // world_size, (rank + 1) * q.shape[1] // world_size)
    masks = []
    original = sol_backend._sol_attn.sol_attn

    def _spy(query, key, value, **kwargs):
      masks.append(_route_mask(query, key))
      return original(query, key, value, **kwargs)

    sol_backend._sol_attn.sol_attn = _spy
    try:
      full_output = _dispatch_attention_fn(q, k, v, backend="sol_attn")
      cp_output = _dispatch_attention_fn(
        q[:, local],
        k[:, local],
        v[:, local],
        backend="sol_attn",
        cp_config=cp_config,
      )
    finally:
      sol_backend._sol_attn.sol_attn = original

    assert len(masks) == 2
    torch.testing.assert_close(masks[1], masks[0][:, rank:rank + 1], rtol=0, atol=0)
    assert _relative_l2(cp_output, full_output[:, local]) <= 3e-3
  finally:
    dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_sol_attn_ulysses_matches_single_card(world_size: int):
  if torch.cuda.device_count() < world_size:
    pytest.skip(f"requires {world_size} CUDA devices")
  mp.spawn(_run_ulysses, args=(world_size, _free_port()), nprocs=world_size, join=True)
