import socket

import pytest
import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as fc
import torch.multiprocessing as mp

from cache_dit.distributed.communicate.primitives import (
  _all_to_all_single_any_o_async,
  _all_to_all_single_any_qkv_async,
  _all_to_all_single_o_async,
  _all_to_all_single_qkv_async,
)


def _free_port() -> int:
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    return sock.getsockname()[1]


def _legacy_qkv(x: torch.Tensor, group) -> torch.Tensor:
  batch, local_seq, heads, head_dim = x.shape
  world = dist.get_world_size(group)
  local_heads = heads // world
  x = x.reshape(batch, local_seq, world, local_heads, head_dim).permute(2, 1, 0, 3, 4).contiguous()
  x = fc.all_to_all_single(x.flatten(), None, None, group).wait()
  x = x.reshape(world, local_seq, batch, local_heads, head_dim)
  return x.flatten(0, 1).permute(1, 0, 2, 3).contiguous()


def _legacy_o(x: torch.Tensor, group) -> torch.Tensor:
  batch, global_seq, local_heads, head_dim = x.shape
  world = dist.get_world_size(group)
  local_seq = global_seq // world
  x = x.reshape(batch, world, local_seq, local_heads, head_dim).permute(1, 3, 0, 2, 4).contiguous()
  x = fc.all_to_all_single(x.flatten(), None, None, group).wait()
  x = x.reshape(world, local_heads, batch, local_seq, head_dim)
  return x.flatten(0, 1).permute(1, 2, 0, 3).contiguous()


def _legacy_any_qkv(x: torch.Tensor, group) -> torch.Tensor:
  batch, local_seq, heads, head_dim = x.shape
  world = dist.get_world_size(group)
  local_heads = heads // world
  seq_sizes = list(range(1, world + 1))
  x = x.reshape(batch, local_seq, world, local_heads, head_dim).permute(2, 1, 0, 3, 4).contiguous()
  x = fc.all_to_all_single(x.flatten(0, 1), seq_sizes, [local_seq] * world, group).wait()
  return x.permute(1, 0, 2, 3).contiguous()


def _legacy_any_o(x: torch.Tensor, group, local_seq: int) -> torch.Tensor:
  batch, global_seq, local_heads, head_dim = x.shape
  world = dist.get_world_size(group)
  seq_sizes = list(range(1, world + 1))
  x = x.permute(1, 0, 2, 3).contiguous()
  x = fc.all_to_all_single(x, [local_seq] * world, seq_sizes, group).wait()
  x = x.reshape(world, local_seq, batch, local_heads, head_dim)
  x = x.permute(2, 1, 0, 3, 4).contiguous()
  return x.reshape(batch, local_seq, world * local_heads, head_dim)


def _worker(rank: int, world_size: int, port: int) -> None:
  torch.cuda.set_device(rank)
  dist.init_process_group("nccl",
                          init_method=f"tcp://127.0.0.1:{port}",
                          rank=rank,
                          world_size=world_size)
  try:
    group = dist.group.WORLD
    torch.manual_seed(rank)
    head_dim = 66

    qkv = torch.randn((1, 3, 8, head_dim), device="cuda", dtype=torch.bfloat16)
    expected_qkv = _legacy_qkv(qkv, group)
    actual_qkv = _all_to_all_single_qkv_async(qkv, group)()
    assert torch.equal(actual_qkv, expected_qkv)

    o = torch.randn((1, 12, 2, head_dim), device="cuda", dtype=torch.bfloat16)
    expected_o = _legacy_o(o, group)
    actual_o = _all_to_all_single_o_async(o, group, NUM_QO_HEAD=8)()
    assert torch.equal(actual_o, expected_o)

    local_seq = rank + 1
    uneven_qkv = torch.randn((1, local_seq, 8, head_dim), device="cuda", dtype=torch.bfloat16)
    expected_uaa_qkv = _legacy_any_qkv(uneven_qkv, group)
    actual_uaa_qkv = _all_to_all_single_any_qkv_async(uneven_qkv, group)()
    assert torch.equal(actual_uaa_qkv, expected_uaa_qkv)

    global_seq = world_size * (world_size + 1) // 2
    uaa_o = torch.randn((1, global_seq, 2, head_dim), device="cuda", dtype=torch.bfloat16)
    expected_uaa_o = _legacy_any_o(uaa_o, group, local_seq)
    actual_uaa_o = _all_to_all_single_any_o_async(uaa_o, group, NUM_QO_HEAD=8,
                                                  Q_S_LOCAL=local_seq)()
    assert torch.equal(actual_uaa_o, expected_uaa_o)
  finally:
    dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4,
                    reason="four CUDA devices are required")
def test_bf16_a2a_alignment_pad_matches_legacy_layout_bitwise():
  mp.spawn(_worker, args=(4, _free_port()), nprocs=4, join=True)
