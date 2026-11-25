import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cache_dit.distributed.communicate.primitives import _all_to_all_fused_qkv_any_fN_async
from cache_dit.distributed.communicate.primitives import _all_to_all_fused_qkv_fN_async
from cache_dit.distributed.communicate.primitives import _all_to_all_single_any_qkv_async
from cache_dit.distributed.communicate.primitives import _all_to_all_single_any_qkv_fN_async
from cache_dit.distributed.communicate.primitives import _all_to_all_single_qkv_async
from cache_dit.distributed.communicate.primitives import _all_to_all_single_qkv_fN_async


def _free_port() -> int:
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    return sock.getsockname()[1]


def _worker(rank: int, world_size: int, port: int, bits: int, group_size: int, quantize_k: bool,
            uneven: bool) -> None:
  torch.cuda.set_device(rank)
  dist.init_process_group("nccl",
                          init_method=f"tcp://127.0.0.1:{port}",
                          rank=rank,
                          world_size=world_size)
  try:
    group = dist.group.WORLD
    torch.manual_seed(2026 + rank)
    seq_len = rank + 1 if uneven else 4
    q = torch.randn((1, seq_len, 5, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, seq_len, 2, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn((1, seq_len, 3, 128), device="cuda", dtype=torch.bfloat16)
    if uneven:
      q_wait = _all_to_all_single_any_qkv_fN_async(q, group, group_size=group_size, bits=bits)
      v_wait = _all_to_all_single_any_qkv_fN_async(v, group, group_size=group_size, bits=bits)
      if quantize_k:
        k_wait = _all_to_all_single_any_qkv_fN_async(k, group, group_size=group_size, bits=bits)
      else:
        k_wait = _all_to_all_single_any_qkv_async(k, group)
      actual_wait = _all_to_all_fused_qkv_any_fN_async(q,
                                                       k,
                                                       v,
                                                       group,
                                                       group_size=group_size,
                                                       bits=bits,
                                                       quantize_k=quantize_k)
    else:
      q_wait = _all_to_all_single_qkv_fN_async(q, group, group_size=group_size, bits=bits)
      v_wait = _all_to_all_single_qkv_fN_async(v, group, group_size=group_size, bits=bits)
      if quantize_k:
        k_wait = _all_to_all_single_qkv_fN_async(k, group, group_size=group_size, bits=bits)
      else:
        k_wait = _all_to_all_single_qkv_async(k, group)
      actual_wait = _all_to_all_fused_qkv_fN_async(q,
                                                   k,
                                                   v,
                                                   group,
                                                   group_size=group_size,
                                                   bits=bits,
                                                   quantize_k=quantize_k)
    expected = (q_wait(), k_wait(), v_wait())
    actual = actual_wait()
    for actual_tensor, expected_tensor in zip(actual, expected):
      assert actual_tensor.shape == expected_tensor.shape
      torch.testing.assert_close(actual_tensor.view(torch.uint16),
                                 expected_tensor.view(torch.uint16),
                                 rtol=0,
                                 atol=0)
  finally:
    dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2,
                    reason="two CUDA devices are required")
@pytest.mark.parametrize("uneven", [False, True])
@pytest.mark.parametrize("bits,group_size,quantize_k", [
  (4, 16, False),
  (4, 16, True),
  (8, 0, True),
  (8, 32, False),
  (6, 16, False),
  (6, 32, True),
])
def test_fused_qkv_a2a_matches_separate_paths(bits: int, group_size: int, quantize_k: bool,
                                              uneven: bool) -> None:
  if bits in (4, 6) and torch.cuda.get_device_capability(0)[0] < 10:
    pytest.skip("FP4/FP6 communication requires native sm100+ conversion support.")
  mp.spawn(_worker,
           args=(2, _free_port(), bits, group_size, quantize_k, uneven),
           nprocs=2,
           join=True)
