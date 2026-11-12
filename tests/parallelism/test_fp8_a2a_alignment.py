import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cache_dit.distributed.communicate.primitives import (
  _all_to_all_single_any_o_async,
  _all_to_all_single_any_o_fp8_async,
  _all_to_all_single_any_qkv_async,
  _all_to_all_single_any_qkv_fp8_async,
  _all_to_all_single_o_async,
  _all_to_all_single_o_fp8_async,
  _all_to_all_single_qkv_async,
  _all_to_all_single_qkv_fp8_async,
)


def _free_port() -> int:
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    return sock.getsockname()[1]


def _assert_fp8_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
  assert actual.dtype == expected.dtype
  assert actual.shape == expected.shape
  assert actual.is_contiguous()
  torch.testing.assert_close(actual.float(), expected.float(), rtol=0.08, atol=0.08)


def _worker(rank: int, world_size: int, port: int) -> None:
  torch.cuda.set_device(rank)
  dist.init_process_group("nccl",
                          init_method=f"tcp://127.0.0.1:{port}",
                          rank=rank,
                          world_size=world_size)
  try:
    group = dist.group.WORLD
    torch.manual_seed(rank)

    qkv = torch.randn((1, 3, 8, 64), device="cuda", dtype=torch.bfloat16)
    expected_qkv = _all_to_all_single_qkv_async(qkv, group)()
    actual_qkv = _all_to_all_single_qkv_fp8_async(qkv, group)()
    _assert_fp8_close(actual_qkv, expected_qkv)

    o = torch.randn((1, 12, 2, 64), device="cuda", dtype=torch.bfloat16)
    expected_o = _all_to_all_single_o_async(o, group, NUM_QO_HEAD=8)()
    actual_o = _all_to_all_single_o_fp8_async(o, group, NUM_QO_HEAD=8)()
    _assert_fp8_close(actual_o, expected_o)

    local_seq = rank + 1
    uneven_qkv = torch.randn((1, local_seq, 8, 64), device="cuda", dtype=torch.bfloat16)
    expected_uaa_qkv = _all_to_all_single_any_qkv_async(uneven_qkv, group)()
    actual_uaa_qkv = _all_to_all_single_any_qkv_fp8_async(uneven_qkv, group)()
    _assert_fp8_close(actual_uaa_qkv, expected_uaa_qkv)

    global_seq = world_size * (world_size + 1) // 2
    uaa_o = torch.randn((1, global_seq, 2, 64), device="cuda", dtype=torch.bfloat16)
    expected_uaa_o = _all_to_all_single_any_o_async(uaa_o,
                                                    group,
                                                    NUM_QO_HEAD=8,
                                                    Q_S_LOCAL=local_seq)()
    actual_uaa_o = _all_to_all_single_any_o_fp8_async(uaa_o,
                                                      group,
                                                      NUM_QO_HEAD=8,
                                                      Q_S_LOCAL=local_seq)()
    _assert_fp8_close(actual_uaa_o, expected_uaa_o)

    invalid_o = torch.randn((1, 6, 2, 64), device="cuda", dtype=torch.bfloat16)
    try:
      _all_to_all_single_o_fp8_async(invalid_o, group, NUM_QO_HEAD=8)
    except ValueError as exc:
      assert "disable ulysses_float8" in str(exc)
    else:
      raise AssertionError("non-divisible S_GLOBAL must fail before the collective")
  finally:
    dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4,
                    reason="four CUDA devices are required")
def test_fp8_a2a_auto_padding_matches_bf16_layout():
  mp.spawn(_worker, args=(4, _free_port()), nprocs=4, join=True)
