import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import cache_dit.distributed.communicate.primitives as primitives
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


def _worker(rank: int, world_size: int, port: int, group_size: int) -> None:
  torch.cuda.set_device(rank)
  dist.init_process_group("nccl",
                          init_method=f"tcp://127.0.0.1:{port}",
                          rank=rank,
                          world_size=world_size)
  tracked_names = (
    "ulysses_fp8_qkv_permute_quant",
    "ulysses_fp8_qkv_permute_dequant",
    "ulysses_fp8_o_permute_quant",
    "ulysses_fp8_o_permute_dequant",
    "ulysses_fp8_per_token_quant",
    "ulysses_fp8_per_token_dequant",
  )
  originals = {name: getattr(primitives, name) for name in tracked_names}
  observed = []
  for name in tracked_names:

    def _wrapped(*args, _name=name, _original=originals[name]):
      observed.append((_name, args[-1]))
      return _original(*args)

    setattr(primitives, name, _wrapped)
  try:
    group = dist.group.WORLD
    torch.manual_seed(rank)

    qkv = torch.randn((1, 3, 8, 72), device="cuda", dtype=torch.bfloat16)
    expected_qkv = _all_to_all_single_qkv_async(qkv, group)()
    actual_qkv = _all_to_all_single_qkv_fp8_async(qkv, group, group_size=group_size)()
    _assert_fp8_close(actual_qkv, expected_qkv)

    o = torch.randn((1, 12, 2, 72), device="cuda", dtype=torch.bfloat16)
    expected_o = _all_to_all_single_o_async(o, group, NUM_QO_HEAD=8)()
    actual_o = _all_to_all_single_o_fp8_async(o, group, NUM_QO_HEAD=8, group_size=group_size)()
    _assert_fp8_close(actual_o, expected_o)

    local_seq = rank + 1
    uneven_qkv = torch.randn((1, local_seq, 8, 72), device="cuda", dtype=torch.bfloat16)
    expected_uaa_qkv = _all_to_all_single_any_qkv_async(uneven_qkv, group)()
    actual_uaa_qkv = _all_to_all_single_any_qkv_fp8_async(uneven_qkv, group,
                                                          group_size=group_size)()
    _assert_fp8_close(actual_uaa_qkv, expected_uaa_qkv)

    global_seq = world_size * (world_size + 1) // 2
    uaa_o = torch.randn((1, global_seq, 2, 72), device="cuda", dtype=torch.bfloat16)
    expected_uaa_o = _all_to_all_single_any_o_async(uaa_o,
                                                    group,
                                                    NUM_QO_HEAD=8,
                                                    Q_S_LOCAL=local_seq)()
    actual_uaa_o = _all_to_all_single_any_o_fp8_async(uaa_o,
                                                      group,
                                                      NUM_QO_HEAD=8,
                                                      Q_S_LOCAL=local_seq,
                                                      group_size=group_size)()
    _assert_fp8_close(actual_uaa_o, expected_uaa_o)

    invalid_o = torch.randn((1, 6, 2, 72), device="cuda", dtype=torch.bfloat16)
    try:
      _all_to_all_single_o_fp8_async(invalid_o, group, NUM_QO_HEAD=8, group_size=group_size)
    except ValueError as exc:
      assert "disable ulysses_float8" in str(exc)
    else:
      raise AssertionError("non-divisible S_GLOBAL must fail before the collective")

    assert {name for name, _ in observed} == set(tracked_names)
    assert len(observed) == 8
    assert all(actual_group_size == group_size for _, actual_group_size in observed)
  finally:
    for name, original in originals.items():
      setattr(primitives, name, original)
    dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4,
                    reason="four CUDA devices are required")
@pytest.mark.parametrize("group_size", [0, 16, 32])
def test_fp8_a2a_auto_padding_matches_bf16_layout(group_size):
  mp.spawn(_worker, args=(4, _free_port(), group_size), nprocs=4, join=True)
