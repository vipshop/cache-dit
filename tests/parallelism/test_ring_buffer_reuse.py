import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cache_dit.distributed.communicate.primitives import _RingP2PComm
from cache_dit.distributed.parallel.ring import RingAttention
import cache_dit.distributed.parallel.ring as ring_module


class _FakeMesh:

  def __init__(self, group):
    self._group = group

  def get_group(self):
    return self._group


def _free_port() -> int:
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.bind(("127.0.0.1", 0))
    return sock.getsockname()[1]


def _forward_op(ctx, query, key, value, attn_mask, dropout_p, is_causal, scale, enable_gqa,
                return_lse, _save_ctx, _cp_config):
  del (ctx, query, attn_mask, dropout_p, is_causal, scale, enable_gqa, return_lse, _save_ctx,
       _cp_config)
  out = key + value
  lse = torch.zeros((*key.shape[:-1], 1), dtype=key.dtype, device=key.device)
  return out, lse


def _worker(rank: int, world_size: int, port: int) -> None:
  torch.cuda.set_device(rank)
  dist.init_process_group("nccl",
                          init_method=f"tcp://127.0.0.1:{port}",
                          rank=rank,
                          world_size=world_size)
  try:
    ring_module.fused_merge_attn_states = None
    buffer_ids = []
    original_send_recv_kv = _RingP2PComm.send_recv_kv

    def _record_buffers(self, k, v, k_buffer=None, v_buffer=None):
      buffer_ids.append((id(k_buffer), id(v_buffer)))
      return original_send_recv_kv(self, k, v, k_buffer=k_buffer, v_buffer=v_buffer)

    _RingP2PComm.send_recv_kv = _record_buffers
    cp_config = SimpleNamespace(_ring_mesh=_FakeMesh(dist.group.WORLD), convert_to_fp32=False)
    query = torch.zeros((1, 2, 3, 4), device="cuda", dtype=torch.float32)
    key = torch.full_like(query, rank + 1)
    value = torch.full_like(query, 10 * (rank + 1))

    out, _ = RingAttention.apply(query, key, value, None, 0.0, False, None, False, True,
                                 _forward_op, None, cp_config)

    expected = torch.full_like(out, sum((i + 1) * 11 for i in range(world_size)) / world_size)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert len(buffer_ids) == world_size - 1
    assert len(set(buffer_ids)) == 2
    assert buffer_ids == [buffer_ids[0], buffer_ids[1], buffer_ids[0]]
  finally:
    dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 4,
                    reason="four CUDA devices are required")
def test_ring_reuses_two_kv_buffer_pairs_without_changing_output():
  mp.spawn(_worker, args=(4, _free_port()), nprocs=4, join=True)
