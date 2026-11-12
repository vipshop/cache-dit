from types import SimpleNamespace

import pytest
import torch

import cache_dit.distributed.communicate.primitives as primitives
from cache_dit.distributed.parallel.config import _ContextParallelConfig


class _FakeMesh:

  def __init__(self, group):
    self._group = group

  def get_group(self):
    return self._group


def test_all2all_comm_requires_ulysses_mesh():
  with pytest.raises(ValueError, match="initialized Ulysses mesh"):
    primitives._All2AllComm(SimpleNamespace(_ulysses_mesh=None))


def test_ring_p2p_comm_requires_ring_mesh():
  with pytest.raises(ValueError, match="initialized ring mesh"):
    primitives._RingP2PComm(SimpleNamespace(_ring_mesh=None))


def test_ring_p2p_comm_derives_group_and_peer_ranks_from_cp_config(monkeypatch):
  group = object()
  cp_config = SimpleNamespace(_ring_mesh=_FakeMesh(group))

  monkeypatch.setattr(primitives.dist, "get_rank", lambda process_group: 1)
  monkeypatch.setattr(primitives.dist, "get_world_size", lambda process_group: 4)
  monkeypatch.setattr(
    primitives.dist,
    "get_global_rank",
    lambda process_group, local_rank: local_rank + 8,
  )

  comm = primitives._RingP2PComm(cp_config)

  assert comm._process_group is group
  assert comm.rank == 1
  assert comm.world_size == 4
  assert comm.send_rank == 10
  assert comm.recv_rank == 8


def test_all2all_comm_selects_impls_once_and_waits(monkeypatch):
  group = object()
  cp_config = SimpleNamespace(_ulysses_mesh=_FakeMesh(group))
  select_qkv_calls = []
  select_o_calls = []
  launch_calls = []

  def _make_impl(kind):

    def _impl(x, launch_group, **metadata):
      launch_calls.append({
        "kind": kind,
        "shape": tuple(x.shape),
        "group": launch_group,
        "metadata": dict(metadata),
      })
      return lambda: kind

    return _impl

  def _select_qkv(_cp_config, fp8=None):
    select_qkv_calls.append((_cp_config, fp8))
    if fp8 is False:
      return _make_impl("k_impl")
    return _make_impl("qv_impl")

  def _select_o(_cp_config, fp8=None):
    select_o_calls.append((_cp_config, fp8))
    if fp8 is False:
      return _make_impl("lse_impl")
    return _make_impl("o_impl")

  monkeypatch.setattr(primitives, "_select_all_to_all_qkv_async_impl", _select_qkv)
  monkeypatch.setattr(primitives, "_select_all_to_all_o_async_impl", _select_o)

  comm = primitives._All2AllComm(cp_config)
  x = torch.randn(2, 3, 4, 5)
  metadata = {"NUM_QO_HEAD": 4, "Q_S_LOCAL": 3}
  returned = comm.init_meta(x)

  assert comm.group is group
  assert returned is comm
  assert select_qkv_calls == [(cp_config, None), (cp_config, False)]
  assert select_o_calls == [(cp_config, None), (cp_config, False)]

  assert comm.send_q(x).wait() == "qv_impl"
  assert comm.send_v(x)() == "qv_impl"
  assert comm.send_k(x).wait() == "k_impl"
  assert comm.send_o(x).wait() == "o_impl"
  assert comm.send_lse(x).wait() == "lse_impl"

  assert launch_calls == [
    {
      "kind": "qv_impl",
      "shape": (2, 3, 4, 5),
      "group": group,
      "metadata": metadata,
    },
    {
      "kind": "qv_impl",
      "shape": (2, 3, 4, 5),
      "group": group,
      "metadata": metadata,
    },
    {
      "kind": "k_impl",
      "shape": (2, 3, 4, 5),
      "group": group,
      "metadata": metadata,
    },
    {
      "kind": "o_impl",
      "shape": (2, 3, 4, 5),
      "group": group,
      "metadata": metadata,
    },
    {
      "kind": "lse_impl",
      "shape": (2, 3, 4, 5),
      "group": group,
      "metadata": metadata,
    },
  ]


def test_ulysses_float8_k_requires_float8():
  with pytest.raises(ValueError, match="requires ulysses_float8=True"):
    _ContextParallelConfig(ulysses_degree=2, ulysses_float8_k=True)

  config = _ContextParallelConfig(ulysses_degree=2, ulysses_float8=True, ulysses_float8_k=True)
  assert config.ulysses_float8_k


def test_all2all_comm_uses_fp8_for_k_only_when_opted_in(monkeypatch):
  cp_config = SimpleNamespace(_ulysses_mesh=_FakeMesh(object()), ulysses_float8_k=True)
  qkv_impl = object()
  monkeypatch.setattr(primitives, "_select_all_to_all_qkv_async_impl",
                      lambda *args, **kwargs: qkv_impl)
  monkeypatch.setattr(primitives, "_select_all_to_all_o_async_impl",
                      lambda *args, **kwargs: object())

  comm = primitives._All2AllComm(cp_config)

  assert comm._k_impl is qkv_impl
  assert comm._q_impl is qkv_impl
  assert comm._v_impl is qkv_impl


@pytest.mark.parametrize("rows,head_size,expected", [
  (12, 64, 68),
  (24576, 128, 130),
  (81900, 128, 132),
])
def test_fp8_o_packed_head_size_keeps_rank_chunks_aligned(rows, head_size, expected):
  packed = primitives._fp8_o_packed_head_size(rows, head_size)

  assert packed == expected
  assert packed >= head_size + 2
  assert (rows * packed) % 16 == 0


@pytest.mark.parametrize("head_size,expected", [
  (66, 72),
  (128, 128),
  (130, 136),
])
def test_bf16_padded_last_dim_aligns_every_row(head_size, expected):
  packed = primitives._padded_last_dim(head_size, element_size=2)

  assert packed == expected
  assert (packed * 2) % 16 == 0


def test_fp8_packed_row_width_covers_uneven_splits():
  packed = primitives._fp8_packed_row_width([1, 2, 3, 4], row_elements=2, min_width=68)

  assert packed == 72
  assert all((size * 2 * packed) % 16 == 0 for size in [1, 2, 3, 4])


def test_all2all_comm_keeps_k_and_lse_off_fp8_by_default(monkeypatch):
  cp_config = SimpleNamespace(_ulysses_mesh=_FakeMesh(object()),
                              ulysses_float8=True,
                              ulysses_float8_k=False)
  impls = iter([object(), object(), object(), object(), object()])
  monkeypatch.setattr(primitives, "_select_all_to_all_qkv_async_impl",
                      lambda *args, **kwargs: next(impls))
  monkeypatch.setattr(primitives, "_select_all_to_all_o_async_impl",
                      lambda *args, **kwargs: next(impls))

  comm = primitives._All2AllComm(cp_config)

  assert comm._k_impl is not comm._q_impl
  assert comm._lse_impl is not comm._q_impl


def test_all2all_comm_implicitly_initializes_metadata_from_first_qkv_send(monkeypatch):
  group = object()
  cp_config = SimpleNamespace(_ulysses_mesh=_FakeMesh(group))
  launch_calls = []

  def _make_impl(kind):

    def _impl(x, launch_group, **metadata):
      launch_calls.append({
        "kind": kind,
        "shape": tuple(x.shape),
        "group": launch_group,
        "metadata": dict(metadata),
      })
      return lambda: kind

    return _impl

  monkeypatch.setattr(
    primitives,
    "_select_all_to_all_qkv_async_impl",
    lambda *_args, fp8=None, **_kwargs: _make_impl("k_impl" if fp8 is False else "qv_impl"))
  monkeypatch.setattr(primitives,
                      "_select_all_to_all_o_async_impl",
                      lambda *_args, fp8=None, **_kwargs: _make_impl("lse_impl"
                                                                     if fp8 is False else "o_impl"))

  comm = primitives._All2AllComm(cp_config)
  x = torch.randn(2, 3, 4, 5)

  assert comm.send_q(x).wait() == "qv_impl"
  assert comm.send_o(x).wait() == "o_impl"

  assert launch_calls == [
    {
      "kind": "qv_impl",
      "shape": (2, 3, 4, 5),
      "group": group,
      "metadata": {
        "NUM_QO_HEAD": 4,
        "Q_S_LOCAL": 3,
      },
    },
    {
      "kind": "o_impl",
      "shape": (2, 3, 4, 5),
      "group": group,
      "metadata": {
        "NUM_QO_HEAD": 4,
        "Q_S_LOCAL": 3,
      },
    },
  ]


def test_all2all_comm_requires_metadata_before_output_send(monkeypatch):
  group = object()
  cp_config = SimpleNamespace(_ulysses_mesh=_FakeMesh(group))

  monkeypatch.setattr(primitives, "_select_all_to_all_qkv_async_impl",
                      lambda *_args, **_kwargs: None)
  monkeypatch.setattr(primitives, "_select_all_to_all_o_async_impl", lambda *_args, **_kwargs: None)

  comm = primitives._All2AllComm(cp_config)
  x = torch.randn(2, 3, 4, 5)

  with pytest.raises(ValueError, match=r"send_q\(\)/send_k\(\)/send_v\(\)"):
    comm.send_o(x)
