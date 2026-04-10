from types import SimpleNamespace

import pytest
import torch

import cache_dit.distributed.core._distributed_primitives as primitives


class _FakeMesh:

  def __init__(self, group):
    self._group = group

  def get_group(self):
    return self._group


def test_all2all_comm_requires_ulysses_mesh():
  with pytest.raises(ValueError, match="initialized Ulysses mesh"):
    primitives._All2AllComm(SimpleNamespace(_ulysses_mesh=None))


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


def test_all2all_comm_requires_metadata_before_send(monkeypatch):
  group = object()
  cp_config = SimpleNamespace(_ulysses_mesh=_FakeMesh(group))

  monkeypatch.setattr(primitives, "_select_all_to_all_qkv_async_impl",
                      lambda *_args, **_kwargs: None)
  monkeypatch.setattr(primitives, "_select_all_to_all_o_async_impl", lambda *_args, **_kwargs: None)

  comm = primitives._All2AllComm(cp_config)
  x = torch.randn(2, 3, 4, 5)

  with pytest.raises(ValueError, match=r"init_meta\(query\)"):
    comm.send_q(x)
