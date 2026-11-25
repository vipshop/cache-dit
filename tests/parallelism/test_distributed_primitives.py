from types import SimpleNamespace

import pytest
import torch

import cache_dit.distributed.communicate.primitives as primitives
from cache_dit.distributed.config import ParallelismConfig
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


@pytest.mark.parametrize("bits,quant_type,group_size", [
  (8, "per_token", 0),
  (8, "per_group_16", 16),
  (8, "per_group_32", 32),
  (6, "per_token", 0),
  (6, "per_group_16", 16),
  (6, "per_group_32", 32),
])
def test_all2all_comm_selects_impls_once_and_waits(monkeypatch, bits, quant_type, group_size):
  group = object()
  cp_config = SimpleNamespace(
    _ulysses_mesh=_FakeMesh(group),
    quant_communicate_kwargs={
      "bits": bits,
      "quant_type": quant_type
    },
  )
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

  def _select_qkv(_cp_config, bits=None):
    select_qkv_calls.append((_cp_config, bits))
    if bits == 0:
      return _make_impl("k_impl")
    return _make_impl("qv_impl")

  def _select_o(_cp_config, bits=None):
    select_o_calls.append((_cp_config, bits))
    if bits == 0:
      return _make_impl("lse_impl")
    return _make_impl("o_impl")

  monkeypatch.setattr(primitives, "_select_all_to_all_qkv_async_impl", _select_qkv)
  monkeypatch.setattr(primitives, "_select_all_to_all_o_async_impl", _select_o)

  comm = primitives._All2AllComm(cp_config)
  x = torch.randn(2, 3, 4, 5)
  metadata = {"NUM_QO_HEAD": 4, "Q_S_LOCAL": 3, "group_size": group_size, "bits": bits}
  returned = comm.init_meta(x)

  assert comm.group is group
  assert returned is comm
  assert select_qkv_calls == [(cp_config, None)]
  assert select_o_calls == [(cp_config, None), (cp_config, 0)]

  assert comm.send_q(x).wait() == "qv_impl"
  assert comm.send_v(x)() == "qv_impl"
  assert comm.send_k(x).wait() == "qv_impl"
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
      "kind": "qv_impl",
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


@pytest.mark.parametrize("selector,bf16_impl,fN_impl,any_bf16_impl,any_fN_impl", [
  (primitives._select_all_to_all_qkv_async_impl, primitives._all_to_all_single_qkv_async,
   primitives._all_to_all_single_qkv_fN_async, primitives._all_to_all_single_any_qkv_async,
   primitives._all_to_all_single_any_qkv_fN_async),
  (primitives._select_all_to_all_o_async_impl, primitives._all_to_all_single_o_async,
   primitives._all_to_all_single_o_fN_async, primitives._all_to_all_single_any_o_async,
   primitives._all_to_all_single_any_o_fN_async),
])
@pytest.mark.parametrize("config_bits,override_bits,expect_fN", [
  (None, None, False),
  (6, None, True),
  (8, None, True),
  (6, 0, False),
  (None, 6, True),
  (None, 8, True),
])
def test_all_to_all_selectors_use_bits(selector, bf16_impl, fN_impl, any_bf16_impl, any_fN_impl,
                                       config_bits, override_bits, expect_fN):
  cp_config = SimpleNamespace(
    ulysses_anything=False,
    quant_communicate_kwargs={} if config_bits is None else {"bits": config_bits},
  )
  selected = selector(cp_config, bits=override_bits)
  assert selected is (fN_impl if expect_fN else bf16_impl)

  cp_config.ulysses_anything = True
  selected = selector(cp_config, bits=override_bits)
  assert selected is (any_fN_impl if expect_fN else any_bf16_impl)

  with pytest.raises(ValueError, match="bits must be"):
    selector(cp_config, bits=7)


def test_unused_all_to_all_async_fn_aliases_removed():
  import cache_dit.attention as attention
  import cache_dit.distributed.communicate as communicate

  for name in ("_all_to_all_qkv_async_fn", "_all_to_all_o_async_fn"):
    assert not hasattr(primitives, name)
    assert not hasattr(communicate, name)
    assert not hasattr(attention, name)


def test_quantize_k_communication_options():
  with pytest.raises(ValueError, match="ulysses_degree must be greater than 1"):
    _ContextParallelConfig(ulysses_degree=1,
                           quant_communicate_kwargs={
                             "bits": 8,
                             "quantize_k": True
                           })

  config = _ContextParallelConfig(ulysses_degree=2,
                                  quant_communicate_kwargs={
                                    "bits": 8,
                                    "quantize_k": True
                                  })
  assert config.quant_communicate_kwargs["quantize_k"]

  default_config = _ContextParallelConfig(ulysses_degree=2, quant_communicate_kwargs={"bits": 8})
  assert default_config.quant_communicate_kwargs["quantize_k"] is True

  disabled_config = _ContextParallelConfig(ulysses_degree=2,
                                           quant_communicate_kwargs={
                                             "bits": 8,
                                             "quantize_k": False
                                           })
  assert disabled_config.quant_communicate_kwargs["quantize_k"] is False


def test_quant_communicate_validation_and_suffix():
  with pytest.raises(ValueError, match="bits must be 6 or 8"):
    _ContextParallelConfig(ulysses_degree=2, quant_communicate_kwargs={"bits": 7})
  with pytest.raises(ValueError, match="Unsupported quant_communicate_kwargs quant_type"):
    _ContextParallelConfig(ulysses_degree=2,
                           quant_communicate_kwargs={
                             "bits": 8,
                             "quant_type": "per_group_8"
                           })
  with pytest.raises(ValueError, match="Unknown quant_communicate_kwargs keys"):
    ParallelismConfig(ulysses_size=2, quant_communicate_kwargs={"unknown": 1})
  with pytest.raises(ValueError, match="requires Ulysses parallelism"):
    ParallelismConfig(quant_communicate_kwargs={"bits": 8})

  default = ParallelismConfig(ulysses_size=2, quant_communicate_kwargs={"bits": 8})
  default_without_k = ParallelismConfig(ulysses_size=2,
                                        quant_communicate_kwargs={
                                          "bits": 8,
                                          "quantize_k": False
                                        })
  group16 = ParallelismConfig(ulysses_size=2,
                              quant_communicate_kwargs={
                                "bits": 8,
                                "quant_type": "per_group_16"
                              })
  group32 = ParallelismConfig(ulysses_size=2,
                              quant_communicate_kwargs={
                                "bits": 8,
                                "quant_type": "per_group_32"
                              })
  fp6_default = ParallelismConfig(ulysses_size=2, quant_communicate_kwargs={"bits": 6})
  fp6_token_k = ParallelismConfig(ulysses_size=2,
                                  quant_communicate_kwargs={
                                    "bits": 6,
                                    "quant_type": "per_token",
                                    "quantize_k": True
                                  })
  assert default.strify() == "Ulysses2_UF8K"
  assert default_without_k.strify() == "Ulysses2_UF8"
  assert group16.strify() == "Ulysses2_UF8K_G16"
  assert group32.strify() == "Ulysses2_UF8K_G32"
  assert fp6_default.strify() == "Ulysses2_UF6K_G32"
  assert fp6_token_k.strify() == "Ulysses2_UF6K_PT"


@pytest.mark.parametrize("comm_kwargs", [
  {
    "bits": 8
  },
  {
    "bits": 8,
    "quantize_k": True
  },
])
def test_all2all_comm_uses_fp8_for_k_when_enabled(monkeypatch, comm_kwargs):
  cp_config = SimpleNamespace(_ulysses_mesh=_FakeMesh(object()),
                              quant_communicate_kwargs=comm_kwargs)
  qkv_impl = object()
  monkeypatch.setattr(primitives, "_select_all_to_all_qkv_async_impl",
                      lambda *args, **kwargs: qkv_impl)
  monkeypatch.setattr(primitives, "_select_all_to_all_o_async_impl",
                      lambda *args, **kwargs: object())

  comm = primitives._All2AllComm(cp_config)

  assert comm._k_impl is qkv_impl
  assert comm._q_impl is qkv_impl
  assert comm._v_impl is qkv_impl


@pytest.mark.parametrize("rows,head_size,group_size,bits,expected", [
  (12, 64, 0, 8, 68),
  (24576, 128, 0, 8, 130),
  (81900, 128, 0, 8, 132),
  (12, 64, 16, 8, 72),
  (12, 64, 32, 8, 68),
  (24576, 128, 16, 8, 144),
  (24576, 128, 32, 8, 136),
  (3, 72, 16, 8, 96),
  (3, 72, 32, 8, 80),
  (12, 64, 0, 6, 52),
  (12, 64, 16, 6, 56),
  (12, 64, 32, 6, 52),
  (24576, 128, 0, 6, 100),
  (24576, 128, 16, 6, 112),
  (24576, 128, 32, 6, 104),
  (3, 72, 0, 6, 64),
  (3, 72, 16, 6, 80),
  (3, 72, 32, 6, 64),
])
def test_fN_o_packed_head_size_keeps_rank_chunks_aligned(rows, head_size, group_size, bits,
                                                         expected):
  packed = primitives._fN_o_packed_head_size(rows, head_size, group_size, bits)

  assert packed == expected
  assert packed >= primitives._fN_row_min_width(head_size, group_size, 2, bits)
  assert (rows * packed) % 16 == 0
  if bits == 6:
    assert packed % 4 == 0


@pytest.mark.parametrize("head_size,expected", [
  (66, 72),
  (128, 128),
  (130, 136),
])
def test_bf16_padded_last_dim_aligns_every_row(head_size, expected):
  packed = primitives._padded_last_dim(head_size, element_size=2)

  assert packed == expected
  assert (packed * 2) % 16 == 0


def test_fN_packed_row_width_covers_uneven_splits():
  packed = primitives._fN_packed_row_width([1, 2, 3, 4], row_elements=2, min_width=68)

  assert packed == 72
  assert all((size * 2 * packed) % 16 == 0 for size in [1, 2, 3, 4])


def test_all2all_comm_keeps_k_and_lse_off_fp8_when_k_quantization_is_disabled(monkeypatch):
  cp_config = SimpleNamespace(_ulysses_mesh=_FakeMesh(object()),
                              quant_communicate_kwargs={
                                "bits": 8,
                                "quantize_k": False
                              })
  impls = iter([object(), object(), object(), object(), object()])
  monkeypatch.setattr(primitives, "_select_all_to_all_qkv_async_impl",
                      lambda *args, **kwargs: next(impls))
  monkeypatch.setattr(primitives, "_select_all_to_all_o_async_impl",
                      lambda *args, **kwargs: next(impls))

  comm = primitives._All2AllComm(cp_config)

  assert comm._k_impl is not comm._q_impl
  assert comm._lse_impl is not comm._q_impl


def test_fused_all2all_comm_is_enabled_only_for_uaa_and_can_be_disabled(monkeypatch):
  monkeypatch.delenv("CACHE_DIT_DISABLE_ULYSSES_FUSED_QKV", raising=False)
  standard_config = SimpleNamespace(_ulysses_mesh=_FakeMesh(object()),
                                    quant_communicate_kwargs={"bits": 8},
                                    ulysses_anything=False)
  uaa_config = SimpleNamespace(_ulysses_mesh=_FakeMesh(object()),
                               quant_communicate_kwargs={"bits": 8},
                               ulysses_anything=True)

  assert not primitives._FusedAll2AllComm(standard_config).fused_enabled
  assert primitives._FusedAll2AllComm(uaa_config).fused_enabled

  monkeypatch.setenv("CACHE_DIT_DISABLE_ULYSSES_FUSED_QKV", "1")
  assert not primitives._FusedAll2AllComm(uaa_config).fused_enabled


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

  monkeypatch.setattr(primitives,
                      "_select_all_to_all_qkv_async_impl",
                      lambda *_args, bits=None, **_kwargs: _make_impl("k_impl"
                                                                      if bits == 0 else "qv_impl"))
  monkeypatch.setattr(primitives,
                      "_select_all_to_all_o_async_impl",
                      lambda *_args, bits=None, **_kwargs: _make_impl("lse_impl"
                                                                      if bits == 0 else "o_impl"))

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
        "group_size": 0,
        "bits": 0,
      },
    },
    {
      "kind": "o_impl",
      "shape": (2, 3, 4, 5),
      "group": group,
      "metadata": {
        "NUM_QO_HEAD": 4,
        "Q_S_LOCAL": 3,
        "group_size": 0,
        "bits": 0,
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
