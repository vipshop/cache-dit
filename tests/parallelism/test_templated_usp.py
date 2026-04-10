from types import SimpleNamespace

import torch

import cache_dit.distributed.core._templated_usp as usp


class _FakeHandle:

  def __init__(self, value):
    self._value = value

  def wait(self):
    return self._value

  def __call__(self):
    return self.wait()


class _FakeComm:

  instances = []

  def __init__(self, cp_config):
    self.cp_config = cp_config
    self.calls = []
    self.metadata_source = None
    _FakeComm.instances.append(self)

  def init_meta(self, query, **kwargs):
    self.metadata_source = query
    self.calls.append(("init_meta", tuple(query.shape), dict(kwargs)))
    return self

  def send_q(self, query):
    self.calls.append(("send_q", tuple(query.shape)))
    return _FakeHandle(query + 1)

  def send_k(self, key):
    self.calls.append(("send_k", tuple(key.shape)))
    return _FakeHandle(key + 2)

  def send_v(self, value):
    self.calls.append(("send_v", tuple(value.shape)))
    return _FakeHandle(value + 3)

  def send_o(self, out):
    self.calls.append(("send_o", tuple(out.shape)))
    return _FakeHandle(out + 4)

  def send_lse(self, lse):
    self.calls.append(("send_lse", tuple(lse.shape)))
    return _FakeHandle(lse + 5)


def test_unified_templated_usp_uses_all2all_comm(monkeypatch):
  _FakeComm.instances.clear()
  captured = {}

  def _fake_ring_apply(
    query,
    key,
    value,
    attn_mask,
    dropout_p,
    is_causal,
    scale,
    enable_gqa,
    return_lse,
    forward_op,
    backward_op,
    _cp_config,
  ):
    captured["query"] = query
    captured["key"] = key
    captured["value"] = value
    out = query + key + value
    if return_lse:
      lse = out[..., 0]
      return out, lse
    return out

  monkeypatch.setattr(usp, "_All2AllComm", _FakeComm)
  monkeypatch.setattr(usp.RingAttention, "apply", _fake_ring_apply)

  cp_config = SimpleNamespace(ulysses_float8=False)
  query = torch.randn(2, 3, 4, 5)
  key = torch.randn(2, 3, 4, 5)
  value = torch.randn(2, 3, 4, 5)

  out, lse = usp.USPAttention.apply(
    query,
    key,
    value,
    None,
    0.0,
    False,
    None,
    False,
    True,
    object(),
    object(),
    cp_config,
  )

  comm = _FakeComm.instances[-1]
  assert comm.metadata_source is query
  assert [call[0] for call in comm.calls] == [
    "init_meta",
    "send_q",
    "send_k",
    "send_v",
    "send_o",
    "send_lse",
  ]
  assert torch.allclose(captured["query"], query + 1)
  assert torch.allclose(captured["key"], key + 2)
  assert torch.allclose(captured["value"], value + 3)
  expected_out = captured["query"] + captured["key"] + captured["value"] + 4
  expected_lse = (captured["query"] + captured["key"] + captured["value"])[..., 0] + 5
  assert torch.allclose(out, expected_out)
  assert torch.allclose(lse, expected_lse)


def test_unified_templated_usp_float8_entrypoint_uses_same_impl(monkeypatch):
  _FakeComm.instances.clear()

  monkeypatch.setattr(usp, "_All2AllComm", _FakeComm)
  monkeypatch.setattr(
    usp.RingAttention,
    "apply",
    lambda query, key, value, *args, **kwargs: query + key + value,
  )

  cp_config = SimpleNamespace(ulysses_float8=True)
  query = torch.randn(2, 3, 4, 5)
  key = torch.randn(2, 3, 4, 5)
  value = torch.randn(2, 3, 4, 5)

  out = usp.USPAttention.apply(
    query,
    key,
    value,
    None,
    0.0,
    False,
    None,
    False,
    False,
    object(),
    object(),
    cp_config,
  )

  comm = _FakeComm.instances[-1]
  assert [call[0] for call in comm.calls] == [
    "init_meta",
    "send_q",
    "send_k",
    "send_v",
    "send_o",
  ]
  expected_out = (query + 1) + (key + 2) + (value + 3) + 4
  assert torch.allclose(out, expected_out)
