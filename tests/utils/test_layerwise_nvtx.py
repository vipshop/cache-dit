"""Exercise NVTX ownership with CPU tensors and replacement push/pop functions."""

import importlib
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from cache_dit.layerwise_nvtx import LayerwiseNvtxMarker, nvtx_scope, stage_range

nvtx_module = importlib.import_module("cache_dit.layerwise_nvtx")


@pytest.fixture
def fake_nvtx(monkeypatch):
  state = SimpleNamespace(events=[], stack=[], fail_push=None, fail_pop=False)

  def push(name):
    state.events.append(("push", name))
    if state.fail_push is not None and state.fail_push(name):
      raise RuntimeError("push failed")
    state.stack.append(name)

  def pop():
    state.events.append(("pop", state.stack.pop()))
    if state.fail_pop:
      raise RuntimeError("pop failed")

  monkeypatch.setattr(nvtx_module, "current_platform", SimpleNamespace(device_type="cuda"))
  monkeypatch.setattr(torch.cuda.nvtx, "range_push", push)
  monkeypatch.setattr(torch.cuda.nvtx, "range_pop", pop)
  return state


def hook_ids(module):
  return tuple(
    tuple(registry)
    for registry in (module._forward_pre_hooks, module._forward_pre_hooks_with_kwargs,
                     module._forward_hooks, module._forward_hooks_with_kwargs,
                     module._forward_hooks_always_called))


def test_compiled_root_keeps_its_range_and_skips_only_its_descendants(fake_nvtx):
  model = torch.nn.Sequential(*[torch.nn.Identity() for _ in range(12)])
  model[1] = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
  model[10] = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
  # Use Module.compile's dispatch slot to exercise the installation boundary on CPU.
  model[1]._compiled_call_impl = model[1]._call_impl
  before = [hook_ids(module) for module in model.modules()]
  inputs = torch.ones(1, 2)
  baseline = model(inputs)
  with LayerwiseNvtxMarker({"transformer": model, "compiled_alias": model[1]}):
    assert torch.equal(model(inputs), baseline)
    assert model[1]._forward_pre_hooks and not model[1][0]._forward_pre_hooks
    assert model[10][0]._forward_pre_hooks
  paths = [name.split(" (")[0] for kind, name in fake_nvtx.events if kind == "push"]
  assert "transformer.1" in paths and "transformer.1.0" not in paths
  assert "transformer.10.0" in paths
  assert [hook_ids(module) for module in model.modules()] == before
  assert not fake_nvtx.stack


def test_stage_nesting_disabling_and_exception_restore_state(fake_nvtx, monkeypatch):
  error = RuntimeError("model failed")
  with pytest.raises(RuntimeError) as raised:
    with nvtx_scope(), stage_range("encode"):
      with nvtx_scope(False), stage_range("disabled"):
        pass
      with nvtx_scope(), stage_range("predict"):
        raise error
  assert raised.value is error
  with stage_range("outside"):
    pass
  assert fake_nvtx.events == [("push", "encode"), ("push", "predict"), ("pop", "predict"),
                              ("pop", "encode")]
  assert fake_nvtx.stack == []
  monkeypatch.setattr(nvtx_module, "current_platform", None)
  with stage_range("outside"), nvtx_scope(False), stage_range("disabled"):
    pass
  assert len(fake_nvtx.events) == 4


@pytest.mark.parametrize("device_type", ["cpu", "npu"])
@pytest.mark.parametrize("entry", [nvtx_scope, lambda: LayerwiseNvtxMarker({})])
def test_enabled_entry_requires_cuda_nvtx(device_type, entry, fake_nvtx, monkeypatch):
  monkeypatch.setattr(nvtx_module, "current_platform", SimpleNamespace(device_type=device_type))
  with pytest.raises(RuntimeError, match="CUDA"):
    with entry():
      pytest.fail("unsupported NVTX must fail at entry")
  assert fake_nvtx.events == []
  with stage_range("outside"):
    pass
  assert fake_nvtx.events == []


def test_failed_stage_push_does_not_pop_outer_range(fake_nvtx):
  fake_nvtx.fail_push = lambda name: name == "inner"
  with nvtx_scope(), stage_range("outer"):
    with pytest.raises(RuntimeError, match="push failed"):
      with stage_range("inner"):
        pytest.fail("failed push must not enter the body")
    assert fake_nvtx.stack == ["outer"]
  assert fake_nvtx.stack == []


@pytest.mark.parametrize("kind", ["stage", "module"])
@pytest.mark.parametrize("model_error", [None, ValueError, KeyboardInterrupt])
def test_pop_failure_preserves_only_the_active_exception(kind, model_error, fake_nvtx):
  error = model_error("model failed") if model_error else None

  class Block(torch.nn.Module):

    def forward(self, value):
      if error:
        raise error
      return value

  block = Block()
  fake_nvtx.fail_pop = True
  # A caller's handled exception must not hide a new cleanup failure.
  try:
    raise LookupError("already handled")
  except LookupError:
    with pytest.raises(model_error or RuntimeError) as caught:
      with nvtx_scope():
        with stage_range("predict") if kind == "stage" else LayerwiseNvtxMarker({"block": block}):
          block(torch.ones(1))
  assert caught.value is error if error else str(caught.value) == "pop failed"
  assert not fake_nvtx.stack
  assert not block._forward_pre_hooks and not block._forward_hooks


def test_kwargs_shapes_and_static_parameters_preserve_inputs_outputs_and_hooks(
    fake_nvtx, monkeypatch):

  class Block(torch.nn.Module):

    def __init__(self):
      super().__init__()
      self.scale = torch.nn.Parameter(torch.ones(2))
      self.proj = torch.nn.Linear(2, 2)

    def forward(self, *, hidden_states, conditioning):
      return self.proj(hidden_states) * self.scale + conditioning["values"][0]

  block = Block()
  inputs = {"hidden_states": torch.ones(3, 2), "conditioning": {"values": [torch.zeros(3, 2)]}}
  observed = []
  handle = block.register_forward_hook(lambda module, args, output: observed.append(output))
  shared = torch.nn.Sequential(block.proj, block.proj)
  outside = torch.nn.Linear(2, 2)
  modules = (block, block.proj, shared, outside)
  before = [hook_ids(m) for m in modules]
  expected = block(**inputs)
  with LayerwiseNvtxMarker({
      "transformer": block,
      "duplicate": block,
      "shared": shared,
      "overlap": block.proj
  }):
    monkeypatch.setattr(
      block, "named_parameters",
      lambda **kwargs: pytest.fail("parameters must be read only at installation"))
    actual = block(**inputs)
    shared(inputs["hidden_states"])
    outside(inputs["hidden_states"])
    assert len(block.proj._forward_pre_hooks) == len(block.proj._forward_hooks) == 1
    with stage_range("manual"):
      pass
    assert observed[-1] is actual
  torch.testing.assert_close(actual, expected)
  assert [hook_ids(m) for m in modules] == before
  names = [name for action, name in fake_nvtx.events if action == "push"]
  assert "transformer (Block)" in names[0]
  assert "params={'scale': (2,)}" in names[0]
  assert "inputs=[(3, 2), (3, 2)]" in names[0]
  assert "transformer.proj (Linear)" in names[1]
  assert "'weight': (2, 2)" in names[1]
  assert [name.split(" (")[0] for name in names] == [
    "transformer", "transformer.proj", "shared", "transformer.proj", "transformer.proj", "manual"
  ]
  assert fake_nvtx.stack == []
  events = list(fake_nvtx.events)
  block(**inputs)
  with stage_range("outside"):
    pass
  assert fake_nvtx.events == events
  handle.remove()


@pytest.mark.parametrize("failure,error_type", [(None, None), ("push", RuntimeError),
                                                ("forward", ValueError),
                                                ("interrupt", KeyboardInterrupt),
                                                ("pre_hook", ValueError)])
def test_recursive_calls_and_failures_restore_ranges_and_hooks(failure, error_type, fake_nvtx):

  class Recursive(torch.nn.Module):

    def forward(self, value, *, depth):
      if depth:
        return self(value, depth=depth - 1)
      if failure == "forward":
        raise ValueError("forward failed")
      if failure == "interrupt":
        raise KeyboardInterrupt("interrupted")
      return value

  module = Recursive()
  before = hook_ids(module)
  if failure == "push":
    fake_nvtx.fail_push = lambda name: len(fake_nvtx.events) == 2
  if failure == "pre_hook":

    def fail_inner(module, args, kwargs):
      if kwargs["depth"] == 1:
        raise ValueError("pre-hook failed")

    handle = module.register_forward_pre_hook(fail_inner, with_kwargs=True)
    before = hook_ids(module)
  with pytest.raises(error_type) if failure else nullcontext():
    with LayerwiseNvtxMarker({"recursive": module}):
      value = torch.ones(1)
      assert module(value, depth=2) is value
  assert hook_ids(module) == before
  assert fake_nvtx.stack == []
  if failure == "pre_hook":
    handle.remove()


@pytest.mark.parametrize("failure", ["root", "register", "remove", "model"])
def test_partial_installation_and_cleanup_preserve_owned_hooks_and_original_error(
    failure, fake_nvtx, monkeypatch):
  module = torch.nn.Sequential(torch.nn.Linear(2, 2))
  existing = module[0].register_forward_hook(lambda *args: None)
  before = hook_ids(module), hook_ids(module[0])
  original_register = module[0].register_forward_hook
  original_error = ValueError("model failed")

  def register(*args, **kwargs):
    if failure == "register":
      raise RuntimeError("register failed")
    handle = original_register(*args, **kwargs)
    remove = handle.remove

    def fail_remove():
      remove()
      raise RuntimeError("remove failed")

    monkeypatch.setattr(handle, "remove", fail_remove)
    return handle

  monkeypatch.setattr(module[0], "register_forward_hook", register)
  with pytest.raises(ValueError if failure in ("root", "model") else RuntimeError) as caught:
    with LayerwiseNvtxMarker({"" if failure == "root" else "transformer": module}):
      assert failure != "register"
      module(torch.ones(1, 2))
      if failure == "model":
        raise original_error
  assert caught.value is original_error if failure == "model" else (
    "nonempty" if failure == "root" else f"{failure} failed") in str(caught.value)
  assert (hook_ids(module), hook_ids(module[0])) == before
  assert not fake_nvtx.stack
  events = list(fake_nvtx.events)
  with stage_range("outside"):
    module(torch.ones(1, 2))
  assert fake_nvtx.events == events
  existing.remove()
