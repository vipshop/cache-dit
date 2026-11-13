"""Regression tests for the owned/family split in `BlockAdapterRegister`.

An adapter registered with `owned=True` encodes knowledge of cache-dit's **own** model
implementation: a class-identity assertion, a layer-KV contract (`force_full_compute`), or a marker
only the vendored transformer sets. A downstream fork can reuse the same class name (sglang's
`LongLive2Transformer3DModel`, vllm-omni's `LongCatVideoAvatarTransformer3DModel`, ...), and before
this split the registry dispatched those foreign classes into the owned adapter, which either
asserted or - worse - silently cached the wrong architecture.

The tests below lock the dispatch boundary:

* a foreign class that merely reuses an owned family name never gets the owned adapter;
* it must not fall through to a shorter family prefix either (a foreign `WanS2V*` must not become a
  plain `Wan` model);
* cache-dit's own classes - including test subclasses of its pipelines - still dispatch;
* family adapters are untouched, and the `_predefined_adapters_has_separate_cfg` fallback keeps
  answering for a rejected class.

Usage:
  python -m pytest tests/api/test_block_adapter_register_owned.py -q
"""

import logging

import pytest
import torch

from cache_dit.caching.block_adapters import BlockAdapterRegister


@pytest.fixture(autouse=True)
def _isolate_registry():
  """Restore the registry tables so the cases cannot leak into each other."""
  adapters = dict(BlockAdapterRegister._adapters)
  owned = set(BlockAdapterRegister._owned_adapters)
  warned = set(BlockAdapterRegister._warned_owned_mismatch)
  try:
    yield
  finally:
    BlockAdapterRegister._adapters.clear()
    BlockAdapterRegister._adapters.update(adapters)
    BlockAdapterRegister._owned_adapters.clear()
    BlockAdapterRegister._owned_adapters.update(owned)
    BlockAdapterRegister._warned_owned_mismatch.clear()
    BlockAdapterRegister._warned_owned_mismatch.update(warned)


class _Block(torch.nn.Module):

  def forward(self, hidden_states, encoder_hidden_states=None):
    return hidden_states


class _TwoStackTransformer(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.transformer_blocks = torch.nn.ModuleList([_Block()])
    self.single_transformer_blocks = torch.nn.ModuleList([_Block()])


def test_foreign_class_with_owned_prefix_is_skipped():
  # Same class name as cache-dit's vendored LongLive 2.0 transformer, but defined outside the
  # package: the owned adapter must not be bound to it.
  ForeignLongLive2 = type("LongLive2Transformer3DModel", (torch.nn.Module, ), {})
  foreign = ForeignLongLive2()

  assert not BlockAdapterRegister.is_supported(foreign)
  assert BlockAdapterRegister.get_adapter(foreign) is None


def test_owned_prefix_does_not_fall_through_to_a_wider_family():
  # "WanS2V" is registered before "Wan"; skipping the owned adapter must stop the lookup instead of
  # binding the foreign class to the generic Wan adapter.
  ForeignWanS2V = type("WanS2VTransformer3DModel", (torch.nn.Module, ), {})
  original = BlockAdapterRegister._adapters["Wan"]
  calls = []

  def _sentinel(*args, **kwargs):
    calls.append(1)
    return original(*args, **kwargs)

  BlockAdapterRegister._adapters["Wan"] = _sentinel
  try:
    assert BlockAdapterRegister.get_adapter(ForeignWanS2V()) is None
    assert calls == [], "foreign WanS2V class fell through to the generic Wan adapter"
  finally:
    BlockAdapterRegister._adapters["Wan"] = original


def test_cache_dit_owned_class_still_dispatches():
  from cache_dit.models.pipelines.longlive2 import LongLive2Pipeline
  from cache_dit.models.pipelines.longlive2.transformer import LongLive2Transformer3DModel

  pipeline = LongLive2Pipeline.__new__(LongLive2Pipeline)
  transformer = LongLive2Transformer3DModel.__new__(LongLive2Transformer3DModel)
  assert BlockAdapterRegister.is_supported(pipeline)
  assert BlockAdapterRegister.is_supported(transformer)

  # Downstream tests and users subclass the pipelines they shipped (keeping the class-name prefix,
  # as `tests/api/test_longlive2_adapter.py` does); the ownership probe walks the MRO so such a
  # subclass keeps dispatching even though its own module is not `cache_dit`.
  class LongLive2TestPipeline(LongLive2Pipeline):
    pass

  derived = LongLive2TestPipeline.__new__(LongLive2TestPipeline)
  assert BlockAdapterRegister.is_supported(derived)


def test_family_prefix_still_dispatches():

  class FluxPipeline(torch.nn.Module):
    pass

  original = BlockAdapterRegister._adapters["Flux"]
  sentinel = object()

  BlockAdapterRegister._adapters["Flux"] = lambda *args, **kwargs: sentinel
  try:
    assert BlockAdapterRegister.is_supported(FluxPipeline())
    assert BlockAdapterRegister.get_adapter(FluxPipeline()) is sentinel
  finally:
    BlockAdapterRegister._adapters["Flux"] = original


def test_flux2_klein_accepts_a_non_diffusers_transformer():
  # `_Flux2Klein` is a diffusers family adapter (its reference class lives in diffusers), so a
  # vendored transformer exposing the same two block stacks must still resolve instead of tripping
  # the strict `isinstance` check.
  transformer = type("_Flux2KleinTransformer2DModel", (_TwoStackTransformer, ), {})()

  adapter = BlockAdapterRegister.get_adapter(transformer)

  assert adapter is not None
  assert adapter.blocks_name == ["transformer_blocks", "single_transformer_blocks"]


def test_hidream_o1_does_not_fall_through_to_hidream():
  # `HiDreamO1Image` is registered before `HiDream`, and the generic HiDream adapter does not assert
  # the class identity, so a fall-through here would silently cache a foreign model.
  ForeignHiDreamO1 = type("HiDreamO1ImageTransformer3DModel", (torch.nn.Module, ), {})

  assert BlockAdapterRegister.get_adapter(ForeignHiDreamO1()) is None


def test_owned_mismatch_is_logged_once_per_class(caplog):
  Foreign = type("LongLive2Transformer3DModel", (torch.nn.Module, ), {})

  with caplog.at_level(logging.WARNING):
    for _ in range(3):
      assert BlockAdapterRegister.get_adapter(Foreign()) is None

  records = [r for r in caplog.records if "reuses cache-dit" in r.getMessage()]
  assert len(records) == 1, "the owned-mismatch warning must be de-duplicated per class name"
  assert "Pass an explicit `BlockAdapter" in records[0].getMessage()


def test_owned_registration_must_stay_supported():
  with pytest.raises(ValueError, match="requires `supported=True`"):
    BlockAdapterRegister.register("SomeOwnedFamily", supported=False, owned=True)


def test_rejected_owned_class_does_not_get_the_cfg_prefix_fallback():
  # `BooguImage` sits in `_predefined_adapters_has_separate_cfg`, but the fallback must not speak for
  # a class refused above: a single-forward vendored model would be counted as two forwards.
  ForeignBooguImage = type("BooguImageTransformer2DModel", (torch.nn.Module, ), {})

  assert BlockAdapterRegister.get_adapter(ForeignBooguImage()) is None
  assert BlockAdapterRegister.has_separate_cfg(ForeignBooguImage()) is False
