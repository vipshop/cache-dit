"""Regression tests for the state bookkeeping of cache-dit's layerwise offload.

The install path has to leave the caller's layout alone: a narrow `module_names=[...]` selection is
an explicit choice to keep the other weights offloaded, while a parameter that hangs directly on a
*non-leaf* module belongs to no target at all and would otherwise stay on the offload device
forever. The AdaLN host residency mode (`--adaln-host-residency`) relies on the second rule.

A separate marker advertises the execution device of an offloaded root, so that a modular pipeline
that builds its own step tensors does not build them on the residency device. It must not be
mistaken for an accelerate offload hook.
"""
from types import SimpleNamespace

import pytest
import torch

from cache_dit.offload import layerwise_offload, remove_layerwise_offload

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA onload device")


class _OrphanStateStack(torch.nn.Module):
  """Two leaf projections plus a non-leaf module that owns a direct parameter."""

  def __init__(self):
    super().__init__()
    self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(2)])
    # `head` owns a child, so leaf discovery never selects it, and `register_tokens` is its own
    # parameter rather than a child's (MiniMax-H3's VAE decoder keeps its register tokens here).
    self.head = torch.nn.Module()
    self.head.register_tokens = torch.nn.Parameter(torch.zeros(2, 4))
    self.head.proj = torch.nn.Linear(4, 4)

  def forward(self, x):
    x = self.blocks[0](x) + self.blocks[1](x)
    return self.head.proj(x) + self.head.register_tokens.sum()


@CUDA
def test_orphan_direct_state_moves_to_onload_device():
  model = _OrphanStateStack()
  layerwise_offload(model, onload_device="cuda", offload_device="cpu")
  try:
    assert model.head.register_tokens.device.type == "cuda"
    # the targets themselves are streamed: they stay on the offload device between calls
    assert model.head.proj.weight.device.type == "cpu"
  finally:
    remove_layerwise_offload(model)


@CUDA
def test_narrow_selection_keeps_unselected_leaves_offloaded():
  model = _OrphanStateStack()
  layerwise_offload(
    model,
    module_names=["blocks.0"],
    onload_device="cuda",
    offload_device="cpu",
  )
  try:
    # an unselected *leaf* is the caller's explicit choice: it must not be pulled onto the device
    assert model.blocks[1].weight.device.type == "cpu"
    assert model.head.proj.weight.device.type == "cpu"
    # an orphan on a non-leaf module is not managed by anyone, so it has to be usable
    assert model.head.register_tokens.device.type == "cuda"
  finally:
    remove_layerwise_offload(model)


def test_execution_device_marker_lifecycle():
  model = _OrphanStateStack()
  assert getattr(model, "_hf_hook", None) is None

  layerwise_offload(model, onload_device="cpu", offload_device="cpu")
  marker = getattr(model, "_hf_hook", None)
  assert marker is not None, "an offloaded root has to advertise its execution device"
  assert torch.device(marker.execution_device) == torch.device("cpu")
  # the marker must not look like an accelerate offload hook, which wraps `forward`
  assert not hasattr(marker, "pre_forward")
  assert not hasattr(marker, "post_forward")

  # torch itself has to accept it: `accelerate.remove_hook_from_module` only calls `detach_hook`
  assert marker.detach_hook(model) is model
  assert marker.init_hook(model) is model

  remove_layerwise_offload(model)
  assert getattr(model, "_hf_hook", None) is None


def _tiny_minimax_h3_transformer():
  from diffusers import MiniMaxH3Transformer3DModel

  return MiniMaxH3Transformer3DModel(
    num_attention_heads=4,
    attention_head_dim=8,
    hidden_size=32,
    num_layers=1,
    num_refiner_layers=1,
    ffn_dim=64,
    in_channels=2,
    audio_in_channels=2,
    patch_size=(1, 2, 2),
    text_dim=4,
    freq_dim=8,
    time_embed_hidden_dim=32,
    time_embed_dim=16,
    rope_freq_dim=1,
  )


def test_layerwise_marker_does_not_disable_forward_pattern_check():
  """`check_forward_pattern` must survive cache-dit's own marker.

  A real accelerate offload hook rewrites `forward`, which is why an existing hook disables the
  pattern check. The marker installed by layerwise offload does not, so the check has to stay on —
  otherwise a model-specific adapter's `check_forward_pattern=True` is silently dropped whenever
  offload is installed before `enable_cache`.
  """
  from cache_dit.caching.block_adapters.block_registers import BlockAdapterRegister
  from cache_dit.offload.layerwise import _LayerwiseExecutionDeviceHook

  transformer = _tiny_minimax_h3_transformer()
  assert BlockAdapterRegister.get_adapter(transformer).check_forward_pattern is True

  # the marker an offloaded root carries advertises a device but does not wrap `forward`
  transformer._hf_hook = _LayerwiseExecutionDeviceHook(torch.device("cpu"))
  assert BlockAdapterRegister.get_adapter(transformer).check_forward_pattern is True

  # a real accelerate hook does wrap it, and then skipping the check is correct
  transformer._hf_hook = SimpleNamespace(
    pre_forward=lambda module, *args, **kwargs: (args, kwargs),
    post_forward=lambda module, output: output,
  )
  assert BlockAdapterRegister.get_adapter(transformer).check_forward_pattern is False


def test_marker_is_removed_when_a_handle_is_removed_directly():
  """`handle.remove()` is a documented public way to uninstall: it must retire the marker too."""
  model = _OrphanStateStack()
  handle = layerwise_offload(model, onload_device="cpu", offload_device="cpu")
  assert getattr(model, "_hf_hook", None) is not None
  handle.remove()
  assert getattr(model, "_hf_hook", None) is None


@CUDA
def test_orphan_state_is_not_migrated_back_on_removal():
  """Freeze the semantics: nobody manages an orphan, so removal leaves it on the onload device."""
  model = _OrphanStateStack()
  layerwise_offload(model, onload_device="cuda", offload_device="cpu")
  assert model.head.register_tokens.device.type == "cuda"
  remove_layerwise_offload(model)
  assert model.head.register_tokens.device.type == "cuda"
  # the targets do go back
  assert model.head.proj.weight.device.type == "cpu"


class _AdalnStack(torch.nn.Module):
  """A transformer-shaped module whose blocks own an `adaln_proj` modulation table."""

  def __init__(self):
    super().__init__()
    self.transformer_blocks = torch.nn.ModuleList()
    for _ in range(2):
      block = torch.nn.Module()
      block.adaln_proj = torch.nn.Linear(4, 8)
      block.ffn = torch.nn.Linear(4, 4)
      self.transformer_blocks.append(block)


@CUDA
def test_compile_is_allowed_while_the_adaln_tables_are_streamed(monkeypatch):
  """`--compile` may stream: the copy runs in a `torch.compiler.disable`d hook, so it is a break."""
  from cache_dit._utils import utils as utils_mod

  installed = []
  monkeypatch.setattr(utils_mod, "layerwise_cpu_offload",
                      lambda module, **kwargs: installed.append(kwargs))

  args = _residency_args(utils_mod)
  args.compile = True

  assert utils_mod.maybe_adaln_host_residency(args, _AdalnStack()) is True
  # only the modulation tables are streamed, and the hook itself is compiled away from the graph
  assert installed and len(installed[0]["module_names"]) == 2


@CUDA
@pytest.mark.parametrize("flag", ["compile_full_graph", "cuda_graph"])
def test_full_graph_and_cuda_graph_are_still_rejected(monkeypatch, flag):
  """`fullgraph` refuses the break, and a CUDA graph would replay the pointers the mirror
  rebinds."""
  from cache_dit._utils import utils as utils_mod

  monkeypatch.setattr(utils_mod, "layerwise_cpu_offload", lambda module, **kwargs: None)
  args = _residency_args(utils_mod)
  args.compile = True
  setattr(args, flag, True)

  with pytest.raises(ValueError, match="stays resident"):
    utils_mod.maybe_adaln_host_residency(args, _AdalnStack())


def _residency_args(utils_mod):
  """Real CLI defaults, so every `layerwise_*` knob the install path reads is present."""
  args = utils_mod.get_base_args(parse=False).parse_args([])
  args.adaln_host_residency = True
  args.compile = False
  args.compile_full_graph = False
  args.cuda_graph = False
  args.quantize = False
  args.parallel_type = ""
  args.device_map_balance = False
  return args
