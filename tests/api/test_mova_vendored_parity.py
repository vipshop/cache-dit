"""Bit-exact parity between the vendored MOVA modules and the released code.

The vendored tree (`cache_dit.models.pipelines.mova`) is a copy of the released `mova` package, so the
one thing that must hold forever is that it computes the same numbers *and* shares the state-dict key
layout. Small synthetic configs on CPU in float32 exercise the real code paths (rope, adaLN, gated
residuals, DAC-style projections, bridge ordering) without loading 73 GB of weights.

The reference checkout is not vendored here, so the whole module skips unless it is present. Point
`MOVA_REF_DIR` at a `mova` source tree (the default matches the layout used while integrating).
"""

import importlib.machinery
import os
import sys
import types

import pytest
import torch

_REF_DIR = os.environ.get("MOVA_REF_DIR", "/tmp/ref/MOVA")

if not os.path.isdir(os.path.join(_REF_DIR, "mova")):
  pytest.skip(f"MOVA reference checkout not found at {_REF_DIR}", allow_module_level=True)


def _disable_hub_kernels():
  """Make the released module take its SDPA fallback instead of fetching flash-attn from the Hub.

  The released importer wraps `get_kernel("kernels-community/flash-attn3")` in a
  `ModuleNotFoundError` handler, but `kernels` raises `ValueError` when the publisher trust status
  cannot be verified (offline), which escapes the fallback entirely.

  :returns: The previous `sys.modules["kernels"]` entry, to be restored by the caller.
  """

  previous = sys.modules.get("kernels")
  stub = types.ModuleType("kernels")

  def _unavailable(*args, **kwargs):
    raise ModuleNotFoundError("hub kernels disabled for the parity test")

  stub.get_kernel = _unavailable
  stub.__spec__ = importlib.machinery.ModuleSpec("kernels", loader=None)
  stub.__version__ = "0.0.0"
  sys.modules["kernels"] = stub
  return previous


def _stub_yunchang():
  """Stand in for `yunchang`, which the released module only needs for `USPAttention`.

  Without it the released fallback defines `class AttnType: FA3 = None`, but `USPAttention.__init__`
  defaults to `AttnType.FA`, so the module cannot even be imported. The stub is never instantiated by
  these tests (they only exercise the plain `AttentionModule` path).

  :returns: The previous `sys.modules` entries, to be restored by the caller.
  """

  previous = (sys.modules.get("yunchang"), sys.modules.get("yunchang.kernels"))
  yunchang = types.ModuleType("yunchang")

  class _LongContextAttention:

    def __init__(self, *args, **kwargs):
      raise RuntimeError("yunchang is stubbed in the parity test")

  yunchang.LongContextAttention = _LongContextAttention
  kernels = types.ModuleType("yunchang.kernels")

  class _AttnType:
    FA = "fa"
    FA3 = "fa3"

  kernels.AttnType = _AttnType
  yunchang.kernels = kernels
  sys.modules["yunchang"] = yunchang
  sys.modules["yunchang.kernels"] = kernels
  return previous


def _import_reference():
  """Import the released modules with the optional backends stubbed out.

  :returns: Tuple of the released `WanModel`, `WanAudioModel` and `DualTowerConditionalBridge`.
  """

  sys.path.insert(0, _REF_DIR)
  previous_kernels = _disable_hub_kernels()
  previous_yunchang, previous_yunchang_kernels = _stub_yunchang()
  try:
    import mova.diffusion.models.wan_video_dit as ref_video_mod
    from mova.diffusion.models.interactionv2 import DualTowerConditionalBridge
    from mova.diffusion.models.wan_audio_dit import WanAudioModel
    from mova.diffusion.models.wan_video_dit import WanModel
  finally:
    # The stubs exist only to make the released package importable; leaving them in `sys.modules`
    # would shadow the real `kernels` / `yunchang` packages for every other test in this process.
    for name, value in (("kernels", previous_kernels), ("yunchang", previous_yunchang),
                        ("yunchang.kernels", previous_yunchang_kernels)):
      if value is None:
        sys.modules.pop(name, None)
      else:
        sys.modules[name] = value

  # The environment ships `flash_attn`, so the released module would route attention through a
  # CUDA-only kernel and the CPU parity run cannot execute. Disable its optional backends so both sides
  # run the same scaled-dot-product path: this compares the vendored math, not the backend choice.
  ref_video_mod.FLASH_ATTN_2_AVAILABLE = False
  ref_video_mod.FLASH_ATTN_3_AVAILABLE = False
  ref_video_mod.SAGE_ATTN_AVAILABLE = False
  return WanModel, WanAudioModel, DualTowerConditionalBridge


RefWanModel, RefWanAudioModel, RefDualTowerConditionalBridge = _import_reference()

from cache_dit.models.pipelines.mova.bridge import DualTowerConditionalBridge  # noqa: E402
from cache_dit.models.pipelines.mova.wan_video_dit import (  # noqa: E402
  WanAudioModel, WanModel, audio_freqs, video_freqs,
)

_VIDEO_CFG = dict(
  dim=128,
  in_dim=36,
  ffn_dim=256,
  out_dim=16,
  text_dim=64,
  freq_dim=32,
  eps=1e-6,
  patch_size=(1, 2, 2),
  num_heads=4,
  num_layers=2,
  has_image_input=False,
)
_AUDIO_CFG = dict(
  dim=64,
  in_dim=32,
  ffn_dim=128,
  out_dim=32,
  text_dim=64,
  freq_dim=32,
  eps=1e-6,
  patch_size=(1, ),
  num_heads=4,
  num_layers=2,
  has_image_input=False,
  vae_type="dac",
)


def _assert_bit_exact(name, reference, actual):
  """Assert one tensor pair is bit-identical.

  :param name: Contract name for the failure message.
  :param reference: Tensor from the released module.
  :param actual: Tensor from the vendored module.
  """

  assert actual.shape == reference.shape, f"{name}: shape {actual.shape} != {reference.shape}"
  assert torch.equal(actual,
                     reference), (f"{name}: max |diff| = "
                                  f"{(actual.float() - reference.float()).abs().max().item():.3e}")


def test_video_tower_is_bit_exact():
  """The vendored video tower reproduces the released forward and the released state-dict keys."""

  torch.manual_seed(1)
  reference = RefWanModel(**_VIDEO_CFG).eval()
  ours = WanModel(**_VIDEO_CFG).eval()
  missing, unexpected = ours.load_state_dict(reference.state_dict(), strict=True)
  assert not missing and not unexpected

  x = torch.randn(1, 36, 3, 4, 4)
  timestep = torch.tensor([500.0])
  context = torch.randn(1, 8, 64)
  with torch.no_grad():
    expected = reference(x, timestep, context)
    t, t_mod = ours.time_embed(timestep)
    ctx = ours.text_embedding(context)
    hidden, grid = ours.patchify(x)
    freqs = video_freqs(ours.freqs, grid[0], grid[1], grid[2], hidden.device)
    for block in ours.blocks:
      hidden = block(hidden, ctx, t_mod, freqs)
    actual = ours.unpatchify(ours.head(hidden, t), grid)

    from mova.diffusion.models.wan_video_dit import sinusoidal_embedding_1d
    expected_t = reference.time_embedding(sinusoidal_embedding_1d(reference.freq_dim, timestep))

  _assert_bit_exact("video tower forward", expected, actual)
  _assert_bit_exact("video time embedding", expected_t, t)


def test_audio_tower_is_bit_exact():
  """The vendored audio tower reproduces the released forward and the released state-dict keys."""

  torch.manual_seed(2)
  reference = RefWanAudioModel(**_AUDIO_CFG).eval()
  ours = WanAudioModel(**_AUDIO_CFG).eval()
  missing, unexpected = ours.load_state_dict(reference.state_dict(), strict=True)
  assert not missing and not unexpected

  x = torch.randn(1, 32, 9)
  timestep = torch.tensor([300.0])
  context = torch.randn(1, 8, 64)
  with torch.no_grad():
    expected = reference(x, timestep, context)
    t, t_mod = ours.time_embed(timestep)
    ctx = ours.text_embedding(context)
    hidden, grid = ours.patchify(x)
    freqs = audio_freqs(ours.freqs, grid[0], hidden.device)
    for block in ours.blocks:
      hidden = block(hidden, ctx, t_mod, freqs)
    actual = ours.unpatchify(ours.head(hidden, t), grid)

  _assert_bit_exact("audio tower forward", expected, actual)


def test_bridge_is_bit_exact_and_keeps_the_ordering():
  """The vendored bridge matches the released one, rope tables included, and keeps v2a's
  ordering."""

  torch.manual_seed(3)
  cfg = dict(
    visual_layers=2,
    audio_layers=2,
    visual_hidden_dim=128,
    audio_hidden_dim=64,
    head_dim=32,
    interaction_strategy="full",
    apply_cross_rope=True,
  )
  reference = RefDualTowerConditionalBridge(**cfg).eval()
  ours = DualTowerConditionalBridge(**cfg).eval()
  missing, unexpected = ours.load_state_dict(reference.state_dict(), strict=True)
  assert not missing and not unexpected

  visual = torch.randn(1, 5, 128)
  audio = torch.randn(1, 3, 64)
  grid = (1, 1, 5)
  with torch.no_grad():
    ref_v_rope, ref_a_rope = reference.build_aligned_freqs(video_fps=24.0,
                                                           grid_size=grid,
                                                           audio_steps=3,
                                                           dtype=torch.float32)
    our_v_rope, our_a_rope = ours.build_aligned_freqs(video_fps=24.0,
                                                      grid_size=grid,
                                                      audio_steps=3,
                                                      dtype=torch.float32)
    ref_v, ref_a = reference(0, visual, audio, x_freqs=ref_v_rope, y_freqs=ref_a_rope)
    our_v, our_a = ours(0, visual, audio, x_freqs=our_v_rope, y_freqs=our_a_rope)
    # v2a must read the pre-layer visual states: zeroing them has to change the audio output.
    _, ref_a_zeroed = reference(0,
                                torch.zeros_like(visual),
                                audio,
                                x_freqs=ref_v_rope,
                                y_freqs=ref_a_rope)

  _assert_bit_exact("bridge aligned rope cos", ref_v_rope[0], our_v_rope[0])
  _assert_bit_exact("bridge aligned rope sin", ref_a_rope[1], our_a_rope[1])
  _assert_bit_exact("bridge visual out", ref_v, our_v)
  _assert_bit_exact("bridge audio out", ref_a, our_a)
  assert not torch.equal(ref_a, ref_a_zeroed), "bridge v2a looks independent of the visual states"
