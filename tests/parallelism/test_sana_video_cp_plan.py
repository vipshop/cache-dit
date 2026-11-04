"""CPU checks for the SANA-Video CP plan and TP layer plan.

SANA-Video is unusual in two ways that are easy to regress silently:

* the self-attention is **linear attention**, whose two aggregates are sums over the token axis, so
  the CP path is one all-reduce rather than a Ulysses head all-to-all;
* the feed-forward reshapes with the `frames/height/width` integers and runs a **3-tap temporal
  convolution**, which is why the plan shards the frame axis (5D `hidden_states`, `split_dim=2`)
  instead of the token axis, and why `conv_temp` gets a halo.

Both are string/shape contracts: a wrong plan still renders a plausible clip. These tests assert
them directly, plus the unarmed bit-exactness of every installed patch.

Usage:
  python -m pytest tests/parallelism/test_sana_video_cp_plan.py -q
"""

import pytest
import torch

from cache_dit.distributed.config import ParallelismConfig
from cache_dit.distributed.transformers.planners import _activate_cp_planners, _activate_tp_planners
from cache_dit.distributed.transformers.register import (
  ContextParallelismPlannerRegister,
  TensorParallelismPlannerRegister,
)
from cache_dit.distributed.transformers.sana_video import (
  SanaVideoContextParallelismPlanner,
  SanaVideoTensorParallelismPlanner,
)

diffusers = pytest.importorskip("diffusers")

if not hasattr(diffusers, "SanaVideoTransformer3DModel"):
  pytest.skip("SANA-Video requires a newer diffusers", allow_module_level=True)

_NUM_BLOCKS = 2
_HEADS = 4
_HEAD_DIM = 8
_IN_CHANNELS = 4
_CAPTION_CHANNELS = 8


def _tiny_transformer():
  from diffusers import SanaVideoTransformer3DModel

  torch.manual_seed(0)
  return SanaVideoTransformer3DModel(
    in_channels=_IN_CHANNELS,
    out_channels=_IN_CHANNELS,
    num_attention_heads=_HEADS,
    attention_head_dim=_HEAD_DIM,
    num_layers=_NUM_BLOCKS,
    num_cross_attention_heads=_HEADS,
    cross_attention_head_dim=_HEAD_DIM,
    cross_attention_dim=_HEADS * _HEAD_DIM,
    caption_channels=_CAPTION_CHANNELS,
    mlp_ratio=2.0,
    sample_size=8,
    patch_size=(1, 2, 2),
  ).eval()


class _FakeMesh:
  """Minimal stand-in for a TP device mesh: the planner reads `size()` before sharding."""

  def __init__(self, size: int):
    self._size = size

  def size(self) -> int:
    return self._size


class _FakeCPConfig:
  """Stand-in for `_ContextParallelConfig`: only the fields the patches read."""

  def __init__(self, ring: int = 1, ulysses: int = 1, anything: bool = False):
    self.mesh_shape = (ring, ulysses)
    self.ulysses_anything = anything
    self._world_size = ring * ulysses


def _small_inputs(frames: int = 2):
  generator = torch.Generator().manual_seed(3)
  return dict(
    hidden_states=torch.randn(1, _IN_CHANNELS, frames, 8, 8, generator=generator),
    encoder_hidden_states=torch.randn(1, 5, _CAPTION_CHANNELS, generator=generator),
    timestep=torch.zeros(1),
    return_dict=False,
  )


def test_cp_plan_splits_the_frame_axis_of_the_5d_grid():
  transformer = _tiny_transformer()
  plan = SanaVideoContextParallelismPlanner()._apply(
    transformer=transformer,
    parallelism_config=ParallelismConfig(ulysses_size=2),
  )

  assert set(plan) == {""}, f"unexpected CP plan keys: {sorted(plan)}"
  subplans = plan[""]
  assert len(subplans) == 2
  split, gather = subplans
  assert isinstance(split, dict)
  hidden = split["hidden_states"]
  # Frame axis of `[B, C, frames, H, W]`: token-axis splitting would desynchronise the block's
  # `frames/height/width` reshape and the temporal convolution.
  assert hidden.split_dim == 2
  assert hidden.expected_dims == 5
  assert hidden.split_output is False
  assert gather.gather_dim == 2
  assert gather.expected_dims == 5


def test_cp_plan_splits_the_i2v_per_frame_timestep():
  """I2V passes a 5D timestep that flattens to the same token grid as `hidden_states`."""

  plan = SanaVideoContextParallelismPlanner()._apply(
    transformer=_tiny_transformer(),
    parallelism_config=ParallelismConfig(ulysses_size=2),
  )
  split = plan[""][0]
  assert set(split) == {"hidden_states", "timestep"}
  timestep = split["timestep"]
  # `expected_dims=5` is what keeps the T2V 1D `[B]` timestep untouched.
  assert timestep.split_dim == 2
  assert timestep.expected_dims == 5
  assert timestep.split_output is False


def test_cp_plan_installs_the_halo_and_rope_patches():
  transformer = _tiny_transformer()
  SanaVideoContextParallelismPlanner()._apply(
    transformer=transformer,
    parallelism_config=ParallelismConfig(ulysses_size=2),
  )
  assert getattr(transformer.rope, "_cp_patched", False)
  for block in transformer.transformer_blocks:
    assert getattr(block.ff.conv_temp, "_cp_patched", False)


def test_cp_plan_rejects_a_frame_count_the_sharder_cannot_split(monkeypatch):
  import cache_dit.distributed.transformers.sana_video as planner_module

  monkeypatch.setattr(planner_module, "_cp_is_armed", lambda owner: True)
  monkeypatch.setattr(planner_module, "_cp_ranks", lambda owner: (0, 2))

  transformer = _tiny_transformer()
  # `frames=81` -> 21 latent frames, which a CP size of 2 cannot split evenly.
  with pytest.raises(ValueError, match="ulysses-anything"):
    planner_module._validate_frame_split(transformer, torch.zeros(1, _IN_CHANNELS, 21, 8, 8))
  # An even latent frame count passes through.
  planner_module._validate_frame_split(transformer, torch.zeros(1, _IN_CHANNELS, 20, 8, 8))


def test_odd_frame_count_passes_on_the_uneven_split(monkeypatch):
  import cache_dit.distributed.transformers.sana_video as planner_module

  monkeypatch.setattr(planner_module, "_cp_is_armed", lambda owner: True)
  monkeypatch.setattr(planner_module, "_cp_ranks", lambda owner: (0, 2))

  transformer = _tiny_transformer()
  transformer._cp_config = _FakeCPConfig(ring=1, ulysses=2, anything=True)
  # 21 latent frames on the uneven path is legal: the halo and the RoPE window work from per-rank
  # lengths, so 11/10 is as correct as 10/10.
  planner_module._validate_frame_split(transformer, torch.zeros(1, _IN_CHANNELS, 21, 8, 8))

  transformer._cp_config = _FakeCPConfig(ring=1, ulysses=2, anything=False)
  with pytest.raises(ValueError, match="ulysses-anything"):
    planner_module._validate_frame_split(transformer, torch.zeros(1, _IN_CHANNELS, 21, 8, 8))


def test_tp_only_ranks_are_not_armed():
  """`_world_size` is the job size, so TP-only ranks would otherwise take the CP paths."""

  import cache_dit.distributed.transformers.sana_video as planner_module

  transformer = _tiny_transformer()
  transformer._cp_config = _FakeCPConfig(ring=1, ulysses=2)
  assert planner_module._cp_is_armed(transformer)
  transformer._cp_config = _FakeCPConfig(ring=1, ulysses=1)
  assert not planner_module._cp_is_armed(transformer)
  del transformer._cp_config
  assert not planner_module._cp_is_armed(transformer)


def test_patches_are_inert_without_an_armed_cp_config():
  reference = _tiny_transformer()
  inputs = _small_inputs()
  with torch.no_grad():
    expected = reference(**inputs)[0]

  transformer = _tiny_transformer()
  # Applying the planner installs the class-level processor patches and the per-instance rope /
  # conv_temp patches; without an armed `_cp_config` every one of them must fall back upstream.
  SanaVideoContextParallelismPlanner()._apply(
    transformer=transformer,
    parallelism_config=ParallelismConfig(ulysses_size=2),
  )
  with torch.no_grad():
    actual = transformer(**inputs)[0]
  assert torch.equal(actual, expected), "the cp-free path must stay bit-identical"


def test_cp_plan_admits_both_ulysses_and_ring():
  # The plan never uses a head all-to-all, so both CP strategies collapse to the same split.
  for kwargs in ({"ulysses_size": 2}, {"ring_size": 2}):
    plan = SanaVideoContextParallelismPlanner()._apply(
      transformer=_tiny_transformer(),
      parallelism_config=ParallelismConfig(**kwargs),
    )
    assert set(plan) == {""}


def test_tp_layer_plan_shards_attention_only(monkeypatch):
  import cache_dit.distributed.transformers.sana_video as planner_module

  captured = []
  monkeypatch.setattr(
    planner_module,
    "parallelize_module",
    lambda module, device_mesh, parallelize_plan: captured.append(parallelize_plan),
  )
  # `_DistributedRMSNorm.from_rmsnorm` needs a real process group; the variance all-reduce itself is
  # covered by the 2-rank tiny test, here only the call sites matter.
  replaced = []

  def _fake_from_rmsnorm(tp_mesh, norm):
    replaced.append(norm)
    return norm

  monkeypatch.setattr(planner_module._DistributedRMSNorm, "from_rmsnorm", _fake_from_rmsnorm)

  transformer = _tiny_transformer()
  _, layer_plans = SanaVideoTensorParallelismPlanner().parallelize_transformer(
    transformer, _FakeMesh(2))

  assert len(layer_plans) == _NUM_BLOCKS
  assert set(layer_plans[0]) == {
    "attn1.to_q",
    "attn1.to_k",
    "attn1.to_v",
    "attn1.to_out.0",
    "attn2.to_q",
    "attn2.to_k",
    "attn2.to_v",
    "attn2.to_out.0",
  }
  # The Conv2d feed-forward cannot be sharded by torch TP and must never appear in the plan.
  assert not any("ff" in key for key in layer_plans[0])
  # Heads follow the local shard, and the across-heads QK norms are replaced on both attentions.
  assert transformer.transformer_blocks[0].attn1.heads == _HEADS // 2
  assert transformer.transformer_blocks[0].attn2.heads == _HEADS // 2
  assert len(replaced) == _NUM_BLOCKS * 4


def test_tp_rejects_an_indivisible_head_count(monkeypatch):
  import cache_dit.distributed.transformers.sana_video as planner_module

  monkeypatch.setattr(planner_module, "parallelize_module", lambda *args, **kwargs: None)
  # 4 heads divide by 2 and 4 but not by 3.
  with pytest.raises(ValueError, match="divisible by tp_size"):
    SanaVideoTensorParallelismPlanner().parallelize_transformer(_tiny_transformer(), _FakeMesh(3))


def test_planners_are_registered_under_the_exact_class_name():
  _activate_cp_planners()
  _activate_tp_planners()

  assert (ContextParallelismPlannerRegister._cp_planner_registry["SanaVideoTransformer3DModel"]
          is SanaVideoContextParallelismPlanner)
  assert (TensorParallelismPlannerRegister._tp_planner_registry["SanaVideoTransformer3DModel"]
          is SanaVideoTensorParallelismPlanner)
  # SANA-Video 2.0 keeps its own planners: the two families must not share a key.
  assert "SanaVideo2Transformer3DModel" in ContextParallelismPlannerRegister._cp_planner_registry
  assert (ContextParallelismPlannerRegister._cp_planner_registry["SanaVideo2Transformer3DModel"]
          is not SanaVideoContextParallelismPlanner)
