"""CPU tests for the MOVA context-parallel and tensor-parallel planners.

The CP plan is small enough to assert literally: one split on the first fused block, one gather on
the last block's **first** output slot (the audio stream is replicated and must not be gathered),
and the ``Bn_compute_blocks >= 1`` requirement that keeps the gather reachable on a cached step.

The TP plan is asserted through its guard rails (head divisibility) because applying it needs a real
process group; the sharded forward is covered by the two-rank parity test.
"""

import os
import socket

import pytest
import torch

from cache_dit.distributed import ParallelismBackend, ParallelismConfig
from cache_dit.distributed.transformers.mova import (
  MOVAContextParallelismPlanner,
  MOVATensorParallelismPlanner,
)
from cache_dit.models.pipelines.mova.bridge import DualTowerConditionalBridge
from cache_dit.models.pipelines.mova.stack import MOVAStack
from cache_dit.models.pipelines.mova.wan_video_dit import WanAudioModel, WanModel

_VIDEO_CFG = dict(
  dim=32,
  in_dim=36,
  ffn_dim=64,
  out_dim=16,
  text_dim=16,
  freq_dim=8,
  eps=1e-6,
  patch_size=(1, 2, 2),
  num_heads=4,
  num_layers=2,
  has_image_input=False,
)
_AUDIO_CFG = dict(
  # `head_dim = dim // num_heads = 16` keeps the 1-D rotary table wide enough for `chunk(3)` (a
  # narrower table returns two chunks and the third buffer cannot be registered) and gives the bridge
  # an even number of conditioning heads (32 / head_dim 8 = 4), which tp_size=2 needs.
  dim=32,
  in_dim=8,
  ffn_dim=32,
  out_dim=8,
  text_dim=16,
  freq_dim=8,
  eps=1e-6,
  patch_size=(1, ),
  num_heads=2,
  num_layers=2,
  has_image_input=False,
  vae_type="dac",
)
_BRIDGE_CFG = dict(
  visual_layers=2,
  audio_layers=2,
  visual_hidden_dim=32,
  audio_hidden_dim=32,
  head_dim=8,
  interaction_strategy="full",
  apply_cross_rope=True,
)

# Unequal conditioner head counts: the released model pairs a 3072-wide video tower with a 1536-wide
# audio one (24 vs 12 heads). The default tiny config keeps both widths equal, which hides any mistake
# that divides one side's head count but not the other's, so the parity test also runs this variant:
# 32 / head_dim 8 = 4 heads one way, 48 / 8 = 6 the other. `dim=48` (not 16) is the narrowest audio
# tower whose 1-D rotary table still chunks three ways (`head_dim` 12, see `_AUDIO_CFG`).
_UNEQUAL_AUDIO_CFG = {**_AUDIO_CFG, "dim": 48, "num_heads": 4}
_UNEQUAL_BRIDGE_CFG = {**_BRIDGE_CFG, "audio_hidden_dim": 48, "head_dim": 8}

_VARIANTS = {
  "default": (_AUDIO_CFG, _BRIDGE_CFG),
  "unequal_conditioner_heads": (_UNEQUAL_AUDIO_CFG, _UNEQUAL_BRIDGE_CFG),
}


def _tiny_stack(variant: str = "default") -> MOVAStack:
  """Build a deterministic two-layer stack.

  :param variant: Key of `_VARIANTS`.
  :returns: A `MOVAStack` on CPU.
  """

  audio_cfg, bridge_cfg = _VARIANTS[variant]
  torch.manual_seed(0)
  return MOVAStack(
    video_dit=WanModel(**_VIDEO_CFG),
    video_dit_2=WanModel(**_VIDEO_CFG),
    audio_dit=WanAudioModel(**audio_cfg),
    dual_tower_bridge=DualTowerConditionalBridge(**bridge_cfg),
  ).eval()


def test_unequal_variant_really_has_unequal_conditioner_heads():
  """The second parity variant must differ in the thing it is meant to cover."""

  stack = _tiny_stack("unequal_conditioner_heads")
  block = stack.blocks[0]
  assert block.a2v_conditioner.inner.num_heads == 4
  assert block.v2a_conditioner.inner.num_heads == 6


def _config(**kwargs) -> ParallelismConfig:
  """Build a parallelism config for the planners.

  :param kwargs: Overrides such as ``ulysses_size``.
  :returns: The config.
  """

  return ParallelismConfig(backend=ParallelismBackend.CACHE_DIT, **kwargs)


def test_cp_plan_splits_only_the_video_stream_and_gathers_the_first_slot():
  """One split on block 0, one gather on the last block's video output."""

  stack = _tiny_stack()
  plan = MOVAContextParallelismPlanner()._apply(transformer=stack,
                                                parallelism_config=_config(ulysses_size=2))

  assert set(plan) == {"blocks.0", "blocks.1"}
  assert set(plan["blocks.0"]) == {"hidden_states"}
  split = plan["blocks.0"]["hidden_states"]
  assert split.split_dim == 1 and split.expected_dims == 3

  gather, audio_slot = plan["blocks.1"]
  assert gather.gather_dim == 1
  assert audio_slot is None, "the replicated audio stream must not be gathered"
  assert stack._cp_requires_min_bn_blocks == 1


def test_cp_plan_rejects_ring():
  """Ring attention is not wired for the bridge's key/value gather."""

  with pytest.raises(NotImplementedError, match="Ring"):
    MOVAContextParallelismPlanner()._apply(transformer=_tiny_stack(),
                                           parallelism_config=_config(ulysses_size=2, ring_size=2))


def test_cp_plan_rejects_ulysses_anything():
  """The gather assumes equally sized shards, which UAA breaks."""

  with pytest.raises(NotImplementedError, match="ulysses-anything"):
    MOVAContextParallelismPlanner()._apply(
      transformer=_tiny_stack(),
      parallelism_config=_config(ulysses_size=2, ulysses_anything=True),
    )


def test_cp_plan_rejects_a_backend_without_context_parallel_support(monkeypatch):
  """A backend that silently drops the CP config must be rejected, not tolerated."""

  from cache_dit.attention.backends import register as backend_register

  monkeypatch.setattr(backend_register._AttnBackendRegistry, "is_context_parallel_available",
                      classmethod(lambda cls, backend: False))
  with pytest.raises(ValueError, match="does not implement context parallelism"):
    MOVAContextParallelismPlanner()._apply(transformer=_tiny_stack(),
                                           parallelism_config=_config(ulysses_size=2))


def test_cp_plan_rejects_an_indivisible_head_count():
  """Video heads must split by `tp_size * ulysses_size`."""

  # 4 video heads cannot be split three ways; a hybrid mesh is avoided on purpose because its config
  # needs a live process group.
  with pytest.raises(ValueError, match="not divisible by ulysses_size"):
    MOVAContextParallelismPlanner()._apply(transformer=_tiny_stack(),
                                           parallelism_config=_config(ulysses_size=3))


_PARITY_TOL = 1e-6


def _free_port() -> int:
  """Pick a free TCP port so the two parametrized runs cannot collide.

  Both ranks of one run share the store port, but a crashed run may leave it bound; a hard-coded
  port turns that into a 10-minute NCCL timeout on the *next* run.

  :returns: A port the OS currently reports as free.
  """

  with socket.socket() as sock:
    sock.bind(("", 0))
    return sock.getsockname()[1]


def _parity_inputs(device, dtype):
  """Fixed inputs shared by the reference and the parallel run.

  :param device: Device of the tensors.
  :param dtype: Dtype of the tensors.
  :returns: Tuple of positional/keyword inputs for `MOVAStack.forward`.
  """

  generator = torch.Generator(device="cpu").manual_seed(7)
  visual = torch.randn(1, 36, 3, 4, 4, generator=generator).to(device=device, dtype=dtype)
  audio = torch.randn(1, 8, 5, generator=generator).to(device=device, dtype=dtype)
  context = torch.randn(1, 4, 16, generator=generator).to(device=device, dtype=dtype)
  timestep = torch.tensor([500.0], device=device)
  return visual, audio, context, timestep


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("mode,variant", [("cp", "default"), ("tp", "default"),
                                          ("tp", "unequal_conditioner_heads")])
def test_parallel_forward_matches_single_rank(mode, variant):
  """One fused step in float32 must match the un-parallelized reference within 1e-6.

  This is the correctness judgment for the parallel plans: bf16 end-to-end PSNR cannot separate
  "different sample" from "wrong sharding", while a float32 relative error can. The TP case runs twice:
  once with equal conditioner widths and once with unequal head counts (4 vs 6), because only the
  second one can catch a head count divided on one bridge direction but not the other.
  """

  import torch.multiprocessing as mp

  mp.spawn(_run_rank, args=(mode, _free_port(), variant), nprocs=2, join=True)


def _run_rank(rank, mode, port, variant="default"):
  import torch.distributed as dist

  from cache_dit.distributed.backend import ParallelismBackend
  from cache_dit.distributed.dispatch import enable_parallelism

  dist.init_process_group("nccl", init_method=f"tcp://localhost:{port}", rank=rank, world_size=2)
  torch.cuda.set_device(rank)
  try:
    stack = _tiny_stack(variant).to("cuda")
    visual, audio, context, timestep = _parity_inputs("cuda", torch.float32)
    with torch.no_grad():
      reference = stack(visual, audio, context, timestep)
      audio_reference = reference[1].float()
      video_reference = reference[0].float()

    if mode == "cp":
      config = ParallelismConfig(ulysses_size=2)
    else:
      config = ParallelismConfig(tp_size=2, backend=ParallelismBackend.NATIVE_PYTORCH)
      # `MOVA_TP_SCOPE=ffn` bisects the plan: only the FFN entries are kept, so a remaining mismatch
      # is an FFN layout problem and a vanishing one points at the attention/norm chain.
      if os.environ.get("MOVA_TP_SCOPE") == "ffn":
        original = MOVATensorParallelismPlanner._block_plan
        MOVATensorParallelismPlanner._block_plan = (lambda self, block, tower, mesh, size: {
          key: value
          for key, value in original(self, block, tower, mesh, size).items() if ".ffn." in key
        })
        MOVATensorParallelismPlanner._conditioner_plan = lambda self, *a, **k: {}
        MOVATensorParallelismPlanner._shard_qk_norms = staticmethod(lambda *a, **k: None)
    enable_parallelism(stack, config)

    with torch.no_grad():
      output = stack(visual, audio, context, timestep)

    # Every rank asserts: the outputs are replicated after the collectives, so a rank-0-only
    # assertion would leave the other rank blocked in `barrier()` until the NCCL timeout turns a
    # plain mismatch into a ten-minute hang.
    for name, ref, out in (("video", video_reference, output[0].float()), ("audio", audio_reference,
                                                                           output[1].float())):
      relative = ((out - ref).abs().mean() / ref.abs().mean().clamp_min(1e-6)).item()
      if rank == 0:
        print(f"[mova-{mode}/{variant}] {name} rel={relative:.3e}")
      assert out.shape == ref.shape
      assert relative < _PARITY_TOL, f"MOVA {mode} {name} diverged (rel={relative:.3e})"
    dist.barrier()
  finally:
    dist.destroy_process_group()


def test_tp_plan_rejects_an_indivisible_head_count():
  """`shard_div_attr` must fail before any parameter is sharded."""

  class _FakeGroup:

    def size(self):
      return 8

    def rank(self):
      return 0

  class _FakeMesh:

    def size(self):
      return 8

    def get_group(self):
      return _FakeGroup()

  with pytest.raises(ValueError, match="not divisible by tp_size"):
    MOVATensorParallelismPlanner()._block_plan(_tiny_stack().blocks[0], "video_block", _FakeMesh(),
                                               8)
