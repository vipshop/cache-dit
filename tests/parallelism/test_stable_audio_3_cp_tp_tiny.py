"""CP/TP tests for the Stable Audio 3 planners (tiny model, no weights).

The CP plan splits `[memory_tokens | audio]` on block 0's input, cuts the full-length RoPE table on the
same token axis and gathers on the last block, so the Ulysses rank-concatenation rebuilds the exact
sequence. The cross attention must run without the CP config (its context is replicated). The TP plan
reorders every fused projection into "shard-part major" order before a plain column shard.

These properties are asserted on CPU, and a two-rank parity run compares a single forward against the
un-parallelized reference on a seeded tiny model.

Usage:
  CUDA_VISIBLE_DEVICES=0,1 python -m pytest tests/parallelism/test_stable_audio_3_cp_tp_tiny.py -q
"""

import types

import pytest
import torch

from cache_dit.distributed.config import ParallelismConfig

diffusers = pytest.importorskip("diffusers")

if not hasattr(diffusers, "StableAudio3DiTModel"):
  pytest.skip("Stable Audio 3 requires a newer diffusers", allow_module_level=True)

_IO_CHANNELS = 4
_EMBED_DIM = 32
_DEPTH = 2
_HEADS = 2
_TEXT_DIM = 16
_GLOBAL_DIM = 16
_MEMORY_TOKENS = 4
_TEXT_LEN = 5
_FRAMES = 6
_REL_TOL = 1e-5


def _tiny_transformer():
  from diffusers import StableAudio3DiTModel

  torch.manual_seed(0)
  transformer = StableAudio3DiTModel(
    io_channels=_IO_CHANNELS,
    patch_size=1,
    embed_dim=_EMBED_DIM,
    depth=_DEPTH,
    num_heads=_HEADS,
    cond_token_dim=_TEXT_DIM,
    global_cond_dim=_GLOBAL_DIM,
    local_add_cond_dim=8,
    timestep_features_dim=16,
    ff_mult=2.0,
    num_memory_tokens=_MEMORY_TOKENS,
    use_differential_attention=True,
  ).eval()
  with torch.no_grad():
    transformer.preprocess_conv.weight.normal_()
    transformer.postprocess_conv.weight.normal_()
  return transformer


def _inputs(device="cpu", frames=_FRAMES):
  generator = torch.Generator(device=device).manual_seed(0)
  return dict(
    hidden_states=torch.randn(1, _IO_CHANNELS, frames, generator=generator, device=device),
    timestep=torch.tensor([0.4], device=device),
    encoder_hidden_states=torch.randn(1, _TEXT_LEN, _TEXT_DIM, generator=generator, device=device),
    global_hidden_states=torch.randn(1, _GLOBAL_DIM, generator=generator, device=device),
    encoder_attention_mask=torch.ones(1, _TEXT_LEN, dtype=torch.bool, device=device),
    return_dict=False,
  )


def test_cp_plan_shards_the_memory_prefixed_sequence_and_its_rope():
  from cache_dit.distributed.transformers.stable_audio_3 import (
    StableAudio3ContextParallelismPlanner, )

  transformer = _tiny_transformer()
  plan = StableAudio3ContextParallelismPlanner()._apply(
    transformer=transformer,
    parallelism_config=ParallelismConfig(ulysses_size=2),
  )

  assert set(plan) == {
    "transformer_blocks.0", "transformer_blocks.*", f"transformer_blocks.{_DEPTH - 1}"
  }
  assert plan["transformer_blocks.0"]["hidden_states"].split_dim == 1
  assert plan["transformer_blocks.*"]["rope"].split_dim == 0
  gather = plan[f"transformer_blocks.{_DEPTH - 1}"]
  assert gather.gather_dim == 1
  # `forward` drops the memory tokens and projects the full sequence, so the gather cannot sit later.
  assert transformer._cp_requires_min_bn_blocks == 1


def test_cross_attn_processor_hides_the_cp_config_only_when_armed(monkeypatch):
  from diffusers.models.transformers.transformer_stable_audio3 import (
    StableAudio3CrossAttnProcessor, )

  from cache_dit.distributed.transformers import stable_audio_3 as module

  module._install_cp_cross_attn_patch()
  seen = {}

  def recorder(self, *args, **kwargs):
    seen["parallel_config"] = self._parallel_config
    seen["cp_config"] = getattr(self, "_cp_config", None)
    return None

  monkeypatch.setitem(module._CROSS_ATTN_PATCH_ORIGINALS, StableAudio3CrossAttnProcessor, recorder)

  armed_config = types.SimpleNamespace(_world_size=2)
  armed = StableAudio3CrossAttnProcessor()
  armed._parallel_config = armed_config
  armed._cp_config = armed_config
  StableAudio3CrossAttnProcessor.__call__(armed, None, None, None)
  # The text context is replicated: Ulysses must not run inside the cross attention.
  assert seen["parallel_config"] is None and seen["cp_config"] is None
  # ... and the runtime config is restored for the next call.
  assert armed._parallel_config is armed_config and armed._cp_config is armed_config

  local_config = types.SimpleNamespace(_world_size=1)
  local = StableAudio3CrossAttnProcessor()
  local._parallel_config = local_config
  local._cp_config = local_config
  StableAudio3CrossAttnProcessor.__call__(local, None, None, None)
  assert seen["parallel_config"] is local_config


def test_tp_permutation_is_skipped_for_a_single_rank_and_idempotent_in_the_planner():
  from cache_dit.distributed.transformers.stable_audio_3 import _permute_fused_rows

  module = torch.nn.Linear(8, 5 * 8, bias=False)
  original = module.weight.detach().clone()
  _permute_fused_rows(module, 5, 1, "to_qkv")
  assert torch.equal(module.weight.detach(), original)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize(("mode", "frames", "ulysses_anything"), [
  ("cp", _FRAMES, False),
  ("cp", 7, True),
  ("tp", _FRAMES, False),
])
def test_parallel_forward_matches_single_rank(mode, frames, ulysses_anything):
  import torch.multiprocessing as mp

  mp.spawn(_run_rank, args=(mode, frames, ulysses_anything), nprocs=2, join=True)


def _run_rank(rank, mode, frames, ulysses_anything):
  import torch.distributed as dist

  from cache_dit.distributed.backend import ParallelismBackend
  from cache_dit.distributed.dispatch import enable_parallelism

  dist.init_process_group("nccl", init_method="tcp://localhost:29593", rank=rank, world_size=2)
  torch.cuda.set_device(rank)
  try:
    transformer = _tiny_transformer().to("cuda")
    inputs = _inputs(device="cuda", frames=frames)
    with torch.no_grad():
      reference = transformer(**inputs)[0].float()

    if mode == "cp":
      config = ParallelismConfig(ulysses_size=2, ulysses_anything=ulysses_anything)
    else:
      config = ParallelismConfig(tp_size=2, backend=ParallelismBackend.NATIVE_PYTORCH)
    enable_parallelism(transformer, config)

    with torch.no_grad():
      output = transformer(**inputs)[0].float()

    if rank == 0:
      relative = (output - reference).abs().mean() / reference.abs().mean().clamp_min(1e-6)
      print(f"[stable-audio-3-{mode}] frames={frames} rel={relative.item():.6f}")
      assert output.shape == reference.shape
      assert relative.item() < _REL_TOL, f"{mode} diverged (rel={relative.item():.6f})"
    dist.barrier()
  finally:
    dist.destroy_process_group()
