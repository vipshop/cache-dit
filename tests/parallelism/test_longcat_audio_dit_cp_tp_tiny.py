"""CP/TP tests for the LongCat-AudioDiT planners (tiny model, no weights).

The plan splits the two embedder *outputs* instead of the transformer entry (the transformer returns
a dataclass) and gathers at `proj_out`; the rotary table is rebuilt full-length and sliced per rank
inside the attention processors, with cross attention kept local because its K/V are replicated.
These properties are asserted on CPU, and a two-rank parity run compares a single forward against
the un-parallelized reference on a seeded tiny model.

Usage:
  CUDA_VISIBLE_DEVICES=4,5 python -m pytest tests/parallelism/test_longcat_audio_dit_cp_tp_tiny.py -q
"""

import types

import pytest
import torch

from cache_dit.distributed.config import ParallelismConfig

diffusers = pytest.importorskip("diffusers")

if not hasattr(diffusers, "LongCatAudioDiTTransformer"):
  pytest.skip("LongCat-AudioDiT requires a newer diffusers", allow_module_level=True)

_DIM = 32
_DEPTH = 2
_HEADS = 2
_TEXT_DIM = 16
_LATENT_DIM = 4
_TEXT_LEN = 5
_FRAMES = 16
_REL_TOL = 2e-2


def _tiny_transformer():
  from diffusers import LongCatAudioDiTTransformer

  torch.manual_seed(0)
  transformer = LongCatAudioDiTTransformer(
    dit_dim=_DIM,
    dit_depth=_DEPTH,
    dit_heads=_HEADS,
    dit_text_dim=_TEXT_DIM,
    latent_dim=_LATENT_DIM,
    ff_mult=2.0,
    qk_norm=True,
  ).eval()
  with torch.no_grad():
    transformer.proj_out.weight.normal_()
    transformer.proj_out.bias.normal_()
    transformer.norm_out.linear.weight.normal_()
    transformer.adaln_global_mlp.mlp[-1].weight.normal_()
  return transformer


def _inputs(device="cpu", frames=_FRAMES):
  generator = torch.Generator(device=device).manual_seed(0)
  return dict(
    hidden_states=torch.randn(1, frames, _LATENT_DIM, generator=generator, device=device),
    encoder_hidden_states=torch.randn(1, _TEXT_LEN, _TEXT_DIM, generator=generator, device=device),
    encoder_attention_mask=torch.ones(1, _TEXT_LEN, dtype=torch.bool, device=device),
    timestep=torch.tensor([0.5], device=device),
    attention_mask=torch.ones(1, frames, dtype=torch.bool, device=device),
    latent_cond=torch.zeros(1, frames, _LATENT_DIM, device=device),
  )


def test_cp_plan_splits_both_embedders_and_gathers_proj_out():
  from cache_dit.distributed.transformers.longcat_audio_dit import (
    LongCatAudioDiTContextParallelismPlanner, )

  transformer = _tiny_transformer()
  plan = LongCatAudioDiTContextParallelismPlanner()._apply(
    transformer=transformer,
    parallelism_config=ParallelismConfig(ulysses_size=2),
  )

  assert set(plan) == {"input_embed", "latent_embed", "proj_out"}
  for name in ("input_embed", "latent_embed"):
    (index, spec), = plan[name].items()
    assert index == 0
    assert spec.split_output is True
    assert spec.split_dim == 1
  assert plan["proj_out"].gather_dim == 1
  # The plan must not touch the transformer entry: `forward` returns a dataclass, which the
  # split/gather hooks cannot process.
  assert "" not in plan


def test_cp_plan_rejects_a_mixed_ring_and_ulysses_mesh():
  from cache_dit.distributed.transformers.longcat_audio_dit import (
    LongCatAudioDiTContextParallelismPlanner, )

  with pytest.raises(NotImplementedError):
    LongCatAudioDiTContextParallelismPlanner()._apply(
      transformer=_tiny_transformer(),
      parallelism_config=ParallelismConfig(ring_size=2, ulysses_size=2),
    )


def test_cp_rope_install_is_idempotent():
  from cache_dit.distributed.transformers.longcat_audio_dit import (
    _CPRotarySliceProcessor,
    _FullRotaryTable,
    _install_cp_rope,
  )

  transformer = _tiny_transformer()
  _install_cp_rope(transformer)
  _install_cp_rope(transformer)

  assert isinstance(transformer.rotary_embed, _FullRotaryTable)
  for block in transformer.blocks:
    assert isinstance(block.self_attn.processor, _CPRotarySliceProcessor)
    assert not isinstance(block.self_attn.processor._processor, _CPRotarySliceProcessor)


def test_rank_offset_uses_the_global_split(monkeypatch):
  import torch.distributed as dist

  from cache_dit.distributed.transformers import longcat_audio_dit as module

  sizes_seen = {}

  def fake_all_gather_object(out, value, group=None):
    sizes = sizes_seen["sizes"]
    out[:] = list(sizes)

  monkeypatch.setattr(dist, "all_gather_object", fake_all_gather_object)
  mesh = types.SimpleNamespace(get_group=lambda: None)
  config = types.SimpleNamespace(
    ulysses_degree=2,
    _ulysses_local_rank=1,
    _ulysses_mesh=mesh,
    ulysses_anything=True,
    ring_degree=1,
    _ring_local_rank=0,
    _ring_mesh=None,
  )

  module._OFFSET_CACHE.clear()
  # 17 frames split as [9, 8] and 16 frames as [8, 8]: rank 1 holds 8 frames either way, but its
  # offset differs, so the cache key has to include the whole split.
  sizes_seen["sizes"] = [9, 8]
  assert module._cp_rank_offset(config, 8) == 9
  sizes_seen["sizes"] = [8, 8]
  assert module._cp_rank_offset(config, 8) == 8


def test_distributed_rmsnorm_shards_the_weight(monkeypatch):
  import torch.distributed as dist
  from diffusers.models.normalization import RMSNorm

  from cache_dit.distributed.transformers.longcat_audio_dit import _DistributedRMSNorm

  monkeypatch.setattr(dist, "all_reduce", lambda *args, **kwargs: None)
  norm = RMSNorm(8, eps=1e-6)
  with torch.no_grad():
    norm.weight.copy_(torch.arange(1.0, 9.0))

  shards = []
  for rank in range(2):
    mesh = types.SimpleNamespace(
      get_group=lambda rank=rank: types.SimpleNamespace(size=lambda: 2, rank=lambda: rank))
    distributed = _DistributedRMSNorm.from_rmsnorm(mesh, norm)
    assert distributed.weight.shape[0] == 4
    shards.append(distributed.weight.detach())
    out = distributed(torch.randn(2, 4))
    assert out.shape == (2, 4)
    assert torch.isfinite(out).all()
  assert torch.equal(torch.cat(shards), norm.weight)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize(("mode", "frames", "ulysses_anything"), [
  ("cp", _FRAMES, False),
  ("cp", 17, True),
  ("tp", _FRAMES, False),
])
def test_parallel_forward_matches_single_rank(mode, frames, ulysses_anything):
  import torch.multiprocessing as mp

  mp.spawn(_run_rank, args=(mode, frames, ulysses_anything), nprocs=2, join=True)


def _run_rank(rank, mode, frames, ulysses_anything):
  import torch.distributed as dist

  from cache_dit.distributed.backend import ParallelismBackend
  from cache_dit.distributed.dispatch import enable_parallelism

  dist.init_process_group("nccl", init_method="tcp://localhost:29583", rank=rank, world_size=2)
  torch.cuda.set_device(rank)
  try:
    transformer = _tiny_transformer().to("cuda")
    inputs = _inputs(device="cuda", frames=frames)
    with torch.no_grad():
      reference = transformer(**inputs).sample.float()

    if mode == "cp":
      config = ParallelismConfig(ulysses_size=2, ulysses_anything=ulysses_anything)
    else:
      config = ParallelismConfig(tp_size=2, backend=ParallelismBackend.NATIVE_PYTORCH)
    enable_parallelism(transformer, config)

    with torch.no_grad():
      output = transformer(**inputs).sample.float()

    if rank == 0:
      relative = (output - reference).abs().mean() / reference.abs().mean().clamp_min(1e-6)
      print(f"[longcat-audiodit-{mode}] frames={frames} rel={relative.item():.6f}")
      assert output.shape == reference.shape
      assert relative.item() < _REL_TOL, f"{mode} diverged (rel={relative.item():.6f})"
    dist.barrier()
  finally:
    dist.destroy_process_group()
