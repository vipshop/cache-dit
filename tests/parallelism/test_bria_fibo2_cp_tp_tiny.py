"""Multi-rank parity test for the Bria Fibo 2.0 CP and TP plumbing (tiny model, no weights).

Fibo 2.0 keeps `freqs_cis` and the per-token `noise_mask` as loop-invariant locals of a plain
`for block in self.layers` loop, so the CP plan has to split them at *every* block, and it gathers
after every block as well: `final_layer` slices the packed sequence with a global `image_start`, and
a cached step never reaches the last block. The TP plan shards the attention heads and the SwiGLU
FFN while leaving `adaLN_modulation` replicated (its output is consumed as `chunk(4)`).

Both paths are compared against a cp-free / tp-free reference on rank 0. A second phase runs a
skip-enabled sequence (a different prompt every step) so the cached path is exercised under the
split/gather as well, not just the warmup-only parity run.

Usage:
  CUDA_VISIBLE_DEVICES=4,5 python -m pytest tests/parallelism/test_bria_fibo2_cp_tp_tiny.py -q
"""

import os

import pytest
import torch
import torch.multiprocessing as mp

try:
  from diffusers.models.transformers.transformer_bria_fibo2 import (  # noqa: F401
    BriaFibo2Transformer2DModel, )
except ImportError:
  pytest.skip("diffusers without Bria Fibo 2.0 support", allow_module_level=True)

_TEXT_LEN = 6
_GIST = 8
_GRID = 8
_IN_CHANNELS = 8
_CAP_FEAT_DIM = 48
# Several bundles, including layer 0 (the block `Fn_compute_blocks=1` re-runs on *every* step) and a
# skippable layer, so the bundle routing is exercised on both sides of a cache hit.
_INJECTION_LAYER_IDS = (0, 2)
_STEPS = 4
_REL_TOL = 2e-2  # bf16 all-reduce order noise on a 3-layer random model
_PORT = 29571


def _build(device):
  torch.manual_seed(0)
  transformer = BriaFibo2Transformer2DModel(
    in_channels=_IN_CHANNELS,
    patch_size=2,
    dim=32,
    n_layers=3,
    n_refiner_layers=1,
    n_heads=2,
    cap_feat_dim=_CAP_FEAT_DIM,
    injection_layer_ids=_INJECTION_LAYER_IDS,
    perceiver_num_layers=1,
    min_num_gist_tokens=_GIST,
    max_num_gist_tokens=_GIST,
    gist_step=_GIST,
    gist_min_text_len=2,
    gist_max_text_len=4,
    # `sum(axes_dims)` must equal `dim // n_heads` (16 here).
    axes_dims=(4, 6, 6),
    axes_lens=(64, 32, 32),
  )
  return transformer.to(device=device, dtype=torch.bfloat16).eval()


def _inputs(seed: int = 3):
  generator = torch.Generator().manual_seed(seed)
  return {
    "hidden_states":
    torch.randn(1, _IN_CHANNELS, 2 * _GRID, 2 * _GRID, generator=generator).to(torch.bfloat16),
    "timestep":
    torch.tensor([0.5]),
    "encoder_hidden_states":
    torch.randn(1, 1 + len(_INJECTION_LAYER_IDS), _TEXT_LEN, _CAP_FEAT_DIM,
                generator=generator).to(torch.bfloat16),
  }


def _bundle(inputs, layer_id: int):
  """Text bundle `layer_id` must read: `encoder_hidden_states[:, k + 1]`."""
  return inputs["encoder_hidden_states"][:, 1 + _INJECTION_LAYER_IDS.index(layer_id)]


def _to_device(inputs, device):
  return {
    key: (value.to(device) if torch.is_tensor(value) else value)
    for key, value in inputs.items()
  }


def _run_rank(rank: int, world: int, mode: str) -> None:
  os.environ["MASTER_ADDR"] = "127.0.0.1"
  os.environ["MASTER_PORT"] = str(_PORT)
  torch.cuda.set_device(rank)
  torch.distributed.init_process_group("nccl", rank=rank, world_size=world)
  device = f"cuda:{rank}"

  try:
    import cache_dit
    from cache_dit import DBCacheConfig
    from cache_dit.caching.block_adapters import BlockAdapterRegister
    from cache_dit.distributed.config import ParallelismConfig

    inputs = _inputs()
    transformer = _build(device)

    reference = None
    if rank == 0:
      with torch.no_grad():
        reference = [
          t.float().cpu() for t in transformer(**_to_device(inputs, device), return_dict=False)
        ]
    torch.distributed.barrier()

    if mode == "cp":
      parallelism_config = ParallelismConfig(ulysses_size=world)
    else:
      parallelism_config = ParallelismConfig(tp_size=world)

    # Cache is disabled (warmup covers the whole run): this test isolates the parallel paths.
    cache_dit.enable_cache(
      BlockAdapterRegister.get_adapter(transformer),
      cache_config=DBCacheConfig(
        Fn_compute_blocks=1,
        Bn_compute_blocks=0,
        max_warmup_steps=100,
        max_cached_steps=0,
        num_inference_steps=8,
        enable_separate_cfg=True,
        residual_diff_threshold=0.0,
      ),
      parallelism_config=parallelism_config,
    )

    with torch.no_grad():
      out = [t.float().cpu() for t in transformer(**_to_device(inputs, device), return_dict=False)]

    if rank == 0:
      got, ref = out[0], reference[0]
      relative = (got - ref).abs().mean() / ref.abs().mean().clamp_min(1e-6)
      print(f"[bria-fibo2-{mode}] rel={relative.item():.6f} shape={tuple(got.shape)}")
      assert got.shape == ref.shape, f"{mode} shape changed: {got.shape} vs {ref.shape}"
      assert relative.item() < _REL_TOL, f"{mode} diverged (rel={relative.item():.6f})"
    torch.distributed.barrier()

    # Phase 2: a skip-enabled sequence, since the run above stays in warmup. `max_warmup_steps=1`
    # plus a huge threshold caches the middle steps, and every step carries a different prompt, so
    # the per-rank stash can be checked against the prompt of the step it ran in.
    steps = [_to_device(_inputs(seed=200 + step), device) for step in range(_STEPS)]
    cached = _build(device)
    cache_dit.enable_cache(
      BlockAdapterRegister.get_adapter(cached),
      cache_config=DBCacheConfig(
        Fn_compute_blocks=1,
        Bn_compute_blocks=0,
        max_warmup_steps=1,
        max_cached_steps=-1,
        num_inference_steps=_STEPS,
        enable_separate_cfg=False,
        residual_diff_threshold=10.0,
      ),
      parallelism_config=parallelism_config,
    )

    executed = {"count": 0}
    handle = cached.layers[_INJECTION_LAYER_IDS[-1]].register_forward_hook(
      lambda module, args, output: executed.__setitem__("count", executed["count"] + 1))
    cached_outputs = []
    try:
      for step_id, step_inputs in enumerate(steps):
        with torch.no_grad():
          cached_outputs.append(cached(**step_inputs, return_dict=False)[0])
        for layer_id in _INJECTION_LAYER_IDS:
          bundle = cached.layers[layer_id]._text_bundle
          expected = _bundle(step_inputs, layer_id)
          assert torch.equal(
            bundle, expected), (f"{mode} step {step_id}: layer {layer_id} kept a stale text bundle")
    finally:
      handle.remove()

    assert executed["count"] < _STEPS, f"{mode}: no step was served from cache"
    assert torch.isfinite(cached_outputs[-1]).all(), f"{mode}: cached run is not finite"

    gathered = [torch.empty_like(cached_outputs[-1]) for _ in range(world)]
    torch.distributed.all_gather(gathered, cached_outputs[-1])
    if rank == 0:
      for other in gathered[1:]:
        relative = (other - gathered[0]).abs().mean() / gathered[0].abs().mean().clamp_min(1e-6)
        print(f"[bria-fibo2-{mode}] cached-step rank delta={relative.item():.6f}")
        assert relative.item() < _REL_TOL, f"{mode} cached step diverged across ranks"
  finally:
    torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
@pytest.mark.parametrize("mode", ["cp", "tp"])
def test_bria_fibo2_parallel_matches_single_rank(mode):
  mp.spawn(_run_rank, args=(2, mode), nprocs=2, join=True)
