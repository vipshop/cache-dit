import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


def _run_distributed_indicator(mode: str):
  """Worker entry for CP/TP/hybrid SEA equivalence against a single full tensor."""

  import torch.distributed as dist
  from types import SimpleNamespace

  from cache_dit.caching.cache_contexts import CachedContextManager, DBCacheConfig
  from cache_dit.caching.cache_contexts.seacache_uitls import apply_sea_filter

  dist.init_process_group("gloo")
  rank = dist.get_rank()
  world = dist.get_world_size()
  uneven = mode == "cp_uneven"
  if mode in ("cp_even", "cp_uneven"):
    cp_size, tp_size = world, 1
  elif mode == "tp":
    cp_size, tp_size = 1, world
  elif mode == "cp_tp":
    cp_size, tp_size = 2, 2
  else:
    raise ValueError(mode)

  mesh = dist.device_mesh.init_device_mesh("cpu", (cp_size, tp_size), mesh_dim_names=("cp", "tp"))
  cp_mesh = mesh["cp"]
  parallel_config = (SimpleNamespace(_flattened_mesh=cp_mesh, ulysses_anything=True)
                     if uneven else SimpleNamespace(_flat_cp_mesh=cp_mesh, ulysses_anything=False))
  cp_rank = cp_mesh.get_local_rank() if cp_size > 1 else 0
  tp_rank = mesh["tp"].get_local_rank() if tp_size > 1 else 0
  feature_shape = (1, 5) if uneven else (2, 3)
  tokens = 5 if uneven else 6
  channels = 4

  generator = torch.Generator().manual_seed(123)
  previous = torch.randn(1, tokens, channels, generator=generator)
  current = torch.randn(1, tokens, channels, generator=generator)
  previous_indices = torch.arange(tokens).tensor_split(cp_size)[cp_rank]
  current_indices = torch.arange(tokens).tensor_split(cp_size)[cp_rank]
  channel_indices = torch.arange(channels).tensor_split(tp_size)[tp_rank]
  previous_local = previous[:, previous_indices][:, :, channel_indices].contiguous()
  current_local = current[:, current_indices][:, :, channel_indices].contiguous()

  config = DBCacheConfig(
    max_warmup_steps=0,
    enable_seacache=True,
    seacache_feature_shape=feature_shape,
    seacache_ab_schedule=((0.9, 0.1), (0.8, 0.2)),
  )
  manager = CachedContextManager(persistent_context=True)
  manager.set_context(manager.new_context(name="default", cache_config=config))
  manager.configure_seacache(parallelism_config=parallel_config)
  manager.mark_step_begin()
  manager.set_Fn_buffer(previous_local)
  manager.mark_step_begin()
  decision = manager.can_cache(current_local, parallelized=True, threshold=0.05)

  filtered_previous = apply_sea_filter(previous, 0.9, 0.1, feature_shape=feature_shape)
  filtered_current = apply_sea_filter(current, 0.8, 0.2, feature_shape=feature_shape)
  expected_diff = ((filtered_previous - filtered_current).abs().mean() /
                   filtered_previous.abs().mean()).item()
  expected_decision = expected_diff < 0.05
  actual_diff = manager.current_context.residual_diffs["1"]
  assert actual_diff == pytest.approx(expected_diff, rel=2e-5, abs=2e-7)
  assert decision == expected_decision
  assert manager.get_seacache_Fn_buffer().shape[1] == tokens
  assert manager.get_Fn_buffer().shape == previous_local.shape

  decisions = [torch.zeros(1, dtype=torch.int64) for _ in range(world)]
  dist.all_gather(decisions, torch.tensor([int(decision)]))
  assert all(int(value) == int(decision) for value in decisions)

  if mode == "cp_even":
    cfg_config = DBCacheConfig(
      max_warmup_steps=0,
      enable_seacache=True,
      enable_separate_cfg=True,
      cfg_diff_compute_separate=False,
      seacache_feature_shape=feature_shape,
      seacache_ab_schedule=((0.9, 0.1), (0.8, 0.2)),
    )
    cfg_manager = CachedContextManager(persistent_context=True)
    cfg_manager.set_context(cfg_manager.new_context(name="cfg", cache_config=cfg_config))
    cfg_manager.configure_seacache(parallelism_config=parallel_config)
    cfg_manager.mark_step_begin()
    cfg_manager.set_Fn_buffer(previous_local)
    cfg_manager.mark_step_begin()
    cfg_manager.set_Fn_buffer(previous_local)
    cfg_manager.mark_step_begin()
    non_cfg_decision = cfg_manager.can_cache(current_local, parallelized=True, threshold=0.05)
    if not non_cfg_decision:
      cfg_manager.set_Fn_buffer(current_local)
    cfg_manager.mark_step_begin()
    cfg_decision = cfg_manager.can_cache(current_local, parallelized=True, threshold=0.05)
    assert cfg_decision == non_cfg_decision
    assert cfg_manager.current_context.cfg_residual_diffs["1"] != -2.0

  if rank == 0:
    print(f"SeaCache {mode}: diff={actual_diff:.8f}, decision={decision}, world={world}")
  dist.destroy_process_group()


@pytest.mark.parametrize(
  "world,mode",
  [(2, "cp_even"), (2, "cp_uneven"), (2, "tp"), (4, "cp_tp")],
)
def test_seacache_distributed_matches_full_tensor(world, mode):
  """Torchrun/Gloo verifies global SEA distance and rank-consistent cache decisions."""

  repo = Path(__file__).resolve().parents[2]
  torchrun = Path(sys.executable).with_name("torchrun")
  env = os.environ.copy()
  env["PYTHONPATH"] = str(repo / "src")
  env["PYTHONDONTWRITEBYTECODE"] = "1"
  result = subprocess.run(
    [
      str(torchrun),
      "--standalone",
      f"--nproc_per_node={world}",
      str(Path(__file__).resolve()),
      "--worker",
      mode,
    ],
    cwd=repo,
    env=env,
    capture_output=True,
    text=True,
    timeout=180,
  )
  assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--worker", action="store_true")
  parser.add_argument("mode")
  args = parser.parse_args()
  if args.worker:
    _run_distributed_indicator(args.mode)
