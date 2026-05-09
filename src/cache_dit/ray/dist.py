from __future__ import annotations

import torch
import torch.distributed as dist


def init_worker_process_group(
  rank: int,
  world_size: int,
  master_port: int,
) -> None:
  """Initialize torch.distributed inside one Ray worker actor.

  :param rank: Global rank assigned by the Ray engine.
  :param world_size: Number of Ray workers participating in the model-parallel group.
  :param master_port: TCPStore port shared by all workers.
  """

  if dist.is_available() and dist.is_initialized():
    return

  backend = "cpu:gloo,cuda:nccl" if torch.cuda.is_available() else "gloo"
  store = dist.TCPStore(
    host_name="127.0.0.1",
    port=master_port,
    world_size=world_size,
    is_master=(rank == 0),
  )
  dist.init_process_group(
    backend=backend,
    store=store,
    rank=rank,
    world_size=world_size,
    device_id=None,
  )
  dist.barrier()


def destroy_worker_process_group() -> None:
  """Destroy the process group owned by a Ray worker actor."""

  if dist.is_available() and dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()
