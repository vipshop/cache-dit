"""Ordering regression for `TileSafeP2PComm`, the transport behind `--parallel-vae`.

The class moves two kinds of message between the same pair of ranks: a five-element int64 shape and the
payload it describes. `nccl` gives one FIFO per peer and **ignores `tag`** (measured: an `isend` with
tag 1 followed by one with tag 0 arrives in send order whatever the receive tag says), so the only
thing that keeps the two channels apart is that every rank posts them in the same order.

A previous implementation sent the shape immediately but deferred the payload to `commit()`. Two
`recv_tensor` calls in a row -- what every tile loop does -- then posted two shape receives before
either payload receive, while the sender interleaved `shape, payload, shape, payload`. The second shape
receive consumed a payload: the receiver read float32 bits as a shape and `torch.empty` raised
"SymIntArrayRef expected to contain only contiguous integers". On a CPU-only `gloo` build the same
mistake surfaces as `Received data size doesn't match expected size` (480 vs 40 bytes).

The test runs on CPU with `gloo`, so it also covers the module on a machine without accelerators. The
two ranks are separate interpreters rather than `mp.spawn` children: `spawn` re-imports the parent's
`__main__`, which under a test runner is not this file.
"""

import socket
import subprocess
import sys

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.distributed.is_available(),
                                reason="needs torch.distributed")

_EXPECTED = [((1, 3, 2, 4, 5), 1.0), ((1, 3, 2, 4, 6), 2.0), ((1, 3, 2, 4, 7), 3.0)]


def _free_port() -> int:
  """Pick a rendezvous port that is free right now.

  :returns: A TCP port number.
  """

  with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    return sock.getsockname()[1]


def _comm(kind: str = "safe"):
  """Build the requested comm with the shape tensors pinned to CPU.

  The class derives the shape device from the backend name, which only contains "cpu" for a CPU
  backend it never sees (gloo maps to an accelerator). Forcing CPU keeps this ordering test runnable
  on a build without accelerators; the ordering rules under test are backend independent.

  :param kind: "safe" for `TileSafeP2PComm`, "batched" for `TileBatchedP2PComm`.
  :returns: The comm instance under test.
  """

  from cache_dit.distributed.autoencoders.utils import TileBatchedP2PComm, TileSafeP2PComm

  comm = TileSafeP2PComm() if kind == "safe" else TileBatchedP2PComm()
  comm._s_device = "cpu"
  comm.set_dims(5)
  return comm


def _worker(rank: int, port: int, kind: str) -> None:
  """Run one rank: rank 0 receives three messages, rank 1 sends them one tensor at a time.

  :param rank: This process's rank.
  :param port: Rendezvous port.
  :param kind: Comm flavour under test.
  """

  import torch.distributed as dist

  dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2)
  group = dist.group.WORLD
  comm = _comm(kind)
  try:
    if rank == 0:
      # Three receives in a row before the first commit, exactly like a tile loop.
      received = [comm.recv_tensor(1, group, dtype=torch.float32) for _ in range(3)]
      comm.sync()
      for index, tensor in enumerate(received):
        expected_shape, expected_value = _EXPECTED[index]
        shape = tuple(tensor.shape)
        assert shape == expected_shape, (f"message {index} arrived with shape {shape}, "
                                         f"expected {expected_shape}")
        value = float(tensor.flatten()[0])
        assert value == expected_value, (f"message {index} carried payload {value}, "
                                         f"expected {expected_value}")
    else:
      relay = torch.full((1, 3, 2, 4, 5), 1.0)
      tile0 = torch.full((1, 3, 2, 4, 6), 2.0)
      tile1 = torch.full((1, 3, 2, 4, 7), 3.0)
      comm.send_tensor(relay, 0, group)
      comm.sync()
      comm.send_tensor(tile0, 0, group)
      comm.send_tensor(tile1, 0, group)
      comm.sync()
    dist.barrier()
  finally:
    dist.destroy_process_group()


@pytest.mark.parametrize("kind", ["safe"])
def test_shapes_and_payloads_stay_aligned_when_receives_are_posted_back_to_back(kind: str):
  """Three receives before the first commit must still see every shape and payload in order.

  This is the interleaving that used to hand a payload buffer to a shape receive.
  """

  port = _free_port()
  workers = [
    subprocess.Popen(
      [sys.executable, __file__, "--rank",
       str(rank), "--port", str(port)],
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True) for rank in (0, 1)
  ]
  outputs = []
  for worker in workers:
    try:
      outputs.append(worker.communicate(timeout=180)[0])
    except subprocess.TimeoutExpired:
      worker.kill()
      outputs.append(worker.communicate()[0] + "\n<timed out: the two ranks deadlocked>")
  for rank, (worker, output) in enumerate(zip(workers, outputs)):
    assert worker.returncode == 0, f"rank {rank} exited with {worker.returncode}:\n{output}"


@pytest.mark.xfail(strict=False,
                   reason="the batched transport is expected to mis-order this interleaving")
@pytest.mark.parametrize("kind", ["batched"])
def test_the_batched_transport_is_known_to_mis_order_this_interleaving(kind: str):
  """Documents why `TileSafeP2PComm` exists: the batched transport fails the same scenario."""

  port = _free_port()
  workers = [
    subprocess.Popen(
      [sys.executable, __file__, "--rank",
       str(rank), "--port",
       str(port), "--comm", kind],
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True) for rank in (0, 1)
  ]
  for worker in workers:
    try:
      worker.communicate(timeout=180)
    except subprocess.TimeoutExpired:
      worker.kill()
      worker.communicate()
  failures = [worker.returncode for worker in workers if worker.returncode != 0]
  assert not failures, "the batched transport unexpectedly handled the interleaving"


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--rank", type=int, required=True)
  parser.add_argument("--port", type=int, required=True)
  parser.add_argument("--comm", default="safe", choices=["safe", "batched"])
  arguments = parser.parse_args()
  _worker(arguments.rank, arguments.port, arguments.comm)
