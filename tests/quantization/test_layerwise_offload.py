from __future__ import annotations

import pytest
import torch

import cache_dit.offload.layerwise as layerwise_module
from cache_dit.offload import get_layerwise_offload_handles
from cache_dit.offload import layerwise_offload
from cache_dit.offload import layerwise_cpu_offload
from cache_dit.offload import remove_layerwise_offload
from cache_dit.offload.layerwise import LayerwiseOffloadHandle
from tests.quantization._svdq_test_utils import make_token_batch
from tests.quantization._svdq_test_utils import make_toy_model

pytestmark = pytest.mark.skipif(
  not torch.cuda.is_available(),
  reason="Layerwise offload tests require CUDA.",
)


def test_layerwise_offload_moves_target_module_to_cuda_and_restores_cpu() -> None:
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=901,
    device="cpu",
    dtype=torch.float32,
  )
  inputs = make_token_batch(
    batch_size=2,
    seq_len=8,
    width=128,
    seed=902,
    device="cpu",
    dtype=torch.float32,
  )

  observed_devices: list[str] = []
  offload_handle = layerwise_offload(
    model,
    module_names=["block.to_q"],
    onload_device="cuda",
  )
  capture_handle = model.block.to_q.register_forward_pre_hook(
    lambda _module, args: observed_devices.append(args[0].device.type))

  try:
    with torch.inference_mode():
      output = model(inputs)
      torch.cuda.synchronize()
  finally:
    capture_handle.remove()
    offload_handle.remove()

  assert torch.isfinite(output).all()
  assert observed_devices == ["cuda"]
  assert output.device.type == "cpu"
  assert model.block.to_q.weight.device.type == "cpu"


def test_layerwise_cpu_offload_preserves_cuda_io_for_full_model() -> None:
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=903,
    device="cpu",
    dtype=torch.float32,
  )
  inputs = make_token_batch(
    batch_size=2,
    seq_len=8,
    width=128,
    seed=904,
    device="cuda",
    dtype=torch.float32,
  )

  offload_handle = layerwise_cpu_offload(model, onload_device="cuda")
  try:
    with torch.inference_mode():
      output = model(inputs)
      torch.cuda.synchronize()
  finally:
    offload_handle.remove()

  assert "block.norm" in offload_handle.module_names
  assert "block.to_q" in offload_handle.module_names
  assert torch.isfinite(output).all()
  assert output.device.type == "cuda"
  assert all(parameter.device.type == "cpu" for parameter in model.parameters())


def test_layerwise_cpu_offload_attaches_handle_to_root_module() -> None:
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=905,
    device="cpu",
    dtype=torch.float32,
  )

  offload_handle = layerwise_cpu_offload(model, onload_device="cuda")

  assert get_layerwise_offload_handles(model) == (offload_handle, )

  removed_count = remove_layerwise_offload(model)

  assert removed_count == 1
  assert get_layerwise_offload_handles(model) == ()
  assert all(parameter.device.type == "cpu" for parameter in model.parameters())


def test_layerwise_cpu_offload_async_transfer_prefetches_next_module() -> None:
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=906,
    device="cpu",
    dtype=torch.float32,
  )
  inputs = make_token_batch(
    batch_size=2,
    seq_len=8,
    width=128,
    seed=907,
    device="cpu",
    dtype=torch.float32,
  )

  observed_next_module_devices: list[str] = []
  offload_handle = layerwise_cpu_offload(
    model,
    module_names=["block.to_q", "block.to_k", "block.to_v", "block.to_out"],
    onload_device="cuda",
    async_transfer=True,
  )
  capture_handle = model.block.to_q.register_forward_pre_hook(
    lambda _module, args: observed_next_module_devices.append(model.block.to_k.weight.device.type))

  try:
    with torch.inference_mode():
      output = model(inputs)
  finally:
    capture_handle.remove()
    offload_handle.remove()

  assert offload_handle.async_transfer is True
  assert observed_next_module_devices == ["cuda"]
  assert torch.isfinite(output).all()
  assert output.device.type == "cpu"
  assert all(parameter.device.type == "cpu" for parameter in model.parameters())


def test_layerwise_cpu_offload_async_transfer_respects_transfer_buckets() -> None:
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=910,
    device="cpu",
    dtype=torch.float32,
  )
  inputs = make_token_batch(
    batch_size=2,
    seq_len=8,
    width=128,
    seed=911,
    device="cpu",
    dtype=torch.float32,
  )

  observed_prefetch_devices: list[tuple[str, str]] = []
  offload_handle = layerwise_cpu_offload(
    model,
    module_names=["block.to_q", "block.to_k", "block.to_v", "block.to_out"],
    onload_device="cuda",
    async_transfer=True,
    transfer_buckets=2,
  )
  capture_handle = model.block.to_q.register_forward_pre_hook(
    lambda _module, _args: observed_prefetch_devices.append(
      (model.block.to_k.weight.device.type, model.block.to_v.weight.device.type)))

  try:
    with torch.inference_mode():
      output = model(inputs)
  finally:
    capture_handle.remove()
    offload_handle.remove()

  assert offload_handle.transfer_buckets == 2
  assert observed_prefetch_devices == [("cuda", "cuda")]
  assert torch.isfinite(output).all()
  assert output.device.type == "cpu"
  assert all(parameter.device.type == "cpu" for parameter in model.parameters())


def test_layerwise_cpu_offload_async_transfer_assigns_distinct_streams_per_bucket(
  monkeypatch, ) -> None:
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=914,
    device="cpu",
    dtype=torch.float32,
  )
  inputs = make_token_batch(
    batch_size=2,
    seq_len=8,
    width=128,
    seed=915,
    device="cpu",
    dtype=torch.float32,
  )

  scheduled_onload_streams: list[tuple[str, int]] = []
  seen_targets: set[str] = set()
  original_schedule_target_onload = LayerwiseOffloadHandle._schedule_target_onload

  def _capture_schedule_target_onload(self, target, *, allow_wait):
    original_schedule_target_onload(self, target, allow_wait=allow_wait)
    stream_index = target.pending_onload_stream_index
    if stream_index is None or target.name in seen_targets:
      return
    seen_targets.add(target.name)
    scheduled_onload_streams.append((target.name, stream_index))

  monkeypatch.setattr(
    LayerwiseOffloadHandle,
    "_schedule_target_onload",
    _capture_schedule_target_onload,
  )

  offload_handle = layerwise_cpu_offload(
    model,
    module_names=["block.to_q", "block.to_k", "block.to_v", "block.to_out"],
    onload_device="cuda",
    async_transfer=True,
    transfer_buckets=2,
  )

  try:
    with torch.inference_mode():
      output = model(inputs)
  finally:
    offload_handle.remove()

  assert torch.isfinite(output).all()
  assert len(offload_handle._copy_streams) == 2
  assert [name for name, _stream_index in scheduled_onload_streams[:3]] == [
    "block.to_q",
    "block.to_k",
    "block.to_v",
  ]
  assert scheduled_onload_streams[1][1] != scheduled_onload_streams[2][1]


def test_layerwise_cpu_offload_async_transfer_emits_distinct_stream_debug_logs(
  monkeypatch, ) -> None:
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=916,
    device="cpu",
    dtype=torch.float32,
  )
  inputs = make_token_batch(
    batch_size=2,
    seq_len=8,
    width=128,
    seed=917,
    device="cpu",
    dtype=torch.float32,
  )

  debug_messages: list[str] = []

  def _capture_debug(message: str, *args) -> None:
    debug_messages.append(message % args if args else message)

  monkeypatch.setattr(layerwise_module.logger, "debug", _capture_debug)

  offload_handle = layerwise_cpu_offload(
    model,
    module_names=["block.to_q", "block.to_k", "block.to_v", "block.to_out"],
    onload_device="cuda",
    async_transfer=True,
    transfer_buckets=2,
  )

  try:
    with torch.inference_mode():
      output = model(inputs)
  finally:
    offload_handle.remove()

  assert torch.isfinite(output).all()
  assert any("copy stream[0]" in message for message in debug_messages)
  assert any("copy stream[1]" in message for message in debug_messages)


def test_layerwise_cpu_offload_async_transfer_caps_global_onload_budget() -> None:
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=912,
    device="cpu",
    dtype=torch.float32,
  )
  inputs = make_token_batch(
    batch_size=2,
    seq_len=8,
    width=128,
    seed=913,
    device="cpu",
    dtype=torch.float32,
  )

  observed_pending_onload_sizes: list[int] = []
  offload_handle = layerwise_cpu_offload(
    model,
    module_names=["block.to_q", "block.to_k", "block.to_v", "block.to_out"],
    onload_device="cuda",
    async_transfer=True,
    transfer_buckets=2,
  )
  capture_handles = [
    module.register_forward_pre_hook(
      lambda _module, _args, handle=offload_handle: observed_pending_onload_sizes.append(
        len(handle._pending_onload_targets)))
    for module in [model.block.to_q, model.block.to_k, model.block.to_v, model.block.to_out]
  ]

  try:
    with torch.inference_mode():
      output = model(inputs)
  finally:
    for capture_handle in capture_handles:
      capture_handle.remove()
    offload_handle.remove()

  assert torch.isfinite(output).all()
  assert observed_pending_onload_sizes
  assert max(observed_pending_onload_sizes) <= 2


def test_layerwise_cpu_offload_async_transfer_drains_pending_remove() -> None:
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=908,
    device="cpu",
    dtype=torch.float32,
  )
  inputs = make_token_batch(
    batch_size=2,
    seq_len=8,
    width=128,
    seed=909,
    device="cuda",
    dtype=torch.float32,
  )

  offload_handle = layerwise_cpu_offload(
    model,
    onload_device="cuda",
    async_transfer=True,
  )
  try:
    with torch.inference_mode():
      output = model(inputs)
  finally:
    offload_handle.remove()

  assert torch.isfinite(output).all()
  assert output.device.type == "cuda"
  assert all(parameter.device.type == "cpu" for parameter in model.parameters())
