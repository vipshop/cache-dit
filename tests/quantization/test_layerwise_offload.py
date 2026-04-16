from __future__ import annotations

import contextlib
import dataclasses
from collections import OrderedDict

import cache_dit
import pytest
import torch
from diffusers import DiffusionPipeline
from torch import nn

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


@dataclasses.dataclass
class _DictLikeModelOutput(OrderedDict):
  pooler_output: torch.Tensor | None = None
  last_hidden_state: torch.Tensor | None = None

  def __post_init__(self) -> None:
    if self.pooler_output is not None:
      self["pooler_output"] = self.pooler_output
    if self.last_hidden_state is not None:
      self["last_hidden_state"] = self.last_hidden_state


class _DictLikeOutputModel(nn.Module):

  def __init__(self) -> None:
    super().__init__()
    self.proj = nn.Linear(8, 8)
    self.norm = nn.LayerNorm(8)

  def forward(self, hidden_states: torch.Tensor) -> _DictLikeModelOutput:
    hidden_states = self.norm(self.proj(hidden_states))
    return _DictLikeModelOutput(
      pooler_output=hidden_states.mean(dim=1),
      last_hidden_state=hidden_states,
    )


class _ToyOffloadPipeline(DiffusionPipeline):

  def __init__(self) -> None:
    self.transformer = make_toy_model(
      embed_dim=128,
      num_heads=4,
      seed=970,
      device="cpu",
      dtype=torch.float32,
    )
    self.proj = nn.Linear(128, 128, device="cpu", dtype=torch.float32)

  def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return self.proj(self.transformer(hidden_states))


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


def test_layerwise_cpu_offload_accepts_diffusion_pipeline() -> None:
  pipe = _ToyOffloadPipeline()
  inputs = make_token_batch(
    batch_size=2,
    seq_len=8,
    width=128,
    seed=971,
    device="cpu",
    dtype=torch.float32,
  )

  offload_handle = layerwise_cpu_offload(pipe, onload_device="cuda")
  try:
    with torch.inference_mode():
      output = pipe(inputs)
      torch.cuda.synchronize()
  finally:
    offload_handle.remove()

  assert isinstance(offload_handle, layerwise_module.LayerwiseOffloadHandleGroup)
  assert len(offload_handle) == 2
  assert "proj" in offload_handle.module_names
  assert all(not module_name.endswith(".") for module_name in offload_handle.module_names)
  assert get_layerwise_offload_handles(pipe) == ()
  assert torch.isfinite(output).all()
  assert output.device.type == "cpu"
  assert all(parameter.device.type == "cpu" for parameter in pipe.transformer.parameters())
  assert all(parameter.device.type == "cpu" for parameter in pipe.proj.parameters())


def test_layerwise_cpu_offload_pipeline_module_names_select_root_modules() -> None:
  pipe = _ToyOffloadPipeline()

  offload_handle = layerwise_cpu_offload(
    pipe,
    module_names=["transformer"],
    onload_device="cuda",
  )
  try:
    assert isinstance(offload_handle, LayerwiseOffloadHandle)
    assert offload_handle.root_module is pipe.transformer
  finally:
    offload_handle.remove()


def test_top_level_cache_dit_exports_layerwise_offload_api() -> None:
  assert cache_dit.layerwise_offload is layerwise_offload
  assert cache_dit.layerwise_cpu_offload is layerwise_cpu_offload
  assert cache_dit.get_layerwise_offload_handles is get_layerwise_offload_handles
  assert cache_dit.remove_layerwise_offload is remove_layerwise_offload


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


def test_layerwise_cpu_offload_preserves_dict_like_model_output_type() -> None:
  model = _DictLikeOutputModel().to(device="cpu", dtype=torch.float32)
  inputs = torch.randn(2, 4, 8, device="cpu", dtype=torch.float32)

  offload_handle = layerwise_cpu_offload(model, onload_device="cuda")
  try:
    with torch.inference_mode():
      output = model(inputs)
      torch.cuda.synchronize()
  finally:
    offload_handle.remove()

  assert isinstance(output, _DictLikeModelOutput)
  assert output.pooler_output is not None
  assert output.last_hidden_state is not None
  assert output.pooler_output.device.type == "cpu"
  assert output.last_hidden_state.device.type == "cpu"
  assert output["pooler_output"].device.type == "cpu"
  assert output["last_hidden_state"].device.type == "cpu"


def test_layerwise_cpu_offload_sync_path_prepares_tensor_state_mirror() -> None:
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=931,
    device="cpu",
    dtype=torch.float32,
  )

  offload_handle = layerwise_cpu_offload(
    model,
    module_names=["block.to_q"],
    onload_device="cuda",
    async_transfer=False,
  )

  try:
    target = offload_handle.targets[0]
    assert target.tensor_states
    assert {tensor_state.kind for tensor_state in target.tensor_states} == {"parameter"}
    assert target.resident_device.type == "cpu"
  finally:
    offload_handle.remove()


def test_layerwise_cpu_offload_sync_path_eval_module_skips_parameter_sync_back(
  monkeypatch, ) -> None:
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=932,
    device="cpu",
    dtype=torch.float32,
  )
  model.eval()
  inputs = make_token_batch(
    batch_size=2,
    seq_len=8,
    width=128,
    seed=933,
    device="cpu",
    dtype=torch.float32,
  )

  offload_handle = layerwise_cpu_offload(
    model,
    module_names=["block.to_q"],
    onload_device="cuda",
    async_transfer=False,
  )
  target = offload_handle.targets[0]
  mirrored_cpu_tensor_ptrs = {
    tensor_state.cpu_tensor.data_ptr()
    for tensor_state in target.tensor_states
  }
  copied_mirror_ptrs: list[int] = []
  original_copy = torch.Tensor.copy_

  def _capture_copy(self, src, *args, **kwargs):
    if self.data_ptr() in mirrored_cpu_tensor_ptrs:
      copied_mirror_ptrs.append(self.data_ptr())
    return original_copy(self, src, *args, **kwargs)

  monkeypatch.setattr(torch.Tensor, "copy_", _capture_copy)

  try:
    with torch.inference_mode():
      output = model(inputs)
      torch.cuda.synchronize()
  finally:
    offload_handle.remove()

  assert torch.isfinite(output).all()
  assert copied_mirror_ptrs == []
  assert model.block.to_q.weight.device.type == "cpu"


def test_layerwise_cpu_offload_sync_path_eval_module_keeps_buffer_sync_back(monkeypatch, ) -> None:

  class _BufferedLeafModule(torch.nn.Module):

    def __init__(self) -> None:
      super().__init__()
      self.weight = torch.nn.Parameter(torch.ones(4, 4, dtype=torch.float32))
      self.register_buffer("scale", torch.ones(4, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
      return inputs @ self.weight * self.scale

  model = _BufferedLeafModule().eval()
  inputs = torch.ones(2, 4, device="cpu", dtype=torch.float32)

  offload_handle = layerwise_cpu_offload(
    model,
    module_names=[""],
    onload_device="cuda",
    async_transfer=False,
  )
  target = offload_handle.targets[0]
  mirror_ptr_to_name = {
    tensor_state.cpu_tensor.data_ptr(): tensor_state.name
    for tensor_state in target.tensor_states
  }
  copied_names: list[str] = []
  original_copy = torch.Tensor.copy_

  def _capture_copy(self, src, *args, **kwargs):
    tensor_name = mirror_ptr_to_name.get(self.data_ptr())
    if tensor_name is not None:
      copied_names.append(tensor_name)
    return original_copy(self, src, *args, **kwargs)

  monkeypatch.setattr(torch.Tensor, "copy_", _capture_copy)

  try:
    with torch.inference_mode():
      output = model(inputs)
      torch.cuda.synchronize()
  finally:
    offload_handle.remove()

  assert torch.isfinite(output).all()
  assert copied_names == ["scale"]
  assert model.weight.device.type == "cpu"
  assert model.scale.device.type == "cpu"


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
  assert offload_handle.transfer_buckets == 1
  assert len(offload_handle._onload_copy_streams) == 1
  assert len(offload_handle._offload_copy_streams) == 1
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
  assert offload_handle.effective_transfer_buckets is None
  assert observed_prefetch_devices == [("cuda", "cuda")]
  assert torch.isfinite(output).all()
  assert output.device.type == "cpu"
  assert all(parameter.device.type == "cpu" for parameter in model.parameters())


def test_layerwise_cpu_offload_async_transfer_prefetch_limit_caps_window() -> None:
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
    prefetch_limit=True,
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
  assert offload_handle.effective_transfer_buckets == 8
  assert max(observed_pending_onload_sizes) <= 3


def test_layerwise_cpu_offload_async_transfer_respects_max_inflight_prefetch_bytes() -> None:
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=940,
    device="cpu",
    dtype=torch.float32,
  )
  inputs = make_token_batch(
    batch_size=2,
    seq_len=8,
    width=128,
    seed=941,
    device="cpu",
    dtype=torch.float32,
  )

  probe_handle = layerwise_cpu_offload(
    model,
    module_names=["block.to_q", "block.to_k", "block.to_v", "block.to_out"],
    onload_device="cuda",
    async_transfer=True,
    transfer_buckets=4,
  )
  single_target_budget = probe_handle.targets[1].prefetch_residency_bytes
  probe_handle.remove()

  observed_pending_onload_sizes: list[int] = []
  offload_handle = layerwise_cpu_offload(
    model,
    module_names=["block.to_q", "block.to_k", "block.to_v", "block.to_out"],
    onload_device="cuda",
    async_transfer=True,
    transfer_buckets=4,
    max_inflight_prefetch_bytes=single_target_budget,
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
  assert max(observed_pending_onload_sizes) <= 1


def test_layerwise_cpu_offload_accepts_string_max_inflight_prefetch_bytes() -> None:
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=942,
    device="cpu",
    dtype=torch.float32,
  )

  offload_handle = layerwise_cpu_offload(
    model,
    module_names=["block.to_q", "block.to_k", "block.to_v", "block.to_out"],
    onload_device="cuda",
    async_transfer=True,
    transfer_buckets=4,
    max_inflight_prefetch_bytes="1KiB",
  )

  try:
    assert offload_handle.max_inflight_prefetch_bytes == 1024
    assert offload_handle.effective_max_inflight_prefetch_bytes == 1024
  finally:
    offload_handle.remove()


def test_layerwise_cpu_offload_rejects_invalid_string_max_inflight_prefetch_bytes() -> None:
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=943,
    device="cpu",
    dtype=torch.float32,
  )

  with pytest.raises(ValueError, match="Expected a positive byte value"):
    layerwise_cpu_offload(
      model,
      module_names=["block.to_q"],
      onload_device="cuda",
      async_transfer=True,
      max_inflight_prefetch_bytes="not-a-size",
    )


def test_layerwise_cpu_offload_async_transfer_onload_does_not_wait_current_stream(
  monkeypatch, ) -> None:
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=918,
    device="cpu",
    dtype=torch.float32,
  )

  offload_handle = layerwise_cpu_offload(
    model,
    module_names=["block.to_q"],
    onload_device="cuda",
    async_transfer=True,
  )
  target = offload_handle.targets[0]

  class _FakeCopyStream:

    def wait_stream(self, _stream) -> None:
      raise AssertionError("Async onload should not wait on the current compute stream.")

  class _FakeEvent:

    def record(self, _stream=None) -> None:
      return

    def synchronize(self) -> None:
      return

    def query(self) -> bool:
      return True

  monkeypatch.setattr(offload_handle, "_select_onload_copy_stream", lambda: (0, _FakeCopyStream()))
  monkeypatch.setattr(
    layerwise_module.torch.cuda,
    "current_stream",
    lambda *_args, **_kwargs:
    (_ for _ in
     ()).throw(AssertionError("Async onload should not query the current compute stream.")),
  )
  monkeypatch.setattr(
    layerwise_module.torch.cuda,
    "stream",
    lambda _stream: contextlib.nullcontext(),
  )
  monkeypatch.setattr(layerwise_module.torch.cuda, "Event", _FakeEvent)

  scheduled_event = None
  scheduled_stream_index = None
  onload_weight_device = None
  try:
    offload_handle._schedule_target_onload(target, allow_wait=True)
    scheduled_event = target.pending_onload_event
    scheduled_stream_index = target.pending_onload_stream_index
    onload_weight_device = model.block.to_q.weight.device.type
    torch.cuda.synchronize()
  finally:
    offload_handle.remove()

  assert scheduled_event is not None
  assert scheduled_stream_index == 0
  assert onload_weight_device == "cuda"
  assert model.block.to_q.weight.device.type == "cpu"


def test_layerwise_cpu_offload_eval_module_skips_parameter_sync_back() -> None:
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=922,
    device="cpu",
    dtype=torch.float32,
  )
  model.eval()

  offload_handle = layerwise_cpu_offload(
    model,
    module_names=["block.to_q"],
    onload_device="cuda",
    async_transfer=True,
  )
  target = offload_handle.targets[0]

  try:
    assert target.tensor_states
    assert {tensor_state.kind for tensor_state in target.tensor_states} == {"parameter"}
    assert all(not offload_handle._tensor_state_requires_sync_back(target, tensor_state)
               for tensor_state in target.tensor_states)
  finally:
    offload_handle.remove()

  def test_layerwise_cpu_offload_eval_module_skips_offload_copy_stream(monkeypatch, ) -> None:
    model = make_toy_model(
      embed_dim=128,
      num_heads=4,
      seed=923,
      device="cpu",
      dtype=torch.float32,
    )
    model.eval()

    offload_handle = layerwise_cpu_offload(
      model,
      module_names=["block.to_q"],
      onload_device="cuda",
      async_transfer=True,
    )
    target = offload_handle.targets[0]

    class _FakeCurrentStream:

      def __init__(self) -> None:
        self.recorded_events = 0

    fake_current_stream = _FakeCurrentStream()

    class _FakeEvent:

      def __init__(self) -> None:
        self.recorded_stream = None

      def record(self, stream=None) -> None:
        self.recorded_stream = stream
        if stream is fake_current_stream:
          fake_current_stream.recorded_events += 1

      def synchronize(self) -> None:
        return

      def query(self) -> bool:
        return True

    monkeypatch.setattr(
      offload_handle,
      "_select_offload_copy_stream",
      lambda: (_ for _ in ()).throw(
        AssertionError("Eval parameter-only offload should not allocate an offload copy stream.")),
    )
    monkeypatch.setattr(
      layerwise_module.torch.cuda,
      "current_stream",
      lambda *_args, **_kwargs: fake_current_stream,
    )
    monkeypatch.setattr(layerwise_module.torch.cuda, "Event", _FakeEvent)

    scheduled_event = None
    scheduled_pending_stream_index = None
    try:
      offload_handle._materialize_onload_sync(target)
      offload_handle._schedule_target_offload(target)
      scheduled_event = target.pending_offload_event
      scheduled_pending_stream_index = target.pending_offload_stream_index
    finally:
      offload_handle.remove()

    assert fake_current_stream.recorded_events == 1
    assert scheduled_event is not None
    assert scheduled_pending_stream_index is None


def test_layerwise_cpu_offload_eval_module_keeps_buffer_sync_back() -> None:

  class _BufferedLeafModule(torch.nn.Module):

    def __init__(self) -> None:
      super().__init__()
      self.weight = torch.nn.Parameter(torch.ones(4, 4, dtype=torch.float32))
      self.register_buffer("scale", torch.ones(4, dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
      return inputs @ self.weight * self.scale

  model = _BufferedLeafModule().eval()

  offload_handle = layerwise_cpu_offload(
    model,
    module_names=[""],
    onload_device="cuda",
    async_transfer=True,
  )
  target = offload_handle.targets[0]

  try:
    sync_back_by_name = {
      tensor_state.name: offload_handle._tensor_state_requires_sync_back(target, tensor_state)
      for tensor_state in target.tensor_states
    }
  finally:
    offload_handle.remove()

  assert sync_back_by_name == {
    "weight": False,
    "scale": True,
  }


def test_layerwise_cpu_offload_full_leaf_coverage_keeps_internal_outputs_on_cuda() -> None:
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=924,
    device="cpu",
    dtype=torch.float32,
  )
  inputs = make_token_batch(
    batch_size=2,
    seq_len=8,
    width=128,
    seed=925,
    device="cpu",
    dtype=torch.float32,
  )

  observed_to_q_output_devices: list[str] = []
  offload_handle = layerwise_cpu_offload(
    model,
    onload_device="cuda",
    async_transfer=True,
  )
  capture_handle = model.block.to_q.register_forward_hook(
    lambda _module, _args, output: observed_to_q_output_devices.append(output.device.type))

  try:
    with torch.inference_mode():
      output = model(inputs)
  finally:
    capture_handle.remove()
    offload_handle.remove()

  assert offload_handle.keep_activations_onload_device is True
  assert observed_to_q_output_devices == ["cuda"]
  assert output.device.type == "cpu"


def test_layerwise_cpu_offload_partial_leaf_coverage_keeps_selected_outputs_on_input_device(
) -> None:
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=926,
    device="cpu",
    dtype=torch.float32,
  )
  inputs = make_token_batch(
    batch_size=2,
    seq_len=8,
    width=128,
    seed=927,
    device="cpu",
    dtype=torch.float32,
  )

  observed_to_q_output_devices: list[str] = []
  offload_handle = layerwise_cpu_offload(
    model,
    module_names=["block.to_q", "block.to_k", "block.to_v", "block.to_out"],
    onload_device="cuda",
    async_transfer=True,
  )
  capture_handle = model.block.to_q.register_forward_hook(
    lambda _module, _args, output: observed_to_q_output_devices.append(output.device.type))

  try:
    with torch.inference_mode():
      output = model(inputs)
  finally:
    capture_handle.remove()
    offload_handle.remove()

  assert offload_handle.keep_activations_onload_device is False
  assert observed_to_q_output_devices == ["cpu"]
  assert output.device.type == "cpu"


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
    prefetch_limit=True,
  )

  try:
    with torch.inference_mode():
      output = model(inputs)
  finally:
    offload_handle.remove()

  assert torch.isfinite(output).all()
  assert len(offload_handle._onload_copy_streams) == 2
  assert len(offload_handle._offload_copy_streams) == 2
  assert [name for name, _stream_index in scheduled_onload_streams[:3]] == [
    "block.to_k",
    "block.to_v",
    "block.to_out",
  ]
  assert scheduled_onload_streams[1][1] != scheduled_onload_streams[2][1]


def test_layerwise_cpu_offload_async_transfer_uses_distinct_onload_and_offload_stream_pools(
  monkeypatch, ) -> None:
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=920,
    device="cpu",
    dtype=torch.float32,
  )
  model.train()
  inputs = make_token_batch(
    batch_size=2,
    seq_len=8,
    width=128,
    seed=921,
    device="cpu",
    dtype=torch.float32,
  )

  selected_streams: list[tuple[str, int, int]] = []
  original_select_onload_copy_stream = LayerwiseOffloadHandle._select_onload_copy_stream
  original_select_offload_copy_stream = LayerwiseOffloadHandle._select_offload_copy_stream

  def _capture_select_onload_copy_stream(self):
    stream_index, stream = original_select_onload_copy_stream(self)
    selected_streams.append(("onload", stream_index, id(stream)))
    return stream_index, stream

  def _capture_select_offload_copy_stream(self):
    stream_index, stream = original_select_offload_copy_stream(self)
    selected_streams.append(("offload", stream_index, id(stream)))
    return stream_index, stream

  monkeypatch.setattr(
    LayerwiseOffloadHandle,
    "_select_onload_copy_stream",
    _capture_select_onload_copy_stream,
  )
  monkeypatch.setattr(
    LayerwiseOffloadHandle,
    "_select_offload_copy_stream",
    _capture_select_offload_copy_stream,
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
  assert selected_streams
  onload_stream_ids = {
    stream_id
    for transfer_kind, _stream_index, stream_id in selected_streams if transfer_kind == "onload"
  }
  offload_stream_ids = {
    stream_id
    for transfer_kind, _stream_index, stream_id in selected_streams if transfer_kind == "offload"
  }
  assert onload_stream_ids
  assert offload_stream_ids
  assert onload_stream_ids <= {id(stream) for stream in offload_handle._onload_copy_streams}
  assert offload_stream_ids <= {id(stream) for stream in offload_handle._offload_copy_streams}
  assert onload_stream_ids.isdisjoint(offload_stream_ids)


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


def test_layerwise_cpu_offload_async_transfer_expands_future_prefetch_window() -> None:
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
    prefetch_limit=True,
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
  assert offload_handle.effective_transfer_buckets == 8
  assert max(observed_pending_onload_sizes) <= 3


def test_layerwise_cpu_offload_async_transfer_clamps_excessive_copy_stream_request(
  monkeypatch, ) -> None:
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=928,
    device="cpu",
    dtype=torch.float32,
  )

  warning_messages: list[str] = []

  def _capture_warning(message: str, *args) -> None:
    warning_messages.append(message % args if args else message)

  monkeypatch.setattr(layerwise_module.logger, "warning", _capture_warning)

  offload_handle = layerwise_cpu_offload(
    model,
    module_names=["block.to_q", "block.to_k", "block.to_v", "block.to_out"],
    onload_device="cuda",
    async_transfer=True,
    transfer_buckets=32,
    max_copy_streams=32,
  )
  try:
    assert offload_handle.transfer_buckets == 32
    assert offload_handle.max_copy_streams == 32
    assert offload_handle.effective_max_copy_streams == 4
    assert len(offload_handle._onload_copy_streams) == 4
    assert len(offload_handle._offload_copy_streams) == 4
  finally:
    offload_handle.remove()

  assert warning_messages
  assert "Clamping layerwise async copy streams from 32 to 4" in warning_messages[0]


def test_layerwise_cpu_offload_async_transfer_default_stream_count_does_not_warn(
  monkeypatch, ) -> None:
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=946,
    device="cpu",
    dtype=torch.float32,
  )

  warning_messages: list[str] = []

  def _capture_warning(message: str, *args) -> None:
    warning_messages.append(message % args if args else message)

  monkeypatch.setattr(layerwise_module.logger, "warning", _capture_warning)

  offload_handle = layerwise_cpu_offload(
    model,
    module_names=["block.to_q", "block.to_k", "block.to_v", "block.to_out"],
    onload_device="cuda",
    async_transfer=True,
    transfer_buckets=1,
  )
  try:
    assert offload_handle.max_copy_streams is None
    assert offload_handle.effective_max_copy_streams == 1
  finally:
    offload_handle.remove()

  assert warning_messages == []


def test_layerwise_cpu_offload_async_transfer_caps_copy_stream_pool_without_warning(
  monkeypatch, ) -> None:
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=928,
    device="cpu",
    dtype=torch.float32,
  )

  warning_messages: list[str] = []

  def _capture_warning(message: str, *args) -> None:
    warning_messages.append(message % args if args else message)

  monkeypatch.setattr(layerwise_module.logger, "warning", _capture_warning)

  offload_handle = layerwise_cpu_offload(
    model,
    module_names=["block.to_q", "block.to_k", "block.to_v", "block.to_out"],
    onload_device="cuda",
    async_transfer=True,
    transfer_buckets=32,
  )
  try:
    assert offload_handle.transfer_buckets == 32
    assert offload_handle.effective_transfer_buckets is None
    assert len(offload_handle._onload_copy_streams) == 4
    assert len(offload_handle._offload_copy_streams) == 4
  finally:
    offload_handle.remove()

  assert warning_messages == []


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


def test_layerwise_cpu_offload_persistent_buckets_keep_prefix_targets_on_cuda() -> None:
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=929,
    device="cpu",
    dtype=torch.float32,
  )
  inputs = make_token_batch(
    batch_size=2,
    seq_len=8,
    width=128,
    seed=930,
    device="cpu",
    dtype=torch.float32,
  )

  offload_handle = layerwise_cpu_offload(
    model,
    module_names=["block.to_q", "block.to_k", "block.to_v", "block.to_out"],
    onload_device="cuda",
    async_transfer=True,
    transfer_buckets=2,
    persistent_buckets=2,
  )

  try:
    assert offload_handle.persistent_buckets == 2
    assert offload_handle.persistent_bins == 1
    assert offload_handle.effective_persistent_buckets == 2
    assert offload_handle.effective_persistent_bins == 1
    assert offload_handle.persistent_module_names == ["block.to_q", "block.to_k"]
    assert model.block.to_q.weight.device.type == "cuda"
    assert model.block.to_k.weight.device.type == "cuda"
    assert model.block.to_v.weight.device.type == "cpu"
    assert model.block.to_out.weight.device.type == "cpu"

    with torch.inference_mode():
      output = model(inputs)
      torch.cuda.synchronize()

    assert torch.isfinite(output).all()
    assert output.device.type == "cpu"
    assert model.block.to_q.weight.device.type == "cuda"
    assert model.block.to_k.weight.device.type == "cuda"
    assert model.block.to_v.weight.device.type == "cpu"
    assert model.block.to_out.weight.device.type == "cpu"
  finally:
    offload_handle.remove()

  assert all(parameter.device.type == "cpu" for parameter in model.parameters())


def test_layerwise_cpu_offload_persistent_bins_select_uniform_target_ranges() -> None:
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=944,
    device="cpu",
    dtype=torch.float32,
  )

  offload_handle = layerwise_cpu_offload(
    model,
    module_names=["block.to_q", "block.to_k", "block.to_v", "block.to_out"],
    onload_device="cuda",
    async_transfer=True,
    persistent_buckets=2,
    persistent_bins=2,
  )

  try:
    assert offload_handle.persistent_buckets == 2
    assert offload_handle.persistent_bins == 2
    assert offload_handle.effective_persistent_buckets == 2
    assert offload_handle.effective_persistent_bins == 2
    assert offload_handle.persistent_module_names == ["block.to_q", "block.to_v"]
    assert offload_handle.persistent_target_spans == [(0, 1), (2, 3)]
    assert model.block.to_q.weight.device.type == "cuda"
    assert model.block.to_k.weight.device.type == "cpu"
    assert model.block.to_v.weight.device.type == "cuda"
    assert model.block.to_out.weight.device.type == "cpu"
  finally:
    offload_handle.remove()


def test_layerwise_cpu_offload_async_transfer_window_keeps_ready_targets_counted() -> None:
  model = nn.Module()
  model.layers = nn.ModuleList([nn.Linear(32, 32, bias=True) for _ in range(7)])

  offload_handle = layerwise_cpu_offload(
    model,
    module_names=[f"layers.{index}" for index in range(7)],
    onload_device="cuda",
    async_transfer=True,
    transfer_buckets=1,
    prefetch_limit=True,
    persistent_buckets=2,
  )

  try:
    first_persistent = offload_handle.targets[0]
    second_persistent = offload_handle.targets[1]
    first_prefetched = offload_handle.targets[2]
    second_prefetched = offload_handle.targets[3]
    third_prefetched = offload_handle.targets[4]
    fourth_prefetched = offload_handle.targets[5]
    refill_target = offload_handle.targets[6]

    offload_handle._prefetch_bucket_targets(first_persistent)
    assert offload_handle.effective_transfer_buckets == 4
    assert offload_handle._pending_onload_targets | offload_handle._ready_onload_targets == {
      2,
      3,
      4,
      5,
    }

    if first_prefetched.pending_onload_event is not None:
      offload_handle._clear_pending_onload(first_prefetched)
    if second_prefetched.pending_onload_event is not None:
      offload_handle._clear_pending_onload(second_prefetched)
    if third_prefetched.pending_onload_event is not None:
      offload_handle._clear_pending_onload(third_prefetched)
    if fourth_prefetched.pending_onload_event is not None:
      offload_handle._clear_pending_onload(fourth_prefetched)
    assert offload_handle._pending_onload_targets == set()
    assert offload_handle._ready_onload_targets == {2, 3, 4, 5}

    offload_handle._prefetch_bucket_targets(second_persistent)
    assert offload_handle._pending_onload_targets == set()
    assert offload_handle._ready_onload_targets == {2, 3, 4, 5}
    assert refill_target.pending_onload_event is None
    assert refill_target.resident_device.type == "cpu"

    offload_handle._consume_prefetched_target(first_prefetched)
    offload_handle._prefetch_bucket_targets(first_prefetched)
    assert offload_handle._ready_onload_targets == {3, 4, 5}
    assert refill_target.pending_onload_event is not None
  finally:
    offload_handle.remove()


def test_layerwise_cpu_offload_persistent_prefix_prefetches_first_non_persistent_targets(
  monkeypatch, ) -> None:
  model = make_toy_model(
    embed_dim=128,
    num_heads=4,
    seed=931,
    device="cpu",
    dtype=torch.float32,
  )
  inputs = make_token_batch(
    batch_size=2,
    seq_len=8,
    width=128,
    seed=932,
    device="cpu",
    dtype=torch.float32,
  )

  scheduled_onload_streams: list[tuple[str, int]] = []
  seen_targets: set[str] = set()
  scheduled_after_first_target: list[list[str]] = []
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
    persistent_buckets=2,
  )
  capture_handle = model.block.to_q.register_forward_pre_hook(
    lambda _module, _args: scheduled_after_first_target.append(
      [name for name, _stream_index in scheduled_onload_streams]))

  try:
    with torch.inference_mode():
      output = model(inputs)
  finally:
    capture_handle.remove()
    offload_handle.remove()

  assert torch.isfinite(output).all()
  assert scheduled_after_first_target == [["block.to_v", "block.to_out"]]
  assert [name for name, _stream_index in scheduled_onload_streams] == [
    "block.to_v",
    "block.to_out",
  ]
