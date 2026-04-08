"""Smoke test for the opaque torchao FP8 scaled_mm path used by compile + CUDA Graph +
float8_per_row.

Run with pytest:
    export CUDA_VISIBLE_DEVICES=7
    pytest tests/kernels/test_opaque_scaled_mm.py -q

Run directly:
    export CUDA_VISIBLE_DEVICES=7
    PYTHONPATH=src python tests/kernels/test_opaque_scaled_mm.py

Notes:
- After setting CUDA_VISIBLE_DEVICES=7, the selected GPU is exposed to the test
    process as the default visible CUDA device.
"""

import torch

try:
  import pytest
except ModuleNotFoundError:
  pytest = None

from cache_dit.quantization.torchao._scaled_mm import (
  enable_opaque_torchao_float8_scaled_mm, )


def _unsupported_reason() -> str | None:
  if not torch.cuda.is_available():
    return "CUDA is not available"
  capability = torch.cuda.get_device_capability()
  if capability < (8, 9):
    return ("The selected CUDA device does not support float8 rowwise quantization: "
            f"capability={capability}")
  return None


def _maybe_skip() -> None:
  reason = _unsupported_reason()
  if reason is not None:
    if pytest is not None:
      pytest.skip(reason)
    raise RuntimeError(reason)


def _run_opaque_scaled_mm_smoke_test() -> None:
  _maybe_skip()

  from torchao.float8.inference import Float8MMConfig
  from torchao.quantization.granularity import PerRow
  from torchao.quantization.quantize_.workflows.float8.float8_tensor import (
    Float8Tensor,
    QuantizeTensorToFloat8Kwargs,
  )

  torch.manual_seed(0)
  enable_opaque_torchao_float8_scaled_mm()
  device = torch.device("cuda")

  mm_config = Float8MMConfig(use_fast_accum=False, pad_inner_dim=False)
  act_quant_kwargs = QuantizeTensorToFloat8Kwargs(
    granularity=PerRow(),
    mm_config=mm_config,
  )

  with torch.no_grad():
    weight = torch.randn(32, 64, device=device, dtype=torch.bfloat16)
    qweight = Float8Tensor.from_hp(
      weight,
      granularity=PerRow(),
      mm_config=mm_config,
      act_quant_kwargs=act_quant_kwargs,
    )
    inputs = torch.randn(4, 64, device=device, dtype=torch.bfloat16)

    eager = torch.nn.functional.linear(inputs, qweight)

    compiled_linear = torch.compile(
      lambda x, w: torch.nn.functional.linear(x, w),
      options={"triton.cudagraphs": True},
    )

    compiled_linear(inputs, qweight)
    compiled = compiled_linear(inputs, qweight)
    torch.cuda.synchronize()

  assert eager.shape == compiled.shape
  assert eager.dtype == compiled.dtype == torch.bfloat16
  assert torch.isfinite(compiled).all()

  max_abs_diff = (eager - compiled).abs().max().item()
  assert max_abs_diff <= 0.5, (
    "Compiled opaque float8 scaled_mm diverged too much from eager path: "
    f"max_abs_diff={max_abs_diff}")


def test_opaque_scaled_mm_compile_cuda_graph_smoke() -> None:
  _run_opaque_scaled_mm_smoke_test()


def main() -> None:
  reason = _unsupported_reason()
  if reason is not None:
    print(f"opaque_scaled_mm smoke test skipped: {reason}")
    raise SystemExit(0)

  _run_opaque_scaled_mm_smoke_test()
  print("opaque_scaled_mm smoke test passed")


if __name__ == "__main__":
  main()
