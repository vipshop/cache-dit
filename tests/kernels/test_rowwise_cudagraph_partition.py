"""Regression coverage for the rowwise FP8 quantize -> scaled_mm CUDA Graph replay hang.

The tests cover two cases:
- raw torchao rowwise quantize + scaled_mm replay still reproduces the hang (currently
    skipped: the hang does not reproduce on torch 2.15 / torchao 0.18 with the torch
    `torch._scaled_mm` path, and the mslk path aborts on sm12x -- see issue #1286),
- Cache-DiT's tagged scaled_mm wrapper keeps the call compiled while letting
    compile + CUDA Graph partition around it so the realistic merged-stream
    attention repro completes.

Both repros pin `KernelPreference.TORCH` for the fp8 weights, because that is the path
Cache-DiT configures for float8_per_row and the one `enable_opaque_torchao_float8_scaled_mm()`
patches; torchao's AUTO would route to mslk, which has no sm12x fp8 rowwise kernel (#1286).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import torch

try:
  import pytest
except ModuleNotFoundError:
  pytest = None

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_REPRO_TIMEOUT_SECONDS = 20
PATCHED_REPRO_TIMEOUT_SECONDS = 180

RAW_ROW_REPRO = textwrap.dedent("""
    import torch
    from torchao.float8.inference import Float8MMConfig, addmm_float8_unwrapped_inference
    from torchao.quantization.granularity import PerRow
    from torchao.quantization.quantize_.common import KernelPreference
    from torchao.quantization.quantize_.workflows.float8.float8_tensor import Float8Tensor, QuantizeTensorToFloat8Kwargs

    # Same `torch._scaled_mm` path as the tagged-wrapper case; see the module docstring.
    _PREF = KernelPreference.TORCH

    M = 512
    K = 3072
    N = 3072
    mm_config = Float8MMConfig(use_fast_accum=True, pad_inner_dim=False)
    kwargs = QuantizeTensorToFloat8Kwargs(granularity=PerRow(), mm_config=mm_config)

    x = torch.randn(1, M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    wq = Float8Tensor.from_hp(w, granularity=PerRow(), mm_config=mm_config, act_quant_kwargs=kwargs, kernel_preference=_PREF)
    b_data = wq.qdata.t()
    b_scale = wq.scale.t().contiguous()

    def fn():
        xq = Float8Tensor.from_hp(x, granularity=PerRow(), mm_config=mm_config, act_quant_kwargs=kwargs, kernel_preference=_PREF)
        a_data = xq.qdata.reshape(-1, xq.qdata.shape[-1])
        a_scale = xq.scale.reshape(-1, 1).contiguous()
        return addmm_float8_unwrapped_inference(
            a_data,
            a_scale,
            b_data,
            b_scale,
            output_dtype=torch.bfloat16,
            bias=None,
            use_fast_accum=True,
        ).reshape(1, M, N)

    out = fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = fn()
    torch.cuda.synchronize()
    print("capture_done", tuple(captured.shape), flush=True)
    graph.replay()
    torch.cuda.synchronize()
    print("replay_done", flush=True)
    """)

PATCHED_MERGED_STREAM_REPRO = textwrap.dedent("""
    import time
    import torch
    import torch.nn.functional as F
    from torchao.float8.inference import Float8MMConfig
    from torchao.quantization.granularity import PerRow
    from torchao.quantization.quantize_.common import KernelPreference
    from torchao.quantization.quantize_.workflows.float8.float8_tensor import Float8Tensor, QuantizeTensorToFloat8Kwargs
    from cache_dit.quantization.torchao.scaled_mm import enable_opaque_torchao_float8_scaled_mm

    # Pin the `torch._scaled_mm` path that Cache-DiT's wrapper patches: torchao's AUTO picks
    # mslk on any sm90+, but mslk 1.4.0 has no sm12x fp8 rowwise kernel and aborts with
    # `RuntimeError: cutlass cannot initialize` (#1286).
    _PREF = KernelPreference.TORCH

    B, SI, ST, H, D = 1, 12288, 512, 24, 128
    S = SI + ST
    C = H * D
    mm_config = Float8MMConfig(use_fast_accum=True, pad_inner_dim=False)
    kwargs = QuantizeTensorToFloat8Kwargs(granularity=PerRow(), mm_config=mm_config)

    def make_weight():
        w = torch.randn(C, C, device="cuda", dtype=torch.bfloat16)
        return Float8Tensor.from_hp(
            w,
            granularity=PerRow(),
            mm_config=mm_config,
            act_quant_kwargs=kwargs,
            kernel_preference=_PREF,
        )

    torch.manual_seed(0)
    enable_opaque_torchao_float8_scaled_mm()
    x_img = torch.randn(B, SI, C, device="cuda", dtype=torch.bfloat16)
    x_txt = torch.randn(B, ST, C, device="cuda", dtype=torch.bfloat16)
    wq = make_weight()
    wk = make_weight()
    wv = make_weight()

    def fn(img, txt, qw, kw, vw):
        q_img = F.linear(img, qw).view(B, SI, H, D)
        k_img = F.linear(img, kw).view(B, SI, H, D)
        v_img = F.linear(img, vw).view(B, SI, H, D)
        q_txt = F.linear(txt, qw).view(B, ST, H, D)
        k_txt = F.linear(txt, kw).view(B, ST, H, D)
        v_txt = F.linear(txt, vw).view(B, ST, H, D)
        q = torch.empty((B, S, H, D), device=img.device, dtype=img.dtype)
        k = torch.empty((B, S, H, D), device=img.device, dtype=img.dtype)
        v = torch.empty((B, S, H, D), device=img.device, dtype=img.dtype)
        q[:, :ST] = q_txt
        q[:, ST:] = q_img
        k[:, :ST] = k_txt
        k[:, ST:] = k_img
        v[:, :ST] = v_txt
        v[:, ST:] = v_img
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()
        return F.scaled_dot_product_attention(q, k, v)

    compiled = torch.compile(fn, options={"triton.cudagraphs": True})
    t0 = time.time()
    out1 = compiled(x_img, x_txt, wq, wk, wv)
    torch.cuda.synchronize()
    t1 = time.time()
    out2 = compiled(x_img, x_txt, wq, wk, wv)
    torch.cuda.synchronize()
    t2 = time.time()
    print("ok", tuple(out1.shape), round(t1 - t0, 3), round(t2 - t1, 3), flush=True)
    assert out2.shape == out1.shape
    """)


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


def _run_python_snippet(code: str, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
  env = os.environ.copy()
  env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
  return subprocess.run(
    [sys.executable, "-c", code],
    cwd=REPO_ROOT,
    env=env,
    text=True,
    capture_output=True,
    timeout=timeout_seconds,
    check=False,
  )


_RAW_REPRO_SKIP_REASON = (
  "The raw rowwise FP8 + CUDA Graph replay hang no longer reproduces with the torch "
  "`torch._scaled_mm` path on torch 2.15 / torchao 0.18, and on sm12x the mslk path aborts earlier "
  "with `RuntimeError: cutlass cannot initialize`, so the hang cannot be observed at all "
  "(issue #1286). The tagged-wrapper case below stays enabled as the regression guard; "
  "re-enable this one only to re-verify the hang on SM90/SM100.")


def test_raw_rowwise_quant_scaled_mm_cudagraph_replay_repro() -> None:
  if pytest is not None:
    pytest.skip(_RAW_REPRO_SKIP_REASON)
  print(f"raw rowwise cudagraph repro skipped: {_RAW_REPRO_SKIP_REASON}")


def test_cudagraph_partitioned_rowwise_merged_stream_repro() -> None:
  _maybe_skip()

  result = _run_python_snippet(
    PATCHED_MERGED_STREAM_REPRO,
    timeout_seconds=PATCHED_REPRO_TIMEOUT_SECONDS,
  )

  assert result.returncode == 0, (
    "Expected the cudagraph-partitioned merged-stream repro to complete.\n"
    f"stdout:\n{result.stdout}\n"
    f"stderr:\n{result.stderr}")
  assert "ok" in result.stdout
