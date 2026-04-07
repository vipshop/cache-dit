"""
cd cache-dit
pytest tests/kernels/test_svdquant_quantizer.py -v -s
"""

import os
from pathlib import Path
import time

import pytest
import torch
from torch import nn

from cache_dit.kernels import svdq_extension_is_available
from cache_dit.quantization.svdquant import SVDQW4A4Linear
from cache_dit.quantization.svdquant import quantize_linear_svdq_w4a4
from tests.kernels._svdq_test_utils import EVALUATED_RANKS
from tests.kernels._svdq_test_utils import RANKS_WITH_BASELINE
from tests.kernels._svdq_test_utils import assert_rank_metric_trend
from tests.kernels._svdq_test_utils import build_empty_quantized_toy_model
from tests.kernels._svdq_test_utils import compute_accuracy_metrics
from tests.kernels._svdq_test_utils import format_markdown_table
from tests.kernels._svdq_test_utils import format_rank_report
from tests.kernels._svdq_test_utils import make_rank_sensitive_linear
from tests.kernels._svdq_test_utils import make_token_batch
from tests.kernels._svdq_test_utils import make_token_samples
from tests.kernels._svdq_test_utils import make_toy_model
from tests.kernels._svdq_test_utils import quantize_toy_model
from tests.kernels._svdq_test_utils import runtime_dtype

_USE_HIGH_PRECISION = os.getenv("CACHE_DIT_SVDQ_TEST_HIGH_PRECISION", "0").lower() == "1"
_USE_FP32_FALLBACK = os.getenv("CACHE_DIT_SVDQ_TEST_FP32_FALLBACK", "1").lower() == "1"
_ENABLE_STREAMING_MEMORY_BENCH = os.getenv("CACHE_DIT_SVDQ_TEST_LARGE_MEMORY", "0").lower() == "1"
_ENABLE_LARGE_HEAD_NUMBER = os.getenv("CACHE_DIT_SVDQ_TEST_LARGE_HEAD_NUM", "0").lower() == "1"
_LARGE_MEMORY_TOTAL_GIB = float(os.getenv("CACHE_DIT_SVDQ_TEST_LARGE_MEMORY_GIB", "10"))
_LARGE_MEMORY_CHUNK_MIB = int(os.getenv("CACHE_DIT_SVDQ_TEST_LARGE_MEMORY_CHUNK_MIB", "256"))
_STREAMING_MEMORY_THRESHOLD_PCT = float(
    os.getenv("CACHE_DIT_SVDQ_TEST_STREAMING_MEMORY_THRESHOLD_PCT", "25")
)
_LARGE_MEMORY_MIN_DEVICE_GIB = float(os.getenv("CACHE_DIT_SVDQ_TEST_LARGE_MEMORY_MIN_GIB", "12"))


def _quantizer_kwargs(**overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "high_precision": _USE_HIGH_PRECISION,
        "fp32_fallback": _USE_FP32_FALLBACK,
        "streaming": True,
        "activation_buffer_flush_sample_count": 1,
        "activation_buffer_flush_cpu_bytes": None,
    }
    kwargs.update(overrides)
    return kwargs


def _current_tolerance() -> tuple[float, float]:
    if _USE_HIGH_PRECISION:
        return 4e-2, 1e-2
    if _USE_FP32_FALLBACK:
        return 6e-2, 2e-2
    return 1e-1, 1e-1


def _supports_low_precision_svd(device: str, dtype: torch.dtype) -> bool:
    try:
        matrix = torch.randn(32, 32, device=device, dtype=dtype)
        torch.linalg.svd(matrix, full_matrices=False)
        if device == "cuda":
            torch.cuda.synchronize()
        return True
    except (NotImplementedError, RuntimeError):
        return False


def _make_large_cpu_calibration_list(
    *,
    in_features: int,
    total_gib: float,
    chunk_mib: int,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    bytes_per_elem = torch.empty((), dtype=dtype).element_size()
    chunk_bytes = chunk_mib * 1024 * 1024
    rows_per_chunk = max(1, chunk_bytes // (in_features * bytes_per_elem))
    total_bytes = int(total_gib * (1024**3))

    calibration: list[torch.Tensor] = []
    allocated = 0
    while allocated < total_bytes:
        remaining_rows = max(1, (total_bytes - allocated) // (in_features * bytes_per_elem))
        rows = min(rows_per_chunk, remaining_rows)
        tensor = torch.zeros((rows, in_features), dtype=dtype, device="cpu")
        calibration.append(tensor)
        allocated += tensor.numel() * tensor.element_size()
    return calibration


def _measure_quantizer_peak_memory(
    linear: nn.Linear,
    representative: list[torch.Tensor],
    *,
    rank: int = 32,
    dtype: torch.dtype,
    streaming: bool,
) -> int:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    _ = quantize_linear_svdq_w4a4(
        linear,
        representative,
        rank=rank,
        device=linear.weight.device,
        torch_dtype=dtype,
        return_state_dict=True,
        **_quantizer_kwargs(streaming=streaming),
    )
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    torch.cuda.empty_cache()
    return peak


def _make_cpu_linear(in_features: int, out_features: int, *, bias: bool = True) -> nn.Linear:
    torch.manual_seed(0)
    linear = nn.Linear(in_features, out_features, bias=bias, device="cpu", dtype=torch.bfloat16)
    return linear


def test_svdquant_quantizer_returns_module_state_dict() -> None:
    linear = _make_cpu_linear(128, 256)
    representative = torch.randn(3, 5, 128, dtype=torch.float32)

    state_dict: dict[str, torch.Tensor] = quantize_linear_svdq_w4a4(
        linear,
        representative,
        rank=16,
        device="cpu",
        torch_dtype=torch.bfloat16,
        return_state_dict=True,
        **_quantizer_kwargs(fp32_fallback=True),
    )

    assert set(state_dict) == {
        "bias",
        "proj_down",
        "proj_up",
        "qweight",
        "smooth_factor",
        "smooth_factor_orig",
        "wscales",
    }
    assert state_dict["qweight"].shape == (256, 64)
    assert state_dict["wscales"].shape == (2, 256)
    assert state_dict["bias"].shape == (256,)
    assert state_dict["smooth_factor"].shape == (128,)
    assert state_dict["smooth_factor_orig"].shape == (128,)
    assert state_dict["proj_down"].shape == (128, 16)
    assert state_dict["proj_up"].shape == (256, 16)


def test_svdquant_quantizer_repairs_invalid_smooth_scales() -> None:
    linear = _make_cpu_linear(128, 128, bias=False)
    with torch.no_grad():
        linear.weight.zero_()

    state_dict: dict[str, torch.Tensor] = quantize_linear_svdq_w4a4(
        linear,
        torch.zeros(4, 128, dtype=torch.float32),
        rank=0,
        device="cpu",
        torch_dtype=torch.bfloat16,
        return_state_dict=True,
        **_quantizer_kwargs(fp32_fallback=True),
    )

    assert torch.equal(state_dict["smooth_factor"], torch.ones_like(state_dict["smooth_factor"]))
    assert torch.equal(
        state_dict["smooth_factor_orig"], torch.ones_like(state_dict["smooth_factor_orig"])
    )
    assert state_dict["proj_down"].shape == (128, 0)
    assert state_dict["proj_up"].shape == (128, 0)


def test_svdquant_quantizer_rejects_unsupported_geometry() -> None:
    linear = _make_cpu_linear(128, 96)

    with pytest.raises(ValueError, match="out_features"):
        quantize_linear_svdq_w4a4(
            linear,
            torch.randn(2, 128, dtype=torch.float32),
            rank=16,
            device="cpu",
            torch_dtype=torch.bfloat16,
            return_state_dict=True,
            **_quantizer_kwargs(fp32_fallback=True),
        )


def test_svdquant_quantizer_state_dict_loads_into_module() -> None:
    linear = _make_cpu_linear(128, 128)
    representative = [
        torch.randn(4, 128, dtype=torch.float32),
        torch.randn(2, 3, 128, dtype=torch.float32),
    ]
    state_dict: dict[str, torch.Tensor] = quantize_linear_svdq_w4a4(
        linear,
        representative,
        rank=16,
        device="cpu",
        torch_dtype=torch.bfloat16,
        return_state_dict=True,
        **_quantizer_kwargs(fp32_fallback=True),
    )

    module = SVDQW4A4Linear.from_linear(
        linear,
        rank=16,
        precision="int4",
        torch_dtype=torch.bfloat16,
        device="cpu",
    )
    incompatible = module.load_state_dict(state_dict, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


def test_svdquant_quantizer_streaming_matches_eager_state_dict() -> None:
    linear = _make_cpu_linear(128, 128)
    representative = [
        torch.randn(4, 128, dtype=torch.bfloat16),
        torch.randn(2, 3, 128, dtype=torch.bfloat16),
    ]

    streamed = quantize_linear_svdq_w4a4(
        linear,
        representative,
        rank=16,
        device="cpu",
        torch_dtype=torch.bfloat16,
        return_state_dict=True,
        high_precision=False,
        fp32_fallback=True,
        streaming=True,
    )
    eager = quantize_linear_svdq_w4a4(
        linear,
        representative,
        rank=16,
        device="cpu",
        torch_dtype=torch.bfloat16,
        return_state_dict=True,
        high_precision=False,
        fp32_fallback=True,
        streaming=False,
    )

    assert set(streamed) == set(eager)
    for key in streamed:
        torch.testing.assert_close(streamed[key], eager[key], rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "buffer_kwargs",
    [
        {"activation_buffer_flush_sample_count": 2},
        {"activation_buffer_flush_cpu_bytes": 256},
        {
            "activation_buffer_flush_sample_count": 3,
            "activation_buffer_flush_cpu_bytes": 256,
        },
    ],
)
def test_svdquant_quantizer_streaming_flush_thresholds_match_eager_state_dict(
    buffer_kwargs: dict[str, int],
) -> None:
    linear = _make_cpu_linear(128, 128)
    representative = [
        torch.randn(4, 128, dtype=torch.bfloat16),
        torch.randn(2, 3, 128, dtype=torch.bfloat16),
        torch.randn(1, 7, 128, dtype=torch.bfloat16),
        torch.randn(6, 128, dtype=torch.bfloat16),
    ]

    buffered = quantize_linear_svdq_w4a4(
        linear,
        (tensor for tensor in representative),
        rank=16,
        device="cpu",
        torch_dtype=torch.bfloat16,
        return_state_dict=True,
        high_precision=False,
        fp32_fallback=True,
        streaming=True,
        **buffer_kwargs,
    )
    eager = quantize_linear_svdq_w4a4(
        linear,
        representative,
        rank=16,
        device="cpu",
        torch_dtype=torch.bfloat16,
        return_state_dict=True,
        high_precision=False,
        fp32_fallback=True,
        streaming=False,
    )

    assert set(buffered) == set(eager)
    for key in buffered:
        torch.testing.assert_close(buffered[key], eager[key], rtol=0.0, atol=0.0)


def test_svdquant_quantizer_low_precision_svd_requires_fallback_when_unsupported() -> None:
    linear = _make_cpu_linear(128, 128)
    representative = torch.randn(2, 128, dtype=torch.bfloat16)

    if _supports_low_precision_svd("cpu", torch.bfloat16):
        quantize_linear_svdq_w4a4(
            linear,
            representative,
            rank=16,
            device="cpu",
            torch_dtype=torch.bfloat16,
            return_state_dict=True,
            high_precision=False,
            fp32_fallback=False,
        )
        return

    with pytest.raises(RuntimeError, match="Low-precision SVD is not supported"):
        quantize_linear_svdq_w4a4(
            linear,
            representative,
            rank=16,
            device="cpu",
            torch_dtype=torch.bfloat16,
            return_state_dict=True,
            high_precision=False,
            fp32_fallback=False,
        )


def test_svdquant_quantizer_runtime_rank32_beats_rank0() -> None:
    if not torch.cuda.is_available() or not svdq_extension_is_available():
        pytest.skip("CUDA runtime validation requires the optional SVDQuant extension.")

    device = "cuda"
    dtype = runtime_dtype()
    in_features = 128
    out_features = 128

    linear = make_rank_sensitive_linear(
        in_features=in_features,
        out_features=out_features,
        seed=17,
        device=device,
        dtype=dtype,
    )
    calibration = make_token_samples(
        num_samples=4,
        batch_size=1,
        seq_len=16,
        width=in_features,
        seed=29,
        device="cpu",
        dtype=dtype,
    )
    rank0_module: SVDQW4A4Linear = quantize_linear_svdq_w4a4(
        linear, calibration, rank=0, device=device, torch_dtype=dtype, **_quantizer_kwargs()
    )
    rank16_module: SVDQW4A4Linear = quantize_linear_svdq_w4a4(
        linear, calibration, rank=16, device=device, torch_dtype=dtype, **_quantizer_kwargs()
    )
    rank32_module: SVDQW4A4Linear = quantize_linear_svdq_w4a4(
        linear, calibration, rank=32, device=device, torch_dtype=dtype, **_quantizer_kwargs()
    )
    rank128_module: SVDQW4A4Linear = quantize_linear_svdq_w4a4(
        linear, calibration, rank=128, device=device, torch_dtype=dtype, **_quantizer_kwargs()
    )

    x = make_token_batch(
        batch_size=2,
        seq_len=16,
        width=in_features,
        seed=41,
        device=device,
        dtype=dtype,
    )
    with torch.inference_mode():
        reference = linear(x)
        rank0_output = rank0_module(x)
        rank16_output = rank16_module(x)
        rank32_output = rank32_module(x)
        rank128_output = rank128_module(x)
        torch.cuda.synchronize()

    metrics_by_rank = {
        0: compute_accuracy_metrics(reference, rank0_output),
        16: compute_accuracy_metrics(reference, rank16_output),
        32: compute_accuracy_metrics(reference, rank32_output),
        128: compute_accuracy_metrics(reference, rank128_output),
    }
    print(format_rank_report("SVDQ linear module accuracy report\n", metrics_by_rank))

    rank0_error = metrics_by_rank[0].mae
    rank16_error = metrics_by_rank[16].mae
    rank32_error = metrics_by_rank[32].mae
    rank128_error = metrics_by_rank[128].mae
    assert rank16_error < rank0_error
    assert rank32_error < rank16_error
    assert rank128_error < rank32_error


def test_svdquant_toymodel_rank_accuracy_roundtrip_report(tmp_path: Path) -> None:
    if not torch.cuda.is_available() or not svdq_extension_is_available():
        pytest.skip("CUDA runtime validation requires the optional SVDQuant extension.")

    device = "cuda"
    dtype = runtime_dtype()  # torch.bfloat16
    num_heads = 16 if not _ENABLE_LARGE_HEAD_NUMBER else 32
    embed_dim = 128 * num_heads

    model = make_toy_model(
        embed_dim=embed_dim,
        num_heads=num_heads,
        seed=0,
        device=device,
        dtype=dtype,
    )
    # case 0: large head number with shorter sequence length to reduce quantization time.
    # case 1: small head number with longer sequence length to better simulate the quantization.
    calibration_samples = make_token_samples(
        num_samples=8,
        batch_size=1,
        seq_len=8192 if not _ENABLE_LARGE_HEAD_NUMBER else 1024,
        width=embed_dim,
        seed=0,
        device=device,
        dtype=dtype,
    )
    # For simplicity, we use the same calibration samples as evaluation inputs. The main
    # goal of this test is to validate the quantizer's offline-to-runtime accuracy trend
    # and state dict integrity, rather than to benchmark on a separate evaluation set.
    eval_inputs = torch.cat(calibration_samples, dim=0)
    H, D, B, S = num_heads, embed_dim, eval_inputs.shape[0], eval_inputs.shape[1]

    metrics_by_rank = {}
    quantization_latency_rows: list[tuple[object, ...]] = []
    # Warmup
    with torch.inference_mode():
        reference = model(eval_inputs)
        torch.cuda.synchronize()
    # Profile reference latency, repeats=10
    with torch.inference_mode():
        start_time = time.perf_counter()
        for _ in range(10):
            _ = model(eval_inputs)
        torch.cuda.synchronize()
        reference_latency = (time.perf_counter() - start_time) / 10
        metrics_by_rank[-1] = compute_accuracy_metrics(
            reference,
            reference,
            latency_ms=reference_latency * 1000,  # reference latency in milliseconds
        )

    for rank in RANKS_WITH_BASELINE:
        quantize_start_time = time.perf_counter()
        quantized_model = quantize_toy_model(
            model,
            calibration_samples,
            rank=rank,
            device=device,
            dtype=dtype,
            high_precision=_USE_HIGH_PRECISION,
            fp32_fallback=_USE_FP32_FALLBACK,
        )
        torch.cuda.synchronize()
        quantize_latency = time.perf_counter() - quantize_start_time
        quantization_latency_rows.append((rank, f"{quantize_latency:.6f}"))

        checkpoint_path = tmp_path / f"svdq_toy_rank{rank}.pt"
        torch.save(
            {
                "model_config": {"embed_dim": embed_dim, "num_heads": num_heads},
                "rank": rank,
                "state_dict": quantized_model.state_dict(),
            },
            checkpoint_path,
        )

        payload = torch.load(checkpoint_path, map_location=device)
        model_config = payload["model_config"]
        reloaded_model = build_empty_quantized_toy_model(
            embed_dim=model_config["embed_dim"],
            num_heads=model_config["num_heads"],
            rank=payload["rank"],
            device=device,
            dtype=dtype,
        )
        incompatible = reloaded_model.load_state_dict(payload["state_dict"], strict=True)
        assert incompatible.missing_keys == []
        assert incompatible.unexpected_keys == []

        # Warmup
        with torch.inference_mode():
            quantized_output = quantized_model(eval_inputs)
            reloaded_output = reloaded_model(eval_inputs)
            torch.cuda.synchronize()

        # Profile and validate outputs, repeats=10
        with torch.inference_mode():
            start_time = time.perf_counter()
            for _ in range(10):
                _ = reloaded_model(eval_inputs)
            torch.cuda.synchronize()
            reloaded_latency = (time.perf_counter() - start_time) / 10
        # May not bitwise-deterministic due to non-determinism in CUDA.
        # BFloat16 atol can be ranged in [4e-3, 8e-3].
        atol, rtol = _current_tolerance()
        torch.testing.assert_close(reloaded_output, quantized_output, rtol=rtol, atol=atol)
        metrics_by_rank[rank] = compute_accuracy_metrics(
            reference,
            reloaded_output,
            reloaded_latency * 1000,  # reloaded latency in milliseconds
        )

    print(
        format_markdown_table(
            "SVDQ ToyModel profiling config\n",
            ("num_heads", "embed_dim", "batch", "seq_len", "high_precision", "fp32_fallback"),
            [(H, D, B, S, _USE_HIGH_PRECISION, _USE_FP32_FALLBACK)],
        )
    )
    print(
        format_markdown_table(
            "SVDQ ToyModel quantization latency\n",
            ("rank", "quantization_s"),
            quantization_latency_rows,
        )
    )
    print(format_rank_report("SVDQ ToyModel accuracy report\n", metrics_by_rank))
    assert_rank_metric_trend(metrics_by_rank, "mae", ranks=RANKS_WITH_BASELINE)
    assert_rank_metric_trend(metrics_by_rank, "rel_l2", ranks=RANKS_WITH_BASELINE)
    for rank in EVALUATED_RANKS:
        assert metrics_by_rank[rank].mae < metrics_by_rank[0].mae


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _ENABLE_STREAMING_MEMORY_BENCH,
    reason="Streaming memory benchmark requires CUDA and CACHE_DIT_SVDQ_TEST_LARGE_MEMORY=1.",
)
def test_svdquant_streaming_memory_peak_is_lower() -> None:
    device_props = torch.cuda.get_device_properties(0)
    total_gib = device_props.total_memory / (1024**3)
    if total_gib < _LARGE_MEMORY_MIN_DEVICE_GIB:
        pytest.skip(
            f"Streaming memory benchmark requires at least {_LARGE_MEMORY_MIN_DEVICE_GIB:.1f} GiB, got {total_gib:.1f} GiB."
        )

    device = torch.device("cuda")
    dtype = runtime_dtype()
    linear = nn.Linear(128, 128, bias=False, device=device, dtype=dtype).eval()
    representative = _make_large_cpu_calibration_list(
        in_features=128,
        total_gib=_LARGE_MEMORY_TOTAL_GIB,
        chunk_mib=_LARGE_MEMORY_CHUNK_MIB,
        dtype=dtype,
    )

    try:
        streaming_peak = _measure_quantizer_peak_memory(
            linear,
            representative,
            dtype=dtype,
            streaming=True,
        )
        eager_peak = _measure_quantizer_peak_memory(
            linear,
            representative,
            dtype=dtype,
            streaming=False,
        )
    except torch.cuda.OutOfMemoryError as exc:
        pytest.skip(f"Not enough free GPU memory for the eager streaming benchmark: {exc}")

    assert eager_peak > streaming_peak
    savings_pct = 100.0 * (eager_peak - streaming_peak) / eager_peak
    print(
        format_markdown_table(
            "SVDQ streaming memory benchmark\n",
            (
                "rank",
                "cpu_calibration_gib",
                "streaming_peak_gib",
                "eager_peak_gib",
                "savings_pct",
            ),
            [
                (
                    32,
                    f"{_LARGE_MEMORY_TOTAL_GIB:.2f}",
                    f"{streaming_peak / 2**30:.4f}",
                    f"{eager_peak / 2**30:.4f}",
                    f"{savings_pct:.2f}",
                )
            ],
        )
    )
    assert savings_pct >= _STREAMING_MEMORY_THRESHOLD_PCT
