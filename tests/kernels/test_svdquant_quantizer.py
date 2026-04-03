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
from tests.kernels._svdq_test_utils import format_rank_report
from tests.kernels._svdq_test_utils import make_rank_sensitive_linear
from tests.kernels._svdq_test_utils import make_token_batch
from tests.kernels._svdq_test_utils import make_token_samples
from tests.kernels._svdq_test_utils import make_toy_model
from tests.kernels._svdq_test_utils import quantize_toy_model
from tests.kernels._svdq_test_utils import runtime_dtype

# For testing purposes, not recommended. Set to True to speed up tests with a potential accuracy regression.
_USE_FAST_SVD = os.getenv("CACHE_DIT_SVDQ_TEST_USE_FAST_SVD", "0").lower() == "1"


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
        device=device,
        dtype=dtype,
    )
    rank0_module: SVDQW4A4Linear = quantize_linear_svdq_w4a4(
        linear, calibration, rank=0, device=device, torch_dtype=dtype
    )
    rank16_module: SVDQW4A4Linear = quantize_linear_svdq_w4a4(
        linear, calibration, rank=16, device=device, torch_dtype=dtype
    )
    rank32_module: SVDQW4A4Linear = quantize_linear_svdq_w4a4(
        linear, calibration, rank=32, device=device, torch_dtype=dtype
    )
    rank128_module: SVDQW4A4Linear = quantize_linear_svdq_w4a4(
        linear, calibration, rank=128, device=device, torch_dtype=dtype
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
    print(format_rank_report("SVDQ linear module accuracy report", metrics_by_rank))

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
    num_heads = 48
    embed_dim = 128 * num_heads

    model = make_toy_model(
        embed_dim=embed_dim,
        num_heads=num_heads,
        seed=0,
        device=device,
        dtype=dtype,
    )
    calibration_samples = make_token_samples(
        num_samples=8,
        batch_size=1,
        seq_len=8192,
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
        print(f"\nQuantizing with rank {rank} with fast_svd={_USE_FAST_SVD} ...")
        quantize_start_time = time.perf_counter()
        quantized_model = quantize_toy_model(
            model,
            calibration_samples,
            rank=rank,
            device=device,
            dtype=dtype,
            fast_svd=_USE_FAST_SVD,
        )
        torch.cuda.synchronize()
        quantize_latency = time.perf_counter() - quantize_start_time
        print(f"Rank {rank} quantization time: {quantize_latency:.2f} seconds")

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
        atol = 4e-2 if not _USE_FAST_SVD else 1e-1  # For testing purposes, not recommended.
        rtol = 1e-2 if not _USE_FAST_SVD else 1e-1
        torch.testing.assert_close(reloaded_output, quantized_output, rtol=rtol, atol=atol)
        metrics_by_rank[rank] = compute_accuracy_metrics(
            reference,
            reloaded_output,
            reloaded_latency * 1000,  # reloaded latency in milliseconds
        )

    print(f"Profiling config: H={H}, D={D}, B={B}, S={S}")
    print(format_rank_report("SVDQ ToyModel accuracy report", metrics_by_rank))
    assert_rank_metric_trend(metrics_by_rank, "mae", ranks=RANKS_WITH_BASELINE)
    assert_rank_metric_trend(metrics_by_rank, "rel_l2", ranks=RANKS_WITH_BASELINE)
    for rank in EVALUATED_RANKS:
        assert metrics_by_rank[rank].mae < metrics_by_rank[0].mae
