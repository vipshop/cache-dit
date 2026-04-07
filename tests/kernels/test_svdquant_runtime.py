"""
cd cache-dit
pytest tests/kernels/test_svdquant_runtime.py -v -s
"""

import pytest
import torch

from cache_dit.kernels import svdq_extension_is_available
from cache_dit.kernels import svdq_gemm_w4a4
from cache_dit.kernels import svdq_gemm_w4a4_ext
from cache_dit.kernels import svdq_quantize_w4a4_act_fuse_lora
from cache_dit.kernels import svdq_quantize_w4a4_wgt
from cache_dit.quantization.svdquant import quantize_linear_svdq_w4a4
from tests.kernels._svdq_test_utils import RANKS_WITH_BASELINE
from tests.kernels._svdq_test_utils import assert_rank_metric_trend
from tests.kernels._svdq_test_utils import compute_accuracy_metrics
from tests.kernels._svdq_test_utils import format_rank_report
from tests.kernels._svdq_test_utils import make_rank_sensitive_linear
from tests.kernels._svdq_test_utils import make_token_batch
from tests.kernels._svdq_test_utils import make_token_samples
from tests.kernels._svdq_test_utils import run_svdq_operator_from_state_dict
from tests.kernels._svdq_test_utils import runtime_dtype


def _require_svdquant_runtime() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the SVDQuant runtime smoke test.")
    if not svdq_extension_is_available():
        pytest.skip(
            "The optional Cache-DiT SVDQuant CUDA extension is not built in this environment."
        )


def test_svdquant_extension_loads_on_cuda_runtime() -> None:
    _require_svdquant_runtime()
    assert svdq_extension_is_available()


def test_svdquant_int4_runtime_smoke() -> None:
    _require_svdquant_runtime()

    device = "cuda"
    dtype = runtime_dtype()
    batch_size = 256
    in_features = 128
    out_features = 128
    rank = 16

    with torch.inference_mode():
        activations = torch.randn(batch_size, in_features, device=device, dtype=dtype)
        smooth = torch.ones(in_features, device=device, dtype=dtype)
        lora_down = torch.zeros(in_features, rank, device=device, dtype=dtype)
        lora_up = torch.zeros(out_features, rank, device=device, dtype=dtype)
        bias = torch.zeros(out_features, device=device, dtype=dtype)

        quantized_activations, ascales, lora_activations = svdq_quantize_w4a4_act_fuse_lora(
            input=activations,
            lora_down=lora_down,
            smooth=smooth,
            fp4=False,
            pad_size=256,
        )

        weights = torch.randn(out_features, in_features, device=device, dtype=dtype)
        qweight, wscales = svdq_quantize_w4a4_wgt(weights)

        output = svdq_gemm_w4a4(
            act=quantized_activations,
            wgt=qweight,
            ascales=ascales,
            wscales=wscales,
            lora_act_in=lora_activations,
            lora_up=lora_up,
            bias=bias,
            act_unsigned=False,
            fp4=False,
            alpha=1.0,
        )
        torch.cuda.synchronize()

    assert quantized_activations.shape == (batch_size, in_features // 2)
    assert ascales.shape == (in_features // 64, batch_size)
    assert qweight.shape == (out_features, in_features // 2)
    assert wscales.shape == (in_features // 64, out_features)
    assert output.shape == (batch_size, out_features)
    assert output.is_cuda
    assert output.dtype == dtype
    assert torch.isfinite(output).all()
    assert output.float().abs().sum().item() > 0.0


def test_svdquant_int4_ext_runtime_smoke() -> None:
    _require_svdquant_runtime()

    device = "cuda"
    dtype = runtime_dtype()
    batch_size = 256
    in_features = 128
    out_features = 128
    rank = 16

    with torch.inference_mode():
        activations = torch.randn(batch_size, in_features, device=device, dtype=dtype)
        smooth = torch.ones(in_features, device=device, dtype=dtype)
        lora_down = torch.zeros(in_features, rank, device=device, dtype=dtype)
        lora_up = torch.zeros(out_features, rank, device=device, dtype=dtype)
        bias = torch.zeros(out_features, device=device, dtype=dtype)

        quantized_activations, ascales, lora_activations = svdq_quantize_w4a4_act_fuse_lora(
            input=activations,
            lora_down=lora_down,
            smooth=smooth,
            fp4=False,
            pad_size=256,
        )

        weights = torch.randn(out_features, in_features, device=device, dtype=dtype)
        qweight, wscales = svdq_quantize_w4a4_wgt(weights)

        base_output = svdq_gemm_w4a4(
            act=quantized_activations,
            wgt=qweight,
            ascales=ascales,
            wscales=wscales,
            lora_act_in=lora_activations,
            lora_up=lora_up,
            bias=bias,
            act_unsigned=False,
            fp4=False,
            alpha=1.0,
        )
        ext_output = torch.empty_like(base_output)
        result = svdq_gemm_w4a4_ext(
            act=quantized_activations,
            wgt=qweight,
            out=ext_output,
            ascales=ascales,
            wscales=wscales,
            lora_act_in=lora_activations,
            lora_up=lora_up,
            bias=bias,
            act_unsigned=False,
            fp4=False,
            alpha=1.0,
        )
        torch.cuda.synchronize()

    assert result is ext_output
    assert ext_output.shape == (batch_size, out_features)
    assert ext_output.is_cuda
    assert ext_output.dtype == dtype
    assert torch.isfinite(ext_output).all()
    torch.testing.assert_close(ext_output, base_output, rtol=0.0, atol=0.0)


def test_svdquant_operator_rank_accuracy_improves_with_rank() -> None:
    _require_svdquant_runtime()

    device = "cuda"
    dtype = runtime_dtype()
    in_features = 128
    out_features = 128

    linear = make_rank_sensitive_linear(
        in_features=in_features,
        out_features=out_features,
        seed=11,
        device=device,
        dtype=dtype,
    )
    calibration_samples = make_token_samples(
        num_samples=8,
        batch_size=1,
        seq_len=16,
        width=in_features,
        seed=101,
        device="cpu",
        dtype=dtype,
    )
    eval_tokens = make_token_batch(
        batch_size=6,
        seq_len=16,
        width=in_features,
        seed=303,
        device=device,
        dtype=dtype,
    )
    eval_matrix = eval_tokens.reshape(-1, in_features)

    metrics_by_rank = {}
    with torch.inference_mode():
        reference = linear(eval_matrix)
        for rank in RANKS_WITH_BASELINE:
            state_dict: dict[str, torch.Tensor] = quantize_linear_svdq_w4a4(
                linear,
                calibration_samples,
                rank=rank,
                device=device,
                torch_dtype=dtype,
                return_state_dict=True,
                high_precision=False,
                fp32_fallback=True,
                streaming=True,
            )
            operator_output = run_svdq_operator_from_state_dict(
                state_dict,
                eval_matrix,
                output_dtype=dtype,
            )
            metrics_by_rank[rank] = compute_accuracy_metrics(reference, operator_output)
        torch.cuda.synchronize()

    print(format_rank_report("SVDQ operator accuracy report", metrics_by_rank))
    assert_rank_metric_trend(metrics_by_rank, "mae", ranks=RANKS_WITH_BASELINE)
    assert_rank_metric_trend(metrics_by_rank, "rel_l2", ranks=RANKS_WITH_BASELINE)
    assert metrics_by_rank[128].mae < metrics_by_rank[0].mae
    assert metrics_by_rank[128].rel_l2 < metrics_by_rank[0].rel_l2
