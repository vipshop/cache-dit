"""Cd cache-dit pytest tests/kernels/test_svdquant_runtime.py -v -s."""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from cache_dit.kernels import cuda_ext_is_available
from cache_dit.kernels import svdq_gemm_w4a4
from cache_dit.kernels import svdq_quantize_w4a4
from cache_dit.quantization.svdquant import quantize_linear_svdq_w4a4
from tests.quantization._svdq_test_utils import RANKS_WITH_BASELINE
from tests.quantization._svdq_test_utils import assert_rank_metric_trend
from tests.quantization._svdq_test_utils import compute_accuracy_metrics
from tests.quantization._svdq_test_utils import format_rank_report
from tests.quantization._svdq_test_utils import make_rank_sensitive_linear
from tests.quantization._svdq_test_utils import make_token_batch
from tests.quantization._svdq_test_utils import make_token_samples
from tests.quantization._svdq_test_utils import pack_int4_weight
from tests.quantization._svdq_test_utils import run_svdq_operator_from_state_dict
from tests.quantization._svdq_test_utils import runtime_dtype


def _require_svdquant_runtime() -> None:
  if not torch.cuda.is_available():
    pytest.skip("CUDA is required for the SVDQuant runtime smoke test.")
  if not cuda_ext_is_available():
    pytest.skip("The optional Cache-DiT SVDQuant CUDA extension is not built in this environment.")


def test_svdquant_extension_loads_on_cuda_runtime() -> None:
  _require_svdquant_runtime()
  assert cuda_ext_is_available()


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

    quantized_activations, ascales, lora_activations = svdq_quantize_w4a4(
      input=activations,
      lora_down=lora_down,
      smooth=smooth,
      fuse_glu=False,
      fp4=False,
      pad_size=256,
    )

    weights = torch.randn(out_features, in_features, device=device, dtype=dtype)
    qweight, wscales = pack_int4_weight(weights)

    output = svdq_gemm_w4a4(
      act=quantized_activations,
      wgt=qweight,
      ascales=ascales,
      wscales=wscales,
      lora_act_in=lora_activations,
      lora_up=lora_up,
      bias=bias,
      wcscales=None,
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


def test_svdquant_fused_gelu_epilogue_matches_gelu_of_plain_output() -> None:
  """The fused qout/oscales/lora_down branch must not disturb the GEMM accumulator.

  Passing the fused arguments switches the kernel to the GELU + quantized-output
  epilogue chain. `out` then holds `gelu(acc)` instead of `acc`, so comparing it
  against `gelu(out_plain)` proves the extra arguments only add epilogue work and
  do not reorder or drop any of the packed inputs.
  """

  _require_svdquant_runtime()

  device = "cuda"
  dtype = runtime_dtype()
  batch_size = 256
  in_features = 128
  out_features = 128
  rank = 32

  with torch.inference_mode():
    activations = torch.randn(batch_size, in_features, device=device, dtype=dtype)
    smooth = torch.ones(in_features, device=device, dtype=dtype)
    lora_down = torch.randn(in_features, rank, device=device, dtype=dtype)
    lora_up = torch.randn(out_features, rank, device=device, dtype=dtype)
    bias = torch.randn(out_features, device=device, dtype=dtype)

    quantized_activations, ascales, lora_activations = svdq_quantize_w4a4(
      input=activations,
      lora_down=lora_down,
      smooth=smooth,
      fuse_glu=False,
      fp4=False,
      pad_size=256,
    )
    weights = torch.randn(out_features, in_features, device=device, dtype=dtype)
    qweight, wscales = pack_int4_weight(weights)

    plain = svdq_gemm_w4a4(
      act=quantized_activations,
      wgt=qweight,
      ascales=ascales,
      wscales=wscales,
      lora_act_in=lora_activations,
      lora_up=lora_up,
      bias=bias,
      fp4=False,
      alpha=1.0,
    )

    qout = torch.zeros(batch_size, out_features // 2, dtype=torch.uint8, device=device)
    oscales = torch.zeros(out_features // 64, batch_size, dtype=dtype, device=device)
    lora_act_out = torch.zeros(batch_size, rank, dtype=torch.float32, device=device)
    fused = svdq_gemm_w4a4(
      act=quantized_activations,
      wgt=qweight,
      ascales=ascales,
      wscales=wscales,
      qout=qout,
      oscales=oscales,
      lora_act_in=lora_activations,
      lora_up=lora_up,
      lora_down=lora_down,
      lora_act_out=lora_act_out,
      bias=bias,
      smooth_factor=torch.ones(out_features, dtype=dtype, device=device),
      fp4=False,
      alpha=1.0,
    )

    # Negative controls: the bias and the LoRA-up term have to move the result, otherwise
    # the O(1) comparison below would still pass with either of them silently dropped.
    without_bias = svdq_gemm_w4a4(
      act=quantized_activations,
      wgt=qweight,
      ascales=ascales,
      wscales=wscales,
      lora_act_in=lora_activations,
      lora_up=lora_up,
      fp4=False,
      alpha=1.0,
    )
    without_lora = svdq_gemm_w4a4(
      act=quantized_activations,
      wgt=qweight,
      ascales=ascales,
      wscales=wscales,
      bias=bias,
      fp4=False,
      alpha=1.0,
    )
    torch.cuda.synchronize()

  expected = F.gelu(plain.float(), approximate="tanh").to(dtype)
  # Both paths share one fp16 accumulator, so the only difference is the extra fp16
  # rounding of the GELU output in `fused`.
  torch.testing.assert_close(fused, expected, rtol=2e-3, atol=2e-3)
  assert not torch.allclose(plain, without_bias)
  assert not torch.allclose(plain, without_lora)
  assert qout.abs().sum().item() > 0
  assert torch.isfinite(oscales.float()).all()
  assert torch.isfinite(lora_act_out).all()
  assert lora_act_out.abs().sum().item() > 0


def test_svdquant_write_dense_false_preserves_the_fused_outputs() -> None:
  """Skipping the dense epilogue must not perturb qout / oscales / lora_act_out."""

  _require_svdquant_runtime()

  device = "cuda"
  dtype = runtime_dtype()
  batch_size = 256
  in_features = 128
  out_features = 128
  rank = 32

  with torch.inference_mode():
    activations = torch.randn(batch_size, in_features, device=device, dtype=dtype)
    smooth = torch.ones(in_features, device=device, dtype=dtype)
    lora_down = torch.randn(in_features, rank, device=device, dtype=dtype)
    lora_up = torch.randn(out_features, rank, device=device, dtype=dtype)
    bias = torch.randn(out_features, device=device, dtype=dtype)

    quantized_activations, ascales, lora_activations = svdq_quantize_w4a4(
      input=activations,
      lora_down=lora_down,
      smooth=smooth,
      fuse_glu=False,
      fp4=False,
      pad_size=256,
    )
    weights = torch.randn(out_features, in_features, device=device, dtype=dtype)
    qweight, wscales = pack_int4_weight(weights)
    smooth_factor = torch.ones(out_features, dtype=dtype, device=device)

    def run(write_dense: bool):
      qout = torch.zeros(batch_size, out_features // 2, dtype=torch.uint8, device=device)
      oscales = torch.zeros(out_features // 64, batch_size, dtype=dtype, device=device)
      lora_act_out = torch.zeros(batch_size, rank, dtype=torch.float32, device=device)
      output = svdq_gemm_w4a4(
        act=quantized_activations,
        wgt=qweight,
        ascales=ascales,
        wscales=wscales,
        qout=qout,
        oscales=oscales,
        lora_act_in=lora_activations,
        lora_up=lora_up,
        lora_down=lora_down,
        lora_act_out=lora_act_out,
        bias=bias,
        smooth_factor=smooth_factor,
        fp4=False,
        alpha=1.0,
        write_dense=write_dense,
      )
      return output, qout.clone(), oscales.clone(), lora_act_out.clone()

    dense_output, dense_qout, dense_oscales, dense_lora_act = run(write_dense=True)
    skipped_output, skipped_qout, skipped_oscales, skipped_lora_act = run(write_dense=False)
    torch.cuda.synchronize()

  assert dense_output.shape == (batch_size, out_features)
  assert skipped_output.numel() == 0
  assert skipped_output.shape == (0, out_features)
  assert skipped_qout.abs().sum().item() > 0
  # qout/oscales are plain stores, so dropping the dense epilogue has to be bitwise invisible.
  assert torch.equal(skipped_qout, dense_qout)
  assert torch.equal(skipped_oscales, dense_oscales)
  # lora_act_out is accumulated with red.relaxed.gpu.global.add.f32 across block-N tiles, so it
  # is only bitwise stable while the grid is a single tile; compare with a tight tolerance
  # instead of betting on that coincidence.
  torch.testing.assert_close(skipped_lora_act, dense_lora_act, rtol=1e-6, atol=1e-6)


def test_svdquant_write_dense_false_rejects_unsupported_combinations() -> None:
  """Every combination the kernel cannot serve has to fail before it reaches the launch."""

  dtype = torch.bfloat16
  common = dict(
    act=torch.empty(256, 64, dtype=torch.uint8),
    wgt=torch.empty(128, 64, dtype=torch.uint8),
    ascales=torch.empty(64, 256, dtype=dtype),
    wscales=torch.empty(64, 128, dtype=dtype),
    fp4=False,
    write_dense=False,
  )
  qout = torch.empty(256, 64, dtype=torch.uint8)
  oscales = torch.empty(2, 256, dtype=dtype)
  smooth_factor = torch.empty(128, dtype=dtype)

  with pytest.raises(ValueError, match="requires qout, oscales and smooth_factor"):
    svdq_gemm_w4a4(**common)
  with pytest.raises(ValueError, match="only available on runtime='native'"):
    svdq_gemm_w4a4(**{**common, "fp4": True}, runtime="cute")
  with pytest.raises(ValueError, match="only dtype sources"):
    svdq_gemm_w4a4(**{
      **common, "fp4": True
    },
                   qout=qout,
                   oscales=oscales,
                   smooth_factor=smooth_factor)


@pytest.mark.parametrize("precision", ["int4", "nvfp4"])
def test_fused_gelu_mlp_matches_the_unfused_two_step(monkeypatch, precision: str) -> None:
  """The fused MLP pass must reproduce fc2(gelu(fc1(x))) and skip the dense store."""

  _require_svdquant_runtime()
  import cache_dit.quantization.svdquant.fused as fused_mod

  device = "cuda"
  dtype = runtime_dtype()
  in_features, hidden, out_features = 256, 512, 256
  rank, tokens = 32, 256

  torch.manual_seed(0)
  fc1_ref = nn.Linear(in_features, hidden, device=device, dtype=dtype)
  fc2_ref = nn.Linear(hidden, out_features, device=device, dtype=dtype)
  x = torch.randn(tokens, in_features, device=device, dtype=dtype)
  hidden_ref = F.gelu(fc1_ref(x), approximate="tanh")

  with torch.inference_mode():
    fc1 = quantize_linear_svdq_w4a4(fc1_ref,
                                    x,
                                    rank=rank,
                                    precision=precision,
                                    device=device,
                                    torch_dtype=dtype)
    fc2 = quantize_linear_svdq_w4a4(fc2_ref,
                                    hidden_ref,
                                    rank=rank,
                                    precision=precision,
                                    device=device,
                                    torch_dtype=dtype)

    expected = fc2(F.gelu(fc1(x), approximate="tanh"))

    dense_flags: list[bool] = []
    original = fused_mod.svdq_gemm_w4a4

    def spy(**kwargs):
      dense_flags.append(kwargs.get("write_dense", True))
      return original(**kwargs)

    monkeypatch.setattr(fused_mod, "svdq_gemm_w4a4", spy)
    fused = fused_mod.fused_gelu_mlp(x, fc1, fc2, pad_size=256)
    fused_mod.fused_gelu_proj(x, fc1, pad_size=256)
    torch.cuda.synchronize()

  got = fused.float()
  ref = expected.float()
  rel_l2 = (got - ref).norm() / ref.norm().clamp_min(1e-12)
  cosine = F.cosine_similarity(got.flatten(), ref.flatten(), dim=0)
  assert rel_l2 < 5e-3, f"{precision} fused MLP rel_l2={rel_l2.item():.3e}"
  assert cosine > 0.999, f"{precision} fused MLP cosine={cosine.item():.6f}"
  # Only the passes' own GEMM calls are intercepted: fused_gelu_mlp issues one fused fc1 GEMM
  # and then hands off to fc2.forward_quant (which resolves svdq_gemm_w4a4 from linear.py), while
  # fused_gelu_proj issues its own dense-output GEMM. So the first entry must skip the dense
  # store and the second must keep it.
  assert len(dense_flags) == 2
  assert dense_flags[0] is False
  assert dense_flags[1] is not False


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
        calibrate_precision="low",
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

  # Absolute accuracy gate on the whole INT4 path (weight pack + activation pack + GEMM +
  # LoRA). A monotonic trend alone survives a uniformly wrong result; a swapped
  # ascales/wscales or a dropped LoRA term lands far outside these bounds.
  assert metrics_by_rank[128].rel_l2 < 0.01
  assert metrics_by_rank[128].cosine > 0.9999
  assert metrics_by_rank[32].rel_l2 < 0.06
