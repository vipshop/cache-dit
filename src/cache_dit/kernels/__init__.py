"""Public CUDA operator APIs provided by Cache-DiT."""

from ._attn import fused_merge_attn_states
from ._comm import fp8_comm_per_token_dequant
from ._comm import fp8_comm_per_token_quant
from ._comm import fp8_comm_qkv_permute_dequant
from ._comm import fp8_comm_qkv_permute_quant
from ._cuda_ext import cuda_ext_get_load_error
from ._cuda_ext import cuda_ext_is_available
from ._models import boogu_fused_qk_norm_rope
from ._models import boogu_gated_rms_norm
from ._models import boogu_scaled_rms_norm
from ._models import qwen_image21_gated_residual
from ._models import qwen_image21_ln_modulate
from ._models import qwen_image21_qk_norm_rope
from ._models import qwen_image21_silu_mul
from ._svdquant import svdq_gemm_w4a4
from ._svdquant import svdq_quantize_w4a4

__all__ = [
  "fp8_comm_per_token_quant",
  "fp8_comm_per_token_dequant",
  "fp8_comm_qkv_permute_quant",
  "fp8_comm_qkv_permute_dequant",
  "fused_merge_attn_states",
  "cuda_ext_is_available",
  "cuda_ext_get_load_error",
  "boogu_scaled_rms_norm",
  "boogu_gated_rms_norm",
  "boogu_fused_qk_norm_rope",
  "qwen_image21_qk_norm_rope",
  "qwen_image21_ln_modulate",
  "qwen_image21_gated_residual",
  "qwen_image21_silu_mul",
  "svdq_quantize_w4a4",
  "svdq_gemm_w4a4",
]
