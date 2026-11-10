"""Public CUDA operator APIs provided by Cache-DiT."""

from ._attn import fused_merge_attn_states
from ._comm import fp8_comm_per_token_dequant
from ._comm import fp8_comm_per_token_quant
from ._comm import fp8_comm_qkv_permute_dequant
from ._comm import fp8_comm_qkv_permute_quant
from ._models import boogu_fused_qk_norm_rope
from ._models import boogu_gated_rms_norm
from ._models import boogu_scaled_rms_norm
from ._models import models_extension_is_available
from ._models import models_get_load_error
from ._svdquant import svdq_extension_is_available
from ._svdquant import svdq_gemm_w4a4
from ._svdquant import svdq_get_load_error
from ._svdquant import svdq_quantize_w4a4

__all__ = [
  "fp8_comm_per_token_quant",
  "fp8_comm_per_token_dequant",
  "fp8_comm_qkv_permute_quant",
  "fp8_comm_qkv_permute_dequant",
  "fused_merge_attn_states",
  "models_get_load_error",
  "models_extension_is_available",
  "boogu_scaled_rms_norm",
  "boogu_gated_rms_norm",
  "boogu_fused_qk_norm_rope",
  "svdq_get_load_error",
  "svdq_extension_is_available",
  "svdq_quantize_w4a4",
  "svdq_gemm_w4a4",
]
