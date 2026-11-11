"""Public CUDA operator APIs provided by Cache-DiT."""

from .attn import fused_merge_attn_states
from .comm import fp8_comm_per_token_dequant
from .comm import fp8_comm_per_token_quant
from .comm import fp8_comm_qkv_permute_dequant
from .comm import fp8_comm_qkv_permute_quant
from .cuda_ext import cuda_ext_get_load_error
from .cuda_ext import cuda_ext_is_available
from .models import boogu_fused_qk_norm_rope
from .models import boogu_gated_rms_norm
from .models import boogu_scaled_rms_norm
from .models import flux2_klein_gated_residual
from .models import flux2_klein_ln_modulate
from .models import flux2_klein_qk_norm_rope
from .models import flux2_klein_qk_norm_rope_join
from .models import flux2_klein_residual_ln_modulate
from .models import flux2_klein_silu_mul
from .models import qwen_image21_gated_residual
from .models import qwen_image21_ln_modulate
from .models import flux3_klein_gated_residual
from .models import flux3_klein_gated_residual_split
from .models import flux3_klein_ln_modulate
from .models import flux3_klein_ln_modulate_cat
from .models import flux3_klein_qk_norm_rope
from .models import qwen_image21_qk_norm_rope
from .models import qwen_image21_silu_mul
from .svdquant import svdq_gemm_w4a4
from .svdquant import svdq_quantize_w4a4

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
  "flux2_klein_qk_norm_rope",
  "flux2_klein_qk_norm_rope_join",
  "flux2_klein_ln_modulate",
  "flux2_klein_residual_ln_modulate",
  "flux2_klein_gated_residual",
  "flux2_klein_silu_mul",
  "qwen_image21_qk_norm_rope",
  "qwen_image21_ln_modulate",
  "qwen_image21_gated_residual",
  "qwen_image21_silu_mul",
  "flux3_klein_qk_norm_rope",
  "flux3_klein_ln_modulate",
  "flux3_klein_ln_modulate_cat",
  "flux3_klein_gated_residual",
  "flux3_klein_gated_residual_split",
  "svdq_quantize_w4a4",
  "svdq_gemm_w4a4",
]
