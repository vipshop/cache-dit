"""Model-specific fused CUDA operators shipped by Cache-DiT."""

from .boogu import boogu_fused_qk_norm_rope
from .boogu import boogu_gated_rms_norm
from .boogu import boogu_scaled_rms_norm
from .flux2_klein import flux2_klein_gated_residual
from .flux2_klein import flux2_klein_ln_modulate
from .flux2_klein import flux2_klein_qk_norm_rope
from .flux2_klein import flux2_klein_qk_norm_rope_join
from .flux2_klein import flux2_klein_residual_ln_modulate
from .flux2_klein import flux2_klein_silu_mul
from .flux3_klein import flux3_klein_gated_residual
from .flux3_klein import flux3_klein_gated_residual_split
from .flux3_klein import flux3_klein_ln_modulate
from .flux3_klein import flux3_klein_ln_modulate_cat
from .flux3_klein import flux3_klein_qk_norm_rope
from .qwen_image21 import qwen_image21_gated_residual
from .qwen_image21 import qwen_image21_ln_modulate
from .qwen_image21 import qwen_image21_qk_norm_rope
from .qwen_image21 import qwen_image21_silu_mul

__all__ = [
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
]
