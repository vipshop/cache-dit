"""Model-specific fused CUDA operators shipped by Cache-DiT."""

from ._boogu import boogu_fused_qk_norm_rope
from ._boogu import boogu_gated_rms_norm
from ._boogu import boogu_scaled_rms_norm
from ._boogu import models_extension_is_available
from ._boogu import models_get_load_error
from ._qwen_image21 import qwen_image21_gated_residual
from ._qwen_image21 import qwen_image21_ln_modulate
from ._qwen_image21 import qwen_image21_modulate_affine
from ._qwen_image21 import qwen_image21_qk_norm_rope
from ._qwen_image21 import qwen_image21_silu_mul

__all__ = [
  "models_extension_is_available",
  "models_get_load_error",
  "boogu_scaled_rms_norm",
  "boogu_gated_rms_norm",
  "boogu_fused_qk_norm_rope",
  "qwen_image21_qk_norm_rope",
  "qwen_image21_ln_modulate",
  "qwen_image21_modulate_affine",
  "qwen_image21_gated_residual",
  "qwen_image21_silu_mul",
]
