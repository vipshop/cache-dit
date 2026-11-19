"""Cache-DiT's native online FP8 quantization backend."""

from .fused import fused_gelu_mlp
from .fused import fused_gelu_proj
from .fused import fused_swiglu_mlp
from .linear import QW8A8Linear
from .passes import apply_fused_mlp_passes
from .quantize import quantize_fp8

__all__ = [
  "QW8A8Linear",
  "apply_fused_mlp_passes",
  "fused_gelu_mlp",
  "fused_gelu_proj",
  "fused_swiglu_mlp",
  "quantize_fp8",
]
