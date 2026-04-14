"""CuTe DSL entrypoints for the in-progress SVDQ runtime v3 path.

The first implementation pass keeps the public packed-tensor contract stable while the actual CuTe
DSL kernels are being built out. The exported wrappers already sit on the final call path used by
`runtime_kernel="v3"`, so later kernel work can replace the compatibility fallback without reworking
the PTQ integration layer.
"""

from .gemm_w4a4_v2 import svdq_gemm_w4a4_v2
from .quantize_w4a4_act_fuse_lora import svdq_quantize_w4a4_act_fuse_lora

__all__ = [
  "svdq_gemm_w4a4_v2",
  "svdq_quantize_w4a4_act_fuse_lora",
]
