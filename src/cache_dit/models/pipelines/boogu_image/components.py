"""PyTorch SwiGLU used when flash_attn's fused activation is unavailable.

Adapted from Boogu-Image (`boogu/models/transformers/components.py`), Apache-2.0.
"""

import torch.nn.functional as F


def swiglu(x, y):
  return F.silu(x.float(), inplace=False).to(x.dtype) * y
