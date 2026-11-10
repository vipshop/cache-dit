"""Pure-PyTorch RMSNorm used by the vendored Boogu-Image model.

Upstream Boogu-Image imports a Triton-fused ``RMSNorm`` when Triton is present.
Cache-DiT ships no Triton kernels, so the vendored model uses this module
instead. It keeps the upstream parameter layout (``weight`` only, no bias) and
exposes the ``zero_centered_weight`` flag that the fused CUDA path reads to tell
an RMSNorm apart from a ``nn.LayerNorm``.
"""

import torch
from torch import nn


class RMSNorm(nn.Module):
  """RMSNorm without any fused-kernel dependency.

  :param hidden_size: Channel count of the normalized dimension.
  :param eps: Numerical epsilon used by the RMS reduction.
  :param dropout_p: Dropout probability applied after normalization.
  :param zero_centered_weight: Initialize/interpret ``weight`` around zero.
  :param elementwise_affine: Whether to learn a per-channel ``weight``.
  :param device: Optional parameter device.
  :param dtype: Optional parameter dtype.
  """

  def __init__(
    self,
    hidden_size: int,
    eps: float = 1e-5,
    dropout_p: float = 0.0,
    zero_centered_weight: bool = False,
    elementwise_affine: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
  ) -> None:
    factory_kwargs = {"device": device, "dtype": dtype}
    super().__init__()

    self.eps = eps
    self.drop = nn.Dropout(dropout_p) if dropout_p > 0.0 else None
    self.zero_centered_weight = zero_centered_weight
    self.elementwise_affine = elementwise_affine
    if elementwise_affine:
      self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
    else:
      self.register_parameter("weight", None)
    self.register_parameter("bias", None)
    self.reset_parameters()

  def reset_parameters(self) -> None:
    if self.weight is None:
      return
    if self.zero_centered_weight:
      nn.init.zeros_(self.weight)
    else:
      nn.init.ones_(self.weight)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Normalize ``x`` over its last dimension.

    :param x: Input activations whose last dimension equals ``hidden_size``.
    :returns: Normalized activations with the input dtype preserved.
    """

    weight = self.weight
    if weight is not None and self.zero_centered_weight:
      weight = weight + 1.0
    out = torch.nn.functional.rms_norm(x, (x.shape[-1], ), weight, self.eps)
    if self.drop is not None:
      out = self.drop(out)
    return out
