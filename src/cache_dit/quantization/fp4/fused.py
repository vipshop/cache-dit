"""Fused runtime helpers for cache-dit's native online FP4 backend."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ...kernels import fp4_quantize_swiglu
from .linear import NVFP4Linear


def _gelu_activation(approximate: str) -> str:
  """Map a diffusers GELU approximation selector to the native EVT name."""

  if approximate == "none":
    return "gelu"
  if approximate == "tanh":
    return "gelu_tanh"
  raise ValueError(f"Unsupported GELU approximation for native FP4: {approximate!r}.")


def fused_gelu_mlp(
  x: torch.Tensor,
  fc1: NVFP4Linear,
  fc2: NVFP4Linear,
  approximate: str = "tanh",
) -> torch.Tensor:
  """Run ``fc2(gelu(fc1(x)))`` with the first GEMM's GELU epilogue.

  FP4 still needs the second layer's row amax and per-16 block scales after
  activation, so its original quantization recipe remains in place.

  :param x: BF16 input activation.
  :param fc1: Quantized input-to-hidden projection.
  :param fc2: Quantized hidden-to-output projection.
  :param approximate: ``none`` for exact GELU or ``tanh`` for the approximation.
  :returns: BF16 MLP output.
  """

  return fc2(fc1(x, activation=_gelu_activation(approximate)))


def fused_gelu_proj(
  x: torch.Tensor,
  fc1: NVFP4Linear,
  approximate: str = "tanh",
) -> torch.Tensor:
  """Run a quantized projection with an exact or tanh GELU epilogue.

  :param x: BF16 input activation.
  :param fc1: Quantized projection.
  :param approximate: ``none`` for exact GELU or ``tanh`` for the approximation.
  :returns: BF16 activated projection output.
  """

  return NVFP4Linear.forward(fc1, x, activation=_gelu_activation(approximate))


def fused_swiglu_mlp(
  x: torch.Tensor,
  up: NVFP4Linear,
  down: NVFP4Linear,
  gate: NVFP4Linear | None = None,
) -> torch.Tensor:
  """Run ``down(up(x) * silu(gate(x)))`` with gating inside the FP4 producer.

  :param x: BF16 input activation.
  :param up: Value projection, producing either H or a half-split 2H tensor.
  :param down: Quantized output projection.
  :param gate: Optional separate gate projection.
  :returns: BF16 gated MLP output.
  """

  hidden = down.in_features
  if gate is None and up.out_features != 2 * hidden:
    raise ValueError("Half-split FP4 SwiGLU expects up.out_features == 2 * down.in_features.")
  if gate is not None and (up.out_features != hidden or gate.out_features != hidden):
    raise ValueError("Separate-gate FP4 SwiGLU expects up/gate output width == down.in_features.")

  value = up(x).reshape(-1, up.out_features)
  gate_value = gate(x).reshape(-1, hidden) if gate is not None else None
  padded_hidden = down.quantized_in_features
  if padded_hidden != hidden:
    if gate_value is None:
      value, gate_value = value.chunk(2, dim=-1)
    value = F.pad(value, (0, padded_hidden - hidden))
    gate_value = F.pad(gate_value, (0, padded_hidden - hidden))

  qx, sx, gx = fp4_quantize_swiglu(value, gate_value)
  return down.forward_quant(qx, sx, gx, x.shape[:-1])


__all__ = ["fused_gelu_mlp", "fused_gelu_proj", "fused_swiglu_mlp"]
