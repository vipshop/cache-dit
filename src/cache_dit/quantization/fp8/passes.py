"""Structural fused-MLP passes for cache-dit's native FP8 layers."""

from __future__ import annotations

import types

from torch import nn

from .linear import QW8A8Linear


def _patch_gelu_mlp(module: nn.Module) -> int:
  """Patch standard diffusers FeedForward blocks with plain GELU."""

  from .fused import fused_gelu_mlp

  count = 0
  for _, parent in module.named_modules():
    if parent.__class__.__name__ != "FeedForward" or hasattr(parent, "_fp8_original_forward"):
      continue
    net = getattr(parent, "net", None)
    if not isinstance(net, nn.ModuleList) or len(net) < 3:
      continue
    activation = net[0]
    fc1 = getattr(activation, "proj", None)
    fc2 = net[2]
    if (activation.__class__.__name__ != "GELU" or not isinstance(fc1, QW8A8Linear)
        or not isinstance(fc2, QW8A8Linear)):
      continue
    approximate = getattr(activation, "approximate", "none")
    parent._fp8_original_forward = parent.forward

    def _forward(self,
                 hidden_states,
                 *args,
                 _fc1=fc1,
                 _fc2=fc2,
                 _approximate=approximate,
                 **kwargs):
      return fused_gelu_mlp(hidden_states, _fc1, _fc2, approximate=_approximate)

    parent.forward = types.MethodType(_forward, parent)
    count += 1
  return count


def _patch_gelu_projection(module: nn.Module) -> int:
  """Patch single-stream ``proj_mlp -> GELU -> concat`` blocks."""

  from .fused import fused_gelu_proj

  count = 0
  for _, parent in module.named_modules():
    if parent.__class__.__name__ == "FeedForward":
      continue
    gelu = None
    layers = []
    for child in parent.children():
      if isinstance(child, nn.GELU):
        gelu = child
      if isinstance(child, QW8A8Linear):
        layers.append(child)
    if gelu is None or len(layers) < 2 or hasattr(gelu, "_fp8_original_forward"):
      continue

    pair = None
    for fc1 in layers:
      for fc2 in layers:
        if (fc1 is not fc2 and fc1.out_features > fc1.in_features
            and fc2.in_features > fc2.out_features
            and fc2.in_features == fc1.in_features + fc1.out_features):
          pair = (fc1, fc2)
          break
      if pair is not None:
        break
    if pair is None:
      continue
    fc1, _ = pair
    approximate = gelu.approximate
    fc1._fp8_original_forward = fc1.forward
    gelu._fp8_original_forward = gelu.forward

    def _fc1_forward(self, x, activation=None, _approximate=approximate):
      return fused_gelu_proj(x, self, approximate=_approximate)

    def _identity(self, x):
      return x

    fc1.forward = types.MethodType(_fc1_forward, fc1)
    gelu.forward = types.MethodType(_identity, gelu)
    count += 1
  return count


def _detect_swiglu(parent: nn.Module) -> tuple[QW8A8Linear, QW8A8Linear | None, QW8A8Linear] | None:
  """Return up/gate/down for separate-gate or diffusers half-split SwiGLU."""

  up = getattr(parent, "proj", None)
  gate = getattr(parent, "gate_layer", None)
  down = getattr(parent, "out", None)
  if (all(isinstance(layer, QW8A8Linear) for layer in (up, gate, down))
      and isinstance(getattr(parent, "activation_fn", None), nn.SiLU)):
    return up, gate, down

  if parent.__class__.__name__ != "FeedForward":
    return None
  net = getattr(parent, "net", None)
  if not isinstance(net, nn.ModuleList) or len(net) < 3:
    return None
  activation = net[0]
  if activation.__class__.__name__ != "SwiGLU":
    return None
  up = getattr(activation, "proj", None)
  down = net[2]
  if isinstance(up, QW8A8Linear) and isinstance(down, QW8A8Linear):
    return up, None, down
  return None


def _patch_swiglu(module: nn.Module) -> int:
  """Patch supported gated MLPs to quantize the gated product directly."""

  from .fused import fused_swiglu_mlp

  count = 0
  for _, parent in module.named_modules():
    if hasattr(parent, "_fp8_original_forward"):
      continue
    target = _detect_swiglu(parent)
    if target is None:
      continue
    up, gate, down = target
    parent._fp8_original_forward = parent.forward

    def _forward(self, hidden_states, *args, _up=up, _gate=gate, _down=down, **kwargs):
      return fused_swiglu_mlp(hidden_states, _up, _down, gate=_gate)

    parent.forward = types.MethodType(_forward, parent)
    count += 1
  return count


def apply_fused_mlp_passes(module: nn.Module) -> dict[str, int]:
  """Install every supported native FP8 fused-MLP pass and return target counts.

  :param module: Root module whose Linear descendants are already quantized.
  :returns: Target counts keyed by ``gelu_mlp``, ``gelu_proj`` and ``swiglu``.
  """

  return {
    "gelu_mlp": _patch_gelu_mlp(module),
    "gelu_proj": _patch_gelu_projection(module),
    "swiglu": _patch_swiglu(module),
  }


__all__ = ["apply_fused_mlp_passes"]
