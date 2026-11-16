"""Fused quantized MLP runtime for SVDQ W4A4 GEMM.

Provides ``fused_gelu_mlp`` (full fc1+GELU+fc2 fusion via qout path),
``fused_gelu_proj`` (fc1+GELU-only fusion returning fp16 output for use
in single-stream blocks where fc2 input is concatenated) and
``fused_swiglu_mlp`` (gated MLP whose gate is applied inside the activation
producer).

Every helper is runtime agnostic: it asks each layer for its own weight pack
through ``SVDQW4A4Linear.fused_gemm_args``, so the same chain runs on the native
GEMM and on the CuTe NVFP4 GEMM with either LoRA-down layout.
"""

from __future__ import annotations

import torch
from torch import nn

from ...kernels.svdquant import svdq_gemm_w4a4
from .linear import SVDQW4A4Linear


def _require_fusable(*modules: nn.Module) -> None:
  """Reject layers that cannot take part in a fused chain.

  The chain feeds the first GEMM's packed output straight into the second layer, so both have to
  live on a runtime that implements the fused epilogue and share one LoRA-down layout: the D the
  first GEMM writes is exactly what the second one reads back.

  :param modules: Quantized linear layers involved in the fused pass.
  :raises RuntimeError: If a layer uses a FlashInfer chain or the layouts disagree.
  """

  layouts = set()
  for module in modules:
    runtime = getattr(module, "runtime_kernel", "native")
    if runtime not in ("native", "cute"):
      raise RuntimeError(f"The fused MLP passes have no epilogue on runtime_kernel={runtime!r} on "
                         f"{type(module).__name__}; use 'native' or 'cute'.")
    layouts.add(getattr(module, "lora_layout", "native"))
  if len(layouts) > 1:
    raise RuntimeError("The fused MLP passes need one LoRA-down layout for the whole chain, got "
                       f"{sorted(layouts)}; the first GEMM writes the D the second one reads.")


@torch.compiler.disable
def fused_gelu_mlp(
  x: torch.Tensor,
  fc1: nn.Module | SVDQW4A4Linear,
  fc2: nn.Module | SVDQW4A4Linear,
  pad_size: int = 256,
) -> torch.Tensor:
  """Fused quantized MLP with GELU activation.

  Combines the first quantized linear layer, GELU activation, and the
  second quantized linear layer into a single CUDA kernel chain: the first
  GEMM applies GELU in-register and directly emits the 4-bit packed
  activation (plus its scales) consumed by the second GEMM, so the fp16
  GELU output never has to be re-quantized by a separate launch. Passing
  ``write_dense=False`` also drops the dense fp16 store, so this pass writes
  no full-size fp16 activation to HBM at all.

  :param x: Input tensor with shape ``[..., in_features]`` and dtype
      ``float16`` or ``bfloat16``.
  :param fc1: First quantized linear layer (``SVDQW4A4Linear``, input
      → hidden).
  :param fc2: Second quantized linear layer (``SVDQW4A4Linear``, hidden
      → output).
  :param pad_size: Batch padding size for CUDA kernel efficiency.
      Default is 256.

  :returns: Output tensor with shape ``[..., fc2.out_features]`` and the
      same dtype as ``x``.
  """
  _require_fusable(fc1, fc2)
  if x.ndim < 2:
    raise ValueError(f"input must have shape [..., in_features], got {tuple(x.shape)}.")

  *leading_shape, channels = x.shape
  token_count = 1
  for extent in leading_shape:
    token_count *= extent
  x = x.reshape(token_count, channels)

  quantized_x, ascales, lora_act = fc1.quantize(x, pad_size=pad_size)

  batch_size_pad = quantized_x.shape[0]
  fc1_out_padded = getattr(fc1, '_padded_out_features', fc1.out_features)

  qout_act, qout_ascales = _allocate_activation_pack(fc1, batch_size_pad, fc1_out_padded, x.dtype)
  # A rank-0 chain has no LoRA term at all, so the fused GEMM gets no down-projection either.
  qout_lora_act = _allocate_lora_act(fc2, batch_size_pad) if fc2.rank > 0 else None

  first = fc1.fused_gemm_args()
  second = fc2.fused_gemm_args()
  svdq_gemm_w4a4(
    act=quantized_x,
    wgt=first["wgt"],
    ascales=ascales,
    wscales=first["wscales"],
    qout=qout_act,
    oscales=qout_ascales,
    lora_act_in=lora_act,
    lora_up=first["lora_up"],
    lora_down=second["lora_down"],
    lora_act_out=qout_lora_act,
    bias=first["bias"],
    smooth_factor=second["smooth"],
    runtime=first["runtime"],
    lora_layout=first["lora_layout"],
    fp4=first["fp4"],
    alpha=first["alpha"],
    wcscales=first["wcscales"],
    write_dense=False,
  )

  output = fc2.forward_quant(qout_act, qout_ascales, qout_lora_act)
  output = output[:token_count]
  if hasattr(fc2, '_needs_n_strip') and fc2._needs_n_strip:
    output = output[:, :fc2.out_features]
  return output.reshape(*leading_shape, fc2.out_features)


@torch.compiler.disable
def fused_gelu_proj(
  x: torch.Tensor,
  fc1: nn.Module | SVDQW4A4Linear,
  pad_size: int = 256,
) -> torch.Tensor:
  """Fused fc1 GEMM + GELU in a single kernel.

  Feeds ``svdq_gemm_w4a4`` the GELU epilogue arguments to combine the first
  quantized linear layer and the GELU activation without a separate GELU
  kernel launch.

  Unlike ``fused_gelu_mlp``, this does **not** fuse the second linear
  layer — the caller is responsible for feeding the returned fp16
  tensor to ``fc2`` (e.g. after concatenation with attention output in
  single-stream blocks).

  :param x: Input tensor with shape ``[..., fc1.in_features]`` and
      dtype ``float16`` or ``bfloat16``.
  :param fc1: First quantized linear layer (``SVDQW4A4Linear``,
      input → hidden).
  :param pad_size: Batch padding size for CUDA kernel efficiency.
      Default is 256.

  :returns: GELU-activated fp16 output with shape
      ``[..., fc1.out_features]``.
  """
  _require_fusable(fc1)
  if x.ndim < 2:
    raise ValueError(f"input must have shape [..., in_features], got {tuple(x.shape)}.")

  *leading_shape, channels = x.shape
  token_count = 1
  for extent in leading_shape:
    token_count *= extent
  x = x.reshape(token_count, channels)

  quantized_x, ascales, lora_act = fc1.quantize(x, pad_size=pad_size)

  batch_size_pad = quantized_x.shape[0]
  fc1_out_padded = getattr(fc1, '_padded_out_features', fc1.out_features)

  # qout / oscales are required by the fused-GELU kernel branch but the
  # caller does not use them — the kernel still populates them.
  qout_act, qout_ascales = _allocate_activation_pack(fc1, batch_size_pad, fc1_out_padded, x.dtype)

  # smooth_factor must match fc1.out_features (output dim N),
  # not fc1.in_features (fc1.smooth_factor is for input quant).
  smooth_ones = torch.ones(
    fc1_out_padded,
    dtype=x.dtype,
    device=x.device,
  )

  first = fc1.fused_gemm_args()
  out = svdq_gemm_w4a4(
    act=quantized_x,
    wgt=first["wgt"],
    ascales=ascales,
    wscales=first["wscales"],
    qout=qout_act,
    oscales=qout_ascales,
    lora_act_in=lora_act,
    lora_up=first["lora_up"],
    bias=first["bias"],
    smooth_factor=smooth_ones,
    runtime=first["runtime"],
    lora_layout=first["lora_layout"],
    fp4=first["fp4"],
    alpha=first["alpha"],
    wcscales=first["wcscales"],
  )

  out = out[:token_count]
  if hasattr(fc1, '_needs_n_strip') and fc1._needs_n_strip:
    out = out[:, :fc1.out_features]
  return out.reshape(*leading_shape, fc1.out_features)


@torch.compiler.disable
def fused_swiglu_mlp(
  x: torch.Tensor,
  up: nn.Module | SVDQW4A4Linear,
  down: nn.Module | SVDQW4A4Linear,
  gate: nn.Module | SVDQW4A4Linear | None = None,
  pad_size: int = 256,
) -> torch.Tensor:
  """Fused quantized gated MLP: ``down(gate(x) * up(x))`` through one producer.

  The gate cannot be applied inside the GEMM epilogue, because the second operand comes from a
  different projection (or from a column range far wider than one output tile). It is therefore
  fused into the activation producer of ``down``: ``up`` (and ``gate`` when it is a separate
  layer) write dense BF16 activations, and the producer multiplies them in place while it packs
  the 4-bit activation and reduces the LoRA-down term. That removes the standalone
  ``silu``/multiply kernel and one full ``[tokens, hidden]`` HBM round trip.

  :param x: Input tensor with shape ``[..., up.in_features]``.
  :param up: Value projection. With ``gate=None`` it must produce ``2 * down.in_features``
      columns so the second half can serve as the gate.
  :param down: Output projection (hidden → output).
  :param gate: Optional separate gate projection with the same output width as ``up``.
  :param pad_size: Batch padding size for CUDA kernel efficiency. Default is 256.
  :returns: Output tensor with shape ``[..., down.out_features]``.
  :raises ValueError: If the widths do not describe a gated MLP.
  """
  _require_fusable(up, down) if gate is None else _require_fusable(up, gate, down)
  if x.ndim < 2:
    raise ValueError(f"input must have shape [..., in_features], got {tuple(x.shape)}.")

  *leading_shape, channels = x.shape
  token_count = 1
  for extent in leading_shape:
    token_count *= extent
  x = x.reshape(token_count, channels)

  hidden = down.in_features
  if gate is None:
    if up.out_features != 2 * hidden:
      raise ValueError(f"fused_swiglu_mlp expects up.out_features == 2 * down.in_features "
                       f"({2 * hidden}) when no separate gate is given, got {up.out_features}.")
  elif up.out_features != hidden or gate.out_features != hidden:
    raise ValueError(f"fused_swiglu_mlp expects up/gate out_features == down.in_features "
                     f"({hidden}), got {up.out_features} and {gate.out_features}.")

  value = up(x)
  if gate is None:
    quantized, ascales, lora_act = down.quantize(value, pad_size=pad_size, fuse_glu=True)
  else:
    quantized, ascales, lora_act = down.quantize(value,
                                                 pad_size=pad_size,
                                                 fuse_glu=True,
                                                 gate=gate(x))

  output = down.forward_quant(quantized, ascales, lora_act)
  output = output[:token_count]
  if hasattr(down, '_needs_n_strip') and down._needs_n_strip:
    output = output[:, :down.out_features]
  return output.reshape(*leading_shape, down.out_features)


def _allocate_activation_pack(
  layer: nn.Module | SVDQW4A4Linear,
  batch_size_pad: int,
  out_features_padded: int,
  dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Allocate the packed activation and its scales for the next layer.

  :param layer: Layer that will consume the pack; its precision selects the scale dtype.
  :param batch_size_pad: Padded token count.
  :param out_features_padded: Padded output width of the producing layer.
  :param dtype: Activation dtype, used by the INT4 scale buffer.
  :returns: ``(packed_activation, activation_scales)``.
  """

  device = layer.qweight.device
  qout = torch.empty(batch_size_pad, out_features_padded // 2, dtype=torch.uint8, device=device)
  if layer.precision == "nvfp4":
    oscales = torch.empty(out_features_padded // 16,
                          batch_size_pad,
                          dtype=torch.float8_e4m3fn,
                          device=device)
  else:
    oscales = torch.empty(out_features_padded // 64, batch_size_pad, dtype=dtype, device=device)
  return qout, oscales


def _allocate_lora_act(layer: nn.Module | SVDQW4A4Linear, batch_size_pad: int) -> torch.Tensor:
  """Allocate the LoRA-down activation buffer in the layout the layer expects.

  :param layer: Layer that will consume the buffer.
  :param batch_size_pad: Padded token count.
  :returns: FP32 fragment buffer for the native layout, BF16 row-major otherwise.
  """

  rank = getattr(layer, "rank", 0)
  dtype = torch.float32 if getattr(layer, "lora_layout", "native") == "native" else torch.bfloat16
  return torch.empty(batch_size_pad, rank, dtype=dtype, device=layer.qweight.device)


__all__ = ["fused_gelu_mlp", "fused_gelu_proj", "fused_swiglu_mlp"]
