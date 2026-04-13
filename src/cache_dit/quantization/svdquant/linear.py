# Adapted from nunchaku's implementation of SVDQW4A4Linear.
import torch
from torch import nn

from ...kernels import svdq_gemm_w4a4
from ...kernels import svdq_gemm_w4a4_v2
from ...kernels import svdq_quantize_w4a4_act_fuse_lora


class SVDQW4A4Linear(nn.Module):
  """Runtime module for packed SVDQ W4A4 linear layers.

  `SVDQW4A4Linear` is the module shape produced by the offline SVDQ
  quantizer. It stores the packed 4-bit weights consumed by
  `svdq_gemm_w4a4` or `svdq_gemm_w4a4_v2`, the per-group scales and
  activation smoothing factors needed to quantize activations at runtime,
  and the optional low-rank residual factors fused around the packed GEMM
  path. `qweight` stores two 4-bit weights per `int8` lane, `wscales`
  stores the packed weight-scale hierarchy, and `proj_down` / `proj_up`
  carry the low-rank correction when `rank > 0`.
  """

  def __init__(
    self,
    in_features: int,
    out_features: int,
    rank: int = 32,
    bias: bool = True,
    precision: str = "int4",
    act_unsigned: bool = False,
    runtime_kernel: str = "v1",
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device | None = None,
  ) -> None:
    """Initialize the packed runtime module.

    :param in_features: Logical input width of the original floating-point
      linear layer. Must be divisible by 2 so two 4-bit weights can be packed
      into one byte. INT4 checkpoints produced by the SVDQ quantizer also use
      64-element scale groups.
    :param out_features: Logical output width of the original linear layer.
    :param rank: Rank of the low-rank residual correction. Use 0 to disable
      the LoRA-style correction path.
    :param bias: Whether to allocate a bias parameter for the packed layer.
    :param precision: Packed weight format. `int4` uses signed INT4 weights
      with 64-element groups, while `nvfp4` uses the NVFP4 path with
      16-element groups and an additional coarse scale hierarchy.
    :param act_unsigned: Whether the runtime activation quantizer should emit
      unsigned 4-bit activations for the kernel path.
    :param runtime_kernel: Runtime packed GEMM implementation. `v1` preserves
      the original kernel path, while `v2` routes the same packed tensors
      through the dedicated plain-path v2 entrypoint.
    :param torch_dtype: Floating-point dtype used for scales, smoothing
      factors, bias, and low-rank tensors.
    :param device: Device where the packed parameters are allocated. Defaults
      to CPU when omitted.
    """

    super().__init__()
    if device is None:
      device = torch.device("cpu")
    if rank < 0:
      raise ValueError(f"rank must be non-negative, got {rank}.")
    if in_features % 2 != 0:
      raise ValueError(f"in_features must be divisible by 2, got {in_features}.")
    if runtime_kernel not in {"v1", "v2"}:
      raise ValueError(f"runtime_kernel must be 'v1' or 'v2', got {runtime_kernel!r}.")

    self.in_features = in_features
    self.out_features = out_features
    self.rank = rank
    self.precision = precision
    self.torch_dtype = torch_dtype
    self.runtime_kernel = runtime_kernel

    if precision == "nvfp4":
      self.group_size = 16
    elif precision == "int4":
      self.group_size = 64
    else:
      raise ValueError(f"Invalid precision: {precision}")
    if in_features % self.group_size != 0:
      raise ValueError(
        f"in_features must be divisible by group_size={self.group_size} for {precision}, got {in_features}."
      )

    self.qweight = nn.Parameter(
      torch.empty(out_features, in_features // 2, dtype=torch.int8, device=device),
      requires_grad=False,
    )
    self.bias = (nn.Parameter(torch.empty(out_features, dtype=torch_dtype, device=device),
                              requires_grad=True) if bias else None)
    self.wscales = nn.Parameter(
      torch.empty(
        in_features // self.group_size,
        out_features,
        dtype=torch_dtype if precision == "int4" else torch.float8_e4m3fn,
        device=device,
      ),
      requires_grad=False,
    )
    self.smooth_factor = nn.Parameter(
      torch.empty(in_features, dtype=torch_dtype, device=device),
      requires_grad=False,
    )
    self.smooth_factor_orig = nn.Parameter(
      torch.empty(in_features, dtype=torch_dtype, device=device),
      requires_grad=False,
    )
    self.proj_down = nn.Parameter(torch.empty(in_features, rank, dtype=torch_dtype, device=device))
    self.proj_up = nn.Parameter(torch.empty(out_features, rank, dtype=torch_dtype, device=device))

    if precision == "nvfp4":
      self.wcscales = nn.Parameter(
        torch.ones(out_features, dtype=torch_dtype, device=device),
        requires_grad=False,
      )
      self.wtscale = 1.0
    else:
      self.wtscale = None
      self.wcscales = None

    self.act_unsigned = act_unsigned

  @classmethod
  def from_linear(cls, linear: nn.Linear, **kwargs) -> "SVDQW4A4Linear":
    """Allocate an `SVDQW4A4Linear` shell from a float `nn.Linear`.

    The returned module matches the source layer's logical geometry, bias
    presence, dtype, and device unless explicitly overridden via `kwargs`.
    This helper only allocates the target module shape; it does not quantize
    or copy the source weights by itself.

    :param linear: Source floating-point linear layer.
    :param kwargs: Additional keyword arguments forwarded to the underlying
      implementation.
    :returns: An uninitialized `SVDQW4A4Linear` shell matching the source
      layer geometry.
    """

    in_features = kwargs.pop("in_features", linear.in_features)
    torch_dtype = kwargs.pop("torch_dtype", linear.weight.dtype)
    device = kwargs.pop("device", linear.weight.device)
    return cls(
      in_features=in_features,
      out_features=linear.out_features,
      bias=linear.bias is not None,
      torch_dtype=torch_dtype,
      device=device,
      **kwargs,
    )

  def forward(self, x: torch.Tensor, output: torch.Tensor | None = None) -> torch.Tensor:
    """Quantize activations, run the packed W4A4 GEMM, and restore the input rank.

    :param x: Input activations with shape `[..., in_features]`.
    :param output: Optional destination buffer. The method accepts either the
      logical `[..., out_features]` shape or the padded 2D shape produced by
      the fused activation quantizer.
    :returns: The output activations with shape `[..., out_features]`.
    """

    if x.ndim < 2:
      raise ValueError(f"input must have shape [..., in_features], got {tuple(x.shape)}.")

    *leading_shape, channels = x.shape
    token_count = 1
    for extent in leading_shape:
      token_count *= extent
    x = x.reshape(token_count, channels)
    quantized_x, ascales, lora_act_out = self.quantize(x)
    use_direct_output = output is not None and output.shape == (
      quantized_x.shape[0],
      self.out_features,
    )
    padded_output = self.forward_quant(
      quantized_x,
      ascales,
      lora_act_out,
      output if use_direct_output else None,
    )

    logical_output = padded_output[:token_count]
    if output is not None and not use_direct_output:
      expected_shape = (*leading_shape, self.out_features)
      if output.shape != expected_shape:
        raise ValueError("output must have shape "
                         f"{expected_shape}, got {tuple(output.shape)}.")
      reshaped_output = output.reshape(token_count, self.out_features)
      reshaped_output.copy_(logical_output)
      logical_output = reshaped_output

    return logical_output.reshape(*leading_shape, self.out_features)

  def quantize(self,
               x: torch.Tensor,
               pad_size: int = 256) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize activations and compute the low-rank down-projection input.

    :param x: Flattened activations with shape `[tokens, in_features]`.
    :param pad_size: Token padding multiple required by the fused activation
      quantizer.
    :returns: A tuple `(quantized_x, ascales, lora_act_out)` ready for
      `forward_quant`.
    """

    return svdq_quantize_w4a4_act_fuse_lora(
      input=x,
      lora_down=self.proj_down,
      smooth=self.smooth_factor,
      fp4=self.precision == "nvfp4",
      pad_size=pad_size,
    )

  def forward_quant(
    self,
    quantized_x: torch.Tensor,
    ascales: torch.Tensor,
    lora_act: torch.Tensor,
    output: torch.Tensor | None = None,
  ) -> torch.Tensor:
    """Run the packed SVDQ GEMM from pre-quantized activation tensors.

    :param quantized_x: Packed activation tensor returned by `quantize`.
    :param ascales: Per-token activation scales returned by `quantize`.
    :param lora_act: Low-rank activation factors returned by `quantize`.
    :param output: Optional destination buffer. When provided, the result is
      copied into this tensor and returned.
    :returns: The padded 2D output tensor produced by the selected packed
      GEMM runtime kernel.
    """

    gemm_kwargs = dict(
      act=quantized_x,
      wgt=self.qweight,
      ascales=ascales,
      wscales=self.wscales,
      bias=self.bias,
      fp4=self.precision == "nvfp4",
      alpha=self.wtscale,
      wcscales=self.wcscales,
      act_unsigned=self.act_unsigned,
      output_dtype=output.dtype if output is not None else self.proj_up.dtype,
    )
    if self.rank > 0:
      gemm_kwargs["lora_act_in"] = lora_act
      gemm_kwargs["lora_up"] = self.proj_up

    if self.runtime_kernel == "v2":
      gemm_op = svdq_gemm_w4a4_v2
    else:
      gemm_op = svdq_gemm_w4a4
    result = gemm_op(**gemm_kwargs)
    if output is not None:
      output.copy_(result)
      return output
    return result

  def __repr__(self) -> str:
    return (f"SVDQW4A4Linear(in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, precision={self.precision}, act_unsigned={self.act_unsigned}, "
            f"runtime_kernel={self.runtime_kernel})")


__all__ = ["SVDQW4A4Linear"]
