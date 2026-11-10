# Adapted from nunchaku's implementation of SVDQW4A4Linear.
import torch
import torch.nn.functional as F
from torch import nn

from ...kernels import svdq_gemm_w4a4
from ...kernels import svdq_quantize_w4a4


class SVDQW4A4Linear(nn.Module):
  """Runtime module for packed SVDQ W4A4 linear layers.

  `SVDQW4A4Linear` is the module shape produced by the offline SVDQ
  quantizer. It stores the packed 4-bit weights consumed by the native or CuTe
  GEMM family, the per-group scales and activation smoothing factors needed at runtime,
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
    runtime_kernel: str | None = None,
    lora_layout: str | None = None,
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
    :param runtime_kernel: Packed GEMM family. `None` selects the plain native
      INT4 kernel for INT4 and the native ext kernel for NVFP4. `cute` is an
      NVFP4-only validation opt-in; it is not enabled by default until its GEMM
      and end-to-end performance meet the release gate.
    :param lora_layout: Layout of the LoRA-down term produced by the
      activation quantizer and consumed by the GEMM. With `runtime_kernel="cute"`,
      `None` defaults to `row_major`, the row-major BF16 `D` emitted by the CuTe
      producer. Other runtimes default to the packed FP32 MMA-fragment `D`.
      Explicit `native` preserves the fragment-D contract with either runtime;
      explicit `row_major` requires `runtime_kernel="cute"`.
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
    if runtime_kernel is not None and runtime_kernel not in {"native", "cute"}:
      raise ValueError("runtime_kernel must be 'native', 'cute', or None; "
                       f"got {runtime_kernel!r}.")
    resolved_runtime_kernel = runtime_kernel
    if resolved_runtime_kernel is None:
      resolved_runtime_kernel = "native"
    if resolved_runtime_kernel == "cute" and precision != "nvfp4":
      raise ValueError("runtime_kernel='cute' is only supported for precision='nvfp4'.")
    if resolved_runtime_kernel == "cute" and torch_dtype != torch.bfloat16:
      raise ValueError("runtime_kernel='cute' currently requires torch_dtype=torch.bfloat16.")
    if lora_layout is not None and lora_layout not in {"native", "row_major"}:
      raise ValueError("lora_layout must be 'native', 'row_major', or None; "
                       f"got {lora_layout!r}.")
    resolved_lora_layout = lora_layout
    if resolved_lora_layout is None:
      resolved_lora_layout = "row_major" if resolved_runtime_kernel == "cute" else "native"
    if resolved_lora_layout == "row_major" and resolved_runtime_kernel != "cute":
      raise ValueError("lora_layout='row_major' is only supported with runtime_kernel='cute'; "
                       "it selects the CuTe producer's row-major BF16 D.")

    if precision == "nvfp4":
      self.group_size = 16
    elif precision == "int4":
      self.group_size = 64
    else:
      raise ValueError(f"Invalid precision: {precision}")

    self.in_features = in_features
    self.out_features = out_features
    self.rank = rank
    self.precision = precision
    self.torch_dtype = torch_dtype
    self.runtime_kernel = resolved_runtime_kernel
    self.lora_layout = resolved_lora_layout
    self._cute_pack_ready = False
    self._cute_pack_version = 1
    self._cute_qweight = None
    self._cute_wscales = None
    self._cute_proj_up = None
    self._cute_proj_down = None
    self._cute_smooth = None
    self._cute_bias = None
    self._cute_wcscales = None

    padded_in = ((in_features + 127) // 128) * 128
    padded_out = ((out_features + 127) // 128) * 128
    self._padded_in_features = padded_in
    self._padded_out_features = padded_out
    self._needs_k_pad = padded_in != in_features
    self._needs_n_strip = padded_out != out_features

    if in_features % 2 != 0:
      raise ValueError(f"in_features must be divisible by 2, got {in_features}.")
    if padded_in % self.group_size != 0:
      raise ValueError(
        f"Padded in_features ({padded_in}) must be divisible by group_size={self.group_size} "
        f"for {precision}, got logical in_features={in_features}.")

    self.qweight = nn.Parameter(
      torch.empty(padded_out, padded_in // 2, dtype=torch.int8, device=device),
      requires_grad=False,
    )
    self.bias = (nn.Parameter(torch.empty(padded_out, dtype=torch_dtype, device=device),
                              requires_grad=True) if bias else None)
    self.wscales = nn.Parameter(
      torch.empty(
        padded_in // self.group_size,
        padded_out,
        dtype=torch_dtype if precision == "int4" else torch.float8_e4m3fn,
        device=device,
      ),
      requires_grad=False,
    )
    self.smooth_factor = nn.Parameter(
      torch.empty(padded_in, dtype=torch_dtype, device=device),
      requires_grad=False,
    )
    self.smooth_factor_orig = nn.Parameter(
      torch.empty(padded_in, dtype=torch_dtype, device=device),
      requires_grad=False,
    )
    self.proj_down = nn.Parameter(torch.empty(padded_in, rank, dtype=torch_dtype, device=device))
    self.proj_up = nn.Parameter(torch.empty(padded_out, rank, dtype=torch_dtype, device=device))

    if precision == "nvfp4":
      self.wcscales = nn.Parameter(
        torch.ones(padded_out, dtype=torch_dtype, device=device),
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

    return self._forward_plain(x, output)

  def _forward_plain(self, x: torch.Tensor, output: torch.Tensor | None = None) -> torch.Tensor:
    """Original forward logic without DTensor awareness (used when TP is disabled)."""

    if x.ndim < 2:
      raise ValueError(f"input must have shape [..., in_features], got {tuple(x.shape)}.")

    *leading_shape, channels = x.shape
    token_count = 1
    for extent in leading_shape:
      token_count *= extent
    x = x.reshape(token_count, channels)

    if self._needs_k_pad:
      x = F.pad(x, (0, self._padded_in_features - self.in_features))

    quantized_x, ascales, lora_act_out = self.quantize(x)
    use_direct_output = output is not None and output.shape == (
      quantized_x.shape[0],
      self._padded_out_features,
    )
    padded_output = self.forward_quant(
      quantized_x,
      ascales,
      lora_act_out,
      output if use_direct_output else None,
    )

    logical_output = padded_output[:token_count]
    if self._needs_n_strip:
      logical_output = logical_output[:, :self.out_features]

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

    if self.lora_layout == "row_major":
      # The CuTe row-major producer consumes the converted pack (`_cute_proj_down` in row-major
      # layout and the unpacked `_cute_smooth` vector) and folds the per-rank scales into the
      # emitted BF16 D. The pack is prepared once at load time, but defensive rebuild keeps direct
      # constructor use working.
      if not self._cute_pack_ready:
        self._prepare_cute_pack()
      return svdq_quantize_w4a4(
        input=x,
        lora_down=self._cute_proj_down if self.rank > 0 else None,
        smooth=self._cute_smooth,
        lora_layout="row_major",
        fp4=True,
        pad_size=pad_size,
      )

    # The native producer owns the fast K-split LoRA-down reduction. Its packed FP32 fragment D is
    # consumed in place by both runtimes, so no BF16 row-major D conversion is needed. Weight
    # packing for the CuTe runtime is prepared separately in forward_quant().
    return svdq_quantize_w4a4(
      input=x,
      lora_down=self.proj_down,
      smooth=self.smooth_factor,
      lora_layout="native",
      fp4=self.precision == "nvfp4",
      pad_size=pad_size,
    )

  def _load_from_state_dict(self, *args, **kwargs) -> None:
    super()._load_from_state_dict(*args, **kwargs)
    # The CuTe pack is a pure function of the loaded checkpoint, so it is rebuilt here, once per
    # load, and never on the forward path.
    self._cute_pack_ready = False
    if self.runtime_kernel == "cute":
      self._prepare_cute_pack()

  def _apply(self, fn, *args, **kwargs):
    """Rebuild the cached CuTe pack across device or dtype conversions.

    The `_cute_*` tensors are plain attributes, so `Module.to`/`.cuda()` would otherwise leave a
    stale pack behind on the previous device.
    """

    rebuild = self._cute_pack_ready
    self._cute_pack_ready = False
    module = super()._apply(fn, *args, **kwargs)
    if rebuild and self.runtime_kernel == "cute":
      self._prepare_cute_pack()
    return module

  def _prepare_cute_pack(self) -> None:
    """Cache CuTe runtime tensors without mutating the native checkpoint parameters.

    The original native parameters remain the only state-dict entries, so saving and reloading
    never serializes the converted layout. The conversion runs once from `_load_from_state_dict`
    and again after any device or dtype move; it is never part of the forward path. The cache is
    private implementation data, not a `nn.Parameter` or registered buffer.

    :returns: None. The converted tensors are stored on `_cute_*` attributes with pack version 1.
    """
    if self._cute_pack_ready:
      return

    from .packing import convert_svdq_nvfp4_pack_to_cute

    (self._cute_qweight, self._cute_wscales, self._cute_proj_up, self._cute_proj_down,
     self._cute_bias, self._cute_wcscales,
     self._cute_smooth) = convert_svdq_nvfp4_pack_to_cute(self.qweight, self.wscales, self.proj_up,
                                                          self.proj_down, self.bias, self.wcscales,
                                                          self.smooth_factor)
    self._cute_pack_ready = True

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

    if self.runtime_kernel == "cute":
      if not self._cute_pack_ready:
        raise RuntimeError(
          "The CuTe weight pack is not built. Load the module with `load_state_dict` (or call "
          "`_prepare_cute_pack`) before running the CuTe runtime.")
      result = svdq_gemm_w4a4(
        act=quantized_x,
        wgt=self._cute_qweight,
        ascales=ascales,
        wscales=self._cute_wscales,
        lora_act_in=lora_act if self.rank > 0 else None,
        lora_up=self._cute_proj_up if self.rank > 0 else None,
        bias=self._cute_bias,
        wcscales=self._cute_wcscales,
        runtime="cute",
        lora_layout=self.lora_layout,
        fp4=True,
        alpha=self.wtscale,
      )
      if output is not None:
        output.copy_(result)
        return output
      return result

    if self.lora_layout != "native":
      raise RuntimeError("runtime_kernel='native' only supports lora_layout='native', got "
                         f"{self.lora_layout!r}; use runtime_kernel='cute' for row-major D.")

    gemm_kwargs = dict(
      act=quantized_x,
      wgt=self.qweight,
      ascales=ascales,
      wscales=self.wscales,
      bias=self.bias,
      runtime="native",
      lora_layout="native",
      fp4=self.precision == "nvfp4",
      alpha=self.wtscale,
      wcscales=self.wcscales,
      act_unsigned=self.act_unsigned,
      output_dtype=output.dtype if output is not None else self.proj_up.dtype,
    )
    gemm_kwargs["lora_act_in"] = lora_act if self.rank > 0 else None
    gemm_kwargs["lora_up"] = self.proj_up if self.rank > 0 else None

    result = svdq_gemm_w4a4(**gemm_kwargs)
    if output is not None:
      output.copy_(result)
      return output
    return result

  def __repr__(self) -> str:
    extra = ""
    if self._needs_k_pad or self._needs_n_strip:
      extra = (f", padded_in={self._padded_in_features}"
               f", padded_out={self._padded_out_features}")
    return (f"SVDQW4A4Linear(in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, precision={self.precision}, act_unsigned={self.act_unsigned}, "
            f"runtime_kernel={self.runtime_kernel}, lora_layout={self.lora_layout}{extra})")


__all__ = ["SVDQW4A4Linear"]
