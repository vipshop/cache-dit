# Cache-DiT's SVDQ packing helpers are adapted from deepcompressor/nunchaku.
# The layout logic below is the Python-side half of the W4A4 kernel contract.
from __future__ import annotations

import typing as tp

import torch

__all__ = [
  "MmaWeightPacker",
  "SVDQWeightPacker",
  "adapt_svdq_module_state_dict",
  "ceil_divide",
  "export_raw_svdq_w4a4_state_dict",
  "fp_quantize",
  "pad",
]


def ceil_divide(value: int, divisor: int) -> int:
  return (value + divisor - 1) // divisor


def pad(
  tensor: torch.Tensor | None,
  divisor: int | tp.Sequence[int],
  dim: int | tp.Sequence[int],
  fill_value: float | int = 0,
) -> torch.Tensor | None:
  if tensor is None:
    return None
  if isinstance(divisor, int):
    if divisor <= 1:
      return tensor
  elif all(item <= 1 for item in divisor):
    return tensor

  shape = list(tensor.shape)
  if isinstance(dim, int):
    assert isinstance(divisor, int)
    shape[dim] = ceil_divide(shape[dim], divisor) * divisor
  else:
    if isinstance(divisor, int):
      divisor = [divisor] * len(dim)
    for axis, axis_divisor in zip(dim, divisor, strict=True):
      shape[axis] = ceil_divide(shape[axis], axis_divisor) * axis_divisor

  result = torch.full(shape, fill_value, dtype=tensor.dtype, device=tensor.device)
  result[tuple(slice(0, extent) for extent in tensor.shape)] = tensor
  return result


def fp_quantize(x: torch.Tensor, codebook: torch.Tensor | None = None) -> torch.Tensor:
  if codebook is None:
    codebook = torch.tensor(
      [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
      ],
      dtype=x.dtype,
      device=x.device,
    )
  return (x.unsqueeze(-1) - codebook.unsqueeze(0)).abs().argmin(dim=-1)


class MmaWeightPacker:
  """Describe the warp-level packing contract for the SVDQ weight-side MMA operand.

  This helper does not quantize anything by itself. It computes how Python should tile and
  distribute packed values across a 32-lane warp so the exported tensors line up with the CUDA
  zgemm kernels in `csrc/kernels/svdq/zgemm`. For the INT4 path it defaults to `comp_n = 16` and
  `comp_k = 256 / bits = 64`, then splits the tile across `8` N-side lanes and `4` K-side lanes.

  Those values mirror the kernel-side contract in `gemm_base.cuh`: the W4A4 path exposes
  `INSN_K = 64` and `WARP_N = 128` at the GEMM layer, while the low-level wrappers in
  `mma_earlycuda.cuh` ultimately issue `mma.sync.aligned.m16n8k64` PTX fragments. The
  `insn_n = 8` bookkeeping here intentionally follows that lower-level fragment width; the
  surrounding kernel composes two `m16n8k64` fragments to reach the higher-level `INSN_N = 16`
  tile.
  """

  def __init__(self,
               bits: int,
               warp_n: int,
               comp_n: int | None = None,
               comp_k: int | None = None) -> None:
    """Initialize the MMA-side packing geometry.

    :param bits: Bit-width of each packed weight element.
    :param warp_n: Output-channel span covered by one warp tile.
    :param comp_n: Optional fragment width along the N dimension. Defaults to the
      W4A4-friendly value `16`.
    :param comp_k: Optional fragment width along the K dimension. Defaults to `256 // bits`,
      which becomes `64` for INT4.
    """

    self.bits = bits
    if self.bits not in (1, 4, 8, 16, 32):
      raise ValueError(f"Unsupported weight bit-width: {bits}.")

    self.comp_n = comp_n if comp_n is not None else 16
    self.comp_k = comp_k if comp_k is not None else 256 // self.bits
    self.insn_n = 8
    self.insn_k = self.comp_k
    if self.insn_k * self.bits not in (128, 256):
      raise ValueError("insn_k * bits must be 128 or 256.")
    if self.comp_n % self.insn_n != 0:
      raise ValueError("comp_n must be divisible by insn_n.")

    self.num_lanes = 32
    self.num_k_lanes = 4
    self.num_n_lanes = 8
    if warp_n < self.comp_n or warp_n % self.comp_n != 0:
      raise ValueError("warp_n must be divisible by comp_n.")
    self.warp_n = warp_n

    self.reg_k = 32 // self.bits
    self.reg_n = 1
    self.k_pack_size = self.comp_k // (self.num_k_lanes * self.reg_k)
    self.n_pack_size = self.comp_n // (self.num_n_lanes * self.reg_n)
    self.pack_size = self.k_pack_size * self.n_pack_size
    if not 1 <= self.pack_size <= 4:
      raise ValueError("pack_size must be between 1 and 4.")

    self.mem_k = self.comp_k
    self.mem_n = warp_n
    self.num_k_packs = self.mem_k // (self.k_pack_size * self.num_k_lanes * self.reg_k)
    self.num_n_packs = self.mem_n // (self.n_pack_size * self.num_n_lanes * self.reg_n)


class SVDQWeightPacker(MmaWeightPacker):
  """Pack SVDQ tensors into the exact layout consumed by the W4A4 CUDA kernels.

  `MmaWeightPacker` defines the lane/register geometry; this class applies that geometry to
  cache-dit's concrete artifacts: signed INT4 weights, grouped scales or FP4 micro-scales, biases
  and smooth factors, plus optional low-rank residual matrices. The output is not an arbitrary
  checkpoint layout. It mirrors the weight-side operand and scale ordering that
  `csrc/kernels/svdq/zgemm/gemm_base.cuh` and `gemm_w4a4.cuh` expect to load.

  On the kernel side, those packed tiles are consumed by `mma_m16n8kx_s32common` in
  `mma_earlycuda.cuh`, which lowers to `mma.sync.aligned.m16n8k64.row.col.s32.u4.s4.s32` or
  `mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32` depending on activation signedness. In PTX
  terms `.row.col` means row-major A and column-major B, `.u4`/`.s4` select unsigned or signed
  4-bit multiplicands, and `.s32` is the accumulator/result type. For the `m16n8k64` fragment
  shape each thread contributes four `.b32` registers for operand A and two `.b32` registers for
  operand B, which is why the padding, permutation and bit-packing logic here is warp-shaped
  instead of being a simple flatten-and-save step.
  """

  def __init__(self, bits: int, warp_n: int = 128) -> None:
    """Initialize the SVDQ-specific packer defaults.

    :param bits: Bit-width of the packed weight operand. The current SVDQ runtime uses `4` for
      the W4A4 path.
    :param warp_n: Output-channel width covered by one packed warp tile. Defaults to `128`,
      matching the migrated W4A4 kernel contract.
    """

    super().__init__(bits=bits, warp_n=warp_n)
    self.num_k_unrolls = 2

  def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
    """Pack an integer weight matrix into the kernel-facing byte layout.

    :param weight: Quantized weight matrix with shape
      `[out_features, in_features]` and dtype `torch.int32`.
    :returns: A packed `torch.int8` tensor whose byte order matches the W4A4
      kernel's warp-level operand loading pattern.
    """

    if weight.dtype != torch.int32:
      raise ValueError(f"Quantized weight must be torch.int32, got {weight.dtype}.")
    out_features, in_features = weight.shape
    if out_features % self.mem_n != 0:
      raise ValueError(f"output channel size ({out_features}) must be divisible by {self.mem_n}.")
    if in_features % (self.mem_k * self.num_k_unrolls) != 0:
      raise ValueError(
        f"input channel size ({in_features}) must be divisible by {self.mem_k * self.num_k_unrolls}."
      )

    n_tiles, k_tiles = out_features // self.mem_n, in_features // self.mem_k
    weight = weight.reshape(
      n_tiles,
      self.num_n_packs,
      self.n_pack_size,
      self.num_n_lanes,
      self.reg_n,
      k_tiles,
      self.num_k_packs,
      self.k_pack_size,
      self.num_k_lanes,
      self.reg_k,
    )
    weight = weight.permute(0, 5, 6, 1, 3, 8, 2, 7, 4, 9).contiguous()

    if self.bits == 4:
      weight = weight.bitwise_and_(0xF)
      shift = torch.arange(0, 32, 4, dtype=torch.int32, device=weight.device)
    elif self.bits == 8:
      weight = weight.bitwise_and_(0xFF)
      shift = torch.arange(0, 32, 8, dtype=torch.int32, device=weight.device)
    else:
      raise NotImplementedError(f"SVDQWeightPacker does not support {self.bits}-bit weights.")
    weight = weight.bitwise_left_shift_(shift)
    weight = weight.sum(dim=-1, dtype=torch.int32)
    return weight.view(dtype=torch.int8).view(out_features, -1)

  def check_if_micro_scale(self, group_size: int) -> bool:
    """Return whether a scale tensor should use the FP4 micro-scale layout.

    Micro-scale packing is selected when one scale value covers exactly four K-lane elements of the
    current instruction tile.

    :param group_size: Number of input channels represented by one scale value.
    :returns: `True` when the scale should be packed with the FP4 micro-scale path.
    """

    return group_size > 0 and self.insn_k == group_size * 4

  def pack_scale(self, scale: torch.Tensor, group_size: int) -> torch.Tensor:
    """Pack standard scale, bias, or smooth tensors for the runtime layout.

    :param scale: Scale-like tensor after any required padding.
    :param group_size: Logical number of input channels represented by one scale value, or `-1` for
        vectors such as bias and smooth factors.
    :returns: A packed scale tensor laid out for warp-level scale loading.
    """

    if self.check_if_micro_scale(group_size=group_size):
      return self.pack_micro_scale(scale, group_size=group_size)

    if scale.dtype not in (torch.float16, torch.bfloat16):
      raise ValueError(f"Scale dtype must be fp16/bf16, got {scale.dtype}.")

    out_features = scale.shape[0]
    s_pack_size = min(max(self.warp_n // self.num_lanes, 2), 8)
    num_s_lanes = min(self.num_lanes, self.warp_n // s_pack_size)
    num_s_packs = self.warp_n // (s_pack_size * num_s_lanes)
    warp_s = num_s_packs * num_s_lanes * s_pack_size
    if warp_s != self.warp_n:
      raise ValueError("warp_n for scales must match the packer warp_n.")

    scale = scale.reshape(out_features // warp_s, num_s_packs, num_s_lanes // 4, s_pack_size // 2,
                          4, 2, -1)
    scale = scale.permute(0, 6, 1, 2, 4, 3, 5).contiguous()
    return scale.view(-1) if group_size == -1 else scale.view(-1, out_features)

  def pack_micro_scale(self, scale: torch.Tensor, group_size: int) -> torch.Tensor:
    """Pack FP4 micro-scales into the tensor layout consumed by the kernel.

    :param scale: Floating-point scale tensor after padding.
    :param group_size: Input channels represented by one micro-scale value.
      The current implementation requires `16`.
    :returns: A packed micro-scale tensor with the lane/interleave order
      expected by the FP4 kernel path.
    """

    if scale.dtype not in (torch.float16, torch.bfloat16):
      raise ValueError(f"Scale dtype must be fp16/bf16, got {scale.dtype}.")
    if group_size != 16:
      raise ValueError("SVDQWeightPacker only supports FP4 micro-scales with group_size=16.")
    scale = scale.to(dtype=torch.float8_e4m3fn)
    out_features = scale.shape[0]

    s_pack_size = min(max(self.warp_n // self.num_lanes, 1), 4)
    num_s_lanes = 32
    num_s_packs = ceil_divide(self.warp_n, s_pack_size * num_s_lanes)
    warp_s = num_s_packs * num_s_lanes * s_pack_size
    if warp_s != self.warp_n:
      raise ValueError("warp_n for scales must match the packer warp_n.")

    scale = scale.view(out_features // warp_s, num_s_packs, s_pack_size, 4, 8, -1,
                       self.insn_k // group_size)
    scale = scale.permute(0, 5, 1, 4, 3, 2, 6).contiguous()
    return scale.view(-1, out_features)

  def pack_lowrank_weight(self, weight: torch.Tensor, down: bool) -> torch.Tensor:
    """Pack a low-rank residual factor into the runtime tensor layout.

    :param weight: Floating-point low-rank matrix. `down=True` expects the down projection shape
        `[rank, channels]`, while `down=False` expects the up projection shape `[channels, rank]`.
    :param down: Whether the tensor belongs to the down-projection branch.
    :returns: A packed low-rank tensor reshaped to the layout expected by the fused runtime path.
    """

    if weight.dtype not in (torch.float16, torch.bfloat16):
      raise ValueError(f"Low-rank weight dtype must be fp16/bf16, got {weight.dtype}.")

    reg_n, reg_k = 1, 2
    pack_n = self.n_pack_size * self.num_n_lanes * reg_n
    pack_k = self.k_pack_size * self.num_k_lanes * reg_k
    weight = pad(weight, divisor=(pack_n, pack_k), dim=(0, 1))
    if down:
      rank, channels = weight.shape
      rank_packs, channel_packs = rank // pack_n, channels // pack_k
      weight = weight.view(rank_packs, pack_n, channel_packs, pack_k).permute(2, 0, 1, 3)
    else:
      channels, rank = weight.shape
      channel_packs, rank_packs = channels // pack_n, rank // pack_k
      weight = weight.view(channel_packs, pack_n, rank_packs, pack_k).permute(0, 2, 1, 3)

    weight = weight.reshape(
      channel_packs,
      rank_packs,
      self.n_pack_size,
      self.num_n_lanes,
      reg_n,
      self.k_pack_size,
      self.num_k_lanes,
      reg_k,
    )
    weight = weight.permute(0, 1, 3, 6, 2, 5, 4, 7).contiguous()
    return weight.view(channels, rank)

  def pad_weight(self, weight: torch.Tensor) -> torch.Tensor:
    """Pad a quantized weight matrix to the packer's warp tile multiples.

    :param weight: Quantized weight matrix before warp-tile padding.
    :returns: The padded quantized weight matrix.
    """

    return tp.cast(
      torch.Tensor,
      pad(weight, divisor=(self.mem_n, self.mem_k * self.num_k_unrolls), dim=(0, 1)),
    )

  def pad_scale(self, scale: torch.Tensor, group_size: int) -> torch.Tensor:
    """Pad scale-like tensors so `pack_scale` can reshape them safely.

    :param scale: Scale, bias, or smooth tensor before packing.
    :param group_size: Logical channels represented by each scale value, or `-1`
    for channel-wise vectors.

    :returns: The padded tensor with fill value `1` in newly created scale slots.
    """

    if group_size > 0 and scale.numel() > scale.shape[0]:
      scale = scale.view(scale.shape[0], 1, -1, 1)
      if self.check_if_micro_scale(group_size=group_size):
        scale = pad(
          scale,
          divisor=(self.warp_n, self.insn_k // group_size),
          dim=(0, 2),
          fill_value=1,
        )
      else:
        scale = pad(scale, divisor=(self.warp_n, self.num_k_unrolls), dim=(0, 2), fill_value=1)
    else:
      scale = pad(scale, divisor=self.warp_n, dim=0, fill_value=1)
    return tp.cast(torch.Tensor, scale)

  def pad_lowrank_weight(self, weight: torch.Tensor, down: bool) -> torch.Tensor:
    """Pad a low-rank factor so its packing axes align with `warp_n`.

    :param weight: Low-rank factor to pad before low-rank packing.
    :param down: Whether the tensor belongs to the down-projection branch.
    :returns: The padded low-rank factor.
    """

    return tp.cast(torch.Tensor, pad(weight, divisor=self.warp_n, dim=1 if down else 0))


def _validate_scale_shape(scale: torch.Tensor, out_features: int,
                          in_features: int) -> tuple[int, int, bool]:
  if scale.numel() == 1:
    scale = scale.view(-1).expand(out_features).reshape(out_features, 1, 1, 1)
    per_tensor_scale = True
  else:
    per_tensor_scale = False
  if (scale.ndim != 4 or scale.shape[1] != 1 or scale.shape[3] != 1
      or scale.shape[0] != out_features):
    raise ValueError("Scale tensor must have shape [out_features, 1, num_groups, 1].")
  num_groups = scale.shape[2]
  group_size = in_features // num_groups
  if in_features != group_size * num_groups:
    raise ValueError("input channel size must equal group_size * num_groups.")
  return num_groups, group_size, per_tensor_scale


def pack_svdq_w4a4_linear_tensors(
  weight: torch.Tensor,
  scale: torch.Tensor,
  bias: torch.Tensor | None = None,
  smooth: torch.Tensor | None = None,
  lora: tuple[torch.Tensor, torch.Tensor] | None = None,
  float_point: bool = False,
  subscale: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor] | None,
    torch.Tensor | None,
]:
  """Pack floating-point SVDQ tensors into the W4A4 kernel layout.

  :param weight: Floating-point residual weight matrix with shape `[out_features, in_features]`.
  :param scale: Per-group weight scales. May be a scalar tensor or a 4D tensor with shape
    `[out_features, 1, num_groups, 1]`.
  :param bias: Optional bias vector for the linear layer.
  :param smooth: Optional per-input-channel smoothing vector.
  :param lora: Optional tuple `(lora_down, lora_up)` containing the low-rank residual factors before
    packing.
  :param float_point: Whether to pack FP4-style unsigned nibble values instead of signed INT4
    values.
  :param subscale: Optional finer-grained 4D scale tensor used by formats that carry an additional
    subscale hierarchy.
  :returns: A tuple `(qweight, wscales, bias, smooth, lora, subscale)` where every tensor is
    rearranged into the layout expected by the SVDQ W4A4 runtime kernels.
  """

  if weight.ndim != 2:
    raise ValueError("Weight tensor must be 2D.")
  if weight.dtype not in (torch.float16, torch.bfloat16):
    raise ValueError(f"Weight dtype must be fp16/bf16, got {weight.dtype}.")

  device, dtype = weight.device, weight.dtype
  out_features, in_features = weight.shape
  num_groups, group_size, per_tensor_scale = _validate_scale_shape(scale, out_features, in_features)

  if subscale is not None:
    if (subscale.ndim != 4 or subscale.shape[1] != 1 or subscale.shape[3] != 1
        or subscale.shape[0] != out_features):
      raise ValueError("Subscale tensor must have shape [out_features, 1, num_subgroups, 1].")
    num_subgroups = subscale.shape[2]
    subgroup_size = in_features // num_subgroups
    if in_features != subgroup_size * num_subgroups:
      raise ValueError("input channel size must equal subgroup_size * num_subgroups.")
    if group_size <= subgroup_size or group_size % subgroup_size != 0:
      raise ValueError("group_size must be divisible by subgroup_size.")
  else:
    num_subgroups, subgroup_size = num_groups, group_size

  weight = weight.to(dtype=torch.float32).view(out_features, 1, num_groups, group_size)
  weight = weight.div_(scale.to(dtype=torch.float32, device=device))
  if subscale is not None:
    weight = weight.view(out_features, 1, num_subgroups, subgroup_size)
    weight = weight.div_(subscale.to(dtype=torch.float32, device=device))
  weight = weight.view(out_features, in_features)

  if float_point:
    weight = fp_quantize(weight)
    if weight.min() < 0 or weight.max() > 15:
      raise ValueError("FP4 quantized weights must be in [0, 15].")
  else:
    weight = weight.round_()
    if weight.min() < -8 or weight.max() > 7:
      raise ValueError("INT4 quantized weights must be in [-8, 7].")

  bias = (torch.zeros(
    (out_features, 1), dtype=dtype, device=device) if bias is None else bias.view(-1, 1))
  smooth = (torch.ones(
    (in_features, 1), dtype=dtype, device=device) if smooth is None else smooth.view(-1, 1))

  packer = SVDQWeightPacker(bits=4)
  weight = packer.pack_weight(packer.pad_weight(weight.to(dtype=torch.int32)))
  scale = packer.pack_scale(
    packer.pad_scale(scale.to(dtype=dtype), group_size=group_size),
    group_size if group_size < in_features else -1,
  )
  if subscale is not None:
    subscale = packer.pack_scale(
      packer.pad_scale(subscale.to(dtype=dtype), group_size=subgroup_size),
      subgroup_size if subgroup_size < in_features else -1,
    )
  bias = packer.pack_scale(packer.pad_scale(bias.to(dtype=dtype), group_size=-1), group_size=-1)
  smooth = packer.pack_scale(packer.pad_scale(smooth.to(dtype=dtype), group_size=-1), group_size=-1)

  if lora is not None:
    lora_down, lora_up = lora
    lora_down = packer.pack_lowrank_weight(packer.pad_lowrank_weight(lora_down, down=True),
                                           down=True)
    lora_up = packer.pack_lowrank_weight(packer.pad_lowrank_weight(lora_up, down=False), down=False)
    lora = (lora_down, lora_up)

  if per_tensor_scale:
    scale = scale.view(-1)[0].view(1)
  return weight, scale, bias, smooth, lora, subscale


def export_raw_svdq_w4a4_state_dict(
  weight: torch.Tensor,
  scale: torch.Tensor,
  bias: torch.Tensor | None = None,
  smooth: torch.Tensor | None = None,
  lora: tuple[torch.Tensor, torch.Tensor] | None = None,
  float_point: bool = False,
  subscale: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
  """Export floating-point SVDQ tensors into the raw packed checkpoint schema.

  :param weight: Floating-point residual weight matrix to quantize and pack.
  :param scale: Per-group weight scales.
  :param bias: Optional bias vector.
  :param smooth: Optional per-input-channel smoothing vector.
  :param lora: Optional tuple `(lora_down, lora_up)` containing the low-rank
    residual factors.
  :param float_point: Whether to use the FP4 packing path instead of INT4.
  :param subscale: Optional subscale hierarchy for formats that require it.
  :returns: A raw packed state dict keyed by serialized tensor roles such as
    `qweight`, `wscales`, `smooth`, `smooth_orig`, and optional `lora_*` /
    `wcscales` entries.
  """

  if lora is not None and smooth is not None:
    lora_down, lora_up = lora
    lora_down = lora_down.to(dtype=torch.float64)
    lora_down = lora_down.div_(smooth.to(dtype=torch.float64).unsqueeze(0))
    lora = (lora_down.to(dtype=weight.dtype), lora_up)

  qweight, packed_scale, packed_bias, packed_smooth, packed_lora, packed_subscale = (
    pack_svdq_w4a4_linear_tensors(
      weight,
      scale=scale,
      bias=bias,
      smooth=smooth,
      lora=lora,
      float_point=float_point,
      subscale=subscale,
    ))

  state_dict: dict[str, torch.Tensor] = {
    "qweight": qweight,
    "bias": packed_bias,
    "smooth": packed_smooth.clone(),
    "smooth_orig": packed_smooth,
  }
  if packed_scale.numel() == 1:
    state_dict["wtscale"] = packed_scale
  else:
    state_dict["wscales"] = packed_scale
  if packed_subscale is not None:
    state_dict["wcscales"] = packed_subscale
  if packed_lora is not None:
    state_dict["lora_down"] = packed_lora[0]
    state_dict["lora_up"] = packed_lora[1]
  return state_dict


def adapt_svdq_module_state_dict(
  raw_state_dict: dict[str, torch.Tensor],
  *,
  in_features: int,
  out_features: int,
  rank: int,
  torch_dtype: torch.dtype,
  device: torch.device | str,
  has_bias: bool,
) -> dict[str, torch.Tensor]:
  """Map raw packed tensors onto the parameter names used by `SVDQW4A4Linear`.

  :param raw_state_dict: Raw packed schema produced by `export_raw_svdq_w4a4_state_dict`.
  :param in_features: Logical input width of the target runtime module.
  :param out_features: Logical output width of the target runtime module.
  :param rank: Low-rank residual rank expected by the runtime module.
  :param torch_dtype: Floating-point dtype for synthesized empty low-rank tensors.
  :param device: Device for synthesized tensors that are absent from the raw state.
  :param has_bias: Whether the target runtime module expects a bias parameter.

  :returns: A module state dict ready for `SVDQW4A4Linear.load_state_dict`.
  """

  module_state_dict = {
    "qweight":
    raw_state_dict["qweight"],
    "smooth_factor":
    raw_state_dict["smooth"],
    "smooth_factor_orig":
    raw_state_dict["smooth_orig"],
    "proj_down":
    raw_state_dict.get(
      "lora_down",
      torch.empty((in_features, rank), dtype=torch_dtype, device=device),
    ),
    "proj_up":
    raw_state_dict.get(
      "lora_up",
      torch.empty((out_features, rank), dtype=torch_dtype, device=device),
    ),
  }
  if "wscales" in raw_state_dict:
    module_state_dict["wscales"] = raw_state_dict["wscales"]
  if has_bias:
    module_state_dict["bias"] = raw_state_dict["bias"]
  if "wtscale" in raw_state_dict:
    module_state_dict["wtscale"] = raw_state_dict["wtscale"]
  if "wcscales" in raw_state_dict:
    module_state_dict["wcscales"] = raw_state_dict["wcscales"]
  return module_state_dict
