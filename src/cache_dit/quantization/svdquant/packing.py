from __future__ import annotations

import typing as tp

import torch

__all__ = [
    "MmaWeightPackerBase",
    "NunchakuWeightPacker",
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


class MmaWeightPackerBase:
    def __init__(
        self, bits: int, warp_n: int, comp_n: int | None = None, comp_k: int | None = None
    ) -> None:
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


class NunchakuWeightPacker(MmaWeightPackerBase):
    def __init__(self, bits: int, warp_n: int = 128) -> None:
        super().__init__(bits=bits, warp_n=warp_n)
        self.num_k_unrolls = 2

    def pack_weight(self, weight: torch.Tensor) -> torch.Tensor:
        if weight.dtype != torch.int32:
            raise ValueError(f"Quantized weight must be torch.int32, got {weight.dtype}.")
        out_features, in_features = weight.shape
        if out_features % self.mem_n != 0:
            raise ValueError(
                f"output channel size ({out_features}) must be divisible by {self.mem_n}."
            )
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
            raise NotImplementedError(
                f"Weight bits {self.bits} are not supported in the minimal packer."
            )
        weight = weight.bitwise_left_shift_(shift)
        weight = weight.sum(dim=-1, dtype=torch.int32)
        return weight.view(dtype=torch.int8).view(out_features, -1)

    def check_if_micro_scale(self, group_size: int) -> bool:
        return group_size > 0 and self.insn_k == group_size * 4

    def pack_scale(self, scale: torch.Tensor, group_size: int) -> torch.Tensor:
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

        scale = scale.reshape(
            out_features // warp_s, num_s_packs, num_s_lanes // 4, s_pack_size // 2, 4, 2, -1
        )
        scale = scale.permute(0, 6, 1, 2, 4, 3, 5).contiguous()
        return scale.view(-1) if group_size == -1 else scale.view(-1, out_features)

    def pack_micro_scale(self, scale: torch.Tensor, group_size: int) -> torch.Tensor:
        if scale.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(f"Scale dtype must be fp16/bf16, got {scale.dtype}.")
        if group_size != 16:
            raise ValueError(
                "The minimal packer only supports FP4 micro-scales with group_size=16."
            )
        scale = scale.to(dtype=torch.float8_e4m3fn)
        out_features = scale.shape[0]

        s_pack_size = min(max(self.warp_n // self.num_lanes, 1), 4)
        num_s_lanes = 32
        num_s_packs = ceil_divide(self.warp_n, s_pack_size * num_s_lanes)
        warp_s = num_s_packs * num_s_lanes * s_pack_size
        if warp_s != self.warp_n:
            raise ValueError("warp_n for scales must match the packer warp_n.")

        scale = scale.view(
            out_features // warp_s, num_s_packs, s_pack_size, 4, 8, -1, self.insn_k // group_size
        )
        scale = scale.permute(0, 5, 1, 4, 3, 2, 6).contiguous()
        return scale.view(-1, out_features)

    def pack_lowrank_weight(self, weight: torch.Tensor, down: bool) -> torch.Tensor:
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
        return tp.cast(
            torch.Tensor,
            pad(weight, divisor=(self.mem_n, self.mem_k * self.num_k_unrolls), dim=(0, 1)),
        )

    def pad_scale(self, scale: torch.Tensor, group_size: int) -> torch.Tensor:
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
                scale = pad(
                    scale, divisor=(self.warp_n, self.num_k_unrolls), dim=(0, 2), fill_value=1
                )
        else:
            scale = pad(scale, divisor=self.warp_n, dim=0, fill_value=1)
        return tp.cast(torch.Tensor, scale)

    def pad_lowrank_weight(self, weight: torch.Tensor, down: bool) -> torch.Tensor:
        return tp.cast(torch.Tensor, pad(weight, divisor=self.warp_n, dim=1 if down else 0))


def _validate_scale_shape(
    scale: torch.Tensor, out_features: int, in_features: int
) -> tuple[int, int, bool]:
    if scale.numel() == 1:
        scale = scale.view(-1).expand(out_features).reshape(out_features, 1, 1, 1)
        per_tensor_scale = True
    else:
        per_tensor_scale = False
    if (
        scale.ndim != 4
        or scale.shape[1] != 1
        or scale.shape[3] != 1
        or scale.shape[0] != out_features
    ):
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
    if weight.ndim != 2:
        raise ValueError("Weight tensor must be 2D.")
    if weight.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(f"Weight dtype must be fp16/bf16, got {weight.dtype}.")

    device, dtype = weight.device, weight.dtype
    out_features, in_features = weight.shape
    num_groups, group_size, per_tensor_scale = _validate_scale_shape(
        scale, out_features, in_features
    )

    if subscale is not None:
        if (
            subscale.ndim != 4
            or subscale.shape[1] != 1
            or subscale.shape[3] != 1
            or subscale.shape[0] != out_features
        ):
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

    bias = (
        torch.zeros((out_features, 1), dtype=dtype, device=device)
        if bias is None
        else bias.view(-1, 1)
    )
    smooth = (
        torch.ones((in_features, 1), dtype=dtype, device=device)
        if smooth is None
        else smooth.view(-1, 1)
    )

    packer = NunchakuWeightPacker(bits=4)
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
    smooth = packer.pack_scale(
        packer.pad_scale(smooth.to(dtype=dtype), group_size=-1), group_size=-1
    )

    if lora is not None:
        lora_down, lora_up = lora
        lora_down = packer.pack_lowrank_weight(
            packer.pad_lowrank_weight(lora_down, down=True), down=True
        )
        lora_up = packer.pack_lowrank_weight(
            packer.pad_lowrank_weight(lora_up, down=False), down=False
        )
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
        )
    )

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
    module_state_dict = {
        "qweight": raw_state_dict["qweight"],
        "smooth_factor": raw_state_dict["smooth"],
        "smooth_factor_orig": raw_state_dict["smooth_orig"],
        "proj_down": raw_state_dict.get(
            "lora_down",
            torch.empty((in_features, rank), dtype=torch_dtype, device=device),
        ),
        "proj_up": raw_state_dict.get(
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
