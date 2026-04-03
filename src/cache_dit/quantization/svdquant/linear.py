import torch
from torch import nn

from ...kernels import svdq_gemm_w4a4
from ...kernels import svdq_quantize_w4a4_act_fuse_lora


class SVDQW4A4Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 32,
        bias: bool = True,
        precision: str = "int4",
        act_unsigned: bool = False,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cpu")
        if rank < 0:
            raise ValueError(f"rank must be non-negative, got {rank}.")
        if in_features % 2 != 0:
            raise ValueError(f"in_features must be divisible by 2, got {in_features}.")

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.precision = precision
        self.torch_dtype = torch_dtype

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
        self.bias = (
            nn.Parameter(
                torch.empty(out_features, dtype=torch_dtype, device=device), requires_grad=True
            )
            if bias
            else None
        )
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
        self.proj_down = nn.Parameter(
            torch.empty(in_features, rank, dtype=torch_dtype, device=device)
        )
        self.proj_up = nn.Parameter(
            torch.empty(out_features, rank, dtype=torch_dtype, device=device)
        )

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
        batch_size, seq_len, channels = x.shape
        token_count = batch_size * seq_len
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
            if output.shape != (token_count, self.out_features):
                raise ValueError(
                    "output must have shape "
                    f"({token_count}, {self.out_features}), got {tuple(output.shape)}."
                )
            output.copy_(logical_output)
            logical_output = output

        return logical_output.reshape(batch_size, seq_len, self.out_features)

    def quantize(
        self, x: torch.Tensor, pad_size: int = 256
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        result = svdq_gemm_w4a4(**gemm_kwargs)
        if output is not None:
            output.copy_(result)
            return output
        return result

    def __repr__(self) -> str:
        return (
            f"SVDQW4A4Linear(in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, precision={self.precision}, act_unsigned={self.act_unsigned})"
        )


__all__ = ["SVDQW4A4Linear"]
