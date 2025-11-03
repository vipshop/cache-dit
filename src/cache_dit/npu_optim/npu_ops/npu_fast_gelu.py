import torch
import torch_npu
import torch.nn as nn

from diffusers.models.activations import GELU as GeluDiffuser

from ..utils import log_replace_info


class NpuFastGelu(nn.GELU):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_fast_gelu(input)


class NpuFastGeluDiffuser(GeluDiffuser):
    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_fast_gelu(gate)


def replace_func():
    from diffusers.models import activations
    activations.GELU = NpuFastGeluDiffuser

    from torch import nn
    nn.GELU = NpuFastGelu


def replace_npu_fast_gelu():
    replace_func()
    log_replace_info("nn.GELU and GELU of Diffusers", "npu_fast_gelu")
