import torch
import torch_npu
import torch.nn as nn

from diffusers.models.activations import GELU as GeluDiffuser

from ..utils import log_replace_info


class NpuFastGelu(nn.GELU):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_fast_gelu(input)

def F_fast_gelu_func(
    input: torch.Tensor,
    approximate: str = "tanh"
) -> torch.Tensor:
    return torch_npu.npu_fast_gelu(input)

def replace_func():
    import torch.nn.functional as F
    F.gelu = F_fast_gelu_func
    
    from torch import nn
    nn.GELU = NpuFastGelu

def replace_npu_fast_gelu():
    replace_func()
    log_replace_info("nn.GELU and GELU of Diffusers", "npu_fast_gelu")
