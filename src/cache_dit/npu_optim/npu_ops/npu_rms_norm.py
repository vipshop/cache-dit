import torch
import torch_npu
import torch.nn as nn

from ..utils import log_replace_info


class NpuRMSNorm(nn.RMSNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_rms_norm(x, self.weight, self.eps)[0]


def replace_func():
    from torch import nn
    nn.RMSNorm = NpuRMSNorm


def replace_npu_rms_norm():
    replace_func()
    log_replace_info("nn.RMSNorm", "npu_rms_norm")