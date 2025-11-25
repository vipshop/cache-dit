
import torch
import torch_npu

import torch.nn as nn

from ..utils import log_replace_info


class NpuLayerNorm(nn.LayerNorm):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_layer_norm_eval(
            inputs,
            normalized_shape=self.normalized_shape,
            weight=self.weight,
            bias=self.bias,
            eps=self.eps,
        )


def replace_func():
    # from torch import nn
    # nn.LayerNorm = NpuLayerNorm

    from diffusers.models import normalization
    normalization.FP32LayerNorm = NpuLayerNorm


def replace_npu_layer_norm_eval():
    replace_func()
    log_replace_info("FP32LayerNorm of Diffusers", "npu_layer_norm_eval")