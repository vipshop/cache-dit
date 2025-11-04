import torch
import torch_npu
import torch.nn as nn
from typing import Dict, Optional, Tuple

from ..utils import log_replace_info


# import ascend_ops
op_path="/home/z00879328/vipshop/scripts/fa_op_test/op_build/build/lib.linux-aarch64-cpython-311/ascend_ops.cpython-311-aarch64-linux-gnu.so"
torch.ops.load_library(op_path)

# class NpuAdaLayerNormSingle(AdaLayerNormZeroSingle):
def AdaLayerNormZeroSingle_forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
    emb = self.linear(self.silu(emb))
    shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
    # x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
    x = torch.ops.ascend_ops.adalayernorm(
        x=x,
        scale=scale_msa,
        shift=shift_msa,
        epsilson=1e-6
    )

    return x, gate_msa

def AdaLayerNormZero_forward(self,
    x: torch.Tensor,
    timestep: Optional[torch.Tensor] = None,
    class_labels: Optional[torch.LongTensor] = None,
    hidden_dtype: Optional[torch.dtype] = None,
    emb: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    if self.emb is not None:
        emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
    emb = self.linear(self.silu(emb))
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
    # x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
    x = torch.ops.ascend_ops.adalayernorm(
        x=x,
        scale=scale_msa,
        shift=shift_msa,
        epsilson=1e-6
    )
    return x, gate_msa, shift_mlp, scale_mlp, gate_mlp

def replace_npu_adalayernorm():
    from diffusers.models import normalization
    normalization.AdaLayerNormZero.forward = AdaLayerNormZero_forward
    normalization.AdaLayerNormZeroSingle.forward = AdaLayerNormZeroSingle_forward
    log_replace_info("AdaLayerNormZero of normalization", "npu_adalayernorm")
    log_replace_info("AdaLayerNormZeroSingle of normalization", "npu_adalayernorm")