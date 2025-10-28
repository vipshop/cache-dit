from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed._tensor import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)


def dit_apply_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
):
    for name, block in model.transformer_blocks.named_children():
        block.attn.heads //= tp_mesh.size()
        layer_plan = {
            "attn.to_q": ColwiseParallel(),
            "attn.to_k": ColwiseParallel(),
            "attn.to_v": ColwiseParallel(),
            "attn.to_out.0": RowwiseParallel(),
            "img_mod.1": ColwiseParallel(output_layouts=Replicate()),
            "img_mlp.net.0.proj": ColwiseParallel(),
            "img_mlp.net.2": RowwiseParallel(),
            "attn.add_q_proj": ColwiseParallel(),
            "attn.add_k_proj": ColwiseParallel(),
            "attn.add_v_proj": ColwiseParallel(),
            "attn.to_add_out": RowwiseParallel(),
            "txt_mod.1": ColwiseParallel(output_layouts=Replicate()),
            "txt_mlp.net.0.proj": ColwiseParallel(),
            "txt_mlp.net.2": RowwiseParallel(),
        }
        parallelize_module(
            module=block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )
    return model
