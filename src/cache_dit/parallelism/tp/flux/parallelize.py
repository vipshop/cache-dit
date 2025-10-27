import torch
from diffusers.models.transformers.transformer_flux import (
    FluxSingleTransformerBlock,
)
from einops import rearrange
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed._tensor import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from cache_dit.logger import init_logger

logger = init_logger(__name__)


def t5_apply_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
):
    for i, block in enumerate(model.encoder.block):
        block.layer[0].SelfAttention.n_heads //= tp_mesh.size()
        block.layer[0].SelfAttention.inner_dim //= tp_mesh.size()
        layer_plan = {
            "layer.0.SelfAttention.q": ColwiseParallel(),
            "layer.0.SelfAttention.k": ColwiseParallel(),
            "layer.0.SelfAttention.v": ColwiseParallel(),
            "layer.0.SelfAttention.o": RowwiseParallel(),
            "layer.1.DenseReluDense.wi_0": ColwiseParallel(),
            "layer.1.DenseReluDense.wi_1": ColwiseParallel(),
            "layer.1.DenseReluDense.wo": RowwiseParallel(),
        }
        if i == 0:
            layer_plan["layer.0.SelfAttention.relative_attention_bias"] = (
                ColwiseParallel()
            )
        parallelize_module(
            module=block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )


def prepare_proj_out_weight(
    single_block: FluxSingleTransformerBlock, tp_group_size
):
    # rowwise
    hidden_dim = 3072
    requires_grad = single_block.proj_out.weight.requires_grad
    linear2_weight_data = single_block.proj_out.weight.data.T.detach().clone()
    out_weight = linear2_weight_data[:hidden_dim, ...]
    out_weight = rearrange(out_weight, "(G D) C -> G D C", G=tp_group_size)
    down_weight = linear2_weight_data.data[hidden_dim:, ...]
    down_weight = rearrange(down_weight, "(G D) C -> G D C", G=tp_group_size)
    new_linear2_weight = torch.cat([out_weight, down_weight], dim=1)
    new_linear2_weight = rearrange(new_linear2_weight, "G D C -> (G D) C")
    single_block.proj_out.weight.data.copy_(new_linear2_weight.T)
    single_block.proj_out.weight.requires_grad_(requires_grad)


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
            "norm1.linear": ColwiseParallel(output_layouts=Replicate()),
            "ff.net.0.proj": ColwiseParallel(),
            "ff.net.2": RowwiseParallel(),
            "attn.add_q_proj": ColwiseParallel(),
            "attn.add_k_proj": ColwiseParallel(),
            "attn.add_v_proj": ColwiseParallel(),
            "attn.to_add_out": RowwiseParallel(),
            "norm1_context.linear": ColwiseParallel(output_layouts=Replicate()),
            "ff_context.net.0.proj": ColwiseParallel(),
            "ff_context.net.2": RowwiseParallel(),
        }
        parallelize_module(
            module=block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    for name, block in model.single_transformer_blocks.named_children():
        prepare_proj_out_weight(block, tp_mesh.size())
        block.attn.heads //= tp_mesh.size()
        layer_plan = {
            "attn.to_q": ColwiseParallel(),
            "attn.to_k": ColwiseParallel(),
            "attn.to_v": ColwiseParallel(),
            "proj_mlp": ColwiseParallel(),
            "proj_out": RowwiseParallel(),
            "norm.linear": ColwiseParallel(output_layouts=Replicate()),
        }
        parallelize_module(
            module=block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )
