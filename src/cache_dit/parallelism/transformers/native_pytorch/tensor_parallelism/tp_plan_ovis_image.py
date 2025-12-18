import torch
from diffusers.models.transformers.transformer_ovis_image import (
    OvisImageSingleTransformerBlock,
    OvisImageTransformerBlock,
)
from einops import rearrange
from torch import nn
from torch.distributed import DeviceMesh, init_device_mesh
from torch.distributed._tensor import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from cache_dit.logger import init_logger
from cache_dit.parallelism.parallel_config import ParallelismConfig

from .tp_plan_registers import (
    TensorParallelismPlanner,
    TensorParallelismPlannerRegister,
)

logger = init_logger(__name__)


@TensorParallelismPlannerRegister.register("OvisImage")
class OvisImageTensorParallelismPlanner(TensorParallelismPlanner):
    def apply(
        self,
        transformer: torch.nn.Module,
        parallelism_config: ParallelismConfig,
        **kwargs,
    ) -> torch.nn.Module:
        assert (
            parallelism_config.tp_size is not None and parallelism_config.tp_size > 1
        ), "parallel_config.tp_size must be set and greater than 1 for tensor parallelism"

        device_type = torch.accelerator.current_accelerator().type
        tp_mesh: DeviceMesh = init_device_mesh(
            device_type=device_type,
            mesh_shape=[parallelism_config.tp_size],
        )

        transformer = self.parallelize_transformer(
            transformer=transformer,
            tp_mesh=tp_mesh,
        )

        return transformer

    def parallelize_transformer(
        self,
        transformer: nn.Module,
        tp_mesh: DeviceMesh,
    ):
        # Ovis-Image use SwiGLU: self.proj = nn.Linear(dim_in, dim_out * 2, bias=bias)
        # hidden_states = self.proj(hidden_states); hidden_states, gate = hidden_states.chunk(2, dim=-1)
        def rearrange_ff_net_0_proj_weight(proj: torch.nn.Linear, tp_group_size):
            # colwise [..,Hd+Gd],Hd=Gd, linear: y=x*A^T, A:[out_dim, in_dim], x:[...,in_dim]
            # -> if tp_group_size=2, permute [...,Hd/2+Gd/2+Hd/2+Gd/2]
            # -> if tp_group_size=4, permute [...,Hd/4+Gd/4+Hd/4+Gd/4+Hd/4+Gd/4+Hd/4+Gd/4]
            # -> finally reshape to [...,(Hd+Gd)]
            dim_out = proj.weight.shape[0] // 2
            requires_grad = proj.weight.requires_grad
            linear1_weight_data = proj.weight.data.detach().clone()  # [out_dim, in_dim]
            new_linear1_weight = torch.zeros_like(linear1_weight_data)
            part1_linear1_weight_data = linear1_weight_data[:dim_out, ...]
            part2_linear1_weight_data = linear1_weight_data[dim_out:, ...]
            split_size = dim_out // tp_group_size
            for i in range(tp_group_size):
                start_idx = i * split_size
                end_idx = (i + 1) * split_size
                new_linear1_weight[i * 2 * split_size : (i * 2 + 1) * split_size, ...] = (
                    part1_linear1_weight_data[start_idx:end_idx, ...]
                )
                new_linear1_weight[(i * 2 + 1) * split_size : (i * 2 + 2) * split_size, ...] = (
                    part2_linear1_weight_data[start_idx:end_idx, ...]
                )

            proj.weight.data.copy_(new_linear1_weight)
            proj.weight.requires_grad_(requires_grad)

        for _, block in transformer.transformer_blocks.named_children():
            assert isinstance(block, OvisImageTransformerBlock)
            rearrange_ff_net_0_proj_weight(block.ff.net[0].proj, tp_mesh.size())
            rearrange_ff_net_0_proj_weight(block.ff_context.net[0].proj, tp_mesh.size())
            block.attn.heads //= tp_mesh.size()
            layer_plan = {
                "attn.to_q": ColwiseParallel(),
                "attn.to_k": ColwiseParallel(),
                "attn.to_v": ColwiseParallel(),
                "attn.to_out.0": RowwiseParallel(),
                "ff.net.0.proj": ColwiseParallel(),
                "ff.net.2": RowwiseParallel(),
                "attn.add_q_proj": ColwiseParallel(),
                "attn.add_k_proj": ColwiseParallel(),
                "attn.add_v_proj": ColwiseParallel(),
                "attn.to_add_out": RowwiseParallel(),
                "ff_context.net.0.proj": ColwiseParallel(),
                "ff_context.net.2": RowwiseParallel(),
            }

            if getattr(block.norm1, "linear", None) is not None:
                layer_plan["norm1.linear"] = ColwiseParallel(output_layouts=Replicate())
            if getattr(block.norm1_context, "linear", None) is not None:
                layer_plan["norm1_context.linear"] = ColwiseParallel(output_layouts=Replicate())
            parallelize_module(
                module=block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_plan,
            )

        # NOTE: special handling for OvisImageSingleTransformerBlock, we have to
        # rearrange the proj_out weight because it contains both out and down
        # projection weights in a single matrix.
        def rearrange_proj_out_weight(single_block: OvisImageSingleTransformerBlock, tp_group_size):
            # rowwise
            hidden_dim = single_block.attn.to_q.weight.shape[0]
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

        def rearrange_proj_mlp_weight(single_block: OvisImageSingleTransformerBlock, tp_group_size):
            # colwise [..,Hd+Gd],Hd=Gd, linear: y=x*A^T, A:[out_dim, in_dim], x:[...,in_dim]
            # -> if tp_group_size=2, permute [...,Hd/2+Gd/2+Hd/2+Gd/2]
            # -> if tp_group_size=4, permute [...,Hd/4+Gd/4+Hd/4+Gd/4+Hd/4+Gd/4+Hd/4+Gd/4]
            # -> finally reshape to [...,(Hd+Gd)]
            mlp_hidden_dim = single_block.proj_mlp.weight.shape[0] // 2
            requires_grad = single_block.proj_mlp.weight.requires_grad
            linear1_weight_data = (
                single_block.proj_mlp.weight.data.detach().clone()
            )  # [out_dim, in_dim]
            new_linear1_weight = torch.zeros_like(linear1_weight_data)
            part1_linear1_weight_data = linear1_weight_data[:mlp_hidden_dim, ...]
            part2_linear1_weight_data = linear1_weight_data[mlp_hidden_dim:, ...]
            split_size = mlp_hidden_dim // tp_group_size
            for i in range(tp_group_size):
                start_idx = i * split_size
                end_idx = (i + 1) * split_size
                new_linear1_weight[i * 2 * split_size : (i * 2 + 1) * split_size, ...] = (
                    part1_linear1_weight_data[start_idx:end_idx, ...]
                )
                new_linear1_weight[(i * 2 + 1) * split_size : (i * 2 + 2) * split_size, ...] = (
                    part2_linear1_weight_data[start_idx:end_idx, ...]
                )

            single_block.proj_mlp.weight.data.copy_(new_linear1_weight)
            single_block.proj_mlp.weight.requires_grad_(requires_grad)

        for _, block in transformer.single_transformer_blocks.named_children():
            rearrange_proj_out_weight(block, tp_mesh.size())
            block.attn.heads //= tp_mesh.size()
            rearrange_proj_mlp_weight(block, tp_mesh.size())
            block.mlp_hidden_dim //= tp_mesh.size()
            # Compute order: proj_mlp, to_q, to_k, to_v, proj_out
            # proj_mlp: dim -> self.mlp_hidden_dim * 2 -> split by mlp_hidden_dim
            layer_plan = {
                "proj_mlp": ColwiseParallel(),
                "attn.to_q": ColwiseParallel(),
                "attn.to_k": ColwiseParallel(),
                "attn.to_v": ColwiseParallel(),
                "proj_out": RowwiseParallel(),
            }
            if getattr(block.norm, "linear", None) is not None:
                layer_plan["norm.linear"] = ColwiseParallel(output_layouts=Replicate())
            parallelize_module(
                module=block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_plan,
            )
        return transformer
