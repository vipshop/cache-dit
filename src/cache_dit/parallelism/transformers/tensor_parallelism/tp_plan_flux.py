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
from cache_dit.parallelism.config import ParallelismConfig

from .tp_plan_registers import (
    TensorParallelismPlanner,
    TensorParallelismPlannerRegister,
)
from .tp_utils import shard_divisible_attr

logger = init_logger(__name__)


@TensorParallelismPlannerRegister.register("Chroma")
@TensorParallelismPlannerRegister.register("HunyuanImage")
@TensorParallelismPlannerRegister.register("HunyuanVideo")
@TensorParallelismPlannerRegister.register("FluxTransformer")
class FluxTensorParallelismPlanner(TensorParallelismPlanner):
    def apply(
        self,
        transformer: torch.nn.Module,
        parallelism_config: ParallelismConfig,
        **kwargs,
    ) -> torch.nn.Module:
        tp_mesh = self.mesh(parallelism_config=parallelism_config)
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
        for _, block in transformer.transformer_blocks.named_children():
            shard_divisible_attr(
                block.attn,
                "heads",
                tp_mesh.size(),
                what="attn",
                context="FluxTensorParallelismPlanner",
            )
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

        # NOTE: special handling for FluxSingleTransformerBlock, we have to
        # rearrange the proj_out weight because it contains both out and down
        # projection weights in a single matrix.
        def rearrange_proj_out_weight(single_block: FluxSingleTransformerBlock, tp_group_size):
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

        for _, block in transformer.single_transformer_blocks.named_children():
            rearrange_proj_out_weight(block, tp_mesh.size())
            shard_divisible_attr(
                block.attn,
                "heads",
                tp_mesh.size(),
                what="attn",
                context="FluxTensorParallelismPlanner(single_block)",
            )
            layer_plan = {
                "attn.to_q": ColwiseParallel(),
                "attn.to_k": ColwiseParallel(),
                "attn.to_v": ColwiseParallel(),
                "proj_mlp": ColwiseParallel(),
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
