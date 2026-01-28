import torch
from diffusers.models.transformers.transformer_flux2 import (
    Flux2SingleTransformerBlock,
    Flux2TransformerBlock,
    Flux2Transformer2DModel,
)
from einops import rearrange

from torch.distributed import DeviceMesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

from cache_dit.logger import init_logger
from cache_dit.parallelism.config import ParallelismConfig
from cache_dit.utils import maybe_empty_cache
from cache_dit.platforms import current_platform

from .tp_plan_registers import TensorParallelismPlanner, TensorParallelismPlannerRegister
from .tp_utils import shard_divisible_attr

logger = init_logger(__name__)


@TensorParallelismPlannerRegister.register("Flux2Transformer")
class Flux2TensorParallelismPlanner(TensorParallelismPlanner):
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

    @classmethod
    def rerangege_swiglu_weight(cls, weight: torch.Tensor, tp_size: int):
        weight = rearrange(weight, "r (g h d) -> r (h g d)", g=2, h=tp_size)
        return weight

    @classmethod
    def rearrange_feedforward_weight(cls, block: Flux2TransformerBlock, tp_size: int):

        block.ff.linear_in.weight.data = cls.rerangege_swiglu_weight(
            block.ff.linear_in.weight.data.T, tp_size
        ).T
        block.ff_context.linear_in.weight.data = cls.rerangege_swiglu_weight(
            block.ff_context.linear_in.weight.data.T, tp_size
        ).T

    @classmethod
    def rearrange_singleblock_weight(cls, block: Flux2SingleTransformerBlock, tp_size: int):
        attn = block.attn
        to_qkv_mlp_proj_weight = attn.to_qkv_mlp_proj.weight.data.T
        qkv, mlp = torch.split(
            to_qkv_mlp_proj_weight,
            [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor],
            dim=-1,
        )

        mlp = cls.rerangege_swiglu_weight(mlp, tp_size)

        def rerangege_qkv_weight(weight: torch.Tensor, tp_size: int):
            weight = rearrange(weight, "r (g h d) -> r (h g d)", g=3, h=tp_size)
            return weight

        qkv = rerangege_qkv_weight(qkv, tp_size)
        qkv = rearrange(qkv, "r (h d) -> r h d", h=tp_size)
        mlp = rearrange(mlp, "r (h d) -> r h d", h=tp_size)
        to_qkv_mlp_proj_weight = torch.cat([qkv, mlp], dim=-1)
        to_qkv_mlp_proj_weight = to_qkv_mlp_proj_weight.flatten(1)
        attn.to_qkv_mlp_proj.weight.data = to_qkv_mlp_proj_weight.T

        # rearrange out projection weight
        out_weight = attn.to_out.weight.data.T
        # FLUX.2-dev, FLUX.2-klein, divide by 4
        attn_out_dim = out_weight.shape[0] // 4
        attn_out_weight = out_weight[:attn_out_dim, ...]
        mlp_out_weight = out_weight[attn_out_dim:, ...]

        attn_out_weight = rearrange(attn_out_weight, "(g d) c -> g d c", g=tp_size)
        mlp_out_weight = rearrange(mlp_out_weight, "(g d) c -> g d c", g=tp_size)

        new_out_weight = torch.cat([attn_out_weight, mlp_out_weight], dim=1)
        new_out_weight = rearrange(new_out_weight, "g d c -> (g d) c")
        attn.to_out.weight.data = new_out_weight.T

    def parallelize_transformer(
        self,
        transformer: Flux2Transformer2DModel,
        tp_mesh: DeviceMesh,
    ):
        tp_size = tp_mesh.get_group().size()
        for _, block in transformer.transformer_blocks.named_children():
            # moving to cuda speed up the rearrangement process significantly
            old_device = next(block.parameters()).device
            block.to(current_platform.device_type)
            self.rearrange_feedforward_weight(block, tp_size)
            block.to(old_device)
            shard_divisible_attr(
                block.attn,
                "heads",
                tp_size,
                what="attn",
                context="Flux2TensorParallelismPlanner",
            )
            layer_plan = {
                "attn.to_q": ColwiseParallel(),
                "attn.to_k": ColwiseParallel(),
                "attn.to_v": ColwiseParallel(),
                "attn.to_out.0": RowwiseParallel(),
                "ff.linear_in": ColwiseParallel(),
                "ff.linear_out": RowwiseParallel(),
                "attn.add_q_proj": ColwiseParallel(),
                "attn.add_k_proj": ColwiseParallel(),
                "attn.add_v_proj": ColwiseParallel(),
                "attn.to_add_out": RowwiseParallel(),
                "ff_context.linear_in": ColwiseParallel(),
                "ff_context.linear_out": RowwiseParallel(),
            }

            parallelize_module(
                module=block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_plan,
            )
        maybe_empty_cache()

        for _, block in transformer.single_transformer_blocks.named_children():
            # moving to cuda speed up the rearrangement process significantly
            old_device = next(block.parameters()).device
            block.to(current_platform.device_type)
            self.rearrange_singleblock_weight(block, tp_size)
            block.to(old_device)
            shard_divisible_attr(
                block.attn,
                "heads",
                tp_size,
                what="attn",
                context="Flux2TensorParallelismPlanner(single_block)",
            )
            shard_divisible_attr(
                block.attn,
                "inner_dim",
                tp_size,
                what="attn",
                context="Flux2TensorParallelismPlanner(single_block)",
            )
            shard_divisible_attr(
                block.attn,
                "mlp_hidden_dim",
                tp_size,
                what="attn",
                context="Flux2TensorParallelismPlanner(single_block)",
            )
            layer_plan = {
                "attn.to_qkv_mlp_proj": ColwiseParallel(),
                "attn.to_out": RowwiseParallel(),
            }
            parallelize_module(
                module=block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_plan,
            )
        maybe_empty_cache()

        return transformer
