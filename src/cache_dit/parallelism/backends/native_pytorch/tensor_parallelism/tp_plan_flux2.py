import torch
from diffusers.models.transformers.transformer_flux2 import (
    Flux2SingleTransformerBlock,
    Flux2TransformerBlock,
)
from einops import rearrange
from torch import nn
from torch.distributed import DeviceMesh, init_device_mesh

from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

from cache_dit.logger import init_logger
from cache_dit.parallelism.parallel_config import ParallelismConfig

from .tp_plan_registers import TensorParallelismPlanner, TensorParallelismPlannerRegister

logger = init_logger(__name__)


@TensorParallelismPlannerRegister.register("Flux2Transformer")
class Flux2TensorParallelismPlanner(TensorParallelismPlanner):
    def apply(
        self,
        transformer: torch.nn.Module,
        parallelism_config: ParallelismConfig,
        **kwargs,
    ) -> torch.nn.Module:
        assert parallelism_config.tp_size is not None and parallelism_config.tp_size > 1, (
            "parallel_config.tp_size must be set and greater than 1 for " "tensor parallelism"
        )

        device_type = torch.accelerator.current_accelerator().type
        tp_mesh: DeviceMesh = init_device_mesh(
            device_type=device_type,
            mesh_shape=[parallelism_config.tp_size],
        )

        transformer = self.parallelize_transformer(
            transformer=transformer,
            tp_mesh=tp_mesh,
        )
        # TODO: Parallelize t5 text encoder via `apply_extra`
        # abstract method and `extra_parallel_kwargs` ?

        return transformer

    def parallelize_text_encoder(
        self,
        transformer: nn.Module,
        tp_mesh: DeviceMesh,
    ):
        for _, block in transformer.model.language_model.layers.named_children():
            layer_plan = {
                "self_attn.q_proj": ColwiseParallel(),
                "self_attn.k_proj": ColwiseParallel(),
                "self_attn.v_proj": ColwiseParallel(),
                "self_attn.o_proj": RowwiseParallel(),
                "mlp.gate_proj": ColwiseParallel(),
                "mlp.up_proj": ColwiseParallel(),
                "mlp.down_proj": RowwiseParallel(),
            }

            parallelize_module(
                module=block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_plan,
            )
        return transformer

    @staticmethod
    def rerangege_swiglu_weight(weight: torch.Tensor, tp_size: int):
        weight = rearrange(weight, "r (g h d) -> r (h g d)", g=2, h=tp_size)
        return weight

    @staticmethod
    def rearrange_feedforward_weight(block: Flux2TransformerBlock, tp_size):

        block.ff.linear_in.weight.data = Flux2TensorParallelismPlanner.rerangege_swiglu_weight(
            block.ff.linear_in.weight.data.T, tp_size
        ).T
        block.ff_context.linear_in.weight.data = (
            Flux2TensorParallelismPlanner.rerangege_swiglu_weight(
                block.ff_context.linear_in.weight.data.T, tp_size
            ).T
        )

    @staticmethod
    def rearrange_singleblock_weight(block: Flux2SingleTransformerBlock, tp_size):
        attn = block.attn
        to_qkv_mlp_proj_weight = attn.to_qkv_mlp_proj.weight.data.T
        qkv, mlp = torch.split(
            to_qkv_mlp_proj_weight,
            [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor],
            dim=-1,
        )

        mlp = Flux2TensorParallelismPlanner.rerangege_swiglu_weight(mlp, tp_size)

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
        assert out_weight.shape[0] == 6144 + 6144 * 3
        attn_out_weight = out_weight[:6144, ...]
        mlp_out_weight = out_weight[6144:, ...]

        attn_out_weight = rearrange(attn_out_weight, "(g d) c -> g d c", g=tp_size)
        mlp_out_weight = rearrange(mlp_out_weight, "(g d) c -> g d c", g=tp_size)

        new_out_weight = torch.cat([attn_out_weight, mlp_out_weight], dim=1)
        new_out_weight = rearrange(new_out_weight, "g d c -> (g d) c")
        attn.to_out.weight.data = new_out_weight.T

    def parallelize_transformer(
        self,
        transformer: nn.Module,
        tp_mesh: DeviceMesh,
    ):
        tp_size = tp_mesh.get_group().size()
        for _, block in transformer.transformer_blocks.named_children():
            # moving to cuda speed up the rearrangement process significantly
            old_device = next(block.parameters()).device
            block.to("cuda")
            self.rearrange_feedforward_weight(block, tp_size)
            block.to(old_device)
            block.attn.heads //= tp_size
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

        for _, block in transformer.single_transformer_blocks.named_children():
            # moving to cuda speed up the rearrangement process significantly
            old_device = next(block.parameters()).device
            block.to("cuda")
            self.rearrange_singleblock_weight(block, tp_size)
            block.to(old_device)
            block.attn.heads //= tp_size
            block.attn.inner_dim //= tp_size
            block.attn.mlp_hidden_dim //= tp_size
            layer_plan = {
                "attn.to_qkv_mlp_proj": ColwiseParallel(),
                "attn.to_out": RowwiseParallel(),
            }
            parallelize_module(
                module=block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_plan,
            )
        return transformer
