import torch
from diffusers import ZImageTransformer2DModel
from torch.distributed import DeviceMesh, init_device_mesh

# from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

from cache_dit.logger import init_logger
from cache_dit.parallelism.parallel_config import ParallelismConfig

# from cache_dit.utils import maybe_empty_cache

from .tp_plan_registers import TensorParallelismPlanner, TensorParallelismPlannerRegister

logger = init_logger(__name__)


@TensorParallelismPlannerRegister.register("ZImage")
class ZImageTensorParallelismPlanner(TensorParallelismPlanner):
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

        return transformer

    def parallelize_transformer(
        self,
        transformer: ZImageTransformer2DModel,
        tp_mesh: DeviceMesh,
    ):
        pass
        # tp_size = tp_mesh.get_group().size()
        # for _, block in transformer.transformer_blocks.named_children():
        #     # moving to cuda speed up the rearrangement process significantly
        #     old_device = next(block.parameters()).device
        #     block.to("cuda")
        #     self.rearrange_feedforward_weight(block, tp_size)
        #     block.to(old_device)
        #     block.attn.heads //= tp_size
        #     layer_plan = {
        #         "attn.to_q": ColwiseParallel(),
        #         "attn.to_k": ColwiseParallel(),
        #         "attn.to_v": ColwiseParallel(),
        #         "attn.to_out.0": RowwiseParallel(),
        #         "ff.linear_in": ColwiseParallel(),
        #         "ff.linear_out": RowwiseParallel(),
        #         "attn.add_q_proj": ColwiseParallel(),
        #         "attn.add_k_proj": ColwiseParallel(),
        #         "attn.add_v_proj": ColwiseParallel(),
        #         "attn.to_add_out": RowwiseParallel(),
        #         "ff_context.linear_in": ColwiseParallel(),
        #         "ff_context.linear_out": RowwiseParallel(),
        #     }

        #     parallelize_module(
        #         module=block,
        #         device_mesh=tp_mesh,
        #         parallelize_plan=layer_plan,
        #     )
        # maybe_empty_cache()

        # for _, block in transformer.single_transformer_blocks.named_children():
        #     # moving to cuda speed up the rearrangement process significantly
        #     old_device = next(block.parameters()).device
        #     block.to("cuda")
        #     self.rearrange_singleblock_weight(block, tp_size)
        #     block.to(old_device)
        #     block.attn.heads //= tp_size
        #     block.attn.inner_dim //= tp_size
        #     block.attn.mlp_hidden_dim //= tp_size
        #     layer_plan = {
        #         "attn.to_qkv_mlp_proj": ColwiseParallel(),
        #         "attn.to_out": RowwiseParallel(),
        #     }
        #     parallelize_module(
        #         module=block,
        #         device_mesh=tp_mesh,
        #         parallelize_plan=layer_plan,
        #     )
        # maybe_empty_cache()

        # return transformer
