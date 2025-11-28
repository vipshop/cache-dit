import torch
from diffusers import ZImageTransformer2DModel
from torch.distributed import DeviceMesh, init_device_mesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

from cache_dit.logger import init_logger
from cache_dit.parallelism.parallel_config import ParallelismConfig

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
        def tp_shard_block(block, tp_size):
            block.attn.heads //= tp_size
            layer_plan = {
                "attention.to_q": ColwiseParallel(),
                "attention.to_k": ColwiseParallel(),
                "attention.to_v": ColwiseParallel(),
                "attention.to_out.0": RowwiseParallel(),
                "feed_forward.w1": ColwiseParallel(),
                "feed_forward.w3": ColwiseParallel(),
                "feed_forward.w2": RowwiseParallel(),
                # saving more memory at the cost of more communication
                # "adaLN_modulation.0": ColwiseParallel(output_layouts=Replicate()),
            }

            parallelize_module(
                module=block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_plan,
            )

        tp_size = tp_mesh.get_group().size()
        for _, block in transformer.noise_refiner.named_children():
            tp_shard_block(block, tp_size)
        for _, block in transformer.context_refiner.named_children():
            tp_shard_block(block, tp_size)
        for _, block in transformer.layers.named_children():
            tp_shard_block(block, tp_size)
