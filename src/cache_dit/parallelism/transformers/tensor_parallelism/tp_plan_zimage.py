import torch
from torch.distributed import DeviceMesh
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

from cache_dit.logger import init_logger
from cache_dit.parallelism.config import ParallelismConfig

from .tp_plan_registers import TensorParallelismPlanner, TensorParallelismPlannerRegister
from .tp_utils import shard_divisible_attr

logger = init_logger(__name__)


@TensorParallelismPlannerRegister.register("Lumina2")
@TensorParallelismPlannerRegister.register("ZImage")
class ZImageTensorParallelismPlanner(TensorParallelismPlanner):
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
        transformer: torch.nn.Module,
        tp_mesh: DeviceMesh,
    ):
        class_name = transformer.__class__.__name__

        def tp_shard_block(block, tp_size):
            attn_mod_name = "attention" if class_name.startswith("ZImage") else "attn"
            ff_linear_name = "w" if class_name.startswith("ZImage") else "linear_"
            attn = getattr(block, attn_mod_name)
            shard_divisible_attr(
                attn,
                "heads",
                tp_size,
                what=attn_mod_name,
                context="ZImageTensorParallelismPlanner",
            )
            layer_plan = {
                f"{attn_mod_name}.to_q": ColwiseParallel(),
                f"{attn_mod_name}.to_k": ColwiseParallel(),
                f"{attn_mod_name}.to_v": ColwiseParallel(),
                f"{attn_mod_name}.to_out.0": RowwiseParallel(),
                f"feed_forward.{ff_linear_name}1": ColwiseParallel(),
                f"feed_forward.{ff_linear_name}3": ColwiseParallel(),
                f"feed_forward.{ff_linear_name}2": RowwiseParallel(),
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

        return transformer
