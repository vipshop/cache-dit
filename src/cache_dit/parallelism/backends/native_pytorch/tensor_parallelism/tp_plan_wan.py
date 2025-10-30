import torch
from torch import nn
from torch.distributed import DeviceMesh, init_device_mesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from cache_dit.logger import init_logger
from cache_dit.parallelism.parallel_config import ParallelismConfig

from .tp_plan_registers import (
    TensorParallelismPlaner,
    TensorParallelismPlanerRegister,
)

logger = init_logger(__name__)


class DistributedRMSNorm(nn.Module):
    def __init__(
        self,
        tp_mesh: DeviceMesh,
        normalized_shape,
        eps,
        elementwise_affine,
        weight,
    ):
        super().__init__()
        self.tp_mesh = tp_mesh
        self.elementwise_affine = elementwise_affine
        self.normalized_shape = normalized_shape
        self.eps = eps
        if self.elementwise_affine:
            assert weight is not None
        self.weight = weight

    @classmethod
    def from_rmsnorm(cls, tp_mesh: DeviceMesh, rmsnorm: nn.RMSNorm):
        if not isinstance(rmsnorm, int):
            assert len(rmsnorm.normalized_shape) == 1

        if rmsnorm.weight is not None:
            tp_size = tp_mesh.get_group().size()
            tp_rank = tp_mesh.get_group().rank()
            weight = rmsnorm.weight.chunk(tp_size, dim=0)[tp_rank]
        else:
            weight = None
        norm = cls(
            tp_mesh=tp_mesh,
            normalized_shape=rmsnorm.normalized_shape,
            eps=rmsnorm.eps,
            elementwise_affine=rmsnorm.elementwise_affine,
            weight=weight,
        )
        return norm

    def forward(self, x):
        if self.elementwise_affine:
            assert x.shape[-1] == self.weight.shape[0]
        mean_square = torch.mean(x * x, dim=-1, keepdim=True)
        torch.distributed.all_reduce(
            mean_square,
            op=torch.distributed.ReduceOp.AVG,
            group=self.tp_mesh.get_group(),
        )
        root_mean_square = torch.sqrt(mean_square + self.eps)
        x_normed = x / root_mean_square
        if self.elementwise_affine:
            x_normed = x_normed * self.weight.to(device=x.device)
        assert x_normed.device.type != "cpu"
        return x_normed


@TensorParallelismPlanerRegister.register("Wan")
class WanTensorParallelismPlaner(TensorParallelismPlaner):
    def apply(
        self,
        transformer: torch.nn.Module,
        parallelism_config: ParallelismConfig,
        **kwargs,
    ) -> torch.nn.Module:
        assert (
            parallelism_config.tp_size is not None
            and parallelism_config.tp_size > 1
        ), (
            "parallel_config.tp_size must be set and greater than 1 for "
            "tensor parallelism"
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
        transformer: nn.Module,
        tp_mesh: DeviceMesh,
    ):
        for _, block in transformer.blocks.named_children():
            block.attn1.heads //= tp_mesh.size()
            block.attn2.heads //= tp_mesh.size()
            layer_plan = {
                "attn1.to_q": ColwiseParallel(),
                "attn1.to_k": ColwiseParallel(),
                "attn1.to_v": ColwiseParallel(),
                "attn1.to_out.0": RowwiseParallel(),
                "attn2.to_q": ColwiseParallel(),
                "attn2.to_k": ColwiseParallel(),
                "attn2.to_v": ColwiseParallel(),
                "attn2.to_out.0": RowwiseParallel(),
                "ffn.net.0.proj": ColwiseParallel(),
                "ffn.net.2": RowwiseParallel(),
                "attn2.add_k_proj": ColwiseParallel(),
                "attn2.add_v_proj": ColwiseParallel(),
            }
            parallelize_module(
                module=block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_plan,
            )

            block.attn1.norm_q = DistributedRMSNorm.from_rmsnorm(
                tp_mesh, block.attn1.norm_q
            )
            block.attn1.norm_k = DistributedRMSNorm.from_rmsnorm(
                tp_mesh, block.attn1.norm_k
            )
            block.attn2.norm_q = DistributedRMSNorm.from_rmsnorm(
                tp_mesh, block.attn2.norm_q
            )
            block.attn2.norm_k = DistributedRMSNorm.from_rmsnorm(
                tp_mesh, block.attn2.norm_k
            )
            if hasattr(block.attn2, "norm_added_k"):
                block.attn2.norm_added_k = DistributedRMSNorm.from_rmsnorm(
                    tp_mesh, block.attn2.norm_added_k
                )
        return transformer
