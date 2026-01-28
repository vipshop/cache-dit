from typing import Optional, Union

import torch
from torch import nn
from torch.distributed import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)
from diffusers.models.transformers.transformer_ltx import LTXVideoAttnProcessor

from cache_dit.logger import init_logger
from cache_dit.parallelism.config import ParallelismConfig

from .tp_plan_registers import (
    TensorParallelismPlanner,
    TensorParallelismPlannerRegister,
)
from .tp_utils import shard_divisible_attr

logger = init_logger(__name__)


class DistributedRMSNorm(nn.Module):
    def __init__(
        self,
        tp_mesh: DeviceMesh,
        normalized_shape: Union[int, list[int], torch.Size],
        eps: Optional[float],
        elementwise_affine: bool,
        weight: torch.nn.parameter.Parameter,
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


class SplitFreqsProcessor:
    def __init__(self, processor: LTXVideoAttnProcessor, tp_size: int, tp_rank: int):
        self.processor = processor
        self.tp_size = tp_size
        self.tp_rank = tp_rank

    @classmethod
    def from_attn_processor(cls, processor: LTXVideoAttnProcessor, tp_size: int, tp_rank: int):
        return cls(
            processor=processor,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )

    def __call__(self, *args, **kwargs):
        assert len(args) == 5
        assert isinstance(args[-1], tuple)
        image_rotary_emb = args[-1]
        assert image_rotary_emb is not None
        cos, sin = image_rotary_emb
        cos = torch.chunk(cos, self.tp_size, dim=-1)[self.tp_rank]
        sin = torch.chunk(sin, self.tp_size, dim=-1)[self.tp_rank]
        image_rotary_emb = (cos, sin)
        return self.processor(
            *(*args[:-1], image_rotary_emb),
            **kwargs,
        )


@TensorParallelismPlannerRegister.register("LTXVideo")
class LTXVideoTensorParallelismPlanner(TensorParallelismPlanner):
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
        tp_size = tp_mesh.get_group().size()
        tp_rank = tp_mesh.get_group().rank()

        def prepare_block(block: nn.Module):
            shard_divisible_attr(
                block.attn1,
                "heads",
                tp_size,
                what="attn1",
                context="LTXVideoTensorParallelismPlanner",
            )
            shard_divisible_attr(
                block.attn2,
                "heads",
                tp_size,
                what="attn2",
                context="LTXVideoTensorParallelismPlanner",
            )
            layer_plan = {
                "attn1.to_q": ColwiseParallel(),
                "attn1.to_k": ColwiseParallel(),
                "attn1.to_v": ColwiseParallel(),
                "attn1.to_out.0": RowwiseParallel(),
                "attn2.to_q": ColwiseParallel(),
                "attn2.to_k": ColwiseParallel(),
                "attn2.to_v": ColwiseParallel(),
                "attn2.to_out.0": RowwiseParallel(),
                "ff.net.0.proj": ColwiseParallel(),
                "ff.net.2": RowwiseParallel(),
            }

            parallelize_module(
                module=block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_plan,
            )

            block.attn1.norm_q = DistributedRMSNorm.from_rmsnorm(tp_mesh, block.attn1.norm_q)
            block.attn1.norm_k = DistributedRMSNorm.from_rmsnorm(tp_mesh, block.attn1.norm_k)
            block.attn2.norm_q = DistributedRMSNorm.from_rmsnorm(tp_mesh, block.attn2.norm_q)
            block.attn2.norm_k = DistributedRMSNorm.from_rmsnorm(tp_mesh, block.attn2.norm_k)

        for _, block in transformer.transformer_blocks.named_children():
            block.attn1.processor = SplitFreqsProcessor.from_attn_processor(
                processor=block.attn1.processor,
                tp_size=tp_size,
                tp_rank=tp_rank,
            )
            prepare_block(block)

        return transformer
