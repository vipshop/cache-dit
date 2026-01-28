import torch
from diffusers.models.attention_processor import MochiAttnProcessor2_0
from diffusers.models.transformers.transformer_mochi import (
    MochiTransformerBlock,
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


class SplitFreqsProcessor:
    def __init__(self, processor: MochiAttnProcessor2_0, tp_size: int, tp_rank: int):
        self.processor = processor
        self.tp_size = tp_size
        self.tp_rank = tp_rank

    @classmethod
    def from_mochi_processor(cls, processor: MochiAttnProcessor2_0, tp_size: int, tp_rank: int):
        return cls(
            processor=processor,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )

    def __call__(self, *args, **kwargs):
        image_rotary_emb = kwargs.pop("image_rotary_emb", None)
        assert image_rotary_emb is not None
        cos, sin = image_rotary_emb
        cos = torch.chunk(cos, self.tp_size, dim=-2)[self.tp_rank]
        sin = torch.chunk(sin, self.tp_size, dim=-2)[self.tp_rank]
        image_rotary_emb = (cos, sin)
        return self.processor(
            *args,
            image_rotary_emb=image_rotary_emb,
            **kwargs,
        )


@TensorParallelismPlannerRegister.register("Mochi")
class MochiTensorParallelismPlanner(TensorParallelismPlanner):
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

    @staticmethod
    def rearrange_feedforward_weight(block: MochiTransformerBlock, tp_size):
        def rerangege_swiglu_weight(weight: torch.Tensor, tp_size: int):
            weight = rearrange(weight, "r (g h d) -> r (h g d)", g=2, h=tp_size)
            return weight

        block.ff.net[0].proj.weight.data = rerangege_swiglu_weight(
            block.ff.net[0].proj.weight.data.T, tp_size
        ).T
        block.ff_context.net[0].proj.weight.data = rerangege_swiglu_weight(
            block.ff_context.net[0].proj.weight.data.T, tp_size
        ).T

    def parallelize_transformer(
        self,
        transformer: nn.Module,
        tp_mesh: DeviceMesh,
    ):

        tp_size = tp_mesh.get_group().size()
        tp_rank = tp_mesh.get_group().rank()

        for name, block in transformer.transformer_blocks.named_children():
            if block.context_pre_only:
                logger.info(f"Skipping tensor parallelism for context pre-only block: {name}")
                continue
            block.attn1.processor = SplitFreqsProcessor.from_mochi_processor(
                processor=block.attn1.processor,
                tp_size=tp_size,
                tp_rank=tp_rank,
            )

            self.rearrange_feedforward_weight(block, tp_size)
            shard_divisible_attr(
                block.attn1,
                "heads",
                tp_size,
                what="attn1",
                context="MochiTensorParallelismPlanner",
            )
            layer_plan = {
                "attn1.to_q": ColwiseParallel(),
                "attn1.to_k": ColwiseParallel(),
                "attn1.to_v": ColwiseParallel(),
                "attn1.to_out.0": RowwiseParallel(),
                "attn1.add_q_proj": ColwiseParallel(),
                "attn1.add_k_proj": ColwiseParallel(),
                "attn1.add_v_proj": ColwiseParallel(),
                "attn1.to_add_out": RowwiseParallel(),
                "ff.net.0.proj": ColwiseParallel(),
                "ff.net.2": RowwiseParallel(),
                "ff_context.net.0.proj": ColwiseParallel(),
                "ff_context.net.2": RowwiseParallel(),
                "norm1.linear": ColwiseParallel(output_layouts=Replicate()),
                "norm1_context.linear": ColwiseParallel(output_layouts=Replicate()),
            }

            parallelize_module(
                module=block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_plan,
            )

        return transformer
