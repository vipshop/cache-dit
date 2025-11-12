import torch
from diffusers.models.attention_processor import CogView4AttnProcessor
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
    TensorParallelismPlanner,
    TensorParallelismPlannerRegister,
)

logger = init_logger(__name__)


class SplitImageRotaryEmbProcessor:
    """Processor to split image_rotary_emb across tensor parallel devices."""

    def __init__(
        self, processor: CogView4AttnProcessor, tp_size: int, tp_rank: int
    ):
        self.processor = processor
        self.tp_size = tp_size
        self.tp_rank = tp_rank

    @classmethod
    def from_cogview4_processor(
        cls, processor: CogView4AttnProcessor, tp_size: int, tp_rank: int
    ):
        return cls(
            processor=processor,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )

    def __call__(self, *args, **kwargs):
        image_rotary_emb = kwargs.pop("image_rotary_emb", None)

        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            # Split the rotary embeddings across the head dimension
            cos = torch.chunk(cos, self.tp_size, dim=0)[self.tp_rank]
            sin = torch.chunk(sin, self.tp_size, dim=0)[self.tp_rank]
            image_rotary_emb = (cos, sin)

        return self.processor(
            *args,
            image_rotary_emb=image_rotary_emb,
            **kwargs,
        )


@TensorParallelismPlannerRegister.register("CogView4")
class CogView4TensorParallelismPlanner(TensorParallelismPlanner):
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
        tp_size = tp_mesh.get_group().size()
        tp_rank = tp_mesh.get_group().rank()

        for _, block in transformer.transformer_blocks.named_children():
            # Replace attention processor with TP-aware version
            block.attn1.processor = (
                SplitImageRotaryEmbProcessor.from_cogview4_processor(
                    processor=block.attn1.processor,
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                )
            )
            block.attn2.processor = (
                SplitImageRotaryEmbProcessor.from_cogview4_processor(
                    processor=block.attn2.processor,
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                )
            )

            # Reduce attention heads for tensor parallelism
            block.attn1.heads //= tp_mesh.size()
            block.attn2.heads //= tp_mesh.size()

            layer_plan = {
                # Self-attention projections
                "attn1.to_q": ColwiseParallel(),
                "attn1.to_k": ColwiseParallel(),
                "attn1.to_v": ColwiseParallel(),
                "attn1.to_out.0": RowwiseParallel(),
                # Cross-attention projections
                "attn2.to_q": ColwiseParallel(),
                "attn2.to_k": ColwiseParallel(),
                "attn2.to_v": ColwiseParallel(),
                "attn2.to_out.0": RowwiseParallel(),
                # Feed-forward networks
                "ff.net.0.proj": ColwiseParallel(),
                "ff.net.2": RowwiseParallel(),
                "ff_context.net.0.proj": ColwiseParallel(),
                "ff_context.net.2": RowwiseParallel(),
            }

            parallelize_module(
                module=block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_plan,
            )

        return transformer
