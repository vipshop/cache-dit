import torch
from transformers import UMT5EncoderModel
from torch.distributed import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from cache_dit.logger import init_logger
from cache_dit.parallelism.config import ParallelismConfig

from .tp_plan_registers import (
    TextEncoderTensorParallelismPlanner,
    TextEncoderTensorParallelismPlannerRegister,
)

logger = init_logger(__name__)


# Text Encoder for Wan2.1, Wan2.2, ChronoEdit, LongCat-Video, SkyReelsV2 series models.
@TextEncoderTensorParallelismPlannerRegister.register("UMT5EncoderModel")
class UMT5EncoderTensorParallelismPlanner(TextEncoderTensorParallelismPlanner):
    def apply(
        self,
        text_encoder: torch.nn.Module,
        parallelism_config: ParallelismConfig,
        **kwargs,
    ) -> torch.nn.Module:
        assert isinstance(
            text_encoder, UMT5EncoderModel
        ), "UMT5EncoderTensorParallelismPlanner can only be applied to UMT5EncoderModel"
        tp_mesh = self.mesh(parallelism_config=parallelism_config)
        text_encoder = self.parallelize_text_encoder(
            text_encoder=text_encoder,
            tp_mesh=tp_mesh,
        )

        return text_encoder

    def parallelize_text_encoder(
        self,
        text_encoder: UMT5EncoderModel,
        tp_mesh: DeviceMesh,
    ):
        from transformers.models.umt5.modeling_umt5 import (
            UMT5Block,
            UMT5Attention,
            UMT5DenseActDense,
            UMT5DenseGatedActDense,
        )

        for i, block in enumerate(text_encoder.encoder.block):
            assert isinstance(block, UMT5Block)
            assert isinstance(block.layer[0].SelfAttention, UMT5Attention)
            block.layer[0].SelfAttention.n_heads //= tp_mesh.size()
            block.layer[0].SelfAttention.inner_dim //= tp_mesh.size()
            if isinstance(block.layer[1].DenseReluDense, UMT5DenseActDense):
                layer_plan = {
                    "layer.0.SelfAttention.q": ColwiseParallel(),
                    "layer.0.SelfAttention.k": ColwiseParallel(),
                    "layer.0.SelfAttention.v": ColwiseParallel(),
                    "layer.0.SelfAttention.o": RowwiseParallel(),
                    "layer.1.DenseReluDense.wi": ColwiseParallel(),
                    "layer.1.DenseReluDense.wo": RowwiseParallel(),
                }
            elif isinstance(block.layer[1].DenseReluDense, UMT5DenseGatedActDense):
                layer_plan = {
                    "layer.0.SelfAttention.q": ColwiseParallel(),
                    "layer.0.SelfAttention.k": ColwiseParallel(),
                    "layer.0.SelfAttention.v": ColwiseParallel(),
                    "layer.0.SelfAttention.o": RowwiseParallel(),
                    "layer.1.DenseReluDense.wi_0": ColwiseParallel(),
                    "layer.1.DenseReluDense.wi_1": ColwiseParallel(),
                    "layer.1.DenseReluDense.wo": RowwiseParallel(),
                }
            else:
                raise NotImplementedError(
                    f"Unsupported feed-forward layer type: {type(block.layer[1].DenseReluDense)}"
                )
            # SelfAttention in UMT5Attention always has relative_attention_bias, nn.Embedding layer.
            if block.layer[0].SelfAttention.has_relative_attention_bias:
                layer_plan["layer.0.SelfAttention.relative_attention_bias"] = ColwiseParallel()

            parallelize_module(
                module=block,
                device_mesh=tp_mesh,
                parallelize_plan=layer_plan,
            )

        return text_encoder
