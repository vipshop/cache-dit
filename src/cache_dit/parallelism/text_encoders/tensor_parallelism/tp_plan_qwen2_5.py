import torch
from typing import Union
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLTextModel
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer

from torch.distributed import DeviceMesh

from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from cache_dit.logger import init_logger
from cache_dit.utils import maybe_empty_cache
from cache_dit.parallelism.config import ParallelismConfig

from .tp_plan_registers import (
    TextEncoderTensorParallelismPlanner,
    TextEncoderTensorParallelismPlannerRegister,
)

logger = init_logger(__name__)


# Text Encoder for Qwen-Image, HunyuanImage-2.1, HunyuanVideo-1.5, Kandinsky-5 series models.
@TextEncoderTensorParallelismPlannerRegister.register("Qwen2_5_VLTextModel")
@TextEncoderTensorParallelismPlannerRegister.register("Qwen2_5_VLForConditionalGeneration")
class Qwen2_5_VLTensorParallelismPlanner(TextEncoderTensorParallelismPlanner):
    def apply(
        self,
        text_encoder: torch.nn.Module,
        parallelism_config: ParallelismConfig,
        **kwargs,
    ) -> torch.nn.Module:
        assert isinstance(
            text_encoder, (Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLTextModel)
        ), (
            "Qwen2_5_VLTensorParallelismPlanner can only be applied to "
            "Qwen2_5_VLForConditionalGeneration or Qwen2_5_VLTextModel"
        )
        tp_mesh = self.mesh(parallelism_config=parallelism_config)
        text_encoder = self.parallelize_text_encoder(
            text_encoder=text_encoder,
            tp_mesh=tp_mesh,
        )

        return text_encoder

    def parallelize_text_encoder(
        self,
        text_encoder: Union[Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLTextModel],
        tp_mesh: DeviceMesh,
    ):

        if isinstance(text_encoder, Qwen2_5_VLForConditionalGeneration):
            model = text_encoder.model.language_model
        else:
            model = text_encoder

        for _, block in model.layers.named_children():
            assert isinstance(block, Qwen2_5_VLDecoderLayer)
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

        if isinstance(text_encoder, Qwen2_5_VLForConditionalGeneration):
            text_encoder.model.language_model = model
        else:
            text_encoder = model

        maybe_empty_cache()

        return text_encoder
