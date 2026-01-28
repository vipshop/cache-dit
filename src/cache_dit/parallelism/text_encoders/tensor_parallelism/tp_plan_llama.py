import torch
from typing import Union
from transformers import LlamaModel, LlamaForCausalLM
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


# Text Encoder HunyunVideo series models.
@TextEncoderTensorParallelismPlannerRegister.register("LlamaModel")
@TextEncoderTensorParallelismPlannerRegister.register("LlamaForCausalLM")
class LlamaTensorParallelismPlanner(TextEncoderTensorParallelismPlanner):
    def apply(
        self,
        text_encoder: torch.nn.Module,
        parallelism_config: ParallelismConfig,
        **kwargs,
    ) -> torch.nn.Module:
        assert isinstance(
            text_encoder, (LlamaModel, LlamaForCausalLM)
        ), "Qwen3TensorParallelismPlanner can only be applied to Llama Language Models."
        tp_mesh = self.mesh(parallelism_config=parallelism_config)
        text_encoder = self.parallelize_text_encoder(
            text_encoder=text_encoder,
            tp_mesh=tp_mesh,
        )

        return text_encoder

    def parallelize_text_encoder(
        self,
        text_encoder: Union[LlamaModel, LlamaForCausalLM],
        tp_mesh: DeviceMesh,
    ):
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        if isinstance(text_encoder, LlamaForCausalLM):
            model = text_encoder.model
        else:
            model = text_encoder

        assert isinstance(model, LlamaModel), "model must be an instance of LlamaModel."
        for _, block in model.layers.named_children():
            assert isinstance(block, LlamaDecoderLayer)
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

        if isinstance(text_encoder, LlamaForCausalLM):
            text_encoder.model = model
        else:
            text_encoder = model

        maybe_empty_cache()

        return text_encoder
