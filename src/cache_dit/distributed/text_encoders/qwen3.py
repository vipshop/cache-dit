import torch
from typing import Dict, List, Tuple, Union
from transformers import Qwen3Model, Qwen3ForCausalLM
from torch.distributed import DeviceMesh

from torch.distributed.tensor.parallel import (
  ParallelStyle,
  ColwiseParallel,
  RowwiseParallel,
  parallelize_module,
)

from ...logger import init_logger
from ...utils import maybe_empty_cache
from ..config import ParallelismConfig

from .register import (
  TextEncoderTensorParallelismPlanner,
  TextEncoderTensorParallelismPlannerRegister,
)

logger = init_logger(__name__)


# Text Encoder for Z-Image, Ovis-Image
@TextEncoderTensorParallelismPlannerRegister.register("Qwen3Model")
@TextEncoderTensorParallelismPlannerRegister.register("Qwen3ForCausalLM")
class Qwen3TensorParallelismPlanner(TextEncoderTensorParallelismPlanner):

  def _apply(
    self,
    text_encoder: torch.nn.Module,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    assert isinstance(
      text_encoder,
      (Qwen3Model, Qwen3ForCausalLM
       )), "Qwen3TensorParallelismPlanner can only be applied to Qwen3 Language Models."
    tp_mesh = self.mesh(parallelism_config=parallelism_config)
    text_encoder, layer_plans = self.parallelize_text_encoder(
      text_encoder=text_encoder,
      tp_mesh=tp_mesh,
    )

    return text_encoder, layer_plans

  def parallelize_text_encoder(
    self,
    text_encoder: Union[Qwen3Model, Qwen3ForCausalLM],
    tp_mesh: DeviceMesh,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    if isinstance(text_encoder, Qwen3ForCausalLM):
      model = text_encoder.model
    else:
      model = text_encoder

    assert isinstance(model, Qwen3Model), "model must be an instance of Qwen3Model."
    layer_plans = []
    for _, block in model.layers.named_children():
      assert isinstance(block, Qwen3DecoderLayer)
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
      layer_plans.append(layer_plan)

    if isinstance(text_encoder, Qwen3ForCausalLM):
      text_encoder.model = model
    else:
      text_encoder = model

    maybe_empty_cache()

    return text_encoder, layer_plans
