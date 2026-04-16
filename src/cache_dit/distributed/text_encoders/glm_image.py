import torch
from typing import Dict, List, Tuple
from transformers import GlmImageForConditionalGeneration

from torch.distributed import DeviceMesh

from torch.distributed.tensor import Replicate
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


@TextEncoderTensorParallelismPlannerRegister.register("GlmImageForConditionalGeneration")
class GlmImageTensorParallelismPlanner(TextEncoderTensorParallelismPlanner):

  def _apply(
    self,
    text_encoder: torch.nn.Module,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    assert isinstance(
      text_encoder, GlmImageForConditionalGeneration
    ), "GlmImageTensorParallelismPlanner can only be applied to GlmImageForConditionalGeneration."
    tp_mesh = self.mesh(parallelism_config=parallelism_config)
    text_encoder, layer_plans = self.parallelize_text_encoder(
      text_encoder=text_encoder,
      tp_mesh=tp_mesh,
    )

    return text_encoder, layer_plans

  def parallelize_text_encoder(
    self,
    text_encoder: GlmImageForConditionalGeneration,
    tp_mesh: DeviceMesh,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    from transformers import GlmImageTextModel
    from transformers.models.glm_image.modeling_glm_image import GlmImageTextDecoderLayer

    model: GlmImageTextModel = text_encoder.model.language_model
    layer_plans = []

    for _, block in model.layers.named_children():
      assert isinstance(block, GlmImageTextDecoderLayer)
      layer_plan = {
        "self_attn.q_proj": ColwiseParallel(),
        "self_attn.k_proj": ColwiseParallel(),
        "self_attn.v_proj": ColwiseParallel(),
        "self_attn.o_proj": RowwiseParallel(),
        # We need to replicate the outputs after ColwiseParallel
        # due to the `chunk` operation.
        # WARN: Tensor parallel for MLP layers will lead to lower performance
        # for GlmImagePipeline, please benchmark before using. I guess it's
        # because the kv cache passing logic in GlmImageTextModel is not
        # compatible with tensor parallelism in cache-dit.
        "mlp.gate_up_proj": ColwiseParallel(output_layouts=Replicate()),
        "mlp.down_proj": ColwiseParallel(output_layouts=Replicate()),
      }
      parallelize_module(
        module=block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_plan,
      )
      layer_plans.append(layer_plan)

    text_encoder.model.language_model = model

    maybe_empty_cache()

    return text_encoder, layer_plans
