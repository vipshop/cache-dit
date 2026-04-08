import torch
from transformers import T5EncoderModel
from typing import Dict, List, Tuple
from torch.distributed import DeviceMesh
from torch.distributed.tensor.parallel import (
  ParallelStyle,
  ColwiseParallel,
  RowwiseParallel,
  parallelize_module,
)

from ....logger import init_logger
from ...config import ParallelismConfig

from .tp_plan_registers import (
  TextEncoderTensorParallelismPlanner,
  TextEncoderTensorParallelismPlannerRegister,
)

logger = init_logger(__name__)


# Text Encoder for FLUX.1, Chroma1-HD, CogVideoX1.5, CogView3-Plus, VisualCloze,
# HiDream, HunyuanImage 2.1, LTXVideo, mochi-preview, PixArt series models.
@TextEncoderTensorParallelismPlannerRegister.register("T5EncoderModel")
class T5EncoderTensorParallelismPlanner(TextEncoderTensorParallelismPlanner):

  def _apply(
    self,
    text_encoder: torch.nn.Module,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:  # text_encoder and layer plans
    assert isinstance(
      text_encoder,
      T5EncoderModel), "T5EncoderTensorParallelismPlanner can only be applied to T5EncoderModel"
    tp_mesh = self.mesh(parallelism_config=parallelism_config)
    text_encoder, layer_plans = self.parallelize_text_encoder(
      text_encoder=text_encoder,
      tp_mesh=tp_mesh,
    )

    return text_encoder, layer_plans

  def parallelize_text_encoder(
    self,
    text_encoder: T5EncoderModel,
    tp_mesh: DeviceMesh,
  ) -> Tuple[T5EncoderModel, List[Dict[str, ParallelStyle]]]:
    from transformers.models.t5.modeling_t5 import (
      T5Block,
      T5Attention,
      T5DenseActDense,
      T5DenseGatedActDense,
    )

    layer_plans = []
    for i, block in enumerate(text_encoder.encoder.block):
      assert isinstance(block, T5Block)
      assert isinstance(block.layer[0].SelfAttention, T5Attention)
      block.layer[0].SelfAttention.n_heads //= tp_mesh.size()
      block.layer[0].SelfAttention.inner_dim //= tp_mesh.size()
      if isinstance(block.layer[1].DenseReluDense, T5DenseActDense):
        layer_plan = {
          "layer.0.SelfAttention.q": ColwiseParallel(),
          "layer.0.SelfAttention.k": ColwiseParallel(),
          "layer.0.SelfAttention.v": ColwiseParallel(),
          "layer.0.SelfAttention.o": RowwiseParallel(),
          "layer.1.DenseReluDense.wi": ColwiseParallel(),
          "layer.1.DenseReluDense.wo": RowwiseParallel(),
        }
      elif isinstance(block.layer[1].DenseReluDense, T5DenseGatedActDense):
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
          f"Unsupported feed-forward layer type: {type(block.layer[1].DenseReluDense)}")
      if block.layer[0].SelfAttention.has_relative_attention_bias:
        layer_plan["layer.0.SelfAttention.relative_attention_bias"] = ColwiseParallel()
      parallelize_module(
        module=block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_plan,
      )
      layer_plans.append(layer_plan)

    return text_encoder, layer_plans
