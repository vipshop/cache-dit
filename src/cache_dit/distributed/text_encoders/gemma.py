import torch
from typing import Dict, List, Tuple, Union
from transformers.models.t5gemma.modeling_t5gemma import T5GemmaEncoder
from transformers import (
  GemmaModel,
  Gemma2Model,
  Gemma3Model,
  GemmaForCausalLM,
  Gemma2ForCausalLM,
  Gemma3ForCausalLM,
  Gemma3ForConditionalGeneration,
)
from transformers.models.gemma.modeling_gemma import GemmaDecoderLayer
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer
from transformers.models.t5gemma.modeling_t5gemma import T5GemmaEncoderLayer

try:
  # transformers 5: Motif-Video ships `T5Gemma2Encoder` (a separate module from `t5gemma`).
  from transformers.models.t5gemma2.modeling_t5gemma2 import (
    T5Gemma2Encoder,
    T5Gemma2EncoderLayer,
  )
except ImportError:  # older transformers
  T5Gemma2Encoder = None
  T5Gemma2EncoderLayer = None

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

_supported_gemma_classes = (
  T5GemmaEncoder,
  GemmaModel,
  Gemma2Model,
  Gemma3Model,
  GemmaForCausalLM,
  Gemma2ForCausalLM,
  Gemma3ForCausalLM,
  Gemma3ForConditionalGeneration,
)
if T5Gemma2Encoder is not None:
  _supported_gemma_classes = _supported_gemma_classes + (T5Gemma2Encoder, )

_supported_layer_classes = (
  GemmaDecoderLayer,
  Gemma2DecoderLayer,
  Gemma3DecoderLayer,
  T5GemmaEncoderLayer,
)
if T5Gemma2EncoderLayer is not None:
  _supported_layer_classes = _supported_layer_classes + (T5Gemma2EncoderLayer, )


# Text Encoder Lumina-Image, prx series models.
@TextEncoderTensorParallelismPlannerRegister.register("T5Gemma2Encoder")
@TextEncoderTensorParallelismPlannerRegister.register("T5GemmaEncoder")
@TextEncoderTensorParallelismPlannerRegister.register("GemmaModel")
@TextEncoderTensorParallelismPlannerRegister.register("Gemma2Model")
@TextEncoderTensorParallelismPlannerRegister.register("Gemma3Model")
@TextEncoderTensorParallelismPlannerRegister.register("GemmaForCausalLM")
@TextEncoderTensorParallelismPlannerRegister.register("Gemma2ForCausalLM")
@TextEncoderTensorParallelismPlannerRegister.register("Gemma3ForCausalLM")
@TextEncoderTensorParallelismPlannerRegister.register("Gemma3ForConditionalGeneration")
class GemmaTensorParallelismPlanner(TextEncoderTensorParallelismPlanner):

  def _apply(
    self,
    text_encoder: torch.nn.Module,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    assert isinstance(
      text_encoder, _supported_gemma_classes
    ), "GemmaTensorParallelismPlanner can only be applied to Gemma Language Models."
    tp_mesh = self.mesh(parallelism_config=parallelism_config)
    text_encoder, layer_plans = self.parallelize_text_encoder(
      text_encoder=text_encoder,
      tp_mesh=tp_mesh,
    )

    return text_encoder, layer_plans

  @staticmethod
  def _resolve_language_model(
    text_encoder: torch.nn.Module,
  ) -> Tuple[torch.nn.Module, Union[torch.nn.Module, None], Union[str, None]]:
    """Unwrap the text stack and report the attribute it hangs off.

    Gemma3 ships as a multimodal container whose text stack sits at a different depth per
    transformers version: `language_model.model.layers` in v4, `model.language_model.layers` in v5.
    `T5Gemma2Encoder` (transformers 5, Motif-Video) nests the stack at `text_model.layers`;
    its `vision_tower` / `multi_modal_projector` are deliberately left untouched.

    :param text_encoder: Gemma-family text encoder, possibly a multimodal container.
    :returns: `(module, owner, attr)` where `owner.attr` is the module, or `(module, None, None)`
      when the module holding `layers` is the encoder itself.
    """
    if isinstance(text_encoder, (GemmaForCausalLM, Gemma2ForCausalLM, Gemma3ForCausalLM)):
      return text_encoder.model, text_encoder, "model"

    module = text_encoder
    owner, attr = None, None
    while not hasattr(module, "layers"):
      for name in ("language_model", "model", "text_model"):
        child = getattr(module, name, None)
        if isinstance(child, torch.nn.Module) and child is not module:
          owner, attr = module, name
          module = child
          break
      else:
        return module, None, None
    return module, owner, attr

  def parallelize_text_encoder(
    self,
    text_encoder: Union[
      T5GemmaEncoder,
      GemmaModel,
      Gemma2Model,
      Gemma3Model,
      GemmaForCausalLM,
      Gemma2ForCausalLM,
      Gemma3ForCausalLM,
      Gemma3ForConditionalGeneration,
    ],
    tp_mesh: DeviceMesh,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:

    model, model_owner, model_owner_attr = self._resolve_language_model(text_encoder)

    if not hasattr(model, "layers"):
      raise AttributeError(
        f"{model.__class__.__name__} object has no attribute 'layers'. "
        "If this is a multi-modal Gemma3 model, expected the language model to be "
        "under `language_model` (and optionally `language_model.model`).")

    assert isinstance(model, torch.nn.Module), "model must be a torch.nn.Module"
    layer_plans = []
    for _, block in model.layers.named_children():
      assert isinstance(block, _supported_layer_classes), (
        f"Unsupported layer type {block.__class__.__name__} for Gemma TP. "
        "Expected one of: "
        f"{[cls.__name__ for cls in _supported_layer_classes]}.")
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

    if model_owner is not None:
      # Write the (already parallelized in place) stack back to the attribute it came from, so the
      # container keeps pointing at the module whose layers now hold the sharded projections.
      setattr(model_owner, model_owner_attr, model)
    else:
      text_encoder = model

    maybe_empty_cache()

    return text_encoder, layer_plans
