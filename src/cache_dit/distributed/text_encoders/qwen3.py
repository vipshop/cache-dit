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

from ...utils import maybe_empty_cache
from ..config import ParallelismConfig
from ..linear_sharding import shard_linear_weights, validate_local_weight_shards

from .register import (
  TextEncoderTensorParallelismPlanner,
  TextEncoderTensorParallelismPlannerRegister,
)

_QWEN3_LAYER_STYLES = {
  "self_attn.q_proj": ColwiseParallel,
  "self_attn.k_proj": ColwiseParallel,
  "self_attn.v_proj": ColwiseParallel,
  "self_attn.o_proj": RowwiseParallel,
  "mlp.gate_proj": ColwiseParallel,
  "mlp.up_proj": ColwiseParallel,
  "mlp.down_proj": RowwiseParallel,
}


def _parallelize_layers(
  layers: torch.nn.ModuleList,
  tp_mesh: DeviceMesh,
  decoder_layer_type: type[torch.nn.Module],
  *,
  local_weight_shards: bool = False,
) -> List[Dict[str, ParallelStyle]]:
  layer_plan = {path: make_style() for path, make_style in _QWEN3_LAYER_STYLES.items()}
  layer_plans = []
  # Visit shared decoder blocks once in the same order as preflight.
  for _, block in layers.named_children():
    if local_weight_shards:
      for path, style in layer_plan.items():
        shard_linear_weights(block.get_submodule(path), tp_mesh, style)
    else:
      assert isinstance(block, decoder_layer_type)
    parallelize_module(module=block, device_mesh=tp_mesh, parallelize_plan=layer_plan)
    layer_plans.append(layer_plan)

  maybe_empty_cache()
  return layer_plans


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
    return self.parallelize_text_encoder(
      text_encoder=text_encoder,
      tp_mesh=tp_mesh,
    )

  def parallelize_text_encoder(
    self,
    text_encoder: Union[Qwen3Model, Qwen3ForCausalLM],
    tp_mesh: DeviceMesh,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    model = text_encoder.model if isinstance(text_encoder, Qwen3ForCausalLM) else text_encoder
    assert isinstance(model, Qwen3Model), "model must be an instance of Qwen3Model."
    return text_encoder, _parallelize_layers(model.layers, tp_mesh, Qwen3DecoderLayer)


@TextEncoderTensorParallelismPlannerRegister.register("Qwen3VLForConditionalGeneration",
                                                      local_weight_shards=True)
class Qwen3VLTextEncoderTensorParallelismPlanner(TextEncoderTensorParallelismPlanner):

  def _preflight_local_weight_shards(
    self,
    text_encoder: torch.nn.Module,
    parallelism_config: ParallelismConfig,
  ) -> DeviceMesh:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer

    tp_mesh = self.mesh(parallelism_config)
    tp_size = tp_mesh.size()

    def projections():
      model = text_encoder.model.language_model
      for attr in ("num_attention_heads", "num_key_value_heads", "intermediate_size"):
        if getattr(model.config, attr) % tp_size != 0:
          raise ValueError(f"{attr} must be divisible by the text encoder TP size {tp_size}.")
      for name, block in model.layers.named_children():
        block_path = f"model.language_model.layers.{name}"
        if not isinstance(block, Qwen3VLTextDecoderLayer):
          raise ValueError(f"{block_path} must be a Qwen3VLTextDecoderLayer, "
                           f"got {type(block).__name__}.")
        for path in _QWEN3_LAYER_STYLES:
          yield f"{block_path}.{path}", True

    validate_local_weight_shards(text_encoder, projections(), tp_mesh)
    return tp_mesh

  def _apply(
    self,
    text_encoder: torch.nn.Module,
    parallelism_config: ParallelismConfig,
    *,
    local_shard_mesh: DeviceMesh | None = None,
    **kwargs,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer

    tp_mesh = self.mesh(parallelism_config) if local_shard_mesh is None else local_shard_mesh
    layer_plans = _parallelize_layers(
      text_encoder.model.language_model.layers,
      tp_mesh,
      Qwen3VLTextDecoderLayer,
      local_weight_shards=local_shard_mesh is not None,
    )
    return text_encoder, layer_plans


@TextEncoderTensorParallelismPlannerRegister.register("Qwen3VLTextModel")
class Qwen3VLTextModelTensorParallelismPlanner(TextEncoderTensorParallelismPlanner):
  """TP for a bare Qwen3-VL text stack, e.g. PRXPixel's `text_encoder`.

  PRXPixel loads `Qwen3VLTextModel` directly (the vision tower is discarded), so unlike the
  `Qwen3VLForConditionalGeneration` planner there is no `.model.language_model` wrapper to
  unwrap. mrope is computed from `position_ids` inside the rotary embedding and never splits
  the sequence, so plain per-projection sharding is enough.
  """

  def _apply(
    self,
    text_encoder: torch.nn.Module,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer

    tp_mesh = self.mesh(parallelism_config)
    layer_plans = _parallelize_layers(text_encoder.layers, tp_mesh, Qwen3VLTextDecoderLayer)
    return text_encoder, layer_plans


@TextEncoderTensorParallelismPlannerRegister.register("Qwen3VLModel")
class Qwen3VLModelTensorParallelismPlanner(Qwen3VLTextModelTensorParallelismPlanner):
  """TP for a complete Qwen3-VL model whose text stack backs Ideogram 4."""

  def _apply(
    self,
    text_encoder: torch.nn.Module,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    _, layer_plans = super()._apply(
      text_encoder=text_encoder.language_model,
      parallelism_config=parallelism_config,
      **kwargs,
    )
    return text_encoder, layer_plans
