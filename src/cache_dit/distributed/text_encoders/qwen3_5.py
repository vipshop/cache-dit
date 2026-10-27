import torch
from typing import Dict, List, Tuple
from torch.distributed.tensor.parallel import (
  ColwiseParallel,
  ParallelStyle,
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


# Text Encoder for MOSS-Image (Qwen3.5, hybrid GatedDeltaNet + full attention).
@TextEncoderTensorParallelismPlannerRegister.register("Qwen3_5Model")
class Qwen3_5TensorParallelismPlanner(TextEncoderTensorParallelismPlanner):
  """Tensor-parallel planner for the Qwen3.5 text stack.

  Qwen3.5 interleaves `linear_attention` (GatedDeltaNet) and `full_attention` layers, and
  transformers ships no `base_model_tp_plan` for it, so the plan is deliberately conservative:

  * every decoder layer shards its MLP (`gate_proj`/`up_proj` colwise, `down_proj` rowwise). The
    residual stream stays full-length after each all-reduce, so the layers around it are unaffected;
  * `full_attention` layers shard `q_proj`/`k_proj`/`v_proj` colwise and `o_proj` rowwise. `q_proj`
    packs `(query, output_gate)` per head (`view(..., -1, head_dim * 2).chunk(2, -1)`), so a
    contiguous column shard keeps whole heads together and needs no weight rearrangement. GQA keeps
    its ratio (`num_key_value_groups`) and `repeat_kv` derives the head counts from the tensor, so
    neither needs extra handling;
  * `linear_attention` layers stay replicated: their conv/scan kernels are DTensor-unsafe, and the
    FLA fallback path materializes the whole layer anyway.

  There is no fallback planner if this one is missing, so `--parallel-text` fails loudly instead of
  silently running the encoder replicated.
  """

  def _apply(
    self,
    text_encoder: torch.nn.Module,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    tp_mesh = self.mesh(parallelism_config=parallelism_config)
    text_encoder, layer_plans = self.parallelize_text_encoder(
      text_encoder=text_encoder,
      tp_mesh=tp_mesh,
    )
    return text_encoder, layer_plans

  def parallelize_text_encoder(
    self,
    text_encoder: torch.nn.Module,
    tp_mesh,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    # `Qwen3_5Model` holds the text stack under `language_model`; wrappers nest it one level deeper.
    model = getattr(text_encoder, "language_model", None)
    if model is None:
      model = getattr(getattr(text_encoder, "model", None), "language_model", text_encoder)
    if not hasattr(model, "layers"):
      raise ValueError(f"Cannot find the decoder stack to shard in "
                       f"{text_encoder.__class__.__name__}.")

    layer_plans: List[Dict[str, ParallelStyle]] = []
    for layer in model.layers:
      layer_plan: Dict[str, ParallelStyle] = {
        "mlp.gate_proj": ColwiseParallel(),
        "mlp.up_proj": ColwiseParallel(),
        "mlp.down_proj": RowwiseParallel(),
      }
      if getattr(layer, "self_attn", None) is not None:
        layer_plan.update({
          "self_attn.q_proj": ColwiseParallel(),
          "self_attn.k_proj": ColwiseParallel(),
          "self_attn.v_proj": ColwiseParallel(),
          "self_attn.o_proj": RowwiseParallel(),
        })
      parallelize_module(module=layer, device_mesh=tp_mesh, parallelize_plan=layer_plan)
      layer_plans.append(layer_plan)

    logger.debug(f"Sharded {len(layer_plans)} Qwen3.5 decoder layers "
                 f"(full-attention + MLP) across tp_size={tp_mesh.size()}.")
    maybe_empty_cache()
    return text_encoder, layer_plans
