from typing import Dict, List, Optional, Tuple

import diffusers
import torch
from diffusers import QwenImageTransformer2DModel
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformerBlock
from torch.distributed import DeviceMesh
from torch.distributed._tensor import Replicate
from torch.distributed.tensor.parallel import (
  ColwiseParallel,
  ParallelStyle,
  RowwiseParallel,
  parallelize_module,
)

from ...distributed.parallel import (
  _ContextParallelInput,
  _ContextParallelModelPlan,
  _ContextParallelOutput,
)
from .async_ulysses import AsyncUlyssesRegistry
from ...logger import init_logger
from ..config import ParallelismConfig
from ..utils import shard_div_attr
from .register import (
  ContextParallelismPlanner,
  ContextParallelismPlannerRegister,
  TensorParallelismPlanner,
  TensorParallelismPlannerRegister,
)

logger = init_logger(__name__)


@ContextParallelismPlannerRegister.register("QwenImage")
class QwenImageContextParallelismPlanner(ContextParallelismPlanner):

  def _apply(
    self,
    transformer: Optional[torch.nn.Module | ModelMixin] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:

    if parallelism_config.ulysses_async and transformer is not None:
      AsyncUlyssesRegistry.apply(transformer)

    if diffusers.__version__ <= "0.36.0":
      _cp_plan = {
        "": {
          "hidden_states":
          _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
          "encoder_hidden_states":
          _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
          "encoder_hidden_states_mask":
          _ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
        },
        "pos_embed": {
          0: _ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
          1: _ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
        },
        "proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
      }
    else:
      # Make CP plan compatible with https://github.com/huggingface/diffusers/pull/12702
      _cp_plan = {
        "transformer_blocks.0": {
          "hidden_states":
          _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
          "encoder_hidden_states":
          _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        "pos_embed": {
          0: _ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
          1: _ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
        },
        "proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
      }

    zero_cond_t = getattr(transformer, "zero_cond_t", False)
    if zero_cond_t:
      # modulate_index: [b, l=seq_len], Qwen-Image-Edit-2511
      _cp_plan.update({
        "transformer_blocks.*": {
          "modulate_index": _ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
        }
      })

    return _cp_plan


@TensorParallelismPlannerRegister.register("QwenImageTransformer2DModel")
class QwenImageTensorParallelismPlanner(TensorParallelismPlanner):
  """Tensor parallelism for the double-stream Qwen-Image 1.x transformer.

  Both `attn.heads` and `attn.inner_dim` must be divided, because
  `QwenDoubleStreamAttnProcessor2_0` reshapes the column-sharded QKV with them and the two
  diffusers generations read them differently: `< 0.40` does
  `unflatten(-1, (attn.heads, -1))` (explicit head count, implied head dim), while `>= 0.40` does
  `head_dim = attn.inner_dim // attn.heads` followed by `unflatten(-1, (-1, head_dim))` (explicit
  head dim, implied head count). Dividing only `heads` keeps the `< 0.40` path correct but makes the
  `>= 0.40` path compute `2 * head_dim`, which breaks the QK-norm (`RMSNorm(head_dim)`) and the
  3-axis RoPE tables.
  """

  def _apply(
    self,
    transformer: torch.nn.Module,
    parallelism_config: ParallelismConfig,
    **kwargs,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    tp_mesh = self.mesh(parallelism_config=parallelism_config)
    transformer, layer_plans = self.parallelize_transformer(
      transformer=transformer,
      tp_mesh=tp_mesh,
    )

    return transformer, layer_plans

  def parallelize_transformer(
    self,
    transformer: QwenImageTransformer2DModel,
    tp_mesh: DeviceMesh,
  ) -> Tuple[torch.nn.Module, List[Dict[str, ParallelStyle]]]:
    tp_size = tp_mesh.size()
    layer_plans = []
    for _, block in transformer.transformer_blocks.named_children():
      assert isinstance(block, QwenImageTransformerBlock)
      shard_div_attr(block.attn, "heads", tp_size)
      shard_div_attr(block.attn, "inner_dim", tp_size)
      layer_plan = {
        "attn.to_q": ColwiseParallel(),
        "attn.to_k": ColwiseParallel(),
        "attn.to_v": ColwiseParallel(),
        "attn.to_out.0": RowwiseParallel(),
        "img_mod.1": ColwiseParallel(output_layouts=Replicate()),
        "img_mlp.net.0.proj": ColwiseParallel(),
        "img_mlp.net.2": RowwiseParallel(),
        "attn.add_q_proj": ColwiseParallel(),
        "attn.add_k_proj": ColwiseParallel(),
        "attn.add_v_proj": ColwiseParallel(),
        "attn.to_add_out": RowwiseParallel(),
        "txt_mod.1": ColwiseParallel(output_layouts=Replicate()),
        "txt_mlp.net.0.proj": ColwiseParallel(),
        "txt_mlp.net.2": RowwiseParallel(),
      }
      parallelize_module(
        module=block,
        device_mesh=tp_mesh,
        parallelize_plan=layer_plan,
      )
      layer_plans.append(layer_plan)

    return transformer, layer_plans
