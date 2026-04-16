from typing import Optional

import torch
from diffusers.models.attention_processor import (
  Attention,
  AttnProcessor2_0,
)  # sdpa
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.dit_transformer_2d import DiTTransformer2DModel

from ...distributed.core import (
  _ContextParallelInput,
  _ContextParallelModelPlan,
  _ContextParallelOutput,
)
from ...logger import init_logger
from ..config import ParallelismConfig
from .pixart import (
  __patch_AttnProcessor2_0__call__,
  __patch_Attention_prepare_attention_mask__,
)
from .register import (
  ContextParallelismPlanner,
  ContextParallelismPlannerRegister,
)

logger = init_logger(__name__)


@ContextParallelismPlannerRegister.register("DiT")
class DiTContextParallelismPlanner(ContextParallelismPlanner):

  def _apply(
    self,
    transformer: Optional[torch.nn.Module | ModelMixin] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:
    assert transformer is not None, "Transformer must be provided."
    assert isinstance(
      transformer,
      DiTTransformer2DModel), "Transformer must be an instance of DiTTransformer2DModel"

    # Apply monkey patch to fix attention mask preparation at class level
    Attention.prepare_attention_mask = __patch_Attention_prepare_attention_mask__
    AttnProcessor2_0.__call__ = __patch_AttnProcessor2_0__call__
    if not hasattr(AttnProcessor2_0, "_parallel_config"):
      AttnProcessor2_0._parallel_config = None
    if not hasattr(AttnProcessor2_0, "_attention_backend"):
      AttnProcessor2_0._attention_backend = None

    _cp_plan = {
      "transformer_blocks.0": {
        "hidden_states": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
      },
      "proj_out_2": _ContextParallelOutput(gather_dim=1, expected_dims=3),
    }
    return _cp_plan
