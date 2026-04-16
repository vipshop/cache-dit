from typing import Optional

import torch
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.cogvideox_transformer_3d import CogVideoXAttnProcessor2_0

from ...distributed.core import (
  _ContextParallelInput,
  _ContextParallelModelPlan,
  _ContextParallelOutput,
)
from ...logger import init_logger
from ..config import ParallelismConfig
from .cogvideox import __patch_CogVideoXAttnProcessor2_0__call__
from .register import (
  ContextParallelismPlanner,
  ContextParallelismPlannerRegister,
)

logger = init_logger(__name__)


@ContextParallelismPlannerRegister.register("ConsisID")
class CosisIDContextParallelismPlanner(ContextParallelismPlanner):

  def _apply(
    self,
    transformer: Optional[torch.nn.Module | ModelMixin] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:

    # ConsisID uses the same attention processor as CogVideoX.
    CogVideoXAttnProcessor2_0.__call__ = __patch_CogVideoXAttnProcessor2_0__call__
    # Also need to patch the parallel config and attention backend
    if not hasattr(CogVideoXAttnProcessor2_0, "_parallel_config"):
      CogVideoXAttnProcessor2_0._parallel_config = None
    if not hasattr(CogVideoXAttnProcessor2_0, "_attention_backend"):
      CogVideoXAttnProcessor2_0._attention_backend = None

    _cp_plan = {
      "transformer_blocks.0": {
        "hidden_states":
        _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        "encoder_hidden_states":
        _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
      },
      "transformer_blocks.*": {
        "image_rotary_emb": [
          _ContextParallelInput(split_dim=0, expected_dims=2, split_output=False),
          _ContextParallelInput(split_dim=0, expected_dims=2, split_output=False),
        ],
      },
      f"transformer_blocks.{len(transformer.transformer_blocks) - 1}": [
        _ContextParallelOutput(gather_dim=1, expected_dims=3),
        _ContextParallelOutput(gather_dim=1, expected_dims=3),
      ],
    }
    return _cp_plan
