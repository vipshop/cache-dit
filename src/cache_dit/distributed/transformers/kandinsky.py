from typing import Optional

import torch
from diffusers.models.modeling_utils import ModelMixin
from ...distributed.core import (
  _ContextParallelInput,
  _ContextParallelOutput,
  _ContextParallelModelPlan,
)

try:
  from diffusers import Kandinsky5Transformer3DModel
except ImportError as exc:
  raise ImportError(
    "Kandinsky5 context parallelism requires diffusers with Kandinsky5Transformer3DModel "
    "support. Please install a recent diffusers version from source: \n"
    "pip3 install git+https://github.com/huggingface/diffusers.git") from exc

from ...logger import init_logger
from ..config import ParallelismConfig
from .register import (
  ContextParallelismPlanner,
  ContextParallelismPlannerRegister,
)

logger = init_logger(__name__)


# NOTE: NOT support sparse attention for Kandinsky5 yet.
@ContextParallelismPlannerRegister.register("Kandinsky5")
class Kandinsky5ContextParallelismPlanner(ContextParallelismPlanner):

  def _apply(
    self,
    transformer: Optional[torch.nn.Module | ModelMixin] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:

    assert isinstance(transformer, Kandinsky5Transformer3DModel
                      ), "Transformer must be an instance of Kandinsky5Transformer3DModel"

    num_blocks = len(transformer.visual_transformer_blocks)
    _cp_plan = {
      "visual_transformer_blocks.0": {
        "visual_embed": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
      },
      "visual_transformer_blocks.*": {
        "text_embed": _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        "rope": _ContextParallelInput(split_dim=1, expected_dims=6, split_output=False),
      },
      # NOTE: Need to gather the visual_embed before final out_layer, because
      # the flatten operation before out_layer needs the full visual_embed.
      f"visual_transformer_blocks.{num_blocks - 1}":
      _ContextParallelOutput(gather_dim=1, expected_dims=3),
    }
    return _cp_plan
