from typing import Optional

import torch
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.transformer_chroma import ChromaTransformer2DModel

from ...distributed import (
  _ContextParallelInput,
  _ContextParallelModelPlan,
  _ContextParallelOutput,
)
from ...logger import init_logger
from ..config import ParallelismConfig
from .register import (
  ContextParallelismPlanner,
  ContextParallelismPlannerRegister,
)

logger = init_logger(__name__)


@ContextParallelismPlannerRegister.register("Chroma")
class ChromaContextParallelismPlanner(ContextParallelismPlanner):

  def _apply(
    self,
    transformer: Optional[torch.nn.Module | ModelMixin] = None,
    parallelism_config: Optional[ParallelismConfig] = None,
    **kwargs,
  ) -> _ContextParallelModelPlan:

    # NOTE: Diffusers native CP plan still not supported
    # for Chroma now.
    self._cp_planner_preferred_native_diffusers = False

    if transformer is not None and self._cp_planner_preferred_native_diffusers:
      assert isinstance(
        transformer,
        ChromaTransformer2DModel), "Transformer must be an instance of ChromaTransformer2DModel"
      if hasattr(transformer, "_cp_plan"):
        if transformer._cp_plan is not None:
          return transformer._cp_plan

    # Otherwise, use the custom CP plan defined here, this maybe
    # a little different from the native diffusers implementation
    # for some models.
    _cp_plan = {
      # Here is a Transformer level CP plan for Chroma, which will
      # only apply the only 1 split hook (pre_forward) on the forward
      # of Transformer, and gather the output after Transformer forward.
      # Pattern of transformer forward, split_output=False:
      #     un-split input -> splited input (inside transformer)
      # Pattern of the transformer_blocks, single_transformer_blocks:
      #     splited input (previous splited output) -> to_qkv/...
      #     -> all2all
      #     -> attn (local head, full seqlen)
      #     -> all2all
      #     -> splited output
      # The `hidden_states` and `encoder_hidden_states` will still keep
      # itself splited after block forward (namely, automatic split by
      # the all2all comm op after attn) for the all blocks.
      # img_ids and txt_ids will only be splited once at the very beginning,
      # and keep splited through the whole transformer forward. The all2all
      # comm op only happens on the `out` tensor after local attn not on
      # img_ids and txt_ids.
      "": {
        "hidden_states":
        _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        "encoder_hidden_states":
        _ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        "img_ids":
        _ContextParallelInput(split_dim=0, expected_dims=2, split_output=False),
        "txt_ids":
        _ContextParallelInput(split_dim=0, expected_dims=2, split_output=False),
      },
      # Then, the final proj_out will gather the splited output.
      #     splited input (previous splited output)
      #     -> all gather
      #     -> un-split output
      "proj_out": _ContextParallelOutput(gather_dim=1, expected_dims=3),
    }
    return _cp_plan
