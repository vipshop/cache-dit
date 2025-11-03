import torch
from typing import Optional
from diffusers.models.modeling_utils import ModelMixin

try:
    from diffusers.models._modeling_parallel import (
        ContextParallelInput,
        ContextParallelOutput,
        ContextParallelModelPlan,
    )
except ImportError:
    raise ImportError(
        "Context parallelism requires the 'diffusers>=0.36.dev0'."
        "Please install latest version of diffusers from source: \n"
        "pip3 install git+https://github.com/huggingface/diffusers.git"
    )
from .cp_plan_registers import (
    ContextParallelismPlanner,
    ContextParallelismPlannerRegister,
)

from cache_dit.logger import init_logger

logger = init_logger(__name__)


@ContextParallelismPlannerRegister.register("QwenImage")
class QwenImageContextParallelismPlanner(ContextParallelismPlanner):
    def apply(
        self,
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:

        # NOTE: Set it as False to use custom CP plan defined here.
        self._cp_planner_preferred_native_diffusers = False

        if (
            transformer is not None
            and self._cp_planner_preferred_native_diffusers
        ):
            from diffusers import QwenImageTransformer2DModel

            assert isinstance(
                transformer, QwenImageTransformer2DModel
            ), "Transformer must be an instance of QwenImageTransformer2DModel"
            if hasattr(transformer, "_cp_plan"):
                return transformer._cp_plan

        # Otherwise, use the custom CP plan defined here, this maybe
        # a little different from the native diffusers implementation
        # for some models.
        _cp_plan = {
            "": {
                "hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
                # NOTE: Due to the joint attention implementation of
                # QwenImageTransformerBlock, we must split the
                # encoder_hidden_states as well.
                "encoder_hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
                # NOTE: But encoder_hidden_states_mask seems very used in
                # QwenImageTransformerBlock, so we do not split it here.
                # "encoder_hidden_states_mask": ContextParallelInput(
                #     split_dim=1, expected_dims=2, split_output=False
                # ),
            },
            "pos_embed": {
                0: ContextParallelInput(
                    split_dim=0, expected_dims=2, split_output=True
                ),
                1: ContextParallelInput(
                    split_dim=0, expected_dims=2, split_output=True
                ),
            },
            "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        }
        return _cp_plan
