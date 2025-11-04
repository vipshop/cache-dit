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


# TODO: Add WanVACETransformer3DModel context parallelism planner.
# NOTE: We choice to use full name to avoid name conflict between
# WanTransformer3DModel and WanVACETransformer3DModel.
@ContextParallelismPlannerRegister.register("WanTransformer3D")
class WanContextParallelismPlanner(ContextParallelismPlanner):
    def apply(
        self,
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:
        if (
            transformer is not None
            and self._cp_planner_preferred_native_diffusers
        ):
            from diffusers import WanTransformer3DModel

            assert isinstance(
                transformer, WanTransformer3DModel
            ), "Transformer must be an instance of WanTransformer3DModel"
            if hasattr(transformer, "_cp_plan"):
                if transformer._cp_plan is not None:
                    return transformer._cp_plan

        # Otherwise, use the custom CP plan defined here, this maybe
        # a little different from the native diffusers implementation
        # for some models.
        _cp_plan = {
            "rope": {
                0: ContextParallelInput(
                    split_dim=1, expected_dims=4, split_output=True
                ),
                1: ContextParallelInput(
                    split_dim=1, expected_dims=4, split_output=True
                ),
            },
            "blocks.0": {
                "hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
            },
            "blocks.*": {
                "encoder_hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
            },
            "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        }
        return _cp_plan
