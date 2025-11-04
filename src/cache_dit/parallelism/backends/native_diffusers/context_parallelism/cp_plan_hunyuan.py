import torch
from typing import Optional
from diffusers.models.modeling_utils import ModelMixin

try:
    from diffusers import HunyuanImageTransformer2DModel
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


@ContextParallelismPlannerRegister.register("HunyuanImage")
class HunyuanImageContextParallelismPlanner(ContextParallelismPlanner):
    def apply(
        self,
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:

        # NOTE: Diffusers native CP plan still not supported
        # for HunyuanImage now.
        self._cp_planner_preferred_native_diffusers = False

        if (
            transformer is not None
            and self._cp_planner_preferred_native_diffusers
        ):
            assert isinstance(
                transformer, HunyuanImageTransformer2DModel
            ), "Transformer must be an instance of HunyuanImageTransformer2DModel"
            if hasattr(transformer, "_cp_plan"):
                if transformer._cp_plan is not None:
                    return transformer._cp_plan

        # Otherwise, use the custom CP plan defined here, this maybe
        # a little different from the native diffusers implementation
        # for some models.
        _cp_plan = {
            "rope": {
                0: ContextParallelInput(
                    split_dim=0, expected_dims=2, split_output=True
                ),
                1: ContextParallelInput(
                    split_dim=0, expected_dims=2, split_output=True
                ),
            },
            "transformer_blocks.0": {
                "hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
            },
            "transformer_blocks.*": {
                "encoder_hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
            },
            "single_transformer_blocks.*": {
                "encoder_hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
            },
            "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        }
        return _cp_plan
