import torch
from typing import Optional
from diffusers.models.modeling_utils import ModelMixin
from diffusers import HeliosTransformer3DModel

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


@ContextParallelismPlannerRegister.register("HeliosTransformer3DModel")
class HeliosContextParallelismPlanner(ContextParallelismPlanner):
    def apply(
        self,
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:

        self._cp_planner_preferred_native_diffusers = False

        if transformer is not None and self._cp_planner_preferred_native_diffusers:
            assert isinstance(
                transformer, HeliosTransformer3DModel
            ), "Transformer must be an instance of HeliosTransformer3DModel"
            if hasattr(transformer, "_cp_plan"):
                if transformer._cp_plan is not None:
                    return transformer._cp_plan

        # Otherwise, use the custom CP plan defined here, this maybe
        # a little different from the native diffusers implementation
        # for some models.
        # NOTE(DefTruth): This cp plan here will raise error due to Helios's special
        # design of concating the history context(frames) and current context(frames)
        # before feeding into the transformer blocks, which makes it hard to shard
        # the input hidden states by sequence dimension correctly.
        # TODO: Add 'history_hidden_states' param to block forward (DON't merged it with 'hidden_states'),
        # and then we can shard the 'current_hidden_states' both 'history_hidden_states' by sequence dim.
        _cp_plan = {
            "blocks.0": {
                "hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
            },
            "blocks.*": {
                "temb": ContextParallelInput(split_dim=1, expected_dims=4, split_output=False),
                "rotary_emb": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
            },
            "blocks.39": ContextParallelOutput(gather_dim=1, expected_dims=3),
        }
        return _cp_plan
