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
    ParallelismConfig,
)

from ....logger import init_logger

logger = init_logger(__name__)


@ContextParallelismPlannerRegister.register("HeliosTransformer3DModel")
class HeliosContextParallelismPlanner(ContextParallelismPlanner):
    def apply(
        self,
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        parallelism_config: Optional[ParallelismConfig] = None,
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

        # NOTE(DefTruth): This cp plan here is ugly but it works, we  will optimize it in the future.
        num_blocks = len(transformer.blocks)
        # NOTE: Due to the complex concat and split ops for history hidden states and current hidden
        # states in Helios, we have to pinned the sharding strategy at 'attn' and 'ffn' level, this
        # will lead to sub-optimal performance because of the extra all-gather and scatter communication
        # overhead, we will optimize it in the future by supporting more flexible sharding strategy.
        _cp_plan = {
            # Input split at attn level and ffn level.
            "blocks.*.attn1": {
                "hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
                "rotary_emb": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
            },
            "blocks.*.attn2": {
                "hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
            },
            "blocks.*.ffn": {
                "hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
            },
            # Output gather at attn level and ffn level.
            **{
                f"blocks.{i}.attn1": ContextParallelOutput(gather_dim=1, expected_dims=3)
                for i in range(num_blocks)
            },
            **{
                f"blocks.{i}.attn2": ContextParallelOutput(gather_dim=1, expected_dims=3)
                for i in range(num_blocks)
            },
            **{
                f"blocks.{i}.ffn": ContextParallelOutput(gather_dim=1, expected_dims=3)
                for i in range(num_blocks)
            },
        }
        return _cp_plan
