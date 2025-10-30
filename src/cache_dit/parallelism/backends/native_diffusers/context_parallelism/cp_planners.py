# Docstring references: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/_modeling_parallel.py#L185
# A dictionary where keys denote the input to be split across context parallel region, and the
# value denotes the sharding configuration.
# If the key is a string, it denotes the name of the parameter in the forward function.
# If the key is an integer, split_output must be set to True, and it denotes the index of the output
# to be split across context parallel region.
# ContextParallelInputType = Dict[
#     Union[str, int], Union[ContextParallelInput, List[ContextParallelInput], Tuple[ContextParallelInput, ...]]
# ]

# A dictionary where keys denote the output to be gathered across context parallel region, and the
# value denotes the gathering configuration.
# ContextParallelOutputType = Union[
#     ContextParallelOutput, List[ContextParallelOutput], Tuple[ContextParallelOutput, ...]
# ]

# A dictionary where keys denote the module id, and the value denotes how the inputs/outputs of
# the module should be split/gathered across context parallel region.
# ContextParallelModelPlan = Dict[str, Union[ContextParallelInputType, ContextParallelOutputType]]

# Example of a ContextParallelModelPlan (QwenImageTransformer2DModel):
#
# Each model should define a _cp_plan attribute that contains information on how to shard/gather
# tensors at different stages of the forward:
#
# ```python
# _cp_plan = {
#     "": {
#         "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
#         "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
#         "encoder_hidden_states_mask": ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
#     },
#     "pos_embed": {
#         0: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
#         1: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
#     },
#     "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
# }
# ```
#
# The dictionary is a set of module names mapped to their respective CP plan. The inputs/outputs of layers will be
# split/gathered according to this at the respective module level. Here, the following happens:
# - "":
#     we specify that we want to split the various inputs across the sequence dim in the pre-forward hook (i.e. before
#     the actual forward logic of the QwenImageTransformer2DModel is run, we will splitthe inputs)
# - "pos_embed":
#     we specify that we want to split the outputs of the RoPE layer. Since there are two outputs (imag & text freqs),
#     we can individually specify how they should be split
# - "proj_out":
#     before returning to the user, we gather the entire sequence on each rank in the post-forward hook (after the linear
#     layer forward has run).
#
# ContextParallelInput:
#     specifies how to split the input tensor in the pre-forward or post-forward hook of the layer it is attached to
#
# ContextParallelOutput:
#     specifies how to gather the input tensor in the post-forward hook in the layer it is attached to

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


__all__ = [
    "ContextParallelismPlanner",
    "ContextParallelismPlannerRegister",
    "FluxContextParallelismPlanner",
    "QwenImageContextParallelismPlanner",
    "WanContextParallelismPlanner",
    "LTXVideoContextParallelismPlanner",
]


# Register context parallelism planner for models
@ContextParallelismPlannerRegister.register("Flux")
class FluxContextParallelismPlanner(ContextParallelismPlanner):
    def apply(
        self,
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:
        if transformer is not None:
            from diffusers import FluxTransformer2DModel

            assert isinstance(
                transformer, FluxTransformer2DModel
            ), "Transformer must be an instance of FluxTransformer2DModel"
            if hasattr(transformer, "_cp_plan"):
                return transformer._cp_plan

        _cp_plan = {
            "": {
                "hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
                "encoder_hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
                "img_ids": ContextParallelInput(
                    split_dim=0, expected_dims=2, split_output=False
                ),
                "txt_ids": ContextParallelInput(
                    split_dim=0, expected_dims=2, split_output=False
                ),
            },
            "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        }
        return _cp_plan


@ContextParallelismPlannerRegister.register("QwenImage")
class QwenImageContextParallelismPlanner(ContextParallelismPlanner):
    def apply(
        self,
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:
        if transformer is not None:
            from diffusers import QwenImageTransformer2DModel

            assert isinstance(
                transformer, QwenImageTransformer2DModel
            ), "Transformer must be an instance of QwenImageTransformer2DModel"
            if hasattr(transformer, "_cp_plan"):
                return transformer._cp_plan

        _cp_plan = _cp_plan = {
            "": {
                "hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
                "encoder_hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
                "encoder_hidden_states_mask": ContextParallelInput(
                    split_dim=1, expected_dims=2, split_output=False
                ),
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
        if transformer is not None:
            from diffusers import WanTransformer3DModel

            assert isinstance(
                transformer, WanTransformer3DModel
            ), "Transformer must be an instance of WanTransformer3DModel"
            if hasattr(transformer, "_cp_plan"):
                return transformer._cp_plan

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


@ContextParallelismPlannerRegister.register("LTXVideo")
class LTXVideoContextParallelismPlanner(ContextParallelismPlanner):
    def apply(
        self,
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:
        if transformer is not None:
            from diffusers import LTXVideoTransformer3DModel

            assert isinstance(
                transformer, LTXVideoTransformer3DModel
            ), "Transformer must be an instance of LTXVideoTransformer3DModel"
            if hasattr(transformer, "_cp_plan"):
                return transformer._cp_plan

        _cp_plan = {
            "": {
                "hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
                "encoder_hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
                "encoder_attention_mask": ContextParallelInput(
                    split_dim=1, expected_dims=2, split_output=False
                ),
            },
            "rope": {
                0: ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=True
                ),
                1: ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=True
                ),
            },
            "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        }
        return _cp_plan
