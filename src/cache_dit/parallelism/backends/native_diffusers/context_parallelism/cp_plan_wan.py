import torch
from typing import Optional
from diffusers.models.modeling_utils import ModelMixin
from diffusers import WanTransformer3DModel, WanVACETransformer3DModel

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
# NOTE: Maybe use full name to avoid name conflict between
# WanTransformer3DModel and WanVACETransformer3DModel?
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
            # Pattern of rope, split_output=True (split output rather than input):
            #    un-split input
            #    -> keep input un-split
            #    -> rope
            #    -> splited output
            "rope": {
                0: ContextParallelInput(
                    split_dim=1, expected_dims=4, split_output=True
                ),
                1: ContextParallelInput(
                    split_dim=1, expected_dims=4, split_output=True
                ),
            },
            # Pattern of blocks.0, split_output=False:
            #     un-split input -> split -> to_qkv/...
            #     -> all2all
            #     -> attn (local head, full seqlen)
            #     -> all2all
            #     -> splited output
            #     (only split hidden_states, not encoder_hidden_states)
            "blocks.0": {
                "hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
            },
            # Pattern of the all blocks, split_output=False:
            #     un-split input -> split -> to_qkv/...
            #     -> all2all
            #     -> attn (local head, full seqlen)
            #     -> all2all
            #     -> splited output
            #    (only split encoder_hidden_states, not hidden_states.
            #    hidden_states has been automatically split in previous
            #    block by all2all comm op after attn)
            # The `encoder_hidden_states` will [NOT] be changed after each block forward,
            # so we need to split it at [ALL] block by the inserted split hook.
            "blocks.*": {
                "encoder_hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
            },
            # Then, the final proj_out will gather the splited output.
            #     splited input (previous splited output)
            #     -> all gather
            #     -> un-split output
            "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        }
        return _cp_plan


@ContextParallelismPlannerRegister.register("WanVACETransformer3D")
class WanVACEContextParallelismPlanner(ContextParallelismPlanner):
    def apply(
        self,
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:

        # NOTE: Now, Diffusers don't have native CP plan for
        # WanVACETransformer3DModel.
        self._cp_planner_preferred_native_diffusers = False

        if (
            transformer is not None
            and self._cp_planner_preferred_native_diffusers
        ):
            assert isinstance(
                transformer, WanVACETransformer3DModel
            ), "Transformer must be an instance of WanVACETransformer3DModel"
            if hasattr(transformer, "_cp_plan"):
                if transformer._cp_plan is not None:
                    return transformer._cp_plan

        # Otherwise, use the custom CP plan defined here, this maybe
        # a little different from the native diffusers implementation
        # for some models.
        _cp_plan = {
            # Pattern of rope, split_output=True (split output rather than input):
            #    un-split input
            #    -> keep input un-split
            #    -> rope
            #    -> splited output
            "rope": {
                0: ContextParallelInput(
                    split_dim=1, expected_dims=4, split_output=True
                ),
                1: ContextParallelInput(
                    split_dim=1, expected_dims=4, split_output=True
                ),
            },
            # Pattern of vace_blocks.0, split_output=False:
            "vace_blocks.0": {
                "control_hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
            },
            "vace_blocks.*": {
                "hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
                "encoder_hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
            },
            # Pattern of blocks.0, split_output=False:
            #     un-split input -> split -> to_qkv/...
            #     -> all2all
            #     -> attn (local head, full seqlen)
            #     -> all2all
            #     -> splited output
            #     (only split hidden_states, not encoder_hidden_states)
            "blocks.0": {
                "hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
            },
            # Pattern of the all blocks, split_output=False:
            #     un-split input -> split -> to_qkv/...
            #     -> all2all
            #     -> attn (local head, full seqlen)
            #     -> all2all
            #     -> splited output
            #    (only split encoder_hidden_states, not hidden_states.
            #    hidden_states has been automatically split in previous
            #    block by all2all comm op after attn)
            # The `encoder_hidden_states` will [NOT] be changed after each block forward,
            # so we need to split it at [ALL] block by the inserted split hook.
            "blocks.*": {
                "encoder_hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
            },
            # Then, the final proj_out will gather the splited output.
            #     splited input (previous splited output)
            #     -> all gather
            #     -> un-split output
            "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        }
        return _cp_plan
