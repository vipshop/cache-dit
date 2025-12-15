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


@ContextParallelismPlannerRegister.register("SkyReelsV2")
class SkyReelsV2ContextParallelismPlanner(ContextParallelismPlanner):
    def apply(
        self,
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:

        # NOTE: Diffusers native CP plan still not supported
        # for SkyReelsV2 now, use custom implementation.
        self._cp_planner_preferred_native_diffusers = False

        # SkyReelsV2 uses WanModel architecture (config: "_class_name": "WanModel")
        # Based on BlockAdapter, it uses Pattern_3 where encoder_hidden_states
        # will NEVER change in the blocks forward loop.
        # This is different from regular Wan which uses Pattern_2.
        _cp_plan = {
            # Pattern of rope, split_output=True (split output rather than input):
            #    un-split input
            #    -> keep input un-split
            #    -> rope
            #    -> splited output
            # SkyReelsV2 (like Wan) has rotary position embeddings
            "rope": {
                0: ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
                1: ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
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
            # Pattern of blocks.*, split_output=False:
            #     splited input (previous splited output) -> to_qkv/...
            #     -> all2all
            #     -> attn (local head, full seqlen)
            #     -> all2all
            #     -> splited output
            # Since encoder_hidden_states never changes (Pattern_3), we need to
            # split it at ALL blocks by the inserted split hook.
            # hidden_states has been automatically split in previous block.
            "blocks.*": {
                "encoder_hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
            },
            # The final proj_out will gather the splited output.
            #     splited input (previous splited output)
            #     -> all gather
            #     -> un-split output
            "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        }
        return _cp_plan
