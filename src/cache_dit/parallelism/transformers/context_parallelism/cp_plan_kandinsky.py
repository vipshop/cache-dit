import torch
from typing import Optional
from diffusers.models.modeling_utils import ModelMixin

try:
    from diffusers import Kandinsky5Transformer3DModel

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


# NOTE: NOT support sparse attention for Kandinsky5 yet.
@ContextParallelismPlannerRegister.register("Kandinsky5")
class Kandinsky5ContextParallelismPlanner(ContextParallelismPlanner):
    def apply(
        self,
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:

        assert isinstance(
            transformer, Kandinsky5Transformer3DModel
        ), "Transformer must be an instance of Kandinsky5Transformer3DModel"

        self._cp_planner_preferred_native_diffusers = False

        if transformer is not None and self._cp_planner_preferred_native_diffusers:
            if hasattr(transformer, "_cp_plan"):
                if transformer._cp_plan is not None:
                    return transformer._cp_plan

        # Otherwise, use the custom CP plan defined here, this maybe
        # a little different from the native diffusers implementation
        # for some models.
        num_blocks = len(transformer.visual_transformer_blocks)
        _cp_plan = {
            # Pattern of blocks 0, split_output=False:
            #     un-split input -> split -> to_qkv/...
            #     -> all2all
            #     -> attn (local head, full seqlen)
            #     -> all2all
            #     -> splited output
            #     (only split visual_embed, not text_embed)
            "visual_transformer_blocks.0": {
                "visual_embed": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
            },
            # Pattern of the all blocks, split_output=False:
            #     un-split input -> split -> to_qkv/...
            #     -> all2all
            #     -> attn (local head, full seqlen)
            #     -> all2all
            #     -> splited output
            #    (only split text_embed, not hidden_states.
            #    hidden_states has been automatically split in previous
            #    block by all2all comm op after attn)
            # The `text_embed` and `rope` will [NOT] be changed after each block ,
            # forward, so we need to split it at [ALL] block by the inserted hook.
            "visual_transformer_blocks.*": {
                "text_embed": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
                "rope": ContextParallelInput(split_dim=1, expected_dims=6, split_output=False),
            },
            # NOTE: Need to gather the visual_embed before final out_layer, because
            # the flatten operation before out_layer needs the full visual_embed.
            f"visual_transformer_blocks.{num_blocks - 1}": ContextParallelOutput(
                gather_dim=1, expected_dims=3
            ),
        }
        return _cp_plan
