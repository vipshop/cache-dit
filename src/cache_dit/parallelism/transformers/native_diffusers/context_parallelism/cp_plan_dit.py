import torch
from typing import Optional
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.dit_transformer_2d import (
    DiTTransformer2DModel,
)
from diffusers.models.attention_processor import (
    Attention,
    AttnProcessor2_0,
)  # sdpa

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
from .cp_plan_pixart import (
    __patch_AttnProcessor2_0__call__,
    __patch_Attention_prepare_attention_mask__,
)


from cache_dit.logger import init_logger

logger = init_logger(__name__)


@ContextParallelismPlannerRegister.register("DiT")
class DiTContextParallelismPlanner(ContextParallelismPlanner):
    def apply(
        self,
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:
        assert transformer is not None, "Transformer must be provided."
        assert isinstance(
            transformer, DiTTransformer2DModel
        ), "Transformer must be an instance of DiTTransformer2DModel"

        self._cp_planner_preferred_native_diffusers = False

        if transformer is not None and self._cp_planner_preferred_native_diffusers:
            if hasattr(transformer, "_cp_plan"):
                if transformer._cp_plan is not None:
                    return transformer._cp_plan

        # Apply monkey patch to fix attention mask preparation at class level
        Attention.prepare_attention_mask = __patch_Attention_prepare_attention_mask__
        AttnProcessor2_0.__call__ = __patch_AttnProcessor2_0__call__
        if not hasattr(AttnProcessor2_0, "_parallel_config"):
            AttnProcessor2_0._parallel_config = None
        if not hasattr(AttnProcessor2_0, "_attention_backend"):
            AttnProcessor2_0._attention_backend = None

        # Otherwise, use the custom CP plan defined here, this maybe
        # a little different from the native diffusers implementation
        # for some models.

        _cp_plan = {
            # Pattern of transformer_blocks.0, split_output=False:
            #     un-split input -> split -> to_qkv/...
            #     -> all2all
            #     -> attn (local head, full seqlen)
            #     -> all2all
            #     -> splited output
            #     (only split hidden_states, not encoder_hidden_states)
            "transformer_blocks.0": {
                "hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
            },
            # Then, the final proj_out will gather the splited output.
            #     splited input (previous splited output)
            #     -> all gather
            #     -> un-split output
            "proj_out_2": ContextParallelOutput(gather_dim=1, expected_dims=3),
        }
        return _cp_plan
