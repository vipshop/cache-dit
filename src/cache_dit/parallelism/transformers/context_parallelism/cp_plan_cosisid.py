import torch
from typing import Optional
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.consisid_transformer_3d import (
    ConsisIDTransformer3DModel,
)
from diffusers.models.transformers.cogvideox_transformer_3d import (
    CogVideoXAttnProcessor2_0,
)

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
from .cp_plan_cogvideox import __patch_CogVideoXAttnProcessor2_0__call__

from cache_dit.logger import init_logger

logger = init_logger(__name__)


@ContextParallelismPlannerRegister.register("ConsisID")
class CosisIDContextParallelismPlanner(ContextParallelismPlanner):
    def apply(
        self,
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:

        # NOTE: Diffusers native CP plan still not supported
        # for ConsisID now.
        self._cp_planner_preferred_native_diffusers = False

        if transformer is not None and self._cp_planner_preferred_native_diffusers:
            assert isinstance(
                transformer, ConsisIDTransformer3DModel
            ), "Transformer must be an instance of ConsisIDTransformer3DModel"
            if hasattr(transformer, "_cp_plan"):
                if transformer._cp_plan is not None:
                    return transformer._cp_plan

        # ConsisID uses the same attention processor as CogVideoX.
        CogVideoXAttnProcessor2_0.__call__ = __patch_CogVideoXAttnProcessor2_0__call__
        # Also need to patch the parallel config and attention backend
        if not hasattr(CogVideoXAttnProcessor2_0, "_parallel_config"):
            CogVideoXAttnProcessor2_0._parallel_config = None
        if not hasattr(CogVideoXAttnProcessor2_0, "_attention_backend"):
            CogVideoXAttnProcessor2_0._attention_backend = None

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
            # Pattern of the rest transformer_blocks, split_output=False:
            #     splited input (previous splited output) -> to_qkv/...
            #     -> all2all
            #     -> attn (local head, full seqlen)
            #     -> all2all
            #     -> splited output
            # The `encoder_hidden_states` will be changed after each block forward,
            # so we need to split it at the first block, and keep it splited (namely,
            # automatically split by the all2all op after attn) for the rest blocks.
            # The `out` tensor of local attn will be splited into `hidden_states` and
            # `encoder_hidden_states` after each block forward, thus both of them
            # will be automatically splited by all2all comm op after local attn.
            "transformer_blocks.0": {
                "hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
                "encoder_hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
            },
            # Pattern of the image_rotary_emb, split at every block, because the it
            # is not automatically splited by all2all comm op and keep un-splited
            # while the block forward finished:
            #    un-split input -> split output
            #    -> after block forward
            #    -> un-split input
            #    un-split input -> split output
            #    ...
            "transformer_blocks.*": {
                "image_rotary_emb": [
                    ContextParallelInput(split_dim=0, expected_dims=2, split_output=False),
                    ContextParallelInput(split_dim=0, expected_dims=2, split_output=False),
                ],
            },
            # NOTE: We should gather both hidden_states and encoder_hidden_states
            # at the end of the last block. Because the subsequent op is:
            # hidden_states = torch.cat([encoder_hidden_states, hidden_states])
            f"transformer_blocks.{len(transformer.transformer_blocks) - 1}": [
                ContextParallelOutput(gather_dim=1, expected_dims=3),
                ContextParallelOutput(gather_dim=1, expected_dims=3),
            ],
        }
        return _cp_plan
