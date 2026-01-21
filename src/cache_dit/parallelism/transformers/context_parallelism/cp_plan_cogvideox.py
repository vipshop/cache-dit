import torch
import functools
from typing import Optional, Tuple
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.cogvideox_transformer_3d import (
    CogVideoXAttnProcessor2_0,
    CogVideoXTransformer3DModel,
)
from diffusers.models.attention_processor import Attention
from diffusers.models.attention_dispatch import dispatch_attention_fn

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


@ContextParallelismPlannerRegister.register("CogVideoX")
class CogVideoXContextParallelismPlanner(ContextParallelismPlanner):
    def apply(
        self,
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:

        # NOTE: Diffusers native CP plan still not supported
        # for CogVideoX now.
        self._cp_planner_preferred_native_diffusers = False

        if transformer is not None and self._cp_planner_preferred_native_diffusers:
            assert isinstance(
                transformer, CogVideoXTransformer3DModel
            ), "Transformer must be an instance of CogVideoXTransformer3DModel"
            if hasattr(transformer, "_cp_plan"):
                if transformer._cp_plan is not None:
                    return transformer._cp_plan

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
            # transformer forward while using CP, since it is not splited here.
            # Then, the final proj_out will gather the splited output.
            #     splited input (previous splited output)
            #     -> all gather
            #     -> un-split output
            "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        }
        return _cp_plan


@functools.wraps(CogVideoXAttnProcessor2_0.__call__)
def __patch_CogVideoXAttnProcessor2_0__call__(
    self: CogVideoXAttnProcessor2_0,
    attn: Attention,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    text_seq_length = encoder_hidden_states.size(1)

    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    batch_size, sequence_length, _ = hidden_states.shape

    # NOTE(DefTruth): attention mask is always None in CogVideoX
    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    # NOTE(DefTruth): no transpose
    query = query.view(batch_size, -1, attn.heads, head_dim)
    key = key.view(batch_size, -1, attn.heads, head_dim)
    value = value.view(batch_size, -1, attn.heads, head_dim)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    # Apply RoPE if needed
    if image_rotary_emb is not None:
        from diffusers.models.embeddings import apply_rotary_emb

        query[:, text_seq_length:] = apply_rotary_emb(
            query[:, text_seq_length:],
            image_rotary_emb,
            sequence_dim=1,
        )
        if not attn.is_cross_attention:
            key[:, text_seq_length:] = apply_rotary_emb(
                key[:, text_seq_length:],
                image_rotary_emb,
                sequence_dim=1,
            )

    # NOTE(DefTruth): Apply dispatch_attention_fn instead of sdpa directly
    hidden_states = dispatch_attention_fn(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        backend=getattr(self, "_attention_backend", None),
        parallel_config=getattr(self, "_parallel_config", None),
    )
    hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)

    # linear proj
    hidden_states = attn.to_out[0](hidden_states)
    # dropout
    hidden_states = attn.to_out[1](hidden_states)

    encoder_hidden_states, hidden_states = hidden_states.split(
        [text_seq_length, hidden_states.size(1) - text_seq_length], dim=1
    )
    return hidden_states, encoder_hidden_states
