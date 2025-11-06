import torch
import functools
from typing import Optional, Tuple
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn

try:
    from nunchaku.models.transformers.transformer_flux_v2 import (
        NunchakuFluxAttention,
        NunchakuFluxFA2Processor,
        NunchakuFluxTransformer2DModelV2,
    )
    from nunchaku.ops.fused import fused_qkv_norm_rottary
    from nunchaku.models.transformers.transformer_qwenimage import (
        NunchakuQwenAttention,
        NunchakuQwenImageNaiveFA2Processor,
        NunchakuQwenImageTransformer2DModel,
    )
except ImportError:
    raise ImportError(
        "NunchakuFluxTransformer2DModelV2 or NunchakuQwenImageTransformer2DModel "
        "requires the 'nunchaku' package. Please install nunchaku before using "
        "the context parallelism for nunchaku 4-bits models."
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

from cache_dit.logger import init_logger

logger = init_logger(__name__)


@ContextParallelismPlannerRegister.register("NunchakuFlux")
class NunchakuFluxContextParallelismPlanner(ContextParallelismPlanner):
    def apply(
        self,
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:

        self._cp_planner_preferred_native_diffusers = False

        if (
            transformer is not None
            and self._cp_planner_preferred_native_diffusers
        ):

            assert isinstance(
                transformer, NunchakuFluxTransformer2DModelV2
            ), "Transformer must be an instance of NunchakuFluxTransformer2DModelV2"
            if hasattr(transformer, "_cp_plan"):
                if transformer._cp_plan is not None:
                    return transformer._cp_plan

        NunchakuFluxFA2Processor.__call__ = (
            __patch_NunchakuFluxFA2Processor__call__
        )
        # Also need to patch the parallel config and attention backend
        if not hasattr(NunchakuFluxFA2Processor, "_parallel_config"):
            NunchakuFluxFA2Processor._parallel_config = None
        if not hasattr(NunchakuFluxFA2Processor, "_attention_backend"):
            NunchakuFluxFA2Processor._attention_backend = None
        if not hasattr(NunchakuFluxAttention, "_parallel_config"):
            NunchakuFluxAttention._parallel_config = None
        if not hasattr(NunchakuFluxAttention, "_attention_backend"):
            NunchakuFluxAttention._attention_backend = None

        # Otherwise, use the custom CP plan defined here, this maybe
        # a little different from the native diffusers implementation
        # for some models.
        _cp_plan = {
            # Here is a Transformer level CP plan for Flux, which will
            # only apply the only 1 split hook (pre_forward) on the forward
            # of Transformer, and gather the output after Transformer forward.
            # Pattern of transformer forward, split_output=False:
            #     un-split input -> splited input (inside transformer)
            # Pattern of the transformer_blocks, single_transformer_blocks:
            #     splited input (previous splited output) -> to_qkv/...
            #     -> all2all
            #     -> attn (local head, full seqlen)
            #     -> all2all
            #     -> splited output
            # The `hidden_states` and `encoder_hidden_states` will still keep
            # itself splited after block forward (namely, automatic split by
            # the all2all comm op after attn) for the all blocks.
            # img_ids and txt_ids will only be splited once at the very beginning,
            # and keep splited through the whole transformer forward. The all2all
            # comm op only happens on the `out` tensor after local attn not on
            # img_ids and txt_ids.
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
            # Then, the final proj_out will gather the splited output.
            #     splited input (previous splited output)
            #     -> all gather
            #     -> un-split output
            "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        }
        return _cp_plan


@functools.wraps(NunchakuFluxFA2Processor.__call__)
def __patch_NunchakuFluxFA2Processor__call__(
    self: NunchakuFluxFA2Processor,
    attn: NunchakuFluxAttention,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    image_rotary_emb: Tuple[torch.Tensor, torch.Tensor] | torch.Tensor = None,
    **kwargs,
) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
    # The original implementation of NunchakuFluxFA2Processor.__call__
    # is not changed here for brevity. In actual implementation, we need to
    # modify the attention computation to support context parallelism.
    if attention_mask is not None:
        raise NotImplementedError("attention_mask is not supported")

    batch_size, _, channels = hidden_states.shape
    assert channels == attn.heads * attn.head_dim
    qkv = fused_qkv_norm_rottary(
        hidden_states,
        attn.to_qkv,
        attn.norm_q,
        attn.norm_k,
        (
            image_rotary_emb[0]
            if isinstance(image_rotary_emb, tuple)
            else image_rotary_emb
        ),
    )

    if attn.added_kv_proj_dim is not None:
        assert encoder_hidden_states is not None
        assert isinstance(image_rotary_emb, tuple)
        qkv_context = fused_qkv_norm_rottary(
            encoder_hidden_states,
            attn.add_qkv_proj,
            attn.norm_added_q,
            attn.norm_added_k,
            image_rotary_emb[1],
        )
        qkv = torch.cat([qkv_context, qkv], dim=1)

    query, key, value = qkv.chunk(3, dim=-1)
    # Original implementation:
    # query = query.view(batch_size, -1, attn.heads, attn.head_dim).transpose(
    #     1, 2
    # )
    # key = key.view(batch_size, -1, attn.heads, attn.head_dim).transpose(1, 2)
    # value = value.view(batch_size, -1, attn.heads, attn.head_dim).transpose(
    #     1, 2
    # )
    # hidden_states = F.scaled_dot_product_attention(
    #     query,
    #     key,
    #     value,
    #     attn_mask=attention_mask,
    #     dropout_p=0.0,
    #     is_causal=False,
    # )
    # hidden_states = hidden_states.transpose(1, 2).reshape(
    #     batch_size, -1, attn.heads * attn.head_dim
    # )
    # hidden_states = hidden_states.to(query.dtype)

    # NOTE(DefTruth): Monkey patch to support context parallelism
    query = query.view(batch_size, -1, attn.heads, attn.head_dim)
    key = key.view(batch_size, -1, attn.heads, attn.head_dim)
    value = value.view(batch_size, -1, attn.heads, attn.head_dim)

    hidden_states = dispatch_attention_fn(
        query,
        key,
        value,
        attn_mask=attention_mask,
        backend=getattr(self, "_attention_backend", None),
        parallel_config=getattr(self, "_parallel_config", None),
    )
    hidden_states = hidden_states.flatten(2, 3)
    hidden_states = hidden_states.to(query.dtype)

    if encoder_hidden_states is not None:
        encoder_hidden_states, hidden_states = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1] :],
        )
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        return hidden_states, encoder_hidden_states
    else:
        # for single transformer block, we split the proj_out into two linear layers
        hidden_states = attn.to_out(hidden_states)
        return hidden_states


@ContextParallelismPlannerRegister.register("NunchakuQwenImage")
class NunchakuQwenImageContextParallelismPlanner(ContextParallelismPlanner):
    def apply(
        self,
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:

        self._cp_planner_preferred_native_diffusers = False

        if (
            transformer is not None
            and self._cp_planner_preferred_native_diffusers
        ):

            assert isinstance(
                transformer, NunchakuQwenImageTransformer2DModel
            ), "Transformer must be an instance of NunchakuQwenImageTransformer2DModel"
            if hasattr(transformer, "_cp_plan"):
                if transformer._cp_plan is not None:
                    return transformer._cp_plan

        # Also need to patch the parallel config and attention backend
        if not hasattr(NunchakuQwenImageNaiveFA2Processor, "_parallel_config"):
            NunchakuQwenImageNaiveFA2Processor._parallel_config = None
        if not hasattr(
            NunchakuQwenImageNaiveFA2Processor, "_attention_backend"
        ):
            NunchakuQwenImageNaiveFA2Processor._attention_backend = None
        if not hasattr(NunchakuQwenAttention, "_parallel_config"):
            NunchakuQwenAttention._parallel_config = None
        if not hasattr(NunchakuQwenAttention, "_attention_backend"):
            NunchakuQwenAttention._attention_backend = None

        # Otherwise, use the custom CP plan defined here, this maybe
        # a little different from the native diffusers implementation
        # for some models.
        _cp_plan = {
            # Here is a Transformer level CP plan for Flux, which will
            # only apply the only 1 split hook (pre_forward) on the forward
            # of Transformer, and gather the output after Transformer forward.
            # Pattern of transformer forward, split_output=False:
            #     un-split input -> splited input (inside transformer)
            # Pattern of the transformer_blocks, single_transformer_blocks:
            #     splited input (previous splited output) -> to_qkv/...
            #     -> all2all
            #     -> attn (local head, full seqlen)
            #     -> all2all
            #     -> splited output
            # The `hidden_states` and `encoder_hidden_states` will still keep
            # itself splited after block forward (namely, automatic split by
            # the all2all comm op after attn) for the all blocks.
            "": {
                "hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
                # NOTE: Due to the joint attention implementation of
                # QwenImageTransformerBlock, we must split the
                # encoder_hidden_states as well.
                "encoder_hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
                # NOTE: But encoder_hidden_states_mask seems never used in
                # QwenImageTransformerBlock, so we do not split it here.
                # "encoder_hidden_states_mask": ContextParallelInput(
                #     split_dim=1, expected_dims=2, split_output=False
                # ),
            },
            # Pattern of pos_embed, split_output=True (split output rather than input):
            #    un-split input
            #    -> keep input un-split
            #    -> rope
            #    -> splited output
            "pos_embed": {
                0: ContextParallelInput(
                    split_dim=0, expected_dims=2, split_output=True
                ),
                1: ContextParallelInput(
                    split_dim=0, expected_dims=2, split_output=True
                ),
            },
            # Then, the final proj_out will gather the splited output.
            #     splited input (previous splited output)
            #     -> all gather
            #     -> un-split output
            "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        }
        return _cp_plan
