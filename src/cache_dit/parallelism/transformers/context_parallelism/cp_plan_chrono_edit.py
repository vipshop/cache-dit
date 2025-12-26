import torch
import functools
from typing import Optional, Tuple
from diffusers.models.modeling_utils import ModelMixin

try:
    from diffusers.models.transformers.transformer_chronoedit import (
        _get_added_kv_projections,
        _get_qkv_projections,
        dispatch_attention_fn,
    )
    from diffusers.models.transformers.transformer_chronoedit import (
        WanAttention as ChronoEditWanAttention,
    )
    from diffusers.models.transformers.transformer_chronoedit import (
        WanAttnProcessor as ChronoEditWanAttnProcessor,
    )
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


@ContextParallelismPlannerRegister.register("ChronoEditTransformer3D")
class ChronoEditContextParallelismPlanner(ContextParallelismPlanner):
    def apply(
        self,
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:

        self._cp_planner_preferred_native_diffusers = False

        if transformer is not None and self._cp_planner_preferred_native_diffusers:
            if hasattr(transformer, "_cp_plan"):
                if transformer._cp_plan is not None:
                    return transformer._cp_plan

        # Otherwise, use the custom CP plan defined here, this maybe
        # a little different from the native diffusers implementation
        # for some models.
        ChronoEditWanAttnProcessor.__call__ = __patch_ChronoEditWanAttnProcessor__call__
        _cp_plan = {
            # Pattern of rope, split_output=True (split output rather than input):
            #    un-split input
            #    -> keep input un-split
            #    -> rope
            #    -> splited output
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
            # NOTE(DefTruth): We need to disable the splitting of encoder_hidden_states because
            # the image_encoder consistently generates 257 tokens for image_embed. This causes
            # the shape of encoder_hidden_states—whose token count is always 769 (512 + 257)
            # after concatenation—to be indivisible by the number of devices in the CP.
            # "blocks.*": {
            #     "encoder_hidden_states": ContextParallelInput(
            #         split_dim=1, expected_dims=3, split_output=False
            #     ),
            # },
            # Then, the final proj_out will gather the splited output.
            #     splited input (previous splited output)
            #     -> all gather
            #     -> un-split output
            "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        }
        return _cp_plan


@functools.wraps(ChronoEditWanAttnProcessor.__call__)
def __patch_ChronoEditWanAttnProcessor__call__(
    self: ChronoEditWanAttnProcessor,
    attn: ChronoEditWanAttention,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    encoder_hidden_states_img = None
    if attn.add_k_proj is not None:
        # 512 is the context length of the text encoder, hardcoded for now
        image_context_length = encoder_hidden_states.shape[1] - 512
        encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
        encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

    query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

    query = attn.norm_q(query)
    key = attn.norm_k(key)

    query = query.unflatten(2, (attn.heads, -1))
    key = key.unflatten(2, (attn.heads, -1))
    value = value.unflatten(2, (attn.heads, -1))

    if rotary_emb is not None:

        def apply_rotary_emb(
            hidden_states: torch.Tensor,
            freqs_cos: torch.Tensor,
            freqs_sin: torch.Tensor,
        ):
            x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
            cos = freqs_cos[..., 0::2]
            sin = freqs_sin[..., 1::2]
            out = torch.empty_like(hidden_states)
            out[..., 0::2] = x1 * cos - x2 * sin
            out[..., 1::2] = x1 * sin + x2 * cos
            return out.type_as(hidden_states)

        query = apply_rotary_emb(query, *rotary_emb)
        key = apply_rotary_emb(key, *rotary_emb)

    # I2V task
    hidden_states_img = None
    if encoder_hidden_states_img is not None:
        key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
        key_img = attn.norm_added_k(key_img)

        key_img = key_img.unflatten(2, (attn.heads, -1))
        value_img = value_img.unflatten(2, (attn.heads, -1))

        hidden_states_img = dispatch_attention_fn(
            query,
            key_img,
            value_img,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            # FIXME(DefTruth): Since the key/value in cross-attention depends
            # solely on encoder_hidden_states_img (img), the (q_chunk * k) * v
            # computation can be parallelized independently. Thus, there is
            # no need to pass the config here.
            parallel_config=None,
        )
        hidden_states_img = hidden_states_img.flatten(2, 3)
        hidden_states_img = hidden_states_img.type_as(query)

    hidden_states = dispatch_attention_fn(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        backend=self._attention_backend,
        # FIXME(DefTruth): Since the key/value in cross-attention depends
        # solely on encoder_hidden_states (text), the (q_chunk * k) * v
        # computation can be parallelized independently. Thus, there is
        # no need to pass the config here.
        parallel_config=(self._parallel_config if encoder_hidden_states is None else None),
    )
    hidden_states = hidden_states.flatten(2, 3)
    hidden_states = hidden_states.type_as(query)

    if hidden_states_img is not None:
        hidden_states = hidden_states + hidden_states_img

    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)
    return hidden_states
