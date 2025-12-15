import torch
import functools
from typing import Optional
import torch.nn.functional as F
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.transformer_ltx import (
    LTXVideoTransformer3DModel,
    LTXAttention,
    AttentionModuleMixin,
    LTXVideoAttnProcessor,
    apply_rotary_emb,
)
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


@ContextParallelismPlannerRegister.register("LTXVideo")
class LTXVideoContextParallelismPlanner(ContextParallelismPlanner):
    def apply(
        self,
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:
        assert transformer is not None, "Transformer must be provided."
        assert isinstance(
            transformer, LTXVideoTransformer3DModel
        ), "Transformer must be an instance of LTXVideoTransformer3DModel"

        # NOTE: The atttention_mask preparation in LTXAttention while using
        # context parallelism is buggy in diffusers v0.36.0.dev0, so we
        # disable the preference to use native diffusers implementation here.
        self._cp_planner_preferred_native_diffusers = False

        if transformer is not None and self._cp_planner_preferred_native_diffusers:
            if hasattr(transformer, "_cp_plan"):
                if transformer._cp_plan is not None:
                    return transformer._cp_plan

        # Apply monkey patch to fix attention mask preparation at class level
        assert issubclass(LTXAttention, AttentionModuleMixin)
        LTXAttention.prepare_attention_mask = __patch__LTXAttention_prepare_attention_mask__
        LTXVideoAttnProcessor.__call__ = __patch__LTXVideoAttnProcessor__call__

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
            # itself splited after block forward, namely, hidden_states will
            # automatically split by the all2all comm op after attn, and the
            # encoder_hidden_states will be keep splited after the entrypoint
            # of transformer forward, for the all blocks.
            "": {
                "hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
                "encoder_hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
                # NOTE: encoder_attention_mask (namely, attention_mask in cross-attn)
                # should never be split across seqlen while using context parallelism
                # for LTXVideoTransformer3DModel. It don't contribute to any computation
                # in parallel or not. So we comment it out here and handle the head-split
                # correctly while using context parallel in the patched attention processor.
                # "encoder_attention_mask": ContextParallelInput(
                #     split_dim=1, expected_dims=2, split_output=False
                # ),
            },
            # Pattern of rope, split_output=True (split output rather than input):
            #    un-split input
            #    -> keep input un-split
            #    -> rope
            #    -> splited output
            "rope": {
                0: ContextParallelInput(split_dim=1, expected_dims=3, split_output=True),
                1: ContextParallelInput(split_dim=1, expected_dims=3, split_output=True),
            },
            # Then, the final proj_out will gather the splited output.
            #     splited input (previous splited output)
            #     -> all gather
            #     -> un-split output
            "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        }
        return _cp_plan


@functools.wraps(LTXAttention.prepare_attention_mask)
def __patch__LTXAttention_prepare_attention_mask__(
    self: LTXAttention,
    attention_mask: torch.Tensor,
    target_length: int,
    batch_size: int,
    out_dim: int = 3,
    # NOTE(DefTruth): Allow specifying head_size for CP
    head_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Prepare the attention mask for the attention computation.

    Args:
        attention_mask (`torch.Tensor`): The attention mask to prepare.
        target_length (`int`): The target length of the attention mask.
        batch_size (`int`): The batch size for repeating the attention mask.
        out_dim (`int`, *optional*, defaults to `3`): Output dimension.

    Returns:
        `torch.Tensor`: The prepared attention mask.
    """
    if head_size is None:
        head_size = self.heads
    if attention_mask is None:
        return attention_mask

    current_length: int = attention_mask.shape[-1]
    if current_length != target_length:
        if attention_mask.device.type == "mps":
            # HACK: MPS: Does not support padding by greater than dimension of input tensor.
            # Instead, we can manually construct the padding tensor.
            padding_shape = (
                attention_mask.shape[0],
                attention_mask.shape[1],
                target_length,
            )
            padding = torch.zeros(
                padding_shape,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([attention_mask, padding], dim=2)
        else:
            # TODO: for pipelines such as stable-diffusion, padding cross-attn mask:
            #       we want to instead pad by (0, remaining_length), where remaining_length is:
            #       remaining_length: int = target_length - current_length
            # TODO: re-enable tests/models/test_models_unet_2d_condition.py#test_model_xattn_padding
            attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

    if out_dim == 3:
        if attention_mask.shape[0] < batch_size * head_size:
            attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
    elif out_dim == 4:
        attention_mask = attention_mask.unsqueeze(1)
        attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

    return attention_mask


@functools.wraps(LTXVideoAttnProcessor.__call__)
def __patch__LTXVideoAttnProcessor__call__(
    self: LTXVideoAttnProcessor,
    attn: "LTXAttention",
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )

    if attention_mask is not None:
        if self._parallel_config is None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )
        else:
            # NOTE(DefTruth): Fix attention mask preparation for context parallelism
            cp_config = getattr(self._parallel_config, "context_parallel_config", None)
            if cp_config is not None and cp_config._world_size > 1:
                head_size = attn.heads // cp_config._world_size
                attention_mask = attn.prepare_attention_mask(
                    attention_mask,
                    sequence_length * cp_config._world_size,
                    batch_size,
                    3,
                    head_size,
                )
                attention_mask = attention_mask.view(
                    batch_size, head_size, -1, attention_mask.shape[-1]
                )

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states

    query = attn.to_q(hidden_states)
    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)

    query = attn.norm_q(query)
    key = attn.norm_k(key)

    if image_rotary_emb is not None:
        query = apply_rotary_emb(query, image_rotary_emb)
        key = apply_rotary_emb(key, image_rotary_emb)

    query = query.unflatten(2, (attn.heads, -1))
    key = key.unflatten(2, (attn.heads, -1))
    value = value.unflatten(2, (attn.heads, -1))

    hidden_states = dispatch_attention_fn(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        backend=self._attention_backend,
        parallel_config=self._parallel_config,
    )
    hidden_states = hidden_states.flatten(2, 3)
    hidden_states = hidden_states.to(query.dtype)

    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)
    return hidden_states
