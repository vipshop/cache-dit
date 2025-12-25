import torch
import functools
from typing import Optional, Tuple, Dict, Any
from torch.distributed import DeviceMesh
from diffusers.models.modeling_utils import ModelMixin
from diffusers import FluxTransformer2DModel
from diffusers.models.transformers.transformer_flux import (
    FluxSingleTransformerBlock,
    FluxAttnProcessor,
    FluxAttention,
    apply_rotary_emb,
    dispatch_attention_fn,
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

from cache_dit.parallelism.attention import _unified_all_to_all_o_async_fn
from cache_dit.parallelism.attention import _unified_all_to_all_qkv_async_fn
from cache_dit.parallelism.attention import _prepare_ulysses_comm_metadata

logger = init_logger(__name__)


@ContextParallelismPlannerRegister.register("FluxTransformer2DModel")
class FluxContextParallelismPlanner(ContextParallelismPlanner):
    def apply(
        self,
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:

        experimental_ulysses_async = kwargs.get("experimental_ulysses_async", False)
        if experimental_ulysses_async:
            FluxAttnProcessor.__call__ = __patch_FluxAttnProcessor_ulysses_async__call__
            FluxSingleTransformerBlock.forward = (
                __patch_FluxSingleTransformerBlock_ulysses_async_forward__
            )
            logger.info(
                "Enabled experimental Async QKV Projection with Ulysses style "
                "Context Parallelism for FluxTransformer2DModel."
            )

        if transformer is not None and self._cp_planner_preferred_native_diffusers:
            assert isinstance(
                transformer, FluxTransformer2DModel
            ), "Transformer must be an instance of FluxTransformer2DModel"
            if hasattr(transformer, "_cp_plan"):
                if transformer._cp_plan is not None:
                    return transformer._cp_plan

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
                "img_ids": ContextParallelInput(split_dim=0, expected_dims=2, split_output=False),
                "txt_ids": ContextParallelInput(split_dim=0, expected_dims=2, split_output=False),
            },
            # Then, the final proj_out will gather the splited output.
            #     splited input (previous splited output)
            #     -> all gather
            #     -> un-split output
            "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        }
        return _cp_plan


# Async Ulysses QKV Proj for FLUX model
# Reference:
# - https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/distributed/sequence_parallel/async_ulysses.py#L43
# - https://github.com/huggingface/diffusers/pull/12727 by @zhangtao0408
def _ulysses_attn_with_async_qkv_proj_flux(
    self: FluxAttnProcessor,
    attn: FluxAttention,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    ulysses_mesh: DeviceMesh = self._parallel_config.context_parallel_config._ulysses_mesh
    group = ulysses_mesh.get_group()

    _all_to_all_o_async_func = _unified_all_to_all_o_async_fn()
    _all_to_all_qv_async_func = _unified_all_to_all_qkv_async_fn()
    _all_to_all_k_async_func = _unified_all_to_all_qkv_async_fn(fp8=False)

    value = attn.to_v(hidden_states)  # type: torch.Tensor
    value = value.unflatten(-1, (attn.heads, -1))
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_value = attn.add_v_proj(encoder_hidden_states)  # type: torch.Tensor
        encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
        value = torch.cat([encoder_value, value], dim=1)

    metadata = _prepare_ulysses_comm_metadata(value)

    # Async all to all for value
    value_wait = _all_to_all_qv_async_func(value, group, **metadata)

    query = attn.to_q(hidden_states)
    query = query.unflatten(-1, (attn.heads, -1))  # type: torch.Tensor
    query = attn.norm_q(query)
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))  # type: torch.Tensor
        encoder_query = attn.norm_added_q(encoder_query)
        query = torch.cat([encoder_query, query], dim=1)
    if image_rotary_emb is not None:
        query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)

    # Async all to all for query
    query_wait = _all_to_all_qv_async_func(query, group, **metadata)

    key = attn.to_k(hidden_states)  # type: torch.Tensor
    key = key.unflatten(-1, (attn.heads, -1))
    key = attn.norm_k(key)
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))  # type: torch.Tensor
        encoder_key = attn.norm_added_k(encoder_key)
        key = torch.cat([encoder_key, key], dim=1)
    if image_rotary_emb is not None:
        key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

    # Async all to all for key
    key_wait = _all_to_all_k_async_func(key, group, **metadata)

    # Ensure the query, key, value are ready
    value = value_wait()
    query = query_wait()
    key = key_wait()

    out = dispatch_attention_fn(
        query,
        key,
        value,
        attn_mask=attention_mask,
        backend=self._attention_backend,
        parallel_config=None,  # set to None to avoid double parallelism
    )  # (B, S_GLOBAL, H_LOCAL, D)

    if encoder_hidden_states is not None:
        # Must be sync all to all for out when encoder_hidden_states is used
        out_wait = _all_to_all_o_async_func(out, group, **metadata)  # (B, S_LOCAL, H_GLOBAL, D)
        out = out_wait()  # type: torch.Tensor

        hidden_states = out.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
            [
                encoder_hidden_states.shape[1],
                hidden_states.shape[1] - encoder_hidden_states.shape[1],
            ],
            dim=1,
        )
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        return hidden_states, encoder_hidden_states
    else:
        # Can be async all to all for out when no encoder_hidden_states
        out_wait = _all_to_all_o_async_func(out, group, **metadata)  # (B, S_LOCAL, H_GLOBAL, D)
        return out_wait


FluxAttnProcessor_original__call__ = FluxAttnProcessor.__call__


@functools.wraps(FluxAttnProcessor_original__call__)
def __patch_FluxAttnProcessor_ulysses_async__call__(
    self: FluxAttnProcessor,
    attn: "FluxAttention",
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if (
        self._parallel_config is not None
        and hasattr(self._parallel_config, "context_parallel_config")
        and self._parallel_config.context_parallel_config is not None
        and self._parallel_config.context_parallel_config.ulysses_degree > 1
    ):
        return _ulysses_attn_with_async_qkv_proj_flux(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )

    # Otherwise, use the original call for non-ulysses case
    return FluxAttnProcessor_original__call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=attention_mask,
        image_rotary_emb=image_rotary_emb,
    )


@functools.wraps(FluxSingleTransformerBlock.forward)
def __patch_FluxSingleTransformerBlock_ulysses_async_forward__(
    self: FluxSingleTransformerBlock,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    text_seq_len = encoder_hidden_states.shape[1]
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    residual = hidden_states
    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)

    joint_attention_kwargs = joint_attention_kwargs or {}
    # Perform attention with Ulysses async QKV proj, the attn_output
    # may be is an instance of AsyncCollectiveTensor.
    attn_output_wait = self.attn(
        hidden_states=norm_hidden_states,
        image_rotary_emb=image_rotary_emb,
        **joint_attention_kwargs,
    )
    # NOTE: Enable the out all2all overlap with mlp computation
    mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))

    # NOTE: Then ensure the attn_output is ready
    if not isinstance(attn_output_wait, torch.Tensor):
        attn_output = attn_output_wait()  # type: torch.Tensor
    else:
        attn_output = attn_output_wait
    attn_output = attn_output.contiguous()
    if attn_output.ndim == 4:
        attn_output = attn_output.flatten(2, 3)

    hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
    gate = gate.unsqueeze(1)
    hidden_states = gate * self.proj_out(hidden_states)
    hidden_states = residual + hidden_states
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    encoder_hidden_states, hidden_states = (
        hidden_states[:, :text_seq_len],
        hidden_states[:, text_seq_len:],
    )
    return encoder_hidden_states, hidden_states
