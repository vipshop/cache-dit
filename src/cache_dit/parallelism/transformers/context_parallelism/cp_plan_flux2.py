import torch
import functools
from typing import Optional
from torch.distributed import DeviceMesh
from diffusers.models.modeling_utils import ModelMixin
from diffusers import Flux2Transformer2DModel
from diffusers.models.transformers.transformer_flux2 import (
    Flux2AttnProcessor,
    Flux2Attention,
    Flux2ParallelSelfAttnProcessor,
    Flux2ParallelSelfAttention,
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
    ParallelismConfig,
)

from ....logger import init_logger

from ...attention import _unified_all_to_all_o_async_fn
from ...attention import _unified_all_to_all_qkv_async_fn
from ...attention import _prepare_ulysses_comm_metadata

logger = init_logger(__name__)


@ContextParallelismPlannerRegister.register("Flux2Transformer2DModel")
class Flux2ContextParallelismPlanner(ContextParallelismPlanner):
    def apply(
        self,
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        parallelism_config: Optional[ParallelismConfig] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:

        self._cp_planner_preferred_native_diffusers = False

        if parallelism_config.ulysses_async:
            Flux2AttnProcessor.__call__ = __patch_flux2_attn_processor
            Flux2ParallelSelfAttnProcessor.__call__ = __patch_flux2_self_attn_processor
            logger.info("Async Ulysses Attention is enabled.")

        if transformer is not None and self._cp_planner_preferred_native_diffusers:
            assert isinstance(
                transformer, Flux2Transformer2DModel
            ), "Transformer must be an instance of Flux2Transformer2DModel"
            if hasattr(transformer, "_cp_plan"):
                if transformer._cp_plan is not None:
                    return transformer._cp_plan

        # Use custom CP plan in cache-dit for better control and flexibility.
        _cp_plan = {
            "": {
                "hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
                "encoder_hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
                "img_ids": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
                "txt_ids": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            },
            "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        }
        return _cp_plan


# Implements async Ulysses communication for Flux2Attention when context parallelism
# is enabled with Ulysses degree > 1. The async communication allows overlapping
# communication with computation for better performance.
def _async_ulysses_attn_flux2(
    self: Flux2AttnProcessor,
    attn: "Flux2Attention",
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    attention_mask: torch.Tensor | None = None,
    image_rotary_emb: torch.Tensor | None = None,
) -> torch.Tensor:
    # Manually expand _get_qkv_projections to support async Ulysses communication.
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

    query = attn.to_q(hidden_states)  # type: torch.Tensor
    query = query.unflatten(-1, (attn.heads, -1))
    query = attn.norm_q(query)
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_query = attn.add_q_proj(encoder_hidden_states)  # type: torch.Tensor
        encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
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
        encoder_key = attn.add_k_proj(encoder_hidden_states)  # type: torch.Tensor
        encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
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
    # Must be sync all to all for out when encoder_hidden_states is used
    out_wait = _all_to_all_o_async_func(out, group, **metadata)  # (B, S_LOCAL, H_GLOBAL, D)
    out = out_wait()  # type: torch.Tensor

    hidden_states = out.flatten(2, 3)
    hidden_states = hidden_states.to(query.dtype)

    if encoder_hidden_states is not None:
        encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
            [
                encoder_hidden_states.shape[1],
                hidden_states.shape[1] - encoder_hidden_states.shape[1],
            ],
            dim=1,
        )
        encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)

    if encoder_hidden_states is not None:
        return hidden_states, encoder_hidden_states
    else:
        return hidden_states


flux2_attn_processor__call__ = Flux2AttnProcessor.__call__


@functools.wraps(flux2_attn_processor__call__)
def __patch_flux2_attn_processor(
    self: Flux2AttnProcessor,
    attn: "Flux2Attention",
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    attention_mask: torch.Tensor | None = None,
    image_rotary_emb: torch.Tensor | None = None,
) -> torch.Tensor:
    if (
        self._parallel_config is not None
        and hasattr(self._parallel_config, "context_parallel_config")
        and self._parallel_config.context_parallel_config is not None
        and self._parallel_config.context_parallel_config.ulysses_degree > 1
    ):
        return _async_ulysses_attn_flux2(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
    else:
        return flux2_attn_processor__call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )


def _async_ulysses_self_attn_flux2(
    self: Flux2ParallelSelfAttnProcessor,
    attn: "Flux2ParallelSelfAttention",
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    image_rotary_emb: torch.Tensor | None = None,
) -> torch.Tensor:
    ulysses_mesh: DeviceMesh = self._parallel_config.context_parallel_config._ulysses_mesh
    group = ulysses_mesh.get_group()

    _all_to_all_o_async_func = _unified_all_to_all_o_async_fn()
    _all_to_all_qv_async_func = _unified_all_to_all_qkv_async_fn()
    _all_to_all_k_async_func = _unified_all_to_all_qkv_async_fn(fp8=False)

    # Parallel in (QKV + MLP in) projection
    hidden_states = attn.to_qkv_mlp_proj(hidden_states)
    qkv, mlp_hidden_states = torch.split(
        hidden_states, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
    )

    # Handle the attention logic
    query, key, value = qkv.chunk(3, dim=-1)

    value = value.unflatten(-1, (attn.heads, -1))
    metadata = _prepare_ulysses_comm_metadata(value)
    # Async all to all for value
    value_wait = _all_to_all_qv_async_func(value, group, **metadata)

    query = query.unflatten(-1, (attn.heads, -1))  # type: torch.Tensor
    query = attn.norm_q(query)
    if image_rotary_emb is not None:
        query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
    # Async all to all for query
    query_wait = _all_to_all_qv_async_func(query, group, **metadata)

    key = key.unflatten(-1, (attn.heads, -1))
    key = attn.norm_k(key)
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
    # Must be sync all to all for out when encoder_hidden_states is used
    out_wait = _all_to_all_o_async_func(out, group, **metadata)  # (B, S_LOCAL, H_GLOBAL, D)

    # Handle the feedforward (FF) logic, overlap with attention output communication
    mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)

    out = out_wait()  # type: torch.Tensor

    hidden_states = out.flatten(2, 3)
    hidden_states = hidden_states.to(query.dtype)

    # Concatenate and parallel output projection
    hidden_states = torch.cat([hidden_states, mlp_hidden_states], dim=-1)
    hidden_states = attn.to_out(hidden_states)

    return hidden_states


flux2_self_attn_processor__call__ = Flux2ParallelSelfAttnProcessor.__call__


@functools.wraps(flux2_self_attn_processor__call__)
def __patch_flux2_self_attn_processor(
    self: Flux2ParallelSelfAttnProcessor,
    attn: "Flux2ParallelSelfAttention",
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    image_rotary_emb: torch.Tensor | None = None,
) -> torch.Tensor:
    if (
        self._parallel_config is not None
        and hasattr(self._parallel_config, "context_parallel_config")
        and self._parallel_config.context_parallel_config is not None
        and self._parallel_config.context_parallel_config.ulysses_degree > 1
    ):
        return _async_ulysses_self_attn_flux2(
            self,
            attn,
            hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
    else:
        return flux2_self_attn_processor__call__(
            self,
            attn,
            hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
