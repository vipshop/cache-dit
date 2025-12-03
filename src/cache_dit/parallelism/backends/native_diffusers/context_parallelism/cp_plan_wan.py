import torch
import functools
from typing import Optional, Tuple
from diffusers.models.modeling_utils import ModelMixin
from diffusers import WanVACETransformer3DModel

try:
    from diffusers.models.transformers.transformer_chronoedit import (
        WanAttention,
        WanAttnProcessor,
        _get_added_kv_projections,
        _get_qkv_projections,
        dispatch_attention_fn,
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

from .attention._distributed_primitives import _wait_tensor
from .attention._distributed_primitives import _all_to_all_single_sync
from .attention._distributed_primitives import _all_to_all_single_async
from .attention._templated_ulysses_anything import is_ulysses_anything_enabled

from cache_dit.logger import init_logger

logger = init_logger(__name__)


@ContextParallelismPlannerRegister.register("ChronoEditTransformer3D")
@ContextParallelismPlannerRegister.register("WanTransformer3D")
class WanContextParallelismPlanner(ContextParallelismPlanner):
    def apply(
        self,
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:

        cls_name = transformer.__class__.__name__ if transformer else ""

        if cls_name.startswith("ChronoEditTransformer3D"):
            self._cp_planner_preferred_native_diffusers = False

        experimental_ulysses_async_qkv_proj = kwargs.get(
            "experimental_ulysses_async_qkv_proj", False
        )
        if experimental_ulysses_async_qkv_proj:
            assert not is_ulysses_anything_enabled(), (
                "experimental_ulysses_async_qkv_proj is not compatible with "
                "experimental_ulysses_anything, please disable one of them."
            )
            WanAttnProcessor.__call__ = __patch_WanAttnProcessor_ulysses_async__call__

            logger.info(
                "Enabled experimental Async QKV Projection with Ulysses style "
                "Context Parallelism for WanTransformer3DModel."
            )

        if transformer is not None and self._cp_planner_preferred_native_diffusers:
            if hasattr(transformer, "_cp_plan"):
                if transformer._cp_plan is not None:
                    return transformer._cp_plan

        # Otherwise, use the custom CP plan defined here, this maybe
        # a little different from the native diffusers implementation
        # for some models.
        if cls_name.startswith("ChronoEditTransformer3D"):
            if not experimental_ulysses_async_qkv_proj:
                WanAttnProcessor.__call__ = __patch_WanAttnProcessor__call__
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
        else:
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
                "blocks.*": {
                    "encoder_hidden_states": ContextParallelInput(
                        split_dim=1, expected_dims=3, split_output=False
                    ),
                },
                # Then, the final proj_out will gather the splited output.
                #     splited input (previous splited output)
                #     -> all gather
                #     -> un-split output
                "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
            }
        return _cp_plan


@functools.wraps(WanAttnProcessor.__call__)
def __patch_WanAttnProcessor__call__(
    self: WanAttnProcessor,
    attn: "WanAttention",
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
            # no need to pass the parallel_config here.
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
        # no need to pass the parallel_config here.
        parallel_config=(self._parallel_config if encoder_hidden_states is None else None),
    )
    hidden_states = hidden_states.flatten(2, 3)
    hidden_states = hidden_states.type_as(query)

    if hidden_states_img is not None:
        hidden_states = hidden_states + hidden_states_img

    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)
    return hidden_states


# NOTE: Support Async Ulysses QKV projection for Wan
def _ulysses_attn_with_async_qkv_proj_wan(
    self: WanAttnProcessor,
    attn: "WanAttention",
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    from torch.distributed import DeviceMesh

    ulysses_mesh: DeviceMesh = self._parallel_config.context_parallel_config._ulysses_mesh
    world_size = self._parallel_config.context_parallel_config.ulysses_degree
    group = ulysses_mesh.get_group()

    encoder_hidden_states_img = None
    if attn.add_k_proj is not None:
        # 512 is the context length of the text encoder, hardcoded for now
        image_context_length = encoder_hidden_states.shape[1] - 512
        encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
        encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

    # NOTE: Reorder to compute Value first to get more opportunity to
    # overlap the computation of norm_q/k and RoPE.
    # Step 1: Project and prepare Value, then start async communication
    value = attn.to_v(hidden_states)
    value = value.unflatten(2, (attn.heads, -1))
    if encoder_hidden_states is not None:
        encoder_value = attn.add_v_proj(encoder_hidden_states)
        encoder_value = encoder_value.unflatten(2, (attn.heads, -1))
        value = torch.cat([encoder_value, value], dim=1)

    B, S_KV_LOCAL, H, D = value.shape
    H_LOCAL = H // world_size

    # 0. Async all to all for value
    value = value.reshape(B, S_KV_LOCAL, world_size, H_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
    value = _all_to_all_single_async(value, group)

    # Step 2: While value is communicating, compute Query
    query = attn.to_q(hidden_states)
    query = query.unflatten(2, (attn.heads, -1))
    query = attn.norm_q(query)
    if encoder_hidden_states is not None:
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
        encoder_query = attn.norm_added_q(encoder_query)
        query = torch.cat([encoder_query, query], dim=1)

    # Apply RoPE to query
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

    # 1. Async all to all for query
    _, S_Q_LOCAL, _, _ = query.shape
    query = query.reshape(B, S_Q_LOCAL, world_size, H_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
    query = _all_to_all_single_async(query, group)

    # Step 3: While query is communicating, compute Key
    key = attn.to_k(hidden_states)
    key = key.unflatten(2, (attn.heads, -1))
    key = attn.norm_k(key)
    if encoder_hidden_states is not None:
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
        encoder_key = attn.norm_added_k(encoder_key)
        key = torch.cat([encoder_key, key], dim=1)

    # Apply RoPE to key
    if rotary_emb is not None:
        key = apply_rotary_emb(key, *rotary_emb)

    # 2. Async all to all for key
    key = key.reshape(B, S_KV_LOCAL, world_size, H_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
    key = _all_to_all_single_async(key, group)

    # Wait for value
    value = _wait_tensor(value)
    value = (
        value.reshape(world_size, S_KV_LOCAL, B, H_LOCAL, D)
        .flatten(0, 1)
        .permute(1, 0, 2, 3)
        .contiguous()
    )

    # Wait for query
    query = _wait_tensor(query)
    query = (
        query.reshape(world_size, S_Q_LOCAL, B, H_LOCAL, D)
        .flatten(0, 1)
        .permute(1, 0, 2, 3)
        .contiguous()
    )

    # Wait for key
    key = _wait_tensor(key)
    key = (
        key.reshape(world_size, S_KV_LOCAL, B, H_LOCAL, D)
        .flatten(0, 1)
        .permute(1, 0, 2, 3)
        .contiguous()
    )

    # Compute attention
    hidden_states = dispatch_attention_fn(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        backend=self._attention_backend,
        parallel_config=None,  # set to None to avoid double parallelism
    )

    # I2V task - handle image encoder hidden states
    # Note: This cross-attention also needs to be computed with the distributed query
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
            parallel_config=None,
        )

    # All-to-all to convert back from (full_head, local_seq) to (local_head, full_seq)
    hidden_states = (
        hidden_states.reshape(B, world_size, S_Q_LOCAL, H_LOCAL, D)
        .permute(1, 3, 0, 2, 4)
        .contiguous()
    )
    hidden_states = _all_to_all_single_sync(hidden_states, group)
    hidden_states = hidden_states.flatten(0, 1).permute(1, 2, 0, 3).contiguous()

    # Reshape back
    hidden_states = hidden_states.flatten(2, 3)
    hidden_states = hidden_states.type_as(query)

    # Also need to all-to-all the image cross-attention output
    if hidden_states_img is not None:
        hidden_states_img = (
            hidden_states_img.reshape(B, world_size, S_Q_LOCAL, H_LOCAL, D)
            .permute(1, 3, 0, 2, 4)
            .contiguous()
        )
        hidden_states_img = _all_to_all_single_sync(hidden_states_img, group)
        hidden_states_img = hidden_states_img.flatten(0, 1).permute(1, 2, 0, 3).contiguous()
        hidden_states_img = hidden_states_img.flatten(2, 3)
        hidden_states_img = hidden_states_img.type_as(query)
        hidden_states = hidden_states + hidden_states_img

    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)
    return hidden_states


WanAttnProcessor_original__call__ = WanAttnProcessor.__call__


@functools.wraps(WanAttnProcessor_original__call__)
def __patch_WanAttnProcessor_ulysses_async__call__(
    self: WanAttnProcessor,
    attn: "WanAttention",
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    if (
        self._parallel_config is not None
        and hasattr(self._parallel_config, "context_parallel_config")
        and self._parallel_config.context_parallel_config is not None
        and self._parallel_config.context_parallel_config.ulysses_degree > 1
    ):
        return _ulysses_attn_with_async_qkv_proj_wan(
            self,
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            rotary_emb,
        )

    # Otherwise, use the original call for non-ulysses case
    return WanAttnProcessor_original__call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states,
        attention_mask,
        rotary_emb,
    )


@ContextParallelismPlannerRegister.register("WanVACETransformer3D")
class WanVACEContextParallelismPlanner(ContextParallelismPlanner):
    def apply(
        self,
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:

        # NOTE: Now, Diffusers don't have native CP plan for
        # WanVACETransformer3DModel.
        self._cp_planner_preferred_native_diffusers = False

        if transformer is not None and self._cp_planner_preferred_native_diffusers:
            assert isinstance(
                transformer, WanVACETransformer3DModel
            ), "Transformer must be an instance of WanVACETransformer3DModel"
            if hasattr(transformer, "_cp_plan"):
                if transformer._cp_plan is not None:
                    return transformer._cp_plan

        # Otherwise, use the custom CP plan defined here, this maybe
        # a little different from the native diffusers implementation
        # for some models.
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
            # Pattern of vace_blocks.0, split_output=False:
            "vace_blocks.0": {
                "control_hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
            },
            "vace_blocks.*": {
                "hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
                "encoder_hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
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
            "blocks.*": {
                "encoder_hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
            },
            # Then, the final proj_out will gather the splited output.
            #     splited input (previous splited output)
            #     -> all gather
            #     -> un-split output
            "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        }
        return _cp_plan
