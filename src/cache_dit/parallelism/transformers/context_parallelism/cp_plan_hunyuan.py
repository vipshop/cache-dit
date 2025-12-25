import torch
import functools
from typing import Optional, Dict, Any, Union, Tuple
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import (
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_hunyuan_video import (
    HunyuanVideoTransformer3DModel,
    HunyuanVideoAttnProcessor2_0,
)
from diffusers.models.attention_processor import Attention
from diffusers.models.attention_dispatch import dispatch_attention_fn

try:
    from diffusers import HunyuanImageTransformer2DModel
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


@ContextParallelismPlannerRegister.register("HunyuanImage")
class HunyuanImageContextParallelismPlanner(ContextParallelismPlanner):
    def apply(
        self,
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:

        # NOTE: Diffusers native CP plan still not supported
        # for HunyuanImage now.
        self._cp_planner_preferred_native_diffusers = False

        if transformer is not None and self._cp_planner_preferred_native_diffusers:
            assert isinstance(
                transformer, HunyuanImageTransformer2DModel
            ), "Transformer must be an instance of HunyuanImageTransformer2DModel"
            if hasattr(transformer, "_cp_plan"):
                if transformer._cp_plan is not None:
                    return transformer._cp_plan

        # Apply monkey patch to fix attention mask preparation while using CP
        assert isinstance(transformer, HunyuanImageTransformer2DModel)
        HunyuanImageTransformer2DModel.forward = __patch__HunyuanImageTransformer2DModel_forward__

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
                0: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
                1: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
            },
            # Pattern of transformer_blocks.0, split_output=False:
            #     un-split input -> split -> to_qkv/...
            #     -> all2all
            #     -> attn (local head, full seqlen)
            #     -> all2all
            #     -> splited output
            # Pattern of the rest transformer_blocks, single_transformer_blocks:
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
            # NOTE: We have to handle the `attention_mask` carefully in monkey-patched
            # transformer forward while using CP, since it is not splited here.
            # Then, the final proj_out will gather the splited output.
            #     splited input (previous splited output)
            #     -> all gather
            #     -> un-split output
            "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        }
        return _cp_plan


# Adapted from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_hunyuanimage.py#L806
@functools.wraps(HunyuanImageTransformer2DModel.forward)
def __patch__HunyuanImageTransformer2DModel_forward__(
    self: HunyuanImageTransformer2DModel,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    timestep_r: Optional[torch.LongTensor] = None,
    encoder_hidden_states_2: Optional[torch.Tensor] = None,
    encoder_attention_mask_2: Optional[torch.Tensor] = None,
    guidance: Optional[torch.Tensor] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
            )

    if hidden_states.ndim == 4:
        batch_size, channels, height, width = hidden_states.shape
        sizes = (height, width)
    elif hidden_states.ndim == 5:
        batch_size, channels, frame, height, width = hidden_states.shape
        sizes = (frame, height, width)
    else:
        raise ValueError(f"hidden_states must be a 4D or 5D tensor, got {hidden_states.shape}")

    post_patch_sizes = tuple(d // p for d, p in zip(sizes, self.config.patch_size))

    # 1. RoPE
    image_rotary_emb = self.rope(hidden_states)

    # 2. Conditional embeddings
    encoder_attention_mask = encoder_attention_mask.bool()
    temb = self.time_guidance_embed(timestep, guidance=guidance, timestep_r=timestep_r)
    hidden_states = self.x_embedder(hidden_states)
    encoder_hidden_states = self.context_embedder(
        encoder_hidden_states, timestep, encoder_attention_mask
    )

    if self.context_embedder_2 is not None and encoder_hidden_states_2 is not None:
        encoder_hidden_states_2 = self.context_embedder_2(encoder_hidden_states_2)

        encoder_attention_mask_2 = encoder_attention_mask_2.bool()

        # reorder and combine text tokens: combine valid tokens first, then padding
        new_encoder_hidden_states = []
        new_encoder_attention_mask = []

        for text, text_mask, text_2, text_mask_2 in zip(
            encoder_hidden_states,
            encoder_attention_mask,
            encoder_hidden_states_2,
            encoder_attention_mask_2,
        ):
            # Concatenate: [valid_mllm, valid_byt5, invalid_mllm, invalid_byt5]
            new_encoder_hidden_states.append(
                torch.cat(
                    [
                        text_2[text_mask_2],  # valid byt5
                        text[text_mask],  # valid mllm
                        text_2[~text_mask_2],  # invalid byt5
                        text[~text_mask],  # invalid mllm
                    ],
                    dim=0,
                )
            )

            # Apply same reordering to attention masks
            new_encoder_attention_mask.append(
                torch.cat(
                    [
                        text_mask_2[text_mask_2],
                        text_mask[text_mask],
                        text_mask_2[~text_mask_2],
                        text_mask[~text_mask],
                    ],
                    dim=0,
                )
            )

        encoder_hidden_states = torch.stack(new_encoder_hidden_states)
        encoder_attention_mask = torch.stack(new_encoder_attention_mask)

    attention_mask = torch.nn.functional.pad(
        encoder_attention_mask, (hidden_states.shape[1], 0), value=True
    )
    # NOTE(DefTruth): Permute attention_mask if context parallel is used.
    # For example, if work size = 2: [H, E] -> [H_0, E_0, H_1, E_1]
    if self._parallel_config is not None:
        cp_config = getattr(self._parallel_config, "context_parallel_config", None)
        if cp_config is not None and cp_config._world_size > 1:
            hidden_mask = attention_mask[:, : hidden_states.shape[1]]
            encoder_mask = attention_mask[:, hidden_states.shape[1] :]
            hidden_mask_splits = torch.chunk(hidden_mask, cp_config._world_size, dim=1)
            encoder_mask_splits = torch.chunk(encoder_mask, cp_config._world_size, dim=1)
            new_attention_mask_splits = []
            for i in range(cp_config._world_size):
                new_attention_mask_splits.append(hidden_mask_splits[i])
                new_attention_mask_splits.append(encoder_mask_splits[i])
            attention_mask = torch.cat(new_attention_mask_splits, dim=1)

    attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [1,N] -> [1,1,1,N]

    # 3. Transformer blocks
    if torch.is_grad_enabled() and self.gradient_checkpointing:
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
            )

        for block in self.single_transformer_blocks:
            hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
            )

    else:
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
            )

        for block in self.single_transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask=attention_mask,
                image_rotary_emb=image_rotary_emb,
            )

    # 4. Output projection
    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)

    # 5. unpatchify
    # reshape: [batch_size, *post_patch_dims, channels, *patch_size]
    out_channels = self.config.out_channels
    reshape_dims = (
        [batch_size] + list(post_patch_sizes) + [out_channels] + list(self.config.patch_size)
    )
    hidden_states = hidden_states.reshape(*reshape_dims)

    # create permutation pattern: batch, channels, then interleave post_patch and patch dims
    # For 4D: [0, 3, 1, 4, 2, 5] -> batch, channels, post_patch_height, patch_size_height, post_patch_width, patch_size_width
    # For 5D: [0, 4, 1, 5, 2, 6, 3, 7] -> batch, channels, post_patch_frame, patch_size_frame, post_patch_height, patch_size_height, post_patch_width, patch_size_width
    ndim = len(post_patch_sizes)
    permute_pattern = [0, ndim + 1]  # batch, channels
    for i in range(ndim):
        permute_pattern.extend([i + 1, ndim + 2 + i])  # post_patch_sizes[i], patch_sizes[i]
    hidden_states = hidden_states.permute(*permute_pattern)

    # flatten patch dimensions: flatten each (post_patch_size, patch_size) pair
    # batch_size, channels, post_patch_sizes[0] * patch_sizes[0], post_patch_sizes[1] * patch_sizes[1], ...
    final_dims = [batch_size, out_channels] + [
        post_patch * patch for post_patch, patch in zip(post_patch_sizes, self.config.patch_size)
    ]
    hidden_states = hidden_states.reshape(*final_dims)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (hidden_states,)

    return Transformer2DModelOutput(sample=hidden_states)


@ContextParallelismPlannerRegister.register("HunyuanVideo")
class HunyuanVideoContextParallelismPlanner(ContextParallelismPlanner):
    def apply(
        self,
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:

        # NOTE: Diffusers native CP plan still not supported
        # for HunyuanImage now.
        self._cp_planner_preferred_native_diffusers = False

        if transformer is not None and self._cp_planner_preferred_native_diffusers:
            assert isinstance(
                transformer, HunyuanVideoTransformer3DModel
            ), "Transformer must be an instance of HunyuanVideoTransformer3DModel"
            if hasattr(transformer, "_cp_plan"):
                if transformer._cp_plan is not None:
                    return transformer._cp_plan

        # Apply monkey patch to fix attention mask preparation while using CP
        assert isinstance(transformer, HunyuanVideoTransformer3DModel)
        HunyuanVideoTransformer3DModel.forward = __patch__HunyuanVideoTransformer3DModel_forward__
        HunyuanVideoAttnProcessor2_0.__call__ = __patch_HunyuanVideoAttnProcessor2_0__call__
        # Also need to patch the parallel config and attention backend
        if not hasattr(HunyuanVideoAttnProcessor2_0, "_parallel_config"):
            HunyuanVideoAttnProcessor2_0._parallel_config = None
        if not hasattr(HunyuanVideoAttnProcessor2_0, "_attention_backend"):
            HunyuanVideoAttnProcessor2_0._attention_backend = None

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
                0: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
                1: ContextParallelInput(split_dim=0, expected_dims=2, split_output=True),
            },
            # Pattern of transformer_blocks.0, split_output=False:
            #     un-split input -> split -> to_qkv/...
            #     -> all2all
            #     -> attn (local head, full seqlen)
            #     -> all2all
            #     -> splited output
            # Pattern of the rest transformer_blocks, single_transformer_blocks:
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
            # NOTE: We have to handle the `attention_mask` carefully in monkey-patched
            # transformer forward while using CP, since it is not splited here.
            # Then, the final proj_out will gather the splited output.
            #     splited input (previous splited output)
            #     -> all gather
            #     -> un-split output
            "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        }
        return _cp_plan


# Adapted from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_hunyuan_video.py#L1032
@functools.wraps(HunyuanVideoTransformer3DModel.forward)
def __patch__HunyuanVideoTransformer3DModel_forward__(
    self: HunyuanVideoTransformer3DModel,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    pooled_projections: torch.Tensor,
    guidance: torch.Tensor = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
) -> Union[Tuple[torch.Tensor], Transformer2DModelOutput]:
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
            )

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p, p_t = self.config.patch_size, self.config.patch_size_t
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p
    post_patch_width = width // p
    first_frame_num_tokens = 1 * post_patch_height * post_patch_width

    # 1. RoPE
    image_rotary_emb = self.rope(hidden_states)

    # 2. Conditional embeddings
    temb, token_replace_emb = self.time_text_embed(timestep, pooled_projections, guidance)

    hidden_states = self.x_embedder(hidden_states)
    encoder_hidden_states = self.context_embedder(
        encoder_hidden_states, timestep, encoder_attention_mask
    )

    # 3. Attention mask preparation
    latent_sequence_length = hidden_states.shape[1]
    condition_sequence_length = encoder_hidden_states.shape[1]
    sequence_length = latent_sequence_length + condition_sequence_length
    attention_mask = torch.ones(
        batch_size,
        sequence_length,
        device=hidden_states.device,
        dtype=torch.bool,
    )  # [B, N]
    effective_condition_sequence_length = encoder_attention_mask.sum(dim=1, dtype=torch.int)  # [B,]
    effective_sequence_length = latent_sequence_length + effective_condition_sequence_length
    indices = torch.arange(sequence_length, device=hidden_states.device).unsqueeze(0)  # [1, N]
    mask_indices = indices >= effective_sequence_length.unsqueeze(1)  # [B, N]
    attention_mask = attention_mask.masked_fill(mask_indices, False)
    # NOTE(DefTruth): Permute attention_mask if context parallel is used.
    # For example, if work size = 2: [H, E] -> [H_0, E_0, H_1, E_1]
    if self._parallel_config is not None:
        cp_config = getattr(self._parallel_config, "context_parallel_config", None)
        if cp_config is not None and cp_config._world_size > 1:
            hidden_mask = attention_mask[:, :latent_sequence_length]
            encoder_mask = attention_mask[:, latent_sequence_length:]
            hidden_mask_splits = torch.chunk(hidden_mask, cp_config._world_size, dim=1)
            encoder_mask_splits = torch.chunk(encoder_mask, cp_config._world_size, dim=1)
            new_attention_mask_splits = []
            for i in range(cp_config._world_size):
                new_attention_mask_splits.append(hidden_mask_splits[i])
                new_attention_mask_splits.append(encoder_mask_splits[i])
            attention_mask = torch.cat(new_attention_mask_splits, dim=1)

    attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, N]

    # 4. Transformer blocks
    if torch.is_grad_enabled() and self.gradient_checkpointing:
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                token_replace_emb,
                first_frame_num_tokens,
            )

        for block in self.single_transformer_blocks:
            hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                token_replace_emb,
                first_frame_num_tokens,
            )

    else:
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                token_replace_emb,
                first_frame_num_tokens,
            )

        for block in self.single_transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                attention_mask,
                image_rotary_emb,
                token_replace_emb,
                first_frame_num_tokens,
            )

    # 5. Output projection
    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size,
        post_patch_num_frames,
        post_patch_height,
        post_patch_width,
        -1,
        p_t,
        p,
        p,
    )
    hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7)
    hidden_states = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (hidden_states,)

    return Transformer2DModelOutput(sample=hidden_states)


@functools.wraps(HunyuanVideoAttnProcessor2_0.__call__)
def __patch_HunyuanVideoAttnProcessor2_0__call__(
    self: HunyuanVideoAttnProcessor2_0,
    attn: Attention,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    image_rotary_emb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if attn.add_q_proj is None and encoder_hidden_states is not None:
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)

    # 1. QKV projections
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    # NOTE(DefTruth): no transpose
    query = query.unflatten(2, (attn.heads, -1))
    key = key.unflatten(2, (attn.heads, -1))
    value = value.unflatten(2, (attn.heads, -1))

    # 2. QK normalization
    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    # 3. Rotational positional embeddings applied to latent stream
    if image_rotary_emb is not None:
        from diffusers.models.embeddings import apply_rotary_emb

        # NOTE(DefTruth): Monkey patch for encoder conditional RoPE
        if attn.add_q_proj is None and encoder_hidden_states is not None:
            query = torch.cat(
                [
                    apply_rotary_emb(
                        query[:, : -encoder_hidden_states.shape[1]],
                        image_rotary_emb,
                        sequence_dim=1,
                    ),
                    query[:, -encoder_hidden_states.shape[1] :],
                ],
                dim=1,
            )
            key = torch.cat(
                [
                    apply_rotary_emb(
                        key[:, : -encoder_hidden_states.shape[1]],
                        image_rotary_emb,
                        sequence_dim=1,
                    ),
                    key[:, -encoder_hidden_states.shape[1] :],
                ],
                dim=1,
            )
        else:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

    # 4. Encoder condition QKV projection and normalization
    if attn.add_q_proj is not None and encoder_hidden_states is not None:
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

        # NOTE(DefTruth): no transpose
        encoder_query = encoder_query.unflatten(2, (attn.heads, -1))
        encoder_key = encoder_key.unflatten(2, (attn.heads, -1))
        encoder_value = encoder_value.unflatten(2, (attn.heads, -1))

        if attn.norm_added_q is not None:
            encoder_query = attn.norm_added_q(encoder_query)
        if attn.norm_added_k is not None:
            encoder_key = attn.norm_added_k(encoder_key)

        query = torch.cat([query, encoder_query], dim=1)
        key = torch.cat([key, encoder_key], dim=1)
        value = torch.cat([value, encoder_value], dim=1)

    # 5. Attention
    # NOTE(DefTruth): use dispatch_attention_fn
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
    # NOTE(DefTruth): no transpose
    hidden_states = hidden_states.flatten(2, 3)
    hidden_states = hidden_states.to(query.dtype)

    # 6. Output projection
    if encoder_hidden_states is not None:
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : -encoder_hidden_states.shape[1]],
            hidden_states[:, -encoder_hidden_states.shape[1] :],
        )

        if getattr(attn, "to_out", None) is not None:
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)

        if getattr(attn, "to_add_out", None) is not None:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

    return hidden_states, encoder_hidden_states
