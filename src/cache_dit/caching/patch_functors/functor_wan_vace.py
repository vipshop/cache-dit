import torch
from typing import Tuple, Optional, Dict, Any, Union, List
from diffusers.models.transformers.transformer_wan_vace import (
    WanTransformerBlock,
    WanVACETransformer3DModel,
    Transformer2DModelOutput,
)
from diffusers.utils import (
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
)
from .functor_base import PatchFunctor
from cache_dit.logger import init_logger

logger = init_logger(__name__)


class WanVACEPatchFunctor(PatchFunctor):

    def _apply(
        self,
        transformer: WanVACETransformer3DModel,
        **kwargs,
    ) -> WanVACETransformer3DModel:
        if hasattr(transformer, "_is_patched"):
            return transformer

        is_patched = False

        _i = 0
        for block in transformer.blocks:
            assert isinstance(block, WanTransformerBlock)
            block._i = _i
            block._vace_layers = transformer.config.vace_layers
            block.forward = __patch_block_forward__.__get__(block)
            _i += 1

        is_patched = True
        cls_name = transformer.__class__.__name__

        if is_patched:
            logger.warning(f"Patched {cls_name} for cache-dit.")
            assert not getattr(transformer, "_is_parallelized", False), (
                "Please call `cache_dit.enable_cache` before Parallelize, "
                "the __patch_transformer_forward__ will overwrite the "
                "parallized forward and cause a downgrade of performance."
            )
            transformer.forward = __patch_transformer_forward__.__get__(transformer)

        transformer._is_patched = is_patched  # True or False

        logger.info(f"Applied {self.__class__.__name__} for {cls_name}, " f"Patch: {is_patched}.")

        return transformer


def __patch_block_forward__(
    self: WanTransformerBlock,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    rotary_emb: torch.Tensor,
    control_hidden_states_list: List[Tuple[torch.Tensor, float]],
) -> Tuple[torch.Tensor, torch.Tensor]:

    if temb.ndim == 4:
        # temb: batch_size, seq_len, 6, inner_dim (wan2.2 ti2v)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table.unsqueeze(0) + temb.float()
        ).chunk(6, dim=2)
        # batch_size, seq_len, 1, inner_dim
        shift_msa = shift_msa.squeeze(2)
        scale_msa = scale_msa.squeeze(2)
        gate_msa = gate_msa.squeeze(2)
        c_shift_msa = c_shift_msa.squeeze(2)
        c_scale_msa = c_scale_msa.squeeze(2)
        c_gate_msa = c_gate_msa.squeeze(2)
    else:
        # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

    # 1. Self-attention
    norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(
        hidden_states
    )
    attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)
    hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

    # 2. Cross-attention
    norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
    attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
    hidden_states = hidden_states + attn_output

    # 3. Feed-forward
    norm_hidden_states = (
        self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa
    ).type_as(hidden_states)
    ff_output = self.ffn(norm_hidden_states)
    hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

    # NOTE(DefTruth): Fused VACE into block forward to support caching.
    i = self._i
    vace_layers = self._vace_layers
    if i in vace_layers:
        control_hint, scale = control_hidden_states_list.pop()
        hidden_states = hidden_states + control_hint * scale

    return hidden_states


def __patch_transformer_forward__(
    self: WanVACETransformer3DModel,
    hidden_states: torch.Tensor,
    timestep: torch.LongTensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    control_hidden_states: torch.Tensor = None,
    control_hidden_states_scale: torch.Tensor = None,
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
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

    batch_size, num_channels, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_patch_num_frames = num_frames // p_t
    post_patch_height = height // p_h
    post_patch_width = width // p_w

    if control_hidden_states_scale is None:
        control_hidden_states_scale = control_hidden_states.new_ones(len(self.config.vace_layers))
    control_hidden_states_scale = torch.unbind(control_hidden_states_scale)
    if len(control_hidden_states_scale) != len(self.config.vace_layers):
        raise ValueError(
            f"Length of `control_hidden_states_scale` {len(control_hidden_states_scale)} should be "
            f"equal to {len(self.config.vace_layers)}."
        )

    # 1. Rotary position embedding
    rotary_emb = self.rope(hidden_states)

    # 2. Patch embedding
    hidden_states = self.patch_embedding(hidden_states)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)

    control_hidden_states = self.vace_patch_embedding(control_hidden_states)
    control_hidden_states = control_hidden_states.flatten(2).transpose(1, 2)
    control_hidden_states_padding = control_hidden_states.new_zeros(
        batch_size,
        hidden_states.size(1) - control_hidden_states.size(1),
        control_hidden_states.size(2),
    )
    control_hidden_states = torch.cat([control_hidden_states, control_hidden_states_padding], dim=1)

    # 3. Time embedding
    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = (
        self.condition_embedder(timestep, encoder_hidden_states, encoder_hidden_states_image)
    )
    timestep_proj = timestep_proj.unflatten(1, (6, -1))

    # 4. Image embedding
    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.concat(
            [encoder_hidden_states_image, encoder_hidden_states], dim=1
        )

    # 5. Transformer blocks
    if torch.is_grad_enabled() and self.gradient_checkpointing:
        # Prepare VACE hints
        control_hidden_states_list = []
        for i, block in enumerate(self.vace_blocks):
            conditioning_states, control_hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                control_hidden_states,
                timestep_proj,
                rotary_emb,
            )
            control_hidden_states_list.append((conditioning_states, control_hidden_states_scale[i]))
        control_hidden_states_list = control_hidden_states_list[::-1]

        for i, block in enumerate(self.blocks):
            hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
            )
            if i in self.config.vace_layers:
                control_hint, scale = control_hidden_states_list.pop()
                hidden_states = hidden_states + control_hint * scale
    else:
        # Prepare VACE hints
        control_hidden_states_list = []
        for i, block in enumerate(self.vace_blocks):
            conditioning_states, control_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                control_hidden_states,
                timestep_proj,
                rotary_emb,
            )
            control_hidden_states_list.append((conditioning_states, control_hidden_states_scale[i]))
        control_hidden_states_list = control_hidden_states_list[::-1]

        for i, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
                control_hidden_states_list,
            )
            # NOTE(DefTruth): Fused into block forward to support caching.
            # if i in self.config.vace_layers:
            #     control_hint, scale = control_hidden_states_list.pop()
            #     hidden_states = hidden_states + control_hint * scale

    # 6. Output norm, projection & unpatchify
    shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

    # Move the shift and scale tensors to the same device as hidden_states.
    # When using multi-GPU inference via accelerate these will be on the
    # first device rather than the last device, which hidden_states ends up
    # on.
    shift = shift.to(hidden_states.device)
    scale = scale.to(hidden_states.device)

    hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(
        hidden_states
    )
    hidden_states = self.proj_out(hidden_states)

    hidden_states = hidden_states.reshape(
        batch_size,
        post_patch_num_frames,
        post_patch_height,
        post_patch_width,
        p_t,
        p_h,
        p_w,
        -1,
    )
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)
