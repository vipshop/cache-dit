import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union
from diffusers import ChromaTransformer2DModel
from diffusers.models.transformers.transformer_chroma import (
    ChromaTransformerBlock,
    ChromaSingleTransformerBlock,
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


class ChromaPatchFunctor(PatchFunctor):

    def _apply(
        self,
        transformer: ChromaTransformer2DModel,
        **kwargs,
    ) -> ChromaTransformer2DModel:
        if hasattr(transformer, "_is_patched"):
            return transformer

        is_patched = False
        for index_block, block in enumerate(transformer.transformer_blocks):
            assert isinstance(block, ChromaTransformerBlock)
            img_offset = 3 * len(transformer.single_transformer_blocks)
            txt_offset = img_offset + 6 * len(transformer.transformer_blocks)
            img_modulation = img_offset + 6 * index_block
            text_modulation = txt_offset + 6 * index_block
            block._img_modulation = img_modulation
            block._text_modulation = text_modulation
            block.forward = __patch_double_forward__.__get__(block)

        for index_block, block in enumerate(transformer.single_transformer_blocks):
            assert isinstance(block, ChromaSingleTransformerBlock)
            start_idx = 3 * index_block
            block._start_idx = start_idx
            block.forward = __patch_single_forward__.__get__(block)

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


# Adapted from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_chroma.py
def __patch_double_forward__(
    self: ChromaTransformerBlock,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    pooled_temb: torch.Tensor,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # TODO: Fuse controlnet into block forward
    img_modulation = self._img_modulation
    text_modulation = self._text_modulation
    temb = torch.cat(
        (
            pooled_temb[:, img_modulation : img_modulation + 6],
            pooled_temb[:, text_modulation : text_modulation + 6],
        ),
        dim=1,
    )

    temb_img, temb_txt = temb[:, :6], temb[:, 6:]
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
        hidden_states, emb=temb_img
    )

    (
        norm_encoder_hidden_states,
        c_gate_msa,
        c_shift_mlp,
        c_scale_mlp,
        c_gate_mlp,
    ) = self.norm1_context(encoder_hidden_states, emb=temb_txt)
    joint_attention_kwargs = joint_attention_kwargs or {}
    if attention_mask is not None:
        attention_mask = attention_mask[:, None, None, :] * attention_mask[:, None, :, None]

    # Attention.
    attention_outputs = self.attn(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        image_rotary_emb=image_rotary_emb,
        attention_mask=attention_mask,
        **joint_attention_kwargs,
    )

    if len(attention_outputs) == 2:
        attn_output, context_attn_output = attention_outputs
    elif len(attention_outputs) == 3:
        attn_output, context_attn_output, ip_attn_output = attention_outputs

    # Process attention outputs for the `hidden_states`.
    attn_output = gate_msa.unsqueeze(1) * attn_output
    hidden_states = hidden_states + attn_output

    norm_hidden_states = self.norm2(hidden_states)
    norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

    ff_output = self.ff(norm_hidden_states)
    ff_output = gate_mlp.unsqueeze(1) * ff_output

    hidden_states = hidden_states + ff_output
    if len(attention_outputs) == 3:
        hidden_states = hidden_states + ip_attn_output

    # Process attention outputs for the `encoder_hidden_states`.

    context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
    encoder_hidden_states = encoder_hidden_states + context_attn_output

    norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
    norm_encoder_hidden_states = (
        norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
    )

    context_ff_output = self.ff_context(norm_encoder_hidden_states)
    encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

    return encoder_hidden_states, hidden_states


# adapted from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_chroma.py
def __patch_single_forward__(
    self: ChromaSingleTransformerBlock,  # Almost same as FluxSingleTransformerBlock
    hidden_states: torch.Tensor,
    pooled_temb: torch.Tensor,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    # TODO: Fuse controlnet into block forward
    start_idx = self._start_idx
    temb = pooled_temb[:, start_idx : start_idx + 3]

    residual = hidden_states
    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
    mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
    joint_attention_kwargs = joint_attention_kwargs or {}

    if attention_mask is not None:
        attention_mask = attention_mask[:, None, None, :] * attention_mask[:, None, :, None]

    attn_output = self.attn(
        hidden_states=norm_hidden_states,
        image_rotary_emb=image_rotary_emb,
        attention_mask=attention_mask,
        **joint_attention_kwargs,
    )

    hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
    gate = gate.unsqueeze(1)
    hidden_states = gate * self.proj_out(hidden_states)
    hidden_states = residual + hidden_states
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    return hidden_states


# Adapted from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_chroma.py
def __patch_transformer_forward__(
    self: ChromaTransformer2DModel,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    attention_mask: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_block_samples=None,
    controlnet_single_block_samples=None,
    return_dict: bool = True,
    controlnet_blocks_repeat: bool = False,
) -> Union[torch.Tensor, Transformer2DModelOutput]:
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if (
            joint_attention_kwargs is not None
            and joint_attention_kwargs.get("scale", None) is not None
        ):
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

    hidden_states = self.x_embedder(hidden_states)

    timestep = timestep.to(hidden_states.dtype) * 1000

    input_vec = self.time_text_embed(timestep)
    pooled_temb = self.distilled_guidance_layer(input_vec)

    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids.ndim == 3:
        logger.warning(
            "Passing `txt_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        logger.warning(
            "Passing `img_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        img_ids = img_ids[0]

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)

    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

    for index_block, block in enumerate(self.transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                pooled_temb,
                image_rotary_emb,
                attention_mask,
            )

        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_temb=pooled_temb,
                image_rotary_emb=image_rotary_emb,
                attention_mask=attention_mask,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # TODO: Fuse controlnet into block forward
        # controlnet residual
        if controlnet_block_samples is not None:
            interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
            interval_control = int(np.ceil(interval_control))
            # For Xlabs ControlNet.
            if controlnet_blocks_repeat:
                hidden_states = (
                    hidden_states
                    + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                )
            else:
                hidden_states = (
                    hidden_states + controlnet_block_samples[index_block // interval_control]
                )

    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    for index_block, block in enumerate(self.single_transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                pooled_temb,
                image_rotary_emb,
                attention_mask,
                joint_attention_kwargs,
            )

        else:
            hidden_states = block(
                hidden_states=hidden_states,
                pooled_temb=pooled_temb,
                image_rotary_emb=image_rotary_emb,
                attention_mask=attention_mask,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # TODO: Fuse controlnet into block forward
        # controlnet residual
        if controlnet_single_block_samples is not None:
            interval_control = len(self.single_transformer_blocks) / len(
                controlnet_single_block_samples
            )
            interval_control = int(np.ceil(interval_control))
            hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                + controlnet_single_block_samples[index_block // interval_control]
            )

    hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]
    temb = pooled_temb[:, -2:]
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)
