import torch
from typing import Tuple, Optional, Dict, Any, Union, List
from diffusers import HiDreamImageTransformer2DModel
from diffusers.models.transformers.transformer_hidream_image import (
    HiDreamBlock,
    HiDreamImageTransformerBlock,
    HiDreamImageSingleTransformerBlock,
    Transformer2DModelOutput,
)
from diffusers.utils import (
    deprecate,
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
)
from .functor_base import PatchFunctor
from cache_dit.logger import init_logger

logger = init_logger(__name__)


class HiDreamPatchFunctor(PatchFunctor):

    def _apply(
        self,
        transformer: HiDreamImageTransformer2DModel,
        **kwargs,
    ) -> HiDreamImageTransformer2DModel:
        if hasattr(transformer, "_is_patched"):
            return transformer

        is_patched = False

        _block_id = 0
        for block in transformer.double_stream_blocks:
            assert isinstance(block, HiDreamBlock)
            block.forward = __patch_block_forward__.__get__(block)
            # NOTE: Patch Inner block and block_id
            _block = block.block
            assert isinstance(_block, HiDreamImageTransformerBlock)
            _block._block_id = _block_id
            _block.forward = __patch_double_forward__.__get__(_block)
            _block_id += 1

        for block in transformer.single_stream_blocks:
            assert isinstance(block, HiDreamBlock)
            block.forward = __patch_block_forward__.__get__(block)
            # NOTE: Patch Inner block and block_id
            _block = block.block
            assert isinstance(_block, HiDreamImageSingleTransformerBlock)
            _block._block_id = _block_id
            _block.forward = __patch_single_forward__.__get__(_block)
            _block_id += 1

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


# Adapted from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_hidream_image.py
def __patch_double_forward__(
    self: HiDreamImageTransformerBlock,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,  # initial_encoder_hidden_states
    hidden_states_masks: Optional[torch.Tensor] = None,
    temb: Optional[torch.Tensor] = None,
    image_rotary_emb: torch.Tensor = None,
    llama31_encoder_hidden_states: List[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Assume block_id was patched in transformer forward:
    # for i, block in enumerate(blocks): block._block_id = i;
    block_id = self._block_id
    initial_encoder_hidden_states_seq_len = encoder_hidden_states.shape[1]
    cur_llama31_encoder_hidden_states = llama31_encoder_hidden_states[block_id]
    cur_encoder_hidden_states = torch.cat(
        [encoder_hidden_states, cur_llama31_encoder_hidden_states],
        dim=1,
    )
    encoder_hidden_states = cur_encoder_hidden_states

    wtype = hidden_states.dtype
    (
        shift_msa_i,
        scale_msa_i,
        gate_msa_i,
        shift_mlp_i,
        scale_mlp_i,
        gate_mlp_i,
        shift_msa_t,
        scale_msa_t,
        gate_msa_t,
        shift_mlp_t,
        scale_mlp_t,
        gate_mlp_t,
    ) = self.adaLN_modulation(temb)[:, None].chunk(12, dim=-1)

    # 1. MM-Attention
    norm_hidden_states = self.norm1_i(hidden_states).to(dtype=wtype)
    norm_hidden_states = norm_hidden_states * (1 + scale_msa_i) + shift_msa_i
    norm_encoder_hidden_states = self.norm1_t(encoder_hidden_states).to(dtype=wtype)
    norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + scale_msa_t) + shift_msa_t

    attn_output_i, attn_output_t = self.attn1(
        norm_hidden_states,
        hidden_states_masks,
        norm_encoder_hidden_states,
        image_rotary_emb=image_rotary_emb,
    )

    hidden_states = gate_msa_i * attn_output_i + hidden_states
    encoder_hidden_states = gate_msa_t * attn_output_t + encoder_hidden_states

    # 2. Feed-forward
    norm_hidden_states = self.norm3_i(hidden_states).to(dtype=wtype)
    norm_hidden_states = norm_hidden_states * (1 + scale_mlp_i) + shift_mlp_i
    norm_encoder_hidden_states = self.norm3_t(encoder_hidden_states).to(dtype=wtype)
    norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + scale_mlp_t) + shift_mlp_t

    ff_output_i = gate_mlp_i * self.ff_i(norm_hidden_states)
    ff_output_t = gate_mlp_t * self.ff_t(norm_encoder_hidden_states)
    hidden_states = ff_output_i + hidden_states
    encoder_hidden_states = ff_output_t + encoder_hidden_states

    initial_encoder_hidden_states = encoder_hidden_states[:, :initial_encoder_hidden_states_seq_len]

    return hidden_states, initial_encoder_hidden_states


# Adapted from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_hidream_image.py
def __patch_single_forward__(
    self: HiDreamImageSingleTransformerBlock,
    hidden_states: torch.Tensor,
    hidden_states_masks: Optional[torch.Tensor] = None,
    temb: Optional[torch.Tensor] = None,
    image_rotary_emb: torch.Tensor = None,
    llama31_encoder_hidden_states: List[torch.Tensor] = None,
) -> torch.Tensor:
    # Assume block_id was patched in transformer forward:
    # for i, block in enumerate(blocks): block._block_id = i;
    block_id = self._block_id
    hidden_states_seq_len = hidden_states.shape[1]
    cur_llama31_encoder_hidden_states = llama31_encoder_hidden_states[block_id]
    hidden_states = torch.cat([hidden_states, cur_llama31_encoder_hidden_states], dim=1)

    wtype = hidden_states.dtype
    (
        shift_msa_i,
        scale_msa_i,
        gate_msa_i,
        shift_mlp_i,
        scale_mlp_i,
        gate_mlp_i,
    ) = self.adaLN_modulation(temb)[:, None].chunk(6, dim=-1)

    # 1. MM-Attention
    norm_hidden_states = self.norm1_i(hidden_states).to(dtype=wtype)
    norm_hidden_states = norm_hidden_states * (1 + scale_msa_i) + shift_msa_i
    attn_output_i = self.attn1(
        norm_hidden_states,
        hidden_states_masks,
        image_rotary_emb=image_rotary_emb,
    )
    hidden_states = gate_msa_i * attn_output_i + hidden_states

    # 2. Feed-forward
    norm_hidden_states = self.norm3_i(hidden_states).to(dtype=wtype)
    norm_hidden_states = norm_hidden_states * (1 + scale_mlp_i) + shift_mlp_i
    ff_output_i = gate_mlp_i * self.ff_i(norm_hidden_states.to(dtype=wtype))
    hidden_states = ff_output_i + hidden_states

    hidden_states = hidden_states[:, :hidden_states_seq_len]

    return hidden_states


# Adapted from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_hidream_image.py
def __patch_block_forward__(
    self: HiDreamBlock,
    hidden_states: torch.Tensor,
    *args,
    **kwargs,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    return self.block(hidden_states, *args, **kwargs)


# Adapted from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_hidream_image.py
def __patch_transformer_forward__(
    self: HiDreamImageTransformer2DModel,
    hidden_states: torch.Tensor,
    timesteps: torch.LongTensor = None,
    encoder_hidden_states_t5: torch.Tensor = None,
    encoder_hidden_states_llama3: torch.Tensor = None,
    pooled_embeds: torch.Tensor = None,
    img_ids: Optional[torch.Tensor] = None,
    img_sizes: Optional[List[Tuple[int, int]]] = None,
    hidden_states_masks: Optional[torch.Tensor] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
    **kwargs,
) -> Union[torch.Tensor, Transformer2DModelOutput]:
    encoder_hidden_states = kwargs.get("encoder_hidden_states", None)

    if encoder_hidden_states is not None:
        deprecation_message = "The `encoder_hidden_states` argument is deprecated. Please use `encoder_hidden_states_t5` and `encoder_hidden_states_llama3` instead."
        deprecate("encoder_hidden_states", "0.35.0", deprecation_message)
        encoder_hidden_states_t5 = encoder_hidden_states[0]
        encoder_hidden_states_llama3 = encoder_hidden_states[1]

    if img_ids is not None and img_sizes is not None and hidden_states_masks is None:
        deprecation_message = "Passing `img_ids` and `img_sizes` with unpachified `hidden_states` is deprecated and will be ignored."
        deprecate("img_ids", "0.35.0", deprecation_message)

    if hidden_states_masks is not None and (img_ids is None or img_sizes is None):
        raise ValueError(
            "if `hidden_states_masks` is passed, `img_ids` and `img_sizes` must also be passed."
        )
    elif hidden_states_masks is not None and hidden_states.ndim != 3:
        raise ValueError(
            "if `hidden_states_masks` is passed, `hidden_states` must be a 3D tensors with shape (batch_size, patch_height * patch_width, patch_size * patch_size * channels)"
        )

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

    # spatial forward
    batch_size = hidden_states.shape[0]
    hidden_states_type = hidden_states.dtype

    # Patchify the input
    if hidden_states_masks is None:
        hidden_states, hidden_states_masks, img_sizes, img_ids = self.patchify(hidden_states)

    # Embed the hidden states
    hidden_states = self.x_embedder(hidden_states)

    # 0. time
    timesteps = self.t_embedder(timesteps, hidden_states_type)
    p_embedder = self.p_embedder(pooled_embeds)
    temb = timesteps + p_embedder

    encoder_hidden_states = [encoder_hidden_states_llama3[k] for k in self.config.llama_layers]

    if self.caption_projection is not None:
        new_encoder_hidden_states = []
        for i, enc_hidden_state in enumerate(encoder_hidden_states):
            enc_hidden_state = self.caption_projection[i](enc_hidden_state)
            enc_hidden_state = enc_hidden_state.view(batch_size, -1, hidden_states.shape[-1])
            new_encoder_hidden_states.append(enc_hidden_state)
        encoder_hidden_states = new_encoder_hidden_states
        encoder_hidden_states_t5 = self.caption_projection[-1](encoder_hidden_states_t5)
        encoder_hidden_states_t5 = encoder_hidden_states_t5.view(
            batch_size, -1, hidden_states.shape[-1]
        )
        encoder_hidden_states.append(encoder_hidden_states_t5)

    txt_ids = torch.zeros(
        batch_size,
        encoder_hidden_states[-1].shape[1]
        + encoder_hidden_states[-2].shape[1]
        + encoder_hidden_states[0].shape[1],
        3,
        device=img_ids.device,
        dtype=img_ids.dtype,
    )
    ids = torch.cat((img_ids, txt_ids), dim=1)
    image_rotary_emb = self.pe_embedder(ids)

    # 2. Blocks
    # NOTE: block_id is no-need anymore.
    initial_encoder_hidden_states = torch.cat(
        [encoder_hidden_states[-1], encoder_hidden_states[-2]], dim=1
    )
    llama31_encoder_hidden_states = encoder_hidden_states
    for bid, block in enumerate(self.double_stream_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states, initial_encoder_hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                initial_encoder_hidden_states,
                hidden_states_masks,
                temb,
                image_rotary_emb,
                llama31_encoder_hidden_states,
            )
        else:
            hidden_states, initial_encoder_hidden_states = block(
                hidden_states,
                initial_encoder_hidden_states,  # encoder_hidden_states
                hidden_states_masks=hidden_states_masks,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                llama31_encoder_hidden_states=llama31_encoder_hidden_states,
            )

    image_tokens_seq_len = hidden_states.shape[1]
    hidden_states = torch.cat([hidden_states, initial_encoder_hidden_states], dim=1)
    if hidden_states_masks is not None:
        # NOTE: Patched
        cur_llama31_encoder_hidden_states = llama31_encoder_hidden_states[0]
        encoder_attention_mask_ones = torch.ones(
            (
                batch_size,
                initial_encoder_hidden_states.shape[1] + cur_llama31_encoder_hidden_states.shape[1],
            ),
            device=hidden_states_masks.device,
            dtype=hidden_states_masks.dtype,
        )
        hidden_states_masks = torch.cat([hidden_states_masks, encoder_attention_mask_ones], dim=1)

    for bid, block in enumerate(self.single_stream_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                hidden_states_masks,
                temb,
                image_rotary_emb,
                llama31_encoder_hidden_states,
            )
        else:
            hidden_states = block(
                hidden_states,
                hidden_states_masks=hidden_states_masks,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                llama31_encoder_hidden_states=llama31_encoder_hidden_states,
            )

    hidden_states = hidden_states[:, :image_tokens_seq_len, ...]
    output = self.final_layer(hidden_states, temb)
    output = self.unpatchify(output, img_sizes, self.training)
    if hidden_states_masks is not None:
        hidden_states_masks = hidden_states_masks[:, :image_tokens_seq_len]

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)
