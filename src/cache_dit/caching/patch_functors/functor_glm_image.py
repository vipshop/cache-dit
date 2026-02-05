import torch
from typing import Tuple, Optional, Dict, Any, Union, List
from diffusers import GlmImageTransformer2DModel
from diffusers.models.transformers.transformer_glm_image import (
    GlmImageTransformerBlock,
    GlmImageKVCache,
    Transformer2DModelOutput,
)


from .functor_base import PatchFunctor
from cache_dit.logger import init_logger

logger = init_logger(__name__)


class GlmImagePatchFunctor(PatchFunctor):

    def _apply(
        self,
        transformer: GlmImageTransformer2DModel,
        blocks: torch.nn.ModuleList = None,
        **kwargs,
    ) -> GlmImageTransformer2DModel:

        if hasattr(transformer, "_is_patched"):
            return transformer

        is_patched = False
        _idx = 0

        for block in transformer.transformer_blocks:
            assert isinstance(block, GlmImageTransformerBlock)
            block._idx = _idx
            block.forward = __patch_block_forward__.__get__(block)
            _idx += 1

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
    self: GlmImageTransformerBlock,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: Optional[torch.Tensor] = None,
    image_rotary_emb: Optional[
        Union[Tuple[torch.Tensor, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]
    ] = None,
    attention_mask: Optional[Dict[str, torch.Tensor]] = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    kv_caches: Optional[GlmImageKVCache] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    # 1. Timestep conditioning
    (
        norm_hidden_states,
        gate_msa,
        shift_mlp,
        scale_mlp,
        gate_mlp,
        norm_encoder_hidden_states,
        c_gate_msa,
        c_shift_mlp,
        c_scale_mlp,
        c_gate_mlp,
    ) = self.norm1(hidden_states, encoder_hidden_states, temb)

    # 2. Attention
    attention_kwargs = attention_kwargs or {}

    attn_hidden_states, attn_encoder_hidden_states = self.attn1(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        image_rotary_emb=image_rotary_emb,
        attention_mask=attention_mask,
        # Patched in cache-dit to avoid dynamic indexing, compatible with cached blocks design
        kv_cache=kv_caches[self._idx] if kv_caches is not None else None,
        **attention_kwargs,
    )
    hidden_states = hidden_states + attn_hidden_states * gate_msa.unsqueeze(1)
    encoder_hidden_states = (
        encoder_hidden_states + attn_encoder_hidden_states * c_gate_msa.unsqueeze(1)
    )

    # 3. Feedforward
    norm_hidden_states = self.norm2(hidden_states) * (
        1 + scale_mlp.unsqueeze(1)
    ) + shift_mlp.unsqueeze(1)
    norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states) * (
        1 + c_scale_mlp.unsqueeze(1)
    ) + c_shift_mlp.unsqueeze(1)

    ff_output = self.ff(norm_hidden_states)
    ff_output_context = self.ff(norm_encoder_hidden_states)
    hidden_states = hidden_states + ff_output * gate_mlp.unsqueeze(1)
    encoder_hidden_states = encoder_hidden_states + ff_output_context * c_gate_mlp.unsqueeze(1)

    return hidden_states, encoder_hidden_states


def __patch_transformer_forward__(
    self: GlmImageTransformer2DModel,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    prior_token_id: torch.Tensor,
    prior_token_drop: torch.Tensor,
    timestep: torch.LongTensor,
    target_size: torch.Tensor,
    crop_coords: torch.Tensor,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
    attention_mask: Optional[torch.Tensor] = None,
    kv_caches: Optional[GlmImageKVCache] = None,
    image_rotary_emb: Optional[
        Union[Tuple[torch.Tensor, torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]
    ] = None,
) -> Union[Tuple[torch.Tensor], Transformer2DModelOutput]:
    batch_size, num_channels, height, width = hidden_states.shape

    # 1. RoPE
    if image_rotary_emb is None:
        image_rotary_emb = self.rope(hidden_states)

    # 2. Patch & Timestep embeddings
    p = self.config.patch_size
    post_patch_height = height // p
    post_patch_width = width // p

    hidden_states = self.image_projector(hidden_states)
    encoder_hidden_states = self.glyph_projector(encoder_hidden_states)
    prior_embedding = self.prior_token_embedding(prior_token_id)
    prior_embedding[prior_token_drop] *= 0.0
    prior_hidden_states = self.prior_projector(prior_embedding)

    hidden_states = hidden_states + prior_hidden_states

    temb = self.time_condition_embed(timestep, target_size, crop_coords, hidden_states.dtype)

    # 3. Transformer blocks
    for idx, block in enumerate(self.transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                attention_mask,
                attention_kwargs,
                kv_caches if kv_caches is not None else None,
            )
        else:
            hidden_states, encoder_hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                attention_mask,
                attention_kwargs,
                # pass kv_caches directly, avoid dynamic indexing
                # make it compatible with cached blocks design
                kv_caches=kv_caches if kv_caches is not None else None,
            )

    # 4. Output norm & projection
    hidden_states = self.norm_out(hidden_states, temb)
    hidden_states = self.proj_out(hidden_states)

    # 5. Unpatchify
    hidden_states = hidden_states.reshape(batch_size, post_patch_height, post_patch_width, -1, p, p)

    # Rearrange tensor from (B, H_p, W_p, C, p, p) to (B, C, H_p * p, W_p * p)
    output = hidden_states.permute(0, 3, 1, 4, 2, 5).flatten(4, 5).flatten(2, 3)

    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)
