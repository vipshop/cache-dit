import torch
from typing import Optional, Union, List
from diffusers import HunyuanDiT2DModel
from diffusers.models.transformers.hunyuan_transformer_2d import (
    HunyuanDiTBlock,
    Transformer2DModelOutput,
)
from .functor_base import PatchFunctor
from cache_dit.logger import init_logger

logger = init_logger(__name__)


class HunyuanDiTPatchFunctor(PatchFunctor):

    def _apply(
        self,
        transformer: HunyuanDiT2DModel,
        **kwargs,
    ) -> HunyuanDiT2DModel:
        if hasattr(transformer, "_is_patched"):
            return transformer

        is_patched = False

        num_layers = transformer.config.num_layers
        layer_id = 0
        for block in transformer.blocks:
            assert isinstance(block, HunyuanDiTBlock)
            block._num_layers = num_layers
            block._layer_id = layer_id
            block.forward = __patch_block_forward__.__get__(block)
            layer_id += 1

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
    self: HunyuanDiTBlock,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    temb: Optional[torch.Tensor] = None,
    image_rotary_emb: torch.Tensor = None,
    controlnet_block_samples: torch.Tensor = None,
    skips: List[torch.Tensor] = [],
) -> torch.Tensor:
    # Notice that normalization is always applied before the real computation in the following blocks.
    # 0. Long Skip Connection
    num_layers = self._num_layers
    layer_id = self._layer_id

    if layer_id > num_layers // 2:
        if controlnet_block_samples is not None:
            skip = skips.pop() + controlnet_block_samples.pop()
        else:
            skip = skips.pop()
    else:
        skip = None

    if self.skip_linear is not None:
        cat = torch.cat([hidden_states, skip], dim=-1)
        cat = self.skip_norm(cat)
        hidden_states = self.skip_linear(cat)

    # 1. Self-Attention
    norm_hidden_states = self.norm1(hidden_states, temb)  # checked: self.norm1 is correct
    attn_output = self.attn1(
        norm_hidden_states,
        image_rotary_emb=image_rotary_emb,
    )
    hidden_states = hidden_states + attn_output

    # 2. Cross-Attention
    hidden_states = hidden_states + self.attn2(
        self.norm2(hidden_states),
        encoder_hidden_states=encoder_hidden_states,
        image_rotary_emb=image_rotary_emb,
    )

    # FFN Layer
    mlp_inputs = self.norm3(hidden_states)
    hidden_states = hidden_states + self.ff(mlp_inputs)

    if layer_id < (num_layers // 2 - 1):
        skips.append(hidden_states)

    return hidden_states


def __patch_transformer_forward__(
    self: HunyuanDiT2DModel,
    hidden_states,
    timestep,
    encoder_hidden_states=None,
    text_embedding_mask=None,
    encoder_hidden_states_t5=None,
    text_embedding_mask_t5=None,
    image_meta_size=None,
    style=None,
    image_rotary_emb=None,
    controlnet_block_samples=None,
    return_dict=True,
) -> Union[torch.Tensor, Transformer2DModelOutput]:
    height, width = hidden_states.shape[-2:]

    hidden_states = self.pos_embed(hidden_states)

    temb = self.time_extra_emb(
        timestep,
        encoder_hidden_states_t5,
        image_meta_size,
        style,
        hidden_dtype=timestep.dtype,
    )  # [B, D]

    # text projection
    batch_size, sequence_length, _ = encoder_hidden_states_t5.shape
    encoder_hidden_states_t5 = self.text_embedder(
        encoder_hidden_states_t5.view(-1, encoder_hidden_states_t5.shape[-1])
    )
    encoder_hidden_states_t5 = encoder_hidden_states_t5.view(batch_size, sequence_length, -1)

    encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_t5], dim=1)
    text_embedding_mask = torch.cat([text_embedding_mask, text_embedding_mask_t5], dim=-1)
    text_embedding_mask = text_embedding_mask.unsqueeze(2).bool()

    encoder_hidden_states = torch.where(
        text_embedding_mask, encoder_hidden_states, self.text_embedding_padding
    )

    skips = []
    for layer, block in enumerate(self.blocks):
        hidden_states = block(
            hidden_states,
            temb=temb,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            controlnet_block_samples=controlnet_block_samples,
            skips=skips,
        )  # (N, L, D)

    if controlnet_block_samples is not None and len(controlnet_block_samples) != 0:
        raise ValueError("The number of controls is not equal to the number of skip connections.")

    # final layer
    hidden_states = self.norm_out(hidden_states, temb.to(torch.float32))
    hidden_states = self.proj_out(hidden_states)
    # (N, L, patch_size ** 2 * out_channels)

    # unpatchify: (N, out_channels, H, W)
    patch_size = self.pos_embed.patch_size
    height = height // patch_size
    width = width // patch_size

    hidden_states = hidden_states.reshape(
        shape=(
            hidden_states.shape[0],
            height,
            width,
            patch_size,
            patch_size,
            self.out_channels,
        )
    )
    hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
    output = hidden_states.reshape(
        shape=(
            hidden_states.shape[0],
            self.out_channels,
            height * patch_size,
            width * patch_size,
        )
    )
    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)
