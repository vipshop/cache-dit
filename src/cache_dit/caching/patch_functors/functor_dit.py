import torch
import torch.nn.functional as F

from typing import Optional, Dict, Any
from diffusers.models.transformers.dit_transformer_2d import (
    DiTTransformer2DModel,
    Transformer2DModelOutput,
)
from .functor_base import PatchFunctor
from cache_dit.logger import init_logger

logger = init_logger(__name__)


class DiTPatchFunctor(PatchFunctor):

    def _apply(
        self,
        transformer: DiTTransformer2DModel,
        **kwargs,
    ) -> DiTTransformer2DModel:
        if hasattr(transformer, "_is_patched"):
            return transformer

        is_patched = False

        transformer._norm1_emb = transformer.transformer_blocks[0].norm1.emb

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


def __patch_transformer_forward__(
    self: DiTTransformer2DModel,
    hidden_states: torch.Tensor,
    timestep: Optional[torch.LongTensor] = None,
    class_labels: Optional[torch.LongTensor] = None,
    cross_attention_kwargs: Dict[str, Any] = None,
    return_dict: bool = True,
):
    height, width = (
        hidden_states.shape[-2] // self.patch_size,
        hidden_states.shape[-1] // self.patch_size,
    )
    hidden_states = self.pos_embed(hidden_states)

    # 2. Blocks
    for block in self.transformer_blocks:
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                None,
                None,
                None,
                timestep,
                cross_attention_kwargs,
                class_labels,
            )
        else:
            hidden_states = block(
                hidden_states,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )

    # 3. Output
    # conditioning = self.transformer_blocks[0].norm1.emb(timestep, class_labels, hidden_dtype=hidden_states.dtype)
    conditioning = self._norm1_emb(timestep, class_labels, hidden_dtype=hidden_states.dtype)
    shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
    hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
    hidden_states = self.proj_out_2(hidden_states)

    # unpatchify
    height = width = int(hidden_states.shape[1] ** 0.5)
    hidden_states = hidden_states.reshape(
        shape=(
            -1,
            height,
            width,
            self.patch_size,
            self.patch_size,
            self.out_channels,
        )
    )
    hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
    output = hidden_states.reshape(
        shape=(
            -1,
            self.out_channels,
            height * self.patch_size,
            width * self.patch_size,
        )
    )

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)
