# Adapted from: https://github.com/chengzeyi/ParaAttention/blob/main/src/para_attn/first_block_cache/diffusers_adapters/hunyuan_video.py
import functools
import unittest
from typing import Any, Dict, Optional, Union

import torch
from diffusers import DiffusionPipeline, HunyuanVideoTransformer3DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import (
    scale_lora_layers,
    unscale_lora_layers,
    USE_PEFT_BACKEND,
)

from cache_dit.cache_factory.first_block_cache import cache_context
from cache_dit.logger import init_logger

try:
    from para_attn.para_attn_interface import SparseKVAttnMode

    def is_sparse_kv_attn_available():
        return True

except ImportError:

    class SparseKVAttnMode:
        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_value, traceback):
            pass

    def is_sparse_kv_attn_available():
        return False


logger = init_logger(__name__)  # pylint: disable=invalid-name


def apply_cache_on_transformer(
    transformer: HunyuanVideoTransformer3DModel,
):
    if getattr(transformer, "_is_cached", False):
        return transformer

    cached_transformer_blocks = torch.nn.ModuleList(
        [
            cache_context.CachedTransformerBlocks(
                transformer.transformer_blocks
                + transformer.single_transformer_blocks,
                transformer=transformer,
            )
        ]
    )
    dummy_single_transformer_blocks = torch.nn.ModuleList()

    original_forward = transformer.forward

    @functools.wraps(transformer.__class__.forward)
    def new_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        pooled_projections: torch.Tensor,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        with (
            unittest.mock.patch.object(
                self,
                "transformer_blocks",
                cached_transformer_blocks,
            ),
            unittest.mock.patch.object(
                self,
                "single_transformer_blocks",
                dummy_single_transformer_blocks,
            ),
        ):
            if getattr(self, "_is_parallelized", False):
                return original_forward(
                    hidden_states,
                    timestep,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    pooled_projections,
                    guidance=guidance,
                    attention_kwargs=attention_kwargs,
                    return_dict=return_dict,
                    **kwargs,
                )
            else:
                if attention_kwargs is not None:
                    attention_kwargs = attention_kwargs.copy()
                    lora_scale = attention_kwargs.pop("scale", 1.0)
                else:
                    lora_scale = 1.0

                if USE_PEFT_BACKEND:
                    # weight the lora layers by setting `lora_scale` for each PEFT layer
                    scale_lora_layers(self, lora_scale)
                else:
                    if (
                        attention_kwargs is not None
                        and attention_kwargs.get("scale", None) is not None
                    ):
                        logger.warning(
                            "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                        )

                batch_size, num_channels, num_frames, height, width = (
                    hidden_states.shape
                )
                p, p_t = self.config.patch_size, self.config.patch_size_t
                post_patch_num_frames = num_frames // p_t
                post_patch_height = height // p
                post_patch_width = width // p

                # 1. RoPE
                image_rotary_emb = self.rope(hidden_states)

                # 2. Conditional embeddings
                temb = self.time_text_embed(
                    timestep, guidance, pooled_projections
                )
                hidden_states = self.x_embedder(hidden_states)
                encoder_hidden_states = self.context_embedder(
                    encoder_hidden_states, timestep, encoder_attention_mask
                )

                # 3. Attention mask preparation
                latent_sequence_length = hidden_states.shape[1]
                latent_attention_mask = torch.ones(
                    batch_size,
                    1,
                    latent_sequence_length,
                    device=hidden_states.device,
                    dtype=torch.bool,
                )  # [B, 1, N]
                attention_mask = torch.cat(
                    [
                        latent_attention_mask,
                        encoder_attention_mask.unsqueeze(1).to(torch.bool),
                    ],
                    dim=-1,
                )  # [B, 1, N + M]

                with SparseKVAttnMode():
                    # 4. Transformer blocks
                    hidden_states, encoder_hidden_states = (
                        self.call_transformer_blocks(
                            hidden_states,
                            encoder_hidden_states,
                            temb,
                            attention_mask,
                            image_rotary_emb,
                        )
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
                hidden_states = (
                    hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
                )

                hidden_states = hidden_states.to(timestep.dtype)

                if USE_PEFT_BACKEND:
                    # remove `lora_scale` from each PEFT layer
                    unscale_lora_layers(self, lora_scale)

                if not return_dict:
                    return (hidden_states,)

                return Transformer2DModelOutput(sample=hidden_states)

    transformer.forward = new_forward.__get__(transformer)

    def call_transformer_blocks(
        self, hidden_states, encoder_hidden_states, *args, **kwargs
    ):
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False}

            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = (
                    torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        *args,
                        **kwargs,
                        **ckpt_kwargs,
                    )
                )

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = (
                    torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        *args,
                        **kwargs,
                        **ckpt_kwargs,
                    )
                )

        else:
            for block in self.transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states, encoder_hidden_states, *args, **kwargs
                )

            for block in self.single_transformer_blocks:
                hidden_states, encoder_hidden_states = block(
                    hidden_states, encoder_hidden_states, *args, **kwargs
                )

        return hidden_states, encoder_hidden_states

    transformer.call_transformer_blocks = call_transformer_blocks.__get__(
        transformer
    )

    transformer._is_cached = True

    return transformer


def apply_cache_on_pipe(
    pipe: DiffusionPipeline,
    *,
    shallow_patch: bool = False,
    residual_diff_threshold=0.06,
    downsample_factor=1,
    warmup_steps=0,
    max_cached_steps=-1,
    **kwargs,
):
    cache_kwargs, kwargs = cache_context.collect_cache_kwargs(
        default_attrs={
            "residual_diff_threshold": residual_diff_threshold,
            "downsample_factor": downsample_factor,
            "warmup_steps": warmup_steps,
            "max_cached_steps": max_cached_steps,
        },
        **kwargs,
    )
    if not getattr(pipe, "_is_cached", False):
        original_call = pipe.__class__.__call__

        @functools.wraps(original_call)
        def new_call(self, *args, **kwargs):
            with cache_context.cache_context(
                cache_context.create_cache_context(
                    **cache_kwargs,
                )
            ):
                return original_call(self, *args, **kwargs)

        pipe.__class__.__call__ = new_call
        pipe.__class__._is_cached = True

    if not shallow_patch:
        apply_cache_on_transformer(pipe.transformer, **kwargs)

    return pipe
