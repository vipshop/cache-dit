import functools
import unittest
import inspect

import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any, Union
from diffusers import DiffusionPipeline, FluxTransformer2DModel
from diffusers.models.transformers.transformer_flux import (
    FluxSingleTransformerBlock,
    Transformer2DModelOutput,
)
from diffusers.utils import (
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
)
from cache_dit.cache_factory.dual_block_cache import (
    cache_context,
    DBCachedTransformerBlocks,
)

from cache_dit.logger import init_logger

logger = init_logger(__name__)


# copy from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux.py#L380
def __patch_single_forward__(
    self: FluxSingleTransformerBlock,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    text_seq_len = encoder_hidden_states.shape[1]
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    residual = hidden_states
    norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
    mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
    joint_attention_kwargs = joint_attention_kwargs or {}
    attn_output = self.attn(
        hidden_states=norm_hidden_states,
        image_rotary_emb=image_rotary_emb,
        **joint_attention_kwargs,
    )

    hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
    gate = gate.unsqueeze(1)
    hidden_states = gate * self.proj_out(hidden_states)
    hidden_states = residual + hidden_states
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    encoder_hidden_states, hidden_states = (
        hidden_states[:, :text_seq_len],
        hidden_states[:, text_seq_len:],
    )
    return encoder_hidden_states, hidden_states


# copy from: https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_flux.py#L631
def __patch_transformer_forward__(
    self: FluxTransformer2DModel,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
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
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
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

    if (
        joint_attention_kwargs is not None
        and "ip_adapter_image_embeds" in joint_attention_kwargs
    ):
        ip_adapter_image_embeds = joint_attention_kwargs.pop(
            "ip_adapter_image_embeds"
        )
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

    for index_block, block in enumerate(self.transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = (
                self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    joint_attention_kwargs,
                )
            )

        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # controlnet residual
        if controlnet_block_samples is not None:
            interval_control = len(self.transformer_blocks) / len(
                controlnet_block_samples
            )
            interval_control = int(np.ceil(interval_control))
            # For Xlabs ControlNet.
            if controlnet_blocks_repeat:
                hidden_states = (
                    hidden_states
                    + controlnet_block_samples[
                        index_block % len(controlnet_block_samples)
                    ]
                )
            else:
                hidden_states = (
                    hidden_states
                    + controlnet_block_samples[index_block // interval_control]
                )

    for index_block, block in enumerate(self.single_transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = (
                self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    joint_attention_kwargs,
                )
            )

        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # controlnet residual
        if controlnet_single_block_samples is not None:
            interval_control = len(self.single_transformer_blocks) / len(
                controlnet_single_block_samples
            )
            interval_control = int(np.ceil(interval_control))
            hidden_states = (
                hidden_states
                + controlnet_single_block_samples[
                    index_block // interval_control
                ]
            )

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)


def maybe_patch_transformer(transformer: FluxTransformer2DModel):
    single_forward_parameters = inspect.signature(
        transformer.single_transformer_blocks[0].forward
    ).parameters.keys()
    if "encoder_hidden_states" not in single_forward_parameters:
        logger.warning("Patch FluxSingleTransformerBlock for DBCache.")
        for block in transformer.single_transformer_blocks:
            block.forward = __patch_single_forward__.__get__(block)

        transformer.forward = __patch_transformer_forward__.__get__(transformer)
        transformer._is_patched = True

    return transformer


def apply_db_cache_on_transformer(
    transformer: FluxTransformer2DModel,
):
    if getattr(transformer, "_is_cached", False):
        return transformer

    transformer = maybe_patch_transformer(transformer)

    cached_transformer_blocks = torch.nn.ModuleList(
        [
            DBCachedTransformerBlocks(
                transformer.transformer_blocks
                + transformer.single_transformer_blocks,
                transformer=transformer,
                return_hidden_states_first=False,
            )
        ]
    )
    dummy_single_transformer_blocks = torch.nn.ModuleList()

    original_forward = transformer.forward

    @functools.wraps(original_forward)
    def new_forward(
        self,
        *args,
        **kwargs,
    ):
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
            return original_forward(
                *args,
                **kwargs,
            )

    transformer.forward = new_forward.__get__(transformer)

    transformer._is_cached = True

    return transformer


def apply_db_cache_on_pipe(
    pipe: DiffusionPipeline,
    *,
    shallow_patch: bool = False,
    residual_diff_threshold=0.05,
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
        apply_db_cache_on_transformer(pipe.transformer)

    return pipe
