import torch
from typing import Optional, Dict, List
from diffusers.models.transformers.transformer_z_image import (
    ZImageTransformer2DModel,
    ZImageTransformerBlock,
    Transformer2DModelOutput,
    SEQ_MULTI_OF,
    pad_sequence,
)

from .functor_base import PatchFunctor
from cache_dit.logger import init_logger

logger = init_logger(__name__)


class ZImageControlNetPatchFunctor(PatchFunctor):
    def _apply(
        self,
        transformer: ZImageTransformer2DModel,
        **kwargs,
    ) -> ZImageTransformer2DModel:
        if hasattr(transformer, "_is_patched"):
            return transformer

        is_patched = False

        for layer_idx, layer in enumerate(transformer.layers):
            if not hasattr(layer, "_is_patched"):
                layer._layer_idx = layer_idx  # type: ignore
                layer.forward = __patch_block_forward__.__get__(layer)

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
    self: ZImageTransformerBlock,
    x: torch.Tensor,
    attn_mask: torch.Tensor,
    freqs_cis: torch.Tensor,
    adaln_input: Optional[torch.Tensor] = None,
    controlnet_block_samples: Optional[Dict[int, torch.Tensor]] = None,
):
    if self.modulation:
        assert adaln_input is not None
        scale_msa, gate_msa, scale_mlp, gate_mlp = (
            self.adaLN_modulation(adaln_input).unsqueeze(1).chunk(4, dim=2)
        )
        gate_msa, gate_mlp = gate_msa.tanh(), gate_mlp.tanh()
        scale_msa, scale_mlp = 1.0 + scale_msa, 1.0 + scale_mlp

        # Attention block
        attn_out = self.attention(
            self.attention_norm1(x) * scale_msa, attention_mask=attn_mask, freqs_cis=freqs_cis
        )
        x = x + gate_msa * self.attention_norm2(attn_out)

        # FFN block
        x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
    else:
        # Attention block
        attn_out = self.attention(
            self.attention_norm1(x), attention_mask=attn_mask, freqs_cis=freqs_cis
        )
        x = x + self.attention_norm2(attn_out)

        # FFN block
        x = x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))

    # ControlNet addition
    if controlnet_block_samples is not None:
        layer_idx = self._layer_idx  # type: ignore
        if layer_idx in controlnet_block_samples:
            controlnet_sample = controlnet_block_samples[layer_idx]

            # NOTE: Make it compatible for context parallelism
            _parallel_config = getattr(self.attention.processor, "_parallel_config", None)
            if _parallel_config is not None:
                cp_config = _parallel_config.context_parallel_config
                if cp_config is not None and cp_config._world_size > 1:
                    # Split controlnet_sample for each device using tensor split
                    #  at sequence dim, which is dim=1.
                    controlnet_sample = torch.tensor_split(
                        controlnet_sample, cp_config._world_size, dim=1
                    )[cp_config._rank]

            x = x + controlnet_sample

    return x


def __patch_transformer_forward__(
    self: ZImageTransformer2DModel,
    x: List[torch.Tensor],
    t,
    cap_feats: List[torch.Tensor],
    controlnet_block_samples: Optional[Dict[int, torch.Tensor]] = None,
    patch_size=2,
    f_patch_size=1,
    return_dict: bool = True,
):
    assert patch_size in self.all_patch_size
    assert f_patch_size in self.all_f_patch_size

    bsz = len(x)
    device = x[0].device
    t = t * self.t_scale
    t = self.t_embedder(t)

    (
        x,
        cap_feats,
        x_size,
        x_pos_ids,
        cap_pos_ids,
        x_inner_pad_mask,
        cap_inner_pad_mask,
    ) = self.patchify_and_embed(x, cap_feats, patch_size, f_patch_size)

    # x embed & refine
    x_item_seqlens = [len(_) for _ in x]
    assert all(_ % SEQ_MULTI_OF == 0 for _ in x_item_seqlens)
    x_max_item_seqlen = max(x_item_seqlens)

    x = torch.cat(x, dim=0)
    x = self.all_x_embedder[f"{patch_size}-{f_patch_size}"](x)

    # Match t_embedder output dtype to x for layerwise casting compatibility
    adaln_input = t.type_as(x)
    x[torch.cat(x_inner_pad_mask)] = self.x_pad_token
    x = list(x.split(x_item_seqlens, dim=0))
    x_freqs_cis = list(
        self.rope_embedder(torch.cat(x_pos_ids, dim=0)).split([len(_) for _ in x_pos_ids], dim=0)
    )

    x = pad_sequence(x, batch_first=True, padding_value=0.0)
    x_freqs_cis = pad_sequence(x_freqs_cis, batch_first=True, padding_value=0.0)
    # Clarify the length matches to satisfy Dynamo due to "Symbolic Shape Inference" to avoid compilation errors
    x_freqs_cis = x_freqs_cis[:, : x.shape[1]]

    x_attn_mask = torch.zeros((bsz, x_max_item_seqlen), dtype=torch.bool, device=device)
    for i, seq_len in enumerate(x_item_seqlens):
        x_attn_mask[i, :seq_len] = 1

    if torch.is_grad_enabled() and self.gradient_checkpointing:
        for layer in self.noise_refiner:
            x = self._gradient_checkpointing_func(layer, x, x_attn_mask, x_freqs_cis, adaln_input)
    else:
        for layer in self.noise_refiner:
            x = layer(x, x_attn_mask, x_freqs_cis, adaln_input)

    # cap embed & refine
    cap_item_seqlens = [len(_) for _ in cap_feats]
    cap_max_item_seqlen = max(cap_item_seqlens)

    cap_feats = torch.cat(cap_feats, dim=0)
    cap_feats = self.cap_embedder(cap_feats)
    cap_feats[torch.cat(cap_inner_pad_mask)] = self.cap_pad_token
    cap_feats = list(cap_feats.split(cap_item_seqlens, dim=0))
    cap_freqs_cis = list(
        self.rope_embedder(torch.cat(cap_pos_ids, dim=0)).split(
            [len(_) for _ in cap_pos_ids], dim=0
        )
    )

    cap_feats = pad_sequence(cap_feats, batch_first=True, padding_value=0.0)
    cap_freqs_cis = pad_sequence(cap_freqs_cis, batch_first=True, padding_value=0.0)
    # Clarify the length matches to satisfy Dynamo due to "Symbolic Shape Inference" to avoid compilation errors
    cap_freqs_cis = cap_freqs_cis[:, : cap_feats.shape[1]]

    cap_attn_mask = torch.zeros((bsz, cap_max_item_seqlen), dtype=torch.bool, device=device)
    for i, seq_len in enumerate(cap_item_seqlens):
        cap_attn_mask[i, :seq_len] = 1

    if torch.is_grad_enabled() and self.gradient_checkpointing:
        for layer in self.context_refiner:
            cap_feats = self._gradient_checkpointing_func(
                layer, cap_feats, cap_attn_mask, cap_freqs_cis
            )
    else:
        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_attn_mask, cap_freqs_cis)

    # unified
    unified = []
    unified_freqs_cis = []
    for i in range(bsz):
        x_len = x_item_seqlens[i]
        cap_len = cap_item_seqlens[i]
        unified.append(torch.cat([x[i][:x_len], cap_feats[i][:cap_len]]))
        unified_freqs_cis.append(torch.cat([x_freqs_cis[i][:x_len], cap_freqs_cis[i][:cap_len]]))
    unified_item_seqlens = [a + b for a, b in zip(cap_item_seqlens, x_item_seqlens)]
    assert unified_item_seqlens == [len(_) for _ in unified]
    unified_max_item_seqlen = max(unified_item_seqlens)

    unified = pad_sequence(unified, batch_first=True, padding_value=0.0)
    unified_freqs_cis = pad_sequence(unified_freqs_cis, batch_first=True, padding_value=0.0)
    unified_attn_mask = torch.zeros((bsz, unified_max_item_seqlen), dtype=torch.bool, device=device)
    for i, seq_len in enumerate(unified_item_seqlens):
        unified_attn_mask[i, :seq_len] = 1

    # NOTE: Already fused controlnet_block_samples into each block forward function.
    if torch.is_grad_enabled() and self.gradient_checkpointing:
        for layer_idx, layer in enumerate(self.layers):
            unified = self._gradient_checkpointing_func(
                layer,
                unified,
                unified_attn_mask,
                unified_freqs_cis,
                adaln_input,
                controlnet_block_samples,
            )
    else:
        for layer_idx, layer in enumerate(self.layers):
            unified = layer(
                unified,
                unified_attn_mask,
                unified_freqs_cis,
                adaln_input,
                controlnet_block_samples,
            )

    unified = self.all_final_layer[f"{patch_size}-{f_patch_size}"](unified, adaln_input)
    unified = list(unified.unbind(dim=0))
    x = self.unpatchify(unified, x_size, patch_size, f_patch_size)

    if not return_dict:
        return (x,)

    return Transformer2DModelOutput(sample=x)
