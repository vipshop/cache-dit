# Mostly copy from https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/parallelism/transformers/context_parallelism/cp_plan_ltxvideo.py
import functools
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.transformer_ltx2 import (
    LTX2Attention,
    LTX2AudioVideoAttnProcessor,
    LTX2VideoTransformer3DModel,
)
from diffusers.models.attention_dispatch import dispatch_attention_fn

try:
    from diffusers.models._modeling_parallel import (
        ContextParallelInput,
        ContextParallelModelPlan,
        ContextParallelOutput,
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


@ContextParallelismPlannerRegister.register("LTX2")
class LTX2ContextParallelismPlanner(ContextParallelismPlanner):
    def apply(
        self,
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:
        assert transformer is not None, "Transformer must be provided."
        assert isinstance(
            transformer, LTX2VideoTransformer3DModel
        ), "Transformer must be an instance of LTX2VideoTransformer3DModel"

        # NOTE:
        # - LTX2ImageToVideoPipeline passes `timestep` as a 2D tensor (B, seq_len) named `video_timestep`.
        # - diffusers native LTX2 `_cp_plan` does NOT shard `timestep`, causing shape mismatch under CP:
        #     hidden_states: (B, seq_len/world, C) but temb built from timestep.flatten(): (B, seq_len, ...)
        #   leading to: RuntimeError size mismatch (1536 vs 6144).
        # So we must use a custom plan for correctness under Ulysses/Ring CP.
        self._cp_planner_preferred_native_diffusers = False

        # Patch attention_mask preparation for CP head sharding + global seq padding
        LTX2Attention.prepare_attention_mask = __patch__LTX2Attention_prepare_attention_mask__  # type: ignore[assignment]
        LTX2AudioVideoAttnProcessor.__call__ = __patch__LTX2AudioVideoAttnProcessor__call__  # type: ignore[assignment]

        rope_type = getattr(getattr(transformer, "config", None), "rope_type", "interleaved")
        if rope_type == "split":
            # split RoPE returns (B, H, T, D/2), shard along T dim
            rope_expected_dims = 4
            rope_split_dim = 2
        else:
            # interleaved RoPE returns (B, T, D), shard along T dim
            rope_expected_dims = 3
            rope_split_dim = 1

        _cp_plan: ContextParallelModelPlan = {
            "": {
                # Shard video/audio latents across sequence
                "hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
                "audio_hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
                # Shard prompt embeds across sequence
                "encoder_hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
                "audio_encoder_hidden_states": ContextParallelInput(
                    split_dim=1, expected_dims=3, split_output=False
                ),
                # IMPORTANT: shard video timestep (B, seq_len) to match sharded hidden_states
                "timestep": ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
                # NOTE: do NOT shard attention masks; handled in patched attention processor
            },
            # Split RoPE outputs to match CP-sharded sequence length
            "rope": {
                0: ContextParallelInput(
                    split_dim=rope_split_dim, expected_dims=rope_expected_dims, split_output=True
                ),
                1: ContextParallelInput(
                    split_dim=rope_split_dim, expected_dims=rope_expected_dims, split_output=True
                ),
            },
            "audio_rope": {
                0: ContextParallelInput(
                    split_dim=rope_split_dim, expected_dims=rope_expected_dims, split_output=True
                ),
                1: ContextParallelInput(
                    split_dim=rope_split_dim, expected_dims=rope_expected_dims, split_output=True
                ),
            },
            "cross_attn_rope": {
                0: ContextParallelInput(
                    split_dim=rope_split_dim, expected_dims=rope_expected_dims, split_output=True
                ),
                1: ContextParallelInput(
                    split_dim=rope_split_dim, expected_dims=rope_expected_dims, split_output=True
                ),
            },
            "cross_attn_audio_rope": {
                0: ContextParallelInput(
                    split_dim=rope_split_dim, expected_dims=rope_expected_dims, split_output=True
                ),
                1: ContextParallelInput(
                    split_dim=rope_split_dim, expected_dims=rope_expected_dims, split_output=True
                ),
            },
            # Gather outputs before returning
            "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
            "audio_proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        }

        return _cp_plan


# Upstream links (for cross-checking when updating diffusers):
# - https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_ltx2.py
# - https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py


@functools.wraps(LTX2Attention.prepare_attention_mask)
def __patch__LTX2Attention_prepare_attention_mask__(
    self: LTX2Attention,
    attention_mask: torch.Tensor,
    target_length: int,
    batch_size: int,
    out_dim: int = 3,
    # NOTE: Allow specifying head_size for CP
    head_size: Optional[int] = None,
) -> torch.Tensor:
    # Differences vs diffusers:
    # - diffusers signature does not accept `head_size` and always uses `self.heads`.
    # - under Context Parallelism, each rank only owns `attn.heads // world_size` heads.
    #   If we keep repeating the mask with the full `self.heads`, the mask shape will not
    #   match the sharded attention computation.
    if head_size is None:
        head_size = self.heads
    if attention_mask is None:
        return attention_mask

    current_length: int = attention_mask.shape[-1]
    if current_length != target_length:
        if attention_mask.device.type == "mps":
            padding_shape = (attention_mask.shape[0], attention_mask.shape[1], target_length)
            padding = torch.zeros(
                padding_shape, dtype=attention_mask.dtype, device=attention_mask.device
            )
            attention_mask = torch.cat([attention_mask, padding], dim=2)
        else:
            attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

    if out_dim == 3:
        if attention_mask.shape[0] < batch_size * head_size:
            attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
    elif out_dim == 4:
        attention_mask = attention_mask.unsqueeze(1)
        attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

    return attention_mask


@functools.wraps(LTX2AudioVideoAttnProcessor.__call__)
def __patch__LTX2AudioVideoAttnProcessor__call__(
    self: LTX2AudioVideoAttnProcessor,
    attn: "LTX2Attention",
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    query_rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    key_rotary_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    # Differences vs diffusers (transformer_ltx2.py):
    # - diffusers always prepares attention_mask using the *local* `sequence_length` and
    #   reshapes it with `attn.heads`.
    # - when Context Parallelism is enabled, `hidden_states` is sharded on seq dim, so
    #   `sequence_length` here is per-rank. However attention_mask typically corresponds
    #   to the *global* sequence length (before sharding), and each rank only uses a shard
    #   of heads (`attn.heads // world_size`).
    # - this patch therefore:
    #   1) uses `target_length = sequence_length * world_size` when CP is active
    #   2) repeats/reshapes the mask using `head_size = attn.heads // world_size`
    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
    )

    if attention_mask is not None:
        if self._parallel_config is None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )
        else:
            cp_config = getattr(self._parallel_config, "context_parallel_config", None)
            if cp_config is not None and cp_config._world_size > 1:
                head_size = attn.heads // cp_config._world_size
                attention_mask = attn.prepare_attention_mask(
                    attention_mask,
                    sequence_length * cp_config._world_size,
                    batch_size,
                    3,
                    head_size,
                )
                attention_mask = attention_mask.view(
                    batch_size, head_size, -1, attention_mask.shape[-1]
                )
            else:
                attention_mask = attn.prepare_attention_mask(
                    attention_mask, sequence_length, batch_size
                )
                attention_mask = attention_mask.view(
                    batch_size, attn.heads, -1, attention_mask.shape[-1]
                )

    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states

    query = attn.to_q(hidden_states)
    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)

    query = attn.norm_q(query)
    key = attn.norm_k(key)

    if query_rotary_emb is not None:
        # Keep RoPE logic identical to upstream: for v2a/a2v cross-attn, K can use separate RoPE.
        if attn.rope_type == "interleaved":
            from diffusers.models.transformers.transformer_ltx2 import apply_interleaved_rotary_emb

            query = apply_interleaved_rotary_emb(query, query_rotary_emb)
            key = apply_interleaved_rotary_emb(
                key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb
            )
        elif attn.rope_type == "split":
            from diffusers.models.transformers.transformer_ltx2 import apply_split_rotary_emb

            query = apply_split_rotary_emb(query, query_rotary_emb)
            key = apply_split_rotary_emb(
                key, key_rotary_emb if key_rotary_emb is not None else query_rotary_emb
            )

    query = query.unflatten(2, (attn.heads, -1))
    key = key.unflatten(2, (attn.heads, -1))
    value = value.unflatten(2, (attn.heads, -1))

    hidden_states = dispatch_attention_fn(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        backend=self._attention_backend,
        parallel_config=self._parallel_config,
    )
    hidden_states = hidden_states.flatten(2, 3)
    hidden_states = hidden_states.to(query.dtype)

    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)
    return hidden_states
