import torch
import functools
from typing import Optional
from torch.distributed import DeviceMesh
from diffusers.models.modeling_utils import ModelMixin
from diffusers import ZImageTransformer2DModel
from diffusers.models.transformers.transformer_z_image import (
    ZSingleStreamAttnProcessor,
    dispatch_attention_fn,
    Attention,
)

try:
    from diffusers.models._modeling_parallel import (
        ContextParallelInput,
        ContextParallelOutput,
        ContextParallelModelPlan,
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
from .attention._distributed_primitives import _unified_all_to_all_fn
from .attention._distributed_primitives import _unified_all_to_all_async_fn
from .attention._templated_ulysses import is_ulysses_anything_enabled
from ..utils import maybe_patch_cp_find_submodule_by_name

from cache_dit.logger import init_logger

logger = init_logger(__name__)


@ContextParallelismPlannerRegister.register("ZImageTransformer2DModel")
class ZImageContextParallelismPlanner(ContextParallelismPlanner):
    def apply(
        self,
        transformer: Optional[torch.nn.Module | ModelMixin] = None,
        **kwargs,
    ) -> ContextParallelModelPlan:

        # NOTE: Diffusers native CP plan still not supported for ZImageTransformer2DModel
        self._cp_planner_preferred_native_diffusers = False

        if transformer is not None and self._cp_planner_preferred_native_diffusers:
            assert isinstance(
                transformer, ZImageTransformer2DModel
            ), "Transformer must be an instance of ZImageTransformer2DModel"
            if hasattr(transformer, "_cp_plan"):
                if transformer._cp_plan is not None:
                    return transformer._cp_plan

        experimental_ulysses_async_qkv_proj = kwargs.get(
            "experimental_ulysses_async_qkv_proj", False
        )
        if experimental_ulysses_async_qkv_proj:
            assert not is_ulysses_anything_enabled(), (
                "experimental_ulysses_async_qkv_proj is not compatible with "
                "experimental_ulysses_anything, please disable one of them."
            )
            ZSingleStreamAttnProcessor.__call__ = (
                __patch_ZSingleStreamAttnProcessor_ulysses_async__call__
            )

            logger.info(
                "Enabled experimental Async QKV Projection with Ulysses style "
                "Context Parallelism for ZImageTransformer2DModel."
            )

        # NOTE: This only a temporary workaround for ZImage to make context parallelism
        # work compatible with DBCache FnB0. The better way is to make DBCache fully
        # compatible with diffusers native context parallelism, e.g., check the split/gather
        # hooks in each block/layer in the initialization of DBCache.
        # Issue: https://github.com/vipshop/cache-dit/issues/498
        maybe_patch_cp_find_submodule_by_name()
        # TODO: Patch rotary embedding function to avoid complex number ops
        n_noise_refiner_layers = len(transformer.noise_refiner)  # 2
        n_context_refiner_layers = len(transformer.context_refiner)  # 2
        # num_layers = len(transformer.layers)  # 30
        _cp_plan = {
            # 0. Hooks for noise_refiner layers, 2
            "noise_refiner.0": {
                "x": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            },
            "noise_refiner.*": {
                "freqs_cis": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            },
            f"noise_refiner.{n_noise_refiner_layers - 1}": ContextParallelOutput(
                gather_dim=1, expected_dims=3
            ),
            # 1. Hooks for context_refiner layers, 2
            "context_refiner.0": {
                "x": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            },
            "context_refiner.*": {
                "freqs_cis": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            },
            f"context_refiner.{n_context_refiner_layers - 1}": ContextParallelOutput(
                gather_dim=1, expected_dims=3
            ),
            # 2. Hooks for main transformer layers, num_layers=30
            "layers.0": {
                "x": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            },
            "layers.*": {
                "freqs_cis": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            },
            # NEED: call maybe_patch_cp_find_submodule_by_name to support ModuleDict like 'all_final_layer'
            "all_final_layer": ContextParallelOutput(gather_dim=1, expected_dims=3),
            # NOTE: The 'all_final_layer' is a ModuleDict of several final layers,
            # each for a specific patch size combination, so we do not add hooks for it here.
            # So, we have to gather the output of the last transformer layer.
            # f"layers.{num_layers - 1}": ContextParallelOutput(gather_dim=1, expected_dims=3),
        }
        return _cp_plan


# TODO: Original implementation using complex numbers, which is not be supported in torch.compile yet.
# May be Reference:
# - https://github.com/triple-Mu/Z-Image-TensorRT/blob/4efc5749e9a0d22344e6c4b8a09d2223dd0a7e17/step_by_step/2-remove-complex-op.py#L26C1-L36C25
# - https://github.com/huggingface/diffusers/pull/12725


# NOTE: Support Async Ulysses QKV projection for Z-Image
def _ulysses_attn_with_async_qkv_proj_zimage(
    self: ZSingleStreamAttnProcessor,
    attn: Attention,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    freqs_cis: Optional[torch.Tensor] = None,
) -> torch.Tensor:

    ulysses_mesh: DeviceMesh = self._parallel_config.context_parallel_config._ulysses_mesh
    world_size = self._parallel_config.context_parallel_config.ulysses_degree
    group = ulysses_mesh.get_group()

    _all_to_all_single = _unified_all_to_all_fn()
    _all_to_all_single_async = _unified_all_to_all_async_fn()

    # Apply RoPE
    def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast("cuda", enabled=False):
            x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
            freqs_cis = freqs_cis.unsqueeze(2)
            x_out = torch.view_as_real(x * freqs_cis).flatten(3)
            return x_out.type_as(x_in)  # todo

    dtype = hidden_states.dtype
    # NOTE: Reorder to compute Value first to get more oppurtunity to
    # overlap the computation of norm_q/k and RoPE.
    value = attn.to_v(hidden_states)  # type: torch.Tensor
    value = value.unflatten(-1, (attn.heads, -1))

    B, S_KV_LOCAL, H, D = value.shape
    H_LOCAL = H // world_size

    # 0. Async all to all for value
    value = value.reshape(B, S_KV_LOCAL, world_size, H_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
    value = _all_to_all_single_async(value, group)

    query = attn.to_q(hidden_states)  # type: torch.Tensor
    query = query.unflatten(-1, (attn.heads, -1))
    if attn.norm_q is not None:  # Apply Norms
        query = attn.norm_q(query)
    if freqs_cis is not None:  # Apply RoPE
        query = apply_rotary_emb(query, freqs_cis)

    # 1. Async all to all for query
    _, S_Q_LOCAL, _, _ = query.shape
    query = query.reshape(B, S_Q_LOCAL, world_size, H_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
    query = _all_to_all_single_async(query, group)

    key = attn.to_k(hidden_states)  # type: torch.Tensor
    key = key.unflatten(-1, (attn.heads, -1))
    if attn.norm_k is not None:  # Apply Norms
        key = attn.norm_k(key)
    if freqs_cis is not None:  # Apply RoPE
        key = apply_rotary_emb(key, freqs_cis)

    # 2. Async all to all for key
    key = key.reshape(B, S_KV_LOCAL, world_size, H_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
    key = _all_to_all_single_async(key, group)

    # (S_GLOBAL, B, H_LOCAL, D) -> (B, S_GLOBAL, H_LOCAL, D)
    value = value()
    value = value.flatten(0, 1).permute(1, 0, 2, 3).contiguous()

    query = query()
    query = query.flatten(0, 1).permute(1, 0, 2, 3).contiguous()

    key = key()
    key = key.flatten(0, 1).permute(1, 0, 2, 3).contiguous()

    # Cast to correct dtype
    query, key = query.to(dtype), key.to(dtype)

    # From [batch, seq_len] to [batch, 1, 1, seq_len] -> broadcast to [batch, heads, seq_len, seq_len]
    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = attention_mask[:, None, None, :]

    # Compute joint attention
    out = dispatch_attention_fn(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        backend=self._attention_backend,
        parallel_config=None,  # set to None to avoid double parallelism
    )

    out = out.reshape(B, world_size, S_Q_LOCAL, H_LOCAL, D).permute(1, 3, 0, 2, 4).contiguous()
    out = _all_to_all_single(out, group)
    hidden_states = out.flatten(0, 1).permute(1, 2, 0, 3).contiguous()

    # Reshape back
    hidden_states = hidden_states.flatten(2, 3)
    hidden_states = hidden_states.to(dtype)

    output = attn.to_out[0](hidden_states)
    if len(attn.to_out) > 1:  # dropout
        output = attn.to_out[1](output)

    return output


ZSingleStreamAttnProcessor_original__call__ = ZSingleStreamAttnProcessor.__call__


@functools.wraps(ZSingleStreamAttnProcessor_original__call__)
def __patch_ZSingleStreamAttnProcessor_ulysses_async__call__(
    self: ZSingleStreamAttnProcessor,
    attn: Attention,
    hidden_states: torch.Tensor,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    freqs_cis: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if (
        self._parallel_config is not None
        and hasattr(self._parallel_config, "context_parallel_config")
        and self._parallel_config.context_parallel_config is not None
        and self._parallel_config.context_parallel_config.ulysses_degree > 1
    ):
        return _ulysses_attn_with_async_qkv_proj_zimage(
            self,
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            freqs_cis,
        )
    else:
        return ZSingleStreamAttnProcessor_original__call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            freqs_cis,
        )
