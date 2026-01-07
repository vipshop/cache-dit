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
from cache_dit.parallelism.attention import _unified_all_to_all_o_async_fn
from cache_dit.parallelism.attention import _unified_all_to_all_qkv_async_fn
from cache_dit.parallelism.attention import _prepare_ulysses_comm_metadata
from cache_dit.parallelism.attention import _maybe_patch_find_submodule
from cache_dit.platforms import current_platform

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

        experimental_ulysses_async = kwargs.get("experimental_ulysses_async", False)
        if experimental_ulysses_async:
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
        _maybe_patch_find_submodule()
        n_noise_refiner_layers = len(transformer.noise_refiner)  # 2
        n_context_refiner_layers = len(transformer.context_refiner)  # 2
        n_layers = len(transformer.layers)  # 30
        # controlnet layer idx: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
        # num_controlnet_samples = len(transformer.layers) // 2  # 15
        has_controlnet = kwargs.get("has_controlnet", None)
        if not has_controlnet:
            # cp plan for ZImageTransformer2DModel if no controlnet
            _cp_plan = {
                # 0. Hooks for noise_refiner layers, 2
                "noise_refiner.0": {
                    "x": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
                },
                "noise_refiner.*": {
                    "freqs_cis": ContextParallelInput(
                        split_dim=1, expected_dims=3, split_output=False
                    ),
                },
                f"noise_refiner.{n_noise_refiner_layers - 1}": ContextParallelOutput(
                    gather_dim=1, expected_dims=3
                ),
                # 1. Hooks for context_refiner layers, 2
                "context_refiner.0": {
                    "x": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
                },
                "context_refiner.*": {
                    "freqs_cis": ContextParallelInput(
                        split_dim=1, expected_dims=3, split_output=False
                    ),
                },
                f"context_refiner.{n_context_refiner_layers - 1}": ContextParallelOutput(
                    gather_dim=1, expected_dims=3
                ),
                # 2. Hooks for main transformer layers, num_layers=30
                "layers.0": {
                    "x": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
                },
                "layers.*": {
                    "freqs_cis": ContextParallelInput(
                        split_dim=1, expected_dims=3, split_output=False
                    ),
                },
                # NEED: call _maybe_patch_find_submodule to support ModuleDict like 'all_final_layer'
                "all_final_layer": ContextParallelOutput(gather_dim=1, expected_dims=3),
                # NOTE: The 'all_final_layer' is a ModuleDict of several final layers,
                # each for a specific patch size combination, so we do not add hooks for it here.
                # So, we have to gather the output of the last transformer layer.
                # f"layers.{num_layers - 1}": ContextParallelOutput(gather_dim=1, expected_dims=3),
            }
        else:
            # Special cp plan for ZImageTransformer2DModel with ZImageControlNetModel
            logger.warning(
                "Using special context parallelism plan for ZImageTransformer2DModel "
                "due to the 'has_controlnet' flag is set to True."
            )
            _cp_plan = {
                # zimage controlnet shared the same refiner as zimage, so, we need to
                # add gather hooks for all layers in noise_refiner and context_refiner.
                # 0. Hooks for noise_refiner layers, 2
                # Insert gather hook after each layers due to the ops: (controlnet)
                # - x = x + noise_refiner_block_samples[layer_idx]
                "noise_refiner.*": {
                    "x": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
                    "freqs_cis": ContextParallelInput(
                        split_dim=1, expected_dims=3, split_output=False
                    ),
                },
                **{
                    f"noise_refiner.{i}": ContextParallelOutput(gather_dim=1, expected_dims=3)
                    for i in range(n_noise_refiner_layers)
                },
                # 1. Hooks for context_refiner layers, 2
                "context_refiner.0": {
                    "x": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
                },
                "context_refiner.*": {
                    "freqs_cis": ContextParallelInput(
                        split_dim=1, expected_dims=3, split_output=False
                    ),
                },
                f"context_refiner.{n_context_refiner_layers - 1}": ContextParallelOutput(
                    gather_dim=1, expected_dims=3
                ),
                # 2. Hooks for main transformer layers, num_layers=30
                # Insert gather hook after each layers due to the ops: (main transformer)
                # - unified + controlnet_block_samples[layer_idx]
                "layers.*": {
                    "x": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
                    "freqs_cis": ContextParallelInput(
                        split_dim=1, expected_dims=3, split_output=False
                    ),
                },
                **{
                    f"layers.{i}": ContextParallelOutput(gather_dim=1, expected_dims=3)
                    for i in range(n_layers)
                },
                # NEED: call _maybe_patch_find_submodule to support ModuleDict like 'all_final_layer'
                "all_final_layer": ContextParallelOutput(gather_dim=1, expected_dims=3),
            }
        return _cp_plan


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
    group = ulysses_mesh.get_group()

    _all_to_all_o_async_func = _unified_all_to_all_o_async_fn()
    _all_to_all_qv_async_func = _unified_all_to_all_qkv_async_fn()
    _all_to_all_k_async_func = _unified_all_to_all_qkv_async_fn(fp8=False)

    # Apply RoPE
    def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(current_platform.device_type, enabled=False):
            x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
            freqs_cis = freqs_cis.unsqueeze(2)
            x_out = torch.view_as_real(x * freqs_cis).flatten(3)
            return x_out.type_as(x_in)  # todo

    dtype = hidden_states.dtype
    query = attn.to_q(hidden_states)  # type: torch.Tensor
    query = query.unflatten(-1, (attn.heads, -1))
    if attn.norm_q is not None:  # Apply Norms
        query = attn.norm_q(query)
    if freqs_cis is not None:  # Apply RoPE
        query = apply_rotary_emb(query, freqs_cis)

    metadata = _prepare_ulysses_comm_metadata(query)

    # Async all to all for query
    query_wait = _all_to_all_qv_async_func(query, group, **metadata)

    key = attn.to_k(hidden_states)  # type: torch.Tensor
    key = key.unflatten(-1, (attn.heads, -1))
    if attn.norm_k is not None:  # Apply Norms
        key = attn.norm_k(key)
    if freqs_cis is not None:  # Apply RoPE
        key = apply_rotary_emb(key, freqs_cis)

    # Async all to all for key
    key_wait = _all_to_all_k_async_func(key, group, **metadata)

    value = attn.to_v(hidden_states)  # type: torch.Tensor
    value = value.unflatten(-1, (attn.heads, -1))

    # Async all to all for value
    value_wait = _all_to_all_qv_async_func(value, group, **metadata)

    # Ensure the query, key, value are ready
    query = query_wait()
    key = key_wait()
    value = value_wait()

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
    )  # (B, S_GLOBAL, H_LOCAL, D)

    out_wait = _all_to_all_o_async_func(out, group, **metadata)  # (B, S_LOCAL, H_GLOBAL, D)
    hidden_states = out_wait()  # type: torch.Tensor

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
