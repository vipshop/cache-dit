import torch
from typing import Optional

try:
    from diffusers.models.attention_dispatch import (
        _AttentionBackendRegistry,
        AttentionBackendName,
        _check_device,
        _check_shape,
        _all_to_all_single,
    )
    from diffusers.models._modeling_parallel import ParallelConfig
except ImportError:
    raise ImportError(
        "Context parallelism requires the 'diffusers>=0.36.dev0'."
        "Please install latest version of diffusers from source: \n"
        "pip3 install git+https://github.com/huggingface/diffusers.git"
    )
from cache_dit.logger import init_logger

logger = init_logger(__name__)


__all__ = [
    "_native_attention",
]


def _is_native_attention_backend_supported_context_parallel() -> bool:
    try:
        return (
            AttentionBackendName.NATIVE
            in _AttentionBackendRegistry._supports_context_parallel
            and _AttentionBackendRegistry._supports_context_parallel[
                AttentionBackendName.NATIVE
            ]
        )
    except AttributeError:
        assert isinstance(
            _AttentionBackendRegistry._supports_context_parallel, set
        )
        return (
            AttentionBackendName.NATIVE.value
            in _AttentionBackendRegistry._supports_context_parallel
        )


if not _is_native_attention_backend_supported_context_parallel():
    logger.warning(
        "Re-registering NATIVE attention backend to enable context parallelism. "
        "This is a temporary workaround and should be removed after the native "
        "attention backend supports context parallelism natively."
    )
    _AttentionBackendRegistry._backends.pop(AttentionBackendName.NATIVE)
    _AttentionBackendRegistry._constraints.pop(AttentionBackendName.NATIVE)
    _AttentionBackendRegistry._supported_arg_names.pop(
        AttentionBackendName.NATIVE
    )

    # NOTE:Remove NATIVE attention backend constraints and re-register it.
    # Here is a temporary workaround to enable context parallelism with
    # native attention backend. We should remove this workaround after
    # the native attention backend supports context parallelism natively.
    # Adapted from: https://github.com/huggingface/diffusers/pull/12563
    @_AttentionBackendRegistry.register(
        AttentionBackendName.NATIVE,
        constraints=[_check_device, _check_shape],
        supports_context_parallel=True,
    )
    def _native_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
        enable_gqa: bool = False,
        return_lse: bool = False,
        _parallel_config: Optional["ParallelConfig"] = None,
    ) -> torch.Tensor:
        if return_lse:
            raise ValueError(
                "Native attention backend does not support setting `return_lse=True`."
            )
        if _parallel_config is None:
            query, key, value = (
                x.permute(0, 2, 1, 3) for x in (query, key, value)
            )
            out = torch.nn.functional.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )
            out = out.permute(0, 2, 1, 3)
        elif _parallel_config.context_parallel_config.ring_degree == 1:
            ulysses_mesh = (
                _parallel_config.context_parallel_config._ulysses_mesh
            )
            world_size = _parallel_config.context_parallel_config.ulysses_degree
            group = ulysses_mesh.get_group()

            batch_size, seq_len_q_local, num_heads, head_dim = query.shape
            _, seq_len_kv_local, _, _ = key.shape
            num_heads_local = num_heads // world_size
            query = (
                query.reshape(
                    batch_size,
                    seq_len_q_local,
                    world_size,
                    num_heads_local,
                    head_dim,
                )
                .permute(2, 1, 0, 3, 4)
                .contiguous()
            )
            key = (
                key.reshape(
                    batch_size,
                    seq_len_kv_local,
                    world_size,
                    num_heads_local,
                    head_dim,
                )
                .permute(2, 1, 0, 3, 4)
                .contiguous()
            )
            value = (
                value.reshape(
                    batch_size,
                    seq_len_kv_local,
                    world_size,
                    num_heads_local,
                    head_dim,
                )
                .permute(2, 1, 0, 3, 4)
                .contiguous()
            )
            query, key, value = (
                _all_to_all_single(x, group) for x in (query, key, value)
            )
            query, key, value = (
                x.flatten(0, 1).permute(1, 2, 0, 3).contiguous()
                for x in (query, key, value)
            )

            out = torch.nn.functional.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )
            out = (
                out.reshape(
                    batch_size,
                    num_heads_local,
                    world_size,
                    seq_len_q_local,
                    head_dim,
                )
                .permute(2, 1, 0, 3, 4)
                .contiguous()
            )
            out = _all_to_all_single(out, group)
            out = out.flatten(0, 1).permute(1, 2, 0, 3).contiguous()
        else:
            raise ValueError(
                "Native attention backend does not support context parallelism with `ring_degree` > 1, try Ulysses Attention instead by specifying `ulysses_degree` > 1."
            )
        return out

else:
    from diffusers.models.attention_dispatch import (
        _native_attention,
    )  # noqa: F401

    logger.info(
        "Native attention backend already supports context parallelism."
    )
