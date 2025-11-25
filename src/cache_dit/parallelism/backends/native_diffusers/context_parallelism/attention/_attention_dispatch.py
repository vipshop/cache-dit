import os
import torch
from typing import Optional

try:
    from diffusers.models.attention_dispatch import (
        _AttentionBackendRegistry,
        AttentionBackendName,
        _check_device,
        _check_shape,
        TemplatedRingAttention,
        TemplatedUlyssesAttention,
    )
    from diffusers.models._modeling_parallel import ParallelConfig
except ImportError:
    raise ImportError(
        "Context parallelism requires the 'diffusers>=0.36.dev0'."
        "Please install latest version of diffusers from source: \n"
        "pip3 install git+https://github.com/huggingface/diffusers.git"
    )
from cache_dit.logger import init_logger
from ._templated_ulysses_anything import TemplatedUlyssesAnythingAttention
from ._templated_ulysses_anything import is_ulysses_anything_enabled


logger = init_logger(__name__)


__all__ = [
    "_native_attention",
]

# Enable custom native attention backend with context parallelism
# by default. Users can set the environment variable to 0 to disable
# this behavior. Default to enabled for better compatibility.
_CACHE_DIT_ENABLE_CUSTOM_CP_NATIVE_ATTN_DISPATCH = bool(
    int(os.getenv("CACHE_DIT_ENABLE_CUSTOM_CP_NATIVE_ATTN_DISPATCH", "1"))
)


def _is_native_attn_supported_context_parallel() -> bool:
    try:
        return (
            AttentionBackendName.NATIVE in _AttentionBackendRegistry._supports_context_parallel
            and _AttentionBackendRegistry._supports_context_parallel[AttentionBackendName.NATIVE]
        )
    except Exception:
        assert isinstance(_AttentionBackendRegistry._supports_context_parallel, set)
        return (
            AttentionBackendName.NATIVE.value
            in _AttentionBackendRegistry._supports_context_parallel
        )


if _CACHE_DIT_ENABLE_CUSTOM_CP_NATIVE_ATTN_DISPATCH:
    logger.warning(
        "Re-registering NATIVE attention backend to enable context parallelism. "
        "This is a temporary workaround and should be removed after the native "
        "attention backend supports context parallelism natively. Please check: "
        "https://github.com/huggingface/diffusers/pull/12563 for more details. "
        "Or, you can disable this behavior by setting the environment variable "
        "`CACHE_DIT_ENABLE_CUSTOM_CP_NATIVE_ATTN_DISPATCH=0`."
    )
    _AttentionBackendRegistry._backends.pop(AttentionBackendName.NATIVE)
    _AttentionBackendRegistry._constraints.pop(AttentionBackendName.NATIVE)
    _AttentionBackendRegistry._supported_arg_names.pop(AttentionBackendName.NATIVE)
    if _is_native_attn_supported_context_parallel():
        if isinstance(_AttentionBackendRegistry._supports_context_parallel, dict):
            _AttentionBackendRegistry._supports_context_parallel.pop(AttentionBackendName.NATIVE)
        else:
            _AttentionBackendRegistry._supports_context_parallel.remove(
                AttentionBackendName.NATIVE.value
            )

    # Re-define templated context parallel attention to support attn mask
    def _templated_context_parallel_attention_v2(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
        enable_gqa: bool = False,
        return_lse: bool = False,
        *,
        forward_op,
        backward_op,
        _parallel_config: Optional["ParallelConfig"] = None,
    ):
        if attn_mask is not None:
            # NOTE(DefTruth): Check if forward_op is native attention forward op
            forward_op_name = forward_op.__name__
            if not forward_op_name == "_native_attention_forward_op":
                raise ValueError(
                    "Templated context parallel attention with attn_mask "
                    "is only supported for native attention backend, "
                    f"but got forward_op: {forward_op_name}."
                )
        if is_causal:
            raise ValueError("Causal attention is not yet supported for templated attention.")
        if enable_gqa:
            raise ValueError("GQA is not yet supported for templated attention.")

        # TODO: add support for unified attention with ring/ulysses degree both being > 1
        if _parallel_config.context_parallel_config.ring_degree > 1:
            return TemplatedRingAttention.apply(
                query,
                key,
                value,
                attn_mask,
                dropout_p,
                is_causal,
                scale,
                enable_gqa,
                return_lse,
                forward_op,
                backward_op,
                _parallel_config,
            )
        elif _parallel_config.context_parallel_config.ulysses_degree > 1:
            if is_ulysses_anything_enabled():
                return TemplatedUlyssesAnythingAttention.apply(
                    query,
                    key,
                    value,
                    attn_mask,
                    dropout_p,
                    is_causal,
                    scale,
                    enable_gqa,
                    return_lse,
                    forward_op,
                    backward_op,
                    _parallel_config,
                )
            else:
                return TemplatedUlyssesAttention.apply(
                    query,
                    key,
                    value,
                    attn_mask,
                    dropout_p,
                    is_causal,
                    scale,
                    enable_gqa,
                    return_lse,
                    forward_op,
                    backward_op,
                    _parallel_config,
                )
        else:
            raise ValueError("Reaching this branch of code is unexpected. Please report a bug.")

    # NOTE:Remove NATIVE attention backend constraints and re-register it.
    # Here is a temporary workaround to enable context parallelism with
    # native attention backend. We should remove this workaround after
    # the native attention backend supports context parallelism natively.
    # Adapted from: https://github.com/huggingface/diffusers/pull/12563

    def _native_attention_forward_op(
        ctx: torch.autograd.function.FunctionCtx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
        enable_gqa: bool = False,
        return_lse: bool = False,
        _save_ctx: bool = True,
        _parallel_config: Optional["ParallelConfig"] = None,
    ):
        # Native attention does not return_lse
        if return_lse:
            raise ValueError("Native attention does not support return_lse=True")

        # used for backward pass
        if _save_ctx:
            ctx.save_for_backward(query, key, value)
            ctx.attn_mask = attn_mask
            ctx.dropout_p = dropout_p
            ctx.is_causal = is_causal
            ctx.scale = scale
            ctx.enable_gqa = enable_gqa

        query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
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

        return out

    def _native_attention_backward_op(
        ctx: torch.autograd.function.FunctionCtx,
        grad_out: torch.Tensor,
        *args,
        **kwargs,
    ):
        query, key, value = ctx.saved_tensors

        query.requires_grad_(True)
        key.requires_grad_(True)
        value.requires_grad_(True)

        query_t, key_t, value_t = (x.permute(0, 2, 1, 3) for x in (query, key, value))
        out = torch.nn.functional.scaled_dot_product_attention(
            query=query_t,
            key=key_t,
            value=value_t,
            attn_mask=ctx.attn_mask,
            dropout_p=ctx.dropout_p,
            is_causal=ctx.is_causal,
            scale=ctx.scale,
            enable_gqa=ctx.enable_gqa,
        )
        out = out.permute(0, 2, 1, 3)

        grad_out_t = grad_out.permute(0, 2, 1, 3)
        grad_query_t, grad_key_t, grad_value_t = torch.autograd.grad(
            outputs=out,
            inputs=[query_t, key_t, value_t],
            grad_outputs=grad_out_t,
            retain_graph=False,
        )

        grad_query = grad_query_t.permute(0, 2, 1, 3)
        grad_key = grad_key_t.permute(0, 2, 1, 3)
        grad_value = grad_value_t.permute(0, 2, 1, 3)

        return grad_query, grad_key, grad_value

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
            raise ValueError("Native attention backend does not support setting `return_lse=True`.")
        if _parallel_config is None:
            query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
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
        else:
            out = _templated_context_parallel_attention_v2(
                query,
                key,
                value,
                attn_mask,
                dropout_p,
                is_causal,
                scale,
                enable_gqa,
                return_lse,
                forward_op=_native_attention_forward_op,
                backward_op=_native_attention_backward_op,
                _parallel_config=_parallel_config,
            )
        return out

else:
    from diffusers.models.attention_dispatch import (
        _native_attention,
    )  # noqa: F401

    logger.info("Native attention backend already supports context parallelism.")
