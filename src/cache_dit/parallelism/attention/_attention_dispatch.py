import torch
import math
from typing import Optional

try:
    from diffusers.models.attention_dispatch import (
        _AttentionBackendRegistry,
        AttentionBackendName,
        _check_device,
        _check_shape,
        _check_qkv_dtype_bf16_or_fp16,
        _check_device_cuda,
    )

    # For sage attention backend re-registration
    from diffusers.models.attention_dispatch import (
        sageattn,
        _sage_attention_forward_op,
        _sage_attention_backward_op,
    )

    # For flash attention 3 backend re-registration
    try:
        from flash_attn_interface import flash_attn_func as flash_attn_3_func

        _flash_attn_3_available = True
    except ImportError:
        flash_attn_3_func = None
        _flash_attn_3_available = False

    # For native npu attention backend re-registration
    try:
        from torch_npu import npu_fusion_attention
    except ImportError:
        npu_fusion_attention = None

    from diffusers.models._modeling_parallel import ParallelConfig
except ImportError:
    raise ImportError(
        "Context parallelism requires the 'diffusers>=0.36.dev0'."
        "Please install latest version of diffusers from source: \n"
        "pip3 install git+https://github.com/huggingface/diffusers.git"
    )
from cache_dit.logger import init_logger
from cache_dit.envs import ENV

from ._templated_ring import UnifiedTemplatedRingAttention
from ._templated_ulysses import UnifiedTemplatedUlyssesAttention
from ._templated_usp import UnifiedTemplatedUSPAttention


logger = init_logger(__name__)
MAX_TOKEN = 2147483647

__all__ = [
    "_native_attention",
    "_sdpa_cudnn_attention",
    "_sage_attention",
    "_flash_attention_3",
    "_native_npu_attention",
]


def _registry_pop_attn_backend(attn_backend: AttentionBackendName):
    _AttentionBackendRegistry._backends.pop(attn_backend, None)
    _AttentionBackendRegistry._constraints.pop(attn_backend, None)
    _AttentionBackendRegistry._supported_arg_names.pop(attn_backend, None)
    if isinstance(_AttentionBackendRegistry._supports_context_parallel, dict):
        _AttentionBackendRegistry._supports_context_parallel.pop(attn_backend, None)
    elif attn_backend in _AttentionBackendRegistry._supports_context_parallel:
        _AttentionBackendRegistry._supports_context_parallel.remove(attn_backend.value)


def _set_new_attn_backend(member: str, value: str):
    # e.g., _set_new_attn_backend("_SDPA_CUDNN", "_sdpa_cudnn")
    new_member = str.__new__(AttentionBackendName, value)
    new_member._name_ = member
    new_member._value_ = value
    setattr(AttentionBackendName, member, new_member)
    AttentionBackendName._member_map_[member] = new_member
    AttentionBackendName._member_names_.append(member)
    AttentionBackendName._value2member_map_[value] = new_member


# Enable custom native attention backend with context parallelism
# by default. Users can set the environment variable to 0 to disable
# this behavior. Default to enabled for better compatibility.
if ENV.CACHE_DIT_ENABLE_CUSTOM_ATTN_DISPATCH:
    _ATTENTION_OPS_ALLOW_ATTN_MASK = [
        "_native_attention_forward_op",
        "_sdpa_cudnn_attention_forward_op",
        "_npu_attention_forward_op",
    ]

    # Re-define templated context parallel attention to support attn mask
    def _unified_templated_context_parallel_attention(
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
            if forward_op_name not in _ATTENTION_OPS_ALLOW_ATTN_MASK:
                raise ValueError(
                    "Templated context parallel attention with attn_mask "
                    "is only supported for native attention backend, "
                    f"but got forward_op: {forward_op_name}."
                )
        if is_causal:
            raise ValueError("Causal attention is not yet supported for templated attention.")
        if enable_gqa:
            raise ValueError("GQA is not yet supported for templated attention.")

        if (
            _parallel_config.context_parallel_config.ring_degree > 1
            and _parallel_config.context_parallel_config.ulysses_degree > 1
        ):
            return UnifiedTemplatedUSPAttention.apply(
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
        elif _parallel_config.context_parallel_config.ring_degree > 1:
            return UnifiedTemplatedRingAttention.apply(
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
            return UnifiedTemplatedUlyssesAttention.apply(
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

    # NOTE: Remove NATIVE attention backend constraints and re-register it.
    # Here is a temporary workaround to enable context parallelism with
    # native attention backend for attn mask support until diffusers
    # officially support it.

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
        # used for backward pass
        if _save_ctx:
            ctx.save_for_backward(query, key, value)
            ctx.attn_mask = attn_mask
            ctx.dropout_p = dropout_p
            ctx.is_causal = is_causal
            ctx.scale = scale
            ctx.enable_gqa = enable_gqa

        if return_lse:
            # Use native flash attention to get lse if return_lse is True
            if attn_mask is not None:
                raise ValueError(
                    "`attn_mask` is not yet supported for native flash attention with lse."
                )
            out, lse = torch.ops.aten._scaled_dot_product_flash_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
            )[:2]
            # [B, H, N, D] -> [B, N, H, D]
            out = out.transpose(1, 2)  # type: torch.Tensor
            lse = lse.transpose(1, 2)  # type: torch.Tensor
            if lse.dim() == 3:
                # [B, N, H] -> [B, N, H, 1]
                lse = lse.unsqueeze(-1)
            return out, lse

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

    # Re-register NATIVE attention backend to allow attn mask while using context parallelism
    _registry_pop_attn_backend(AttentionBackendName.NATIVE)

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
            out = _unified_templated_context_parallel_attention(
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

    logger.info(
        "Re-registered NATIVE attention backend to enable context parallelism "
        "with attn mask in cache-dit. You can disable this behavior by: "
        "export CACHE_DIT_ENABLE_CUSTOM_ATTN_DISPATCH=0."
    )

    def _sdpa_cudnn_attention_forward_op(
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
            raise ValueError("cudnn attention with sdpa does not support return_lse=True")

        # used for backward pass
        if _save_ctx:
            ctx.save_for_backward(query, key, value)
            ctx.attn_mask = attn_mask
            ctx.dropout_p = dropout_p
            ctx.is_causal = is_causal
            ctx.scale = scale
            ctx.enable_gqa = enable_gqa

        query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
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

    def _sdpa_cudnn_attention_backward_op(
        ctx: torch.autograd.function.FunctionCtx,
        grad_out: torch.Tensor,
        *args,
        **kwargs,
    ):
        raise NotImplementedError("Backward for cudnn attention with sdpa is not implemented yet.")

    # Register _sdpa_cudnn_attention backend to allow attn mask while using context parallelism
    _set_new_attn_backend("_SDPA_CUDNN", "_sdpa_cudnn")
    assert hasattr(AttentionBackendName, "_SDPA_CUDNN")

    @_AttentionBackendRegistry.register(
        AttentionBackendName._SDPA_CUDNN,  # type: AttentionBackendName
        constraints=[_check_device, _check_shape],
        supports_context_parallel=True,
    )
    def _sdpa_cudnn_attention(
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
        lse = None
        if _parallel_config is None and not return_lse:
            query, key, value = (x.permute(0, 2, 1, 3).contiguous() for x in (query, key, value))
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.CUDNN_ATTENTION):
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
            out = _unified_templated_context_parallel_attention(
                query,
                key,
                value,
                attn_mask,
                dropout_p,
                is_causal,
                scale,
                enable_gqa,
                return_lse,
                forward_op=_sdpa_cudnn_attention_forward_op,
                backward_op=_sdpa_cudnn_attention_backward_op,
                _parallel_config=_parallel_config,
            )
            if return_lse:
                out, lse = out

        return (out, lse) if return_lse else out

    logger.info(
        "Registered new attention backend: _SDPA_CUDNN to enable context "
        "parallelism with attn mask in cache-dit. You can disable it by: "
        "export CACHE_DIT_ENABLE_CUSTOM_ATTN_DISPATCH=0."
    )

    _registry_pop_attn_backend(AttentionBackendName.SAGE)

    @_AttentionBackendRegistry.register(
        AttentionBackendName.SAGE,
        constraints=[_check_device_cuda, _check_qkv_dtype_bf16_or_fp16, _check_shape],
        supports_context_parallel=True,
    )
    def _sage_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool = False,
        scale: Optional[float] = None,
        return_lse: bool = False,
        _parallel_config: Optional["ParallelConfig"] = None,
    ) -> torch.Tensor:
        lse = None
        if _parallel_config is None:
            out = sageattn(
                q=query,
                k=key,
                v=value,
                tensor_layout="NHD",
                is_causal=is_causal,
                sm_scale=scale,
                return_lse=return_lse,
            )
            if return_lse:
                out, lse, *_ = out
        else:
            out = _unified_templated_context_parallel_attention(
                query,
                key,
                value,
                None,
                0.0,
                is_causal,
                scale,
                False,
                return_lse,
                forward_op=_sage_attention_forward_op,
                backward_op=_sage_attention_backward_op,
                _parallel_config=_parallel_config,
            )
            if return_lse:
                out, lse = out

        return (out, lse) if return_lse else out

    logger.info(
        "Re-registered SAGE attention backend to enable context parallelism "
        "with FP8 Attention in cache-dit. You can disable this behavior by: "
        "export CACHE_DIT_ENABLE_CUSTOM_ATTN_DISPATCH=0."
    )

    # Flash Attention 3 forward op implementation (inference only)
    def _flash_attention_3_forward_op(
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
        """Flash Attention 3 forward operation for cache-dit (inference only)."""
        if attn_mask is not None:
            raise ValueError("`attn_mask` is not yet supported for flash-attn 3.")
        if enable_gqa:
            raise ValueError("`enable_gqa` is not yet supported for flash-attn 3.")
        if dropout_p > 0.0:
            raise ValueError("`dropout_p` > 0 is not yet supported for flash-attn 3.")

        if scale is None:
            scale = query.shape[-1] ** (-0.5)

        if _save_ctx:
            logger.warning(
                "Flash Attention 3 is configured for inference only, but _save_ctx=True was passed. "
                "Context will not be saved."
            )

        # Hardcoded parameters for FA3
        window_size = (-1, -1)
        softcap = 0.0
        deterministic = False

        out = flash_attn_3_func(
            q=query,
            k=key,
            v=value,
            softmax_scale=scale,
            causal=is_causal,
            qv=None,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            window_size=window_size,
            attention_chunk=0,
            softcap=softcap,
            num_splits=1,
            pack_gqa=None,
            deterministic=deterministic,
            sm_margin=0,
            return_attn_probs=return_lse,
        )
        if return_lse:
            out, lse = out
            lse = lse.permute(0, 2, 1)
            return out, lse
        else:
            return out

    def _flash_attention_3_backward_op(
        ctx: torch.autograd.function.FunctionCtx,
        grad_out: torch.Tensor,
        *args,
        **kwargs,
    ):
        """Flash Attention 3 backward operation for cache-dit."""
        raise NotImplementedError(
            "Backward pass for Flash Attention 3 with context parallelism is not implemented yet in cache-dit."
        )

    # Re-register Flash Attention 3 backend
    if _flash_attn_3_available:
        if hasattr(AttentionBackendName, "_FLASH_3"):
            _registry_pop_attn_backend(AttentionBackendName._FLASH_3)
        else:
            logger.info("AttentionBackendName._FLASH_3 not found, creating new backend.")
            _set_new_attn_backend("_FLASH_3", "_flash_3")
            assert hasattr(AttentionBackendName, "_FLASH_3")

        @_AttentionBackendRegistry.register(
            AttentionBackendName._FLASH_3,
            constraints=[_check_device, _check_qkv_dtype_bf16_or_fp16, _check_shape],
            supports_context_parallel=True,
        )
        def _flash_attention_3(
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            scale: Optional[float] = None,
            is_causal: bool = False,
            return_lse: bool = False,
            _parallel_config: Optional["ParallelConfig"] = None,
        ) -> torch.Tensor:
            lse = None
            if _parallel_config is None:
                # Non-parallel: use native flash-attn-3
                window_size = (-1, -1)
                softcap = 0.0
                deterministic = False
                out = flash_attn_3_func(
                    q=query,
                    k=key,
                    v=value,
                    softmax_scale=scale,
                    causal=is_causal,
                    qv=None,
                    q_descale=None,
                    k_descale=None,
                    v_descale=None,
                    window_size=window_size,
                    attention_chunk=0,
                    softcap=softcap,
                    num_splits=1,
                    pack_gqa=None,
                    deterministic=deterministic,
                    sm_margin=0,
                    return_attn_probs=return_lse,
                )
                if return_lse:
                    out, lse = out
                    lse = lse.permute(0, 2, 1)
            else:
                # Parallel: use cache-dit's optimized implementation
                out = _unified_templated_context_parallel_attention(
                    query,
                    key,
                    value,
                    None,  # attn_mask not supported by FA3
                    0.0,  # dropout_p
                    is_causal,
                    scale,
                    False,  # enable_gqa
                    return_lse,
                    forward_op=_flash_attention_3_forward_op,
                    backward_op=_flash_attention_3_backward_op,
                    _parallel_config=_parallel_config,
                )
                if return_lse:
                    out, lse = out

            return (out, lse) if return_lse else out

        logger.info(
            "Re-registered FLASH_3 attention backend to enable context parallelism "
            "with Ulysses Anything/Float8 in cache-dit. You can disable this behavior by: "
            "export CACHE_DIT_ENABLE_CUSTOM_ATTN_DISPATCH=0."
        )
    else:
        _flash_attention_3 = None  # type: ignore[assignment]
        logger.info("Flash Attention 3 not available, skipping _FLASH_3 backend registration.")

    _registry_pop_attn_backend(AttentionBackendName._NATIVE_NPU)

    def _maybe_modify_attn_mask_npu(
        query: torch.Tensor, key: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ):
        # Skip Attention Mask if all values are 1, `None` mask can speedup the computation
        if attn_mask is not None and torch.all(attn_mask != 0):
            attn_mask = None

        # Reshape Attention Mask: [batch_size, seq_len_k] -> [batch_size, 1, sqe_len_q, seq_len_k]
        # https://www.hiascend.com/document/detail/zh/Pytorch/730/apiref/torchnpuCustomsapi/docs/context/torch_npu-npu_fusion_attention.md
        if (
            attn_mask is not None
            and attn_mask.ndim == 2
            and attn_mask.shape[0] == query.shape[0]
            and attn_mask.shape[1] == key.shape[1]
        ):
            B, Sq, Skv = attn_mask.shape[0], query.shape[1], key.shape[1]
            attn_mask = ~attn_mask.to(torch.bool)
            attn_mask = attn_mask.unsqueeze(1).expand(B, Sq, Skv).unsqueeze(1).contiguous()

        return attn_mask

    @_AttentionBackendRegistry.register(
        AttentionBackendName._NATIVE_NPU,
        constraints=[_check_device, _check_qkv_dtype_bf16_or_fp16, _check_shape],
        supports_context_parallel=True,
    )
    def _native_npu_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        scale: Optional[float] = None,
        return_lse: bool = False,
        _parallel_config: Optional["ParallelConfig"] = None,
    ) -> torch.Tensor:
        if return_lse:
            raise ValueError("NPU attention backend does not support setting `return_lse=True`.")
        if _parallel_config is None:
            attn_mask = _maybe_modify_attn_mask_npu(query, key, attn_mask)
            out = npu_fusion_attention(
                query,
                key,
                value,
                atten_mask=attn_mask,
                input_layout="BSND",
                scale=1.0 / math.sqrt(query.shape[-1]) if scale is None else scale,
                pre_tockens=MAX_TOKEN,
                next_tockens=MAX_TOKEN,
                head_num=query.size(2),
            )[0]
        else:
            out = _unified_templated_context_parallel_attention(
                query,
                key,
                value,
                attn_mask,
                dropout_p,
                None,
                1.0 / math.sqrt(query.shape[-1]) if scale is None else scale,
                None,
                return_lse,
                forward_op=_npu_attention_forward_op,
                backward_op=_npu_attention_backward_op,
                _parallel_config=_parallel_config,
            )
        return out

    logger.info(
        "Re-registered _NATIVE_NPU attention backend to enable context parallelism "
        "You can disable this behavior by: "
        "export CACHE_DIT_ENABLE_CUSTOM_ATTN_DISPATCH=0."
    )

    def _npu_attention_forward_op(
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
        if return_lse:
            raise ValueError("NPU attention backend does not support setting `return_lse=True`.")

        attn_mask = _maybe_modify_attn_mask_npu(query, key, attn_mask)
        out = npu_fusion_attention(
            query,
            key,
            value,
            atten_mask=attn_mask,
            input_layout="BSND",
            scale=scale,
            pre_tockens=MAX_TOKEN,
            next_tockens=MAX_TOKEN,
            head_num=query.size(2),
        )[0]

        return out

    def _npu_attention_backward_op(
        ctx: torch.autograd.function.FunctionCtx,
        grad_out: torch.Tensor,
        *args,
        **kwargs,
    ):
        raise NotImplementedError("Backward pass is not implemented for Npu Fusion Attention.")

else:
    from diffusers.models.attention_dispatch import (
        _native_attention,
        _sage_attention,
    )  # noqa: F401

    try:
        from diffusers.models.attention_dispatch import _flash_attention_3  # noqa: F401
    except ImportError:
        _flash_attention_3 = None  # type: ignore[assignment]

    _sdpa_cudnn_attention = None  # type: ignore[assignment]
    _native_npu_attention = None  # type: ignore[assignment]

    logger.info("Skipped custom attention backend registration in cache-dit.")
