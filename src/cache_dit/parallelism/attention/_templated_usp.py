import torch
from typing import Optional
from ._distributed_primitives import (
    _all_to_all_single_qkv_async,
    _all_to_all_single_o_async,
    _all_to_all_single_qkv_fp8_async,
    _all_to_all_single_o_fp8_async,
    _prepare_ulysses_comm_metadata,
)

try:
    from ._templated_ring import UnifiedTemplatedRingAttention
    from ._templated_ulysses import is_ulysses_float8_enabled
    from diffusers.models._modeling_parallel import ParallelConfig
except ImportError:
    raise ImportError(
        "Context parallelism requires the 'diffusers>=0.36.dev0'."
        "Please install latest version of diffusers from source: \n"
        "pip3 install git+https://github.com/huggingface/diffusers.git"
    )


__all__ = ["UnifiedTemplatedUSPAttention"]


class UnifiedTemplatedUSPAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        dropout_p: float,
        is_causal: bool,
        scale: Optional[float],
        enable_gqa: bool,
        return_lse: bool,
        forward_op,
        backward_op,
        _parallel_config: Optional["ParallelConfig"] = None,
    ):
        if is_ulysses_float8_enabled():
            return _TemplatedUSPAttentionFloat8.apply(
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
            return _TemplatedUSPAttention.apply(
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

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_out: torch.Tensor,
        *args,
    ):
        raise NotImplementedError("USP attention backward is not implemented yet.")


class _TemplatedUSPAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        dropout_p: float,
        is_causal: bool,
        scale: Optional[float],
        enable_gqa: bool,
        return_lse: bool,
        forward_op,
        backward_op,
        _parallel_config: Optional["ParallelConfig"] = None,
    ):
        # USP step 0: Apply Ulysses group all-to-all to collect distributed Q, K, V
        ulysses_mesh = _parallel_config.context_parallel_config._ulysses_mesh
        ulysses_group = ulysses_mesh.get_group()

        metadata = _prepare_ulysses_comm_metadata(query)
        query_wait = _all_to_all_single_qkv_async(query, ulysses_group, **metadata)
        key_wait = _all_to_all_single_qkv_async(key, ulysses_group, **metadata)
        value_wait = _all_to_all_single_qkv_async(value, ulysses_group, **metadata)

        query = query_wait()  # type: torch.Tensor
        key = key_wait()  # type: torch.Tensor
        value = value_wait()  # type: torch.Tensor

        # USP step 1: Apply Ring attention to process the collected partial Q, K, V
        out = UnifiedTemplatedRingAttention.apply(
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

        if return_lse:
            out, lse, *_ = out

        # USP step 2: Apply Ulysses group all-to-all to redistribute the output
        out_wait = _all_to_all_single_o_async(out, ulysses_group, **metadata)

        if return_lse:
            if lse.dim() == 3:
                lse = lse.unsqueeze(-1)  # (B, S_Q_GLOBAL, H_LOCAL, D=1)
            lse_wait = _all_to_all_single_o_async(lse, ulysses_group, **metadata)
            out = out_wait()  # type: torch.Tensor
            lse = lse_wait()  # type: torch.Tensor
            lse = lse.squeeze(-1).contiguous()  # (B, S_Q_GLOBAL, H_LOCAL)
        else:
            out = out_wait()  # type: torch.Tensor
            lse = None
        return (out, lse) if return_lse else out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_out: torch.Tensor,
        *args,
    ):
        raise NotImplementedError("USP attention backward is not implemented yet.")


class _TemplatedUSPAttentionFloat8(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        dropout_p: float,
        is_causal: bool,
        scale: Optional[float],
        enable_gqa: bool,
        return_lse: bool,
        forward_op,
        backward_op,
        _parallel_config: Optional["ParallelConfig"] = None,
    ):
        # USP step 0: Apply Ulysses group all-to-all to collect distributed Q, K, V
        ulysses_mesh = _parallel_config.context_parallel_config._ulysses_mesh
        ulysses_group = ulysses_mesh.get_group()

        metadata = _prepare_ulysses_comm_metadata(query)
        # Use async all_to_all to overlap comm and quant/dequant computation
        # NOTE: Currently, we choose to keep K in FP16/BF16 format to keep higher
        # precision during softmax computation: Softmax(Q@K^T) which is sensitive to
        # numerical instability. So we only use float8 all_to_all for Q, V and O.
        # TODO: We should relax this design and support all QKV in float8 format while
        # the K-per-channel-smooth (e.g., in SageAttention) is used to improve numerical
        # stability. Using this smooth technique before All-to-All on K may introduce
        # extra AllReduce communication overhead.
        key_wait = _all_to_all_single_qkv_async(key, ulysses_group, **metadata)
        query_wait = _all_to_all_single_qkv_fp8_async(query, ulysses_group, **metadata)
        value_wait = _all_to_all_single_qkv_fp8_async(value, ulysses_group, **metadata)

        query = query_wait()  # type: torch.Tensor
        value = value_wait()  # type: torch.Tensor
        key = key_wait()  # type: torch.Tensor

        # USP step 1: Apply Ring attention to process the collected partial Q, K, V
        out = UnifiedTemplatedRingAttention.apply(
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

        if return_lse:
            out, lse, *_ = out

        # USP step 2: Apply Ulysses group all-to-all to redistribute the output
        out_wait = _all_to_all_single_o_fp8_async(out, ulysses_group, **metadata)

        if return_lse:
            if lse.dim() == 3:
                lse = lse.unsqueeze(-1)  # (B, S_Q_GLOBAL, H_LOCAL, D=1)
            lse_wait = _all_to_all_single_o_fp8_async(lse, ulysses_group, **metadata)
            out = out_wait()  # type: torch.Tensor
            lse = lse_wait()  # type: torch.Tensor
            lse = lse.squeeze(-1).contiguous()  # (B, S_Q_GLOBAL, H_LOCAL)
        else:
            out = out_wait()  # type: torch.Tensor
            lse = None
        return (out, lse) if return_lse else out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_out: torch.Tensor,
        *args,
    ):
        raise NotImplementedError("USP attention backward is not implemented yet.")
