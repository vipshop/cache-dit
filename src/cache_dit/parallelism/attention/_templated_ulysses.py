import copy
import functools
from typing import Optional, Tuple, List

import torch
import torch.distributed as dist

try:
    from diffusers.models._modeling_parallel import ParallelConfig
    from diffusers.hooks.context_parallel import EquipartitionSharder
except ImportError:
    raise ImportError(
        "Context parallelism requires the 'diffusers>=0.36.dev0'."
        "Please install latest version of diffusers from source: \n"
        "pip3 install git+https://github.com/huggingface/diffusers.git"
    )
from ._distributed_primitives import (
    _get_rank_world_size,
    _gather_size_by_comm,
    # All to all for Ulysses Attention
    _all_to_all_single_o_async,
    _all_to_all_single_qkv_fp8_async,
    _all_to_all_single_o_fp8_async,
    _all_to_all_single_qkv_uneven_heads_async,
    _all_to_all_single_o_uneven_heads_async,
    # All to all for Ulysses Anything Attention
    _all_to_all_single_any_o_async,
    _all_to_all_single_any_qkv_async,
    _all_to_all_single_any_o_fp8_async,
    _all_to_all_single_any_qkv_fp8_async,
    _all_to_all_single_qkv_async,
    # Helper functions for preparing communication metadata
    _prepare_ulysses_comm_metadata,
)

from cache_dit.envs import ENV
from cache_dit.logger import init_logger

logger = init_logger(__name__)


__all__ = [
    "UnifiedTemplatedUlyssesAttention",
    "EquipartitionSharder",
    "enable_ulysses_anything",
    "is_ulysses_anything_enabled",
    "disable_ulysses_anything",
    "enable_ulysses_float8",
    "is_ulysses_float8_enabled",
    "disable_ulysses_float8",
    "is_ulysses_heads_no_padding",
]


class UnifiedTemplatedUlyssesAttention(torch.autograd.Function):
    """A unified wrapper for all Ulysses Attention variants in cache-dit."""

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
        if is_ulysses_anything_enabled():
            # Ulysses Anything Attention: Any sequence length and any head num supported.
            if is_ulysses_float8_enabled():
                return _TemplatedUlyssesAnythingAttentionFloat8.apply(
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
                return _TemplatedUlyssesAnythingAttention.apply(
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
            # Ulysses Attention: Support even sequence length and any head num.
            if is_ulysses_float8_enabled():
                return _TemplatedUlyssesAttentionFloat8.apply(
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
                if is_ulysses_heads_no_padding():
                    return _TemplatedUlyssesAttentionUnEvenHeads.apply(
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
                    return _TemplatedUlyssesAttention.apply(
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


# Re-implement Ulysses Attention with custom async all-to-all communication in cache-dit
# Use '_' prefix to avoid name conflict with diffusers' TemplatedUlyssesAttention.
class _TemplatedUlyssesAttention(torch.autograd.Function):
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
        ulysses_mesh = _parallel_config.context_parallel_config._ulysses_mesh
        group = ulysses_mesh.get_group()

        ctx.forward_op = forward_op
        ctx.backward_op = backward_op
        ctx._parallel_config = _parallel_config

        metadata = _prepare_ulysses_comm_metadata(query)
        query_wait = _all_to_all_single_qkv_async(query, group, **metadata)
        key_wait = _all_to_all_single_qkv_async(key, group, **metadata)
        value_wait = _all_to_all_single_qkv_async(value, group, **metadata)

        query = query_wait()  # type: torch.Tensor
        key = key_wait()  # type: torch.Tensor
        value = value_wait()  # type: torch.Tensor
        out = forward_op(
            ctx,
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
            return_lse,
            _save_ctx=False,
            _parallel_config=_parallel_config,
        )
        if return_lse:
            out, lse, *_ = out

        out_wait = _all_to_all_single_o_async(out, group, **metadata)

        if return_lse:
            lse = lse.unsqueeze(-1)  # (B, S_Q_GLOBAL, H_LOCAL, D=1)
            lse_wait = _all_to_all_single_o_async(lse, group, **metadata)
            out = out_wait()  # type: torch.Tensor
            lse = lse_wait()  # type: torch.Tensor
            lse = lse.squeeze(-1).contiguous()  # (B, S_Q_LOCAL, H_GLOBAL)
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
        raise NotImplementedError(
            "Backward pass for Ulysses Attention in cache-dit is not implemented yet."
        )


class _TemplatedUlyssesAttentionUnEvenHeads(torch.autograd.Function):
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
        ulysses_mesh = _parallel_config.context_parallel_config._ulysses_mesh
        group = ulysses_mesh.get_group()

        ctx.forward_op = forward_op
        ctx.backward_op = backward_op
        ctx._parallel_config = _parallel_config

        metadata = _prepare_ulysses_comm_metadata(query)
        # Async all to all for query, key, value with uneven heads communication
        query_wait = _all_to_all_single_qkv_uneven_heads_async(query, group, **metadata)
        key_wait = _all_to_all_single_qkv_uneven_heads_async(key, group, **metadata)
        value_wait = _all_to_all_single_qkv_uneven_heads_async(value, group, **metadata)

        query = query_wait()  # type: torch.Tensor
        key = key_wait()  # type: torch.Tensor
        value = value_wait()  # type: torch.Tensor

        out = forward_op(
            ctx,
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
            return_lse,
            _save_ctx=False,
            _parallel_config=_parallel_config,
        )
        if return_lse:
            out, lse, *_ = out

        # out: (B, S_Q_GLOBAL, H_LOCAL, D) -> (B, S_Q_LOCAL, H_GLOBAL, D)
        out_wait = _all_to_all_single_o_uneven_heads_async(out, group, **metadata)

        if return_lse:
            # NOTE: DON'T use float8 all_to_all for out and lse, as it may
            # cause more numerical instability.
            lse = lse.unsqueeze(-1)  # (B, S_Q_GLOBAL, H_LOCAL, D=1)
            lse_wait = _all_to_all_single_o_uneven_heads_async(lse, group, **metadata)
            out = out_wait()  # type: torch.Tensor
            lse = lse_wait()  # type: torch.Tensor
            lse = lse.squeeze(-1).contiguous()  # (B, S_Q_LOCAL, H_GLOBAL)
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
        raise NotImplementedError(
            "Backward pass for Ulysses Attention in cache-dit is not implemented yet."
        )


class _TemplatedUlyssesAttentionFloat8(torch.autograd.Function):
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
        ulysses_mesh = _parallel_config.context_parallel_config._ulysses_mesh
        group = ulysses_mesh.get_group()

        ctx.forward_op = forward_op
        ctx.backward_op = backward_op
        ctx._parallel_config = _parallel_config

        metadata = _prepare_ulysses_comm_metadata(query)
        # Use async all_to_all to overlap comm and quant/dequant computation
        # NOTE: Currently, we choose to keep K in FP16/BF16 format to keep higher
        # precision during softmax computation: Softmax(Q@K^T) which is sensitive to
        # numerical instability. So we only use float8 all_to_all for Q, V and O.
        # TODO: We should relax this design and support all QKV in float8 format while
        # the K-per-channel-smooth (e.g., in SageAttention) is used to improve numerical
        # stability. Using this smooth technique before All-to-All on K may introduce
        # extra AllReduce communication overhead.
        key_wait = _all_to_all_single_qkv_async(key, group, **metadata)
        query_wait = _all_to_all_single_qkv_fp8_async(query, group, **metadata)
        value_wait = _all_to_all_single_qkv_fp8_async(value, group, **metadata)

        query = query_wait()  # type: torch.Tensor
        value = value_wait()  # type: torch.Tensor
        key = key_wait()  # type: torch.Tensor

        out = forward_op(
            ctx,
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
            return_lse,
            _save_ctx=False,
            _parallel_config=_parallel_config,
        )
        if return_lse:
            out, lse, *_ = out

        out_wait = _all_to_all_single_o_fp8_async(out, group, **metadata)

        if return_lse:
            # NOTE: DON'T use float8 all_to_all for out and lse, as it may
            # cause more numerical instability.
            lse = lse.unsqueeze(-1)  # (B, S_Q_GLOBAL, H_LOCAL, D=1)
            lse_wait = _all_to_all_single_o_async(lse, group, **metadata)
            out = out_wait()  # type: torch.Tensor
            lse = lse_wait()  # type: torch.Tensor
            lse = lse.squeeze(-1).contiguous()  # (B, S_Q_LOCAL, H_GLOBAL)
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
        raise NotImplementedError(
            "Backward pass for Ulysses Attention Float8 in cache-dit is not implemented yet."
        )


class _TemplatedUlyssesAnythingAttention(torch.autograd.Function):

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
        **kwargs,
    ):
        ulysses_mesh = _parallel_config.context_parallel_config._ulysses_mesh
        group = ulysses_mesh.get_group()

        ctx.forward_op = forward_op
        ctx.backward_op = backward_op
        ctx._parallel_config = _parallel_config

        metadata = _prepare_ulysses_comm_metadata(query)
        query_wait = _all_to_all_single_any_qkv_async(query, group, **metadata)
        key_wait = _all_to_all_single_any_qkv_async(key, group, **metadata)
        value_wait = _all_to_all_single_any_qkv_async(value, group, **metadata)

        query = query_wait()  # type: torch.Tensor
        key = key_wait()  # type: torch.Tensor
        value = value_wait()  # type: torch.Tensor

        out = forward_op(
            ctx,
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
            return_lse,
            _save_ctx=False,
            _parallel_config=_parallel_config,
        )
        if return_lse:
            out, lse, *_ = out

        # out: (B, S_Q_GLOBAL, H_LOCAL, D) -> (B, S_Q_LOCAL, H_GLOBAL, D)
        out_wait = _all_to_all_single_any_o_async(out, group, **metadata)

        if return_lse:
            # lse: (B, S_Q_GLOBAL, H_LOCAL)
            lse = lse.unsqueeze(-1)  # (B, S_Q_GLOBAL, H_LOCAL, D=1)
            lse_wait = _all_to_all_single_any_o_async(lse, group, **metadata)
            out = out_wait()  # type: torch.Tensor
            lse = lse_wait()  # type: torch.Tensor
            lse = lse.squeeze(-1).contiguous()  # (B, S_Q_LOCAL, H_GLOBAL)
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
        raise NotImplementedError(
            "Backward pass for Ulysses Anything Attention in cache-dit is not implemented yet."
        )


class _TemplatedUlyssesAnythingAttentionFloat8(torch.autograd.Function):

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
        **kwargs,
    ):
        # TODO: Should we only use float8 all_to_all for VO not QK? The softmax in
        # QK may cause more numerical instability than P@V matrix multiplication.
        ulysses_mesh = _parallel_config.context_parallel_config._ulysses_mesh
        group = ulysses_mesh.get_group()

        ctx.forward_op = forward_op
        ctx.backward_op = backward_op
        ctx._parallel_config = _parallel_config

        metadata = _prepare_ulysses_comm_metadata(query)
        # Use async all_to_all to overlap comm and quant/dequant computation
        # NOTE: Currently, we choose to keep K in FP16/BF16 format to keep higher
        # precision during softmax computation: Softmax(Q@K^T) which is sensitive to
        # numerical instability. So we only use float8 all_to_all for Q, V and O.
        # TODO: We should relax this design and support all QKV in float8 format while
        # the K-per-channel-smooth (e.g., in SageAttention) is used to improve numerical
        # stability. Using this smooth technique before All-to-All on K may introduce
        # extra AllReduce communication overhead.
        key_wait = _all_to_all_single_any_qkv_async(key, group, **metadata)
        query_wait = _all_to_all_single_any_qkv_fp8_async(query, group, **metadata)
        value_wait = _all_to_all_single_any_qkv_fp8_async(value, group, **metadata)

        query = query_wait()  # type: torch.Tensor
        value = value_wait()  # type: torch.Tensor
        key = key_wait()  # type: torch.Tensor

        out = forward_op(
            ctx,
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
            return_lse,
            _save_ctx=False,
            _parallel_config=_parallel_config,
        )
        if return_lse:
            out, lse, *_ = out

        out_wait = _all_to_all_single_any_o_fp8_async(out, group, **metadata)

        if return_lse:
            # NOTE: DON'T use float8 all_to_all for out and lse, as it may
            # cause more numerical instability.
            lse = lse.unsqueeze(-1)  # (B, S_Q_GLOBAL, H_LOCAL, D=1)
            lse_wait = _all_to_all_single_any_o_async(lse, group, **metadata)
            out = out_wait()  # type: torch.Tensor
            lse = lse_wait()  # type: torch.Tensor
            lse = lse.squeeze(-1).contiguous()  # (B, S_Q_LOCAL, H_GLOBAL)
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
        raise NotImplementedError(
            "Backward pass for Ulysses Anything Attention Float8 in cache-dit is not implemented yet."
        )


@functools.lru_cache(maxsize=64)
def _fill_gather_shapes(
    shape: Tuple[int], gather_dims: Tuple[int], dim: int, world_size: int
) -> List[List[int]]:
    gather_shapes = []
    for i in range(world_size):
        # WARN: deepcopy to avoid modifying the original shape
        rank_shape = list(copy.deepcopy(shape))
        rank_shape[dim] = gather_dims[i]
        gather_shapes.append(rank_shape)
    return gather_shapes


@torch.compiler.allow_in_graph
def _all_gather_anything(  # noqa: F811
    tensor: torch.Tensor,
    dim: int,
    group: dist.device_mesh.DeviceMesh,
) -> torch.Tensor:
    _, world_size = _get_rank_world_size(group)
    tensor = tensor.contiguous()
    shape = tensor.shape
    rank_dim = shape[dim]
    gather_dims = _gather_size_by_comm(rank_dim, group)

    # NOTE: The `if` branch will introduce graph break for torch.compile,
    # so, we choose to disable the even split optimization for now.

    gather_shapes = _fill_gather_shapes(
        tuple(shape),
        tuple(gather_dims),
        dim,
        world_size,
    )

    gathered_tensors = [
        torch.empty(
            shape,
            device=tensor.device,
            dtype=tensor.dtype,
        )
        for shape in gather_shapes
    ]

    dist.all_gather(gathered_tensors, tensor, group=group)
    gathered_tensor = torch.cat(gathered_tensors, dim=dim)
    return gathered_tensor


# NOTE: dist.all_gather, Gathers tensors from the whole group in a list.
# Complex and uneven sized tensors are supported.
class AllGatherAnythingFunction(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        tensor: torch.Tensor,
        dim: int,
        group: dist.device_mesh.DeviceMesh,
    ):
        ctx.dim = dim
        ctx.group = group
        ctx.world_size = dist.get_world_size(group)
        ctx.rank = dist.get_rank(group)
        gathered_tensor = _all_gather_anything(tensor, dim, group)
        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        # NOTE: We use `tensor_split` instead of chunk, because the `chunk`
        # function may return fewer than the specified number of chunks!
        grad_splits = torch.tensor_split(grad_output, ctx.world_size, dim=ctx.dim)
        return grad_splits[ctx.rank], None, None


# NOTE: We use `tensor_split` instead of chunk, because the `chunk`
# function may return fewer than the specified number of chunks! For example,
# x = torch.tensor([1,2,3,4,5]), torch.chunk(x, 4) will return only 3 chunks:
# (tensor([1, 2]), tensor([3, 4]), tensor([5])). This behavior can lead to
# inconsistencies when sharding tensors across multiple devices. In contrast,
# tensor_split will always return the specified number of chunks, the last chunk
# may be smaller if the tensor size is not divisible by the number of chunks.
# For example, torch.tensor_split(x, 4) will return 4 chunks:
# (tensor([1, 2]), tensor([3]), tensor([4]), tensor([5])).
@classmethod
@functools.wraps(EquipartitionSharder.shard)
def shard_anything(
    cls: EquipartitionSharder,
    tensor: torch.Tensor,
    dim: int,
    mesh: dist.device_mesh.DeviceMesh,
    **kwargs,
) -> torch.Tensor:
    assert tensor.size()[dim] >= mesh.size(), (
        f"Cannot shard tensor of size {tensor.size()} along dim {dim} "
        f"across mesh of size {mesh.size()}."
    )
    return tensor.tensor_split(mesh.size(), dim=dim)[dist.get_rank(mesh.get_group())]


# NOTE: We use AllGatherAnythingFunction to support gathering
# tensors with complex and uneven sizes across all ranks. It handles the
# case where the tensor size (the seq_len of hidden_states) along the
# specified dimension is not divisible by the number of ranks in the mesh.
@classmethod
@functools.wraps(EquipartitionSharder.unshard)
def unshard_anything(
    cls,
    tensor: torch.Tensor,
    dim: int,
    mesh: torch.distributed.device_mesh.DeviceMesh,
    **kwargs,
) -> torch.Tensor:
    tensor = tensor.contiguous()
    tensor = AllGatherAnythingFunction.apply(tensor, dim, mesh.get_group())
    return tensor


# Environment variable flags for Ulysses Attention variants in cache-dit.
def is_ulysses_heads_no_padding() -> bool:
    return ENV.CACHE_DIT_UNEVEN_HEADS_COMM_NO_PAD


def enable_ulysses_anything(**kwargs):
    try:
        if ENV.CACHE_DIT_ENABELD_ULYSSES_ANYTHING:
            # function for TemplatedUlyssesAnythingAttention.
            if EquipartitionSharder.shard != shard_anything:
                EquipartitionSharder.shard = shard_anything
                EquipartitionSharder.unshard = unshard_anything
                logger.warning(
                    "Ulysses Anything Attention is already enabled in cache-dit. "
                    "but EquipartitionSharder.shard/unshard is not set correctly, "
                    "resetting it to the correct shard/unshard_anything function."
                )
            return

        ENV.CACHE_DIT_ENABELD_ULYSSES_ANYTHING = True

        logger.warning(
            "Ulysses Anything Attention is enabled in cache-dit. "
            "Please note that this is an experimental feature and "
            "may not be fully tested."
        )

        # Ensure the EquipartitionSharder uses our modified shard_anything
        # function for TemplatedUlyssesAnythingAttention.
        if EquipartitionSharder.shard != shard_anything:
            EquipartitionSharder.shard = shard_anything
            EquipartitionSharder.unshard = unshard_anything
            logger.info(
                "EquipartitionSharder.shard/unshard is set to shard/unshard_anything function "
                "for Ulysses Anything Attention."
            )
    except Exception as e:
        ENV.CACHE_DIT_ENABELD_ULYSSES_ANYTHING = False
        logger.error(f"Failed to enable Ulysses Anything Attention in cache-dit due to error: {e}")
        pass


def is_ulysses_anything_enabled(**kwargs) -> bool:
    return ENV.CACHE_DIT_ENABELD_ULYSSES_ANYTHING


def disable_ulysses_anything(**kwargs):
    ENV.CACHE_DIT_ENABELD_ULYSSES_ANYTHING = False
    logger.info("Ulysses Anything Attention is manually disabled in cache-dit.")


# Float8 flags for Ulysses/Ulysses Anything Attention
def _enable_ulysses_anything_float8(**kwargs):
    try:
        if ENV.CACHE_DIT_ENABELD_ULYSSES_ANYTHING_FLOAT8:
            # function for TemplatedUlyssesAnythingAttention.
            if EquipartitionSharder.shard != shard_anything:
                EquipartitionSharder.shard = shard_anything
                EquipartitionSharder.unshard = unshard_anything
                logger.warning(
                    "Ulysses Anything Attention Float8 is already enabled in cache-dit. "
                    "but EquipartitionSharder.shard/unshard is not set correctly, "
                    "resetting it to the correct shard/unshard_anything function."
                )
            return

        ENV.CACHE_DIT_ENABELD_ULYSSES_ANYTHING_FLOAT8 = True

        logger.warning(
            "Ulysses Anything Attention Float8 is enabled in cache-dit. "
            "Please note that this is an experimental feature and "
            "may not be fully tested."
        )

        # Ensure the EquipartitionSharder uses our modified shard_anything
        # function for TemplatedUlyssesAnythingAttention.
        if EquipartitionSharder.shard != shard_anything:
            EquipartitionSharder.shard = shard_anything
            EquipartitionSharder.unshard = unshard_anything
            logger.info(
                "EquipartitionSharder.shard/unshard is set to shard/unshard_anything function "
                "for Ulysses Anything Attention Float8."
            )
    except Exception as e:
        ENV.CACHE_DIT_ENABELD_ULYSSES_ANYTHING_FLOAT8 = False
        logger.error(
            f"Failed to enable Ulysses Anything Attention Float8 in cache-dit due to error: {e}"
        )
        pass


def _is_ulysses_anything_float8_enabled(**kwargs) -> bool:
    return ENV.CACHE_DIT_ENABELD_ULYSSES_ANYTHING_FLOAT8


def _disable_ulysses_anything_float8(**kwargs) -> bool:
    ENV.CACHE_DIT_ENABELD_ULYSSES_ANYTHING_FLOAT8 = False
    logger.info("Ulysses Anything Attention Float8 is manually disabled in cache-dit.")


def enable_ulysses_float8(**kwargs):

    # Check if Ulysses Anything Attention is already enabled
    if is_ulysses_anything_enabled():
        _enable_ulysses_anything_float8()
        return

    ENV.CACHE_DIT_ENABELD_ULYSSES_FLOAT8 = True
    logger.warning(
        "Ulysses Attention Float8 is enabled in cache-dit. "
        "Please note that this is an experimental feature and "
        "may not be fully tested."
    )


def is_ulysses_float8_enabled(**kwargs) -> bool:
    return ENV.CACHE_DIT_ENABELD_ULYSSES_FLOAT8 or _is_ulysses_anything_float8_enabled()


def disable_ulysses_float8(**kwargs) -> bool:
    ENV.CACHE_DIT_ENABELD_ULYSSES_FLOAT8 = False
    logger.info("Ulysses Attention Float8 is manually disabled in cache-dit.")
    if is_ulysses_anything_enabled():
        _disable_ulysses_anything_float8()
