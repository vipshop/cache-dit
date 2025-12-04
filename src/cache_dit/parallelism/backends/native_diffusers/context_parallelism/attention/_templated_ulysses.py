import os
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
from cache_dit.logger import init_logger
from ._distributed_primitives import (
    _get_rank_world_size,
    _gather_size_by_comm,
    _all_to_all_single_any_o,
    _all_to_all_single_any_qkv,
    _all_to_all_single_any_o_fp8,
    _all_to_all_single_any_qkv_fp8,
    _all_to_all_single_fp8,
    _all_to_all_single,
)

logger = init_logger(__name__)

__all__ = [
    "TemplatedUlyssesAnythingAttention",
    "TemplatedUlyssesAnythingAttentionFloat8",
    "TemplatedUlyssesAttentionFloat8",
    "EquipartitionSharder",
    "enable_ulysses_anything",
    "is_ulysses_anything_enabled",
    "disable_ulysses_anything",
    "enable_ulysses_anything_float8",
    "is_ulysses_anything_float8_enabled",
    "disable_ulysses_anything_float8",
    "enable_ulysses_float8",
    "is_ulysses_float8_enabled",
    "disable_ulysses_float8",
]


class TemplatedUlyssesAnythingAttention(torch.autograd.Function):

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
        world_size = _parallel_config.context_parallel_config.ulysses_degree
        group = ulysses_mesh.get_group()

        ctx.forward_op = forward_op
        ctx.backward_op = backward_op
        ctx._parallel_config = _parallel_config

        B, S_Q_LOCAL, H, D = query.shape
        _, S_KV_LOCAL, _, _ = key.shape
        H_LOCAL = H // world_size
        # (world_size, S_LOCAL, B, H_LOCAL, D)
        query = (
            query.reshape(B, S_Q_LOCAL, world_size, H_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
        )
        key = key.reshape(B, S_KV_LOCAL, world_size, H_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
        value = (
            value.reshape(B, S_KV_LOCAL, world_size, H_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
        )
        query, key, value = (_all_to_all_single_any_qkv(x, group) for x in (query, key, value))
        # (S_GLOBAL, B, H_LOCAL, D) -> (B, S_GLOBAL, H_LOCAL, D)
        query, key, value = (x.permute(1, 0, 2, 3).contiguous() for x in (query, key, value))

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
            _save_ctx=True,
            _parallel_config=_parallel_config,
        )
        if return_lse:
            out, lse, *_ = out

        # out: (B, S_Q_GLOBAL, H_LOCAL, D) -> (B, S_Q_LOCAL, H_GLOBAL, D)
        out = _all_to_all_single_any_o(out, group).contiguous()

        if return_lse:
            # lse: (B, S_Q_GLOBAL, H_LOCAL)
            lse = lse.unsqueeze(-1)  # (B, S_Q_GLOBAL, H_LOCAL, D=1)
            lse = (
                _all_to_all_single_any_o(lse, group).squeeze(-1).contiguous()
            )  # (B, S_Q_LOCAL, H_GLOBAL)
        else:
            lse = None

        return (out, lse) if return_lse else out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_out: torch.Tensor,
        *args,
    ):
        raise NotImplementedError(
            "Backward pass for Ulysses Anything Attention is not implemented yet."
        )


class TemplatedUlyssesAnythingAttentionFloat8(torch.autograd.Function):

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
        world_size = _parallel_config.context_parallel_config.ulysses_degree
        group = ulysses_mesh.get_group()

        ctx.forward_op = forward_op
        ctx.backward_op = backward_op
        ctx._parallel_config = _parallel_config

        B, S_Q_LOCAL, H, D = query.shape
        _, S_KV_LOCAL, _, _ = key.shape
        H_LOCAL = H // world_size
        # (world_size, S_LOCAL, B, H_LOCAL, D)
        query = (
            query.reshape(B, S_Q_LOCAL, world_size, H_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
        )
        key = key.reshape(B, S_KV_LOCAL, world_size, H_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
        value = (
            value.reshape(B, S_KV_LOCAL, world_size, H_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
        )
        query, key, value = (_all_to_all_single_any_qkv_fp8(x, group) for x in (query, key, value))
        # (S_GLOBAL, B, H_LOCAL, D) -> (B, S_GLOBAL, H_LOCAL, D)
        query, key, value = (x.permute(1, 0, 2, 3).contiguous() for x in (query, key, value))

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
            _save_ctx=True,
            _parallel_config=_parallel_config,
        )
        if return_lse:
            out, lse, *_ = out

        # out: (B, S_Q_GLOBAL, H_LOCAL, D) -> (B, S_Q_LOCAL, H_GLOBAL, D)
        out = _all_to_all_single_any_o_fp8(out, group).contiguous()

        if return_lse:
            # lse: (B, S_Q_GLOBAL, H_LOCAL)
            lse = lse.unsqueeze(-1)  # (B, S_Q_GLOBAL, H_LOCAL, D=1)
            # NOTE: DON'T use float8 all_to_all for lse, as it may cause numerical instability
            lse = (
                _all_to_all_single_any_o(lse, group).squeeze(-1).contiguous()
            )  # (B, S_Q_LOCAL, H_GLOBAL)
        else:
            lse = None

        return (out, lse) if return_lse else out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_out: torch.Tensor,
        *args,
    ):
        raise NotImplementedError(
            "Backward pass for Ulysses Anything Attention Float8 is not implemented yet."
        )


class TemplatedUlyssesAttentionFloat8(torch.autograd.Function):
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
        # TODO: Should we only use float8 all_to_all for VO not QK? The softmax in
        # QK may cause more numerical instability than P@V matrix multiplication.
        ulysses_mesh = _parallel_config.context_parallel_config._ulysses_mesh
        world_size = _parallel_config.context_parallel_config.ulysses_degree
        group = ulysses_mesh.get_group()

        ctx.forward_op = forward_op
        ctx.backward_op = backward_op
        ctx._parallel_config = _parallel_config

        B, S_Q_LOCAL, H, D = query.shape
        _, S_KV_LOCAL, _, _ = key.shape
        H_LOCAL = H // world_size
        query = (
            query.reshape(B, S_Q_LOCAL, world_size, H_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
        )
        key = key.reshape(B, S_KV_LOCAL, world_size, H_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
        value = (
            value.reshape(B, S_KV_LOCAL, world_size, H_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
        )
        query, key, value = (_all_to_all_single_fp8(x, group) for x in (query, key, value))
        query, key, value = (
            x.flatten(0, 1).permute(1, 0, 2, 3).contiguous() for x in (query, key, value)
        )

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
            _save_ctx=True,
            _parallel_config=_parallel_config,
        )
        if return_lse:
            out, lse, *_ = out

        out = out.reshape(B, world_size, S_Q_LOCAL, H_LOCAL, D).permute(1, 3, 0, 2, 4).contiguous()
        out = _all_to_all_single_fp8(out, group)
        out = out.flatten(0, 1).permute(1, 2, 0, 3).contiguous()

        if return_lse:
            lse = lse.reshape(B, world_size, S_Q_LOCAL, H_LOCAL).permute(1, 3, 0, 2).contiguous()
            # NOTE: DON'T use float8 all_to_all for lse, as it may cause numerical instability
            lse = _all_to_all_single(lse, group)
            lse = lse.flatten(0, 1).permute(1, 2, 0).contiguous()
        else:
            lse = None

        return (out, lse) if return_lse else out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_out: torch.Tensor,
        *args,
    ):
        raise NotImplementedError(
            "Backward pass for Ulysses Attention Float8 is not implemented yet."
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


_CACHE_DIT_ENABELD_ULYSSES_ANYTHING = (
    os.environ.get("CACHE_DIT_ENABELD_ULYSSES_ANYTHING", "0") == "1"
)
_CACHE_DIT_ENABELD_ULYSSES_ANYTHING_FLOAT8 = (
    os.environ.get("CACHE_DIT_ENABELD_ULYSSES_ANYTHING_FLOAT8", "0") == "1"
)
_CACHE_DIT_ENABELD_ULYSSES_FLOAT8 = os.environ.get("CACHE_DIT_ENABELD_ULYSSES_FLOAT8", "0") == "1"


def enable_ulysses_anything(**kwargs):
    global _CACHE_DIT_ENABELD_ULYSSES_ANYTHING
    try:
        if _CACHE_DIT_ENABELD_ULYSSES_ANYTHING:
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

        _CACHE_DIT_ENABELD_ULYSSES_ANYTHING = True

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
        _CACHE_DIT_ENABELD_ULYSSES_ANYTHING = False
        logger.error(f"Failed to enable Ulysses Anything Attention in cache-dit due to error: {e}")
        pass


def is_ulysses_anything_enabled(**kwargs) -> bool:
    global _CACHE_DIT_ENABELD_ULYSSES_ANYTHING
    return _CACHE_DIT_ENABELD_ULYSSES_ANYTHING


def disable_ulysses_anything(**kwargs):
    global _CACHE_DIT_ENABELD_ULYSSES_ANYTHING
    _CACHE_DIT_ENABELD_ULYSSES_ANYTHING = False
    logger.info("Ulysses Anything Attention is manually disabled in cache-dit.")


def enable_ulysses_anything_float8(**kwargs):
    global _CACHE_DIT_ENABELD_ULYSSES_ANYTHING_FLOAT8
    try:
        if _CACHE_DIT_ENABELD_ULYSSES_ANYTHING_FLOAT8:
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

        _CACHE_DIT_ENABELD_ULYSSES_ANYTHING_FLOAT8 = True

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
        _CACHE_DIT_ENABELD_ULYSSES_ANYTHING_FLOAT8 = False
        logger.error(
            f"Failed to enable Ulysses Anything Attention Float8 in cache-dit due to error: {e}"
        )
        pass


def is_ulysses_anything_float8_enabled(**kwargs) -> bool:
    global _CACHE_DIT_ENABELD_ULYSSES_ANYTHING_FLOAT8
    return _CACHE_DIT_ENABELD_ULYSSES_ANYTHING_FLOAT8


def disable_ulysses_anything_float8(**kwargs) -> bool:
    global _CACHE_DIT_ENABELD_ULYSSES_ANYTHING_FLOAT8
    _CACHE_DIT_ENABELD_ULYSSES_ANYTHING_FLOAT8 = False
    logger.info("Ulysses Anything Attention Float8 is manually disabled in cache-dit.")


def enable_ulysses_float8(**kwargs):
    global _CACHE_DIT_ENABELD_ULYSSES_FLOAT8
    _CACHE_DIT_ENABELD_ULYSSES_FLOAT8 = True
    logger.warning(
        "Ulysses Attention Float8 is enabled in cache-dit. "
        "Please note that this is an experimental feature and "
        "may not be fully tested."
    )


def is_ulysses_float8_enabled(**kwargs) -> bool:
    global _CACHE_DIT_ENABELD_ULYSSES_FLOAT8
    return _CACHE_DIT_ENABELD_ULYSSES_FLOAT8


def disable_ulysses_float8(**kwargs) -> bool:
    global _CACHE_DIT_ENABELD_ULYSSES_FLOAT8
    _CACHE_DIT_ENABELD_ULYSSES_FLOAT8 = False
    logger.info("Ulysses Attention Float8 is manually disabled in cache-dit.")
