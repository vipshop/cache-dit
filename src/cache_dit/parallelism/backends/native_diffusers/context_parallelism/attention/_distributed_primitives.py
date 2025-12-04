import functools
from typing import Tuple, List, Callable

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as fc

from cache_dit.logger import init_logger
from cache_dit.kernels import per_token_quant_fp8, per_token_dequant_fp8

logger = init_logger(__name__)

# Some helper distributed primitive functions for context parallel attention.


# Reference:
# - https://github.com/pytorch/pytorch/blob/f58a680d09e13658a52c6ba05c63c15759846bcc/torch/distributed/_functional_collectives.py#L827
# - https://github.com/pytorch/pytorch/blob/f58a680d09e13658a52c6ba05c63c15759846bcc/torch/distributed/_functional_collectives.py#L246
# - https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_dispatch.py#L1012
# For fullgraph=True tracing compatibility (since FakeTensor does not have a `wait` method):
def _wait_tensor(tensor) -> torch.Tensor:
    if isinstance(tensor, fc.AsyncCollectiveTensor):
        tensor = tensor.wait()

    return tensor


def _all_to_all_single(x: torch.Tensor, group) -> torch.Tensor:
    shape = x.shape
    x = x.flatten()
    x = fc.all_to_all_single(x, None, None, group)
    x = x.reshape(shape)
    x = _wait_tensor(x)
    return x


def _get_rank_world_size(
    group: dist.ProcessGroup,
) -> Tuple[int, int]:
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    return rank, world_size


@functools.lru_cache(maxsize=128)
def _gather_size_by_comm(S_LOCAL: int, group: dist.ProcessGroup) -> List[int]:
    world_size = dist.get_world_size(group=group)
    # HACK: Use Gloo backend for all_gather to avoid H2D and D2H overhead
    comm_backends = str(dist.get_backend(group=group))
    # NOTE: e.g., dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    gather_device = "cpu" if "cpu" in comm_backends else torch.device("cuda")
    gathered_sizes = [
        torch.empty((1,), device=gather_device, dtype=torch.int64) for _ in range(world_size)
    ]
    dist.all_gather(
        gathered_sizes,
        torch.tensor([S_LOCAL], device=gather_device, dtype=torch.int64),
        group=group,
    )

    gathered_sizes = [s[0].item() for s in gathered_sizes]
    # NOTE: DON'T use tolist here due to graph break - Explanation:
    # Backend compiler `inductor` failed with aten._local_scalar_dense.default
    return gathered_sizes


# NOTE: Temporary workaround for torch.compile issue with float8 tensor view.
# Issue: https://github.com/vipshop/cache-dit/issues/513
@torch.compiler.disable
def _tensor_bitcast(x: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    assert x.nbytes % dtype.itemsize == 0, f"x.nbytes must be divisible by {dtype.itemsize}"
    x = x.contiguous()
    return x.view(dtype)


@torch.compiler.allow_in_graph
def _all_to_all_single_any_qkv(
    x: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    shape = x.shape  # (world_size, S_LOCAL, B, H_LOCAL, D)
    (world_size, S_LOCAL, B, H_LOCAL, D) = shape
    input_split_sizes = [S_LOCAL] * world_size
    # S_LOCAL maybe not equal for all ranks in dynamic shape case,
    # since we don't know the actual shape before this timing, thus,
    # we have to use all gather to collect the S_LOCAL first.
    output_split_sizes = _gather_size_by_comm(S_LOCAL, group)
    # NOTE: The `if` branch will introduce graph break for torch.compile,
    # so, we choose to disable the even split optimization implementation
    # _all_to_all_single for now.
    x = x.flatten(0, 1)  # (world_size * S_LOCAL, B, H_LOCAL, D)
    x = fc.all_to_all_single(x, output_split_sizes, input_split_sizes, group)
    x = _wait_tensor(x)  # (S_GLOBAL, B, H_LOCAL, D)
    return x


@torch.compiler.allow_in_graph
def _all_to_all_single_any_o(
    out: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    rank, world_size = _get_rank_world_size(group)
    shape = out.shape  # (B, S_GLOBAL, H_LOCAL, D)
    (B, S_GLOBAL, H_LOCAL, D) = shape

    # NOTE: The `if` branch will introduce graph break for torch.compile,
    # so, we choose to disable the even split optimization implementation
    # _all_to_all_single for now.
    out = out.flatten(0, 1).contiguous()  # (B*S_GLOBAL, H_LOCAL, D)
    # NOTE: May use tensor_split here to ensure the same split policy
    # that we have used in the EquipartitionSharder sharding strategy. Please
    # note that the 'tensor_split' Splits a tensor into multiple sub-tensors,
    # all of which are views of input, thus may not introduce extra IO access.
    input_split_sizes = [o.shape[0] for o in torch.tensor_split(out, world_size, dim=0)]
    # input_split: e.g, B*S_GLOBAL=1*9 input splits across ranks [[5,4], [5,4],..]
    # output_split: e.g, B*S_GLOBAL=1*9 output splits across ranks [[5,5], [4,4],..]
    output_split_sizes = [input_split_sizes[rank]] * world_size
    out = fc.all_to_all_single(out, output_split_sizes, input_split_sizes, group)
    out = _wait_tensor(out)  # (S_LOCAL*world_size, H_LOCAL, D)
    # NOTE: We can not simply reshape here, because the collective tensors
    # are stacked at dim=0(SeqLen), we need to first split them and then concat at
    # dim=1(Head), otherwise the result will be incorrect due to the linear layout
    # of the tensor in memory.
    H_GLOBAL = H_LOCAL * world_size
    S_LOCAL = out.shape[0] // world_size
    # TODO: How to avoid extra memory IO access here?
    out = torch.cat(out.tensor_split(world_size, dim=0), dim=1)  # (B*S_LOCAL, H_GLOBAL, D)
    out = out.reshape(B, S_LOCAL, H_GLOBAL, D)  # (B, S_LOCAL, H_GLOBAL, D)
    return out


def _all_to_all_single_fp8(x: torch.Tensor, group) -> torch.Tensor:
    shape = x.shape
    x_fp8_with_scale = per_token_quant_fp8(x)  # type: torch.Tensor
    shape_with_scale = x_fp8_with_scale.shape  # (world_size, S_LOCAL, B, H_LOCAL, D + itemsize)
    x_fp8_with_scale = x_fp8_with_scale.flatten()
    x_fp8_with_scale = fc.all_to_all_single(x_fp8_with_scale, None, None, group)
    x_fp8_with_scale = _wait_tensor(x_fp8_with_scale)
    x_fp8_with_scale = x_fp8_with_scale.reshape(shape_with_scale)
    x = per_token_dequant_fp8(x_fp8_with_scale)
    x = x.reshape(shape)
    return x


@torch.compiler.allow_in_graph
def _all_to_all_single_any_qkv_fp8(
    x: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    shape = x.shape  # (world_size, S_LOCAL, B, H_LOCAL, D)
    (world_size, S_LOCAL, B, H_LOCAL, D) = shape
    input_split_sizes = [S_LOCAL] * world_size
    # S_LOCAL maybe not equal for all ranks in dynamic shape case,
    # since we don't know the actual shape before this timing, thus,
    # we have to use all gather to collect the S_LOCAL first.
    output_split_sizes = _gather_size_by_comm(S_LOCAL, group)
    # NOTE: The `if` branch will introduce graph break for torch.compile,
    # so, we choose to disable the even split optimization implementation
    # _all_to_all_single for now.
    x_fp8_with_scale = per_token_quant_fp8(x)
    x_fp8_with_scale = x_fp8_with_scale.flatten(0, 1)
    x_fp8_with_scale = fc.all_to_all_single(
        x_fp8_with_scale, output_split_sizes, input_split_sizes, group
    )
    x_fp8_with_scale = _wait_tensor(x_fp8_with_scale)
    x = per_token_dequant_fp8(x_fp8_with_scale)
    return x


@torch.compiler.allow_in_graph
def _all_to_all_single_any_o_fp8(
    out: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    rank, world_size = _get_rank_world_size(group)
    shape = out.shape  # (B, S_GLOBAL, H_LOCAL, D)
    (B, S_GLOBAL, H_LOCAL, D) = shape
    out_fp8_with_scale = per_token_quant_fp8(out)

    # NOTE: The `if` branch will introduce graph break for torch.compile,
    # so, we choose to disable the even split optimization implementation
    # _all_to_all_single for now.
    out_fp8_with_scale = out_fp8_with_scale.flatten(0, 1).contiguous()  # (B*S_GLOBAL, H_LOCAL, D)
    # NOTE: May use tensor_split here to ensure the same split policy
    # that we have used in the EquipartitionSharder sharding strategy. Please
    # note that the 'tensor_split' Splits a tensor into multiple sub-tensors,
    # all of which are views of input, thus may not introduce extra IO access.
    input_split_sizes = [
        o.shape[0] for o in torch.tensor_split(out_fp8_with_scale, world_size, dim=0)
    ]
    # input_split: e.g, B*S_GLOBAL=1*9 input splits across ranks [[5,4], [5,4],..]
    # output_split: e.g, B*S_GLOBAL=1*9 output splits across ranks [[5,5], [4,4],..]
    output_split_sizes = [input_split_sizes[rank]] * world_size
    out_fp8_with_scale = fc.all_to_all_single(
        out_fp8_with_scale, output_split_sizes, input_split_sizes, group
    )
    out_fp8_with_scale = _wait_tensor(out_fp8_with_scale)  # (S_LOCAL*world_size, H_LOCAL, D)
    # NOTE: We can not simply reshape here, because the collective tensors
    # are stacked at dim=0(SeqLen), we need to first split them and then concat at
    # dim=1(Head), otherwise the result will be incorrect due to the linear layout
    # of the tensor in memory.
    H_GLOBAL = H_LOCAL * world_size
    S_LOCAL = out_fp8_with_scale.shape[0] // world_size
    # TODO: How to avoid extra memory IO access here?
    out_fp8_with_scale = torch.cat(
        out_fp8_with_scale.tensor_split(world_size, dim=0), dim=1
    )  # (B*S_LOCAL, H_GLOBAL, D)
    out = per_token_dequant_fp8(out_fp8_with_scale)
    out = out.reshape(B, S_LOCAL, H_GLOBAL, D)  # (B, S_LOCAL, H_GLOBAL, D)
    return out


@torch.compiler.allow_in_graph
def _gather_split_any_o(  # noqa: F811
    out: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    # NOTE: This is an alternative implementation of _all_to_all_single
    # for any o. It use all_gather and split, which may be less efficient.
    rank, world_size = _get_rank_world_size(group)
    # (B, S_GLOBAL, H_LOCAL, D)
    # all gather to get (B, S_GLOBAL, H_GLOBAL, D) at H_GLOBAL dim
    out_gathered = [torch.empty_like(out) for _ in range(world_size)]
    dist.all_gather(out_gathered, out, group=group)
    out_gathered = torch.cat(out_gathered, dim=2)
    # (B, S_GLOBAL, H_GLOBAL, D) -> (B, S_Q_LOCAL, H_GLOBAL, D)
    out = out_gathered.tensor_split(world_size, dim=1)[rank]
    return out


# Asynchronous all to all variants. Currently only non-any_qkvo version is supported.
# TODO: Implement any_qkvo asynchronous all to all variants.
def _all_to_all_single_qkv_async(
    x: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    _, world_size = _get_rank_world_size(group)
    B, S_LOCAL, H, D = x.shape
    H_LOCAL = H // world_size
    x = x.reshape(B, S_LOCAL, world_size, H_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
    _shape = x.shape  # (world_size, S_LOCAL, B, H_LOCAL, D)

    x = x.flatten()
    x = fc.all_to_all_single(x, None, None, group)

    def wait() -> torch.Tensor:
        nonlocal x
        out = _wait_tensor(x)
        # (world_size, S_LOCAL, B, H_LOCAL, D)
        # -> (S_GLOBAL, B, H_LOCAL, D)
        # -> (B, S_GLOBAL, H_LOCAL, D)
        out = out.reshape(_shape).flatten(0, 1).permute(1, 0, 2, 3).contiguous()
        return out

    return wait


def _all_to_all_single_o_async(
    x: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    _, world_size = _get_rank_world_size(group)
    B, S_GLOBAL, H_LOCAL, D = x.shape
    S_LOCAL = S_GLOBAL // world_size
    # (B, S_GLOBAL, H_LOCAL, D) -> (world_size, H_LOCAL, B, S_LOCAL, D)
    x = x.reshape(B, world_size, S_LOCAL, H_LOCAL, D).permute(1, 3, 0, 2, 4).contiguous()
    _shape = x.shape  # (world_size, H_LOCAL, B, S_LOCAL, D)

    x = x.flatten()
    x = fc.all_to_all_single(x, None, None, group)

    def wait() -> torch.Tensor:
        nonlocal x
        out = _wait_tensor(x)
        # (world_size, H_LOCAL, B, S_LOCAL, D)
        # -> (H_GLOBAL, B, S_LOCAL, D)
        # -> (B, H_GLOBAL, S_LOCAL, D)
        out = out.reshape(_shape).flatten(0, 1).permute(1, 0, 2, 3).contiguous()
        return out

    return wait


def _all_to_all_single_qkv_fp8_async(
    x: torch.Tensor,
    group: dist.ProcessGroup,
) -> Callable[[], torch.Tensor]:
    _, world_size = _get_rank_world_size(group)
    B, S_LOCAL, H, D = x.shape
    H_LOCAL = H // world_size
    x = x.reshape(B, S_LOCAL, world_size, H_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
    _shape = x.shape  # (world_size, S_LOCAL, B, H_LOCAL, D)

    x_fp8_with_scale = per_token_quant_fp8(x)  # type: torch.Tensor
    shape_with_scale = x_fp8_with_scale.shape  # (world_size, S_LOCAL, B, H_LOCAL, D + itemsize)
    x_fp8_with_scale = x_fp8_with_scale.flatten()
    x_fp8_with_scale = fc.all_to_all_single(x_fp8_with_scale, None, None, group)

    def wait() -> torch.Tensor:
        nonlocal x_fp8_with_scale
        x_fp8_with_scale = _wait_tensor(x_fp8_with_scale)
        x_fp8_with_scale = x_fp8_with_scale.reshape(shape_with_scale)
        out = per_token_dequant_fp8(x_fp8_with_scale)
        # (world_size, S_LOCAL, B, H_LOCAL, D)
        # -> (S_GLOBAL, B, H_LOCAL, D)
        # -> (B, S_GLOBAL, H_LOCAL, D)
        out = out.reshape(_shape).flatten(0, 1).permute(1, 0, 2, 3).contiguous()
        return out

    return wait


def _all_to_all_single_o_fp8_async(
    x: torch.Tensor,
    group: dist.ProcessGroup,
) -> Callable[[], torch.Tensor]:
    _, world_size = _get_rank_world_size(group)
    B, S_GLOBAL, H_LOCAL, D = x.shape
    S_LOCAL = S_GLOBAL // world_size
    # (B, S_GLOBAL, H_LOCAL, D) -> (world_size, H_LOCAL, B, S_LOCAL, D)
    x = x.reshape(B, world_size, S_LOCAL, H_LOCAL, D).permute(1, 3, 0, 2, 4).contiguous()
    _shape = x.shape  # (world_size, H_LOCAL, B, S_LOCAL, D)

    x_fp8_with_scale = per_token_quant_fp8(x)
    shape_with_scale = x_fp8_with_scale.shape  # (world_size, H_LOCAL, B, S_LOCAL, D + itemsize)
    x_fp8_with_scale = x_fp8_with_scale.flatten()
    x_fp8_with_scale = fc.all_to_all_single(x_fp8_with_scale, None, None, group)

    def wait() -> torch.Tensor:
        nonlocal x_fp8_with_scale
        x_fp8_with_scale = _wait_tensor(x_fp8_with_scale)
        x_fp8_with_scale = x_fp8_with_scale.reshape(shape_with_scale)
        out = per_token_dequant_fp8(x_fp8_with_scale)
        # (world_size, H_LOCAL, B, S_LOCAL, D)
        # -> (H_GLOBAL, B, S_LOCAL, D)
        # -> (B, H_GLOBAL, S_LOCAL, D)
        out = out.reshape(_shape).flatten(0, 1).permute(1, 0, 2, 3).contiguous()
        return out

    return wait


# Unified functions to select proper all to all implementations according to
# Ulysses Float8 or other settings. Mainly used in Async Ulysses Attention.
# TODO: Refactor basic any_qkvo and non-any_qkvo all2all functions to have
# the same output shape, thus make the unified functions more general and clean.


def _unified_all_to_all_qkv_async_fn() -> Callable[[], torch.Tensor]:
    # TODO: Add any_qkvo async all2all support.
    from ._templated_ulysses import is_ulysses_float8_enabled

    if is_ulysses_float8_enabled():
        return _all_to_all_single_qkv_fp8_async
    else:
        return _all_to_all_single_qkv_async


def _unified_all_to_all_o_async_fn() -> Callable[[], torch.Tensor]:
    # TODO: Add any_qkvo all2all support.
    from ._templated_ulysses import is_ulysses_float8_enabled

    if is_ulysses_float8_enabled():
        return _all_to_all_single_o_fp8_async
    else:
        return _all_to_all_single_o_async
