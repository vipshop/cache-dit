import functools
from typing import Tuple, List, Callable

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as fc
import torch.nn.functional as F

from cache_dit.logger import init_logger
from cache_dit.kernels import per_token_quant_fp8, per_token_dequant_fp8

logger = init_logger(__name__)

# Some helper distributed primitive functions for context parallel attention.

# NOTE: We should always use the asynchronous all to all variants to keep the uified input/output shape
# for any_qkvo and non-any_qkvo cases, otherwise, the input/output shape will be different, which makes
# the unified function implementation complex and ugly.


# Reference:
# - https://github.com/pytorch/pytorch/blob/f58a680d09e13658a52c6ba05c63c15759846bcc/torch/distributed/_functional_collectives.py#L827
# - https://github.com/pytorch/pytorch/blob/f58a680d09e13658a52c6ba05c63c15759846bcc/torch/distributed/_functional_collectives.py#L246
# - https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_dispatch.py#L1012
# For fullgraph=True tracing compatibility (since FakeTensor does not have a `wait` method):
def _wait_tensor(tensor) -> torch.Tensor:
    if isinstance(tensor, fc.AsyncCollectiveTensor):
        tensor = tensor.wait()

    return tensor


def _get_rank_world_size(
    group: dist.ProcessGroup,
) -> Tuple[int, int]:
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    return rank, world_size


@functools.lru_cache(maxsize=128)
def _gather_size_by_comm(S_LOCAL: int, group: dist.ProcessGroup) -> List[int]:
    r"""Gather the local sequence length from all ranks.
    S_LOCAL: int, local sequence length
    return: List[int], list of sequence lengths from all ranks
    """
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


# Helper functions to pad/unpad head dimension for QKV and O projections
def _maybe_pad_qkv_head(
    x: torch.Tensor,
    H: int,
    group: dist.ProcessGroup,
) -> Tuple[torch.Tensor, int]:
    r"""Maybe pad the head dimension to be divisible by world_size.
    x: torch.Tensor, shape (B, S_LOCAL, H, D)
    H: int, original global head num
    return: Tuple[torch.Tensor, int], padded tensor (B, S_LOCAL, H + H_PAD, D) and H_PAD
    """
    _, world_size = _get_rank_world_size(group)
    H_PAD = 0
    if H % world_size != 0:
        H_PAD = world_size - (H % world_size)
        NEW_H_LOCAL = (H + H_PAD) // world_size
        # e.g., Allow: H=30, world_size=8 -> NEW_H_LOCAL=4, PAD=2.
        # NOT ALLOW: H=30, world_size=16 -> NEW_H_LOCAL=2, PAD=14.
        assert (
            H_PAD < NEW_H_LOCAL
        ), f"Padding head num {H_PAD} should be less than new local head num {NEW_H_LOCAL}"
        x = F.pad(x, (0, 0, 0, H_PAD)).contiguous()
    return x, H_PAD


def _maybe_unpad_qkv_head(
    x: torch.Tensor,
    H_PAD: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    r"""Maybe unpad the head dimension.
    x: torch.Tensor, shape (B, S_GLOBAL, H_LOCAL + H_PAD, D)
    H_PAD: int, head padding num
    return: torch.Tensor, unpadded tensor (B, S_GLOBAL, H_LOCAL, D)
    """
    rank, world_size = _get_rank_world_size(group)
    # Only the last rank may have padding
    if H_PAD > 0 and rank == world_size - 1:
        x = x[:, :, :-H_PAD, :]
    return x.contiguous()


def _maybe_pad_o_head(
    x: torch.Tensor,
    H: int,
    group: dist.ProcessGroup,
) -> Tuple[torch.Tensor, int]:
    r"""Maybe pad the head dimension to be divisible by world_size.
    x: torch.Tensor, shape (B, S_GLOBAL, H_LOCAL, D)
    H: int, original global head num
    return: Tuple[torch.Tensor, int], padded tensor (B, S_GLOBAL, H_LOCAL + H_PAD, D) and H_PAD
    """
    if H is None:
        return x, 0

    rank, world_size = _get_rank_world_size(group)
    H_PAD = 0
    # Only the last rank may need padding
    if H % world_size != 0:
        # We need to broadcast H_PAD to all ranks to keep consistency
        # in unpadding step later for all ranks.
        H_PAD = world_size - (H % world_size)
        NEW_H_LOCAL = (H + H_PAD) // world_size
        assert (
            H_PAD < NEW_H_LOCAL
        ), f"Padding head num {H_PAD} should be less than new local head num {NEW_H_LOCAL}"
        if rank == world_size - 1:
            x = F.pad(x, (0, 0, 0, H_PAD)).contiguous()
    return x, H_PAD


def _maybe_unpad_o_head(
    x: torch.Tensor,
    H_PAD: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    r"""Maybe unpad the head dimension.
    x: torch.Tensor, shape (B, S_LOCAL, H_GLOBAL + H_PAD, D)
    H_PAD: int, head padding num
    return: torch.Tensor, unpadded tensor (B, S_LOCAL, H_GLOBAL, D)
    """
    if H_PAD > 0:
        x = x[:, :, :-H_PAD, :]
    return x.contiguous()


# Helper functions to for all-to-all communication with Ulysses Attention
def _prepare_extra_comm_kwargs(
    query: torch.Tensor,
    **kwargs,
) -> dict:
    num_qo_head = query.shape[2]  # (B, S_LOCAL, H_GLOBAL, D)
    extra_kwargs = {}
    extra_kwargs["num_qo_head"] = num_qo_head
    # Add other kwargs if needed in future
    return extra_kwargs


def _all_to_all_single_qkv_async(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    **kwargs,
) -> torch.Tensor:
    r"""
    x: torch.Tensor, shape (B, S_LOCAL, H, D)
    return: Callable that returns (B, S_GLOBAL, H_LOCAL, D)
    """
    _, world_size = _get_rank_world_size(group)
    B, S_LOCAL, H, D = x.shape
    x, H_PAD = _maybe_pad_qkv_head(x, H, group)
    H_LOCAL = (H + H_PAD) // world_size
    x = x.reshape(B, S_LOCAL, world_size, H_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
    _shape = x.shape  # (world_size, S_LOCAL, B, H_LOCAL, D)

    x = x.flatten()
    x = fc.all_to_all_single(x, None, None, group)

    def wait() -> torch.Tensor:
        nonlocal x, H_PAD
        x = _wait_tensor(x)
        # (world_size, S_LOCAL, B, H_LOCAL, D)
        # -> (S_GLOBAL, B, H_LOCAL, D)
        # -> (B, S_GLOBAL, H_LOCAL, D)
        x = x.reshape(_shape).flatten(0, 1).permute(1, 0, 2, 3).contiguous()
        x = _maybe_unpad_qkv_head(x, H_PAD, group)
        return x

    return wait


def _all_to_all_single_o_async(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    **kwargs,
) -> torch.Tensor:
    r"""
    x: torch.Tensor, shape (B, S_GLOBAL, H_LOCAL, D)
    return: Callable that returns (B, S_LOCAL, H_GLOBAL, D)
    """
    # Assume H is provided in kwargs, since we can't infer H from x's shape.
    # The padding logic needs H to determine if padding is necessary.
    H = kwargs.get("num_qo_head", None)
    _, world_size = _get_rank_world_size(group)
    x, H_PAD = _maybe_pad_o_head(x, H, group)
    B, S_GLOBAL, H_LOCAL, D = x.shape
    S_LOCAL = S_GLOBAL // world_size
    # (B, S_GLOBAL, H_LOCAL, D) -> (world_size, H_LOCAL, B, S_LOCAL, D)
    x = x.reshape(B, world_size, S_LOCAL, H_LOCAL, D).permute(1, 3, 0, 2, 4).contiguous()
    _shape = x.shape  # (world_size, H_LOCAL, B, S_LOCAL, D)

    x = x.flatten()
    x = fc.all_to_all_single(x, None, None, group)

    def wait() -> torch.Tensor:
        nonlocal x, H_PAD
        x = _wait_tensor(x)
        # (world_size, H_LOCAL, B, S_LOCAL, D)
        # -> (H_GLOBAL, B, S_LOCAL, D)
        # -> (B, S_LOCAL, H_GLOBAL, D)
        x = x.reshape(_shape).flatten(0, 1).permute(1, 2, 0, 3).contiguous()
        x = _maybe_unpad_o_head(x, H_PAD, group)
        return x

    return wait


def _all_to_all_single_qkv_fp8_async(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    **kwargs,
) -> Callable[..., torch.Tensor]:
    r"""
    x: torch.Tensor, shape (B, S_LOCAL, H, D)
    return: Callable that returns (B, S_GLOBAL, H_LOCAL, D)
    """
    _, world_size = _get_rank_world_size(group)
    B, S_LOCAL, H, D = x.shape
    x, H_PAD = _maybe_pad_qkv_head(x, H, group)
    H_LOCAL = (H + H_PAD) // world_size
    x = x.reshape(B, S_LOCAL, world_size, H_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
    _shape = x.shape  # (world_size, S_LOCAL, B, H_LOCAL, D)

    x = per_token_quant_fp8(x)  # type: torch.Tensor
    shape_with_scale = x.shape  # (world_size, S_LOCAL, B, H_LOCAL, D + itemsize)
    x = x.flatten()
    x = fc.all_to_all_single(x, None, None, group)

    def wait() -> torch.Tensor:
        nonlocal x, H_PAD
        x = _wait_tensor(x)
        x = x.reshape(shape_with_scale)
        x = per_token_dequant_fp8(x)
        # (world_size, S_LOCAL, B, H_LOCAL, D)
        # -> (S_GLOBAL, B, H_LOCAL, D)
        # -> (B, S_GLOBAL, H_LOCAL, D)
        x = x.reshape(_shape).flatten(0, 1).permute(1, 0, 2, 3).contiguous()
        x = _maybe_unpad_qkv_head(x, H_PAD, group)
        return x

    return wait


def _all_to_all_single_o_fp8_async(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    **kwargs,
) -> Callable[..., torch.Tensor]:
    r"""
    x: torch.Tensor, shape (B, S_GLOBAL, H_LOCAL, D)
    return: Callable that returns (B, S_LOCAL, H_GLOBAL, D)
    """
    # Assume H is provided in kwargs, since we can't infer H from x's shape.
    # The padding logic needs H to determine if padding is necessary.
    H = kwargs.get("num_qo_head", None)
    _, world_size = _get_rank_world_size(group)
    x, H_PAD = _maybe_pad_o_head(x, H, group)
    B, S_GLOBAL, H_LOCAL, D = x.shape
    S_LOCAL = S_GLOBAL // world_size
    # (B, S_GLOBAL, H_LOCAL, D) -> (world_size, H_LOCAL, B, S_LOCAL, D)
    x = x.reshape(B, world_size, S_LOCAL, H_LOCAL, D).permute(1, 3, 0, 2, 4).contiguous()
    _shape = x.shape  # (world_size, H_LOCAL, B, S_LOCAL, D)

    x = per_token_quant_fp8(x)
    shape_with_scale = x.shape  # (world_size, H_LOCAL, B, S_LOCAL, D + itemsize)
    x = x.flatten()
    x = fc.all_to_all_single(x, None, None, group)

    def wait() -> torch.Tensor:
        nonlocal x, H_PAD
        x = _wait_tensor(x)
        x = x.reshape(shape_with_scale)
        x = per_token_dequant_fp8(x)
        # (world_size, H_LOCAL, B, S_LOCAL, D)
        # -> (H_GLOBAL, B, S_LOCAL, D)
        # -> (B, H_GLOBAL, S_LOCAL, D)
        x = x.reshape(_shape).flatten(0, 1).permute(1, 2, 0, 3).contiguous()
        x = _maybe_unpad_o_head(x, H_PAD, group)
        return x

    return wait


@torch.compiler.allow_in_graph
def _all_to_all_single_any_qkv_async(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    **kwargs,
) -> Callable[..., torch.Tensor]:
    r"""
    x: torch.Tensor, shape (B, S_LOCAL, H, D)
    return: Callable that returns (B, S_GLOBAL, H_LOCAL, D)
    """
    _, world_size = _get_rank_world_size(group)
    B, S_LOCAL, H, D = x.shape
    x, H_PAD = _maybe_pad_qkv_head(x, H, group)
    H_LOCAL = (H + H_PAD) // world_size
    # (world_size, S_LOCAL, B, H_LOCAL, D)
    x = x.reshape(B, S_LOCAL, world_size, H_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()

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

    def wait() -> torch.Tensor:
        nonlocal x, H_PAD
        x = _wait_tensor(x)  # (S_GLOBAL, B, H_LOCAL, D)
        # (S_GLOBAL, B, H_LOCAL, D)
        # -> (B, S_GLOBAL, H_LOCAL, D)
        x = x.permute(1, 0, 2, 3).contiguous()
        x = _maybe_unpad_qkv_head(x, H_PAD, group)
        return x

    return wait


@torch.compiler.allow_in_graph
def _all_to_all_single_any_o_async(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    **kwargs,
) -> Callable[..., torch.Tensor]:
    r"""
    x: torch.Tensor, shape (B, S_GLOBAL, H_LOCAL, D)
    return: Callable that returns (B, S_LOCAL, H_GLOBAL, D)
    """
    # Assume H is provided in kwargs, since we can't infer H from x's shape.
    # The padding logic needs H to determine if padding is necessary.
    H = kwargs.get("num_qo_head", None)
    rank, world_size = _get_rank_world_size(group)
    x, H_PAD = _maybe_pad_o_head(x, H, group)
    shape = x.shape  # (B, S_GLOBAL, H_LOCAL, D)
    (B, S_GLOBAL, H_LOCAL, D) = shape

    x = x.flatten(0, 1).contiguous()  # (B*S_GLOBAL, H_LOCAL, D)
    # NOTE: May use tensor_split here to ensure the same split policy
    # that we have used in the EquipartitionSharder sharding strategy. Please
    # note that the 'tensor_split' Splits a tensor into multiple sub-tensors,
    # all of which are views of input, thus may not introduce extra IO access.
    input_split_sizes = [o.shape[0] for o in torch.tensor_split(x, world_size, dim=0)]
    # input_split: e.g, B*S_GLOBAL=1*9 input splits across ranks [[5,4], [5,4],..]
    # output_split: e.g, B*S_GLOBAL=1*9 output splits across ranks [[5,5], [4,4],..]
    output_split_sizes = [input_split_sizes[rank]] * world_size
    x = fc.all_to_all_single(x, output_split_sizes, input_split_sizes, group)

    def wait() -> torch.Tensor:
        nonlocal x, H_PAD
        x = _wait_tensor(x)  # (S_LOCAL*world_size, H_LOCAL, D)
        # NOTE: We can not simply reshape here, because the collective tensors
        # are stacked at dim=0(SeqLen), we need to first split them and then concat at
        # dim=1(Head), otherwise the result will be incorrect due to the linear layout
        # of the tensor in memory.
        H_GLOBAL = H_LOCAL * world_size
        S_LOCAL = x.shape[0] // world_size
        # TODO: How to avoid extra memory IO access here?
        x = torch.cat(x.tensor_split(world_size, dim=0), dim=1)  # (B*S_LOCAL, H_GLOBAL, D)
        x = x.reshape(B, S_LOCAL, H_GLOBAL, D)  # (B, S_LOCAL, H_GLOBAL, D)
        x = _maybe_unpad_o_head(x, H_PAD, group)
        return x

    return wait


@torch.compiler.allow_in_graph
def _all_to_all_single_any_qkv_fp8_async(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    **kwargs,
) -> Callable[..., torch.Tensor]:
    r"""
    x: torch.Tensor, shape (B, S_LOCAL, H, D)
    return: Callable that returns (B, S_GLOBAL, H_LOCAL, D)
    """
    _, world_size = _get_rank_world_size(group)
    B, S_LOCAL, H, D = x.shape
    x, H_PAD = _maybe_pad_qkv_head(x, H, group)
    H_LOCAL = (H + H_PAD) // world_size
    # (world_size, S_LOCAL, B, H_LOCAL, D)
    x = x.reshape(B, S_LOCAL, world_size, H_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()

    input_split_sizes = [S_LOCAL] * world_size
    # S_LOCAL maybe not equal for all ranks in dynamic shape case,
    # since we don't know the actual shape before this timing, thus,
    # we have to use all gather to collect the S_LOCAL first.
    output_split_sizes = _gather_size_by_comm(S_LOCAL, group)
    # NOTE: The `if` branch will introduce graph break for torch.compile,
    # so, we choose to disable the even split optimization implementation
    # _all_to_all_single for now.
    x = per_token_quant_fp8(x)
    x = x.flatten(0, 1)
    x = fc.all_to_all_single(x, output_split_sizes, input_split_sizes, group)

    def wait() -> torch.Tensor:
        nonlocal x, H_PAD
        x = _wait_tensor(x)
        x = per_token_dequant_fp8(x)
        x = x.permute(1, 0, 2, 3).contiguous()
        x = _maybe_unpad_qkv_head(x, H_PAD, group)
        return x

    return wait


@torch.compiler.allow_in_graph
def _all_to_all_single_any_o_fp8_async(
    x: torch.Tensor,
    group: dist.ProcessGroup,
    **kwargs,
) -> Callable[..., torch.Tensor]:
    r"""
    x: torch.Tensor, shape (B, S_GLOBAL, H_LOCAL, D)
    return: Callable that returns (B, S_LOCAL, H_GLOBAL, D)
    """
    # Assume H is provided in kwargs, since we can't infer H from x's shape.
    # The padding logic needs H to determine if padding is necessary.
    H = kwargs.get("num_qo_head", None)
    rank, world_size = _get_rank_world_size(group)
    x, H_PAD = _maybe_pad_o_head(x, H, group)
    shape = x.shape  # (B, S_GLOBAL, H_LOCAL, D)
    (B, S_GLOBAL, H_LOCAL, D) = shape
    x = per_token_quant_fp8(x)

    # NOTE: The `if` branch will introduce graph break for torch.compile,
    # so, we choose to disable the even split optimization implementation
    # _all_to_all_single for now.
    x = x.flatten(0, 1).contiguous()  # (B*S_GLOBAL, H_LOCAL, D)
    # NOTE: May use tensor_split here to ensure the same split policy
    # that we have used in the EquipartitionSharder sharding strategy. Please
    # note that the 'tensor_split' Splits a tensor into multiple sub-tensors,
    # all of which are views of input, thus may not introduce extra IO access.
    input_split_sizes = [o.shape[0] for o in torch.tensor_split(x, world_size, dim=0)]
    # input_split: e.g, B*S_GLOBAL=1*9 input splits across ranks [[5,4], [5,4],..]
    # output_split: e.g, B*S_GLOBAL=1*9 output splits across ranks [[5,5], [4,4],..]
    output_split_sizes = [input_split_sizes[rank]] * world_size
    x = fc.all_to_all_single(x, output_split_sizes, input_split_sizes, group)

    def wait() -> torch.Tensor:
        nonlocal x, H_PAD
        x = _wait_tensor(x)  # (S_LOCAL*world_size, H_LOCAL, D)
        # NOTE: We can not simply reshape here, because the collective tensors
        # are stacked at dim=0(SeqLen), we need to first split them and then concat at
        # dim=1(Head), otherwise the result will be incorrect due to the linear layout
        # of the tensor in memory.
        H_GLOBAL = H_LOCAL * world_size
        S_LOCAL = x.shape[0] // world_size
        # TODO: How to avoid extra memory IO access here?
        x = torch.cat(x.tensor_split(world_size, dim=0), dim=1)  # (B*S_LOCAL, H_GLOBAL, D)
        x = per_token_dequant_fp8(x)
        x = x.reshape(B, S_LOCAL, H_GLOBAL, D)  # (B, S_LOCAL, H_GLOBAL, D)
        x = _maybe_unpad_o_head(x, H_PAD, group)
        return x

    return wait


# Unified functions to select proper all to all implementations according to
# Ulysses Float8 or other settings. Mainly used in Async Ulysses Attention.
# TODO: Refactor basic any_qkvo and non-any_qkvo all2all functions to have
# the same output shape, thus make the unified functions more general and clean.


def _unified_all_to_all_qkv_async_fn(
    disable_fp8: bool = False,
) -> Callable[..., torch.Tensor]:
    from ._templated_ulysses import is_ulysses_float8_enabled
    from ._templated_ulysses import is_ulysses_anything_enabled

    if is_ulysses_anything_enabled():
        if is_ulysses_float8_enabled() and not disable_fp8:
            return _all_to_all_single_any_qkv_fp8_async
        return _all_to_all_single_any_qkv_async
    else:
        if is_ulysses_float8_enabled() and not disable_fp8:
            return _all_to_all_single_qkv_fp8_async
        return _all_to_all_single_qkv_async


def _unified_all_to_all_o_async_fn(
    disable_fp8: bool = False,
) -> Callable[..., torch.Tensor]:
    from ._templated_ulysses import is_ulysses_float8_enabled
    from ._templated_ulysses import is_ulysses_anything_enabled

    if is_ulysses_anything_enabled():
        if is_ulysses_float8_enabled() and not disable_fp8:
            return _all_to_all_single_any_o_fp8_async
        return _all_to_all_single_any_o_async
    else:
        if is_ulysses_float8_enabled() and not disable_fp8:
            return _all_to_all_single_o_fp8_async
        return _all_to_all_single_o_async
