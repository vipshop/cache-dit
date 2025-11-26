import functools
from typing import Tuple, List

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as fc

from cache_dit.logger import init_logger

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


def _all_to_all_single_sync(
    x: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    return _all_to_all_single(x, group)


def _all_to_all_single_async(
    x: torch.Tensor,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    # TODO: should we use dist.all_to_all_single with async_op=True here?
    x = x.flatten()
    x = fc.all_to_all_single(x, None, None, group)
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
