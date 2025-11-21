import os
import functools
from enum import Enum
from typing import Optional, Tuple, List

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as fc

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

logger = init_logger(__name__)

__all__ = [
    "TemplatedUlyssesAnythingAttention",
    "EquipartitionSharder",
    "enable_ulysses_anything",
    "is_ulysses_anything_enabled",
    "disable_ulysses_anything",
]


# Reference:
# - https://github.com/pytorch/pytorch/blob/f58a680d09e13658a52c6ba05c63c15759846bcc/torch/distributed/_functional_collectives.py#L827
# - https://github.com/pytorch/pytorch/blob/f58a680d09e13658a52c6ba05c63c15759846bcc/torch/distributed/_functional_collectives.py#L246
# - https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_dispatch.py#L1012
# For fullgraph=True tracing compatibility (since FakeTensor does not have a `wait` method):
# TODO: How to avoid unwaited collective calls warnings in torch.compile graphs?
def _wait_tensor(tensor):
    if isinstance(tensor, fc.AsyncCollectiveTensor):
        tensor = tensor.wait()

    return tensor


# Reference:
# - https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_dispatch.py#L1012
def _all_to_all_single(x: torch.Tensor, group) -> torch.Tensor:
    shape = x.shape
    # HACK: We need to flatten because despite making tensors contiguous, torch single-file-ization
    # to benchmark triton codegen fails somewhere:
    # buf25 = torch.ops._c10d_functional.all_to_all_single.default(buf24, [1, 1], [1, 1], '3')
    # ValueError: Tensors must be contiguous
    x = x.flatten()
    x = fc.all_to_all_single(x, None, None, group)
    x = x.reshape(shape)
    x = _wait_tensor(x)
    return x


@torch.compiler.disable
def _maybe_get_rank_world_size(
    group: dist.ProcessGroup,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> Tuple[int, int]:
    if world_size is None:
        world_size = dist.get_world_size(group=group)
    if rank is None:
        rank = dist.get_rank(group=group)
    return rank, world_size


@torch.compiler.disable
def _check_all_sizes_same(sizes: List[int]) -> bool:
    first_size = sizes[0]
    for s in sizes:
        if s != first_size:
            return False
    return True


# NOTE: Disable torch.compile to avoid compile error - Explanation:
# Backend compiler `inductor` failed with aten._local_scalar_dense.default
@torch.compiler.disable
def _tensor_tolist(tensor: torch.Tensor) -> List[int]:
    return tensor.tolist()


@torch.compiler.disable
def _split_sizes(S_GLOBAL: int, world_size: int) -> List[int]:
    assert world_size > 0, "world_size must be greater than 0"
    assert S_GLOBAL >= world_size, "S_GLOBAL must be greater than or equal to world_size"

    base = S_GLOBAL // world_size
    remainder = S_GLOBAL % world_size

    splits = [base + 1 if i < remainder else base for i in range(world_size)]

    return splits


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
    gathered_sizes = fc.all_gather_tensor(
        torch.tensor(S_LOCAL, device=x.device), gather_dim=0, group=group
    )
    gathered_sizes = _wait_tensor(gathered_sizes)

    # NOTE(DefTruth): Please note that torch.compile will raise an NCCL
    # timeout error here - 'Watchdog caught collective operation timeout:
    # WorkNCCL(SeqNum=435, OpType=ALLTOALL_BASE ...'), so, we choose to
    # introduce a graph break here as a temporary workaround.
    output_split_sizes = _tensor_tolist(gathered_sizes)
    # NOTE(DefTruth): Using _all_to_all_single if the gathered_sizes
    # are all equal, which may be more efficient.
    # torch._dynamo.graph_break()
    if _check_all_sizes_same(output_split_sizes):
        x = _all_to_all_single(x, group)
        # (world_size * S_LOCAL, B, H_LOCAL, D)
        x = x.flatten(0, 1).contiguous()
        return x  # (S_GLOBAL, B, H_LOCAL, D)

    # torch._dynamo.graph_break()
    x = x.flatten(0, 1)  # (world_size * S_LOCAL, B, H_LOCAL, D)
    x = fc.all_to_all_single(x, output_split_sizes, input_split_sizes, group)
    x = _wait_tensor(x)  # (S_GLOBAL, B, H_LOCAL, D)
    return x


def _all_to_all_single_any_o(
    out: torch.Tensor,
    group: dist.ProcessGroup,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> torch.Tensor:
    rank, world_size = _maybe_get_rank_world_size(group, rank, world_size)
    shape = out.shape  # (B, S_GLOBAL, H_LOCAL, D)
    (B, S_GLOBAL, H_LOCAL, D) = shape

    # If S_GLOBAL is divisible by world_size, we can use the more
    # efficient _all_to_all_single implementation.
    # torch._dynamo.graph_break()
    if S_GLOBAL % world_size == 0:
        # (B, S_GLOBAL, H_LOCAL, D) -> (world_size, H_LOCAL, B, S_Q_LOCAL, D)
        out = (
            out.reshape(B, world_size, S_GLOBAL // world_size, H_LOCAL, D)
            .permute(1, 3, 0, 2, 4)
            .contiguous()
        )
        out = _all_to_all_single(out, group)
        # (world_size * H_LOCAL, B, S_Q_LOCAL, D) -> (B, S_Q_LOCAL, H_GLOBAL, D)
        out = out.flatten(0, 1).permute(1, 2, 0, 3).contiguous()
        return out

    # torch._dynamo.graph_break()
    out = out.flatten(0, 1).contiguous()  # (B*S_GLOBAL, H_LOCAL, D)
    # NOTE(DefTruth): May use tensor_split here to ensure the same split policy
    # that we have used in the EquipartitionSharder sharding strategy. Please
    # note that the 'tensor_split' Splits a tensor into multiple sub-tensors,
    # all of which are views of input, thus may not introduce extra IO access.
    # input_split_sizes = _split_sizes(S_GLOBAL * B, world_size)
    input_split_sizes = [o.shape[0] for o in torch.tensor_split(out, world_size, dim=0)]
    # input_split: e.g, B*S_GLOBAL=1*9 input splits across ranks [[5,4], [5,4],..]
    # output_split: e.g, B*S_GLOBAL=1*9 output splits across ranks [[5,5], [4,4],..]
    output_split_sizes = [input_split_sizes[rank]] * world_size
    out = fc.all_to_all_single(out, output_split_sizes, input_split_sizes, group)
    out = _wait_tensor(out)  # (S_LOCAL*world_size, H_LOCAL, D)
    # NOTE(DefTruth): We can not simply reshape here, because the collective tensors
    # are stacked at dim=0(SeqLen), we need to first split them and then concat at
    # dim=1(Head), otherwise the result will be incorrect due to the linear layout
    # of the tensor in memory.
    H_GLOBAL = H_LOCAL * world_size
    S_LOCAL = out.shape[0] // world_size
    out = torch.cat(out.tensor_split(world_size, dim=0), dim=1)  # (B*S_LOCAL, H_GLOBAL, D)
    out = out.reshape(B, S_LOCAL, H_GLOBAL, D).contiguous()  # (B, S_LOCAL, H_GLOBAL, D)
    return out


def _gather_split_any_o(  # noqa: F811
    out: torch.Tensor,
    group: dist.ProcessGroup,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> torch.Tensor:
    # NOTE(DefTruth): This is an alternative implementation of _all_to_all_single
    # for any o. It use all_gather and split, which may be less efficient.
    rank, world_size = _maybe_get_rank_world_size(group, rank, world_size)
    # (B, S_GLOBAL, H_LOCAL, D)
    # all gather to get (B, S_GLOBAL, H_GLOBAL, D) at H_GLOBAL dim
    out_gathered = [torch.empty_like(out) for _ in range(world_size)]
    dist.all_gather(out_gathered, out, group=group)
    out_gathered = torch.cat(out_gathered, dim=2)
    # (B, S_GLOBAL, H_GLOBAL, D) -> (B, S_Q_LOCAL, H_GLOBAL, D)
    out = out_gathered.tensor_split(world_size, dim=1)[rank].contiguous()
    return out


class _CommType(Enum):
    ALL_TO_ALL = "all_to_all_single"
    GATHER_SPLIT = "gather_split"


class TemplatedUlyssesAnythingAttention(torch.autograd.Function):

    _o_split_comm = _CommType.ALL_TO_ALL

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
        if TemplatedUlyssesAnythingAttention._o_split_comm == _CommType.ALL_TO_ALL:
            out = _all_to_all_single_any_o(out, group)
        else:
            out = _gather_split_any_o(out, group)

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


# NOTE(DefTruth): We use `tensor_split` instead of chunk, because the `chunk`
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


_CACHE_DIT_ENABELD_ULYSSES_ANYTHING = (
    os.environ.get("CACHE_DIT_ENABELD_ULYSSES_ANYTHING", "0") == "1"
)


def enable_ulysses_anything(**kwargs):
    global _CACHE_DIT_ENABELD_ULYSSES_ANYTHING
    try:
        if _CACHE_DIT_ENABELD_ULYSSES_ANYTHING:
            # function for TemplatedUlyssesAnythingAttention.
            if EquipartitionSharder.shard != shard_anything:
                EquipartitionSharder.shard = shard_anything
                logger.warning(
                    "Ulysses Anything Attention is already enabled in cache-dit. "
                    "but EquipartitionSharder.shard is not set correctly, "
                    "resetting it to the correct shard_anything function."
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
            logger.info(
                "EquipartitionSharder.shard is set to shard_anything function "
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
