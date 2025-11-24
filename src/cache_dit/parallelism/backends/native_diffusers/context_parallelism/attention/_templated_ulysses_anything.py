import os
import copy
import functools
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
def _wait_tensor(tensor):
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
