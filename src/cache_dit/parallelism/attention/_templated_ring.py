# TODO: Support TemplatedRingAttention in cache-dit with PyTorch context-parallel api.
# Reference: https://docs.pytorch.org/tutorials/unstable/context_parallel.html
import torch
import triton
import triton.language as tl
from typing import Optional, Tuple
import torch.distributed as dist
from diffusers.utils.import_utils import is_torch_version


try:
    from diffusers.models.attention_dispatch import TemplatedRingAttention
    from diffusers.models._modeling_parallel import ParallelConfig
except ImportError:
    raise ImportError(
        "Context parallelism requires the 'diffusers>=0.36.dev0'."
        "Please install latest version of diffusers from source: \n"
        "pip3 install git+https://github.com/huggingface/diffusers.git"
    )


__all__ = ["UnifiedTemplatedRingAttention"]


class UnifiedTemplatedRingAttention(torch.autograd.Function):
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
        if _parallel_config.context_parallel_config.rotate_method == "allgather":
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
        elif _parallel_config.context_parallel_config.rotate_method == "p2p":
            return _TemplatedRotatedRingAttention.apply(
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
            raise ValueError(
                f"Unsupported rotate_method: {_parallel_config.context_parallel_config.rotate_method}"
            )


class _TemplatedRingAttention(TemplatedRingAttention):
    """A wrapper of diffusers' TemplatedRingAttention to avoid name conflict."""

    pass


# Adapted from: https://github.com/zhuzilin/ring-flash-attention/blob/main/ring_flash_attn/utils.py#L98
class _RotatedRingComm:
    def __init__(self, process_group: dist.ProcessGroup):
        self._process_group = process_group
        self._ops = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size

        if process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)

    def send_recv(
        self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor

        send_op = dist.P2POp(dist.isend, to_send, self.send_rank, group=self._process_group)
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        self._ops.append(send_op)
        self._ops.append(recv_op)
        return res

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = dist.batch_isend_irecv(self._ops)

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        self._reqs = None
        self._ops = []

    def send_recv_kv(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_buffer: Optional[torch.Tensor] = None,
        v_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        next_k, next_v = self.send_recv(k, k_buffer), self.send_recv(v, v_buffer)
        self.commit()
        return next_k, next_v


class _TemplatedRotatedRingAttention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        query: torch.Tensor,  # (B, S_LOCAL, H, D)
        key: torch.Tensor,  # (B, S_LOCAL, H, D)
        value: torch.Tensor,  # (B, S_LOCAL, H, D)
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
        ring_mesh = _parallel_config.context_parallel_config._ring_mesh
        group = ring_mesh.get_group()

        comm = _RotatedRingComm(group)

        prev_out = prev_lse = None

        ctx.forward_op = forward_op
        ctx.backward_op = backward_op
        ctx.q_shape = query.shape
        ctx.kv_shape = key.shape
        ctx._parallel_config = _parallel_config

        next_k, next_v = None, None

        for step in range(comm.world_size):
            if step + 1 != comm.world_size:
                next_k, next_v = comm.send_recv_kv(key, value)

            # [B, N, H, D], [B, N, H, 1]
            out, lse = forward_op(
                ctx,
                query,
                key,
                value,
                attn_mask,
                dropout_p,
                is_causal,
                scale,
                enable_gqa,
                True,
                _save_ctx=step == 0,
                _parallel_config=_parallel_config,
            )

            # Refer to:
            # https://github.com/huggingface/diffusers/pull/12693#issuecomment-3627519544
            if is_torch_version("<", "2.9.0"):
                lse = lse.unsqueeze(-1)

            # Use _fused_merge_attn_states to combine the attention outputs and lses
            if prev_out is not None:
                # out = prev_out - F.sigmoid(lse - prev_lse) * (prev_out - out)
                # lse = prev_lse - F.logsigmoid(prev_lse - lse)
                out, lse = _fused_merge_attn_states(
                    prev_out,
                    prev_lse,
                    out,
                    lse,
                )

            prev_out, prev_lse = out, lse

            if step + 1 != comm.world_size:
                comm.wait()
                key, value = next_k, next_v

        out = out.to(query.dtype)
        lse = lse.squeeze(-1)

        return (out, lse) if return_lse else out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_out: torch.Tensor,
        *args,
    ):
        raise NotImplementedError(
            "Backward pass is not implemented for _TemplatedRotatedRingAttention."
        )


def _fused_merge_attn_states(
    prev_out: torch.Tensor,  # [B, N, H, D]
    prev_lse: torch.Tensor,  # [B, N, H, 1]
    suff_out: torch.Tensor,
    suff_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, N, H, D = suff_out.shape  # Batch, Seq_len, Num_heads, Head_dim
    # Flatten the batch and sequence dimensions
    prev_out = prev_out.flatten(0, 1).contiguous()  # [B*N, H, D]
    suff_out = suff_out.flatten(0, 1).contiguous()  # [B*N, H, D]
    prev_lse = prev_lse.flatten(0, 1).squeeze(-1).contiguous()  # [B*N, H]
    suff_lse = suff_lse.flatten(0, 1).squeeze(-1).contiguous()  # [B*N, H]

    out = torch.empty_like(suff_out).contiguous()
    lse = torch.empty_like(suff_lse).contiguous()

    _fused_merge_attn_states_kernel[(B * N, H)](
        out,
        lse,
        prev_out,
        prev_lse,
        suff_out,
        suff_lse,
        D,
        triton.next_power_of_2(D),
    )

    # Reshape back to original shape
    out = out.view(B, N, H, D)
    lse = lse.view(B, N, H, 1)
    return out, lse


@triton.jit
def _fused_merge_attn_states_kernel(
    out_ptr: tl.tensor,  # [B*N, H, D],
    lse_ptr: tl.tensor,  # [B*N, H],
    prev_out_ptr: tl.tensor,  # [B*N, H, D],
    prev_lse_ptr: tl.tensor,  # [B*N, H],
    suff_out_ptr: tl.tensor,  # [B*N, H, D],
    suff_lse_ptr: tl.tensor,  # [B*N, H],
    HEAD_SIZE: tl.constexpr,
    PADDED_HEAD_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    num_heads = tl.num_programs(1)

    # NOTE(DefTruth): Use float32 for numerical stability
    prev_lse = tl.load(prev_lse_ptr + token_idx * num_heads + head_idx).to(tl.float32)
    suff_lse = tl.load(suff_lse_ptr + token_idx * num_heads + head_idx).to(tl.float32)
    prev_lse = float("-inf") if prev_lse == float("inf") else prev_lse
    suff_lse = float("-inf") if suff_lse == float("inf") else suff_lse

    head_arange = tl.arange(0, PADDED_HEAD_SIZE)
    head_mask = head_arange < HEAD_SIZE
    prev_out = tl.load(
        prev_out_ptr + token_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE + head_arange,
        mask=head_mask,
    ).to(tl.float32)
    suff_out = tl.load(
        suff_out_ptr + token_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE + head_arange,
        mask=head_mask,
    ).to(tl.float32)

    # compute: out = prev_out - F.sigmoid(lse - prev_lse) * (prev_out - out)
    out = prev_out - tl.sigmoid(suff_lse - prev_lse) * (prev_out - suff_out)
    out = out.to(out_ptr.dtype.element_ty)
    tl.store(
        out_ptr + token_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE + head_arange,
        out,
        mask=head_mask,
    )

    # compute: lse = prev_lse - F.logsigmoid(prev_lse - lse)
    lse = prev_lse - tl.log(tl.sigmoid(prev_lse - suff_lse))
    lse = lse.to(lse_ptr.dtype.element_ty)
    tl.store(lse_ptr + token_idx * num_heads + head_idx, lse)
