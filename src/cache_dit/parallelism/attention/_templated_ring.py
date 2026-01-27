import torch
from typing import Optional, Tuple
import torch.distributed as dist
import torch.nn.functional as F

from diffusers.utils.import_utils import is_torch_version

try:
    from ...kernels import fused_merge_attn_states

    _is_triton_kernel_available = True
except ImportError:
    fused_merge_attn_states = None
    _is_triton_kernel_available = False

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
            return _TemplatedRingAllGatherAttention.apply(
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
            return _TemplatedRingBatchedP2PAttention.apply(
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


class _TemplatedRingAllGatherAttention(TemplatedRingAttention):
    """A wrapper of diffusers' TemplatedRingAttention to avoid name conflict."""

    pass


# Adapted from: https://github.com/zhuzilin/ring-flash-attention/blob/main/ring_flash_attn/utils.py#L98
class _RingBatchedP2PComm:
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
        to_send = to_send.contiguous()
        if recv_tensor is None:
            res = torch.empty_like(to_send).contiguous()
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


class _TemplatedRingBatchedP2PAttention(torch.autograd.Function):
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
        ring_group = ring_mesh.get_group()

        comm = _RingBatchedP2PComm(ring_group)

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
                True,  # return_lse
                _save_ctx=step == 0,
                _parallel_config=_parallel_config,
            )

            # Refer to:
            # https://github.com/huggingface/diffusers/pull/12693#issuecomment-3627519544
            if is_torch_version("<", "2.9.0") and lse.dim() == 3:
                lse = lse.unsqueeze(-1)

            # Use _fused_merge_attn_states to combine the attention outputs and lses
            if prev_out is not None:
                # out = prev_out - F.sigmoid(lse - prev_lse) * (prev_out - out)
                # lse = prev_lse - F.logsigmoid(prev_lse - lse)
                if _is_triton_kernel_available:
                    out, lse = fused_merge_attn_states(
                        prev_out,
                        prev_lse,
                        out,
                        lse,
                    )
                else:
                    if _parallel_config.context_parallel_config.convert_to_fp32:
                        out = out.to(torch.float32)
                        lse = out.to(torch.float32)

                    out = prev_out - F.sigmoid(lse - prev_lse) * (prev_out - out)
                    lse = prev_lse - F.logsigmoid(prev_lse - lse)

            prev_out, prev_lse = out, lse

            if step + 1 != comm.world_size:
                comm.wait()
                key, value = next_k, next_v

        out = out.to(query.dtype)
        lse = lse.squeeze(-1)  # (B, S_Q_GLOBAL, H_LOCAL)

        return (out, lse) if return_lse else out

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_out: torch.Tensor,
        *args,
    ):
        raise NotImplementedError(
            "Backward pass is not implemented for _TemplatedRingP2PAttention."
        )
