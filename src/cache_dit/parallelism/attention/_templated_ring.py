# TODO: Support TemplatedRingAttention in cache-dit with PyTorch context-parallel api.
# Reference: https://docs.pytorch.org/tutorials/unstable/context_parallel.html
import torch
from typing import Optional

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
        return _TemplatedRingAttention.apply(
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


class _TemplatedRingAttention(TemplatedRingAttention):
    """A wrapper of diffusers' TemplatedRingAttention to avoid name conflict."""

    pass
