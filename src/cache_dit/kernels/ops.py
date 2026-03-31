# Currently we only have triton kernels for cache-dit, but we can re-implement
# some of them in CuteDSL for better performance if needed in the future. In
# that case, we can keep the same operator definitions in this file and just
# change the underlying implementations.
import torch
from functools import partial
from typing import Tuple, Callable
from .backend import KernelBackend


_KERNEL_BE_FN = Callable[..., KernelBackend]
# Currently we only support triton backend, but we may support more backends
# in the future, such as cuteDSL, etc. In that case, we can automatically
# select the kernel backend based on the hardware and software environment,
# or provide an interface for users to specify the kernel backend.
_DEFAULT_BE_FN = lambda: KernelBackend.TRITON


def _fp8_comm_per_token_quant_impl(
    x: torch.Tensor,
    backend_fn: _KERNEL_BE_FN = _DEFAULT_BE_FN,
) -> torch.Tensor:
    backend = backend_fn()
    if backend == KernelBackend.TRITON:
        from .triton import fp8_comm_per_token_quant

        return fp8_comm_per_token_quant(x)
    else:
        raise ValueError(f"kernel backend: {backend} is not supported now!")


def _fp8_comm_per_token_dequant_impl(
    x: torch.Tensor,
    backend_fn: _KERNEL_BE_FN = _DEFAULT_BE_FN,
) -> torch.Tensor:
    backend = backend_fn()
    if backend == KernelBackend.TRITON:
        from .triton import fp8_comm_per_token_dequant

        return fp8_comm_per_token_dequant(x)
    else:
        raise ValueError(f"kernel backend: {backend} is not supported now!")


def _fp8_comm_qkv_permute_quant_impl(
    x: torch.Tensor,
    backend_fn: _KERNEL_BE_FN = _DEFAULT_BE_FN,
) -> torch.Tensor:
    backend = backend_fn()
    if backend == KernelBackend.TRITON:
        from .triton import fp8_comm_qkv_permute_quant

        return fp8_comm_qkv_permute_quant(x)
    else:
        raise ValueError(f"kernel backend: {backend} is not supported now!")


def _fp8_comm_qkv_permute_dequant_impl(
    quant_x: torch.Tensor,
    backend_fn: _KERNEL_BE_FN = _DEFAULT_BE_FN,
) -> torch.Tensor:
    backend = backend_fn()
    if backend == KernelBackend.TRITON:
        from .triton import fp8_comm_qkv_permute_dequant

        return fp8_comm_qkv_permute_dequant(quant_x)
    else:
        raise ValueError(f"kernel backend: {backend} is not supported now!")


def _fused_merge_attn_states_impl(
    prev_out: torch.Tensor,
    prev_lse: torch.Tensor,
    suff_out: torch.Tensor,
    suff_lse: torch.Tensor,
    backend_fn: _KERNEL_BE_FN = _DEFAULT_BE_FN,
) -> Tuple[torch.Tensor, torch.Tensor]:
    backend = backend_fn()
    if backend == KernelBackend.TRITON:
        from .triton import fused_merge_attn_states

        return fused_merge_attn_states(prev_out, prev_lse, suff_out, suff_lse)
    else:
        raise ValueError(f"kernel backend: {backend} is not supported now!")


fp8_comm_per_token_quant: Callable[..., torch.Tensor] = partial(
    _fp8_comm_per_token_quant_impl,
    backend_fn=_DEFAULT_BE_FN,
)
fp8_comm_per_token_dequant: Callable[..., torch.Tensor] = partial(
    _fp8_comm_per_token_dequant_impl,
    backend_fn=_DEFAULT_BE_FN,
)
fp8_comm_qkv_permute_quant: Callable[..., torch.Tensor] = partial(
    _fp8_comm_qkv_permute_quant_impl,
    backend_fn=_DEFAULT_BE_FN,
)
fp8_comm_qkv_permute_dequant: Callable[..., torch.Tensor] = partial(
    _fp8_comm_qkv_permute_dequant_impl,
    backend_fn=_DEFAULT_BE_FN,
)
fused_merge_attn_states: Callable[..., Tuple[torch.Tensor, torch.Tensor]] = partial(
    _fused_merge_attn_states_impl,
    backend_fn=_DEFAULT_BE_FN,
)

__all__ = [
    # FP8 related ops
    "fp8_comm_per_token_quant",
    "fp8_comm_per_token_dequant",
    "fp8_comm_qkv_permute_quant",
    "fp8_comm_qkv_permute_dequant",
    # Attention related ops
    "fused_merge_attn_states",
]
