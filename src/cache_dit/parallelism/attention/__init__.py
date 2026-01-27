from cache_dit.envs import ENV
from ._context_parallel_utils import _ExtendedContextParallelConfig
from ._distributed_primitives import (
    _unified_all_to_all_o_async_fn,
    _unified_all_to_all_qkv_async_fn,
    _prepare_ulysses_comm_metadata,
)
from ._experimental_utils import (
    _is_diffusers_parallelism_available,
    _maybe_patch_find_submodule,
)
from ._templated_ulysses import (
    enable_ulysses_anything,
    enable_ulysses_float8,
)


def _maybe_register_custom_attn_backends():
    """Maybe re-register native attention backend to enable context parallelism."""
    if not ENV.CACHE_DIT_ENABLE_CUSTOM_ATTN_DISPATCH:
        return

    if ENV.CACHE_DIT_ENABLE_CUSTOM_ATTN_ALREADY_DISPATCH:
        return

    ENV.CACHE_DIT_ENABLE_CUSTOM_ATTN_ALREADY_DISPATCH = True

    try:
        from ._attention_dispatch import (
            _native_attention,
            _sdpa_cudnn_attention,
            _sage_attention,
            _flash_attention_3,
            _native_npu_attention,
        )
    except ImportError as e:
        raise e
