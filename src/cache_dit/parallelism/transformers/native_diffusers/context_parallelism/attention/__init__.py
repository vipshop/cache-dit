from cache_dit.envs import ENV


def _maybe_register_custom_attn_backends():
    """Maybe re-register native attention backend to enable context parallelism."""
    if not ENV.CACHE_DIT_ENABLE_CUSTOM_ATTN_DISPATCH:
        return

    if ENV.CACHE_DIT_ENABLE_CUSTOM_ATTN_ALREADY_DISPATCH:
        return

    ENV.CACHE_DIT_ENABLE_CUSTOM_ATTN_ALREADY_DISPATCH = True

    from ._attention_dispatch import (
        _native_attention,
        _sdpa_cudnn_attention,
        _sage_attention,
        _flash_attention_3,
    )
