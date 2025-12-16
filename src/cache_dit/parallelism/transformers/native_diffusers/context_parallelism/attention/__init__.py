from cache_dit.envs import ENV


def _maybe_register_custom_attn_backends():
    """Maybe re-register native attention backend to enable context parallelism."""
    # Import custom attention backend ensuring registration
    if (
        ENV.CACHE_DIT_ENABLE_CUSTOM_ATTN_DISPATCH
        and not ENV.CACHE_DIT_ENABLE_CUSTOM_ATTN_ALREADY_DISPATCH
    ):
        from ._attention_dispatch import (
            _native_attention,
            _sdpa_cudnn_attention,
            _sage_attention,
        )

        ENV.CACHE_DIT_ENABLE_CUSTOM_ATTN_ALREADY_DISPATCH = True
