def maybe_resigter_native_attention_backend():
    """Maybe re-register native attention backend to enable context parallelism."""
    # Import custom attention backend ensuring registration
    from ._attention_dispatch import _native_attention
