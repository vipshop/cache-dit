def _maybe_resigter_attn_backends():
    """Maybe re-register native attention backend to enable context parallelism."""
    # Import custom attention backend ensuring registration
    from ._attention_dispatch import _native_attention, _sdpa_cudnn_attention, _sage_attention
