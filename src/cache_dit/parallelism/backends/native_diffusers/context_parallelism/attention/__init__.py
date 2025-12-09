def maybe_resigter_native_attention_backend():
    """Maybe re-register native attention backend to enable context parallelism."""
    # Import custom attention backend ensuring registration
    from ._attention_dispatch import _native_attention, _sdpa_cudnn_attention, _sage_attention


from ._templated_ulysses import (
    enable_ulysses_anything,
    is_ulysses_anything_enabled,
    disable_ulysses_anything,
    enable_ulysses_float8,
    is_ulysses_float8_enabled,
    disable_ulysses_float8,
)
