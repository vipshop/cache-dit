def maybe_resigter_native_attention_backend():
    """Maybe re-register native attention backend to enable context parallelism."""
    # Import custom attention backend ensuring registration
    from ._attention_dispatch import _native_attention


from ._templated_ulysses_anything import (
    enable_ulysses_anything,
    is_ulysses_anything_enabled,
    disable_ulysses_anything,
)
