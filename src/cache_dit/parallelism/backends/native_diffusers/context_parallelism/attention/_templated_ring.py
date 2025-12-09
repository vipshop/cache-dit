# TODO: Support TemplatedRingAttention in cache-dit with PyTorch context-parallel api.
# Reference: https://docs.pytorch.org/tutorials/unstable/context_parallel.html
try:
    from diffusers.models.attention_dispatch import (
        TemplatedRingAttention,
    )
except ImportError:
    raise ImportError(
        "Context parallelism requires the 'diffusers>=0.36.dev0'."
        "Please install latest version of diffusers from source: \n"
        "pip3 install git+https://github.com/huggingface/diffusers.git"
    )


class _TemplatedRingAttention(TemplatedRingAttention):
    """A wrapper of diffusers' TemplatedRingAttention to avoid name conflict."""

    pass
