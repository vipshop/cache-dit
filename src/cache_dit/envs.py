import os


class ENV(object):
    # ENVs for cache-dit

    # Logging ENVs
    CACHE_DIT_LOG_LEVEL: str = os.environ.get("CACHE_DIT_LOG_LEVEL", "info")
    CACHE_DIT_LOG_DIR: str = os.environ.get("CACHE_DIT_LOG_DIR", None)

    # Parallelism ENVs

    # Enable custom attention backend dispatch for context parallelism
    # in cache-dit by default. Users can set the environment variable
    # to 0 to disable this behavior. Default to enabled for better
    # compatibility and performance.
    CACHE_DIT_ENABLE_CUSTOM_ATTN_DISPATCH: bool = bool(
        int(os.getenv("CACHE_DIT_ENABLE_CUSTOM_ATTN_DISPATCH", "1"))
    )

    # Avoid re-registering custom attention backend dispatch in cache-dit.
    # Inner use only. Users should not set this variable directly.
    CACHE_DIT_ENABLE_CUSTOM_ATTN_ALREADY_DISPATCH: bool = bool(
        int(os.getenv("CACHE_DIT_ENABLE_CUSTOM_ATTN_ALREADY_DISPATCH", "0"))
    )

    # Environment variable flags for Ulysses Attention variants in cache-dit.
    # Enable Ulysses Anything Attention by setting the environment variable to 1.
    # Otherwise, users can set it by 'exprimental_ulysses_anything' argument in
    # ContextParallelism.
    CACHE_DIT_ENABELD_ULYSSES_ANYTHING: bool = bool(
        int(os.environ.get("CACHE_DIT_ENABELD_ULYSSES_ANYTHING", "0"))
    )

    # Enable Ulysses Anything Attention Float8 by setting the environment variable to 1.
    # Otherwise, users can set it by 'experimental_ulysses_anything=True' and
    # 'experimental_ulysses_float=True' arguments in ContextParallelism.
    CACHE_DIT_ENABELD_ULYSSES_ANYTHING_FLOAT8: bool = bool(
        int(os.environ.get("CACHE_DIT_ENABELD_ULYSSES_ANYTHING_FLOAT8", "0"))
    )

    # Enable Ulysses Attention by setting the environment variable to 1.
    # Otherwise, users can set it by 'experimental_ulysses_float8' argument in
    # ContextParallelism.
    CACHE_DIT_ENABELD_ULYSSES_FLOAT8: bool = bool(
        int(os.environ.get("CACHE_DIT_ENABELD_ULYSSES_FLOAT8", "0"))
    )

    # Enable unpadded communication for uneven attention heads without padding
    # by setting the environment variable to 1.
    CACHE_DIT_UNEVEN_HEADS_COMM_NO_PAD: bool = bool(
        int(os.environ.get("CACHE_DIT_UNEVEN_HEADS_COMM_NO_PAD", "0"))
    )

    # Models ENVs

    # Users should never use this variable directly, it is only for developers
    # to control whether to enable dummy blocks for FLUX, default to enabled.
    CACHE_DIT_FLUX_ENABLE_DUMMY_BLOCKS: bool = bool(
        int(os.environ.get("CACHE_DIT_FLUX_ENABLE_DUMMY_BLOCKS", "1"))
    )

    # Torch compile ENVs

    # Enable epilogue and prologue fusion in cache-dit compile optimizations
    CACHE_DIT_EPILOGUE_PROLOGUE_FUSION: bool = bool(
        int(os.environ.get("CACHE_DIT_EPILOGUE_PROLOGUE_FUSION", "0"))
    )

    # Enable compile compute-communication (all reduce) overlap in cache-dit by
    # default. Users can set the environment variable to 0 to disable this behavior.
    # Default to enabled for better performance.
    CACHE_DIT_ENABLE_COMPILE_COMPUTE_COMM_OVERLAP: bool = bool(
        int(os.environ.get("CACHE_DIT_ENABLE_COMPILE_COMPUTE_COMM_OVERLAP", "1"))
    )

    # Force disable custom compile config in cache-dit by setting the environment
    # variable to 1. Otherwise, cache-dit will set custom compile configs for
    # better performance during torch.compile.
    CACHE_DIT_FORCE_DISABLE_CUSTOM_COMPILE_CONFIG: bool = bool(
        int(os.environ.get("CACHE_DIT_FORCE_DISABLE_CUSTOM_COMPILE_CONFIG", "0"))
    )

    # Patch Functors ENVs

    # Force disable the checking of whether the model is from diffusers in patch functors.
    # Users can set the environment variable to 1 to disable this behavior.
    CACHE_DIT_PATCH_FUNCTOR_DISABLE_DIFFUSERS_CHECK: bool = bool(
        int(os.environ.get("CACHE_DIT_PATCH_FUNCTOR_DISABLE_DIFFUSERS_CHECK", "0"))
    )
