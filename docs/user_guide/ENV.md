# Configurable Environment Variables

This document summarizes all core configurable environment variables in cache-dit, which control key functionalities such as logging behavior, parallel computing strategies, model adaptation, compilation optimization, and patch tool logic.

- <span style="color:#c77dff;">CACHE_DIT_LOG_LEVEL</span>, default `"info"`, Controls the logging level of cache-dit.
- <span style="color:#c77dff;">CACHE_DIT_LOG_DIR</span>, default `None`, Specifies the directory for cache-dit log files (if not set, logs are output to console by default).
- <span style="color:#c77dff;">CACHE_DIT_UNEVEN_HEADS_COMM_NO_PAD</span>, default `False (0)`, Enables unpadded communication for uneven attention heads (avoids padding overhead) when set to 1.
- <span style="color:#c77dff;">CACHE_DIT_FLUX_ENABLE_DUMMY_BLOCKS</span>, default `True (1)`, For **developer use only** – controls whether dummy blocks are enabled for FLUX models (enabled by default). Users should NOT use this variable directly.
- <span style="color:#c77dff;">CACHE_DIT_EPILOGUE_PROLOGUE_FUSION</span>, default `False (0)`, Enables epilogue and prologue fusion in cache-dit's torch.compile optimizations.
- <span style="color:#c77dff;">CACHE_DIT_ENABLE_COMPILE_COMPUTE_COMM_OVERLAP</span>, default `True (1)`, Enables compute-communication (all-reduce) overlap during cache-dit compilation. Enabled by default for better performance; set to 0 to disable.
- <span style="color:#c77dff;">CACHE_DIT_FORCE_DISABLE_CUSTOM_COMPILE_CONFIG</span>, default `False (0)`, Forces disabling cache-dit's custom `torch.compile` configurations when set to 1 (by default, custom configs are used for better performance).
- <span style="color:#c77dff;">CACHE_DIT_PATCH_FUNCTOR_DISABLE_DIFFUSERS_CHECK</span>, default `False (0)`, Disables the check for whether the model originates from the diffusers library in patch functors when set to 1.
- <span style="color:#c77dff;">CACHE_DIT_FORCE_ONLY_RANK0_LOGGING</span>, default `True (1)`, Forces only rank 0 to output logs (recommended for distributed training to avoid cluttered logs). Set to 0 to allow logging from all ranks.
- <span style="color:#c77dff;">CACHE_DIT_ENABLE_LOGGERS_SUPPRESS</span>, default `False (0)`, Force enable loggers suppress in cache-dit by setting the environment variable to 1. By default, cache-dit DON'T suppresses some noisy loggers. Users can set this variable to 1 to suppress these loggers globally, which is recommended for better log readability.
- <span style="color:#c77dff;">CACHE_DIT_DISABLE_EXCLUDE_FOR_QUANTIZE_AFTER_TP</span>, default `False (0)`, For **developer use only** – controls whether to disable the temporary workaround of excluding some layers from quantization after applying tensor parallelism. Users should NOT use this variable directly.

## Key Notes

1. Boolean-type variables are parsed from integer values: `1` = `True` (enabled), `0` = `False` (disabled).

2. Variables marked "internal use only" or "developer use only" should not be modified by end users.

3. Ulysses variants such as `ulysses_anything`, `ulysses_float8`, and `ulysses_async` are configured through `ParallelismConfig` and internal `_ContextParallelConfig`, not environment variables.
