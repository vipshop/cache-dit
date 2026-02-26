# Configurable Environment Variables

This document summarizes all core configurable environment variables in cache-dit, which control key functionalities such as logging behavior, parallel computing strategies, model adaptation, compilation optimization, and patch tool logic.

## Environment Variables

|Environment Variable Name|Type|Default Value|Description|
|---|---|---|---|
|CACHE_DIT_LOG_LEVEL|str|`"info"`|Controls the logging level of cache-dit.|
|CACHE_DIT_LOG_DIR|str|`None`|Specifies the directory for cache-dit log files (if not set, logs are output to console by default).|
|CACHE_DIT_ENABLE_CUSTOM_ATTN_DISPATCH|bool|`True` (1)|Enables custom attention backend dispatch for context parallelism. Enabled by default for better compatibility and performance; set to 0 to disable this behavior.|
|CACHE_DIT_ENABLE_CUSTOM_ATTN_ALREADY_DISPATCH|bool|`False` (0)|For **internal use only** – avoids re-registering the custom attention backend dispatch. Users should NOT set this variable directly.|
|CACHE_DIT_ENABLE_ULYSSES_ANYTHING|bool|`False` (0)|Enables Ulysses Anything Attention when set to 1. Alternative configuration: use the `experimental_ulysses_anything` argument in `ContextParallelism`.|
|CACHE_DIT_ENABLE_ULYSSES_ANYTHING_FLOAT8|bool|`False` (0)|Enables Ulysses Anything Attention Float8 when set to 1. Alternative configuration: use `experimental_ulysses_anything=True` and `experimental_ulysses_float=True` in `ContextParallelism`.|
|CACHE_DIT_ENABLE_ULYSSES_FLOAT8|bool|`False` (0)|Enables Ulysses Attention Float8 when set to 1. Alternative configuration: use the `experimental_ulysses_float8` argument in `ContextParallelism`.|
|CACHE_DIT_UNEVEN_HEADS_COMM_NO_PAD|bool|`False` (0)|Enables unpadded communication for uneven attention heads (avoids padding overhead) when set to 1.|
|CACHE_DIT_FLUX_ENABLE_DUMMY_BLOCKS|bool|`True` (1)|For **developer use only** – controls whether dummy blocks are enabled for FLUX models (enabled by default). Users should NOT use this variable directly.|
|CACHE_DIT_EPILOGUE_PROLOGUE_FUSION|bool|`False` (0)|Enables epilogue and prologue fusion in cache-dit's torch.compile optimizations.|
|CACHE_DIT_ENABLE_COMPILE_COMPUTE_COMM_OVERLAP|bool|`True` (1)|Enables compute-communication (all-reduce) overlap during cache-dit compilation. Enabled by default for better performance; set to 0 to disable.|
|CACHE_DIT_FORCE_DISABLE_CUSTOM_COMPILE_CONFIG|bool|`False` (0)|Forces disabling cache-dit's custom `torch.compile` configurations when set to 1 (by default, custom configs are used for better performance).|
|CACHE_DIT_PATCH_FUNCTOR_DISABLE_DIFFUSERS_CHECK|bool|`False` (0)|Disables the check for whether the model originates from the diffusers library in patch functors when set to 1.|
|CACHE_DIT_FORCE_ONLY_RANK0_LOGGING|bool|`True` (1)|Forces only rank 0 to output logs (recommended for distributed training to avoid cluttered logs). Set to 0 to allow logging from all ranks.|

### Key Notes

1. Boolean-type variables are parsed from integer values: `1` = `True` (enabled), `0` = `False` (disabled).

2. Variables marked "internal use only" or "developer use only" should not be modified by end users.

3. Most variables have alternative configuration via code arguments (e.g., `ContextParallelism` parameters) besides environment variables.
