# API Documentation

<div id="api-docs"></div>  

Unified Cache API for almost Any Diffusion Transformers (with Transformer Blocks that match the specific Input and Output patterns). For a good balance between performance and precision, DBCache is configured by default with F8B0, 8 warmup steps, and unlimited cached steps. All the configurable params are listed beflows.

## API: enable_cache

```python
def enable_cache(...) -> Union[DiffusionPipeline, BlockAdapter, Transformer]
```

## Function Description

The `enable_cache` function serves as a unified caching interface designed to optimize the performance of diffusion transformer models by implementing an intelligent caching mechanism known as `DBCache`. This API is engineered to be compatible with nearly `all` diffusion transformer architectures that feature transformer blocks adhering to standard input-output patterns, eliminating the need for architecture-specific modifications.  

By strategically caching intermediate outputs of transformer blocks during the diffusion process, `DBCache` significantly reduces redundant computations without compromising generation quality. The caching mechanism works by tracking residual differences between consecutive steps, allowing the model to reuse previously computed features when these differences fall below a configurable threshold. This approach maintains a balance between computational efficiency and output precision.  

The default configuration (`F8B0, 8 warmup steps, unlimited cached steps`) is carefully tuned to provide an optimal tradeoff for most common use cases. The "F8B0" configuration indicates that the first 8 transformer blocks are used to compute stable feature differences, while no final blocks are employed for additional fusion. The warmup phase ensures the model establishes sufficient feature representation before caching begins, preventing potential degradation of output quality.  

This function seamlessly integrates with both standard diffusion pipelines and custom block adapters, making it versatile for various deployment scenarios—from research prototyping to production environments where inference speed is critical. By abstracting the complexity of caching logic behind a simple interface, it enables developers to enhance model performance with minimal code changes.

## Quick Start

```python
>>> import cache_dit
>>> from diffusers import DiffusionPipeline
>>> pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image") # Can be any diffusion pipeline
>>> cache_dit.enable_cache(pipe) # One-line code with default cache options.
>>> output = pipe(...) # Just call the pipe as normal.
>>> stats = cache_dit.summary(pipe) # Then, get the summary of cache acceleration stats.
>>> cache_dit.disable_cache(pipe) # Disable cache and run original pipe.
```

## Parameter Description

- **pipe_or_adapter**(`DiffusionPipeline`, `BlockAdapter` or `Transformer`, *required*):  
  The standard Diffusion Pipeline or custom BlockAdapter (from cache-dit or user-defined).
  For example: `cache_dit.enable_cache(FluxPipeline(...))`.

- **cache_config**(`DBCacheConfig`, *required*, defaults to DBCacheConfig()):  
  Basic DBCache config for cache context, defaults to DBCacheConfig(). The configurable parameters are listed below:
  - `Fn_compute_blocks`: (`int`, *required*, defaults to 8):  
    Specifies that `DBCache` uses the**first n**Transformer blocks to fit the information at time step t, enabling the calculation of a more stable L1 difference and delivering more accurate information to subsequent blocks.
    Please check https://github.com/vipshop/cache-dit/blob/main/docs/DBCache.md for more details of DBCache.
  - `Bn_compute_blocks`: (`int`, *required*, defaults to 0):  
    Further fuses approximate information in the**last n**Transformer blocks to enhance prediction accuracy. These blocks act as an auto-scaler for approximate hidden states that use residual cache.
  - `residual_diff_threshold`: (`float`, *required*, defaults to 0.08):  
    The value of residual difference threshold, a higher value leads to faster performance at the cost of lower precision.
  - `max_accumulated_residual_diff_threshold`: (`float`, *optional*, defaults to None):  
    The maximum accumulated relative l1 diff threshold for Cache. If set, when the
    accumulated relative l1 diff exceeds this threshold, the caching strategy will be
    disabled for current step. This is useful for some cases where the input condition
    changes significantly in a single step. Default None means this feature is disabled.  
  - `max_warmup_steps`: (`int`, *required*, defaults to 8):  
    DBCache does not apply the caching strategy when the number of running steps is less than or equal to this value, ensuring the model sufficiently learns basic features during warmup.
  - `warmup_interval`: (`int`, *required*, defaults to 1):    
    Skip interval in warmup steps, e.g., when warmup_interval is 2, only 0, 2, 4, ... steps
    in warmup steps will be computed, others will use dynamic cache.
  - `max_cached_steps`: (`int`, *required*, defaults to -1):  
    DBCache disables the caching strategy when the previous cached steps exceed this value to prevent precision degradation.
  - `max_continuous_cached_steps`: (`int`, *required*, defaults to -1):  
    DBCache disables the caching strategy when the previous continuous cached steps exceed this value to prevent precision degradation.
  - `enable_separate_cfg`: (`bool`, *required*, defaults to None):  
    Whether to use separate cfg or not, such as in Wan 2.1, Qwen-Image. For models that fuse CFG and non-CFG into a single forward step, set enable_separate_cfg as False. Examples include: CogVideoX, HunyuanVideo, Mochi, etc.
  - `cfg_compute_first`: (`bool`, *required*, defaults to False):    
    Whether to compute cfg forward first, default is False, meaning:  
    0, 2, 4, ... -> non-CFG step; 1, 3, 5, ... -> CFG step.
  - `cfg_diff_compute_separate`: (`bool`, *required*, defaults to True):    
    Whether to compute separate difference values for CFG and non-CFG steps, default is True. If False, we will use the computed difference from the current non-CFG transformer step for the current CFG step.
  - `num_inference_steps` (`int`, *optional*, defaults to None):  
    num_inference_steps for DiffusionPipeline, used to adjust some internal settings
    for better caching performance. For example, we will refresh the cache once the
    executed steps exceed num_inference_steps if num_inference_steps is provided.
  - `steps_computation_mask`: (`List[int]`, *optional*, defaults to None):  
    This param introduce LeMiCa/EasyCache style compute mask for steps. It is a list
    of length num_inference_steps indicating whether to compute each step or not.
    1 means must compute, 0 means use dynamic/static cache. If provided, will override
    other settings to decide whether to compute each step.  
  - `steps_computation_policy`: (`str`, *optional*, defaults to "dynamic"):  
    The computation policy for steps when using steps_computation_mask. It can be
    "dynamic" or "static". "dynamic" means using dynamic cache for steps marked as 0
    in steps_computation_mask, while "static" means using static cache for those steps.

- **calibrator_config** (`CalibratorConfig`, *optional*, defaults to None):  
  Config for calibrator. If calibrator_config is not None, it means the user wants to use DBCache with a specific calibrator, such as taylorseer, foca, and so on.

- **params_modifiers** ('ParamsModifier', *optional*, defaults to None):  
  Modify cache context parameters for specific blocks. The configurable parameters are listed below:
  - `cache_config`: (`DBCacheConfig`, *required*, defaults to DBCacheConfig()):  
    The same as the 'cache_config' parameter in the cache_dit.enable_cache() interface.
  - `calibrator_config`: (`CalibratorConfig`, *optional*, defaults to None):  
    The same as the 'calibrator_config' parameter in the cache_dit.enable_cache() interface.
  - `**kwargs`: (`dict`, *optional*, defaults to {}):  
    The same as the 'kwargs' parameter in the cache_dit.enable_cache() interface.

- **parallelism_config** (`ParallelismConfig`, *optional*, defaults to None):  
    Config for Parallelism. If parallelism_config is not None, it means the user wants to enable
    parallelism for cache-dit.
    - `backend`: (`ParallelismBackend`, *required*, defaults to "ParallelismBackend.NATIVE_DIFFUSER"):  
        Parallelism backend, currently only NATIVE_DIFFUSER and NVTIVE_PYTORCH are supported.
        For context parallelism, only NATIVE_DIFFUSER backend is supported, for tensor parallelism,
        only NATIVE_PYTORCH backend is supported.
    - `ulysses_size`: (`int`, *optional*, defaults to None):  
        The size of Ulysses cluster. If ulysses_size is not None, enable Ulysses style parallelism.
        This setting is only valid when backend is NATIVE_DIFFUSER.
    - `ring_size`: (`int`, *optional*, defaults to None):  
        The size of ring for ring parallelism. If ring_size is not None, enable ring attention.
        This setting is only valid when backend is NATIVE_DIFFUSER.
    - `tp_size`: (`int`, *optional*, defaults to None):  
        The size of tensor parallelism. If tp_size is not None, enable tensor parallelism.
        This setting is only valid when backend is NATIVE_PYTORCH.
    - `parallel_kwargs` (`dict`, *optional*):  
       Additional kwargs for parallelism backends. For example, for NATIVE_DIFFUSER backend, it can include:
       - `cp_plan`: The custom context parallelism plan pass by user.
       - `attention_backend`: str, The attention backend for parallel attention, e.g, 'native', 'flash', 'sage', etc.
       - `experimental_ulysses_anything: bool, Whether to enable the ulysses anything attention to support arbitrary sequence length and arbitrary number of heads.
       - `experimental_ulysses_async: bool, Whether to enable the ulysses async attention to overlap communication and computation.
       - `experimental_ulysses_float8: bool, Whether to enable the ulysses float8 attention to use fp8 for faster communication.
       - `ring_rotate_method`: str, The ring rotate method, default is `p2p`:   
          - `p2p`: Use batch_isend_irecv ops to rotate the key and value tensors. This method is more efficient due to th better overlap of communication and computation (default).
          - `allgather`: Use allgather to gather the key and value tensors.
       - `ring_convert_to_fp32`: bool, Whether to convert the value output and lse of ring attention to fp32. Default to True to avoid numerical issues.
        
- **attention_backend** (`str`, *optional*, defaults to None):  
  Custom attention backend in cache-dit for non-parallelism case. If attention_backend is 
  not None, set the attention backend for the transformer module. Supported backends include: 
  "native", "_sdpa_cudnn", "sage", "flash", "flash", "_native_npu", etc. Prefer attention_backend
  in parallelism_config when both are provided.

- **kwargs** (`dict`, *optional*, defaults to {}):   
  Other cache context keyword arguments. Please check https://github.com/vipshop/cache-dit/blob/main/src/cache_dit/caching/cache_contexts/cache_context.py for more details.
