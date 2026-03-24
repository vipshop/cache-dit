# Attention Backend

## Available backend

Cache-DiT supports multiple Attention backends for better performance. The supported list is as follows:

|backend|details|parallelism|attn_mask|
|:---|:---|:---|:---|    
|<span style="color:#c77dff;">native</span>| Native SDPA Attention, w/ cache-dit optimized|✅|✅|  
|<span style="color:#c77dff;">_sdpa_cudnn</span>| CUDNN Attention via SDPA API, w/ cache-dit optimized|✅|✅|
|<span style="color:#c77dff;">_native_cudnn</span>| CUDNN Attention via SDPA API, w/o cache-dit optimized|✅|✖️|
|<span style="color:#c77dff;">flash</span>| official FlashAttention-2|✅|✖️| 
|<span style="color:#c77dff;">_flash_3</span>| official FlashAttention-3|✅|✖️|
|<span style="color:#c77dff;">sage</span>| FP8 SageAttention|✅|✖️|
|<span style="color:#c77dff;">_native_npu</span>| Optimized Ascend NPU Attention|✅|✅|
|<span style="color:#c77dff;">_npu_fia</span>| NPU Attention for Ring Parallelism|✅|✅|


## Single GPU Inference

Users can specify Attention backend by setting the <span style="color:#c77dff;">attention_backend</span> parameter of <span style="color:#c77dff;">enable_cache</span> API or use <span style="color:#c77dff;">set_attn_backend</span> interface directly.  

```python
import cache_dit

# Setting the `attention_backend` parameter of `enable_cache` API
cache_dit.enable_cache(pipe_or_adapter, ..., attention_backend="_sdpa_cudnn")
# Or, use `set_attn_backend` interface directly.  
cache_dit.set_attn_backend(pipe_or_adapter, attention_backend="_sdpa_cudnn")
```


## Distributed inference

Users also can specify Attention backend by setting the <span style="color:#c77dff;">attention_backend</span> parameter of <span style="color:#c77dff;">parallelism_config</span> in the cases of distributed inference:

```python
from cache_dit import ParallelismConfig

cache_dit.enable_cache(
    pipe_or_adapter, 
    cache_config=DBCacheConfig(...),
    parallelism_config=ParallelismConfig(
        ulysses_size=2, # or, tp_size=2
        # flash, native(sdpa), _native_cudnn, _sdpa_cudnn, sage
        attention_backend="_sdpa_cudnn",
    ),
)
```

## FP8 Attention

<div id="fp8-attention"></div>

For FP8 Attention, users must install `sage-attention`. Then, pass the <span style="color:#c77dff;">sage</span> attention backend to the <span style="color:#c77dff;">parallelism_config</span> as an extra parameter. Please note that <span style="color:#c77dff;">attention mask</span> is not currently supported for FP8 sage attention.

```python
# pip3 install git+https://github.com/thu-ml/SageAttention.git 
from cache_dit import ParallelismConfig

cache_dit.enable_cache(
    pipe_or_adapter, 
    cache_config=DBCacheConfig(...),
    parallelism_config=ParallelismConfig(
        ulysses_size=2, # or, tp_size=2
        attention_backend="sage",
    ),
)
```
