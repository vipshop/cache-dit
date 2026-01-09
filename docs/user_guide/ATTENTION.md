# Attention Backend

## Available backend

Cache-DiT supports multiple Attention backends for better performance. The supported list is as follows:

|backend|details|context/tensor parallel|attn_mask|
|:---:|:---:|:---:|:---:|    
|native| Native SDPA Attention, w/ cache-dit optimized|✅|✅|  
|_sdpa_cudnn| CUDNN Attention via SDPA API, w/ cache-dit optimized|✅|✅|
|_native_cudnn| CUDNN Attention via SDPA API, w/o cache-dit optimized|✅|✖️|
|flash| official FlashAttention-2|✅|✖️| 
|_flash_3| official FlashAttention-3|✅|✖️|
|sage| FP8 SageAttention|✅|✖️|
|_native_npu| Ascend NPU Attention|✅|✅|

Users can specify Attention backend by setting the attention_backend parameter of parallel_kwargs:

```python
from cache_dit import ParallelismConfig

cache_dit.enable_cache(
    pipe_or_adapter, 
    cache_config=DBCacheConfig(...),
    parallelism_config=ParallelismConfig(
        ulysses_size=2, # or, tp_size=2
        parallel_kwargs={
            # flash, native(sdpa), _native_cudnn, _sdpa_cudnn, sage
            "attention_backend": "_sdpa_cudnn",
        },
    ),
)
```

## FP8 Attention

<div id="fp8-attention"></div>

For FP8 Attention, users must install `sage-attention`. Then, pass the `sage` attention backend to the parallelism configuration as an extra parameter. Please note that `attention mask` is not currently supported for FP8 sage attention.

```python
# pip3 install "cache-dit[parallelism]"
# pip3 install git+https://github.com/thu-ml/SageAttention.git 
from cache_dit import ParallelismConfig

cache_dit.enable_cache(
    pipe_or_adapter, 
    cache_config=DBCacheConfig(...),
    parallelism_config=ParallelismConfig(
        ulysses_size=2, # or, tp_size=2
        parallel_kwargs={
            # flash, native(sdpa), _native_cudnn, _sdpa_cudnn, sage
            "attention_backend": "sage",
        },
    ),
)
# torchrun --nproc_per_node=2 parallel_fp8_cache.py
```
