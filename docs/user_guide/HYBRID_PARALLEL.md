# Hybrid CP(USP) and TP

cache-dit fully supports hybrid Context Parallelism (including USP) and Tensor Parallelism. Thus, it can scale up the performance of large DiT models such as FLUX.2 (112 GiB total) and Qwen-Image (56 GiB total) on low-VRAM devices (e.g., NVIDIA L20/A30, < 48 GiB) with no precision loss. `Hybrid CP (USP) + TP` is faster than vanilla Tensor Parallelism. For examples:

```python
# pip3 install "cache-dit[parallelism]"
from cache_dit import ParallelismConfig

# Ulysses + TP
cache_dit.enable_cache(
    pipe_or_adapter, 
    cache_config=DBCacheConfig(...),
    parallelism_config=ParallelismConfig(
        ulysses_size=4, 
        tp_size=2,
        parallel_kwargs={
            "extra_parallel_modules": [pipe.text_encoder], # FLUX.2
        },
    ),
)

# Ring + TP
cache_dit.enable_cache(
    pipe_or_adapter, 
    cache_config=DBCacheConfig(...),
    parallelism_config=ParallelismConfig(
        ring_size=4, 
        tp_size=2,
        parallel_kwargs={
            "extra_parallel_modules": [pipe.text_encoder], # FLUX.2
        },
    ),
)

# USP + TP
cache_dit.enable_cache(
    pipe_or_adapter, 
    cache_config=DBCacheConfig(...),
    parallelism_config=ParallelismConfig(
        ulysses_size=2,
        ring_size=2, 
        tp_size=2,
        parallel_kwargs={
            "extra_parallel_modules": [pipe.text_encoder], # FLUX.2
        },
    ),
)

# torchrun --nproc_per_node=8 parallel_hybrid.py
```

From the table below (FLUX.2-dev, 112 GiB total), it is clear that `Ulysses-4-TP-2` delivers higher throughput than `TP-8`. This allows it to better scale the performance of FLUX.2-dev on an 8×L20 (<48 GiB) GPU node. (Note: The text encoder is always be parallelized; GiB = GiB per GPU; CP = Ulysses or Ring; USP = Ulysses + Ring)

|TP-2|CP-2/4/8|TP-4|TP-8|Ulysses-4-TP-2|Ulysses-2-TP-4|Ring-4-TP-2|Ring-2-TP-4|USP-2-2-TP-2|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|OOM|OOM|32.40GiB|19.92GiB|41.85GiB|27.23GiB|41.85GiB|27.23GiB|41.85GiB|
|OOM|OOM|27.72s|21.37s|15.21s|17.98s|17.37s|17.13s|16.06s|
