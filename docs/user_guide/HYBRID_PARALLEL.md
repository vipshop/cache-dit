# Hybrid CP (USP) + TP

cache-dit fully supports hybrid Context Parallelism (including USP) and Tensor Parallelism. Thus, it can scale up the performance of large DiT models such as FLUX.2 (112 GiB total) and Qwen-Image (56 GiB total) on low-VRAM devices (e.g., NVIDIA L20/A30, < 48 GiB) with no precision loss. `Hybrid CP (USP) + TP` is faster than vanilla Tensor Parallelism.

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
    ),
)

# Ring + TP
cache_dit.enable_cache(
    pipe_or_adapter, 
    cache_config=DBCacheConfig(...),
    parallelism_config=ParallelismConfig(
        ring_size=4, 
        tp_size=2,
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
    ),
)

# torchrun --nproc_per_node=8 parallel_hybrid.py
```
