# Hybrid 2D/3D/5D Parallelism

## Overviews

cache-dit fully supports hybrid Context Parallelism (including USP) and Tensor Parallelism (namely, 2D or 3D Parallelism). Thus, it can scale up the performance of large DiT models such as FLUX.2 (**112 GiB**❗️❗️ total), Qwen-Image (56 GiB total) and LTX-2 (84 GiB total) on low-VRAM devices (e.g., NVIDIA L20, A30, H20, A800, H800, ..., **<96 GiB**❗️❗️) with no precision loss. `Hybrid CP (USP) + TP` is faster than vanilla Tensor Parallelism and compatible with cache. 

From the table below (FLUX.2-dev, **112 GiB**❗️❗️), it is clear that `Ulysses-4-TP-2` delivers higher throughput than `TP-8`. This allows it to better scale the performance of FLUX.2-dev on an 8×L20 (<48 GiB) GPU node. (Note: The text encoder is always be parallelized; GiB = GiB per GPU; USP = Ulysses + Ring)

|TP-2|Ring-2/4/8|Ulysses-2/4/8|TP-4|TP-8|Ulysses-4-TP-2|
|:---:|:---:|:---:|:---:|:---:|:---:|
|OOM|OOM|OOM|32.40GiB|19.92GiB|41.85GiB|
|OOM|OOM|OOM|27.72s|21.37s|🎉**15.21s**|
|Ulysses-2-TP-4|Ring-4-TP-2|Ring-2-TP-4|USP-2-2-TP-2|Ulysses-2-TP-4 + Cache|Ulysses-4-TP-2 + Cache|
|27.23GiB|41.85GiB|27.23GiB|41.85GiB|27.33GiB|41.90GiB|
|17.98s|17.37s|17.13s|16.06s|🎉**9.00s**|🎉**7.73s**|


## 2D, 3D and 5D Parallelism

Users can set both `ulysses_size/ring_size(CP, USP)` and `tp_size(TP)` to values greater than 1 to enable hybrid **2D** or **complex 3D** parallelism for the DiT transformer module. The **2D/3D** hybrid parallelism for the Transformer module in cache-dit is fully compatible with Text Encoder Parallelism (**TE-P**) and Autoencoder Parallelism (**VAE-P**). Thus, you can combine all these parallelism mechanisms to construct a sophisticated **5D** parallelism architecture for **large-scale DiTs**!

- 2D Transformer Parallelism: Ulysses + TP

```python
from cache_dit import ParallelismConfig

# 2D Parallelism: Ulysses + TP
cache_dit.enable_cache(
    pipe_or_adapter, 
    cache_config=DBCacheConfig(...),
    parallelism_config=ParallelismConfig(
        ulysses_size=4, tp_size=2,
    ),
)
```

- 2D Transformer Parallelism: Ring + TP

```python
from cache_dit import ParallelismConfig

# 2D Parallelism: Ring + TP
cache_dit.enable_cache(
    pipe_or_adapter, 
    cache_config=DBCacheConfig(...),
    parallelism_config=ParallelismConfig(
        ring_size=4, tp_size=2,
    ),
)
```

- 3D Transformer Parallelism: USP + TP

```python
from cache_dit import ParallelismConfig

# 3D Parallelism: USP + TP
cache_dit.enable_cache(
    pipe_or_adapter, 
    cache_config=DBCacheConfig(...),
    parallelism_config=ParallelismConfig(
        ulysses_size=2, ring_size=2, tp_size=2,
    ),
)
```

- 5D Parallelism: 2D/3D Transformer Parallelsim + TE-P + VAE-P

```python
from cache_dit import ParallelismConfig

# 2D/3D Parallelism + TE-P
cache_dit.enable_cache(
    pipe_or_adapter, 
    cache_config=DBCacheConfig(...),
    parallelism_config=ParallelismConfig(
        # ulysses_size=2, ring_size=2, tp_size=2, # 3D Parallelism
        ulysses_size=4, tp_size=2, # or, 2D Parallelsim
        # e.g, FLUX.2, we can also parallize the Text Encoder and VAE
        # module to further reduce the memory usage on low-VRAM devices.
        parallel_kwargs={
            "extra_parallel_modules": [pipe.text_encoder, pipe.vae], 
        },
    ),
)
```

## Quick Examples

```bash
torchrun --nproc_per_node=4 -m cache_dit.generate flux2 --parallel tp --parallel-text --track-memory
torchrun --nproc_per_node=8 -m cache_dit.generate flux2 --parallel tp --parallel-text --track-memory
torchrun --nproc_per_node=8 -m cache_dit.generate flux2 --parallel tp_ulysses --parallel-text --track-memory
torchrun --nproc_per_node=8 -m cache_dit.generate flux2 --parallel ulysses_tp --parallel-text --track-memory
torchrun --nproc_per_node=8 -m cache_dit.generate flux2 --parallel ring_tp --parallel-text --track-memory
torchrun --nproc_per_node=8 -m cache_dit.generate flux2 --parallel tp_ring --parallel-text --track-memory
torchrun --nproc_per_node=8 -m cache_dit.generate flux2 --parallel usp_tp --parallel-text --track-memory
torchrun --nproc_per_node=8 -m cache_dit.generate flux2 --parallel tp_ulysses --parallel-text --track-memory --cache
torchrun --nproc_per_node=8 -m cache_dit.generate flux2 --parallel ulysses_tp --parallel-text --track-memory --cache
torchrun --nproc_per_node=8 -m cache_dit.generate flux2 --parallel usp_tp --parallel-text --track-memory --cache
```
