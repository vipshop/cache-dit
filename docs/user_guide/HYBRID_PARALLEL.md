# Hybrid 2D/3D Parallelism

## Overviews

cache-dit fully supports hybrid Context Parallelism (including USP) and Tensor Parallelism (namely, <span style="color:hotpink;">2D or 3D Parallelism</span>). Thus, it can <span style="color:hotpink;">scale up the performance of large DiT</span> models such as FLUX.2 (**112 GiB**❗️❗️ total), Qwen-Image (56 GiB total) and LTX-2 (84 GiB total) on low-VRAM devices (e.g., NVIDIA L20, A30, H20, A800, H800, ..., **<96 GiB**❗️❗️) with no precision loss. `Hybrid CP (USP) + TP` is faster than vanilla Tensor Parallelism and fully compatible with TE-P, VAE-P and Cache acceleration. 

### Image Generation 

From the table below (Image Generation: FLUX.2-dev, **112 GiB**❗️❗️), it is clear that `Ulysses-4-TP-2` delivers higher throughput than `TP-8`. This allows it to better scale the performance of FLUX.2-dev on an 8×L20 (<48 GiB) GPU node. (Note: The text encoder is always be parallelized; GiB = GiB per GPU; USP = Ulysses + Ring)

|TP-2|Ring-2/4/8|Ulysses-2/4/8|TP-4|TP-8|<span style="color:hotpink;">Ulysses-4-TP-2</span>|
|:---:|:---:|:---:|:---:|:---:|:---:|
|OOM❗️|OOM❗️|OOM❗️|32.40GiB|19.92GiB|41.85GiB|
|OOM❗️|OOM❗️|OOM❗️|27.72s|21.37s|🎉**15.21s**|
|Ulysses-2-TP-4|Ring-4-TP-2|Ring-2-TP-4|USP-2-2-TP-2|<span style="color:hotpink;">Ulysses-2-TP-4 + Cache</span>|<span style="color:hotpink;">Ulysses-4-TP-2 + Cache</span>|
|27.23GiB|41.85GiB|27.23GiB|41.85GiB|27.33GiB|41.90GiB|
|17.98s|17.37s|17.13s|16.06s|🎉**9.00s**|🎉**7.73s**|  

### Video Generation 

From the table below (Video Generation: LTX-2, **84 GiB**❗️❗️), it is clear that `Ulysses-2-TP-2` delivers higher throughput than `TP-4`. This also shows that hybrid `CP(USP) + TP` allows the better scaling of the performance for LTX-2 on 4×L20 (<48 GiB). (Note: The text encoder is always be parallelized; GiB = GiB per GPU; TP-2, Ring-2/4/8, Ulysses-2/4/8: OOM❗️)

|LTX-2, L20, TP-4|LTX-2, L20, <span style="color:hotpink;">Ulysses-2-TP-2</span>|
|:---:|:---:|  
|26.75GiB|35.38GiB|
|143.49s|🎉**110.95s**|
|<img src="https://github.com/vipshop/cache-dit/raw/main/docs/assets/ltx2_t2v.512x768x121.C0_Q0_NONE_TP4_TEP_VAEP.gif" >|<img src="https://github.com/vipshop/cache-dit/raw/main/docs/assets/ltx2_t2v.512x768x121.C0_Q0_NONE_Ulysses2_TP2_TEP_VAEP_ulysses_anything.gif" >|


## 2D, 3D, 5D Parallelism and Cache

Users can set both **ulysses_size/ring_size(CP, USP)** and **tp_size(TP)** to values greater than 1 to enable hybrid **2D** or **complex 3D** parallelism for the DiT transformer module. The <span style="color:hotpink;">**2D/3D**</span> hybrid parallelism for the Transformer module in cache-dit is fully compatible with Text Encoder Parallelism (<span style="color:hotpink;">**TE-P**</span>), Autoencoder Parallelism (<span style="color:hotpink;">**VAE-P**</span>) and Cache acceleration. Thus, you can combine all these parallelism mechanisms to construct a sophisticated <span style="color:hotpink;">**5D**</span> parallelism + <span style="color:hotpink;">**Cache**</span> architecture for <span style="color:hotpink;">**large-scale DiTs**</span> to achieve the best performance on low-VRAM devices.

- 🎉2D Transformer Parallelism: <span style="color:hotpink;">Ulysses + TP</span>

```python
from cache_dit import ParallelismConfig

cache_dit.enable_cache(
    pipe_or_adapter, 
    parallelism_config=ParallelismConfig(
        ulysses_size=4, tp_size=2,
    ),
)
```

- 🎉2D Transformer Parallelism: <span style="color:hotpink;">Ring + TP</span>

```python
from cache_dit import ParallelismConfig

cache_dit.enable_cache(
    pipe_or_adapter, 
    parallelism_config=ParallelismConfig(
        ring_size=4, tp_size=2,
    ),
)
```

- 🎉3D Transformer Parallelism: <span style="color:hotpink;">USP + TP</span>

```python
from cache_dit import ParallelismConfig

cache_dit.enable_cache(
    pipe_or_adapter, 
    parallelism_config=ParallelismConfig(
        ulysses_size=2, ring_size=2, tp_size=2,
    ),
)
```

- 🎉5D Parallelism + Cache: <span style="color:hotpink;">2D/3D</span> Transformer Parallelsim + <span style="color:hotpink;">TE-P</span> + <span style="color:hotpink;">VAE-P</span> + <span style="color:hotpink;">Cache</span>

```python
from cache_dit import DBCacheConfig, ParallelismConfig

cache_dit.enable_cache(
    pipe_or_adapter, 
    cache_config=DBCacheConfig(...), # w/ Cache
    parallelism_config=ParallelismConfig(
        # ulysses_size=2, ring_size=2, tp_size=2, # 3D Parallelism
        ulysses_size=4, tp_size=2, # or, 2D Parallelsim
        # e.g, FLUX.2, we can also parallelize the Text Encoder and VAE
        # module to further reduce the memory usage on low-VRAM devices.
        extra_parallel_modules=[
            pipe.text_encoder, 
            pipe.vae,
        ],
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
