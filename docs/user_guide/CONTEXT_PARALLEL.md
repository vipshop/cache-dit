# Context Parallelism

## Hybrid Context Parallelism

<div id="context-parallelism"></div>

cache-dit is compatible with context parallelism. Currently, we support the use of `Hybrid Cache` + `Context Parallelism` scheme (via NATIVE_DIFFUSER parallelism backend) in cache-dit. Users can use Context Parallelism to further accelerate the speed of inference! For more details, please refer to [📚examples](https://github.com/vipshop/cache-dit/tree/main/examples). Currently, cache-dit supported context parallelism for [FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-dev), 🔥[FLUX.2](https://huggingface.co/black-forest-labs/FLUX.2-dev), [Qwen-Image](https://github.com/QwenLM/Qwen-Image), [Qwen-Image-Lightning](https://github.com/ModelTC/Qwen-Image-Lightning), [LTXVideo](https://huggingface.co/Lightricks/LTX-Video), [Wan 2.1](https://github.com/Wan-Video/Wan2.1), [Wan 2.2](https://github.com/Wan-Video/Wan2.2), [HunyuanImage-2.1](https://huggingface.co/tencent/HunyuanImage-2.1), [HunyuanVideo](https://huggingface.co/hunyuanvideo-community/HunyuanVideo), [CogVideoX 1.0](https://github.com/zai-org/CogVideo), [CogVideoX 1.5](https://github.com/zai-org/CogVideo), [CogView 3/4](https://github.com/zai-org/CogView4) and [VisualCloze](https://github.com/lzyhha/VisualCloze), etc. cache-dit will support more models in the future.

```python
# pip3 install "cache-dit[parallelism]"
from cache_dit import ParallelismConfig

cache_dit.enable_cache(
    pipe_or_adapter, 
    cache_config=DBCacheConfig(...),
    # Set ulysses_size > 1 to enable ulysses style context parallelism.
    parallelism_config=ParallelismConfig(ulysses_size=2),
)
# torchrun --nproc_per_node=2 parallel_cache.py
```

|L20x1| Ulysses-2 | Ulysses-4 | + compile |
|:---:|:---:|:---:|:---:|  
|FLUX, 23.56s| 13.80s | 8.28s | 7.27s |
|<img src="https://github.com/vipshop/cache-dit/raw/main/docs/assets/flux.1024x1024.C0_Q0_NONE.png" width=222px>|<img src="https://github.com/vipshop/cache-dit/raw/main/docs/assets/flux.1024x1024.C0_Q0_NONE_Ulysses2.png" width=222px>|<img src="https://github.com/vipshop/cache-dit/raw/main/docs/assets/flux.1024x1024.C0_Q0_NONE_Ulysses4.png" width=222px>|<img src="https://github.com/vipshop/cache-dit/raw/main/docs/assets/flux.1024x1024.C1_Q0_NONE_Ulysses4.png" width=222px>|

## UAA: Ulysses Anything Attention 

<div id="ulysses-anything-attention"></div>

✅**Any Sequence Length**: We have implemented the **[📚UAA: Ulysses Anything Attention](#uaa-ulysses-anything-attention)**: An Ulysses Attention that supports **arbitrary sequence length** with ✅**zero padding** and **nearly ✅zero theoretical communication overhead**. The default Ulysses Attention requires that the sequence len of hidden states **must be divisible by the number of devices**. This imposes **significant limitations** on the practical application of Ulysses.


```python
# pip3 install "cache-dit[parallelism]"
from cache_dit import ParallelismConfig

cache_dit.enable_cache(
    pipe_or_adapter, 
    cache_config=DBCacheConfig(...),
    # Set `experimental_ulysses_anything` as True to enable UAA
    parallelism_config=ParallelismConfig(
        ulysses_size=2,
        parallel_kwargs={
            "experimental_ulysses_anything": True
        },
    ),
)
# torchrun --nproc_per_node=2 parallel_cache_ulysses_anything.py
```

For example, in the T2I and I2V tasks, the length of prompts input by users is often variable, and it is difficult to ensure that this length is divisible by the number of devices. To address this issue, we have developed a **✅padding-free** Ulysses Attention (UAA) for **arbitrary sequence length**, which enhances the versatility of Ulysses.

```python
dist.init_process_group(backend="cpu:gloo,cuda:nccl")
```
Compared to Ulysses Attention, in **UAA**, we have only added an **extra all-gather** op for scalar types to gather the seq_len value of each rank. To avoid multiple forced CUDA sync caused by H2D and D2H transfers, please add the **✅gloo** backend in `init_process_group`. This will significantly reduce communication latency.

<p align="center">
    ✅<b>Any Sequence Length</b><br>
    U*: Ulysses Attention, <b>UAA: Ulysses Anything Attenton</b>, UAA*: UAA + Gloo, Device: NVIDIA L20<br>
    FLUX.1-Dev w/o CPU Offload, 28 steps; Qwen-Image w/ CPU Offload, 50 steps; Gloo: Extra All Gather w/ Gloo
</p>

|CP2 w/ U* |CP2 w/ UAA* | CP2 w/ UAA |  L20x1 | CP2 w/ UAA* | CP2 w/ U* |  L20x1 |  CP2 w/ UAA* | 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|FLUX, 13.87s|**🎉13.88s**|14.75s|23.25s| **🎉13.75s**|Qwen, 132s|181s|**🎉133s**|
|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/uaa/flux.C0_Q0_NONE_Ulysses2.png" width=110px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/uaa/flux.C0_Q0_NONE_Ulysses2_ulysses_anything.png" width=110px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/uaa/flux.C0_Q0_NONE_Ulysses2_ulysses_anything.png" width=110px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/uaa/flux.1008x1008.C0_Q0_NONE.png" width=110px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets//uaa/flux.1008x1008.C0_Q0_NONE_Ulysses2_ulysses_anything.png" width=110px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/uaa/qwen-image.1312x1312.C0_Q0_NONE_Ulysses2.png" width=110px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/uaa/qwen-image.1328x1328.C0_Q0_NONE.png" width=110px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/uaa/qwen-image.1328x1328.C0_Q0_NONE_Ulysses2_ulysses_anything.png" width=110px>|
|1024x1024|1024x1024|1024x1024|1008x1008|1008x1008|1312x1312|1328x1328|1328x1328|
|✔️U* ✔️UAA|✔️U* ✔️UAA|✔️U* ✔️UAA| NO CP|❌U* ✔️UAA|✔️U* ✔️UAA|NO CP|❌U* ✔️UAA|


✅**Any Head Num**: By the way, Ulysses Attention and UAA in cache-dit now **support arbitrary numbers of heads** via additional padding and unpadding operations implemented before and after all-to-all. The overhead incurred by these extra padding and unpadding steps can be **partially hidden** through asynchronous communication. This support for arbitrary head counts is **automatically activated** whenever the number of heads is not divisible by the world size. For Example: 


<p align="center">
    ✅<b>Any Head Num</b><br>
    Ulysses: Ulysses Attention, <b>FP8 Ulysses: Ulysses w/ FP8 All2All</b>, Device: NVIDIA L20<br>
    🔥<b>Z-Image</b> (Head=30, ❌<b>CAN NOT</b> divisible by 4), 1024x1024, 9 steps.
</p>

|Ulysses 2, L20|Ulysses 4|FP8 Ulysses 4| + Cache | + FP8 DQ | 
|:---:|:---:|:---:|:---:|:---:|    
|1024x1024, 3.19s|1024x1024, 1.98s|1024x1024, 1.89s|1024x1024, 1.63s|1024x1024, 1.23s|    
|<img width="180" height="180" alt="zimage C1_Q0_NONE_Ulysses2_sdpa_cudnn" src="https://github.com/vipshop/cache-dit/raw/main/docs/assets/zimage.C1_Q0_NONE_Ulysses2_sdpa_cudnn.png" />|<img width="180" height="180" alt="zimage C1_Q0_NONE_Ulysses4_sdpa_cudnn" src="https://github.com/vipshop/cache-dit/raw/main/docs/assets/zimage.C1_Q0_NONE_Ulysses4_sdpa_cudnn.png" />|<img width="180" height="180" alt="zimage C1_Q0_NONE_Ulysses4_ulysses_float8_sdpa_cudnn" src="https://github.com/vipshop/cache-dit/raw/main/docs/assets/zimage.C1_Q0_NONE_Ulysses4_ulysses_float8_sdpa_cudnn.png" />|<img width="180" height="180" alt="zimage C1_Q0_DBCache_F1B0_W4I1M0MC0_R0 6_SCM111110101_dynamic_CFG0_T0O0_Ulysses4_S2_ulysses_float8_sdpa_cudnn" src="https://github.com/vipshop/cache-dit/raw/main/docs/assets/zimage.C1_Q0_DBCache_F1B0_W4I1M0MC0_R0.6_SCM111110101_dynamic_CFG0_T0O0_Ulysses4_S2_ulysses_float8_sdpa_cudnn.png" />|<img width="180" height="180" alt="zimage C1_Q1_float8_DBCache_F1B0_W4I1M0MC0_R0 6_SCM111110101_dynamic_CFG0_T0O0_Ulysses4_S2_ulysses_float8_sdpa_cudnn" src="https://github.com/vipshop/cache-dit/raw/main/docs/assets/zimage.C1_Q1_float8_DBCache_F1B0_W4I1M0MC0_R0.6_SCM111110101_dynamic_CFG0_T0O0_Ulysses4_S2_ulysses_float8_sdpa_cudnn.png" />|    


We have also implemented a ✅**padding-free** version that support any head num. Please be informed that this solution cannot be used when seq len is not divisible by world size. Users can enable this feature through environment variables:

```bash
export CACHE_DIT_UNEVEN_HEADS_COMM_NO_PAD=1 # NOT WORK if seq len is also not divisible by world size
```

Important: Please note that **Ulysses Anything Attention (UAA)** is currently an **experimental** feature. It has not undergone large-scale testing, and may introduce a slight performance degradation while the `cpu:gloo` commucation backend is not available.

## Async Ulysses QKV Projection

<div id="ulysses-async"></div>


![async_ulysses](https://github.com/vipshop/cache-dit/raw/main/assets/parallelism/async_ulysses.png)

Inspired by [ByteDance-Seed/VeOmni: Async Ulysses CP](https://github.com/ByteDance-Seed/VeOmni/blob/main/veomni/distributed/sequence_parallel/async_ulysses.py), we have also added support for **Async Ulysses QKV Projection** for certain models in cache-dit. This enables partial overlap of communication and computation, which can further enhance the performance of Ulysses style Context Parallelism. Currently, only the 🔥[FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-dev), 🔥[Qwen-Image](https://github.com/QwenLM/Qwen-Image), 🔥[Z-Image](https://github.com/Tongyi-MAI/Z-Image) and 🔥[Ovis-Image](https://github.com/AIDC-AI/Ovis-Image) models are supported, and more models will be added in the future—stay tuned!

```python
# pip3 install "cache-dit[parallelism]"
from cache_dit import ParallelismConfig

cache_dit.enable_cache(
    pipe_or_adapter, 
    cache_config=DBCacheConfig(...),
    # Set `experimental_ulysses_async` as True to enable Async Ulysses QKV Projection.
    parallelism_config=ParallelismConfig(
        ulysses_size=2,
        parallel_kwargs={
            "experimental_ulysses_async": True
        },
    ),
)
# torchrun --nproc_per_node=2 parallel_cache_ulysses_async.py
```


<p align="center">
    Ulysses: Standard Ulysses Attention, <b>Async Ulysses</b>: Ulysses Attenton with Async QKV Projection
</p>

|L20x2 w/ Ulysses| w/ Async Ulysses|w/ Ulysses + compile| w/ Async Ulysses + compile|
|:---:|:---:|:---:|:---:|  
|FLUX.1, 13.87s|**🎉13.20s**|12.21s|**🎉11.97s**|
|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/parallelism/flux.1024x1024.C0_Q0_NONE_Ulysses2.png" width=222px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/parallelism/flux.1024x1024.C0_Q0_NONE_Ulysses2_ulysses_async_qkv_proj.png" width=222px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/parallelism/flux.1024x1024.C1_Q0_NONE_Ulysses2.png" width=222px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/parallelism/flux.1024x1024.C1_Q0_NONE_Ulysses2_ulysses_async_qkv_proj.png" width=222px>

## Async FP8 Ulysses Attention

<div id="ulysses-async-fp8"></div>

![async_ulysses_fp8](https://github.com/vipshop/cache-dit/raw/main/assets/parallelism/async_ulysses_fp8.png)

cache-dit has implemented **Async FP8 Ulysses Attention** for **🔥all** supported DiTs. This optimization reduces communication latency while preserving high precision. Users can enable this feature by setting `experimental_ulysses_float8=True`. To maintain higher precision during softmax computation—where `Softmax(Q@K^T)` is sensitive to numerical instability—we currently retain `K in FP16/BF16` format. Float8-optimized all_to_all communication is therefore only applied to Q, V, and O.

```python
# pip3 install "cache-dit[parallelism]"
from cache_dit import ParallelismConfig

cache_dit.enable_cache(
    pipe_or_adapter, 
    cache_config=DBCacheConfig(...),
    # Set `experimental_ulysses_float8` as True to enable Async FP8 Ulysses Attention
    parallelism_config=ParallelismConfig(
        ulysses_size=2,
        parallel_kwargs={
            "experimental_ulysses_float8": True
        },
    ),
)
# torchrun --nproc_per_node=2 parallel_cache_ulysses_float8.py
```

|L20x2 w/ Ulysses| w/ Ulysses FP8|w/ Ulysses + compile|w/ Ulysses FP8 + compile|
|:---:|:---:|:---:|:---:|
|FLUX.1, 13.87s|**🎉13.36s**|12.21s|**🎉11.54s**|
|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/parallelism/flux.1024x1024.C0_Q0_NONE_Ulysses2.png" width=222px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/parallelism/flux.1024x1024.C0_Q0_NONE_Ulysses2_ulysses_float8.png" width=222px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/parallelism/flux.1024x1024.C1_Q0_NONE_Ulysses2.png" width=222px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/parallelism/flux.1024x1024.C1_Q0_NONE_Ulysses2_ulysses_float8.png" width=222px>|


## Ring Attention with Batched P2P  

Currently, cache-dit support 2 ring_rotate_method, namely, `allgather` and `p2p`. `allgather`: Use allgather to gather the key and value tensors (default). `p2p`: Use batch_isend_irecv ops to rotate the key and value tensors. This method is more efficient due to th better overlap of communication and computation.

```python
# pip3 install "cache-dit[parallelism]"
from cache_dit import ParallelismConfig

cache_dit.enable_cache(
    pipe_or_adapter, 
    cache_config=DBCacheConfig(...),
    # Set `ring_rotate_method` as 'p2p' to enable the faster ring attention implementation
    parallelism_config=ParallelismConfig(
        ring_size=2,
        parallel_kwargs={
            "ring_rotate_method": "p2p",
        },
    ),
)
# torchrun --nproc_per_node=2 parallel_ring_batched_p2p.py
```
