# NVIDIA GPU Benchmark

## NVIDIA L20

### Z-Image-ControlNet: Hybrid Cache + Parallelism

|Z-Image-ControlNet| Context Parallel: Ulysses 2 |  Context Parallel: Ulysses 4 | + ControlNet Parallel |
|:---:|:---:|:---:|:---:|
|Base L20x1: 22s|15.7s|12.7s|**🚀7.71s**|
| <img src="https://github.com/vipshop/cache-dit/raw/main/examples/assets/zimage_controlnet.1728x992.C0_Q0_NONE.png" width=200px> | <img src="https://github.com/vipshop/cache-dit/raw/main/examples/assets/zimage_controlnet.1728x992.C0_Q0_NONE_Ulysses2.png" width=200px> | <img src="https://github.com/vipshop/cache-dit/raw/main/examples/assets/zimage_controlnet.1728x992.C0_Q0_NONE_Ulysses4.png" width=200px> | <img src="https://github.com/vipshop/cache-dit/raw/main/examples/assets/zimage_controlnet.1728x992.C0_Q0_NONE_Ulysses4_CNP.png" width=200px> |
| **+ Hybrid Cache** | **+ Torch Compile** | **+ Async Ulyess CP** | **+ FP8 All2All + CUDNN ATTN** | 
|**🚀6.85s**|6.45s|6.38s|**🚀6.19s, 5.47s**|
| <img src="https://github.com/vipshop/cache-dit/raw/main/examples/assets/zimage_controlnet.1728x992.C0_Q0_DBCache_F1B0_W4I1M0MC3_R0.6_SCM111101001_dynamic_CFG0_T0O0_Ulysses4_S2_CNP.png" width=200px> | <img src="https://github.com/vipshop/cache-dit/raw/main/examples/assets/zimage_controlnet.1728x992.C1_Q0_DBCache_F1B0_W4I1M0MC3_R0.6_SCM111101001_dynamic_CFG0_T0O0_Ulysses4_S2_CNP.png" width=200px> |<img src="https://github.com/vipshop/cache-dit/raw/main/examples/assets/zimage_controlnet.1728x992.C1_Q0_DBCache_F1B0_W4I1M0MC3_R0.6_SCM111101001_dynamic_CFG0_T0O0_Ulysses4_S2_ulysses_async_CNP.png" width=200px> | <img src="https://github.com/vipshop/cache-dit/raw/main/examples/assets/zimage_controlnet.1728x992.C1_Q0_DBCache_F1B0_W4I1M0MC3_R0.6_SCM111101001_dynamic_CFG0_T0O0_Ulysses4_S2_ulysses_float8_CNP_sdpa_cudnn.png" width=200px> 

### FLUX.1-dev: Hybrid Cache + Parallelism

|Baseline(L20x1)|F1B0 (0.08)|F1B0 (0.20)|F8B8 (0.15)|F12B12 (0.20)|
|:---:|:---:|:---:|:---:|:---:|
|24.85s|15.59s|8.58s|15.41s|15.11s|
|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/NONE_R0.08_S0.png width=130px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBCACHE_F1B0S1_R0.08_S11.png width=130px> | <img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBCACHE_F1B0S1_R0.2_S19.png width=130px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBCACHE_F8B8S1_R0.15_S15.png width=130px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBCACHE_F12B12S4_R0.2_S16.png width=130px>|
|**Baseline(L20x1)**|**F1B0 (0.08)**|**F8B8 (0.12)**|**F8B12 (0.12)**|**F8B16 (0.20)**|
|27.85s|6.04s|5.88s|5.77s|6.01s|
|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/TEXTURE_NONE_R0.08.png width=130px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/TEXTURE_DBCACHE_F1B0_R0.08.png width=130px> |<img src=https://github.com/vipshop/cache-dit/raw/main/assets/TEXTURE_DBCACHE_F8B8_R0.12.png width=130px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/TEXTURE_DBCACHE_F8B12_R0.12.png width=130px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/TEXTURE_DBCACHE_F8B16_R0.2.png width=130px>|

### UAA: Ulysses Anything Attention  

#### Qwen-Image & FLUX.1-dev

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

#### Z-Image-Turbo

<p align="center">
    ✅<b>Any Head Num</b><br>
    Ulysses: Ulysses Attention, <b>FP8 Ulysses: Ulysses w/ FP8 All2All</b>, Device: NVIDIA L20<br>
    🔥<b>Z-Image</b> (Head=30, ❌<b>CAN NOT</b> divisible by 4), 1024x1024, 9 steps.
</p>

|Ulysses 2, L20|Ulysses 4|FP8 Ulysses 4| + Cache | + FP8 DQ | 
|:---:|:---:|:---:|:---:|:---:|    
|1024x1024, 3.19s|1024x1024, 1.98s|1024x1024, 1.89s|1024x1024, 1.63s|1024x1024, 1.23s|    
|<img width="180" height="180" alt="zimage C1_Q0_NONE_Ulysses2_sdpa_cudnn" src="https://github.com/vipshop/cache-dit/raw/main/docs/assets/zimage.C1_Q0_NONE_Ulysses2_sdpa_cudnn.png" />|<img width="180" height="180" alt="zimage C1_Q0_NONE_Ulysses4_sdpa_cudnn" src="https://github.com/vipshop/cache-dit/raw/main/docs/assets/zimage.C1_Q0_NONE_Ulysses4_sdpa_cudnn.png" />|<img width="180" height="180" alt="zimage C1_Q0_NONE_Ulysses4_ulysses_float8_sdpa_cudnn" src="https://github.com/vipshop/cache-dit/raw/main/docs/assets/zimage.C1_Q0_NONE_Ulysses4_ulysses_float8_sdpa_cudnn.png" />|<img width="180" height="180" alt="zimage C1_Q0_DBCache_F1B0_W4I1M0MC0_R0 6_SCM111110101_dynamic_CFG0_T0O0_Ulysses4_S2_ulysses_float8_sdpa_cudnn" src="https://github.com/vipshop/cache-dit/raw/main/docs/assets/zimage.C1_Q0_DBCache_F1B0_W4I1M0MC0_R0.6_SCM111110101_dynamic_CFG0_T0O0_Ulysses4_S2_ulysses_float8_sdpa_cudnn.png" />|<img width="180" height="180" alt="zimage C1_Q1_float8_DBCache_F1B0_W4I1M0MC0_R0 6_SCM111110101_dynamic_CFG0_T0O0_Ulysses4_S2_ulysses_float8_sdpa_cudnn" src="https://github.com/vipshop/cache-dit/raw/main/docs/assets/zimage.C1_Q1_float8_DBCache_F1B0_W4I1M0MC0_R0.6_SCM111110101_dynamic_CFG0_T0O0_Ulysses4_S2_ulysses_float8_sdpa_cudnn.png" />|   

### Async Ulysses QKV Projection

#### FLUX.1-dev

<p align="center">
    Ulysses: Standard Ulysses Attention, <b>Async Ulysses</b>: Ulysses Attenton with Async QKV Projection
</p>

|L20x2 w/ Ulysses| w/ Async Ulysses|w/ Ulysses + compile| w/ Async Ulysses + compile|
|:---:|:---:|:---:|:---:|  
|FLUX.1, 13.87s|**🎉13.20s**|12.21s|**🎉11.97s**|
|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/parallelism/flux.1024x1024.C0_Q0_NONE_Ulysses2.png" width=222px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/parallelism/flux.1024x1024.C0_Q0_NONE_Ulysses2_ulysses_async_qkv_proj.png" width=222px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/parallelism/flux.1024x1024.C1_Q0_NONE_Ulysses2.png" width=222px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/parallelism/flux.1024x1024.C1_Q0_NONE_Ulysses2_ulysses_async_qkv_proj.png" width=222px>


### Async FP8 Ulysses Attention

#### FLUX.1-dev

|L20x2 w/ Ulysses| w/ Ulysses FP8|w/ Ulysses + compile|w/ Ulysses FP8 + compile|
|:---:|:---:|:---:|:---:|
|FLUX.1, 13.87s|**🎉13.36s**|12.21s|**🎉11.54s**|
|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/parallelism/flux.1024x1024.C0_Q0_NONE_Ulysses2.png" width=222px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/parallelism/flux.1024x1024.C0_Q0_NONE_Ulysses2_ulysses_float8.png" width=222px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/parallelism/flux.1024x1024.C1_Q0_NONE_Ulysses2.png" width=222px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/parallelism/flux.1024x1024.C1_Q0_NONE_Ulysses2_ulysses_float8.png" width=222px>|


## NVIDIA H100

|Model|Baseline H100x1|Ulysses 2| + FA3| + cache| + compile|  
|:---:|:---:|:---:|:---:|:---:|:---:|
FLUX.1-dev: 50 steps |  9.30s |  6.04s | 5.99s  | 2.60s | 1.92s |
Qwen-Image: 50 steps | 18.49s  | 12.81s  | 12.75s  |  5.67s | 4.20s |

Reproduce command:

```shell
# FLUX.1-dev: 50 steps
python3 generate.py flux --steps 50
torchrun --nproc_per_node=2 generate.py flux --steps 50 --parallel ulysses
torchrun --nproc_per_node=2 generate.py flux --steps 50 --parallel ulysses --attn _flash_3
torchrun --nproc_per_node=2 generate.py flux --steps 50 --parallel ulysses --attn _flash_3 --cache
torchrun --nproc_per_node=2 generate.py flux --steps 50 --parallel ulysses --attn _flash_3 --cache --compile
# Qwen-Image: 50 steps
python3 generate.py qwen_image --steps 50
torchrun --nproc_per_node=2 generate.py qwen_image --steps 50 --parallel ulysses
torchrun --nproc_per_node=2 generate.py qwen_image --steps 50 --parallel ulysses --attn _flash_3
torchrun --nproc_per_node=2 generate.py qwen_image --steps 50 --parallel ulysses --attn _flash_3 --cache
torchrun --nproc_per_node=2 generate.py qwen_image --steps 50 --parallel ulysses --attn _flash_3 --cache --compile
```

## NVIDIA H800

|Model|Baseline H800x1|Ulysses 2| + FA3| + cache| + compile|  
|:---:|:---:|:---:|:---:|:---:|:---:|
|FLUX.1-dev: 50 steps||||||
|Qwen-Image: 50 steps||||||

