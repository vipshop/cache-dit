# TE-P, VAE-P and CN-P 

## TE-P: Parallelize Text Encoder

<div id="parallel-text-encoder"></div>

Users can set the `extra_parallel_modules` parameter in parallelism_config (when using Tensor Parallelism or Context Parallelism) to specify additional modules that need to be parallelized beyond the main transformer — e.g, `text_encoder` in `Flux2Pipeline`. It can further reduce the per-GPU memory requirement and slightly improve the inference performance of the text encoder. 

Currently, cache-dit supported text encoder parallelism for **T5Encoder, UMT5Encoder, Llama, Gemma 1/2/3, Mistral, Mistral-3, Qwen-3, Qwen-2.5 VL, Glm and Glm-4** model series, namely, supported almost **[🔥ALL](../supported_matrix/NVIDIA_GPU.md)** pipelines in diffusers.

```python
from cache_dit import ParallelismConfig

cache_dit.enable_cache(
    pipe, 
    cache_config=DBCacheConfig(...),
    parallelism_config=ParallelismConfig(
        tp_size=2,
        parallel_kwargs={
            "extra_parallel_modules": [pipe.text_encoder], # FLUX.2
        },
    ),
)

cache_dit.enable_cache(
    pipe, 
    cache_config=DBCacheConfig(...),
    parallelism_config=ParallelismConfig(
        ulysses_size=2,
        parallel_kwargs={
            "extra_parallel_modules": [pipe.text_encoder], # FLUX.2
        },
    ),
)
```

## VAE-P: Parallelize Auto Encoder

<div id="parallel-auto-encoder"></div>

Currently, cache-dit supported auto encoder (vae) parallelism for **AutoencoderKL, AutoencoderKLQwenImage, AutoencoderKLWan, and AutoencoderKLHunyuanVideo** series, namely, supported almost **[🔥ALL](../supported_matrix/NVIDIA_GPU.md)** pipelines in diffusers. It can further reduce the per-GPU memory requirement and slightly improve the inference performance of the auto encoder. Users can set it by `extra_parallel_modules` parameter in parallelism_config, for example:

```python
from cache_dit import ParallelismConfig

cache_dit.enable_cache(
    pipe, 
    cache_config=DBCacheConfig(...),
    parallelism_config=ParallelismConfig(
        ulysses_size=2,
        parallel_kwargs={
            "extra_parallel_modules": [pipe.vae],
        },
    ),
)
```

From the table below (Image Generation: FLUX.2-Klein-4B), it is clear that `Ulysses-4 + VAE-P-4` delivers higher throughput than `Ulysses-4` alone, while also significantly reducing the per-GPU memory usage thus can avoid OOM issues on low-VRAM devices. Furthermore, the image quality remains nearly identical between the two approaches while the inference speed is slightly improved with VAE parallelism.

|FLUX.2-Klein-4B Ulysses-4|FLUX.2-Klein-4B Ulysses-4 + VAE-P-4|
|:---:|:---:|
|3.74s, 24.46GiB per GPU|🎉**3.37s, 17.34GiB per GPU**|
|<img src="https://github.com/vipshop/cache-dit/raw/main/docs/assets/flux2_klein_4b.2048x2048_C0_Q0_NONE_Ulysses4.png" >|<img src="https://github.com/vipshop/cache-dit/raw/main/docs/assets/flux2_klein_4b.2048x2048_C0_Q0_NONE_Ulysses4_VAEP.png" >|


## CN-P: Parallelize ControlNet

<div id="parallel-controlnet"></div>

Further, cache-dit even supported Controlnet Parallelism (CN-P) for specific models, such as Z-Image-Turbo with ControlNet. Users can set it by `extra_parallel_modules` parameter in parallelism_config, for example:

```python
from cache_dit import ParallelismConfig

cache_dit.enable_cache(
    pipe, 
    cache_config=DBCacheConfig(...),
    parallelism_config=ParallelismConfig(
        ulysses_size=2,
        # case: Z-Image-Turbo-Fun-ControlNet-2.1
        parallel_kwargs={
            "extra_parallel_modules": [pipe.controlnet],
        },
    ),
)
# torchrun --nproc_per_node=2 parallel_cache.py
```

## Hybrid TE-P + VAE-P + CN-P

User can also combine the above techniques together to further reduce the per-GPU memory usage and improve the inference performance, for example:

```python
from cache_dit import DBCacheConfig, ParallelismConfig

cache_dit.enable_cache(
    pipe_or_adapter, 
    cache_config=DBCacheConfig(...), # w/ Cache
    parallelism_config=ParallelismConfig(
        ulysses_size=4, tp_size=2, # 2D Parallelsim
        # e.g, Z-Image-Turbo with ControlNet, we can also parallelize the 
        # Text Encoder, VAE and ControlNet module to further reduce the 
        # memory usage on low-VRAM devices.
        parallel_kwargs={
            "extra_parallel_modules": [
                pipe.text_encoder, 
                pipe.vae,
                pipe.controlnet, # only support for Z-Image-Turbo currently
            ], 
        },
    ),
)
```

From the table below (Image Generation: FLUX.2-Klein-4B), it is clear that combining TE-P and VAE-P with Ulysses-4 (`Ulysses-4 + VAE-P-4 + TE-P-4`) results in a significant reduction in per-GPU memory usage compared to using Ulysses-4 alone. This combined approach not only minimizes memory consumption but also enhances inference speed, making it a highly efficient solution for deploying large diffusion models on devices with limited VRAM.

|FLUX.2-Klein-4B Ulysses-4| Ulysses-4 + VAE-P-4|Ulysses-4 + TE-P-4|Ulysses-4 + VAE-P-4 + TE-P-4|
|:---:|:---:|:---:|:---:|
|24.46GiB per GPU| 🎉17.34GiB per GPU|🎉19.37GiB per GPU|🎉12.25GiB per GPU|
