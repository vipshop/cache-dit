# Extra Modules Parallelism

## Parallelize Text Encoder

<div id="parallel-text-encoder"></div>

Users can set the `extra_parallel_modules` parameter in parallelism_config (when using Tensor Parallelism or Context Parallelism) to specify additional modules that need to be parallelized beyond the main transformer — e.g, `text_encoder` in `Flux2Pipeline`. It can further reduce the per-GPU memory requirement and slightly improve the inference performance of the text encoder. 

Currently, cache-dit supported text encoder parallelism for **T5Encoder, UMT5Encoder, Llama, Gemma 1/2/3, Mistral, Mistral-3, Qwen-3, Qwen-2.5 VL, Glm and Glm-4** model series, namely, supported almost **[🔥ALL](./SUPPORTED.md)** pipelines in diffusers.

```python
# pip3 install "cache-dit[parallelism]"
from cache_dit import ParallelismConfig

# Transformer Tensor Parallelism + Text Encoder Tensor Parallelism
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

# Transformer Context Parallelism + Text Encoder Tensor Parallelism
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
# torchrun --nproc_per_node=2 parallel_cache.py
```

## Parallelize Auto Encoder (VAE)

<div id="parallel-auto-encoder"></div>

Currently, cache-dit supported auto encoder (vae) parallelism for **AutoencoderKL, AutoencoderKLQwenImage, AutoencoderKLWan, and AutoencoderKLHunyuanVideo** series, namely, supported almost **[🔥ALL](./SUPPORTED.md)** pipelines in diffusers. It can further reduce the per-GPU memory requirement and slightly improve the inference performance of the auto encoder. Users can set it by `extra_parallel_modules` parameter in parallelism_config, for example:

```python
# pip3 install "cache-dit[parallelism]"
from cache_dit import ParallelismConfig

# Transformer Context Parallelism + Text Encoder Tensor Parallelism + VAE Data Parallelism
cache_dit.enable_cache(
    pipe, 
    cache_config=DBCacheConfig(...),
    parallelism_config=ParallelismConfig(
        ulysses_size=2,
        parallel_kwargs={
            "extra_parallel_modules": [pipe.text_encoder, pipe.vae], # FLUX.1
        },
    ),
)
# torchrun --nproc_per_node=2 parallel_cache.py
```

## Parallelize ControlNet

<div id="parallel-controlnet"></div>

Further, cache-dit even supported controlnet parallelism for specific models, such as Z-Image-Turbo with ControlNet. Users can set it by `extra_parallel_modules` parameter in parallelism_config, for example:

```python
# pip3 install "cache-dit[parallelism]"
from cache_dit import ParallelismConfig

# Transformer Context Parallelism + Text Encoder Tensor Parallelism 
# + VAE Data Parallelism + ControlNet Context Parallelism
cache_dit.enable_cache(
    pipe, 
    cache_config=DBCacheConfig(...),
    parallelism_config=ParallelismConfig(
        ulysses_size=2,
        # case: Z-Image-Turbo-Fun-ControlNet-2.1
        parallel_kwargs={
            "extra_parallel_modules": [pipe.text_encoder, pipe.vae, pipe.controlnet],
        },
    ),
)
# torchrun --nproc_per_node=2 parallel_cache.py
```
