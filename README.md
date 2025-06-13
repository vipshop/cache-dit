<div align="center">
  <p align="center">
    <h2>⚡️ DBCache: Dual Block Cache for Diffusion Transformers.</h2>
  </p>
   <img src=./assets/dbcache.png >
  <div align='center'>
        <img src=https://img.shields.io/badge/Language-Python-brightgreen.svg >
        <img src=https://img.shields.io/badge/PRs-welcome-9cf.svg >
        <img src=https://img.shields.io/badge/pypi-pass-brightgreen.svg >
        <img src=https://img.shields.io/badge/python-3.10|3.11|3.12-skyblue.svg >
        <img src=https://img.shields.io/badge/Release-v0.1.0-brightgreen.svg >
 </div>
</div>

⚡️ **DBCache**: Dual Block Cache for Diffusion Transformers. We have enhanced `FBCache` into a more general algorithm, namely `DBCache`, enabling it to achieve fully `UNet-style` cache acceleration for DiT models. Different configurations of `Cache Blocks` (such as F8B8) can be customized in DBCache and it can be entirely `training-free`. DBCache can strike a `perfect balance` between performance and precision! Moreover, DBCache is a **plug-and-play** solution that works hand-in-hand with `ParaAttention`. Users can easily tap into its **Context Parallel** features for distributed inference.

|CacheType|Baseline|FBCache(0.08)|FBCache(0.12)|FBCache(0.20)|
|:---:|:---:|:---:|:---:|:---:|
|Cached Steps|0|11|11|19|
|Latency(s)|24.8|15.5|12.9|8.5|
|Image|![](./assets/NONE_R0.08_S0.png)|![](./assets/FBCACHE_R0.08_S11.png)|![](./assets/FBCACHE_R0.2_S19.png)|![](./assets/FBCACHE_R0.2_S19.png)|
|CacheType|DBCache F8B8(0.08)|DBCache F8B8(0.12)|DBCache F8B12(0.20)|DBCache F8B16(0.20)|  
|Cached Steps|9|12|18|18|  
|Latency(s)|19.2|17.3|14.6|15.7|
|Image|![](./assets/DBCACHE_F8B8S1_R0.08_S9.png)|![](./assets/DBCACHE_F8B8S1_R0.12_S12.png)|![](./assets/DBCACHE_F8B12S1_R0.2_S18.png)|![](./assets/DBCACHE_F8B16S1_R0.2_S18.png)|

<div align="center">
  <p align="center">
    NVIDIA L20, Steps: 28, Prompt: "A cat holding a sign that says hello world with complex background"
  </p>
</div>

The case shows that even at a large threshold, such as 0.2 (cached steps is 18), under the DBCache <b>F8B16</b> configuration, the style of the kitten's textured fur and text can still be maintained. Thus, users can use DBCache to strike a balance between performance and precision!   

💡 NOTE: The codebase of [DBCache](./src/cache_dit/) was adapted from [ParaAttention](https://github.com/chengzeyi/ParaAttention/tree/main/src/para_attn/first_block_cache). Many thanks! This project is still in its early development stages and currently provides some documentation and examples for reference. More features will be added in the future.

## ©️Citations

```BibTeX
@misc{DBCache@2025,
  title={DBCache: Dual Block Cache for Diffusion Transformers.},
  url={https://github.com/vipshop/DBCache.git},
  note={Open-source software available at https://github.com/vipshop/DBCache.git},
  author={vipshop.com},
  year={2025}
}
```

## 📖Contents 

<div id="contents"></div>  

- [⚙️Installation](#️installation)  
- [⚡️Dual Block Cache](#dbcache)
- [🎉First Block Cache](#fbcache)
- [⚡️Context Parallelism](#context-parallelism)  
- [👋Contribute](#contribute)
- [©️License](#license)


## ⚙️Installation  

<div id="installation"></div>

You can install `DBCache` from PyPI:

```bash
pip3 install 'torch==2.7.0'
pip3 install cache-dit
```

or just install it from sources:

```bash
git clone https://github.com/vipshop/DBCache.git
cd DBCache && git submodule update --init --recursive
pip3 install 'torch==2.7.0' 'setuptools>=64' 'setuptools_scm>=8'

pip3 install -e '.[dev]' --no-build-isolation # build editable package
python3 -m build && pip3 install ./dist/cache_dit-*.whl # or build whl first and then install it.
```

## ⚡️DBCache: Dual Block Cache  

<div id="dbcache"></div>


```python
from diffusers import FluxPipeline
from cache_dit.cache_factory import apply_cache_on_pipe, CacheType

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

# Default options, F8B8, good balance between performance and precision
cache_options = CacheType.default_options(CacheType.DBCache)

# Custom options, F8B16, higher precision
cache_options = {
    "cache_type": CacheType.DBCache,
    "warmup_steps": 8,
    "max_cached_steps": 8,    # -1 means no limit
    "Fn_compute_blocks": 8,   # Fn, F8, etc.
    "Bn_compute_blocks": 16,  # Bn, B16, etc.
    "residual_diff_threshold": 0.12,
}

apply_cache_on_pipe(pipe, **cache_options)
```

## 🎉FBCache: First Block Cache  

<div id="fbcache"></div>


```python
from diffusers import FluxPipeline
from cache_dit.cache_factory import apply_cache_on_pipe, CacheType

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

# Using FBCache directly
cache_options = CacheType.default_options(CacheType.FBCache)

# Or using DBCache with F1B0. 
# Fn=1, Bn=0, means FB Cache, otherwise, Dual Block Cache
cache_options = {
    "cache_type": CacheType.DBCache,
    "warmup_steps": 8,
    "max_cached_steps": 8,   # -1 means no limit
    "Fn_compute_blocks": 1,  # Fn, F1, etc.
    "Bn_compute_blocks": 0,  # Bn, B0, etc.
    "residual_diff_threshold": 0.12,
}

apply_cache_on_pipe(pipe, **cache_options)
```

## ⚡️Context Parallelism

<div id="context-parallelism"></div>  

DBCache is a **plug-and-play** solution that works hand-in-hand with [ParaAttention](https://github.com/chengzeyi/ParaAttention). Users can **easily tap into** its Context Parallel features for distributed inference. For example, Firstly, install `para-attn` from PyPI:

```bash
pip3 install para-attn  # or install `para-attn` from sources.
```

Then, you can run DBCache with **Context Parallel** on 4 GPUs:

```python
from diffusers import FluxPipeline
from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe
from cache_dit.cache_factory import apply_cache_on_pipe, CacheType

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

# Context Parallel from ParaAttention
parallelize_pipe(
    pipe, mesh=init_context_parallel_mesh(
        pipe.device.type, max_ulysses_dim_size=4
    )
)

# DBCache with F8B8 from this library
apply_cache_on_pipe(
    pipe, **CacheType.default_options(CacheType.DBCache)
)
```

## 👋Contribute 
<div id="contribute"></div>

How to contribute? Star this repo or check [CONTRIBUTE.md](./CONTRIBUTE.md).


## ©️License   

<div id="license"></div>


We have followed the original License from [ParaAttention](https://github.com/chengzeyi/ParaAttention), please check [LICENSE.md](./LICENSE.md) for more details.
