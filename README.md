<div align="center">
  <p align="center">
    <h2>⚡️ DBCache: Dual Block Cache for Diffusion Transformers </h2>
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

⚡️ **DBCache**: Dual Block Cache for Diffusion Transformers. We have enhanced `FBCache` into a more general algorithm, namely `DBCache`, enabling it to achieve fully `UNet-style` cache acceleration for DiT models. Different configurations of compute blocks (such as **F8B8**) can be customized in DBCache and it can be entirely `training-free`. DBCache can strike a `perfect balance` between performance and precision! Moreover, DBCache is a **plug-and-play** solution that works hand-in-hand with `ParaAttention`. Users can easily tap into its **Context Parallel** features for distributed inference.

|DBCache|Baseline(w/o Cache)|F1B0(0.08)|F1B0(0.20)| F12B12(0.20)|F16B16(0.20)|
|:---:|:---:|:---:|:---:|:---:|:---:|
|Latency(s)|24.8|15.5|8.5|15.1|17.7|
|Image|<img src=./assets/NONE_R0.08_S0.png width=100px>|<img src=./assets/DBCACHE_F1B0S1_R0.08_S11.png width=100px> |<img src=./assets/DBCACHE_F1B0S1_R0.2_S19.png width=100px>|<img src=./assets/DBCACHE_F12B12S4_R0.2_S16.png width=100px>|<img src=./assets/DBCACHE_F16B16S4_R0.2_S13.png width=100px>|

<div align="center">
  <p align="center">
    NVIDIA L20, Steps: 28, Prompt: "A cat holding a sign that says hello world with complex background"
  </p>
</div>

|DBCache(L20x4)|Baseline(L20x1)|F1B0(0.08)|F8B8(0.12)| F8B12(0.20)|F8B16(0.20)|
|:---:|:---:|:---:|:---:|:---:|:---:|
|Latency(s)|27.85|6.04|5.88|5.77|6.01|
|Image|<img src=https://github.com/user-attachments/assets/70ea57f4-d8f2-415b-8a96-d8315974a5e6 width=100px>|<img src=https://github.com/user-attachments/assets/fc0e1a67-19cc-44aa-bf50-04696e7978a0 width=100px> |<img src=https://github.com/user-attachments/assets/d1434896-628c-436b-95ad-43c085a8629e width=100px>|<img src=https://github.com/user-attachments/assets/aaa42cd2-57de-4c4e-8bfb-913018a8251d width=100px>|<img src=https://github.com/user-attachments/assets/dc0ba2a4-ef7c-436d-8a39-67055deab92f width=100px>|

<div align="center">
  <p align="center">
    DBCache: NVIDIA L20x4, Steps: 20, case to show the texture recovery ability of DBCache.
  </p>
</div>

These cases study demonstrates that even with a relatively high threshold (such as 0.2) under the DBCache **F12B12** or **F8B16** configuration, the detailed texture of the kitten's fur & color cloth, and the clarity of text can still be preserved. This suggests that users can leverage DBCache to effectively balance performance and precision in their workflows!

## ©️Citations

```BibTeX
@misc{DBCache@2025,
  title={DBCache: Dual Block Cache for Diffusion Transformers},
  url={https://github.com/vipshop/DBCache.git},
  note={Open-source software available at https://github.com/vipshop/DBCache.git},
  author={vipshop.com},
  year={2025}
}
```

## 💡Notes

The codebase of [DBCache](./src/cache_dit/) was adapted from [ParaAttention](https://github.com/chengzeyi/ParaAttention/tree/main/src/para_attn/first_block_cache). Many thanks! This project is still in its early development stages and currently provides some documentation and examples for reference.

## 📖Contents 

<div id="contents"></div>  

- [⚙️Installation](#️installation)  
- [⚡️Dual Block Cache](#dbcache)
- [🎉First Block Cache](#fbcache)
- [⚡️Dynamic Block Prune](#dbprune)
- [🎉Context Parallelism](#context-parallelism)  
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

![image](https://github.com/user-attachments/assets/c2a382b9-0ccd-46f4-aacc-87857b4a4de8)

**DBCache** provides configurable parameters for custom optimization, enabling a balanced trade-off between performance and precision:

- **Fn**: Specifies that DBCache uses the **first n** Transformer blocks to fit the information at time step t, enabling the calculation of a more stable L1 diff and delivering more accurate information to subsequent blocks.
- **Bn**: Further fuses approximate information in the **last n** Transformer blocks to enhance prediction accuracy. These blocks act as an auto-scaler for approximate hidden states that use residual cache.
- **warmup_steps**: (default: 0) DBCache does not apply the caching strategy when the number of running steps is less than or equal to this value, ensuring the model sufficiently learns basic features during warmup.
- **max_cached_steps**:  (default: -1) DBCache disables the caching strategy when the running steps exceed this value to prevent precision degradation.


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

![image](https://github.com/user-attachments/assets/748a64fb-fc3d-4280-b664-29a4c7b6c685)

**DBCache** is a more general algorithm than **FBCache**. When Fn=1 and Bn=0, **DBCache** behaves identically to **FBCache**. Therefore, you can either use the original **FBCache** implementation directly or configure **DBCache** with **F1B0** settings to achieve the same functionality.

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

## ⚡️DBPrune: Dynamic Block Prune

<div id="dbprune"></div>  

![image](https://github.com/user-attachments/assets/932b6360-9533-4352-b176-4c4d84bd4695)

We have further implemented a new **Dynamic Block Prune** algorithm with Residual Cache for Diffusion Transformers, which is referred to as **DBPrune**. (Note: DBPrune is currently in the experimental phase, and we kindly invite you to stay tuned for upcoming updates.)

```python
from diffusers import FluxPipeline
from cache_dit.cache_factory import apply_cache_on_pipe, CacheType

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

# Using DBPrune
cache_options = CacheType.default_options(CacheType.DBPrune)

apply_cache_on_pipe(pipe, **cache_options)
```


## 🎉Context Parallelism

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
