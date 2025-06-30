<div align="center">
  <p align="center">
    <h2>🤗 CacheDiT: A Training-free and Easy-to-use Cache Acceleration <br>Toolbox for Diffusion Transformers</h2>
  </p>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/cache-dit-v1.png >
  <div align='center'>
      <img src=https://img.shields.io/badge/Language-Python-brightgreen.svg >
      <img src=https://img.shields.io/badge/PRs-welcome-9cf.svg >
      <img src=https://img.shields.io/badge/PyPI-pass-brightgreen.svg >
      <img src=https://static.pepy.tech/badge/cache-dit >
      <img src=https://img.shields.io/badge/Python-3.10|3.11|3.12-9cf.svg >
      <img src=https://img.shields.io/badge/Release-v0.2.2-brightgreen.svg >
 </div>
  <p align="center">
    DeepCache is for UNet not DiT. Most DiT cache speedups are complex and not training-free. CacheDiT offers <br>a set of training-free cache accelerators for DiT: <b>🔥<a href="#dbcache">DBCache</a>, <a href="#dbprune">DBPrune</a>, <a href="#taylorseer">TaylorSeer</a>, <a href="#fbcache">FBCache</a></b>, etc🔥
  </p>
</div>

<div align="center">
  <p align="center">
    <b>♥️ Please consider to leave a ⭐️ Star to support us ~ ♥️</b>
  </p>
</div> 

## 🤗 Introduction 

<div align="center">
  <p align="center">
    <h3>🔥DBCache: Dual Block Caching for Diffusion Transformers</h3>
  </p>
</div> 

**DBCache**: **Dual Block Caching** for Diffusion Transformers. We have enhanced `FBCache` into a more general and customizable cache algorithm, namely `DBCache`, enabling it to achieve fully `UNet-style` cache acceleration for DiT models. Different configurations of compute blocks (**F8B12**, etc.) can be customized in DBCache. Moreover, it can be entirely **training**-**free**. DBCache can strike a perfect **balance** between performance and precision!

<div align="center">
  <p align="center">
    DBCache, <b> L20x1 </b>, Steps: 28, "A cat holding a sign that says hello world with complex background"
  </p>
</div>

|Baseline(L20x1)|F1B0 (0.08)|F1B0 (0.20)|F8B8 (0.15)|F12B12 (0.20)|F16B16 (0.20)|
|:---:|:---:|:---:|:---:|:---:|:---:|
|24.85s|15.59s|8.58s|15.41s|15.11s|17.74s|
|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/NONE_R0.08_S0.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBCACHE_F1B0S1_R0.08_S11.png width=105px> | <img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBCACHE_F1B0S1_R0.2_S19.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBCACHE_F8B8S1_R0.15_S15.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBCACHE_F12B12S4_R0.2_S16.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBCACHE_F16B16S4_R0.2_S13.png width=105px>|
|**Baseline(L20x1)**|**F1B0 (0.08)**|**F8B8 (0.12)**|**F8B12 (0.12)**|**F8B16 (0.20)**|**F8B20 (0.20)**|
|27.85s|6.04s|5.88s|5.77s|6.01s|6.20s|
|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/TEXTURE_NONE_R0.08.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/TEXTURE_DBCACHE_F1B0_R0.08.png width=105px> |<img src=https://github.com/vipshop/cache-dit/raw/main/assets/TEXTURE_DBCACHE_F8B8_R0.12.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/TEXTURE_DBCACHE_F8B12_R0.12.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/TEXTURE_DBCACHE_F8B16_R0.2.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/TEXTURE_DBCACHE_F8B20_R0.2.png width=105px>|

<div align="center">
  <p align="center">
    DBCache, <b> L20x4 </b>, Steps: 20, case to show the texture recovery ability of DBCache
  </p>
</div>

These case studies demonstrate that even with relatively high thresholds (such as 0.12, 0.15, 0.2, etc.) under the DBCache **F12B12** or **F8B16** configuration, the detailed texture of the kitten's fur, colored cloth, and the clarity of text can still be preserved. This suggests that users can leverage DBCache to effectively balance performance and precision in their workflows! 

<div align="center">
  <p align="center">
    <h3>🔥DBPrune: Dynamic Block Prune with Residual Caching</h3>
  </p>
</div> 

**DBPrune**: We have further implemented a new **Dynamic Block Prune** algorithm based on **Residual Caching** for Diffusion Transformers, referred to as DBPrune. DBPrune caches each block's hidden states and residuals, then **dynamically prunes** blocks during inference by computing the L1 distance between previous hidden states. When a block is pruned, its output is approximated using the cached residuals.

<div align="center">
  <p align="center">
    DBPrune, <b> L20x1 </b>, Steps: 28, "A cat holding a sign that says hello world with complex background"
  </p>
</div>

|Baseline(L20x1)|Pruned(24%)|Pruned(35%)|Pruned(38%)|Pruned(45%)|Pruned(60%)|
|:---:|:---:|:---:|:---:|:---:|:---:|
|24.85s|19.43s|16.82s|15.95s|14.24s|10.66s|
|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/NONE_R0.08_S0.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBPRUNE_F1B0_R0.03_P24.0_T19.43s.png width=105px> | <img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBPRUNE_F1B0_R0.04_P34.6_T16.82s.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBPRUNE_F1B0_R0.05_P38.3_T15.95s.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBPRUNE_F1B0_R0.06_P45.2_T14.24s.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBPRUNE_F1B0_R0.2_P59.5_T10.66s.png width=105px>|

<div align="center">
  <p align="center">
    <h3>🔥Context Parallelism and Torch Compile</h3>
  </p>
</div> 

Moreover, **CacheDiT** are **plug-and-play** solutions that works hand-in-hand with [ParaAttention](https://github.com/chengzeyi/ParaAttention). Users can easily tap into its **Context Parallelism** features for distributed inference. CacheDiT is designed to work compatibly with **torch.compile.** You can easily use CacheDiT with torch.compile to further achieve a better performance.

<div align="center">
  <p align="center">
  DBPrune + <b>torch.compile + context parallelism</b> <br>Steps: 28, "A cat holding a sign that says hello world with complex background"
  </p>
</div>

|Baseline|Pruned(24%)|Pruned(35%)|Pruned(38%)|Pruned(45%)|Pruned(60%)|
|:---:|:---:|:---:|:---:|:---:|:---:|
|+compile:20.43s|16.25s|14.12s|13.41s|12.00s|8.86s|
|+L20x4:7.75s|6.62s|6.03s|5.81s|5.24s|3.93s|
|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/U0_C1_NONE_R0.08_S0_T20.43s.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/U0_C1_DBPRUNE_F1B0_R0.03_P24.0_T16.25s.png width=105px> | <img src=https://github.com/vipshop/cache-dit/raw/main/assets/U0_C1_DBPRUNE_F1B0_R0.04_P34.6_T14.12s.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/U0_C1_DBPRUNE_F1B0_R0.045_P38.2_T13.41s.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/U0_C1_DBPRUNE_F1B0_R0.055_P45.1_T12.00s.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/U0_C1_DBPRUNE_F1B0_R0.2_P59.5_T8.86s.png width=105px>|

## ©️Citations

```BibTeX
@misc{CacheDiT@2025,
  title={CacheDiT: A Training-free and Easy-to-use cache acceleration Toolbox for Diffusion Transformers},
  url={https://github.com/vipshop/cache-dit.git},
  note={Open-source software available at https://github.com/vipshop/cache-dit.git},
  author={vipshop.com},
  year={2025}
}
```

## 👋Reference

<div id="reference"></div>

The **CacheDiT** codebase is adapted from [FBCache](https://github.com/chengzeyi/ParaAttention/tree/main/src/para_attn/first_block_cache). Special thanks to their excellent work! 

## 📖Contents 

<div id="contents"></div>  

- [⚙️Installation](#️installation)
- [🔥Supported Models](#supported)
- [⚡️Dual Block Cache](#dbcache)
- [🔥Hybrid TaylorSeer](#taylorseer)
- [🎉First Block Cache](#fbcache)
- [⚡️Dynamic Block Prune](#dbprune)
- [🎉Context Parallelism](#context-parallelism)  
- [🔥Torch Compile](#compile)
- [👋Contribute](#contribute)
- [©️License](#license)

## ⚙️Installation  

<div id="installation"></div>

You can install the stable release of `cache-dit` from PyPI:

```bash
pip3 install cache-dit
```
Or you can install the latest develop version from GitHub:

```bash
pip3 install git+https://github.com/vipshop/cache-dit.git
```

## 🔥Supported Models  

<div id="supported"></div>

- [🚀FLUX.1](https://github.com/vipshop/cache-dit/raw/main/examples)  
- [🚀Mochi](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀CogVideoX](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀CogVideoX1.5](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀Wan2.1](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀HunyuanVideo](https://github.com/vipshop/cache-dit/raw/main/examples)


## ⚡️DBCache: Dual Block Cache  

<div id="dbcache"></div>

![](https://github.com/vipshop/cache-dit/raw/main/assets/dbcache-v1.png)

**DBCache** provides configurable parameters for custom optimization, enabling a balanced trade-off between performance and precision:

- **Fn**: Specifies that DBCache uses the **first n** Transformer blocks to fit the information at time step t, enabling the calculation of a more stable L1 diff and delivering more accurate information to subsequent blocks.
- **Bn**: Further fuses approximate information in the **last n** Transformer blocks to enhance prediction accuracy. These blocks act as an auto-scaler for approximate hidden states that use residual cache.

![](https://github.com/vipshop/cache-dit/raw/main/assets/dbcache-fnbn-v1.png)

- **warmup_steps**: (default: 0) DBCache does not apply the caching strategy when the number of running steps is less than or equal to this value, ensuring the model sufficiently learns basic features during warmup.
- **max_cached_steps**:  (default: -1) DBCache disables the caching strategy when the previous cached steps exceed this value to prevent precision degradation.
- **residual_diff_threshold**: The value of residual diff threshold, a higher value leads to faster performance at the cost of lower precision.

For a good balance between performance and precision, DBCache is configured by default with **F8B8**, 8 warmup steps, and unlimited cached steps.

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
Moreover, users configuring higher **Bn** values (e.g., **F8B16**) while aiming to maintain good performance can specify **Bn_compute_blocks_ids** to work with Bn. DBCache will only compute the specified blocks, with the remaining estimated using the previous step's residual cache.

```python
# Custom options, F8B16, higher precision with good performance.
cache_options = {
    # 0, 2, 4, ..., 14, 15, etc. [0,16)
    "Bn_compute_blocks_ids": CacheType.range(0, 16, 2),
    # If the L1 difference is below this threshold, skip Bn blocks 
    # not in `Bn_compute_blocks_ids`(1, 3,..., etc), Otherwise, 
    # compute these blocks.
    "non_compute_blocks_diff_threshold": 0.08,
}
```

<div align="center">
  <p align="center">
    DBCache, <b> L20x1 </b>, Steps: 28, "A cat holding a sign that says hello world with complex background"
  </p>
</div>

|Baseline(L20x1)|F1B0 (0.08)|F1B0 (0.20)|F8B8 (0.15)|F12B12 (0.20)|F16B16 (0.20)|
|:---:|:---:|:---:|:---:|:---:|:---:|
|24.85s|15.59s|8.58s|15.41s|15.11s|17.74s|
|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/NONE_R0.08_S0.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBCACHE_F1B0S1_R0.08_S11.png width=105px> | <img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBCACHE_F1B0S1_R0.2_S19.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBCACHE_F8B8S1_R0.15_S15.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBCACHE_F12B12S4_R0.2_S16.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBCACHE_F16B16S4_R0.2_S13.png width=105px>|

## 🔥Hybrid TaylorSeer

<div id="taylorseer"></div>

We have supported the [TaylorSeers: From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers](https://arxiv.org/pdf/2503.06923) algorithm to further improve the precision of DBCache in cases where the cached steps are large, namely, **Hybrid TaylorSeer + DBCache**. At timesteps with significant intervals, the feature similarity in diffusion models decreases substantially, significantly harming the generation quality. 

$$
\mathcal{F}\_{\text {pred }, m}\left(x_{t-k}^l\right)=\mathcal{F}\left(x_t^l\right)+\sum_{i=1}^m \frac{\Delta^i \mathcal{F}\left(x_t^l\right)}{i!\cdot N^i}(-k)^i
$$

**TaylorSeer** employs a differential method to approximate the higher-order derivatives of features and predict features in future timesteps with Taylor series expansion. The TaylorSeer implemented in CacheDiT supports both hidden states and residual cache types. That is $\mathcal{F}\_{\text {pred }, m}\left(x_{t-k}^l\right)$ can be a residual cache or a hidden-state cache.

```python
cache_options = {
    # TaylorSeer options
    "enable_taylorseer": True,
    "enable_encoder_taylorseer": True,
    # Taylorseer cache type cache be hidden_states or residual.
    "taylorseer_cache_type": "residual",
    # Higher values of n_derivatives will lead to longer 
    # computation time but may improve precision significantly.
    "taylorseer_kwargs": {
        "n_derivatives": 2, # default is 2.
    },
    "warmup_steps": 3, # n_derivatives + 1
    "residual_diff_threshold": 0.12,
}
``` 

> [!Important]
> Please note that if you have used TaylorSeer as the calibrator for approximate hidden states, the **Bn** param of DBCache can be set to **0**. In essence, DBCache's Bn is also act as a calibrator, so you can choose either Bn > 0 or TaylorSeer. We recommend using the configuration scheme of **TaylorSeer** + **DBCache FnB0**.

<div align="center">
  <p align="center">
    <b>DBCache F1B0 + TaylorSeer</b>, L20x1, Steps: 28, <br>"A cat holding a sign that says hello world with complex background"
  </p>
</div>

|Baseline(L20x1)|F1B0 (0.12)|+TaylorSeer|F1B0 (0.15)|+TaylorSeer|+compile| 
|:---:|:---:|:---:|:---:|:---:|:---:|
|24.85s|12.85s|12.86s|10.27s|10.28s|8.48s|
|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/NONE_R0.08_S0.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/U0_C0_DBCACHE_F1B0S1W0T0ET0_R0.12_S14_T12.85s.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/U0_C0_DBCACHE_F1B0S1W0T1ET1_R0.12_S14_T12.86s.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/U0_C0_DBCACHE_F1B0S1W0T0ET0_R0.15_S17_T10.27s.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/U0_C0_DBCACHE_F1B0S1W0T1ET1_R0.15_S17_T10.28s.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/U0_C1_DBCACHE_F1B0S1W0T1ET1_R0.15_S17_T8.48s.png width=105px>|

## 🎉FBCache: First Block Cache  

<div id="fbcache"></div>

![](https://github.com/vipshop/cache-dit/raw/main/assets/fbcache-v1.png)

**DBCache** is a more general cache algorithm than **FBCache**. When Fn=1 and Bn=0, DBCache behaves identically to FBCache. Therefore, you can either use the original FBCache implementation directly or configure **DBCache** with **F1B0** settings to achieve the same functionality.

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

![](https://github.com/vipshop/cache-dit/raw/main/assets/dbprune-v1.png)

We have further implemented a new **Dynamic Block Prune** algorithm based on **Residual Caching** for Diffusion Transformers, which is referred to as **DBPrune**. DBPrune caches each block's hidden states and residuals, then dynamically prunes blocks during inference by computing the L1 distance between previous hidden states. When a block is pruned, its output is approximated using the cached residuals. DBPrune is currently in the experimental phase, and we kindly invite you to stay tuned for upcoming updates.

```python
from diffusers import FluxPipeline
from cache_dit.cache_factory import apply_cache_on_pipe, CacheType

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

# Using DBPrune with default options
cache_options = CacheType.default_options(CacheType.DBPrune)

apply_cache_on_pipe(pipe, **cache_options)
```

We have also brought the designs from DBCache to DBPrune to make it a more general and customizable block prune algorithm. You can specify the values of **Fn** and **Bn** for higher precision, or set up the non-prune blocks list **non_prune_blocks_ids** to avoid aggressive pruning. For example:

```python
# Custom options for DBPrune
cache_options = {
    "cache_type": CacheType.DBPrune,
    "residual_diff_threshold": 0.05,
    # Never prune the first `Fn` and last `Bn` blocks.
    "Fn_compute_blocks": 8,  # default 1
    "Bn_compute_blocks": 8,  # default 0
    "warmup_steps": 8,  # default -1
    # Disables the pruning strategy when the previous 
    # pruned steps greater than this value.
    "max_pruned_steps": 12,  # default, -1 means no limit
    # Enable dynamic prune threshold within step, higher 
    # `max_dynamic_prune_threshold` value may introduce a more 
    # ageressive pruning strategy.
    "enable_dynamic_prune_threshold": True,
    "max_dynamic_prune_threshold": 2 * 0.05,
    # (New thresh) = mean(previous_block_diffs_within_step) * 1.25
    # (New thresh) = ((New thresh) if (New thresh) <
    # max_dynamic_prune_threshold else residual_diff_threshold)
    "dynamic_prune_threshold_relax_ratio": 1.25,
    # The step interval to update residual cache. For example, 
    # 2: means the update steps will be [0, 2, 4, ...].
    "residual_cache_update_interval": 1,
    # You can set non-prune blocks to avoid ageressive pruning. 
    # For example, FLUX.1 has 19 + 38 blocks, so we can set it 
    # to 0, 2, 4, ..., 56, etc.
    "non_prune_blocks_ids": [],
}

apply_cache_on_pipe(pipe, **cache_options)
```

> [!Important]
> Please note that for GPUs with lower VRAM, DBPrune may not be suitable for use on video DiTs, as it caches the hidden states and residuals of each block, leading to higher GPU memory requirements. In such cases, please use DBCache, which only caches the hidden states and residuals of 2 blocks.

<div align="center">
  <p align="center">
    DBPrune, <b> L20x1 </b>, Steps: 28, "A cat holding a sign that says hello world with complex background"
  </p>
</div>

|Baseline(L20x1)|Pruned(24%)|Pruned(35%)|Pruned(38%)|Pruned(45%)|Pruned(60%)|
|:---:|:---:|:---:|:---:|:---:|:---:|
|24.85s|19.43s|16.82s|15.95s|14.24s|10.66s|
|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/NONE_R0.08_S0.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBPRUNE_F1B0_R0.03_P24.0_T19.43s.png width=105px> | <img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBPRUNE_F1B0_R0.04_P34.6_T16.82s.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBPRUNE_F1B0_R0.05_P38.3_T15.95s.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBPRUNE_F1B0_R0.06_P45.2_T14.24s.png width=105px>|<img src=https://github.com/vipshop/cache-dit/raw/main/assets/DBPRUNE_F1B0_R0.2_P59.5_T10.66s.png width=105px>|

## 🎉Context Parallelism

<div id="context-parallelism"></div>  

**CacheDiT** are **plug-and-play** solutions that works hand-in-hand with [ParaAttention](https://github.com/chengzeyi/ParaAttention). Users can **easily tap into** its **Context Parallelism** features for distributed inference. Firstly, install `para-attn` from PyPI:

```bash
pip3 install para-attn  # or install `para-attn` from sources.
```

Then, you can run **DBCache** or **DBPrune** with **Context Parallelism** on 4 GPUs:

```python
import torch.distributed as dist
from diffusers import FluxPipeline
from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe
from cache_dit.cache_factory import apply_cache_on_pipe, CacheType

# Init distributed process group
dist.init_process_group()
torch.cuda.set_device(dist.get_rank())

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

# DBPrune with default options from this library
apply_cache_on_pipe(
    pipe, **CacheType.default_options(CacheType.DBPrune)
)

dist.destroy_process_group()
```
Then, run the python test script with `torchrun`:
```bash
torchrun --nproc_per_node=4 parallel_cache.py
```

## 🔥Torch Compile

<div id="compile"></div>  

By the way, **CacheDiT** is designed to work compatibly with **torch.compile.** You can easily use CacheDiT with torch.compile to further achieve a better performance. For example:

```python
apply_cache_on_pipe(
    pipe, **CacheType.default_options(CacheType.DBPrune)
)
# Compile the Transformer module
pipe.transformer = torch.compile(pipe.transformer)
```
However, users intending to use **CacheDiT** for DiT with **dynamic input shapes** should consider increasing the **recompile** **limit** of `torch._dynamo`. Otherwise, the recompile_limit error may be triggered, causing the module to fall back to eager mode. 
```python
torch._dynamo.config.recompile_limit = 96  # default is 8
torch._dynamo.config.accumulated_recompile_limit = 2048  # default is 256
```

Please check [bench.py](./bench/bench.py) for more details.

## 👋Contribute 
<div id="contribute"></div>

How to contribute? Star ⭐️ this repo to support us or check [CONTRIBUTE.md](https://github.com/vipshop/cache-dit/raw/main/CONTRIBUTE.md).

## ©️License   

<div id="license"></div>


We have followed the original License from [ParaAttention](https://github.com/chengzeyi/ParaAttention), please check [LICENSE](https://github.com/vipshop/cache-dit/raw/main/LICENSE) for more details.
