<div align="center">
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/cache-dit-logo.png height="120">

  <p align="center">
    An <b>Unified</b> and Training-free <b>Cache Acceleration</b> Toolbox for <b>Diffusion Transformers</b> <br>
    ♥️ <b>Cache Acceleration</b> with <b>One-line</b> Code ~ ♥️
  </p>
  <div align='center'>
      <img src=https://img.shields.io/badge/Language-Python-brightgreen.svg >
      <img src=https://img.shields.io/badge/PRs-welcome-9cf.svg >
      <img src=https://img.shields.io/badge/PyPI-pass-brightgreen.svg >
      <img src=https://static.pepy.tech/badge/cache-dit >
      <img src=https://img.shields.io/badge/Python-3.10|3.11|3.12-9cf.svg >
      <img src=https://img.shields.io/badge/Release-v0.2-brightgreen.svg >
 </div>
  <p align="center">
  🔥<b><a href="#unified">Unified Cache APIs</a> | <a href="#dbcache">DBCache</a> | <a href="#taylorseer">Hybrid TaylorSeer</a> | <a href="#cfg">Hybrid Cache CFG</a></b>🔥
  </p>
  <p align="center">
  🎉Now, <b>cache-dit</b> covers <b>Most</b> mainstream <b>Diffusers'</b> Pipelines</b>🎉<br>
  🔥<b><a href="#supported">Qwen-Image</a> | <a href="#supported">FLUX.1</a> | <a href="#supported">Wan 2.1</a> | <a href="#supported"> ... </a> | <a href="#supported">CogVideoX</a></b>🔥
  </p>
</div>

## 🔥News  

- [2025-08-19] 🔥[**Qwen-Image-Edit**](https://github.com/QwenLM/Qwen-Image) **2x⚡️** speedup! Check example [run_qwen_image_edit.py](./examples/run_qwen_image_edit.py).
- [2025-08-12] 🎉First caching mechanism in [QwenLM/Qwen-Image](https://github.com/QwenLM/Qwen-Image) with **[cache-dit](https://github.com/vipshop/cache-dit)**, check the [PR](https://github.com/QwenLM/Qwen-Image/pull/61). 
- [2025-08-11] 🔥[**Qwen-Image**](https://github.com/QwenLM/Qwen-Image) **1.8x⚡️** speedup! Please refer [run_qwen_image.py](./examples/run_qwen_image.py) as an example.

<details>
<summary> Previous News </summary>  
  
- [2025-08-10] 🔥[**FLUX.1-Kontext-dev**](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) is supported! Please refer [run_flux_kontext.py](./examples/run_flux_kontext.py) as an example.
- [2025-07-18] 🎉First caching mechanism in [🤗huggingface/flux-fast](https://github.com/huggingface/flux-fast) with **[cache-dit](https://github.com/vipshop/cache-dit)**, check the [PR](https://github.com/huggingface/flux-fast/pull/13). 
- [2025-07-13] **[🤗flux-faster](https://github.com/xlite-dev/flux-faster)** is released! **3.3x** speedup for FLUX.1 on NVIDIA L20 with **[cache-dit](https://github.com/vipshop/cache-dit)**.

</details>
 
## 📖Contents 

<div id="contents"></div>  

- [⚙️Installation](#️installation)
- [🔥Supported Models](#supported)
- [🎉Unified Cache APIs](#unified)
- [⚡️Dual Block Cache](#dbcache)
- [🔥Hybrid TaylorSeer](#taylorseer)
- [⚡️Hybrid Cache CFG](#cfg)
- [⚙️Torch Compile](#compile)
- [🛠Metrics CLI](#metrics)

## ⚙️Installation  

<div id="installation"></div>

You can install the stable release of `cache-dit` from PyPI:

```bash
pip3 install -U cache-dit
```
Or you can install the latest develop version from GitHub:

```bash
pip3 install git+https://github.com/vipshop/cache-dit.git
```

## 🔥Supported Models  

<div id="supported"></div>

Currently, **cache-dit** library supports almost **Any** Diffusion Transformers (with **Transformer Blocks** that match the specific Input and Output **patterns**). Please check [🎉Unified Cache APIs](#unified) for more details. Here are just some of the tested models listed:

- [🚀Qwen-Image-Edit](https://github.com/vipshop/cache-dit/raw/main/examples)  
- [🚀Qwen-Image](https://github.com/vipshop/cache-dit/raw/main/examples)  
- [🚀FLUX.1-dev](https://github.com/vipshop/cache-dit/raw/main/examples)  
- [🚀FLUX.1-Fill-dev](https://github.com/vipshop/cache-dit/raw/main/examples)  
- [🚀FLUX.1-Kontext-dev](https://github.com/vipshop/cache-dit/raw/main/examples)  
- [🚀CogVideoX](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀CogVideoX1.5](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀Wan2.1-T2V](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀Wan2.1-FLF2V](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀HunyuanVideo](https://github.com/vipshop/cache-dit/raw/main/examples)

<details>
<summary> More Pipelines </summary>  

- [🚀mochi-1-preview](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀LTXVideo](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀Allegro](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀CogView3Plus](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀CogView4](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀Cosmos](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀EasyAnimate](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀SkyReelsV2](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀SD3](https://github.com/vipshop/cache-dit/raw/main/examples)
  
</details>

## 🎉Unified Cache APIs

<div id="unified"></div>  

Currently, for any **Diffusion** models with **Transformer Blocks** that match the specific **Input/Output patterns**, we can use the **Unified Cache APIs** from **cache-dit**, namely, the `cache_dit.enable_cache(...)` API. The **Unified Cache APIs** are currently in the experimental phase; please stay tuned for updates. The supported patterns are listed as follows:

![](https://github.com/vipshop/cache-dit/raw/main/assets/patterns.png)

After the `cache_dit.enable_cache(...)` API is called, you just need to call the pipe as normal. The `pipe` param can be **any** Diffusion Pipeline. Please refer to [Qwen-Image](./examples/run_qwen_image_uapi.py) as an example. 
```python
import cache_dit
from diffusers import DiffusionPipeline 

# can be any diffusion pipeline
pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image")

# one line code with default cache options.
cache_dit.enable_cache(pipe) 

# or, enable cache with custom pattern settings.
cache_dit.enable_cache(
    pipe, 
    transformer=pipe.transformer,
    blocks=pipe.transformer.transformer_blocks,
    return_hidden_states_first=False,
)

# just call the pipe as normal.
output = pipe(...)

# then, summary the cache stats.
stats = cache_dit.summary(pipe)
```

After finishing each inference of `pipe(...)`, you can call the `cache_dit.summary(...)` API on pipe to get the details of the cache stats for the current inference (markdown table format). You can set `details` param as `True` to show more details of cache stats.

```python
⚡️Cache Steps and Residual Diffs Statistics: QwenImagePipeline

| Cache Steps | Diffs P00 | Diffs P25 | Diffs P50 | Diffs P75 | Diffs P95 |
|-------------|-----------|-----------|-----------|-----------|-----------|
| 23          | 0.04      | 0.082     | 0.115     | 0.152     | 0.245     |
...
```

## ⚡️DBCache: Dual Block Cache  

<div id="dbcache"></div>

![](https://github.com/vipshop/cache-dit/raw/main/assets/dbcache-fnbn-v1.png)

**DBCache**: **Dual Block Caching** for Diffusion Transformers. Different configurations of compute blocks (**F8B12**, etc.) can be customized in DBCache, enabling a balanced trade-off between performance and precision. Moreover, it can be entirely **training**-**free**. Please check [DBCache.md](./docs/DBCache.md) docs for more design details.

- **Fn**: Specifies that DBCache uses the **first n** Transformer blocks to fit the information at time step t, enabling the calculation of a more stable L1 diff and delivering more accurate information to subsequent blocks.
- **Bn**: Further fuses approximate information in the **last n** Transformer blocks to enhance prediction accuracy. These blocks act as an auto-scaler for approximate hidden states that use residual cache.

```python
import cache_dit
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

# Default options, F8B0, 8 warmup steps, and unlimited cached 
# steps for good balance between performance and precision
cache_dit.enable_cache(pipe)

# Custom options, F8B8, higher precision
cache_dit.enable_cache(
    pipe,
    warmup_steps=8,      # steps do not cache
    max_cached_steps=-1, # -1 means no limit
    Fn_compute_blocks=8, # Fn, F8, etc.
    Bn_compute_blocks=8, # Bn, B8, etc.
    residual_diff_threshold=0.12,
)
```
Moreover, users configuring higher **Bn** values (e.g., **F8B16**) while aiming to maintain good performance can specify **Bn_compute_blocks_ids** to work with Bn. DBCache will only compute the specified blocks, with the remaining estimated using the previous step's residual cache.

```python
# Custom options, F8B16, higher precision with good performance.
cache_dit.enable_cache(
    pipe,
    Fn_compute_blocks=8,  # Fn, F8, etc.
    Bn_compute_blocks=16, # Bn, B16, etc.
    # 0, 2, 4, ..., 14, 15, etc. [0,16)
    Bn_compute_blocks_ids=cache_dit.block_range(0, 16, 2),
    # If the L1 difference is below this threshold, skip Bn blocks 
    # not in `Bn_compute_blocks_ids`(1, 3,..., etc), Otherwise, 
    # compute these blocks.
    non_compute_blocks_diff_threshold=0.08,
)
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

**TaylorSeer** employs a differential method to approximate the higher-order derivatives of features and predict features in future timesteps with Taylor series expansion. The TaylorSeer implemented in cache-dit supports both hidden states and residual cache types. That is $\mathcal{F}\_{\text {pred }, m}\left(x_{t-k}^l\right)$ can be a residual cache or a hidden-state cache.

```python
cache_dit.enable_cache(
    pipe,
    enable_taylorseer=True,
    enable_encoder_taylorseer=True,
    # Taylorseer cache type cache be hidden_states or residual.
    taylorseer_cache_type="residual",
    # Higher values of n_derivatives will lead to longer 
    # computation time but may improve precision significantly.
    taylorseer_kwargs={
        "n_derivatives": 2, # default is 2.
    },
    warmup_steps=3, # prefer: >= n_derivatives + 1
    residual_diff_threshold=0.12
)
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

## ⚡️Hybrid Cache CFG

<div id="cfg"></div>

cache-dit supports caching for **CFG (classifier-free guidance)**. For models that fuse CFG and non-CFG into a single forward step, or models that do not include CFG (classifier-free guidance) in the forward step, please set `do_separate_classifier_free_guidance` param to **False (default)**. Otherwise, set it to True. For examples:

```python
cache_dit.enable_cache(
    pipe, 
    ...,
    # CFG: classifier free guidance or not
    # For model that fused CFG and non-CFG into single forward step,
    # should set do_separate_classifier_free_guidance as False.
    # For example, set it as True for Wan 2.1 and set it as False 
    # for FLUX.1, HunyuanVideo, CogVideoX, Mochi.
    do_separate_classifier_free_guidance=True, # Wan 2.1, Qwen-Image
    # Compute cfg forward first or not, default False, namely, 
    # 0, 2, 4, ..., -> non-CFG step; 1, 3, 5, ... -> CFG step.
    cfg_compute_first=False,
    # Compute spearate diff values for CFG and non-CFG step, 
    # default True. If False, we will use the computed diff from 
    # current non-CFG transformer step for current CFG step.
    cfg_diff_compute_separate=True,
)
```

## ⚙️Torch Compile

<div id="compile"></div>  

By the way, **cache-dit** is designed to work compatibly with **torch.compile.** You can easily use cache-dit with torch.compile to further achieve a better performance. For example:

```python
cache_dit.enable_cache(pipe)

# Compile the Transformer module
pipe.transformer = torch.compile(pipe.transformer)
```
However, users intending to use **cache-dit** for DiT with **dynamic input shapes** should consider increasing the **recompile** **limit** of `torch._dynamo`. Otherwise, the recompile_limit error may be triggered, causing the module to fall back to eager mode. 
```python
torch._dynamo.config.recompile_limit = 96  # default is 8
torch._dynamo.config.accumulated_recompile_limit = 2048  # default is 256
```

Please check [bench.py](./bench/bench.py) for more details.


## 🛠Metrics CLI

<div id="metrics"></div>    

You can utilize the APIs provided by cache-dit to quickly evaluate the accuracy losses caused by different cache configurations. For example:

```python
from cache_dit.metrics import compute_psnr
from cache_dit.metrics import compute_video_psnr
from cache_dit.metrics import FrechetInceptionDistance  # FID

FID = FrechetInceptionDistance()
image_psnr, n = compute_psnr("true.png", "test.png") # Num: n
image_fid,  n = FID.compute_fid("true_dir", "test_dir")
video_psnr, n = compute_video_psnr("true.mp4", "test.mp4") # Frames: n
```

Please check [test_metrics.py](./tests/test_metrics.py) for more details. Or, you can use `cache-dit-metrics-cli` tool. For examples: 

```bash
cache-dit-metrics-cli -h  # show usage
# all: PSNR, FID, SSIM, MSE, ..., etc.
cache-dit-metrics-cli all  -i1 true.png -i2 test.png  # image
cache-dit-metrics-cli all  -i1 true_dir -i2 test_dir  # image dir
cache-dit-metrics-cli all  -v1 true.mp4 -v2 test.mp4  # video
cache-dit-metrics-cli all  -v1 true_dir -v2 test_dir  # video dir
cache-dit-metrics-cli fid  -i1 true_dir -i2 test_dir  # FID
cache-dit-metrics-cli psnr -i1 true_dir -i2 test_dir  # PSNR
```

## 👋Contribute 
<div id="contribute"></div>

How to contribute? Star ⭐️ this repo to support us or check [CONTRIBUTE.md](./CONTRIBUTE.md).

## ©️License   

<div id="license"></div>

The **cache-dit** codebase is adapted from FBCache. Special thanks to their excellent work! We have followed the original License from FBCache, please check [LICENSE](./LICENSE) for more details.

## ©️Citations

<div id="citations"></div>

```BibTeX
@misc{cache-dit@2025,
  title={cache-dit: An Unified and Training-free Cache Acceleration Toolbox for Diffusion Transformers},
  url={https://github.com/vipshop/cache-dit.git},
  note={Open-source software available at https://github.com/vipshop/cache-dit.git},
  author={vipshop.com},
  year={2025}
}
```
