<div align="center">
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/cache-dit-logo.png height="120">

<p align="center">
    A <b>Unified</b> and Training-free <b>Cache Acceleration</b> Toolbox for <b>Diffusion Transformers</b> <br>
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
  🎉Now, <b>cache-dit</b> covers <b>most</b> mainstream Diffusers' <b>DiT</b> Pipelines🎉<br>
  🔥<a href="#supported">Qwen-Image</a> | <a href="#supported">FLUX.1</a> | <a href="#supported">Wan 2.1</a> | <a href="#supported"> Wan 2.2 </a> | <a href="#supported">HunyuanVideo</a>🔥<br>
  🔥<a href="#supported">HunyuanDiT</a> | <a href="#supported">HiDream</a> | <a href="#supported">Mochi</a> | <a href="#supported"> CogVideoX </a> | <a href="#supported">CogVideoX1.5</a>🔥<br>
  🔥<a href="#supported">CogView3Plus</a> | <a href="#supported">CogView4</a> | <a href="#supported">Chroma</a> | <a href="#supported"> LTXVideo </a> | <a href="#supported">PixArt</a>🔥<br>
  🔥<a href="#supported">Cosmos</a> | <a href="#supported">SkyReelsV2</a> | <a href="#supported">VisualCloze</a> | <a href="#supported"> OmniGen </a> | <a href="#supported">Lumina 1/2</a>🔥<br>
  🔥<a href="#supported">Allegro</a> | <a href="#supported">EasyAnimate</a> | <a href="#supported">SD 3/3.5</a> | <a href="#supported"> ... </a> | <a href="#supported">DiT-XL</a>🔥
  </p>
</div>
<div align='center'>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/gifs/wan2.2.C0_Q0_NONE.gif width=160px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/gifs/wan2.2.C1_Q0_DBCACHE_F1B0_W2M8MC2_T1O2_R0.08.gif width=160px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/gifs/wan2.2.C1_Q1_fp8_w8a8_dq_DBCACHE_F1B0_W2M8MC2_T1O2_R0.08.gif width=160px>
  <p><b>🔥Wan2.2 MoE</b> | <b><a href="https://github.com/vipshop/cache-dit">+cache-dit</a>:~2.0x↑🎉</b> | +FP8 DQ:<b>~2.4x↑🎉</b></p>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/qwen-image.C0_Q0_NONE.png width=160px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/qwen-image.C1_Q0_DBCACHE_F8B0_W8M0MC0_T1O4_R0.12_S23.png width=160px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/qwen-image.C1_Q1_fp8_w8a8_dq_DBCACHE_F8B0_W8M0MC0_T1O4_R0.12_S18.png width=160px>
  <p><b>🔥Qwen-Image</b> | <b><a href="https://github.com/vipshop/cache-dit">+cache-dit</a>:~1.8x↑🎉</b> | +FP8 DQ:<b>~2.2x↑🎉</b></p>
  <img src=./assets/qwen-image-lightning.C0_Q0_NONE.png width=200px>
  <img src=./assets/qwen-image-lightning.C1_Q0_DBCACHE_F16B16_W4M2MC1_T0O2_R0.5_S2.png width=200px>
  <p><b>🔥Qwen-Image-Lightning</b> | <b><a href="https://github.com/vipshop/cache-dit">+cache-dit</a></b>+fuse-lora+...:<b>~1.7x↑🎉</b>
  <br>♥️ Please consider to leave a <b>⭐️ Star</b> to support us ~ ♥️</p>
</div>

## 🔥News  

- [2025-09-08] 🎉[**Qwen-Image-Lightning**](https://github.com/ModelTC/Qwen-Image-Lightning) **1.7x↑🎉** speedup! w/ **[cache-dit](https://github.com/vipshop/cache-dit)** + **fuse-lora + compile**
- [2025-09-03] 🎉[**Wan2.2-MoE**](https://github.com/Wan-Video) **2.4x↑🎉** speedup! Please refer to [run_wan_2.2.py](./examples/pipeline/run_wan_2.2.py) as an example.
- [2025-08-19] 🔥[**Qwen-Image-Edit**](https://github.com/QwenLM/Qwen-Image) **2x↑🎉** speedup! Check the example: [run_qwen_image_edit.py](./examples/pipeline/run_qwen_image_edit.py).
- [2025-08-12] 🎉First caching mechanism in [QwenLM/Qwen-Image](https://github.com/QwenLM/Qwen-Image) with **[cache-dit](https://github.com/vipshop/cache-dit)**, check this [PR](https://github.com/QwenLM/Qwen-Image/pull/61). 
- [2025-08-11] 🔥[**Qwen-Image**](https://github.com/QwenLM/Qwen-Image) **1.8x↑🎉** speedup! Please refer to [run_qwen_image.py](./examples/pipeline/run_qwen_image.py) as an example.
- [2025-07-13] 🎉[**FLUX.1-dev**](https://github.com/xlite-dev/flux-faster) **3.3x↑🎉** speedup! NVIDIA L20 with **[cache-dit](https://github.com/vipshop/cache-dit)** + **compile + FP8 DQ**.

<details>
<summary> Previous News </summary>  

- [2025-09-01] 📚[**Hybird Forward Pattern**](#unified) is supported! Please check [FLUX.1-dev](./examples/run_flux_adapter.py) as an example.
- [2025-08-10] 🔥[**FLUX.1-Kontext-dev**](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) is supported! Please refer [run_flux_kontext.py](./examples/pipeline/run_flux_kontext.py) as an example.
- [2025-07-18] 🎉First caching mechanism in [🤗huggingface/flux-fast](https://github.com/huggingface/flux-fast) with **[cache-dit](https://github.com/vipshop/cache-dit)**, check the [PR](https://github.com/huggingface/flux-fast/pull/13). 

</details>

## 📖Contents 

<div id="contents"></div>  

- [⚙️Installation](#️installation)
- [🔥Supported Models](#supported)
- [🎉Unified Cache APIs](#unified)
  - [📚Forward Pattern Matching](#unified)
  - [🎉Cache with One-line Code](#unified)
  - [🔥Automatic Block Adapter](#unified)
  - [📚Hybird Forward Pattern](#unified)
  - [🤖Cache Acceleration Stats](#unified)
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

- [🚀Qwen-Image-Lightning](https://github.com/vipshop/cache-dit/raw/main/examples)  
- [🚀Qwen-Image-Edit](https://github.com/vipshop/cache-dit/raw/main/examples)  
- [🚀Qwen-Image](https://github.com/vipshop/cache-dit/raw/main/examples)  
- [🚀FLUX.1-dev](https://github.com/vipshop/cache-dit/raw/main/examples)  
- [🚀FLUX.1-Fill-dev](https://github.com/vipshop/cache-dit/raw/main/examples)  
- [🚀FLUX.1-Kontext-dev](https://github.com/vipshop/cache-dit/raw/main/examples)  
- [🚀CogVideoX](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀CogVideoX1.5](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀Wan2.2-T2V](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀Wan2.1-T2V](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀Wan2.1-FLF2V](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀mochi-1-preview](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀HunyuanVideo](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀HunyuanDiT](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀HiDream-I1-Full](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀PixArt-Alpha](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀PixArt-Sigma](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀SD-3/3.5](https://github.com/vipshop/cache-dit/raw/main/examples)

</details>

## 🎉Unified Cache APIs

<div id="unified"></div>  

### 📚Forward Pattern Matching 

Currently, for any **Diffusion** models with **Transformer Blocks** that match the specific **Input/Output patterns**, we can use the **Unified Cache APIs** from **cache-dit**, namely, the `cache_dit.enable_cache(...)` API. The **Unified Cache APIs** are currently in the experimental phase; please stay tuned for updates. The supported patterns are listed as follows:

![](https://github.com/vipshop/cache-dit/raw/main/assets/patterns-v1.png)

### ♥️Cache Acceleration with One-line Code

In most cases, you only need to call **one-line** of code, that is `cache_dit.enable_cache(...)`. After this API is called, you just need to call the pipe as normal. The `pipe` param can be **any** Diffusion Pipeline. Please refer to [Qwen-Image](./examples/pipeline/run_qwen_image.py) as an example. 

```python
import cache_dit
from diffusers import DiffusionPipeline 

# Can be any diffusion pipeline
pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image")

# One-line code with default cache options.
cache_dit.enable_cache(pipe) 

# Just call the pipe as normal.
output = pipe(...)

# Disable cache and run original pipe.
cache_dit.disable_cache(pipe)
```

### 🔥Automatic Block Adapter

But in some cases, you may have a **modified** Diffusion Pipeline or Transformer that is not located in the diffusers library or not officially supported by **cache-dit** at this time. The **BlockAdapter** can help you solve this problems. Please refer to [🔥Qwen-Image w/ BlockAdapter](./examples/adapter/run_qwen_image_adapter.py) as an example.

```python
from cache_dit import ForwardPattern, BlockAdapter

# Use 🔥BlockAdapter with `auto` mode.
cache_dit.enable_cache(
    BlockAdapter(
        # Any DiffusionPipeline, Qwen-Image, etc.  
        pipe=pipe, auto=True,
        # Check `📚Forward Pattern Matching` documentation and hack the code of
        # of Qwen-Image, you will find that it has satisfied `FORWARD_PATTERN_1`.
        forward_pattern=ForwardPattern.Pattern_1,
    ),   
)

# Or, manually setup transformer configurations.
cache_dit.enable_cache(
    BlockAdapter(
        pipe=pipe, # Qwen-Image, etc.
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        forward_pattern=ForwardPattern.Pattern_1,
    ), 
)
```
For such situations, **BlockAdapter** can help you quickly apply various cache acceleration features to your own Diffusion Pipelines and Transformers. Please check the [📚BlockAdapter.md](./docs/BlockAdapter.md) for more details.

### 📚Hybird Forward Pattern

Sometimes, a Transformer class will contain more than one transformer `blocks`. For example, **FLUX.1** (HiDream, Chroma, etc) contains transformer_blocks and single_transformer_blocks (with different forward patterns). The **BlockAdapter** can also help you solve this problem. Please refer to [📚FLUX.1](./examples/adapter/run_flux_adapter.py) as an example.

```python
# For diffusers <= 0.34.0, FLUX.1 transformer_blocks and 
# single_transformer_blocks have different forward patterns.
cache_dit.enable_cache(
    BlockAdapter(
        pipe=pipe, # FLUX.1, etc.
        transformer=pipe.transformer,
        blocks=[
            pipe.transformer.transformer_blocks,
            pipe.transformer.single_transformer_blocks,
        ],
        forward_pattern=[
            ForwardPattern.Pattern_1,
            ForwardPattern.Pattern_3,
        ],
    ),
)
```

### 🤖Cache Acceleration Stats Summary

After finishing each inference of `pipe(...)`, you can call the `cache_dit.summary()` API on pipe to get the details of the **Cache Acceleration Stats** for the current inference. 
```python
stats = cache_dit.summary(pipe)
```

You can set `details` param as `True` to show more details of cache stats. (markdown table format) Sometimes, this may help you analyze what values of the residual diff threshold would be better.

```python
⚡️Cache Steps and Residual Diffs Statistics: QwenImagePipeline

| Cache Steps | Diffs Min | Diffs P25 | Diffs P50 | Diffs P75 | Diffs P95 | Diffs Max |
|-------------|-----------|-----------|-----------|-----------|-----------|-----------|
| 23          | 0.045     | 0.084     | 0.114     | 0.147     | 0.241     | 0.297     |
```

## ⚡️DBCache: Dual Block Cache  

<div id="dbcache"></div>

![](https://github.com/vipshop/cache-dit/raw/main/assets/dbcache-v1.png)

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
    max_warmup_steps=8,  # steps do not cache
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
    # Higher values of order will lead to longer computation time
    taylorseer_order=2, # default is 2.
    max_warmup_steps=3, # prefer: >= order + 1
    residual_diff_threshold=0.12
)s
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

cache-dit supports caching for **CFG (classifier-free guidance)**. For models that fuse CFG and non-CFG into a single forward step, or models that do not include CFG (classifier-free guidance) in the forward step, please set `enable_spearate_cfg` param to **False (default)**. Otherwise, set it to True. For examples:

```python
cache_dit.enable_cache(
    pipe, 
    ...,
    # CFG: classifier free guidance or not
    # For model that fused CFG and non-CFG into single forward step,
    # should set enable_spearate_cfg as False. For example, set it as True 
    # for Wan 2.1/Qwen-Image and set it as False for FLUX.1, HunyuanVideo, 
    # CogVideoX, Mochi, LTXVideo, Allegro, CogView3Plus, EasyAnimate, SD3, etc.
    enable_spearate_cfg=True, # Wan 2.1, Qwen-Image, CogView4, Cosmos, SkyReelsV2, etc.
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
```

## 👋Contribute 
<div id="contribute"></div>

How to contribute? Star ⭐️ this repo to support us or check [CONTRIBUTE.md](./CONTRIBUTE.md).

<div align='center'>
<a href="https://star-history.com/#vipshop/cache-dit&Date">
  <picture align='center'>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=vipshop/cache-dit&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=vipshop/cache-dit&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=vipshop/cache-dit&type=Date" width=400px />
  </picture>
</a>
</div>

## ©️Acknowledgements

<div id="Acknowledgements"></div>

The **cache-dit** codebase is adapted from FBCache. Over time its codebase diverged a lot, and **cache-dit** API is no longer compatible with FBCache. 

## ©️Citations

<div id="citations"></div>

```BibTeX
@misc{cache-dit@2025,
  title={cache-dit: A Unified and Training-free Cache Acceleration Toolbox for Diffusion Transformers},
  url={https://github.com/vipshop/cache-dit.git},
  note={Open-source software available at https://github.com/vipshop/cache-dit.git},
  author={vipshop.com},
  year={2025}
}
```
