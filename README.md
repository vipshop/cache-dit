<div align="center">
  <p align="center">
    <h2>🤗 CacheDiT: An Unified and Training-free Cache Acceleration <br>Toolbox for Diffusion Transformers</h2>
  </p>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/cache-dit-v1.png >
  <div align='center'>
      <img src=https://img.shields.io/badge/Language-Python-brightgreen.svg >
      <img src=https://img.shields.io/badge/PRs-welcome-9cf.svg >
      <img src=https://img.shields.io/badge/PyPI-pass-brightgreen.svg >
      <img src=https://static.pepy.tech/badge/cache-dit >
      <img src=https://img.shields.io/badge/Python-3.10|3.11|3.12-9cf.svg >
      <img src=https://img.shields.io/badge/Release-v0.2-brightgreen.svg >
 </div>
  🔥<b><a href="#unified">Unified Cache APIs</a> | <a href="#dbcache">DBCache</a> | <a href="#taylorseer">Hybrid TaylorSeer</a> | <a href="#cfg">Hybrid Cache CFG</a></b>🔥
</div>

<div align="center">
  <p align="center">
    ♥️ Cache <b>Acceleration</b> with <b>One-line</b> Code ~ ♥️
  </p>
</div> 


## 🔥News  

- [2025-08-19] 🔥[**Qwen-Image-Edit**](https://github.com/QwenLM/Qwen-Image) **~2x⚡️** speedup! Check example [run_qwen_image_edit.py](./examples/run_qwen_image_edit.py).
- [2025-08-18] 🎉Early **[Unified Cache APIs](#unified)** released! Check [Qwen-Image w/ UAPI](./examples/run_qwen_image_uapi.py) as an example.
- [2025-08-12] 🎉First caching mechanism in [QwenLM/Qwen-Image](https://github.com/QwenLM/Qwen-Image) with **[cache-dit](https://github.com/vipshop/cache-dit)**, check the [PR](https://github.com/QwenLM/Qwen-Image/pull/61). 
- [2025-08-11] 🔥[**Qwen-Image**](https://github.com/QwenLM/Qwen-Image) **~1.8x⚡️** speedup! Please refer [run_qwen_image.py](./examples/run_qwen_image.py) as an example.
- [2025-08-10] 🔥[FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) is supported! Please refer [run_flux_kontext.py](./examples/run_flux_kontext.py) as an example.
- [2025-07-18] 🎉First caching mechanism in [🤗huggingface/flux-fast](https://github.com/huggingface/flux-fast) with **[cache-dit](https://github.com/vipshop/cache-dit)**, check the [PR](https://github.com/huggingface/flux-fast/pull/13). 
- [2025-07-13] **[🤗flux-faster](https://github.com/xlite-dev/flux-faster)** is released! **3.3x** speedup for FLUX.1 on NVIDIA L20 with `cache-dit`.

## 📖Contents 

<div id="contents"></div>  

- [⚙️Installation](#️installation)
- [🔥Supported Models](#supported)
- [🎉Unified Cache APIs](#unified)
- [⚡️Dual Block Cache](#dbcache)
- [🔥Hybrid TaylorSeer](#taylorseer)
- [⚡️Hybrid Cache CFG](#cfg)
- [🔥Torch Compile](#compile)
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
- [🚀mochi-1-preview](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀CogVideoX](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀CogVideoX1.5](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀Wan2.1-T2V](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀Wan2.1-FLF2V](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀HunyuanVideo](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀LTXVideo](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀Allegro](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀CogView3Plus](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀CogView4](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀Cosmos](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀EasyAnimate](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀SkyReelsV2](https://github.com/vipshop/cache-dit/raw/main/examples)
- [🚀SD3](https://github.com/vipshop/cache-dit/raw/main/examples)

## 🎉Unified Cache APIs

<div id="unified"></div>  


Currently, for any **Diffusion** models with **Transformer Blocks** that match the specific **Input/Output pattern**, we can use the **Unified Cache APIs** from **cache-dit**. The supported patterns are listed as follows:

```bash
(IN: hidden_states, encoder_hidden_states, ...) -> (OUT: hidden_states, encoder_hidden_states)  
(IN: hidden_states, encoder_hidden_states, ...) -> (OUT: encoder_hidden_states, hidden_states)  
(IN: hidden_states, encoder_hidden_states, ...) -> (OUT: hidden_states)
(IN: hidden_states, ...) -> (OUT: hidden_states) # TODO, DiT, Lumina2, etc. 
```

Please refer to [Qwen-Image w/ UAPI](./examples/run_qwen_image_uapi.py) as an example. The `pipe` parameter can be **Any** Diffusion Pipelines. The **Unified Cache APIs** are currently in the experimental phase, please stay tuned for updates. 

```python
import cache_dit
from diffusers import DiffusionPipeline # Can be [Any] Diffusion Pipeline

pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image")

# One line code with default cache options.
cache_dit.enable_cache(pipe) 

# Or, enable cache with custom setting.
cache_dit.enable_cache(
    pipe, transformer=pipe.transformer,
    blocks=pipe.transformer.transformer_blocks,
    return_hidden_states_first=False,
    **cache_dit.default_options(),
)

# summary cache stats.
cache_dit.summary(pipe)
```

After finishing each inference of `pipe(...)`, you can call the `cache_dict.summary` API on pipe to get the details of the cache stats for the current inference (markdown table format). For example:

```bash
Cache Steps and Residual Diffs Statistics:

| Cache Steps | Diffs P00 | Diffs P25 | Diffs P50 | Diffs P75 | Diffs P95 |
|-------------|-----------|-----------|-----------|-----------|-----------|
| 14          | 0.033     | 0.083     | 0.116     | 0.144     | 0.243     |
```

## ⚡️DBCache: Dual Block Cache  

<div id="dbcache"></div>

![](https://github.com/vipshop/cache-dit/raw/main/assets/dbcache-v1.png)


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


**DBCache** provides configurable parameters for custom optimization, enabling a balanced trade-off between performance and precision:

- **Fn**: Specifies that DBCache uses the **first n** Transformer blocks to fit the information at time step t, enabling the calculation of a more stable L1 diff and delivering more accurate information to subsequent blocks.
- **Bn**: Further fuses approximate information in the **last n** Transformer blocks to enhance prediction accuracy. These blocks act as an auto-scaler for approximate hidden states that use residual cache.

![](https://github.com/vipshop/cache-dit/raw/main/assets/dbcache-fnbn-v1.png)

- **warmup_steps**: (default: 0) DBCache does not apply the caching strategy when the number of running steps is less than or equal to this value, ensuring the model sufficiently learns basic features during warmup.
- **max_cached_steps**:  (default: -1) DBCache disables the caching strategy when the previous cached steps exceed this value to prevent precision degradation.
- **residual_diff_threshold**: The value of residual diff threshold, a higher value leads to faster performance at the cost of lower precision.

For a good balance between performance and precision, DBCache is configured by default with **F8B0**, 8 warmup steps, and unlimited cached steps.

```python
import cache_dit
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
).to("cuda")

# Default options, F8B0, good balance between performance and precision
cache_options = cache_dit.default_options()

# Custom options, F8B8, higher precision
cache_options = {
    "cache_type": cache_dit.DBCache,
    "warmup_steps": 8,
    "max_cached_steps": -1, # -1 means no limit
    "Fn_compute_blocks": 8, # Fn, F8, etc.
    "Bn_compute_blocks": 8, # Bn, B8, etc.
    "residual_diff_threshold": 0.12,
}

cache_dit.enable_cache(pipe, **cache_options)
```
Moreover, users configuring higher **Bn** values (e.g., **F8B16**) while aiming to maintain good performance can specify **Bn_compute_blocks_ids** to work with Bn. DBCache will only compute the specified blocks, with the remaining estimated using the previous step's residual cache.

```python
# Custom options, F8B16, higher precision with good performance.
cache_options = {
    # 0, 2, 4, ..., 14, 15, etc. [0,16)
    "Bn_compute_blocks_ids": cache_dit.block_range(0, 16, 2),
    # If the L1 difference is below this threshold, skip Bn blocks 
    # not in `Bn_compute_blocks_ids`(1, 3,..., etc), Otherwise, 
    # compute these blocks.
    "non_compute_blocks_diff_threshold": 0.08,
}
```

## 🔥Hybrid TaylorSeer

<div id="taylorseer"></div>

We have supported the [TaylorSeers: From Reusing to Forecasting: Accelerating Diffusion Models with TaylorSeers](https://arxiv.org/pdf/2503.06923) algorithm to further improve the precision of DBCache in cases where the cached steps are large, namely, **Hybrid TaylorSeer + DBCache**. At timesteps with significant intervals, the feature similarity in diffusion models decreases substantially, significantly harming the generation quality. 

$$
\mathcal{F}\_{\text {pred }, m}\left(x_{t-k}^l\right)=\mathcal{F}\left(x_t^l\right)+\sum_{i=1}^m \frac{\Delta^i \mathcal{F}\left(x_t^l\right)}{i!\cdot N^i}(-k)^i
$$

**TaylorSeer** employs a differential method to approximate the higher-order derivatives of features and predict features in future timesteps with Taylor series expansion. The TaylorSeer implemented in cache-dit supports both hidden states and residual cache types. That is $\mathcal{F}\_{\text {pred }, m}\left(x_{t-k}^l\right)$ can be a residual cache or a hidden-state cache.

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
    "warmup_steps": 3, # prefer: >= n_derivatives + 1
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

## ⚡️Hybrid Cache CFG

<div id="cfg"></div>

cache-dit supports caching for **CFG (classifier-free guidance)**. For models that fuse CFG and non-CFG into a single forward step, or models that do not include CFG (classifier-free guidance) in the forward step, please set `do_separate_classifier_free_guidance` param to **False (default)**. Otherwise, set it to True. For examples:

```python
cache_options = {
    # CFG: classifier free guidance or not
    # For model that fused CFG and non-CFG into single forward step,
    # should set do_separate_classifier_free_guidance as False.
    # For example, set it as True for Wan 2.1 and set it as False 
    # for FLUX.1, HunyuanVideo, CogVideoX, Mochi.
    "do_separate_classifier_free_guidance": True, # Wan 2.1, Qwen-Image
    # Compute cfg forward first or not, default False, namely, 
    # 0, 2, 4, ..., -> non-CFG step; 1, 3, 5, ... -> CFG step.
    "cfg_compute_first": False,
    # Compute spearate diff values for CFG and non-CFG step, 
    # default True. If False, we will use the computed diff from 
    # current non-CFG transformer step for current CFG step.
    "cfg_diff_compute_separate": True,
}
```

## 🔥Torch Compile

<div id="compile"></div>  

By the way, **cache-dit** is designed to work compatibly with **torch.compile.** You can easily use cache-dit with torch.compile to further achieve a better performance. For example:

```python
cache_dit.enable_cache(
    pipe, **cache_dit.default_options()
)
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
