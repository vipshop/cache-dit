<div align="center">
  <p align="center">
    <h2 align="center">
        <img src=https://github.com/vipshop/cache-dit/raw/main/assets/cache-dit-logo.png height="90" align="left">
        A Unified and Flexible Inference Engine with 🤗🎉<br>Hybrid Cache Acceleration and Parallelism for DiTs<br>
        <a href="https://pepy.tech/projects/cache-dit"><img src=https://static.pepy.tech/personalized-badge/cache-dit?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=BLUE&left_text=downloads></a>
        <img src=https://img.shields.io/github/license/vipshop/cache-dit.svg?color=blue >
        <img src=https://img.shields.io/github/release/vipshop/cache-dit.svg >
        <a href="https://huggingface.co/docs/diffusers/main/en/optimization/cache_dit"><img src=https://img.shields.io/badge/🤗Diffusers-ecosystem-yellow.svg ></a> 
        <a href="https://hellogithub.com/repository/vipshop/cache-dit" target="_blank"><img src="https://api.hellogithub.com/v1/widgets/recommend.svg?rid=b8b03b3b32a449ea84cfc2b96cd384f3&claim_uid=ofSCbzTmdeQk3FD&theme=small" alt="Featured｜HelloGitHub" /></a> 
        <img src=https://img.shields.io/badge/Models-30+-hotpink.svg >
    </h2>
  </p>
</div>

<!--
<a href="https://pepy.tech/projects/cache-dit"><img src=https://static.pepy.tech/personalized-badge/cache-dit?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=GREEN&left_text=downloads></a>
<a href="https://pypi.org/project/cache-dit/"><img src=https://img.shields.io/pypi/dm/cache-dit.svg ></a> 
<img src=https://img.shields.io/github/stars/vipshop/cache-dit.svg?style=dark >
-->

## 🔥Hightlight

We are excited to announce that the **first API-stable version (v1.0.0)** of cache-dit has finally been released!
**[cache-dit](https://github.com/vipshop/cache-dit)** is a **Unified** and **Flexible** Inference Engine for 🤗Diffusers, enabling acceleration with just ♥️**one line**♥️ of code. Key features: **Unified Cache APIs**, **Forward Pattern Matching**, **Automatic Block Adapter**, **DBCache**, **DBPrune**, **Hybrid TaylorSeer Calibrator**, **Hybrid Cache CFG**, **Context Parallelism**, **Tensor Parallelism**, **Torch Compile Compatible** and **🎉SOTA** performance.

```bash
pip3 install -U cache-dit # pip3 install git+https://github.com/vipshop/cache-dit.git
```
You can install the stable release of cache-dit from PyPI, or the latest development version from GitHub. Then try ♥️ Cache Acceleration with just **one line** of code ~ ♥️
```python
>>> import cache_dit
>>> from diffusers import DiffusionPipeline
>>> pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image") # Can be any diffusion pipeline
>>> cache_dit.enable_cache(pipe) # One-line code with default cache options.
>>> output = pipe(...) # Just call the pipe as normal.
>>> stats = cache_dit.summary(pipe) # Then, get the summary of cache acceleration stats.
>>> cache_dit.disable_cache(pipe) # Disable cache and run original pipe.
```

### 📚Core Features

- **[🎉Full 🤗Diffusers Support](./docs/User_Guide.md#supported-pipelines)**: Notably, **[cache-dit](https://github.com/vipshop/cache-dit)** now supports nearly **all** of Diffusers' **DiT-based** pipelines, include **[30+](./examples/pipeline/)** series, nearly **[100+](./examples/pipeline/)** pipelines, such as FLUX.1, Qwen-Image, Qwen-Image-Lightning, Wan 2.1/2.2, HunyuanImage-2.1, HunyuanVideo, HiDream, AuraFlow, CogView3Plus, CogView4, CogVideoX, LTXVideo, ConsisID, SkyReelsV2, VisualCloze, PixArt, Chroma, Mochi, SD 3.5, DiT-XL, etc.  
- **[🎉Extremely Easy to Use](./docs/User_Guide.md#unified-cache-apis)**: In most cases, you only need **one line** of code: `cache_dit.enable_cache(...)`. After calling this API, just use the pipeline as normal.   
- **[🎉Easy New Model Integration](./docs/User_Guide.md#automatic-block-adapter)**: Features like **Unified Cache APIs**, **Forward Pattern Matching**, **Automatic Block Adapter**, **Hybrid Forward Pattern**, and **Patch Functor** make it highly functional and flexible. For example, we achieved 🎉 Day 1 support for [HunyuanImage-2.1](https://github.com/Tencent-Hunyuan/HunyuanImage-2.1) with 1.7x speedup w/o precision loss—even before it was available in the Diffusers library.  
- **[🎉State-of-the-Art Performance](./bench/)**: Compared with algorithms including Δ-DiT, Chipmunk, FORA, DuCa, TaylorSeer and FoCa, cache-dit achieved the **SOTA** performance w/ **7.4x↑🎉** speedup on ClipScore!
- **[🎉Support for 4/8-Steps Distilled Models](./bench/)**: Surprisingly, cache-dit's **DBCache** works for extremely few-step distilled models—something many other methods fail to do.  
- **[🎉Compatibility with Other Optimizations](./docs/User_Guide.md#️torch-compile)**: Designed to work seamlessly with torch.compile, Quantization ([torchao](./examples/quantize/), [🔥nunchaku](./examples/quantize/)), CPU or Sequential Offloading, **[🔥Context Parallelism](./docs/User_Guide.md/#️hybrid-context-parallelism)**, **[🔥Tensor Parallelism](./docs/User_Guide.md#️hybrid-tensor-parallelism)**, etc.  
- **[🎉Hybrid Cache Acceleration](./docs/User_Guide.md#taylorseer-calibrator)**: Now supports hybrid **Block-wise Cache + Calibrator** schemes (e.g., DBCache or DBPrune + TaylorSeerCalibrator). DBCache or DBPrune acts as the **Indicator** to decide *when* to cache, while the Calibrator decides *how* to cache. More mainstream cache acceleration algorithms (e.g., FoCa) will be supported in the future, along with additional benchmarks—stay tuned for updates!  
- **[🤗Diffusers Ecosystem Integration](https://huggingface.co/docs/diffusers/main/en/optimization/cache_dit)**: 🔥**cache-dit** has joined the Diffusers community ecosystem as the **first** DiT-specific cache acceleration framework! Check out the documentation here: <a href="https://huggingface.co/docs/diffusers/main/en/optimization/cache_dit"><img src=https://img.shields.io/badge/🤗Diffusers-ecosystem-yellow.svg ></a>

![](https://github.com/vipshop/cache-dit/raw/main/assets/clip-score-bench.png)

## 🔥 Supported DiTs

> [!Tip] 
> One **Model Series** may contain **many** pipelines. cache-dit applies optimizations at the **Transformer** level; thus, any pipelines that include the supported transformer are already supported by cache-dit. ✅: known work and official supported now; ✖️: unofficial supported now, but maybe support in the future; **4-bits**: w/ nunchaku + svdq int4.

<div align="center">

| 📚Model | Cache  | CP | TP | 📚Model | Cache  | CP | TP |
|:---|:---|:---|:---|:---|:---|:---|:---|
| **🎉[FLUX.1](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✅ | **🎉[FLUX.1 4-bits](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✖️ |
| **🎉[Qwen-Image](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✅ | **🎉[Qwen-Image 4-bits](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✖️ |
| **🎉[Qwen...Lightning](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✅ | **🎉[Qwen...Lightning 4-bits](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✖️ |
| **🎉[CogVideoX](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✖️ | **🎉[OmniGen](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ |
| **🎉[Wan 2.1](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✅ | **🎉[PixArt Sigma](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✖️ |
| **🎉[Wan 2.1 VACE](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✅ | **🎉[PixArt Alpha](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✖️ |
| **🎉[Wan 2.2](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✅ | **🎉[CogVideoX 1.5](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✖️ |
| **🎉[HunyuanVideo](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✅ | **🎉[Sana](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ |
| **🎉[LTX](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✖️ | **🎉[VisualCloze](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✅ |
| **🎉[Allegro](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ | **🎉[AuraFlow](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ |
| **🎉[CogView4](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✖️ | **🎉[ShapE](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ |
| **🎉[CogView3Plus](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✖️ | **🎉[Chroma](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ️✅ |
| **🎉[Cosmos](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ | **🎉[HiDream](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ |
| **🎉[EasyAnimate](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ | **🎉[HunyuanDiT](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✅ |
| **🎉[SkyReelsV2](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ | **🎉[HunyuanDiTPAG](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ |
| **🎉[StableDiffusion3](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ | **🎉[Kandinsky5](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✅️ |
| **🎉[ConsisID](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✖️ | **🎉[PRX](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ |
| **🎉[DiT](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✖️ | **🎉[HunyuanImage](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✅ | ✅ |
| **🎉[Amused](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ | **🎉[LongCatVideo](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ |
| **🎉[StableAudio](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ | **🎉[Bria](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ |
| **🎉[Mochi](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✅ | **🎉[Lumina](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline)** | ✅ | ✖️ | ✖️ |

</div>

<details align='center'>
<summary>🔥<b>Click</b> here to show many <b>Image/Video</b> cases🔥</summary>
  
<p align='center'>
  🎉Now, cache-dit covers almost All Diffusers' DiT Pipelines🎉 <br>
   🔥<a href="./examples/pipeline">Qwen-Image</a> | <a href="./examples/pipeline">Qwen-Image-Edit</a> | <a href="./examples/pipeline">Qwen-Image-Edit-Plus </a> 🔥<br>
    🔥<a href="./examples/pipeline">FLUX.1</a> | <a href="./examples/pipeline">Qwen-Image-Lightning 4/8 Steps</a> | <a href="./examples/pipeline"> Wan 2.1 </a> | <a href="./examples/pipeline"> Wan 2.2 </a>🔥<br>
    🔥<a href="./examples/pipeline">HunyuanImage-2.1</a> | <a href="./examples/pipeline">HunyuanVideo</a> | <a href="./examples/pipeline">HunyuanDiT</a> | <a href="./examples/pipeline">HiDream</a> | <a href="./examples/pipeline">AuraFlow</a>🔥<br>
    🔥<a href="./examples/pipeline">CogView3Plus</a> | <a href="./examples/pipeline">CogView4</a> | <a href="./examples/pipeline">LTXVideo</a> | <a href="./examples/pipeline">CogVideoX</a> | <a href="./examples/">CogVideoX 1.5</a> | <a href="./examples/">ConsisID</a>🔥<br>
    🔥<a href="./examples/pipeline">Cosmos</a> | <a href="./examples/pipeline">SkyReelsV2</a> | <a href="./examples/pipeline">VisualCloze</a> | <a href="./examples/pipeline">OmniGen 1/2</a> | <a href="./examples/pipeline">Lumina 1/2</a> | <a href="./examples/pipeline">PixArt</a>🔥<br>
    🔥<a href="./examples/pipeline">Chroma</a> | <a href="./examples/pipeline">Sana</a> | <a href="./examples/pipeline">Allegro</a> | <a href="./examples/pipeline">Mochi</a> | <a href="./examples/pipeline">SD 3/3.5</a> | <a href="./examples/pipeline">Amused</a> | <a href="./examples/pipeline"> ... </a> | <a href="./examples/pipeline">DiT-XL</a>🔥
</p>
  
<div align='center'>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/gifs/wan2.2.C0_Q0_NONE.gif width=124px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/gifs/wan2.2.C1_Q0_DBCACHE_F1B0_W2M8MC2_T1O2_R0.08.gif width=124px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/gifs/hunyuan_video.C0_L0_Q0_NONE.gif width=126px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/gifs/hunyuan_video.C0_L0_Q0_DBCACHE_F1B0_W8M0MC2_T0O2_R0.12_S27.gif width=126px>
  <p><b>🔥Wan2.2 MoE</b> | <a href="https://github.com/vipshop/cache-dit">+cache-dit</a>:2.0x↑🎉 | <b>HunyuanVideo</b> | <a href="https://github.com/vipshop/cache-dit">+cache-dit</a>:2.1x↑🎉</p>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/qwen-image.C0_Q0_NONE.png width=160px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/qwen-image.C1_Q0_DBCACHE_F8B0_W8M0MC0_T1O4_R0.12_S23.png width=160px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/flux.C0_Q0_NONE_T23.69s.png width=90px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/flux.C0_Q0_DBCACHE_F1B0_W4M0MC0_T1O2_R0.15_S16_T11.39s.png width=90px>
  <p><b>🔥Qwen-Image</b> | <a href="https://github.com/vipshop/cache-dit">+cache-dit</a>:1.8x↑🎉 | <b>FLUX.1-dev</b> | <a href="https://github.com/vipshop/cache-dit">+cache-dit</a>:2.1x↑🎉</p>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/qwen-image-lightning.4steps.C0_L1_Q0_NONE.png width=160px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/qwen-image-lightning.4steps.C0_L1_Q0_DBCACHE_F16B16_W2M1MC1_T0O2_R0.9_S1.png width=160px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/hunyuan-image-2.1.C0_L0_Q1_fp8_w8a16_wo_NONE.png width=90px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/hunyuan-image-2.1.C0_L0_Q1_fp8_w8a16_wo_DBCACHE_F8B0_W8M0MC2_T1O2_R0.12_S25.png width=90px>
  <p><b>🔥Qwen...Lightning</b> | <a href="https://github.com/vipshop/cache-dit">+cache-dit</a>:1.14x↑🎉 | <b>HunyuanImage</b> | <a href="https://github.com/vipshop/cache-dit">+cache-dit</a>:1.7x↑🎉</p>
  <img src=https://github.com/vipshop/cache-dit/raw/main/examples/data/bear.png width=125px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/qwen-image-edit.C0_L0_Q0_NONE.png width=125px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/qwen-image-edit.C0_L0_Q0_DBCACHE_F8B0_W8M0MC0_T0O2_R0.08_S18.png width=125px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/qwen-image-edit.C0_L0_Q0_DBCACHE_F1B0_W8M0MC2_T0O2_R0.12_S24.png width=125px>
  <p><b>🔥Qwen-Image-Edit</b> | Input w/o Edit | Baseline | <a href="https://github.com/vipshop/cache-dit">+cache-dit</a>:1.6x↑🎉 | 1.9x↑🎉 </p>
</div>
<div align='center'>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/flux-kontext-cat.C0_L0_Q0_NONE.png width=100px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/flux-kontext.C0_L0_Q0_NONE.png width=100px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/flux-kontext.C0_L0_Q0_DBCACHE_F8B0_W8M0MC0_T0O2_R0.08_S10.png width=100px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/flux-kontext.C0_L0_Q0_DBCACHE_F1B0_W8M0MC2_T0O2_R0.12_S12.png width=100px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/flux-kontext.C0_L0_Q0_DBCACHE_F1B0_W2M0MC2_T0O2_R0.15_S15.png width=100px>
  <p><b>🔥FLUX-Kontext-dev</b> | Baseline | <a href="https://github.com/vipshop/cache-dit">+cache-dit</a>:1.3x↑🎉 | 1.7x↑🎉 | 2.0x↑ 🎉</p>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/hidream.C0_L0_Q0_NONE.png width=100px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/hidream.C0_L0_Q0_DBCACHE_F1B0_W8M0MC0_T0O2_R0.08_S24.png width=100px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/cogview4.C0_L0_Q0_NONE.png width=100px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/cogview4.C0_L0_Q0_DBCACHE_F8B0_W8M0MC0_T0O2_R0.08_S15.png width=100px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/cogview4.C0_L0_Q0_DBCACHE_F1B0_W4M0MC4_T0O2_R0.2_S22.png width=100px>
  <p><b>🔥HiDream-I1</b> | <a href="https://github.com/vipshop/cache-dit">+cache-dit</a>:1.9x↑🎉 | <b>CogView4</b> | <a href="https://github.com/vipshop/cache-dit">+cache-dit</a>:1.4x↑🎉 | 1.7x↑🎉</p>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/cogview3_plus.C0_L0_Q0_NONE.png width=100px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/cogview3_plus.C0_L0_Q0_DBCACHE_F8B0_W8M0MC0_T0O2_R0.08_S15.png width=100px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/cogview3_plus.C0_L0_Q0_DBCACHE_F1B0_W8M0MC2_T0O2_R0.08_S25.png width=100px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/chroma1-hd.C0_L0_Q0_NONE.png width=100px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/chroma1-hd.C0_L0_Q0_DBCACHE_F1B0_W8M0MC0_T0O2_R0.08_S20.png width=100px>
  <p><b>🔥CogView3</b> | <a href="https://github.com/vipshop/cache-dit">+cache-dit</a>:1.5x↑🎉 | 2.0x↑🎉| <b>Chroma1-HD</b> | <a href="https://github.com/vipshop/cache-dit">+cache-dit</a>:1.9x↑🎉</p>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/gifs/mochi.C0_L0_Q0_NONE.gif width=125px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/gifs/mochi.C0_L0_Q0_DBCACHE_F8B0_W8M0MC0_T0O2_R0.08_S34.gif width=125px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/gifs/skyreels_v2.C0_L0_Q0_NONE.gif width=125px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/gifs/skyreels_v2.C0_L0_Q0_DBCACHE_F8B0_W8M0MC0_T0O2_R0.12_S17.gif width=125px>
  <p><b>🔥Mochi-1-preview</b> | <a href="https://github.com/vipshop/cache-dit">+cache-dit</a>:1.8x↑🎉 | <b>SkyReelsV2</b> | <a href="https://github.com/vipshop/cache-dit">+cache-dit</a>:1.6x↑🎉</p>
  <img src=https://github.com/vipshop/cache-dit/raw/main/examples/data/visualcloze/00555_00.jpg width=100px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/examples/data/visualcloze/12265_00.jpg width=100px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/visualcloze-512.C0_L0_Q0_NONE.png width=100px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/visualcloze-512.C0_L0_Q0_DBCACHE_F8B0_W8M0MC0_T0O2_R0.08_S15.png width=100px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/visualcloze-512.C0_L0_Q0_DBCACHE_F1B0_W8M0MC0_T0O2_R0.08_S18.png width=100px>
  <p><b>🔥VisualCloze-512</b> | Model | Cloth | Baseline | <a href="https://github.com/vipshop/cache-dit">+cache-dit</a>:1.4x↑🎉 | 1.7x↑🎉 </p>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/gifs/ltx-video.C0_L0_Q0_NONE.gif width=144px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/gifs/ltx-video.C0_L0_Q0_DBCACHE_F1B0_W8M0MC0_T0O2_R0.15_S13.gif width=144px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/gifs/cogvideox1.5.C0_L0_Q0_NONE.gif width=105px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/gifs/cogvideox1.5.C0_L0_Q0_DBCACHE_F1B0_W8M0MC0_T0O2_R0.12_S22.gif width=105px>
  <p><b>🔥LTX-Video-0.9.7</b> | <a href="https://github.com/vipshop/cache-dit">+cache-dit</a>:1.7x↑🎉 | <b>CogVideoX1.5</b> | <a href="https://github.com/vipshop/cache-dit">+cache-dit</a>:2.0x↑🎉</p>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/omingen-v1.C0_L0_Q0_NONE.png width=100px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/omingen-v1.C0_L0_Q0_DBCACHE_F8B0_W8M0MC0_T0O2_R0.08_S24.png width=100px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/omingen-v1.C0_L0_Q0_DBCACHE_F1B0_W8M0MC0_T1O2_R0.08_S38.png width=100px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/lumina2.C0_L0_Q0_NONE.png width=100px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/lumina2.C0_L0_Q0_DBCACHE_F1B0_W2M0MC2_T0O2_R0.12_S14.png width=100px>
  <p><b>🔥OmniGen-v1</b> | <a href="https://github.com/vipshop/cache-dit">+cache-dit</a>:1.5x↑🎉 | 3.3x↑🎉 | <b>Lumina2</b> | <a href="https://github.com/vipshop/cache-dit">+cache-dit</a>:1.9x↑🎉</p>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/gifs/allegro.C0_L0_Q0_NONE.gif width=117px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/gifs/allegro.C0_L0_Q0_DBCACHE_F8B0_W8M0MC0_T0O2_R0.26_S27.gif width=117px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/auraflow.C0_L0_Q0_NONE.png width=133px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/auraflow.C0_L0_Q0_DBCACHE_F1B0_W8M0MC2_T0O2_R0.08_S28.png width=133px>
  <p><b>🔥Allegro</b> | <a href="https://github.com/vipshop/cache-dit">+cache-dit</a>:1.36x↑🎉 | <b>AuraFlow-v0.3</b> | <a href="https://github.com/vipshop/cache-dit">+cache-dit</a>:2.27x↑🎉 </p>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/sana.C0_L0_Q0_NONE.png width=100px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/sana.C0_L0_Q0_DBCACHE_F8B0_W8M0MC2_T0O2_R0.25_S6.png width=100px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/sana.C0_L0_Q0_DBCACHE_F1B0_W8M0MC2_T0O2_R0.3_S8.png width=100px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/pixart-sigma.C0_L0_Q0_NONE.png width=100px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/pixart-sigma.C0_L0_Q0_DBCACHE_F8B0_W8M0MC0_T0O2_R0.08_S28.png width=100px>
  <p><b>🔥Sana</b> | <a href="https://github.com/vipshop/cache-dit">+cache-dit</a>:1.3x↑🎉 | 1.6x↑🎉| <b>PixArt-Sigma</b> | <a href="https://github.com/vipshop/cache-dit">+cache-dit</a>:2.3x↑🎉</p>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/pixart-alpha.C0_L0_Q0_NONE.png width=100px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/pixart-alpha.C0_L0_Q0_DBCACHE_F8B0_W8M0MC0_T0O2_R0.05_S27.png width=100px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/pixart-alpha.C0_L0_Q0_DBCACHE_F8B0_W8M0MC0_T0O2_R0.08_S32.png width=100px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/sd_3_5.C0_L0_Q0_NONE.png width=100px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/sd_3_5.C0_L0_Q0_DBCACHE_F1B0_W8M0MC3_T0O2_R0.12_S30.png width=100px>
  <p><b>🔥PixArt-Alpha</b> | <a href="https://github.com/vipshop/cache-dit">+cache-dit</a>:1.6x↑🎉 | 1.8x↑🎉| <b>SD 3.5</b> | <a href="https://github.com/vipshop/cache-dit">+cache-dit</a>:2.5x↑🎉</p>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/amused.C0_L0_Q0_NONE.png width=100px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/amused.C0_L0_Q0_DBCACHE_F8B0_W8M0MC0_T0O2_R0.34_S1.png width=100px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/amused.C0_L0_Q0_DBCACHE_F8B0_W8M0MC0_T0O2_R0.38_S2.png width=100px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/dit-xl.C0_L0_Q0_NONE.png width=100px>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/dit-xl.C0_L0_Q0_DBCACHE_F1B0_W8M0MC2_T0O2_R0.15_S11.png width=100px>
  <p><b>🔥Asumed</b> | <a href="https://github.com/vipshop/cache-dit">+cache-dit</a>:1.1x↑🎉 | 1.2x↑🎉 | <b>DiT-XL-256</b> | <a href="https://github.com/vipshop/cache-dit">+cache-dit</a>:1.8x↑🎉
  <br>♥️ Please consider to leave a <b>⭐️ Star</b> to support us ~ ♥️</p>
</div>

</details>

## 📖Table of Contents

<div id="user-guide"></div>

For more advanced features such as **Unified Cache APIs**, **Forward Pattern Matching**, **Automatic Block Adapter**, **Hybrid Forward Pattern**, **Patch Functor**, **DBCache**, **DBPrune**, **TaylorSeer Calibrator**, **Hybrid Cache CFG**, **Context Parallelism** and **Tensor Parallelism**, please refer to the [🎉User_Guide.md](./docs/User_Guide.md) for details.

- [⚙️Installation](./docs/User_Guide.md#️installation)
- [🔥Supported DiTs](./docs/User_Guide.md#supported)
- [🔥Benchmarks](./docs/User_Guide.md#benchmarks)
- [🎉Unified Cache APIs](./docs/User_Guide.md#unified-cache-apis)
  - [📚Forward Pattern Matching](./docs/User_Guide.md#forward-pattern-matching)
  - [📚Cache with One-line Code](./docs/User_Guide.md#%EF%B8%8Fcache-acceleration-with-one-line-code)
  - [🔥Automatic Block Adapter](./docs/User_Guide.md#automatic-block-adapter)
  - [📚Hybrid Forward Pattern](./docs/User_Guide.md#hybrid-forward-pattern)
  - [📚Implement Patch Functor](./docs/User_Guide.md#implement-patch-functor)
  - [📚Transformer-Only Interface](./docs/User_Guide.md#transformer-only-interface)
  - [📚How to use ParamsModifier](./docs/User_Guide.md#how-to-use-paramsmodifier)
  - [🤖Cache Acceleration Stats](./docs/User_Guide.md#cache-acceleration-stats-summary)
- [⚡️DBCache: Dual Block Cache](./docs/User_Guide.md#️dbcache-dual-block-cache)
- [⚡️DBPrune: Dynamic Block Prune](./docs/User_Guide.md#️dbprune-dynamic-block-prune)
- [⚡️Hybrid Cache CFG](./docs/User_Guide.md#️hybrid-cache-cfg)
- [🔥Hybrid TaylorSeer Calibrator](./docs/User_Guide.md#taylorseer-calibrator)
- [⚡️Hybrid Context Parallelism](./docs/User_Guide.md#context-parallelism)
- [⚡️Hybrid Tensor Parallelism](./docs/User_Guide.md#tensor-parallelism)
- [🤖Low-bits Quantization](./docs/User_Guide.md#quantization)
- [🛠Metrics Command Line](./docs/User_Guide.md#metrics-cli)
- [⚙️Torch Compile](./docs/User_Guide.md#️torch-compile)
- [📚API Documents](./docs/User_Guide.md#api-documentation)

## 👋Contribute 
<div id="contribute"></div>

How to contribute? Star ⭐️ this repo to support us or check [CONTRIBUTE.md](https://github.com/vipshop/cache-dit/raw/main/CONTRIBUTE.md).

<div align='center'>
<a href="https://star-history.com/#vipshop/cache-dit&Date">
  <picture align='center'>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=vipshop/cache-dit&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=vipshop/cache-dit&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=vipshop/cache-dit&type=Date" width=400px />
  </picture>
</a>

</div>

## 🎉Projects Using CacheDiT

Here is a curated list of open-source projects integrating **CacheDiT**, including popular repositories like [jetson-containers](https://github.com/dusty-nv/jetson-containers/blob/master/packages/diffusion/cache_edit/build.sh), [flux-fast](https://github.com/huggingface/flux-fast), and [sdnext](https://github.com/vladmandic/sdnext/discussions/4269). 🎉**CacheDiT** has been **recommended** by: [Wan2.2](https://github.com/Wan-Video/Wan2.2), [Qwen-Image-Lightning](https://github.com/ModelTC/Qwen-Image-Lightning), [Qwen-Image](https://github.com/QwenLM/Qwen-Image), [LongCat-Video](https://github.com/meituan-longcat/LongCat-Video), [Kandinsky-5](https://github.com/ai-forever/Kandinsky-5), [🤗diffusers](https://huggingface.co/docs/diffusers/main/en/optimization/cache_dit) and [HelloGitHub](https://hellogithub.com/repository/vipshop/cache-dit), among others.

## ©️Acknowledgements

Special thanks to vipshop's Computer Vision AI Team for supporting document, testing and production-level deployment of this project.

## ©️Citations

<div id="citations"></div>

```BibTeX
@misc{cache-dit@2025,
  title={cache-dit: A Unified and Flexible Inference Engine with Hybrid Cache Acceleration and Parallelism for Diffusers.},
  url={https://github.com/vipshop/cache-dit.git},
  note={Open-source software available at https://github.com/vipshop/cache-dit.git},
  author={DefTruth, vipshop.com},
  year={2025}
}
```
