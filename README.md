<div align="center">
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/cache-dit-logo.png height="120">

  <p align="center">
    A <b>Unified</b>, Flexible and Training-free <b>Cache Acceleration</b> Framework for <b>🤗Diffusers</b> <br>
    ♥️ Cache Acceleration with <b>One-line</b> Code ~ ♥️
  </p>
  <div align='center'>
      <img src="./assets/image-reward-bench.png" width=580px >
  </div>
  <div align='center'>
      <a href="https://huggingface.co/docs/diffusers/main/en/optimization/cache_dit"><img src=https://img.shields.io/badge/🤗Diffusers-ecosystem-yellow.svg ></a>
      <img src=https://img.shields.io/badge/Language-Python-brightgreen.svg >
      <img src=https://img.shields.io/badge/PRs-welcome-blue.svg >
      <img src=https://img.shields.io/badge/PyPI-pass-brightgreen.svg >
      <img src=https://static.pepy.tech/badge/cache-dit >
      <img src=https://img.shields.io/github/stars/vipshop/cache-dit.svg?style=dark >
  </div>
  <div align='center'>
      <a href="./README.md">📚English</a> | <a href="./README_CN.md">📚中文阅读 </a> | <a href="./docs/User_Guide.md#api-documentation"> 📚API Documentation </a> | <a href="https://huggingface.co/docs/diffusers/main/en/optimization/cache_dit">🤗Diffusers' Docs</a>
  </div>
  <p align="center">
    🎉Now, <b>cache-dit</b> covers almost <b>All</b> Diffusers' <b>DiT</b> Pipelines🎉<br>
    🔥<a href="#supported">Qwen-Image</a> | <a href="#supported">FLUX.1</a> | <a href="#supported">Qwen-Image-Lightning</a> | <a href="#supported"> Wan 2.1 </a> | <a href="#supported"> Wan 2.2 </a>🔥<br>
    🔥<a href="#supported">HunyuanImage-2.1</a> | <a href="#supported">HunyuanVideo</a> | <a href="#supported">HunyuanDiT</a> | <a href="#supported">HiDream</a> | <a href="#supported">AuraFlow</a>🔥<br>
    🔥<a href="#supported">CogView3Plus</a> | <a href="#supported">CogView4</a> | <a href="#supported">LTXVideo</a> | <a href="#supported">CogVideoX</a> | <a href="#supported">CogVideoX 1.5</a> | <a href="#supported">ConsisID</a>🔥<br>
    🔥<a href="#supported">Cosmos</a> | <a href="#supported">SkyReelsV2</a> | <a href="#supported">VisualCloze</a> | <a href="#supported">OmniGen 1/2</a> | <a href="#supported">Lumina 1/2</a> | <a href="#supported">PixArt</a>🔥<br>
    🔥<a href="#supported">Chroma</a> | <a href="#supported">Sana</a> | <a href="#supported">Allegro</a> | <a href="#supported">Mochi</a> | <a href="#supported">SD 3/3.5</a> | <a href="#supported">Amused</a> | <a href="#supported"> ... </a> | <a href="#supported">DiT-XL</a>🔥
  </p>
</div>


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
  <p><b>🔥Qwen-Image-Edit</b> | Input w/o Edit | Baseline | <a href="https://github.com/vipshop/cache-dit">+cache-dit</a>:1.6x↑🎉 | 1.9x↑🎉 
  <br>♥️ Please consider to leave a <b>⭐️ Star</b> to support us ~ ♥️
  </p>
</div>

<details align='center'>

<summary>Click here to show more Image/Video cases</summary>

<div  align='center'>
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

## 🔥Hightlight

We are excited to announce that the **first API-stable version (v1.0.0)** of cache-dit has finally been released!

**[cache-dit](https://github.com/vipshop/cache-dit)** is a **Unified**, **Flexible**, and **Training-free** cache acceleration framework for 🤗 Diffusers, enabling cache acceleration with just **one line** of code. Key features include **Unified Cache APIs**, **Forward Pattern Matching**, **Automatic Block Adapter**, **Hybrid Forward Pattern**, **DBCache**, **TaylorSeer Calibrator**, and **Cache CFG**.   

### 📚Core Features

- **Full 🤗 Diffusers Support**: Notably, **[cache-dit](https://github.com/vipshop/cache-dit)** now supports nearly **all** of Diffusers' **DiT-based** pipelines, such as Qwen-Image, FLUX.1, Qwen-Image-Lightning, Wan 2.1, Wan 2.2, HunyuanImage-2.1, HunyuanVideo, HunyuanDiT, HiDream, AuraFlow, CogView3Plus, CogView4, LTXVideo, CogVideoX 1.5, ConsisID, SkyReelsV2, VisualCloze, OmniGen 1/2, Lumina 1/2, PixArt, Chroma, Sana, Allegro, Mochi, SD 3.5, Amused, and DiT-XL.  
- **Extremely Easy to Use**: In most cases, you only need **♥️ one line ♥️** of code: `cache_dit.enable_cache(...)`. After calling this API, just use the pipeline as normal.   
- **Easy New Model Integration**: Features like **Unified Cache APIs**, **Forward Pattern Matching**, **Automatic Block Adapter**, **Hybrid Forward Pattern**, and **Patch Functor** make it highly functional and flexible. For example, we achieved 🎉 Day 1 support for [HunyuanImage-2.1](https://github.com/Tencent-Hunyuan/HunyuanImage-2.1)—even before it was available in the Diffusers library.  
- **State-of-the-Art Performance**: Compared with algorithms including Δ-DiT, Chipmunk, FORA, DuCa, TaylorSeer, and FoCa, cache-dit's DBCache achieves the best accuracy when the speedup ratio is below 3x.  
- **Support for 4/8-Step Distilled Models**: Surprisingly, cache-dit's DBCache works for extremely few-step distilled models—something many other methods fail to do.  
- **Compatibility with Other Optimizations**: Designed to work seamlessly with torch.compile, model CPU offload, sequential CPU offload, group offloading, etc.  
- **Hybrid Cache Acceleration**: Now supports hybrid **DBCache + Calibrator** schemes (e.g., DBCache + TaylorSeerCalibrator). DBCache acts as the **Indicator** to decide *when* to cache, while the Calibrator decides *how* to cache. More mainstream cache acceleration algorithms (e.g., FoCa) will be supported in the future, along with additional benchmarks—stay tuned for updates!  
- **🤗 Diffusers Ecosystem Integration**: 🔥 **cache-dit** has joined the 🤗 Diffusers community ecosystem as the **first** DiT-specific cache acceleration framework! Check out the documentation here: **[Diffusers Docs](https://huggingface.co/docs/diffusers/main/en/optimization/cache_dit)**. <a href="https://huggingface.co/docs/diffusers/main/en/optimization/cache_dit"><img src=https://img.shields.io/badge/🤗Diffusers-ecosystem-yellow.svg ></a>


## 🔥News  

- [2025-09-24] 🔥**cache-dit** has now joined the 🤗 Diffusers community ecosystem as the **first** cache acceleration framework for DiTs! Check out the documentation here: **[Diffusers Docs](https://huggingface.co/docs/diffusers/main/en/optimization/cache_dit)**. <a href="https://huggingface.co/docs/diffusers/main/en/optimization/cache_dit"><img src=https://img.shields.io/badge/🤗Diffusers-ecosystem-yellow.svg ></a>
- [2025-09-10] 🎉Day 1 support [**HunyuanImage-2.1**](https://github.com/Tencent-Hunyuan/HunyuanImage-2.1) with **1.7x↑🎉** speedup! Check this [example](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline/run_hunyuan_image_2.1.py).
- [2025-09-08] 🔥[**Qwen-Image-Lightning**](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline/run_qwen_image_lightning.py) **7.1/3.5 steps🎉** inference with **[DBCache: F16B16](https://github.com/vipshop/cache-dit)**.
- [2025-09-03] 🎉[**Wan2.2-MoE**](https://github.com/Wan-Video) **2.4x↑🎉** speedup! Please refer to [run_wan_2.2.py](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline/run_wan_2.2.py) as an example.
- [2025-08-19] 🔥[**Qwen-Image-Edit**](https://github.com/QwenLM/Qwen-Image) **2x↑🎉** speedup! Check the example: [run_qwen_image_edit.py](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline/run_qwen_image_edit.py).
- [2025-08-11] 🔥[**Qwen-Image**](https://github.com/QwenLM/Qwen-Image) **1.8x↑🎉** speedup! Please refer to [run_qwen_image.py](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline/run_qwen_image.py) as an example.

<details>
<summary> Previous News </summary>  

- [2025-07-13] 🎉[**FLUX.1-dev**](https://github.com/xlite-dev/flux-faster) **3.3x↑🎉** speedup! NVIDIA L20 with **[cache-dit](https://github.com/vipshop/cache-dit)** + **compile + FP8 DQ**.
- [2025-09-08] 🎉First caching mechanism in [Qwen-Image-Lightning](https://github.com/ModelTC/Qwen-Image-Lightning) with **[cache-dit](https://github.com/vipshop/cache-dit)**, check this [PR](https://github.com/ModelTC/Qwen-Image-Lightning/pull/35). 
- [2025-09-08] 🎉First caching mechanism in [Wan2.2](https://github.com/Wan-Video/Wan2.2) with **[cache-dit](https://github.com/vipshop/cache-dit)**, check this [PR](https://github.com/Wan-Video/Wan2.2/pull/127) for more details. 
- [2025-08-12] 🎉First caching mechanism in [QwenLM/Qwen-Image](https://github.com/QwenLM/Qwen-Image) with **[cache-dit](https://github.com/vipshop/cache-dit)**, check this [PR](https://github.com/QwenLM/Qwen-Image/pull/61). 
- [2025-09-01] 📚[**Hybird Forward Pattern**](#unified) is supported! Please check [FLUX.1-dev](https://github.com/vipshop/cache-dit/blob/main/examples/run_flux_adapter.py) as an example.
- [2025-08-10] 🔥[**FLUX.1-Kontext-dev**](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) is supported! Please refer [run_flux_kontext.py](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline/run_flux_kontext.py) as an example.
- [2025-07-18] 🎉First caching mechanism in [🤗huggingface/flux-fast](https://github.com/huggingface/flux-fast) with **[cache-dit](https://github.com/vipshop/cache-dit)**, check the [PR](https://github.com/huggingface/flux-fast/pull/13). 

</details>

## 📖Contents 

<div id="contents"></div>  

- [⚙️Installation](#️installation)
- [🔥Benchmarks](#benchmarks)
- [🔥Quick Start](#quick-start)
- [📚Pattern Matching](#forward-pattern-matching)
- [⚡️Dual Block Cache](#dbcache)
- [🔥TaylorSeer Calibrator](#taylorseer)
- [📚Hybrid Cache CFG](#cfg)
- [🎉User Guide](#user-guide)
- [©️Citations](#citations)

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

## 🔥Benchmarks

<div id="benchmarks"></div>

![image-reward-bench](https://github.com/vipshop/cache-dit/raw/main/assets/image-reward-bench.png)

The comparison between **cache-dit: DBCache** and algorithms such as Δ-DiT, Chipmunk, FORA, DuCa, TaylorSeer and FoCa is as follows. Now, in the comparison with a speedup ratio less than **3x**, cache-dit achieved the best accuracy. Surprisingly, cache-dit: DBCache still works in the extremely few-step distill model. For a complete benchmark, please refer to [📚Benchmarks](https://github.com/vipshop/cache-dit/raw/main/bench/). 

| Method | TFLOPs(↓) | SpeedUp(↑) | ImageReward(↑) | Clip Score(↑) |
| --- | --- | --- | --- | --- |
| [**FLUX.1**-dev]: 50 steps | 3726.87 | 1.00× | 0.9898 | 32.404 |
| [**FLUX.1**-dev]: 60% steps | 2231.70 | 1.67× | 0.9663 | 32.312 |
| Δ-DiT(N=2) | 2480.01 | 1.50× | 0.9444 | 32.273 |
| Δ-DiT(N=3) | 1686.76 | 2.21× | 0.8721 | 32.102 |
| [**FLUX.1**-dev]: 34% steps | 1264.63 | 3.13× | 0.9453 | 32.114 |
| Chipmunk | 1505.87 | 2.47× | 0.9936 | 32.776 |
| FORA(N=3) | 1320.07 | 2.82× | 0.9776 | 32.266 |
| **[DBCache(F=4,B=0,W=4,MC=4)](https://github.com/vipshop/cache-dit)** | 1400.08 | **2.66×** | **1.0065** | 32.838 |
| **[DBCache+TaylorSeer(F=1,B=0,O=1)](https://github.com/vipshop/cache-dit)** | 1153.05 | **3.23×** | **1.0221** | 32.819 |
| DuCa(N=5) | 978.76 | 3.80× | 0.9955 | 32.241 |
| TaylorSeer(N=4,O=2) | 1042.27 | 3.57× | 0.9857 | 32.413 |
| **[DBCache(F=1,B=0,W=4,MC=6)](https://github.com/vipshop/cache-dit)** | 944.75 | **3.94×** | 0.9997 | 32.849 |
| **[DBCache+TaylorSeer(F=1,B=0,O=1)](https://github.com/vipshop/cache-dit)** | 944.75 | **3.94×** | **1.0107** | 32.865 |
| **[FoCa(N=5): arxiv.2508.16211](https://arxiv.org/pdf/2508.16211)** | 893.54 | **4.16×** | **1.0029** | **32.948** |

<details>
<summary> Show all comparison </summary>  

![clip-score-bench](https://github.com/vipshop/cache-dit/raw/main/assets/clip-score-bench.png)

| Method | TFLOPs(↓) | SpeedUp(↑) | ImageReward(↑) | Clip Score(↑) |
| --- | --- | --- | --- | --- |
| [**FLUX.1**-dev]: 50 steps | 3726.87 | 1.00× | 0.9898 | 32.404 |
| [**FLUX.1**-dev]: 60% steps | 2231.70 | 1.67× | 0.9663 | 32.312 |
| Δ-DiT(N=2) | 2480.01 | 1.50× | 0.9444 | 32.273 |
| Δ-DiT(N=3) | 1686.76 | 2.21× | 0.8721 | 32.102 |
| [**FLUX.1**-dev]: 34% steps | 1264.63 | 3.13× | 0.9453 | 32.114 |
| Chipmunk | 1505.87 | 2.47× | 0.9936 | 32.776 |
| FORA(N=3) | 1320.07 | 2.82× | 0.9776 | 32.266 |
| **[DBCache(F=4,B=0,W=4,MC=4)](https://github.com/vipshop/cache-dit)** | 1400.08 | **2.66×** | **1.0065** | 32.838 |
| DuCa(N=5) | 978.76 | 3.80× | 0.9955 | 32.241 |
| TaylorSeer(N=4,O=2) | 1042.27 | 3.57× | 0.9857 | 32.413 |
| **[DBCache+TaylorSeer(F=1,B=0,O=1)](https://github.com/vipshop/cache-dit)** | 1153.05 | **3.23×** | **1.0221** | 32.819 |
| **[DBCache(F=1,B=0,W=4,MC=6)](https://github.com/vipshop/cache-dit)** | 944.75 | **3.94×** | 0.9997 | 32.849 |
| **[DBCache+TaylorSeer(F=1,B=0,O=1)](https://github.com/vipshop/cache-dit)** | 944.75 | **3.94×** | **1.0107** | 32.865 |
| **[FoCa(N=5): arxiv.2508.16211](https://arxiv.org/pdf/2508.16211)** | 893.54 | **4.16×** | **1.0029** | **32.948** |
| [**FLUX.1**-dev]: 22% steps | 818.29 | 4.55× | 0.8183 | 31.772 |
| FORA(N=4) | 967.91 | 3.84× | 0.9730 | 32.142 |
| ToCa(N=8) | 784.54 | 4.74× | 0.9451 | 31.993 |
| DuCa(N=7) | 760.14 | 4.89× | 0.9757 | 32.066 |
| TeaCache(l=0.8) | 892.35 | 4.17× | 0.8683 | 31.704 |
| **[DBCache(F=4,B=0,W=4,MC=10)](https://github.com/vipshop/cache-dit)** | 816.65 | 4.56x | 0.8245 | 32.191 |
| TaylorSeer(N=5,O=2) | 893.54 | 4.16× | 0.9768 | 32.467 |
| **[FoCa(N=7): arxiv.2508.16211](https://arxiv.org/pdf/2508.16211)** | 670.44 | **5.54×** | **0.9891** | **32.920** |
| FORA(N=7) | 670.14 | 5.55× | 0.7418 | 31.519 |
| ToCa(N=12) | 644.70 | 5.77× | 0.7155 | 31.808 |
| DuCa(N=10) | 606.91 | 6.13× | 0.8382 | 31.759 |
| TeaCache(l=1.2) | 669.27 | 5.56× | 0.7394 | 31.704 |
| **[DBCache(F=1,B=0,W=4,MC=10)](https://github.com/vipshop/cache-dit)** | 651.90 | **5.72x** | 0.8796 | **32.318** |
| TaylorSeer(N=7,O=2) | 670.44 | 5.54× | 0.9128 | 32.128 |
| **[FoCa(N=8): arxiv.2508.16211](https://arxiv.org/pdf/2508.16211)** | 596.07 | **6.24×** | **0.9502** | **32.706** |

NOTE: Except for DBCache, other performance data are referenced from the paper [FoCa, arxiv.2508.16211](https://arxiv.org/pdf/2508.16211).

</details>

## 🔥Quick Start 

<div id="unified"></div>  

<div id="quick-start"></div>

In most cases, you only need to call ♥️**one-line**♥️ of code, that is `cache_dit.enable_cache(...)`. After this API is called, you just need to call the pipe as normal. The `pipe` param can be **any** Diffusion Pipeline. Please refer to [Qwen-Image](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline/run_qwen_image.py) as an example. 

```python
>>> import cache_dit
>>> from diffusers import DiffusionPipeline
>>> pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image") # Can be any diffusion pipeline
>>> cache_dit.enable_cache(pipe) # One-line code with default cache options.
>>> output = pipe(...) # Just call the pipe as normal.
>>> stats = cache_dit.summary(pipe) # Then, get the summary of cache acceleration stats.
>>> cache_dit.disable_cache(pipe) # Disable cache and run original pipe.
```

## 📚Forward Pattern Matching 

<div id="supported"></div>

<div id="forward-pattern-matching"></div>  

cache-dit works by matching specific input/output patterns as shown below.

![](https://github.com/vipshop/cache-dit/raw/main/assets/patterns-v1.png)

Please check [🎉Examples](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline) for more details. Here are just some of the tested models listed.

```python
>>> import cache_dit
>>> cache_dit.supported_pipelines()
(30, ['Flux*', 'Mochi*', 'CogVideoX*', 'Wan*', 'HunyuanVideo*', 'QwenImage*', 'LTX*', 'Allegro*',
'CogView3Plus*', 'CogView4*', 'Cosmos*', 'EasyAnimate*', 'SkyReelsV2*', 'StableDiffusion3*',
'ConsisID*', 'DiT*', 'Amused*', 'Bria*', 'Lumina*', 'OmniGen*', 'PixArt*', 'Sana*', 'StableAudio*',
'VisualCloze*', 'AuraFlow*', 'Chroma*', 'ShapE*', 'HiDream*', 'HunyuanDiT*', 'HunyuanDiTPAG*'])
```

<details>
<summary> Show all pipelines </summary>  

- [🚀HunyuanImage-2.1](https://github.com/vipshop/cache-dit/blob/main/examples)  
- [🚀Qwen-Image-Lightning](https://github.com/vipshop/cache-dit/blob/main/examples)
- [🚀Qwen-Image-Edit](https://github.com/vipshop/cache-dit/blob/main/examples)  
- [🚀Qwen-Image](https://github.com/vipshop/cache-dit/blob/main/examples)  
- [🚀FLUX.1-dev](https://github.com/vipshop/cache-dit/blob/main/examples)  
- [🚀FLUX.1-Fill-dev](https://github.com/vipshop/cache-dit/blob/main/examples)  
- [🚀FLUX.1-Kontext-dev](https://github.com/vipshop/cache-dit/blob/main/examples)
- [🚀CogView4](https://github.com/vipshop/cache-dit/blob/main/examples)
- [🚀Wan2.2-T2V](https://github.com/vipshop/cache-dit/blob/main/examples)
- [🚀HunyuanVideo](https://github.com/vipshop/cache-dit/blob/main/examples)
- [🚀HiDream-I1-Full](https://github.com/vipshop/cache-dit/blob/main/examples)
- [🚀HunyuanDiT](https://github.com/vipshop/cache-dit/blob/main/examples)
- [🚀Wan2.1-T2V](https://github.com/vipshop/cache-dit/blob/main/examples)
- [🚀Wan2.1-FLF2V](https://github.com/vipshop/cache-dit/blob/main/examples)
- [🚀SkyReelsV2](https://github.com/vipshop/cache-dit/blob/main/examples)  
- [🚀Chroma1-HD](https://github.com/vipshop/cache-dit/blob/main/examples)  
- [🚀CogVideoX1.5](https://github.com/vipshop/cache-dit/blob/main/examples)
- [🚀CogView3-Plus](https://github.com/vipshop/cache-dit/blob/main/examples)
- [🚀CogVideoX](https://github.com/vipshop/cache-dit/blob/main/examples)
- [🚀VisualCloze](https://github.com/vipshop/cache-dit/blob/main/examples)  
- [🚀LTXVideo](https://github.com/vipshop/cache-dit/blob/main/examples)  
- [🚀OmniGen](https://github.com/vipshop/cache-dit/blob/main/examples)  
- [🚀Lumina2](https://github.com/vipshop/cache-dit/blob/main/examples)  
- [🚀mochi-1-preview](https://github.com/vipshop/cache-dit/blob/main/examples)
- [🚀AuraFlow-v0.3](https://github.com/vipshop/cache-dit/blob/main/examples)
- [🚀PixArt-Alpha](https://github.com/vipshop/cache-dit/blob/main/examples)
- [🚀PixArt-Sigma](https://github.com/vipshop/cache-dit/blob/main/examples)
- [🚀NVIDIA Sana](https://github.com/vipshop/cache-dit/blob/main/examples)
- [🚀SD-3/3.5](https://github.com/vipshop/cache-dit/blob/main/examples)
- [🚀ConsisID](https://github.com/vipshop/cache-dit/blob/main/examples)
- [🚀Allegro](https://github.com/vipshop/cache-dit/blob/main/examples)
- [🚀Amused](https://github.com/vipshop/cache-dit/blob/main/examples)
- [🚀DiT-XL](https://github.com/vipshop/cache-dit/blob/main/examples)
- ...

</details>

## ⚡️DBCache: Dual Block Cache  

<div id="dbcache"></div>

![](https://github.com/vipshop/cache-dit/raw/main/assets/dbcache-v1.png)

**DBCache**: **Dual Block Caching** for Diffusion Transformers. Different configurations of compute blocks (**F8B12**, etc.) can be customized in DBCache, enabling a balanced trade-off between performance and precision. Moreover, it can be entirely **training**-**free**. Please Check the [DBCache](https://github.com/vipshop/cache-dit/blob/main/docs/DBCache.md) and [User Guide](https://github.com/vipshop/cache-dit/blob/main/docs/User_Guide.md#dbcache) docs for details.

```python
# Default options, F8B0, 8 warmup steps, and unlimited cached 
# steps for good balance between performance and precision
cache_dit.enable_cache(pipe_or_adapter)

# Custom options, F8B8, higher precision
from cache_dit import BasicCacheConfig

cache_dit.enable_cache(
    pipe_or_adapter,
    cache_config=BasicCacheConfig(
        max_warmup_steps=8,  # steps do not cache
        max_cached_steps=-1, # -1 means no limit
        Fn_compute_blocks=8, # Fn, F8, etc.
        Bn_compute_blocks=8, # Bn, B8, etc.
        residual_diff_threshold=0.12,
    ),
)
```  

## 🔥TaylorSeer Calibrator

<div id="taylorseer"></div>

The [TaylorSeers](https://huggingface.co/papers/2503.06923) algorithm further improves the precision of DBCache in cases where the cached steps are large (Hybrid TaylorSeer + DBCache). At timesteps with significant intervals, the feature similarity in diffusion models decreases substantially, significantly harming the generation quality. 

TaylorSeer employs a differential method to approximate the higher-order derivatives of features and predict features in future timesteps with Taylor series expansion. The TaylorSeer implemented in CacheDiT supports both hidden states and residual cache types. F_pred can be a residual cache or a hidden-state cache.

```python
from cache_dit import BasicCacheConfig, TaylorSeerCalibratorConfig

cache_dit.enable_cache(
    pipe_or_adapter,
    # Basic DBCache w/ FnBn configurations
    cache_config=BasicCacheConfig(
        max_warmup_steps=8,  # steps do not cache
        max_cached_steps=-1, # -1 means no limit
        Fn_compute_blocks=8, # Fn, F8, etc.
        Bn_compute_blocks=8, # Bn, B8, etc.
        residual_diff_threshold=0.12,
    ),
    # Then, you can use the TaylorSeer Calibrator to approximate 
    # the values in cached steps, taylorseer_order default is 1.
    calibrator_config=TaylorSeerCalibratorConfig(
        taylorseer_order=1,
    ),
)
``` 

> [!TIP]  
> The `Bn_compute_blocks` parameter of DBCache can be set to `0` if you use TaylorSeer as the calibrator for approximate hidden states. DBCache's `Bn_compute_blocks` also acts as a calibrator, so you can choose either `Bn_compute_blocks` > 0 or TaylorSeer. We recommend using the configuration scheme of TaylorSeer + DBCache FnB0.

## 📚Hybrid Cache CFG

<div id="cfg"></div>

cache-dit supports caching for CFG (classifier-free guidance). For models that fuse CFG and non-CFG into a single forward step, or models that do not include CFG (classifier-free guidance) in the forward step, please set `enable_separate_cfg` parameter  to `False (default, None)`. Otherwise, set it to `True`. 

```python
from cache_dit import BasicCacheConfig

cache_dit.enable_cache(
    pipe_or_adapter, 
    cache_config=BasicCacheConfig(
        ...,
        # For example, set it as True for Wan 2.1/Qwen-Image 
        # and set it as False for FLUX.1, HunyuanVideo, CogVideoX, etc.
        enable_separate_cfg=True,
    ),
)
```

## 🎉User Guide

<div id="user-guide"></div>

For more advanced features such as **Unified Cache APIs**, **Forward Pattern Matching**, **Automatic Block Adapter**, **Hybrid Forward Pattern**, **DBCache**, **TaylorSeer Calibrator**, and **Hybrid Cache CFG**, please refer to the [🎉User_Guide.md](./docs/User_Guide.md) for details.

- [⚙️Installation](./docs/User_Guide.md#️installation)
- [🔥Benchmarks](./docs/User_Guide.md#benchmarks)
- [🔥Supported Pipelines](./docs/User_Guide.md#supported-pipelines)
- [🎉Unified Cache APIs](./docs/User_Guide.md#unified-cache-apis)
  - [📚Forward Pattern Matching](./docs/User_Guide.md#forward-pattern-matching)
  - [📚Cache with One-line Code](./docs/User_Guide.md#%EF%B8%8Fcache-acceleration-with-one-line-code)
  - [🔥Automatic Block Adapter](./docs/User_Guide.md#automatic-block-adapter)
  - [📚Hybird Forward Pattern](./docs/User_Guide.md#hybird-forward-pattern)
  - [📚Implement Patch Functor](./docs/User_Guide.md#implement-patch-functor)
  - [🤖Cache Acceleration Stats](./docs/User_Guide.md#cache-acceleration-stats-summary)
- [⚡️Dual Block Cache](./docs/User_Guide.md#️dbcache-dual-block-cache)
- [🔥TaylorSeer Calibrator](./docs/User_Guide.md#taylorseer-calibrator)
- [⚡️Hybrid Cache CFG](./docs/User_Guide.md#️hybrid-cache-cfg)
- [⚙️Torch Compile](./docs/User_Guide.md#️torch-compile)
- [🛠Metrics CLI](./docs/User_Guide.md#metrics-cli)
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

## ©️Acknowledgements

<div id="Acknowledgements"></div>

The **cache-dit** codebase is adapted from FBCache. Over time its codebase diverged a lot, and **cache-dit** API is no longer compatible with FBCache. 

## ©️Special Acknowledgements

Special thanks to vipshop's Computer Vision AI Team for supporting document, testing and production-level deployment of this project.

## ©️Citations

<div id="citations"></div>

```BibTeX
@misc{cache-dit@2025,
  title={cache-dit: A Unified, Flexible and Training-free Cache Acceleration Framework for Diffusers.},
  url={https://github.com/vipshop/cache-dit.git},
  note={Open-source software available at https://github.com/vipshop/cache-dit.git},
  author={vipshop.com},
  year={2025}
}
```
