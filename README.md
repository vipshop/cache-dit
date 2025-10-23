📚English | <a href="./README_CN.md">📚中文阅读 </a> 

<div align="center">
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/cache-dit-logo.png height="120">
  <p align="center">
    A <b>Unified</b>, Flexible and Training-free <b>Cache Acceleration</b> Framework for <b>🤗Diffusers</b> <br>
    ♥️ Cache Acceleration with <b>One-line</b> Code ~ ♥️ <br>
    🔥<a href="./docs/User_Guide.md">Forward Pattern Matching</a> | <a href="./docs/User_Guide.md">Automatic Block Adapter</a>🔥 <br>
    🔥<a href="./docs/User_Guide.md"><b>DBCache</b></a> | <a href="./docs/User_Guide.md"><b>DBPrune</b></a> | <a href="./docs/User_Guide.md">Hybrid <b>TaylorSeer</b> Calibrator</a> | <a href="./docs/User_Guide.md">Cache CFG</a>🔥<br>
    🔥<a href="./docs/User_Guide.md"><b>Context Parallelism</b></a> | <a href="./docs/User_Guide.md">Torch Compile Compatible</a> | <a href="./docs/User_Guide.md">SOTA</a>🔥 
  </p>
  <div align='center'>
      <img src=https://img.shields.io/badge/Language-Python-brightgreen.svg >
      <img src=https://img.shields.io/badge/PRs-welcome-blue.svg >
      <img src=https://img.shields.io/badge/PyPI-pass-brightgreen.svg >
      <img src=https://static.pepy.tech/badge/cache-dit >
      <img src=https://img.shields.io/github/stars/vipshop/cache-dit.svg?style=dark >
      <img src=https://img.shields.io/badge/Release-v1.0-brightgreen.svg >
  </div>
  <p align="center">
    🎉Now, <b>cache-dit</b> covers almost <b>All</b> Diffusers' <b>DiT</b> Pipelines🎉<br>
    🔥<a href="./examples/pipeline">Qwen-Image</a> | <a href="./examples/pipeline">Qwen-Image-Edit</a> | <a href="./examples/pipeline">Qwen-Image-Edit-Plus </a> 🔥<br>
    🔥<a href="./examples/pipeline">FLUX.1</a> | <a href="./examples/pipeline">Qwen-Image-Lightning 4/8 Steps</a> | <a href="./examples/pipeline"> Wan 2.1 </a> | <a href="./examples/pipeline"> Wan 2.2 </a>🔥<br>
    🔥<a href="./examples/pipeline">HunyuanImage-2.1</a> | <a href="./examples/pipeline">HunyuanVideo</a> | <a href="./examples/pipeline">HunyuanDiT</a> | <a href="./examples/pipeline">HiDream</a> | <a href="./examples/pipeline">AuraFlow</a>🔥<br>
    🔥<a href="./examples/pipeline">CogView3Plus</a> | <a href="./examples/pipeline">CogView4</a> | <a href="./examples/pipeline">LTXVideo</a> | <a href="./examples/pipeline">CogVideoX</a> | <a href="./examples/">CogVideoX 1.5</a> | <a href="./examples/">ConsisID</a>🔥<br>
    🔥<a href="./examples/pipeline">Cosmos</a> | <a href="./examples/pipeline">SkyReelsV2</a> | <a href="./examples/pipeline">VisualCloze</a> | <a href="./examples/pipeline">OmniGen 1/2</a> | <a href="./examples/pipeline">Lumina 1/2</a> | <a href="./examples/pipeline">PixArt</a>🔥<br>
    🔥<a href="./examples/pipeline">Chroma</a> | <a href="./examples/pipeline">Sana</a> | <a href="./examples/pipeline">Allegro</a> | <a href="./examples/pipeline">Mochi</a> | <a href="./examples/pipeline">SD 3/3.5</a> | <a href="./examples/pipeline">Amused</a> | <a href="./examples/pipeline"> ... </a> | <a href="./examples/pipeline">DiT-XL</a>🔥
    <br>♥️ Please consider to leave a <b>⭐️ Star</b> to support us ~ ♥️ 
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
  <p><b>🔥Qwen-Image-Edit</b> | Input w/o Edit | Baseline | <a href="https://github.com/vipshop/cache-dit">+cache-dit</a>:1.6x↑🎉 | 1.9x↑🎉 </p>
</div>

<details align='center'>
<summary>🔥<b>Click</b> here to show many <b>Image/Video</b> cases🔥</summary>

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

## 🔥Hightlight <a href="https://huggingface.co/docs/diffusers/main/en/optimization/cache_dit"><img src=https://img.shields.io/badge/🤗Diffusers-ecosystem-yellow.svg ></a>

We are excited to announce that the **first API-stable version (v1.0.0)** of cache-dit has finally been released!
**[cache-dit](https://github.com/vipshop/cache-dit)** is a **Unified**, **Flexible**, and **Training-free** cache acceleration framework for 🤗 Diffusers, enabling cache acceleration with just **one line** of code. The key features of cache-dit includes: **Unified Cache APIs**, **Forward Pattern Matching**, **Automatic Block Adapter**, **DBCache**, **DBPrune**, **Hybrid TaylorSeer Calibrator**, **Hybrid Cache CFG**, **Context Parallelism**, **Torch Compile Compatible** and **🎉SOTA** performance.

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
- **[🎉Compatibility with Other Optimizations](./docs/User_Guide.md#️torch-compile)**: Designed to work seamlessly with torch.compile, Quantization ([torchao](./examples/quantize/), [🔥nunchaku](./examples/quantize/)), CPU or Sequential Offloading, **[🔥Context Parallelism](./docs/User_Guide.md/#️hybrid-context-parallelism)**, Tensor Parallelism, etc.  
- **[🎉Hybrid Cache Acceleration](./docs/User_Guide.md#taylorseer-calibrator)**: Now supports hybrid **Block-wise Cache + Calibrator** schemes (e.g., DBCache or DBPrune + TaylorSeerCalibrator). DBCache or DBPrune acts as the **Indicator** to decide *when* to cache, while the Calibrator decides *how* to cache. More mainstream cache acceleration algorithms (e.g., FoCa) will be supported in the future, along with additional benchmarks—stay tuned for updates!  
- **[🤗Diffusers Ecosystem Integration](https://huggingface.co/docs/diffusers/main/en/optimization/cache_dit)**: 🔥**cache-dit** has joined the Diffusers community ecosystem as the **first** DiT-specific cache acceleration framework! Check out the documentation here: <a href="https://huggingface.co/docs/diffusers/main/en/optimization/cache_dit"><img src=https://img.shields.io/badge/🤗Diffusers-ecosystem-yellow.svg ></a>

![](https://github.com/vipshop/cache-dit/raw/main/assets/clip-score-bench.png)

## 🔥Important News

- 2025.10.23: 🎉Now cache-dit supported the [Kandinsky5 T2V](https://github.com/ai-forever/Kandinsky-5) and [Photoroom/PRX](https://github.com/huggingface/diffusers/pull/12456) pipelines.
- 2025.10.20: 🔥Now cache-dit supported the **[Hybrid Cache + Context Parallelism](./docs/User_Guide.md/#️hybrid-context-parallelism)** scheme!🔥
- 2025.10.16: 🎉cache-dit + [**🔥nunchaku 4-bits**](https://github.com/nunchaku-tech/nunchaku) supported: [Qwen-Image-Lightning 4/8 steps](./examples/quantize/).
- 2025.10.15: 🎉cache-dit now supported [**🔥nunchaku**](https://github.com/nunchaku-tech/nunchaku): Qwen-Image/FLUX.1 [4-bits examples](./examples/quantize/)
- 2025.10.13: 🎉cache-dit achieved the **SOTA** performance w/ **7.4x↑🎉** speedup on ClipScore!
- 2025.10.10: 🔥[**Qwen-Image-ControlNet-Inpainting**](https://huggingface.co/InstantX/Qwen-Image-ControlNet-Inpainting) **2.3x↑🎉** speedup! Check the [example](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline/run_qwen_image_controlnet_inpaint.py).
- 2025.09.26: 🔥[**Qwen-Image-Edit-Plus(2509)**](https://github.com/QwenLM/Qwen-Image) **2.1x↑🎉** speedup! Please check the [example](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline/run_qwen_image_edit_plus.py).
- 2025.09.10: 🎉Day 1 support [**HunyuanImage-2.1**](https://github.com/Tencent-Hunyuan/HunyuanImage-2.1) with **1.7x↑🎉** speedup! Check this [example](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline/run_hunyuan_image_2.1.py).
- 2025.09.08: 🔥[**Qwen-Image-Lightning**](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline/run_qwen_image_lightning.py) **7.1/3.5 steps🎉** inference with **[DBCache: F16B16](https://github.com/vipshop/cache-dit)**.
- 2025.09.03: 🎉[**Wan2.2-MoE**](https://github.com/Wan-Video) **2.4x↑🎉** speedup! Please refer to [run_wan_2.2.py](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline/run_wan_2.2.py) as an example.
- 2025.08.19: 🔥[**Qwen-Image-Edit**](https://github.com/QwenLM/Qwen-Image) **2x↑🎉** speedup! Check the example: [run_qwen_image_edit.py](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline/run_qwen_image_edit.py).
- 2025.08.11: 🔥[**Qwen-Image**](https://github.com/QwenLM/Qwen-Image) **1.8x↑🎉** speedup! Please refer to [run_qwen_image.py](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline/run_qwen_image.py) as an example.

<details>
<summary>Previous News</summary>

- 2025.09.25: 🎉The **first API-stable version (v1.0.0)** of cache-dit has finally been released!
- 2025.09.25: 🔥**cache-dit** has joined the Diffusers community ecosystem: <a href="https://huggingface.co/docs/diffusers/main/en/optimization/cache_dit"><img src=https://img.shields.io/badge/🤗Diffusers-ecosystem-yellow.svg ></a>
- 2025.09.08: 🎉First caching mechanism in [Wan2.2](https://github.com/Wan-Video/Wan2.2) with **[cache-dit](https://github.com/vipshop/cache-dit)**, check this [PR](https://github.com/Wan-Video/Wan2.2/pull/127) for more details.
- 2025.09.08: 🎉First caching mechanism in [Qwen-Image-Lightning](https://github.com/ModelTC/Qwen-Image-Lightning) with **[cache-dit](https://github.com/vipshop/cache-dit)**, check this [PR](https://github.com/ModelTC/Qwen-Image-Lightning/pull/35).
- 2025.08.10: 🔥[**FLUX.1-Kontext-dev**](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) is supported! Please refer [run_flux_kontext.py](https://github.com/vipshop/cache-dit/blob/main/examples/pipeline/run_flux_kontext.py) as an example.
- 2025.08.12: 🎉First caching mechanism in [QwenLM/Qwen-Image](https://github.com/QwenLM/Qwen-Image) with **[cache-dit](https://github.com/vipshop/cache-dit)**, check this [PR](https://github.com/QwenLM/Qwen-Image/pull/61).
- 2025.07.18: 🎉First caching mechanism in [🤗huggingface/flux-fast](https://github.com/huggingface/flux-fast) with **[cache-dit](https://github.com/vipshop/cache-dit)**, check the [PR](https://github.com/huggingface/flux-fast/pull/13).
- 2025.07.13: 🎉[**FLUX.1-dev**](https://github.com/xlite-dev/flux-faster) **3.3x↑🎉** speedup! NVIDIA L20 with **[cache-dit](https://github.com/vipshop/cache-dit)** + **compile + FP8 DQ**.

</details>

## 📚User Guide

<div id="user-guide"></div>

For more advanced features such as **Unified Cache APIs**, **Forward Pattern Matching**, **Automatic Block Adapter**, **Hybrid Forward Pattern**, **Patch Functor**, **DBCache**, **TaylorSeer Calibrator**, and **Hybrid Cache CFG**, please refer to the [🎉User_Guide.md](./docs/User_Guide.md) for details.

- [⚙️Installation](./docs/User_Guide.md#️installation)
- [🔥Benchmarks](./docs/User_Guide.md#benchmarks)
- [🔥Supported Pipelines](./docs/User_Guide.md#supported-pipelines)
- [🎉Unified Cache APIs](./docs/User_Guide.md#unified-cache-apis)
  - [📚Forward Pattern Matching](./docs/User_Guide.md#forward-pattern-matching)
  - [📚Cache with One-line Code](./docs/User_Guide.md#%EF%B8%8Fcache-acceleration-with-one-line-code)
  - [🔥Automatic Block Adapter](./docs/User_Guide.md#automatic-block-adapter)
  - [📚Hybrid Forward Pattern](./docs/User_Guide.md#hybrid-forward-pattern)
  - [📚Implement Patch Functor](./docs/User_Guide.md#implement-patch-functor)
  - [🤖Cache Acceleration Stats](./docs/User_Guide.md#cache-acceleration-stats-summary)
- [⚡️DBCache: Dual Block Cache](./docs/User_Guide.md#️dbcache-dual-block-cache)
- [⚡️DBPrune: Dynamic Block Prune](./docs/User_Guide.md#️dbprune-dynamic-block-prune)
- [⚡️Hybrid Cache CFG](./docs/User_Guide.md#️hybrid-cache-cfg)
- [🔥Hybrid TaylorSeer Calibrator](./docs/User_Guide.md#taylorseer-calibrator)
- [⚡️Hybrid Context Parallelism](./docs/User_Guide.md#context-paralleism)
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

Here is a curated list of open-source projects integrating **CacheDiT**, including popular repositories like [jetson-containers](https://github.com/dusty-nv/jetson-containers/blob/master/packages/diffusion/cache_edit/build.sh) ![](https://img.shields.io/github/stars/dusty-nv/jetson-containers.svg), [flux-fast](https://github.com/huggingface/flux-fast) ![](https://img.shields.io/github/stars/huggingface/flux-fast.svg), and [sdnext](https://github.com/vladmandic/sdnext/discussions/4269) ![](https://img.shields.io/github/stars/vladmandic/sdnext.svg). **CacheDiT** has also been **recommended** by [Wan2.2](https://github.com/Wan-Video/Wan2.2) ![](https://img.shields.io/github/stars/Wan-Video/Wan2.2.svg), [Qwen-Image-Lightning](https://github.com/ModelTC/Qwen-Image-Lightning) ![](https://img.shields.io/github/stars/ModelTC/Qwen-Image-Lightning.svg), [Qwen-Image](https://github.com/QwenLM/Qwen-Image) ![](https://img.shields.io/github/stars/QwenLM/Qwen-Image.svg), and <a href="https://huggingface.co/docs/diffusers/main/en/optimization/cache_dit"><img src="https://img.shields.io/badge/🤗Diffusers-ecosystem-yellow.svg"></a> ![](https://img.shields.io/github/stars/huggingface/diffusers.svg), among others. We would be grateful if you could let us know if you have used CacheDiT.

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
