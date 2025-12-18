<div align="center">
  <p align="center">
    <h2 align="center">
        <img src=https://github.com/vipshop/cache-dit/raw/main/assets/cache-dit-logo.png height="90" align="left">
        A PyTorch-native and Flexible Inference Engine with <br>Hybrid Cache Acceleration and Parallelism for 🤗DiTs<br>
        <a href="https://pepy.tech/projects/cache-dit"><img src=https://static.pepy.tech/badge/cache-dit/month ></a>
        <img src=https://img.shields.io/github/release/vipshop/cache-dit.svg?color=GREEN >
        <img src="https://img.shields.io/github/license/vipshop/cache-dit.svg?color=blue">
        <a href="https://huggingface.co/docs/diffusers/main/en/optimization/cache_dit"><img src=https://img.shields.io/badge/🤗Diffusers-ecosystem-yellow.svg ></a> 
        <a href="https://hellogithub.com/repository/vipshop/cache-dit" target="_blank"><img src="https://api.hellogithub.com/v1/widgets/recommend.svg?rid=b8b03b3b32a449ea84cfc2b96cd384f3&claim_uid=ofSCbzTmdeQk3FD&theme=small" alt="Featured｜HelloGitHub" /></a> 
    </h2>
  </p>

|Baseline|SCM S S*|SCM F D*|SCM U D*|+TS|+compile|+FP8*|   
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|24.85s|15.4s|11.4s|8.2s|8.2s|**🎉7.1s**|**🎉4.5s**|
|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/steps_mask/flux.NONE.png" width=90px>|<img src="assets/steps_mask/static.png" width=90px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/steps_mask/flux.DBCache_F1B0_W8I1M0MC0_R0.2_SCM1111110100010000100000100000_dynamic_T0O0_S15.png" width=90px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/steps_mask/flux.DBCache_F1B0_W8I1M0MC0_R0.3_SCM111101000010000010000001000000_dynamic_T0O0_S19.png" width=90px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/steps_mask/flux.DBCache_F1B0_W8I1M0MC0_R0.35_SCM111101000010000010000001000000_dynamic_T1O1_S19.png" width=90px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/steps_mask/flux.DBCache_F1B0_W8I1M0MC0_R0.35_SCM111101000010000010000001000000_dynamic_T1O1_S19.png" width=90px>|<img src="./assets/steps_mask/flux.C1_Q1_float8_DBCache_F1B0_W8I1M0MC0_R0.35_SCM111101000010000010000001000000_dynamic_T1O1_S19.png" width=90px>|

<p align="center">
  Scheme: <b>DBCache + SCM(steps_computation_mask) + TS(TaylorSeer) + FP8*</b>, L20x1, S*: static cache, <br><b>D*: dynamic cache</b>, <b>S</b>: Slow, <b>F</b>: Fast, <b>U</b>: Ultra Fast, <b>TS</b>: TaylorSeer, <b>FP8*</b>: FP8 DQ + Sage, <b>FLUX.1</b>-Dev
</p>

<img src=https://github.com/vipshop/cache-dit/raw/main/assets/speedup_v4.png>

<!--
<p align="center">
    U*: Ulysses Attention, <b>UAA: Ulysses Anything Attenton</b>, UAA*: UAA + Gloo, Device: NVIDIA L20<br>
    FLUX.1-Dev w/o CPU Offload, 28 steps; Qwen-Image w/ CPU Offload, 50 steps; Gloo: Extra All Gather w/ Gloo
</p>

|CP2 U* |CP2 UAA* |  L20x1 | CP2 UAA* | CP2 U* |  L20x1 |  CP2 UAA* | 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|FLUX, 13.87s|**🎉13.88s**|23.25s| **🎉13.75s**|Qwen, 132s|181s|**🎉133s**|
|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/uaa/flux.C0_Q0_NONE_Ulysses2.png" width=90px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/uaa/flux.C0_Q0_NONE_Ulysses2_ulysses_anything.png" width=90px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/uaa/flux.1008x1008.C0_Q0_NONE.png" width=90px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets//uaa/flux.1008x1008.C0_Q0_NONE_Ulysses2_ulysses_anything.png" width=90px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/uaa/qwen-image.1312x1312.C0_Q0_NONE_Ulysses2.png" width=90px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/uaa/qwen-image.1328x1328.C0_Q0_NONE.png" width=90px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/uaa/qwen-image.1328x1328.C0_Q0_NONE_Ulysses2_ulysses_anything.png" width=90px>|
|1024x1024|1024x1024|1008x1008|1008x1008|1312x1312|1328x1328|1328x1328|
|✔️U* ✔️UAA|✔️U* ✔️UAA| NO CP|❌U* ✔️UAA|✔️U* ✔️UAA| NO CP|❌U* ✔️UAA|

<p align="center">
 <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/docs/cache_dit.md">
  <img src="https://img.shields.io/badge/🔥Latest_News-🎉SGLang_Diffusion_x_🤗Cache_DiT_ready!🔥-blue?style=for-the-badge&labelColor=darkblue&logo=github" alt="SGLang Diffusion x Cache-DiT News">
 </a><br>
 <a href="https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/cache_dit_acceleration/">
  <img src="https://img.shields.io/badge/🔥Latest_News-🎉vLLM_Omni_x_🤗Cache_DiT_ready!🔥-blue?style=for-the-badge&labelColor=darkblue&logo=github" alt="vLLM Omni x Cache-DiT News">
 </a>
</p>

-->

<p align="center">
 <a href="https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/docs/cache_dit.md">
  <img src="https://img.shields.io/badge/🔥News-🎉SGLang_Diffusion_x_🤗Cache_DiT🔥-skyblue?style=for-the-badge&labelColor=darkblue&logo=github" alt="SGLang Diffusion x Cache-DiT News" ></a>
 <a href="https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/acceleration/cache_dit_acceleration/" >
  <img src="https://img.shields.io/badge/🎉vLLM_Omni_x_🤗Cache_DiT🔥-skyblue?style=for-the-badge&labelColor=darkblue&logo=github" alt="vLLM Omni x Cache-DiT News"></a>
</p>

</div>

## 🔥Hightlight

We are excited to announce that the 🎉[**v1.1.0**](https://github.com/vipshop/cache-dit/releases/tag/v1.1.0) version of cache-dit has finally been released! It brings **[🔥Context Parallelism](./docs/User_Guide.md/#️hybrid-context-parallelism)** and **[🔥Tensor Parallelism](./docs/User_Guide.md#️hybrid-tensor-parallelism)** to cache-dit, thus making it a **[PyTorch-native](./)** and **[Flexible](./)** Inference Engine for 🤗DiTs. Key features: **Unified Cache APIs**, **Forward Pattern Matching**, **Block Adapter**, **DBCache**, **DBPrune**, **Cache CFG**, **TaylorSeer**, **[SCM](./docs/User_Guide.md#scm-steps-computation-masking)**, **Context Parallelism (w/ [UAA](./docs/User_Guide.md#uaa-ulysses-anything-attention))**, **Tensor Parallelism** and **🎉SOTA** performance.

```bash
pip3 install -U cache-dit # Also, pip3 install git+https://github.com/huggingface/diffusers.git (latest)
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

- **[🎉Full 🤗Diffusers Support](./docs/User_Guide.md#supported-pipelines)**: Notably, **[cache-dit](https://github.com/vipshop/cache-dit)** now supports nearly **all** of Diffusers' **DiT-based** pipelines, include **[30+](./examples/)** series, nearly **[100+](./examples/)** pipelines, such as FLUX.1, Qwen-Image, Qwen-Image-Lightning, Wan 2.1/2.2, HunyuanImage-2.1, HunyuanVideo, HiDream, AuraFlow, CogView3Plus, CogView4, CogVideoX, LTXVideo, ConsisID, SkyReelsV2, VisualCloze, PixArt, Chroma, Mochi, SD 3.5, DiT-XL, etc.  
- **[🎉Extremely Easy to Use](./docs/User_Guide.md#unified-cache-apis)**: In most cases, you only need **one line** of code: `cache_dit.enable_cache(...)`. After calling this API, just use the pipeline as normal.   
- **[🎉Easy New Model Integration](./docs/User_Guide.md#automatic-block-adapter)**: Features like **Unified Cache APIs**, **Forward Pattern Matching**, **Automatic Block Adapter**, **Hybrid Forward Pattern**, and **Patch Functor** make it highly functional and flexible. For example, we achieved 🎉 Day 1 support for [HunyuanImage-2.1](https://github.com/Tencent-Hunyuan/HunyuanImage-2.1) with 1.7x speedup w/o precision loss—even before it was available in the Diffusers library.  
- **[🎉State-of-the-Art Performance](./bench/)**: Compared with algorithms including Δ-DiT, Chipmunk, FORA, DuCa, TaylorSeer and FoCa, cache-dit achieved the **SOTA** performance w/ **7.4x↑🎉** speedup on ClipScore!
- **[🎉Support for 4/8-Steps Distilled Models](./bench/)**: Surprisingly, cache-dit's **DBCache** works for extremely few-step distilled models—something many other methods fail to do.  
- **[🎉Compatibility with Other Optimizations](./docs/User_Guide.md#️torch-compile)**: Designed to work seamlessly with torch.compile, Quantization ([torchao](./examples/quantize/), [🔥nunchaku](./examples/quantize/)), CPU or Sequential Offloading, **[🔥Context Parallelism](./docs/User_Guide.md/#️hybrid-context-parallelism)**, **[🔥Tensor Parallelism](./docs/User_Guide.md#️hybrid-tensor-parallelism)**, etc.  
- **[🎉Hybrid Cache Acceleration](./docs/User_Guide.md#taylorseer-calibrator)**: Now supports hybrid **Block-wise Cache + Calibrator** schemes (e.g., DBCache or DBPrune + TaylorSeerCalibrator). DBCache or DBPrune acts as the **Indicator** to decide *when* to cache, while the Calibrator decides *how* to cache. More mainstream cache acceleration algorithms (e.g., FoCa) will be supported in the future, along with additional benchmarks—stay tuned for updates!  
- **[🤗Diffusers Ecosystem Integration](https://huggingface.co/docs/diffusers/main/en/optimization/cache_dit)**: 🔥**cache-dit** has joined the Diffusers community ecosystem as the **first** DiT-specific cache acceleration framework! Check out the documentation here: <a href="https://huggingface.co/docs/diffusers/main/en/optimization/cache_dit"><img src=https://img.shields.io/badge/🤗Diffusers-ecosystem-yellow.svg ></a>
- **[🎉HTTP Serving Support](./docs/SERVING.md)**: Built-in HTTP serving capabilities for production deployment. Supports **text-to-image**, **image editing**, **multi-image editing**, **text-to-video**, and **image-to-video** generation with simple REST API. Easy integration with existing applications and services.

![](https://github.com/vipshop/cache-dit/raw/main/assets/clip-score-bench-v2.png)

<!--

The comparison between **cache-dit** and other algorithms shows that within a speedup ratio (TFLOPs) less than 🎉**4x**, cache-dit achieved the **SOTA** performance. Please refer to [📚Benchmarks](https://github.com/vipshop/cache-dit/tree/main/bench/) for more details.

<div align="center">

| Method | TFLOPs(↓) | SpeedUp(↑) | ImageReward(↑) | Clip Score(↑) |
| --- | --- | --- | --- | --- |
| [**FLUX.1**-dev]: 50 steps | 3726.87 | 1.00× | 0.9898 | 32.404 |
| Chipmunk | 1505.87 | 2.47× | 0.9936 | 32.776 |
| FORA(N=3) | 1320.07 | 2.82× | 0.9776 | 32.266 |
| **[DBCache(S)](https://github.com/vipshop/cache-dit)** | 1400.08 | **2.66×** | **1.0065** | 32.838 |
| DuCa(N=5) | 978.76 | 3.80× | 0.9955 | 32.241 |
| TeaCache(l=0.8) | 892.35 | 4.17× | 0.8683 | 31.704 |
| TaylorSeer(N=4,O=2) | 1042.27 | 3.57× | 0.9857 | 32.413 |
| **[DBCache(S)+TS](https://github.com/vipshop/cache-dit)** | 1153.05 | **3.23×** | **1.0221** | 32.819 |
| **[DBCache(M)+TS](https://github.com/vipshop/cache-dit)** | 944.75 | **3.94×** | **1.0107** | 32.865 |
| FoCa(N=5) | 893.54 | **4.16×** | 1.0029 | **32.948** |
| [**FLUX.1**-dev]: 22% steps | 818.29 | 4.55× | 0.8183 | 31.772 |
| TaylorSeer(N=7,O=2) | 670.44 | 5.54× | 0.9128 | 32.128 |
| FoCa(N=8) | 596.07 | 6.24× | 0.9502 | **32.706** |
| **[DBCache(F)+TS](https://github.com/vipshop/cache-dit)** | 651.90 | **5.72x** | **0.9526** | 32.568 |
| **[DBCache(U)+TS](https://github.com/vipshop/cache-dit)** | 505.47 | **7.37x** | 0.8645 | **32.719** |

</div>

🎉Surprisingly, **cache-dit** still works in the **extremely few-step** distill model, such as **Qwen-Image-Lightning**, with the F16B16 config, the PSNR is 34.8 and the ImageReward is 1.26. It maintained a relatively high precision.
<div align="center">

| Config                     |  PSNR(↑)      | Clip Score(↑) | ImageReward(↑) | TFLOPs(↓)   | SpeedUp(↑) |
|----------------------------|-----------|------------|--------------|----------|------------|
| **[Full 4 steps]**   | INF       | 35.5797    | 1.2630       | 274.33   | 1.00x       |
| F24B24          | 36.3242   | 35.6224    | 1.2630       | 264.74   | 1.04x       |
| F16B16         | 34.8163   | 35.6109    | 1.2614       | 244.25   | 1.12x       |
| F12B12         | 33.8953   | 35.6535    | 1.2549       | 234.63   | 1.17x       |
| F8B8         | 33.1374   | 35.7284    | 1.2517       | 224.29   | 1.22x       |
| F1B0            | 31.8317   | 35.6651    | 1.2397       | 206.90   | 1.33x       |

</div>
-->

## 🔥Supported DiTs

> [!Tip] 
> One **Model Series** may contain **many** pipelines. cache-dit applies optimizations at the **Transformer** level; thus, any pipelines that include the supported transformer are already supported by cache-dit. ✔️: known work and official supported now; ✖️: unofficial supported now, but maybe support in the future; **[`Q`](https://github.com/nunchaku-tech/nunchaku)**: **4-bits** models w/ [nunchaku](https://github.com/nunchaku-tech/nunchaku) **W4A4**; **TE**: Text Encoder Parallelism; **💡[C*](./)**: **Hybrid Cache** Acceleration.

<div align="center">

| 📚Model | [C*](./)  | CP | TP | TE | 📚Model | [C*](./)  | CP | TP | TE |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| **🔥[Z-Image](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✔️ | ✔️ | ✔️ | **🔥[Ovis-Image](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✔️ | ✔️ | ✔️ |
| **🔥[FLUX.2](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✔️ | ✔️ | ✔️ | **🔥[HuyuanVideo 1.5](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✖️ | ✖️ | ✔️ |
| **🎉[FLUX.1](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✔️ | ✔️ | ✔️ | **🎉[FLUX.1 `Q`](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✔️ | ✖️ | ✔️ |
| **🎉[FLUX.1-Fill](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✔️ | ✔️ | ✔️ | **🎉[Qwen-Image `Q`](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✔️ | ✖️ | ✔️ |
| **🎉[Qwen-Image](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✔️ | ✔️ | ✔️ | **🎉[Qwen...Edit `Q`](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✔️ | ✖️ | ✔️ |
| **🎉[Qwen...Edit](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✔️ | ✔️ | ✔️ | **🎉[Qwen.E.Plus `Q`](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✔️ | ✖️ | ✔️ |
| **🎉[Qwen..Light](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✔️ | ✔️ | ✔️ | **🎉[Qwen...Light `Q`](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✔️ | ✖️ | ✔️ |
| **🎉[Qwen..Control](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✔️ | ✔️ | ✔️ | **🎉[Qwen.E.Light `Q`](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✔️ | ✖️ | ✔️ |
| **🎉[Wan 2.1](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✔️ | ✔️ | ✔️ | **🎉[Mochi](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✖️ | ✔️ | ✔️ |
| **🎉[Wan 2.1 VACE](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✔️ | ✔️ | ✔️ | **🎉[HiDream](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✖️ | ✖️ | ✔️ |
| **🎉[Wan 2.2](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✔️ | ✔️ | ✔️ | **🎉[HunyuanDiT](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✖️ | ✔️ | ✔️ |
| **🎉[HunyuanVideo](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✔️ | ✔️ | ✔️ | **🎉[Sana](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✖️ | ✖️ | ✔️ |
| **🎉[ChronoEdit](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✔️ | ✔️ | ✔️ | **🎉[Bria](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✖️ | ✖️ | ✔️ |
| **🎉[CogVideoX](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✔️ | ✔️ | ✔️ | **🎉[SkyReelsV2](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✔️  | ✔️  | ✔️ |
| **🎉[CogVideoX 1.5](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✔️ | ✔️ | ✔️ | **🎉[Lumina 1/2](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✖️ | ✔️ | ✔️ |
| **🎉[CogView4](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✔️ | ✔️ | ✔️ | **🎉[DiT-XL](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✔️ | ✖️ | ✔️ |
| **🎉[CogView3Plus](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✔️ | ✔️ | ✔️ | **🎉[Allegro](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✖️ | ✖️ | ✔️ |
| **🎉[PixArt Sigma](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✔️ | ✔️ | ✔️ | **🎉[Cosmos](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✖️ | ✖️ | ✔️ |
| **🎉[PixArt Alpha](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✔️ | ✔️ | ✔️ | **🎉[OmniGen](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✖️ | ✖️ | ✔️ |
| **🎉[Chroma-HD](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✔️ | ️✔️ | ✔️ | **🎉[EasyAnimate](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✖️ | ✖️ | ✔️ |
| **🎉[VisualCloze](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✔️ | ✔️ | ✔️ | **🎉[StableDiffusion3](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✖️ | ✖️ | ✔️ |
| **🎉[HunyuanImage](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✔️ | ✔️ | ✔️ | **🎉[PRX T2I](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✖️ | ✖️ | ✔️ |
| **🎉[Kandinsky5](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✔️️ | ✔️️ | ✔️ | **🎉[Amused](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✖️ | ✖️ | ✔️ |
| **🎉[LTXVideo](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✔️ | ✔️ | ✔️ | **🎉[AuraFlow](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✖️ | ✖️ | ✔️ |
| **🎉[ConsisID](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✔️ | ✔️ | ✔️ | **🎉[LongCatVideo](https://github.com/vipshop/cache-dit/blob/main/examples)** | ✔️ | ✖️ | ✖️ | ✔️ |

</div>

<details align='center'>
<summary>🔥<b>Click</b> here to show many <b>Image/Video</b> cases🔥</summary>
  
<p align='center'>
  🎉Now, cache-dit covers almost All Diffusers' DiT Pipelines🎉 <br>
   🔥<a href="./examples">Qwen-Image</a> | <a href="./examples">Qwen-Image-Edit</a> | <a href="./examples">Qwen-Image-Edit-Plus </a> 🔥<br>
    🔥<a href="./examples">FLUX.1</a> | <a href="./examples">Qwen-Image-Lightning 4/8 Steps</a> | <a href="./examples"> Wan 2.1 </a> | <a href="./examples"> Wan 2.2 </a>🔥<br>
    🔥<a href="./examples">HunyuanImage-2.1</a> | <a href="./examples">HunyuanVideo</a> | <a href="./examples">HunyuanDiT</a> | <a href="./examples">HiDream</a> | <a href="./examples">AuraFlow</a>🔥<br>
    🔥<a href="./examples">CogView3Plus</a> | <a href="./examples">CogView4</a> | <a href="./examples">LTXVideo</a> | <a href="./examples">CogVideoX</a> | <a href="./examples/">CogVideoX 1.5</a> | <a href="./examples/">ConsisID</a>🔥<br>
    🔥<a href="./examples">Cosmos</a> | <a href="./examples">SkyReelsV2</a> | <a href="./examples">VisualCloze</a> | <a href="./examples">OmniGen 1/2</a> | <a href="./examples">Lumina 1/2</a> | <a href="./examples">PixArt</a>🔥<br>
    🔥<a href="./examples">Chroma</a> | <a href="./examples">Sana</a> | <a href="./examples">Allegro</a> | <a href="./examples">Mochi</a> | <a href="./examples">SD 3/3.5</a> | <a href="./examples">Amused</a> | <a href="./examples"> ... </a> | <a href="./examples">DiT-XL</a>🔥
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

For more advanced features such as **Unified Cache APIs**, **Forward Pattern Matching**, **Automatic Block Adapter**, **Hybrid Forward Pattern**, **Patch Functor**, **DBCache**, **DBPrune**, **TaylorSeer Calibrator**, **SCM**, **Hybrid Cache CFG**, **Context Parallelism (w/ UAA)** and **Tensor Parallelism**, please refer to the [🎉User_Guide.md](./docs/User_Guide.md) for details.

### 🚀Quick Links

- [📊Examples](./examples/) - The **easiest** way to enable **hybrid cache acceleration** and **parallelism** for DiTs with cache-dit is to start with our examples for popular models: FLUX, Z-Image, Qwen-Image, Wan, etc.
- [🌐HTTP Serving](./docs/SERVING.md) - Deploy cache-dit models with HTTP API for **text-to-image**, **image editing**, **multi-image editing**, and **text-to-video** generation.
- [❓FAQ](./FAQ.md) - Frequently asked questions including attention backend configuration, troubleshooting, and optimization tips.

### 📚Documentation
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
- [🤖SCM: Steps Computation Masking](./docs/User_Guide.md#steps-mask)
- [⚡️Hybrid Context Parallelism](./docs/User_Guide.md#context-parallelism)
- [🤖UAA: Ulysses Anything Attention](./docs/User_Guide.md#ulysses-anything-attention)
- [🤖Async Ulysses QKV Projection](./docs/User_Guide.md#ulysses-async)
- [🤖Async FP8 Ulysses Attention](./docs/User_Guide.md#ulysses-async-fp8)
- [⚡️Hybrid Tensor Parallelism](./docs/User_Guide.md#tensor-parallelism)
- [🤖Parallelize Text Encoder](./docs/User_Guide.md#parallel-text-encoder)
- [🤖Low-bits Quantization](./docs/User_Guide.md#quantization)
- [🤖How to use FP8 Attention](./docs/User_Guide.md#fp8-attention)
- [🛠Metrics Command Line](./docs/User_Guide.md#metrics-cli)
- [⚙️Torch Compile](./docs/User_Guide.md#️torch-compile)
- [📊Torch Profiler Usage](./docs/PROFILER.md)
- [📚API Documents](./docs/User_Guide.md#api-documentation)

## 👋Contribute 
<div id="contribute"></div>

How to contribute? Star ⭐️ this repo to support us or check [CONTRIBUTE.md](https://github.com/vipshop/cache-dit/raw/main/docs/CONTRIBUTE.md).

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

Here is a curated list of open-source projects integrating **CacheDiT**, including popular repositories like [jetson-containers](https://github.com/dusty-nv/jetson-containers/blob/master/packages/diffusion/cache_edit/build.sh), [flux-fast](https://github.com/huggingface/flux-fast), [sdnext](https://github.com/vladmandic/sdnext/discussions/4269), 🔥[vLLM-Omni](https://github.com/vllm-project/vllm-omni/blob/main/docs/user_guide/acceleration/cache_dit_acceleration.md), and 🔥[SGLang Diffusion](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/docs/cache_dit.md). 🎉CacheDiT has been **recommended** by many famous opensource projects: 🔥[Z-Image](https://github.com/Tongyi-MAI/Z-Image), 🔥[Wan 2.2](https://github.com/Wan-Video/Wan2.2), 🔥[Qwen-Image](https://github.com/QwenLM/Qwen-Image), 🔥[LongCat-Video](https://github.com/meituan-longcat/LongCat-Video), [Qwen-Image-Lightning](https://github.com/ModelTC/Qwen-Image-Lightning), [Kandinsky-5](https://github.com/ai-forever/Kandinsky-5), [LeMiCa](https://github.com/UnicomAI/LeMiCa), [🤗diffusers](https://huggingface.co/docs/diffusers/main/en/optimization/cache_dit), [HelloGitHub](https://hellogithub.com/repository/vipshop/cache-dit) and [GaintPandaCV](https://mp.weixin.qq.com/s/ZBr3veg7EF5kuiHpYmGGjQ).

## ©️Acknowledgements

Special thanks to vipshop's Computer Vision AI Team for supporting document, testing and production-level deployment of this project. We learned the design and reused code from the following projects: [🤗diffusers](https://huggingface.co/docs/diffusers), [SGLang](https://github.com/sgl-project/sglang), [ParaAttention](https://github.com/chengzeyi/ParaAttention), [xDiT](https://github.com/xdit-project/xDiT), [TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer) and [LeMiCa](https://github.com/UnicomAI/LeMiCa).

## ©️Citations

<div id="citations"></div>

```BibTeX
@misc{cache-dit@2025,
  title={cache-dit: A PyTorch-native and Flexible Inference Engine with Hybrid Cache Acceleration and Parallelism for DiTs.},
  url={https://github.com/vipshop/cache-dit.git},
  note={Open-source software available at https://github.com/vipshop/cache-dit.git},
  author={DefTruth, vipshop.com},
  year={2025}
}
```
