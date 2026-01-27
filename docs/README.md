<div align="center">
  <p align="center">
    <h2 align="center">
        <img src=https://github.com/vipshop/cache-dit/raw/main/assets/cache-dit-logo-v2.png align="center"><br>
        A PyTorch-native and Flexible Inference Engine with <br>Hybrid Cache Acceleration and Parallelism for 🤗DiTs<br>
        <a href="https://pepy.tech/projects/cache-dit"><img src=https://static.pepy.tech/personalized-badge/cache-dit?period=monthly&units=ABBREVIATION&left_color=GRAY&right_color=BLUE&left_text=downloads/month ></a>
        <img src=https://img.shields.io/github/release/vipshop/cache-dit.svg?color=GREEN >
        <img src="https://img.shields.io/github/license/vipshop/cache-dit.svg?color=blue">
        <a href="https://cache-dit.readthedocs.io/en/latest/COMMUNITY/"><img src=https://img.shields.io/badge/🤗-Community-orange.svg ></a> 
        <a href="https://hellogithub.com/repository/vipshop/cache-dit" target="_blank"><img src="https://api.hellogithub.com/v1/widgets/recommend.svg?rid=b8b03b3b32a449ea84cfc2b96cd384f3&claim_uid=ofSCbzTmdeQk3FD&theme=small" alt="Featured｜HelloGitHub" /></a> 
    </h2>
  </p>
</div>

<p style="text-align:center">
<script async defer src="https://buttons.github.io/buttons.js"></script>
<a class="github-button" href="https://github.com/vipshop/cache-dit" data-show-count="true" data-size="large" aria-label="Star">Star</a>
<a class="github-button" href="https://github.com/vipshop/cache-dit/subscription" data-show-count="true" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
<a class="github-button" href="https://github.com/vipshop/cache-dit/fork" data-show-count="true" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
</p>

|Baseline|SCM Slow|SCM Fast|SCM Ultra|+compile|+FP8*|+CP2|   
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|24.85s|15.4s|11.4s|8.2s|**🎉7.1s**|**🎉4.5s**|**🎉2.9s**|
|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/steps_mask/flux.NONE.png" width=90px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/steps_mask/static.png" width=90px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/steps_mask/flux.DBCache_F1B0_W8I1M0MC0_R0.2_SCM1111110100010000100000100000_dynamic_T0O0_S15.png" width=90px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/steps_mask/flux.DBCache_F1B0_W8I1M0MC0_R0.3_SCM111101000010000010000001000000_dynamic_T0O0_S19.png" width=90px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/steps_mask/flux.DBCache_F1B0_W8I1M0MC0_R0.35_SCM111101000010000010000001000000_dynamic_T1O1_S19.png" width=90px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/steps_mask/flux.C1_Q1_float8_DBCache_F1B0_W8I1M0MC0_R0.35_SCM111101000010000010000001000000_dynamic_T1O1_S19.png" width=90px>|<img src="https://github.com/vipshop/cache-dit/raw/main/assets/steps_mask/flux.1024x1024.C1_Q1_float8_DBCache_F1B0_W8I1M0MC0_R0.35_SCM1111001000001000000100000001_dynamic_CFG0_T1O1_Ulysses2_S19_ulysses_float8_sage.png" width=90px>|

**🤗Why Cache-DiT❓❓**Cache-DiT is built on top of the Diffusers library and now supports nearly **[🔥ALL](https://cache-dit.readthedocs.io/en/latest/)** DiTs from Diffusers, including over **[🤗70+](https://github.com/vipshop/cache-dit)** DiTs. Please refer to our online documentation at [readthedocs.io](https://cache-dit.readthedocs.io/en/latest/) for more details. The optimizations made by Cache-DiT include:  

- 🎉**Hybrid Cache Acceleration** ([**DBCache**](https://cache-dit.readthedocs.io/en/latest/user_guide/CACHE_API/#dbcache-dual-block-cache), DBPrune, [**TaylorSeer**](https://cache-dit.readthedocs.io/en/latest/user_guide/CACHE_API/#hybrid-taylorseer-calibrator), [**SCM**](https://cache-dit.readthedocs.io/en/latest/user_guide/CACHE_API/#scm-steps-computation-masking) and more)
- 🎉**Context Parallelism** (w/ Diffusers' CP APIs, Ulysses, Ring, [**Ulysses Anything**](https://cache-dit.readthedocs.io/en/latest/user_guide/CONTEXT_PARALLEL/#uaa-ulysses-anything-attention), USP)
- 🎉**Tensor Parallelism** (w/ PyTorch native DTensor and Tensor Parallelism APIs)
- 🎉**Text Encoder Parallelism** (w/ PyTorch DTensor and Tensor Parallelism APIs)
- 🎉**Auto Encoder (VAE) Parallelism** (w/ Data or Tile Parallelism, avoid OOM)
- 🎉**ControlNet Parallelism** (w/ Context Parallelism for ControlNet module)
- 🎉Built-in **HTTP serving** deployment support with simple REST APIs
- 🎉**Natively** compatible with **Compile**, **Offloading**, **Quantization**, ...
- 🎉Integration into **vLLM-Omni**, **SGLang Diffusion**, SD.Next, ...
- 🎉**Natively** supports **NVIDIA GPUs**, [**Ascend NPUs**](https://cache-dit.readthedocs.io/en/latest/user_guide/ASCEND_NPU/) (>= 1.2.0), ...  

## 🔥Latest News 

- [2026/01] **[🎉v1.2.0 Major Release](https://github.com/vipshop/cache-dit)** is ready: New Models Support(Z-Image, FLUX.2, LTX-2, etc), Request level Cache Context, HTTP Serving, [Ulysses Anything](https://cache-dit.readthedocs.io/en/latest/user_guide/CONTEXT_PARALLEL/#uaa-ulysses-anything-attention), TE-P, VAE-P, CN-P and [Ascend NPUs](https://cache-dit.readthedocs.io/en/latest/user_guide/ASCEND_NPU/) Support.

## 🚀Quick Start 

You can install the cache-dit from PyPI or from source: 
```bash
pip3 install -U cache-dit # or, pip3 install git+https://github.com/vipshop/cache-dit.git
```
Then try ♥️ Cache Acceleration with just **one line** of code ~ ♥️
```python
>>> import cache_dit
>>> from diffusers import DiffusionPipeline
>>> # The pipe can be any diffusion pipeline.
>>> pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image")
>>> # Cache Acceleration with One-line code.
>>> cache_dit.enable_cache(pipe)
>>> # Or, Hybrid Cache Acceleration + Parallelism.
>>> from cache_dit import DBCacheConfig, ParallelismConfig
>>> cache_dit.enable_cache(
...   pipe, cache_config=DBCacheConfig(), 
...   parallelism_config=ParallelismConfig(ulysses_size=2)
... )
>>> from cache_dit import load_configs
>>> # Or, Load Acceleration config from a custom yaml file.
>>> cache_dit.enable_cache(pipe, **load_configs("config.yaml"))
>>> # Set custom attention backend for better performance.
>>> cache_dit.set_attn_backend(pipe, attention_backend=...)
>>> output = pipe(...) # Just call the pipe as normal.
```
Please refer to our online documentation at [readthedocs.io](https://cache-dit.readthedocs.io/en/latest/) for more details.

## 🚀Quick Links

- [📊Examples](https://github.com/vipshop/cache-dit/tree/main/examples/) - The **easiest** way to enable **hybrid cache acceleration** and **parallelism** for DiTs with cache-dit is to start with our examples for popular models: FLUX, Z-Image, Qwen-Image, Wan, etc.
- [🌐HTTP Serving](https://cache-dit.readthedocs.io/en/latest) - Deploy cache-dit models with HTTP API for **text-to-image**, **image editing**, **multi-image editing**, and **text/image-to-video** generation.
- [🎉User Guide](https://cache-dit.readthedocs.io/en/latest/) - For more advanced features, please refer to the [🎉User Guide](https://cache-dit.readthedocs.io/en/latest/) for details.
- [❓FAQ](https://cache-dit.readthedocs.io/en/latest) - Frequently asked questions including attention backend configuration, troubleshooting, and optimization tips.

## 🌐Community Integration

- 🔥[Ascend NPU x Cache-DiT](https://cache-dit.readthedocs.io/en/latest/user_guide/ASCEND_NPU/)
- 🎉[Diffusers x Cache-DiT](https://huggingface.co/docs/diffusers/main/en/optimization/cache_dit)
- 🎉[SGLang Diffusion x Cache-DiT](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/docs/cache_dit.md)
- 🎉[vLLM-Omni x Cache-DiT](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion/cache_dit_acceleration/)
- 🎉[Nunchaku x Cache-DiT](https://nunchaku.tech/docs/nunchaku/usage/cache.html#cache-dit)
- 🎉[SD.Next x Cache-DiT](https://github.com/vladmandic/sdnext/blob/master/modules/cachedit.py)
- 🎉[stable-diffusion.cpp x Cache-DiT](https://github.com/leejet/stable-diffusion.cpp/blob/master/cache_dit.hpp)
- 🎉[jetson-containers x Cache-DiT](https://github.com/dusty-nv/jetson-containers/tree/master/packages/diffusion/cache_edit)

## ©️Acknowledgements

Special thanks to vipshop's Computer Vision AI Team for supporting document, testing and deployment of this project. We learned the design and reused code from the following projects: [Diffusers](https://huggingface.co/docs/diffusers), [SGLang](https://github.com/sgl-project/sglang), [vLLM-Omni](https://github.com/vllm-project/vllm-omni), [ParaAttention](https://github.com/chengzeyi/ParaAttention), [xDiT](https://github.com/xdit-project/xDiT), [TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer) and [LeMiCa](https://github.com/UnicomAI/LeMiCa).

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
