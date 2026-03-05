<div align="center">
  <p align="center">
    <h2 align="center">
        <img src=https://github.com/vipshop/cache-dit/raw/main/assets/cache-dit-logo-v2.png width=185px align="left">
        A PyTorch-native and Flexible Inference Engine with <br>Hybrid Cache Acceleration and Parallelism for 🤗DiTs<br>
        <a href="https://pepy.tech/projects/cache-dit"><img src=https://static.pepy.tech/personalized-badge/cache-dit?period=total&units=ABBREVIATION&left_color=GRAY&right_color=BLUE&left_text=downloads/pypi ></a>
        <a href="https://pypi.org/project/cache-dit/"><img src=https://img.shields.io/github/release/vipshop/cache-dit.svg?color=GREEN ></a>
        <img src="https://img.shields.io/github/license/vipshop/cache-dit.svg?color=blue">
        <a href="https://cache-dit.readthedocs.io/en/latest/COMMUNITY/"><img src=https://img.shields.io/badge/🤗-Community-orange.svg ></a> 
        <a href="https://hellogithub.com/repository/vipshop/cache-dit" target="_blank"><img src="https://api.hellogithub.com/v1/widgets/recommend.svg?rid=b8b03b3b32a449ea84cfc2b96cd384f3&claim_uid=ofSCbzTmdeQk3FD&theme=small" alt="Featured｜HelloGitHub" /></a> 
    </h2>
  </p>
</div>

**🤗Why Cache-DiT❓❓**Cache-DiT is built on top of the Diffusers library and now supports nearly [ALL](https://cache-dit.readthedocs.io/en/latest/supported_matrix/NVIDIA_GPU/) DiTs from Diffusers. It provides [hybrid cache acceleration](https://cache-dit.readthedocs.io/en/latest/user_guide/CACHE_API/) (DBCache, TaylorSeer, SCM, etc.) and comprehensive [parallelism](https://cache-dit.readthedocs.io/en/latest/user_guide/CONTEXT_PARALLEL/) optimizations, including Context Parallelism, Tensor Parallelism, hybrid 2D or 3D parallelism, and dedicated extra parallel support for Text Encoder, VAE, and ControlNet. 

<div align="center">
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/arch.png width=850px>
</div>

<!--
- 🎉**Hybrid Cache Acceleration** ([**DBCache**](https://cache-dit.readthedocs.io/en/latest/user_guide/CACHE_API/#dbcache-dual-block-cache), DBPrune, [**TaylorSeer**](https://cache-dit.readthedocs.io/en/latest/user_guide/CACHE_API/#hybrid-taylorseer-calibrator), [**SCM**](https://cache-dit.readthedocs.io/en/latest/user_guide/CACHE_API/#scm-steps-computation-masking), Cache CFG and more)
- 🎉**Context Parallelism** ([**CP**](https://cache-dit.readthedocs.io/en/latest/user_guide/CONTEXT_PARALLEL/) w/ Ulysses, Ring, **[USP](https://arxiv.org/pdf/2405.07719)**, [**Ulysses Anything**](https://cache-dit.readthedocs.io/en/latest/user_guide/CONTEXT_PARALLEL/#uaa-ulysses-anything-attention), [**Async CP**](https://cache-dit.readthedocs.io/en/latest/user_guide/CONTEXT_PARALLEL/#async-ulysses-qkv-projection), FP8 Comm)
- 🎉**Tensor Parallelism** ([**TP**](https://cache-dit.readthedocs.io/en/latest/user_guide/HYBRID_PARALLEL/) w/ PyTorch native DTensor and Tensor Parallelism APIs, avoid OOM)
- 🎉**Hybrid [2D](https://cache-dit.readthedocs.io/en/latest/user_guide/HYBRID_PARALLEL/) and [3D](https://cache-dit.readthedocs.io/en/latest/user_guide/HYBRID_PARALLEL/) Parallelism** (w/ 💥[**USP + TP**](https://cache-dit.readthedocs.io/en/latest/user_guide/HYBRID_PARALLEL/), scale up the performance of [**Large DiTs**](https://cache-dit.readthedocs.io/en/latest/user_guide/HYBRID_PARALLEL/))
- 🎉**Text Encoder Parallelism** ([**TE-P**](https://cache-dit.readthedocs.io/en/latest/user_guide/EXTRA_PARALLEL) w/ PyTorch native DTensor and Tensor Parallelism APIs)
- 🎉**Auto Encoder Parallelism** ([**VAE-P**](https://cache-dit.readthedocs.io/en/latest/user_guide/EXTRA_PARALLEL) w/ Data or Tile Parallelism, slightly faster, avoid OOM)
- 🎉**ControlNet Parallelism** ([**CN-P**](https://cache-dit.readthedocs.io/en/latest/user_guide/EXTRA_PARALLEL) w/ Context Parallelism for some ControlNet modules)
- 🎉Compatible with [**Compile**](https://cache-dit.readthedocs.io/en/latest/user_guide/COMPILE/), CPU Offloading, [**Quantization**](https://cache-dit.readthedocs.io/en/latest/user_guide/QUANTIZATION/) (TorchAo, nunchaku), ...
- 🎉Fully integrated into [**vLLM-Omni**](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion/cache_dit_acceleration/), [**SGLang Diffusion**](https://docs.sglang.io/diffusion/performance/cache/cache_dit.html), SD.Next, ComfyUI, ...
- 🎉**Natively** supports **NVIDIA GPUs**, [**Ascend NPUs**](https://cache-dit.readthedocs.io/en/latest/user_guide/ASCEND_NPU/) (>= 1.2.0), ... 
-->

Cache-DiT is compatible with compilation, CPU offloading, and quantization (TorchAo, nunchaku), fully integrates with [SGLang Diffusion](https://docs.sglang.io/diffusion/performance/cache/cache_dit.html), [vLLM-Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion/cache_dit_acceleration/), ComfyUI, and runs natively on NVIDIA GPUs and Ascend NPUs. Cache-DiT is really **fast**, **easy to use**, and **flexible** for various DiTs (online docs at 📘[readthedocs.io](https://cache-dit.readthedocs.io/en/latest/)).

<div align="center">
  <img src=assets/speedup_v5.png width=800px>
</div>

## 🔥Latest News 

- [2026/02] **[🎉v1.2.1](https://github.com/vipshop/cache-dit)** release is ready, the major updates including: [Ring](https://cache-dit.readthedocs.io/en/latest/user_guide/CONTEXT_PARALLEL) Attention w/ [batched P2P](https://cache-dit.readthedocs.io/en/latest/user_guide/CONTEXT_PARALLEL), [USP](https://cache-dit.readthedocs.io/en/latest/user_guide/CONTEXT_PARALLEL/) (Hybrid Ring and Ulysses), Hybrid 2D and 3D Parallelism (💥[USP + TP](https://cache-dit.readthedocs.io/en/latest/user_guide/HYBRID_PARALLEL/)),  VAE-P Comm overhead reduce.
- [2026/01] **[🎉v1.2.0](https://github.com/vipshop/cache-dit)** stable release is ready: New Models Support (Z-Image, FLUX.2, LTX-2, etc), Request level Cache Context, HTTP Serving, [Ulysses Anything](https://cache-dit.readthedocs.io/en/latest/user_guide/CONTEXT_PARALLEL/#uaa-ulysses-anything-attention), TE-P, VAE-P, CN-P and [Ascend NPUs](https://cache-dit.readthedocs.io/en/latest/user_guide/ASCEND_NPU/) support.


## 🚀Quick Start 

You can install the cache-dit from PyPI or from source: 
```bash
pip3 install -U cache-dit # or, pip3 install git+https://github.com/vipshop/cache-dit.git
```
Then accelerate your DiTs with just **♥️one line♥️** of code ~  
```python
>>> import cache_dit
>>> from diffusers import DiffusionPipeline
>>> # The pipe can be any diffusion pipeline.
>>> pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image")
>>> # Cache Acceleration with One-line code.
>>> cache_dit.enable_cache(pipe)
>>> # Or, Hybrid Cache Acceleration + 1D Parallelism.
>>> from cache_dit import DBCacheConfig, ParallelismConfig
>>> cache_dit.enable_cache(
...   pipe, cache_config=DBCacheConfig(), # w/ default
...   parallelism_config=ParallelismConfig(ulysses_size=2))
>>> # Or, Use Distributed Inference without Cache Acceleration.
>>> cache_dit.enable_cache(
...   pipe, parallelism_config=ParallelismConfig(ulysses_size=2))
>>> # Or, Hybrid Cache Acceleration + 2D Parallelism.
>>> cache_dit.enable_cache(
...   pipe, cache_config=DBCacheConfig(), # w/ default
...   parallelism_config=ParallelismConfig(ulysses_size=2, tp_size=2))
>>> from cache_dit import load_configs
>>> # Or, Load Acceleration config from a custom yaml file.
>>> cache_dit.enable_cache(pipe, **load_configs("config.yaml"))
>>> # Optional, set attention backend for better performance.
>>> cache_dit.set_attn_backend(pipe, attention_backend=...)
>>> output = pipe(...) # Just call the pipe as normal.
```

For more advanced features, please refer to our online documentation at 📘[readthedocs.io](https://cache-dit.readthedocs.io/en/latest/).

## 🌐Community Integration

- 🎉[ComfyUI x Cache-DiT](https://github.com/Jasonzzt/ComfyUI-CacheDiT)
- 🎉[(Intel) llm-scaler x Cache-DiT](https://github.com/intel/llm-scaler/tree/main/omni#cache-dit--torchcompile-acceleration)
- 🎉[Ascend NPU x Cache-DiT](https://cache-dit.readthedocs.io/en/latest/user_guide/ASCEND_NPU/)
- 🎉[Diffusers x Cache-DiT](https://huggingface.co/docs/diffusers/main/en/optimization/cache_dit)
- 🎉[SGLang Diffusion x Cache-DiT](https://docs.sglang.io/diffusion/performance/cache/cache_dit.html)
- 🎉[vLLM-Omni x Cache-DiT](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion/cache_dit_acceleration/)
- 🎉[Nunchaku x Cache-DiT](https://nunchaku.tech/docs/nunchaku/usage/cache.html#cache-dit)
- 🎉[SD.Next x Cache-DiT](https://github.com/vladmandic/sdnext/blob/master/modules/cachedit.py)
- 🎉[stable-diffusion.cpp x Cache-DiT](https://github.com/leejet/stable-diffusion.cpp/blob/master/cache_dit.hpp)
- 🎉[jetson-containers x Cache-DiT](https://github.com/dusty-nv/jetson-containers/tree/master/packages/diffusion/cache_edit)


## ©️Acknowledgements

Special thanks to vipshop's Computer Vision AI Team for supporting document, testing and deployment of this project. We learned the design and reused code from the following projects: [Diffusers](https://huggingface.co/docs/diffusers), [SGLang](https://github.com/sgl-project/sglang), [vLLM](https://github.com/vllm-project/vllm), [vLLM-Omni](https://github.com/vllm-project/vllm-omni), [ParaAttention](https://github.com/chengzeyi/ParaAttention), [xDiT](https://github.com/xdit-project/xDiT) and [TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer).


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
<p align="center">Star ⭐ this repo if you find it useful!</p>
