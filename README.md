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
</div>

## 🤗Why Cache-DiT?

**Cache-DiT** is built on top of the Diffusers library. Notably, Cache-DiT now supports nearly **ALL** DiT-based models from Diffusers, including over [🤗65+](https://github.com/vipshop/cache-dit) DiT-based models and nearly [100+](https://github.com/vipshop/cache-dit) pipelines. The optimizations made by Cache-DiT for diffusers include: 

- 🎉**Hybrid Cache Acceleration** (DBCache, TaylorSeer, SCM and more)
- 🎉**Context Parallelism** (w/ Ulysses Anything Attention, FP8 All2All, Async Ulysses CP)
- 🎉**Tensor Parallelism** (w/ PyTorch native DTensor and Tensor Parallel API)
- 🎉T**ext Encoder Parallelism** (Tensor Parallelism)
- 🎉**AutoEncoder (VAE) Parallelism** (latest, Data/Tile Parallelism)
- 🎉**ControlNet Parallelism** (currently, Context Parallelism)
- 🎉Compatible with **compile, offload, quantization**, ...
- 🎉Built-in **HTTP serving** support with simple REST API
- 🎉**vLLM-Omni**, **SGLang Diffusion**, SD.Next, ... integration
- 🎉**NVIDIA GPU**, **Ascend NPU** support (latest)

Please refer to our online documentation at [readthedocs.io](https://cache-dit-dev.readthedocs.io/en/latest/) for more details.

## 🚀Quick Start 

You can install the cache-dit from PyPI or from source: 
```bash
pip3 install -U cache-dit # or, pip3 install git+https://github.com/vipshop/cache-dit.git
```
Then try ♥️ Cache Acceleration with just **one line** of code ~ ♥️
```python
>>> import cache_dit
>>> from diffusers import DiffusionPipeline
>>> pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image") # Can be any diffusion pipeline
>>> cache_dit.enable_cache(pipe) # One-line code with default cache options.
>>> output = pipe(...) # Just call the pipe as normal.
>>> stats = cache_dit.summary(pipe) # Then, get the summary of cache acceleration stats.
>>> cache_dit.disable_cache(pipe) # Disable cache and run original pipe.
```

## 🚀Quick Links

- [📊Examples](https://github.com/vipshop/cache-dit/tree/main/examples/) - The **easiest** way to enable **hybrid cache acceleration** and **parallelism** for DiTs with cache-dit is to start with our examples for popular models: FLUX, Z-Image, Qwen-Image, Wan, etc.
- [🌐HTTP Serving](https://cache-dit-dev.readthedocs.io/en/latest/SERVING/) - Deploy cache-dit models with HTTP API for **text-to-image**, **image editing**, **multi-image editing**, and **text/image-to-video** generation.
- [🎉User Guide](https://cache-dit-dev.readthedocs.io/en/latest/User_Guide/) - For more advanced features, please refer to the [🎉User_Guide.md](https://cache-dit-dev.readthedocs.io/en/latest/User_Guide/) for details.
- [❓FAQ](https://cache-dit-dev.readthedocs.io/en/latest/FAQ/) - Frequently asked questions including attention backend configuration, troubleshooting, and optimization tips.

## ©️Acknowledgements

Special thanks to vipshop's Computer Vision AI Team for supporting document, testing and production-level deployment of this project. We learned the design and reused code from the following projects: [🤗Diffusers](https://huggingface.co/docs/diffusers), [SGLang](https://github.com/sgl-project/sglang), [ParaAttention](https://github.com/chengzeyi/ParaAttention), [xDiT](https://github.com/xdit-project/xDiT), [TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer) and [LeMiCa](https://github.com/UnicomAI/LeMiCa).

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
