<div align="center">
  <p align="center">
    <h2 align="center">
      <img src=https://github.com/vipshop/cache-dit/raw/main/assets/cache-dit-logo-v2.png width=185px align="left">
      ⚡️🎉A PyTorch-native Inference Engine with Cache, <br>Parallelism, Quantization and CPU Offload for DiTs<br>
      <a href="https://pepy.tech/projects/cache-dit"><img src=https://static.pepy.tech/personalized-badge/cache-dit?period=total&units=ABBREVIATION&left_color=GRAY&right_color=BLUE&left_text=downloads/pypi ></a>
      <a href="https://pypi.org/project/cache-dit/"><img src=https://img.shields.io/github/release/vipshop/cache-dit.svg?color=GREEN ></a>
      <img src="https://img.shields.io/github/license/vipshop/cache-dit.svg?color=blue">
      <a href="https://cache-dit.readthedocs.io/en/latest/COMMUNITY/"><img src=https://img.shields.io/badge/🤗-Community-orange.svg ></a> 
      <a href="https://hellogithub.com/repository/vipshop/cache-dit" target="_blank"><img src="https://api.hellogithub.com/v1/widgets/recommend.svg?rid=b8b03b3b32a449ea84cfc2b96cd384f3&claim_uid=ofSCbzTmdeQk3FD&theme=small" alt="Featured｜HelloGitHub" /></a> 
    </h2>
  </p>
</div>

**🤗Why Cache-DiT❓❓**Cache-DiT is built on top of the 🤗[Diffusers](https://github.com/huggingface/diffusers) library and now supports nearly [ALL](https://cache-dit.readthedocs.io/en/latest/supported_matrix/NVIDIA_GPU/) DiTs from Diffusers. It provides [hybrid cache acceleration](https://cache-dit.readthedocs.io/en/latest/user_guide/CACHE_API/) (DBCache, TaylorSeer, SCM, etc.) and comprehensive [parallelism](https://cache-dit.readthedocs.io/en/latest/user_guide/CONTEXT_PARALLEL/) optimizations, including Context Parallelism, Tensor Parallelism, hybrid 2D or 3D parallelism, and dedicated extra parallelism support for Text Encoder, VAE, and ControlNet. 

<div align="center">
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/arch_v2.png width=815px>
</div>

Cache-DiT is compatible with compilation, CPU Offloading, and quantization, fully integrates with [SGLang Diffusion](https://docs.sglang.io/diffusion/performance/cache/cache_dit.html), [vLLM-Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion/cache_acceleration/cache_dit/), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/pull/12548), ComfyUI, and runs natively on NVIDIA GPUs, Ascend NPUs and AMD GPUs. Cache-DiT is **fast**, **easy to use**, and **flexible** for various DiTs (online docs at 📘[cache-dit.io](https://cache-dit.readthedocs.io/en/latest/)).

<div align="center">
  <p align="center">
    <i>⚡️9x speedup by Cache-DiT with Cache, Context Parallelism and Compilation</i>
  </p>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/speedup_v5.png width=800px>
</div>

<div align="center">
  <p> <h2>🚀Quick Start: Cache, Parallelism and Quantization</h2> </p>
</div>

First, you can install the cache-dit from PyPI or install from source: 

```bash
uv pip install -U cache-dit # PyPI, stable release.
uv pip install git+https://github.com/vipshop/cache-dit.git # latest.
```

Then, try to accelerate your DiTs with just **♥️one line♥️** of code ~  

```python
>>> import cache_dit
>>> from diffusers import DiffusionPipeline
>>> pipe = DiffusionPipeline.from_pretrained(...).to("cuda")
>>> cache_dit.enable_cache(pipe) # Cache Acceleration with One-line code.
>>> from cache_dit import DBCacheConfig, ParallelismConfig
>>> cache_dit.enable_cache( # Or, Hybrid Cache Acceleration + Parallelism.
...   pipe, cache_config=DBCacheConfig(), # w/ default
...   parallelism_config=ParallelismConfig(ulysses_size=2))
>>> from cache_dit import DBCacheConfig, ParallelismConfig, QuantizeConfig
>>> cache_dit.enable_cache( # Or, Hybrid Cache + Parallelism + Quantization.
...   pipe, cache_config=DBCacheConfig(), # w/ default
...   parallelism_config=ParallelismConfig(ulysses_size=2),
...   quantize_config=QuantizeConfig(quant_type=...))
>>> output = pipe(...) # Then, just call the pipe as normal.
```

<div align="center">
  <p> <h2>🚀Quick Start: SVDQuant (W4A4) PTQ/DQ workflow</h2> </p>
</div>

First, install Cache-DiT with SVDQuant support (Experimental):

```bash
# Required: CUDA 13.0+, PyTorch 2.11+, Ubuntu 22.04+.
uv pip install -U cache-dit-cu13 # PyPI, stable release.
CACHE_DIT_BUILD_SVDQUANT=1 uv pip install -e ".[quantization]" # latest.
```

Then, try to quantize your model with just **♥️a few lines♥️** of code ~

```python
>>> from cache_dit import QuantizeConfig
>>> pipe = DiffusionPipeline.from_pretrained(...).to("cuda")
>>> # DQ: "...{dtype}_r{rank}_dq", PTQ: "...{dtype}_r{rank}"
>>> pipe.transformer = cache_dit.quantize(
...   pipe.transformer, quant_config=QuantizeConfig(
...   # ✅ NO calibration needed for SVDQ DQ in Cache-DiT! 🎉
...   quant_type="svdq_{int4|nvfp4}_r{32|64|128|256|...}_dq",
...   svdq_kwargs={"smooth_strategy": "few_shot"})) 
>>> output = pipe(...) # Then, just call the pipe as normal.
```

<div align="center">
  <p> <h2>🚀Quick Start: Bucket-style Layerwise CPU Offload</h2> </p>
</div>

Cache-DiT's offer a bucket-style layerwise offload tech that with nearly zero (**<5%🎉**) latency overhead.

```python
>>> import cache_dit
>>> cache_dit.layerwise_offload(
...   pipe, # Any nn.Module: pipe, transformer, text_encoder, etc.
...   onload_device="cuda",
...   offload_device="cpu",
...   async_transfer=True,
...   transfer_buckets=4,
...   persistent_buckets=64,
...   persistent_bins=8,
...   prefetch_limit=True,
...   max_copy_streams=4,
...   max_inflight_prefetch_bytes="8gib")
>>> output = pipe(...) # Then, just call the pipe as normal.
```

For more advanced features, please refer to our online documentation at 📘[cache-dit.io](https://cache-dit.readthedocs.io/en/latest/user_guide/OVERVIEWS/).

## 🌐Community Integration

- 🎉[ComfyUI x Cache-DiT](https://github.com/Jasonzzt/ComfyUI-CacheDiT)
- 🎉[(Intel) llm-scaler x Cache-DiT](https://github.com/intel/llm-scaler/tree/main/omni#cache-dit--torchcompile-acceleration)
- 🎉[Diffusers x Cache-DiT](https://huggingface.co/docs/diffusers/main/en/optimization/cache_dit)
- 🎉[TensorRT-LLM x Cache-DiT](https://github.com/NVIDIA/TensorRT-LLM/pull/12548)
- 🎉[SGLang Diffusion x Cache-DiT](https://docs.sglang.io/diffusion/performance/cache/cache_dit.html)
- 🎉[vLLM-Omni x Cache-DiT](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion/cache_acceleration/cache_dit/)
- 🎉[Nunchaku x Cache-DiT](https://nunchaku.tech/docs/nunchaku/usage/cache.html#cache-dit)
- 🎉[SD.Next x Cache-DiT](https://github.com/vladmandic/sdnext/blob/master/modules/cachedit.py)
- 🎉[stable-diffusion.cpp x Cache-DiT](https://github.com/leejet/stable-diffusion.cpp/blob/master/cache_dit.hpp)
- 🎉[jetson-containers x Cache-DiT](https://github.com/dusty-nv/jetson-containers/tree/master/packages/cv/diffusion/cache_edit)


## ©️Acknowledgements

Special thanks to vipshop's Computer Vision AI Team for supporting testing and deployment of this project. We learned and reused codes from: [Diffusers](https://github.com/huggingface/diffusers), [SGLang](https://github.com/sgl-project/sglang), [vLLM-Omni](https://github.com/vllm-project/vllm-omni), [Nunchaku](https://github.com/nunchaku-ai/nunchaku), [xDiT](https://github.com/xdit-project/xDiT) and [TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer).


## ©️Citations

<div id="citations"></div>

```BibTeX
@misc{cache-dit@2025,
  title={Cache-DiT: A PyTorch-native Inference Engine with Cache, Parallelism and Quantization for Diffusion Transformers.},
  url={https://github.com/vipshop/cache-dit.git},
  note={Open-source software available at https://github.com/vipshop/cache-dit.git},
  author={DefTruth, vipshop.com, etc.},
  year={2025}
}
```
