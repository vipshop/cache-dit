<div align="center">
  <p align="center">
    <h2 align="center">
        🤗🎉Cache-DiT: A PyTorch-native Inference Engine with <br>Hybrid Cache Acceleration and Massive Parallelism for DiTs 
    </h2>
    <a href="https://pepy.tech/projects/cache-dit"><img src=https://static.pepy.tech/personalized-badge/cache-dit?period=total&units=ABBREVIATION&left_color=GRAY&right_color=BLUE&left_text=downloads/pypi ></a>
    <a href="https://pypi.org/project/cache-dit/"><img src=https://img.shields.io/github/release/vipshop/cache-dit.svg?color=GREEN ></a>
    <img src="https://img.shields.io/github/license/vipshop/cache-dit.svg?color=blue">
    <a href="https://cache-dit.readthedocs.io/en/latest/COMMUNITY/"><img src=https://img.shields.io/badge/🤗-Community-orange.svg ></a> 
    <a href="https://hellogithub.com/repository/vipshop/cache-dit" target="_blank"><img src="https://api.hellogithub.com/v1/widgets/recommend.svg?rid=b8b03b3b32a449ea84cfc2b96cd384f3&claim_uid=ofSCbzTmdeQk3FD&theme=small" alt="Featured｜HelloGitHub" /></a><br>
    <script async defer src="https://buttons.github.io/buttons.js"></script>
    <a class="github-button" href="https://github.com/vipshop/cache-dit" data-show-count="true" data-size="large" aria-label="Star">Star</a>
    <a class="github-button" href="https://github.com/vipshop/cache-dit/subscription" data-show-count="true" data-icon="octicon-eye" data-size="large" aria-label="Watch">Watch</a>
    <a class="github-button" href="https://github.com/vipshop/cache-dit/fork" data-show-count="true" data-icon="octicon-repo-forked" data-size="large" aria-label="Fork">Fork</a>
  </p>
</div>


**🤗Why Cache-DiT❓❓**Cache-DiT is built on top of the 🤗[Diffusers](https://github.com/huggingface/diffusers) library and now supports nearly [ALL](https://cache-dit.readthedocs.io/en/latest/supported_matrix/NVIDIA_GPU/) DiTs from Diffusers. It provides [hybrid cache acceleration](https://cache-dit.readthedocs.io/en/latest/user_guide/CACHE_API/) (DBCache, TaylorSeer, SCM, etc.) and comprehensive [parallelism](https://cache-dit.readthedocs.io/en/latest/user_guide/CONTEXT_PARALLEL/) optimizations, including Context Parallelism, Tensor Parallelism, hybrid 2D or 3D parallelism, and dedicated extra parallelism support for Text Encoder, VAE, and ControlNet.  

<div align="center">
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/arch_v2.png width=815px>
</div>

Cache-DiT is compatible with compilation, CPU Offloading, and quantization, fully integrates with [SGLang Diffusion](https://docs.sglang.io/diffusion/performance/cache/cache_dit.html), [vLLM-Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion/cache_acceleration/cache_dit/), ComfyUI, and runs natively on NVIDIA GPUs, Ascend NPUs and AMD GPUs. Cache-DiT is **fast**, **easy to use**, and **flexible** for various DiTs (online docs at 📘[readthedocs.io](https://cache-dit.readthedocs.io/en/latest/)).

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
uv pip install -U cache-dit # or, uv pip install git+https://github.com/vipshop/cache-dit.git
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
...   quantize_config=QuantizeConfig(quant_type="float8_per_row"))
>>> output = pipe(...) # Then, just call the pipe as normal.
```

For more advanced features, please refer to our online documentation at 📘[Documentation](https://cache-dit.readthedocs.io/en/latest/user_guide/OVERVIEWS/).

<div align="center">
  <p> <h2>🚀Quick Start: SVDQuant (W4A4) PTQ workflow</h2> </p>
</div>

First, build Cache-DiT from source with SVDQuant support (Experimental):

```bash
git clone https://github.com/vipshop/cache-dit.git && cd cache-dit
CACHE_DIT_BUILD_SVDQUANT=1 uv pip install -e ".[quantization]" --no-build-isolation
```

Then, try to quantize your model with just **♥️a few lines♥️** of code ~

```python
>>> from cache_dit import QuantizeConfig
>>> pipe = DiffusionPipeline.from_pretrained(...).to("cuda")
>>> # 0. Define the calibration function for PTQ.
>>> def calibrate_fn(**_: object) -> None:
...     with torch.inference_mode():
...         for prompt in calibration_prompts:
...             _ = pipe(prompt=prompt, ...)
>>> # 1. Build the QuantizeConfig for SVDQuant PTQ.
>>> quant_config = QuantizeConfig(
...     quant_type="svdq_int4_r32", # _r{rank}, e.g., r16, r32, r64, r128, etc.
...     calibrate_fn=calibrate_fn,
...     serialize_to=..., 
... )
>>> # 2. Apply quantization with `cache_dit.quantize(...)` API.
>>> pipe.transformer = cache_dit.quantize(pipe.transformer, quant_config) 
>>> output = pipe(...) # 3. Use the quantized model for inference.
>>> # 4. Or, reload the quantized model from disk for inference.
>>> pipe.transformer = cache_dit.load(pipe.transformer, ...)
```

For more advanced features, please refer to our online documentation at 📘[Low-bits Quantization](https://cache-dit.readthedocs.io/en/latest/user_guide/QUANTIZATION/).

## 🌐Community Integration

- 🎉[ComfyUI x Cache-DiT](https://github.com/Jasonzzt/ComfyUI-CacheDiT)
- 🎉[(Intel) llm-scaler x Cache-DiT](https://github.com/intel/llm-scaler/tree/main/omni#cache-dit--torchcompile-acceleration)
- 🎉[Diffusers x Cache-DiT](https://huggingface.co/docs/diffusers/main/en/optimization/cache_dit)
- 🎉[SGLang Diffusion x Cache-DiT](https://docs.sglang.io/diffusion/performance/cache/cache_dit.html)
- 🎉[vLLM-Omni x Cache-DiT](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion/cache_acceleration/cache_dit/)
- 🎉[Nunchaku x Cache-DiT](https://nunchaku.tech/docs/nunchaku/usage/cache.html#cache-dit)
- 🎉[SD.Next x Cache-DiT](https://github.com/vladmandic/sdnext/blob/master/modules/cachedit.py)
- 🎉[stable-diffusion.cpp x Cache-DiT](https://github.com/leejet/stable-diffusion.cpp/blob/master/cache_dit.hpp)
- 🎉[jetson-containers x Cache-DiT](https://github.com/dusty-nv/jetson-containers/tree/master/packages/cv/diffusion/cache_edit)


## ©️Acknowledgements

Special thanks to vipshop's Computer Vision AI Team for supporting document, testing and deployment of this project. We learned the design and reused code from the following projects: [Diffusers](https://github.com/huggingface/diffusers), [SGLang](https://github.com/sgl-project/sglang), [vLLM](https://github.com/vllm-project/vllm), [vLLM-Omni](https://github.com/vllm-project/vllm-omni), [ParaAttention](https://github.com/chengzeyi/ParaAttention), [xDiT](https://github.com/xdit-project/xDiT) and [TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer).


## ©️Citations

<div id="citations"></div>

```BibTeX
@misc{cache-dit@2025,
  title={Cache-DiT: A PyTorch-native Inference Engine with Hybrid Cache Acceleration and Massive Parallelism for DiTs.},
  url={https://github.com/vipshop/cache-dit.git},
  note={Open-source software available at https://github.com/vipshop/cache-dit.git},
  author={DefTruth, vipshop.com, etc.},
  year={2025}
}
```
