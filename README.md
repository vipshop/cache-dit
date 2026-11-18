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

**🤗Cache-DiT** is a PyTorch-native inference engine with [cache](https://cache-dit.readthedocs.io/en/latest/user_guide/CACHE_API/) (DBCache, TaylorSeers, FoCa, SCM, SVD-Cache, DMD-Cache, etc), [quantization](https://cache-dit.readthedocs.io/en/latest/user_guide/QUANTIZATION/), [offload](https://cache-dit.readthedocs.io/en/latest/user_guide/OFFLOAD/) and [parallelism](https://cache-dit.readthedocs.io/en/latest/user_guide/CONTEXT_PARALLEL/) (CP/TP/TE-P/VAE-P/Hybrid) for DiTs. 

<div align="center">
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/arch_v2.png width=815px>
</div>

Cache-DiT is compatible with compilation, offload, and quantization, fully integrates with [SGLang Diffusion](https://docs.sglang.io/diffusion/performance/cache/cache_dit.html), [vLLM-Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion/cache_acceleration/cache_dit/), [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/models/visual-generation.html#cache-dit), [xDiT](https://github.com/xdit-project/xDiT/tree/main/xfuser/model_executor/cache/adapters/cache_dit), and runs natively on NVIDIA GPUs, Huawei Ascend NPUs and AMD GPUs. Cache-DiT is **fast**, **easy to use**, and **flexible** for various DiTs.

<div align="center">
  <p align="center">
    <i>⚡️9x speedup by Cache-DiT with Cache, Context Parallelism and Compilation</i>
  </p>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/speedup_v5.png width=800px>
</div>

Cache-DiT supports **60+ DiT pipeline families (150+ Variants)** from 🤗Diffusers, covering the vast majority of DiT-based pipelines. For full support matrix and detailed usage, please refer to our docs at 📘[cache-dit.io](https://cache-dit.readthedocs.io/en/latest/).

<div align="center">
  <p> <h2>🚀Quick Start: Cache, Parallelism and Quantization</h2> </p>
</div>

```bash
pip install -U cache-dit # PyPI, stable release.
pip install git+https://github.com/vipshop/cache-dit.git # latest.
```

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

## 🌐Community Integration

- 🎉[xDiT x Cache-DiT](https://github.com/xdit-project/xDiT/tree/main/xfuser/model_executor/cache/adapters/cache_dit)
- 🎉[ComfyUI x Cache-DiT](https://github.com/Jasonzzt/ComfyUI-CacheDiT)
- 🎉[Diffusers x Cache-DiT](https://huggingface.co/docs/diffusers/main/en/optimization/cache_dit)
- 🎉[MindIE-SD x Cache-DiT](https://mindie-sd.readthedocs.io/en/latest/zh/features/supported_matrix.html)
- 🎉[TensorRT-LLM x Cache-DiT](https://nvidia.github.io/TensorRT-LLM/models/visual-generation.html#cache-dit)
- 🎉[SGLang Omni x Cache-DiT](https://github.com/sgl-project/sglang-omni/blob/abc7640446629a837d043355ff7a756eb2a0190a/sglang_omni/models/minimax_music3/dit.py#L329)
- 🎉[SGLang Diffusion x Cache-DiT](https://docs.sglang.io/diffusion/performance/cache/cache_dit.html)
- 🎉[vLLM-Omni x Cache-DiT](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion/cache_acceleration/cache_dit/)
- 🎉[Nunchaku x Cache-DiT](https://nunchaku.tech/docs/nunchaku/usage/cache.html#cache-dit)
- 🎉[SD.Next x Cache-DiT](https://github.com/vladmandic/sdnext/blob/master/modules/cachedit.py)

## ©️Acknowledgements

Special thanks to vipshop's Computer Vision AI Team for supporting testing and deployment of this project. We learned and reused codes from: [Diffusers](https://github.com/huggingface/diffusers), [SGLang](https://github.com/sgl-project/sglang), [vLLM-Omni](https://github.com/vllm-project/vllm-omni), [Nunchaku](https://github.com/nunchaku-ai/nunchaku), [xDiT](https://github.com/xdit-project/xDiT) and [TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer).


## ©️Citations

<div id="citations"></div>

```BibTeX
@misc{cache-dit@2025,
  title={Cache-DiT: A PyTorch-native Inference Engine with Cache, Parallelism, Quantization and CPU Offload for DiTs.},
  url={https://github.com/vipshop/cache-dit.git},
  note={Open-source software available at https://github.com/vipshop/cache-dit.git},
  author={DefTruth, vipshop.com, etc.},
  year={2025}
}
```
