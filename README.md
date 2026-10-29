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

Cache-DiT is compatible with compilation, CPU Offloading, and quantization, fully integrates with [SGLang Diffusion](https://docs.sglang.io/diffusion/performance/cache/cache_dit.html), [vLLM-Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion/cache_acceleration/cache_dit/), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/pull/12548), ComfyUI, and runs natively on NVIDIA GPUs, Huawei Ascend NPUs and AMD GPUs. Cache-DiT is **fast**, **easy to use**, and **flexible** for various DiTs (online docs at 📘[cache-dit.io](https://cache-dit.readthedocs.io/en/latest/)).

<div align="center">
  <p align="center">
    <i>⚡️9x speedup by Cache-DiT with Cache, Context Parallelism and Compilation</i>
  </p>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/speedup_v5.png width=800px>
</div>

<div align="center">
  <p> <h2>🚀Quick Start: Cache, Parallelism and Quantization</h2> </p>
</div>

```bash
pip3 install -U cache-dit # PyPI, stable release.
pip3 install git+https://github.com/vipshop/cache-dit.git # latest.
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

<div align="center">
  <p> <h2>🚀Quick Start: SVDQuant (W4A4) PTQ/DQ workflow</h2> </p>
</div>

```bash
# Required: CUDA >=13.0, PyTorch >=2.11 (a CUDA 13 build), GLIBC 2.38.
pip3 install -U cache-dit --extra-index-url https://vipshop.github.io/cache-dit/whl/cu13/ 
CACHE_DIT_BUILD_SVDQUANT=1 uv pip install -e ".[quantization]" # latest.
```

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

```python
>>> import cache_dit
>>> cache_dit.layerwise_offload(
...   pipe, # nn.Module: pipe, transformer, text_encoder, etc.
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

## 📋Supported DiT Models

Cache-DiT supports **40+ DiT pipeline families (120+ Variants)** from 🤗Diffusers, covering the vast majority of DiT-based pipelines. For full support matrix and detailed usage, please refer to our documentation at 📘[cache-dit.io](https://cache-dit.readthedocs.io/en/latest/).

[diffusers-url]: https://github.com/huggingface/diffusers/tree/main/src/diffusers/models/transformers

<div align="center">

| Modality | Pipeline Series | Model Sources | Variants | C/P/Q/OF |
|:----------|:----------------|:-------------|:----------|:-------|
| Image | FLUX.3 Klein 🔥 | [`diffusers@0.42.0`][diffusers-url] | 1 | ✔️/✔️/✔️/✔️ |
| Image | Bria-Fibo 2.0 🔥 | [`diffusers@0.42.0`][diffusers-url] | 2 | ✔️/✔️/✔️/✔️ |
| Video | Kandinsky 6.0 (T2VA) 🔥 | [`diffusers@0.42.0`][diffusers-url] | 1 | ✔️/✔️/✔️/✖️ |
| Image | Qwen-Image-2.1 🔥| [`diffusers@0.41.0`][diffusers-url] | 3 | ✔️/✔️/✔️/✔️ |
| Video | LTX2.5 (DFR/T2V/I2V) 🔥| [`diffusers@0.41.0`][diffusers-url] | 6 | ✔️/✔️/✔️/✔️ |
| Video | Echo (JoyAI-Echo) 🔥 | [`diffusers@0.41.0`][diffusers-url] | 1 | ✔️/✔️/✔️/✔️ |
| Video | MiniMax-H3 (T2VA) 🔥 | [`diffusers@0.40.0`][diffusers-url] | 1 | ✔️/✔️/✔️/✔️ |
| Video | MiniMax-H3 (FL2VA) 🔥 | [`diffusers@0.40.0`][diffusers-url] | 1 | ✔️/✔️/✔️/✔️ |  
| Video | MiniMax-H3 (Ref2VA) 🔥 | [`diffusers@0.40.0`][diffusers-url] | 1 | ✔️/✔️/✔️/✔️ |  
| Image | Joy-Image| [`diffusers@0.39.0`][diffusers-url] | 1 | ✔️/✔️/✔️/✔️ |
| Image | Krea-2 | [`diffusers@0.39.0`][diffusers-url] | 1 | ✔️/✔️/✔️/✔️ |
| Image | Ideogram 4 🔥 | [`diffusers@0.39.0`][diffusers-url] | 1 | ✔️/✔️/✔️/✔️ |
| Image | Anima | [`diffusers@0.39.0`][diffusers-url] | 1 | ✔️/✔️/✔️/✔️ |
| Video | Any-Flow | [`diffusers@0.39.0`][diffusers-url] | 1 | ✔️/✔️/✔️/✔️ |
| Video | Any-Flow-FAR | [`diffusers@0.39.0`][diffusers-url] | 1 | ✖️/✖️/✔️/✔️ |
| Video | Cosmos3 (Omni) 🔥| [`diffusers@0.39.0`][diffusers-url] | 2 | ✔️/✖️/✔️/✔️ |
| Video | Motif-Video (T2V/I2V) 🔥 | [`diffusers@0.39.0`][diffusers-url] | 2 | ✔️/✔️/✔️/✔️ |
| Image | ERNIE-Image 🔥 | [`diffusers@0.38.0`][diffusers-url] | 1 | ✔️/✔️/✔️/✔️ |
| Image | NucleusMoE-Image 🔥 | [`diffusers@0.38.0`][diffusers-url] | 1 | ✔️/✔️/✔️/✔️ |
| Image | FLUX.2 Klein 🔥 | [`diffusers@0.37.0`][diffusers-url] | 3 | ✔️/✔️/✔️/✔️ |
| Image | GLM-Image 🔥 | [`diffusers@0.37.0`][diffusers-url] | 1 | ✔️/✔️/✔️/✔️ |
| Image | LongCat-Image 🔥 | [`diffusers@0.37.0`][diffusers-url] | 2 | ✔️/✔️/✔️/✔️ |
| Video | LTX2 | [`diffusers@0.37.0`][diffusers-url] | 6 | ✔️/✔️/✔️/✔️ |
| Video | Helios 🔥 | [`diffusers@0.37.0`][diffusers-url] | 2 | ✔️/✔️/✔️/✔️ |
| Image | FLUX.2 🔥 | [`diffusers@0.36.0`][diffusers-url] | 1 | ✔️/✔️/✔️/✔️ |
| Image | Z-Image 🔥 | [`diffusers@0.36.0`][diffusers-url] | 6 | ✔️/✔️/✔️/✔️ |
| Image | Bria-Fibo | [`diffusers@0.36.0`][diffusers-url] | 2 | ✔️/✔️/✔️/✔️ |
| Image | Ovis-Image | [`diffusers@0.36.0`][diffusers-url] | 1 | ✔️/✔️/✔️/✔️ |
| Image | Hunyuan-Image 🔥 | [`diffusers@0.36.0`][diffusers-url] | 2 | ✔️/✔️/✔️/✔️ |
| Image | Bria | [`diffusers@0.36.0`][diffusers-url] | 1 | ✔️/✖️/✔️/✔️ |
| Image | PRX / PRXPixel 🔥| [`diffusers@0.36.0`][diffusers-url] | 2 | ✔️/✔️/✔️/✔️ |
| Video | HunyuanVideo 1.5 🔥| [`diffusers@0.36.0`][diffusers-url] | 2 | ✔️/✖️/✔️/✔️ |
| Video | ChronoEdit | [`diffusers@0.36.0`][diffusers-url] | 1 | ✔️/✔️/✔️/✔️ |
| Video | Kandinsky5 (T2V/I2V) | [`diffusers@0.36.0`][diffusers-url] | 4 | ✔️/✔️/✔️/✔️ |
| Other | LucyEdit | [`diffusers@0.36.0`][diffusers-url]  | 1 | ✔️/✔️/✔️/✔️ |
| Image | Qwen-Image 🔥 | [`diffusers@0.35.0`][diffusers-url] | 9+ | ✔️/✔️/✔️/✔️ |
| Video | Wan 2.2 (T2V/I2V) 🔥| [`diffusers@0.35.0`][diffusers-url] | 3 | ✔️/✔️/✔️/✔️ |
| Video | SkyReelsV2 | [`diffusers@0.35.0`][diffusers-url] | 5 | ✔️/✔️/✔️/✔️ |
| Image | HiDream | [`diffusers@0.34.0`][diffusers-url] | 1 | ✔️/✖️/✔️/✔️ |
| Image | Chroma | [`diffusers@0.34.0`][diffusers-url] | 3 | ✔️/✔️/✔️/✔️ |
| Image | VisualCloze | [`diffusers@0.34.0`][diffusers-url]  | 2 | ✔️/✔️/✔️/✔️ |
| Video | Cosmos | [`diffusers@0.34.0`][diffusers-url] | 7+ | ✔️/✖️/✔️/✔️ |
| Video | Wan-VACE 🔥| [`diffusers@0.34.0`][diffusers-url] | 1 | ✔️/✔️/✔️/✔️ |
| Image | CogView4 | [`diffusers@0.33.0`][diffusers-url] | 2 | ✔️/✔️/✔️/✔️ |
| Image | OmniGen | [`diffusers@0.33.0`][diffusers-url] | 1 | ✔️/✖️/✔️/✔️ |
| Image | Lumina 2 | [`diffusers@0.33.0`][diffusers-url] | 2 | ✔️/✖️/✔️/✔️ |
| Video | Wan 2.1 (T2V/I2V) 🔥| [`diffusers@0.33.0`][diffusers-url] | 3 | ✔️/✔️/✔️/✔️ |
| Video | EasyAnimate | [`diffusers@0.33.0`][diffusers-url] | 3 | ✔️/✖️/✔️/✔️ |
| Video | ConsisID | [`diffusers@0.33.0`][diffusers-url] | 1 | ✔️/✔️/✔️/✔️ |
| Image | Sana (image) | [`diffusers@0.32.0`][diffusers-url] | 4 | ✔️/✖️/✔️/✔️ |
| Video | HunyuanVideo | [`diffusers@0.32.0`][diffusers-url] | 4 | ✔️/✔️/✔️/✔️ |
| Video | Mochi | [`diffusers@0.32.0`][diffusers-url] | 1 | ✔️/✔️/✔️/✔️ |
| Video | Allegro | [`diffusers@0.32.0`][diffusers-url] | 1 | ✔️/✖️/✔️/✔️ |
| Video | LTX | [`diffusers@0.32.0`][diffusers-url] | 5 | ✔️/✔️/✔️/✔️ |
| Image | CogView3Plus | [`diffusers@0.31.0`][diffusers-url] | 1 | ✔️/✔️/✔️/✔️ |
| Image | FLUX | [`diffusers@0.30.0`][diffusers-url] | 10+ | ✔️/✔️/✔️/✔️ |
| Image | HunyuanDiT-PAG | [`diffusers@0.30.0`][diffusers-url] | 1 | ✔️/✔️/✔️/✔️ |
| Image | AuraFlow | [`diffusers@0.30.0`][diffusers-url] | 1 | ✔️/✖️/✔️/✔️ |
| Image | Lumina | [`diffusers@0.30.0`][diffusers-url] | 2 | ✔️/✖️/✔️/✔️ |
| Video | CogVideoX | [`diffusers@0.30.0`][diffusers-url] | 4 | ✔️/✔️/✔️/✔️ |
| Audio | StableAudio | [`diffusers@0.30.0`][diffusers-url] | 1 | ✔️/✖️/✔️/✔️ |
| Image | SD3 | [`diffusers@0.29.0`][diffusers-url] | 3 | ✔️/✖️/✔️/✔️ |
| Image | HunyuanDiT | [`diffusers@0.28.1`][diffusers-url] | 1 | ✔️/✔️/✔️/✔️ |
| Image | PixArt-Sigma | [`diffusers@0.28.0`][diffusers-url] | 1 | ✔️/✔️/✔️/✔️ |
| Image | Amused | [`diffusers@0.25.0`][diffusers-url] | 1 | ✔️/✖️/✔️/✔️ |
| Image | PixArt-Alpha | [`diffusers@0.22.0`][diffusers-url] | 1 | ✔️/✔️/✔️/✔️ |
| 3D/Other | Shap-E | [`diffusers@0.18.0`][diffusers-url] | 1 | ✔️/✖️/✔️/✔️ |
| Image | DiT | [`diffusers@0.12.0`][diffusers-url] | 1 | ✔️/✔️/✔️/✔️ |

<i> <b>C</b>: Hybrid Cache (DBCache + Calibrator: TaylorSeer/DMD/FoCa/SCM); <b>P</b>: Parallelism (Ulysses/Ring/USP/TP<br>TE-P/VAE-P/2D-P/3D-P); <b>Q</b>: Quantization; <b>OF</b>: Bucket-style Layerwise Offload; @: Minimum version </i>

</div>

<div align="center">
  <p> <h2>🚀Built-in Pipelines Beyond 🤗Diffusers</h2>
</div>

[cache-dit-url]: https://github.com/vipshop/cache-dit/tree/main/src/cache_dit/models/pipelines

<div align="center">

| Modality | Pipeline Series | Model Sources | Variants | C/P/Q/OF |
|:----------|:----------------|:-------------|:----------|:-------|
| **Image** | MOSS-Image 🔥 | [`cache-dit@main`][cache-dit-url] | 2 | ✔️/✔️/✔️/✔️ |
| Image | Ming-Image 0.1 🔥 | [`cache-dit@main`][cache-dit-url] | 2 | ✔️/✔️/✔️/✔️ |
| Image | HiDream-O1-Image 🔥 | [`cache-dit@main`][cache-dit-url] | 1 | ✔️/✔️/✔️/✔️ |

<i>Cache-DiT also provides built-in pipelines for models that are not currently included in 🤗Diffusers</i>
</div>

## 🤖Agentic Workflows  

Cache-DiT provides a [model-integration](./.github/skills/cache-dit-model-integration/SKILL.md) SKILL to help users integrate new DiT pipelines into Cache-DiT, including **Cache, CP, TP, TE-P, VAE-P** and carefully designed test cases. Users can use it with Coding Agents, e.g, [GitHub Copilot](https://docs.github.com/en/copilot), [Claude Code](https://claude.ai), [Open Code](https://opencode.ai/). 

```bash
gh skill install vipshop/cache-dit cache-dit-model-integration --allow-hidden-dirs
```

> [!NOTE]
> Please note that quantization and layerwise offload in Cache-DiT are generally supported for **nn.Module**, thus no extra integration is needed for new DiT pipelines or transformers. 

## 🌐Community Integration

- 🎉[MindIE-SD x Cache-DiT](https://github.com/vipshop/cache-dit/pull/1004)
- 🎉[xDiT x Cache-DiT](https://github.com/xdit-project/xDiT/tree/main/xfuser/model_executor/cache/adapters/cache_dit)
- 🎉[ComfyUI x Cache-DiT](https://github.com/Jasonzzt/ComfyUI-CacheDiT)
- 🎉[Diffusers x Cache-DiT](https://huggingface.co/docs/diffusers/main/en/optimization/cache_dit)
- 🎉[TensorRT-LLM x Cache-DiT](https://github.com/NVIDIA/TensorRT-LLM/pull/12548)
- 🎉[SGLang Omni x Cache-DiT](https://github.com/sgl-project/sglang-omni/blob/abc7640446629a837d043355ff7a756eb2a0190a/sglang_omni/models/minimax_music3/dit.py#L329)
- 🎉[SGLang Diffusion x Cache-DiT](https://docs.sglang.io/diffusion/performance/cache/cache_dit.html)
- 🎉[vLLM-Omni x Cache-DiT](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion/cache_acceleration/cache_dit/)
- 🎉[Nunchaku x Cache-DiT](https://nunchaku.tech/docs/nunchaku/usage/cache.html#cache-dit)
- 🎉[SD.Next x Cache-DiT](https://github.com/vladmandic/sdnext/blob/master/modules/cachedit.py)
- 🎉[stable-diffusion.cpp x Cache-DiT](https://github.com/leejet/stable-diffusion.cpp/blob/master/cache_dit.hpp)
- 🎉[jetson-containers x Cache-DiT](https://github.com/dusty-nv/jetson-containers/tree/master/packages/cv/diffusion/cache_edit)
- 🎉[RH-MinMax-H3 x Cache-DiT](https://github.com/HM-RunningHub/ComfyUI_RH_MinMaxH3)


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
