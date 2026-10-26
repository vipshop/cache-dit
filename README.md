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
pip3 install -U cache-dit # PyPI, stable release.
pip3 install git+https://github.com/vipshop/cache-dit.git # latest.
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

First, install the pre-built CUDA package, or build Cache-DiT from source with SVDQuant support (Experimental):

```bash
# Required: CUDA 13.0+, PyTorch 2.11+ (a CUDA 13 build), and the GLIBC floor of the
# release (2.38; see the platform tags on the index page). SVDQuant built with cu13.
pip3 install -U cache-dit --extra-index-url https://vipshop.github.io/cache-dit/whl/cu13/ 
CACHE_DIT_BUILD_SVDQUANT=1 uv pip install -e ".[quantization]" # latest.
```

Then, try to quantize your model with just **♥️a few lines♥️** of codes ~

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

**Bucket-style** Layerwise Offload w/ nearly zero (**<5%🎉**) latency overhead ~

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

For more advanced features, please refer to our online documentation at 📘[cache-dit.io](https://cache-dit.readthedocs.io/en/latest/user_guide/OVERVIEWS/).

## 📋Supported DiT Models

Cache-DiT supports **40+ DiT pipeline families (120+ Variants)** from 🤗Diffusers, covering the vast majority of DiT-based pipelines. For full support matrix and detailed usage, please refer to our documentation at 📘[cache-dit.io](https://cache-dit.readthedocs.io/en/latest/).

<div align="center">

| Modality | Pipeline Series | Transformer | Variants | C/P/Q/OF |
|:----------|:----------------|:-------------|:----------|:-------|
| **Image** | FLUX | [`FluxTransformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 10+ | ✔️/✔️/✔️/✔️ |
| Image | FLUX.2 | [`Flux2Transformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✔️/✔️/✔️ |
| Image | FLUX.2 Klein | [`Flux2Transformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 3 | ✔️/✔️/✔️/✔️ |
| Image | FLUX.3 Klein 🔥 | [`Flux3Transformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✔️/✔️/✔️ |
| Image | ERNIE-Image | [`ErnieImageTransformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✔️/✔️/✔️ |
| Image | Qwen-Image | [`QwenImageTransformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 9+ | ✔️/✔️/✔️/✔️ |
| Image | Qwen-Image-2.1 🔥| [`QwenImage21Transformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 3 | ✔️/✔️/✔️/✔️ |
| Image | Z-Image | [`ZImageTransformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 6 | ✔️/✔️/✔️/✔️ |
| Image | Joy-Image| [`JoyImageEditTransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✔️/✔️/✔️ |
| Image | Krea-2 | [`Krea2Transformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✔️/✔️/✔️ |
| Image | Bria-Fibo | [`BriaFiboTransformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 2 | ✔️/✔️/✔️/✔️ |
| Image | Bria-Fibo 2.0 | [`BriaFibo2Transformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 2 | ✔️/✔️/✔️/✔️ |
| Image | GLM-Image | [`GlmImageTransformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✔️/✔️/✔️ |
| Image | LongCat-Image | [`LongCatImageTransformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 2 | ✔️/✔️/✔️/✔️ |
| Image | Ovis-Image | [`OvisImageTransformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✔️/✔️/✔️ |
| Image | SD3 | [`SD3Transformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 3 | ✔️/✖️/✔️/✔️ |
| Image | PixArt-Alpha | [`PixArtTransformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✔️/✔️/✔️ |
| Image | PixArt-Sigma | [`PixArtTransformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✔️/✔️/✔️ |
| Image | Sana (image) | [`SanaTransformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 4 | ✔️/✖️/✔️/✔️ |
| Image | DiT | [`DiTTransformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✔️/✔️/✔️ |
| Image | HunyuanDiT | [`HunyuanDiT2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✔️/✔️/✔️ |
| Image | HunyuanDiT-PAG | [`HunyuanDiT2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✔️/✔️/✔️ |
| Image | AuraFlow | [`AuraFlowTransformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✖️/✔️/✔️ |
| Image | CogView4 | [`CogView4Transformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 2 | ✔️/✔️/✔️/✔️ |
| Image | CogView3Plus | [`CogView3PlusTransformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✔️/✔️/✔️ |
| Image | Hunyuan-Image | [`HunyuanImageTransformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 2 | ✔️/✔️/✔️/✔️ |
| Image | HiDream | [`HiDreamImageTransformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✖️/✔️/✔️ |
| Image | Bria | [`BriaTransformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✖️/✔️/✔️ |
| Image | Chroma | [`ChromaTransformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 3 | ✔️/✔️/✔️/✔️ |
| Image | PRX / PRXPixel | [`PRXTransformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 2 | ✔️/✔️/✔️/✔️ |
| Image | Anima | [`CosmosTransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✔️/✔️/✔️ |
| Image | OmniGen | [`OmniGenTransformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✖️/✔️/✔️ |
| Image | Lumina | [`LuminaNextDiT2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 2 | ✔️/✖️/✔️/✔️ |
| Image | Lumina 2 | [`Lumina2Transformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 2 | ✔️/✖️/✔️/✔️ |
| Image | VisualCloze | [`FluxTransformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py)  | 2 | ✔️/✔️/✔️/✔️ |
| Image | Amused | [`UVit2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✖️/✔️/✔️ |
| **Video** | CogVideoX | [`CogVideoXTransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 4 | ✔️/✔️/✔️/✔️ |
| Video | Wan (T2V/I2V) | [`WanTransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 3 | ✔️/✔️/✔️/✔️ |
| Video | Any-Flow | [`AnyFlowTransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✔️/✔️/✔️ |
| Video | Any-Flow-FAR | [`AnyFlowFARTransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✖️/✖️/✔️/✔️ |
| Video | Wan-VACE | [`WanVACETransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✔️/✔️/✔️ |
| Video | HunyuanVideo | [`HunyuanVideoTransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 4 | ✔️/✔️/✔️/✔️ |
| Video | HunyuanVideo 1.5 | [`HunyuanVideo15Transformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 2 | ✔️/✖️/✔️/✔️ |
| Video | Mochi | [`MochiTransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✔️/✔️/✔️ |
| Video | Allegro | [`AllegroTransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✖️/✔️/✔️ |
| Video | EasyAnimate | [`EasyAnimateTransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 3 | ✔️/✖️/✔️/✔️ |
| Video | ConsisID | [`ConsisIDTransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✔️/✔️/✔️ |
| Video | Cosmos | [`CosmosTransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 7+ | ✔️/✖️/✔️/✔️ |
| Video | Cosmos3 (Omni) | [`Cosmos3OmniTransformer`](src/cache_dit/caching/patch_functors/functor_cosmos3.py) | 2 | ✔️/✖️/✔️/✔️ |
| Video | LTX | [`LTXVideoTransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 5 | ✔️/✔️/✔️/✔️ |
| Video | LTX2 | [`LTX2VideoTransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 6 | ✔️/✔️/✔️/✔️ |
| Video | LTX2.5 (DFR/T2V/I2V) | [`LTX2VideoTransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 6 | ✔️/✔️/✔️/✔️ |
| Video | Echo (JoyAI-Echo) | [`LTX2VideoTransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✔️/✔️/✔️ |
| Video | Helios | [`HeliosTransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 2 | ✔️/✔️/✔️/✔️ |
| Video | ChronoEdit | [`ChronoEditTransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✔️/✔️/✔️ |
| Video | SkyReelsV2 | [`SkyReelsV2Transformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 5 | ✔️/✔️/✔️/✔️ |
| Video | Kandinsky5 (T2V/I2V) | [`Kandinsky5Transformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 4 | ✔️/✔️/✔️/✔️ |
| Video | MiniMax-H3 (T2VA) 🔥 | [`MiniMaxH3Transformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✔️/✔️/✔️ |
| Video | MiniMax-H3 (FL2VA) 🔥 | [`MiniMaxH3Transformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✔️/✔️/✔️ |  
| Video | MiniMax-H3 (Ref2VA) 🔥 | [`MiniMaxH3Transformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✔️/✔️/✔️ |  
| Video | Motif-Video (T2V/I2V) | [`MotifVideoTransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 2 | ✔️/✔️/✔️/✔️ |
| **Audio** | StableAudio | [`StableAudioDiTModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✖️/✔️/✔️ |
| **3D/Other** | Shap-E | [`PriorTransformer`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✖️/✔️/✔️ |
| Other | LucyEdit | [`WanTransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py)  | 1 | ✔️/✔️/✔️/✔️ |

<i> <b>C</b>: Hybrid Cache (DBCache + Calibrator: TaylorSeer/DMD/FoCa/SCM); <b>P</b>: Parallelism (Ulysses/Ring/USP/TP<br>TE-P/VAE-P/2D-P/3D-P); <b>Q</b>: Quantization (W8A8, W4A4); <b>OF</b>: Bucket-style Layerwise Offload (~0 overhead) </i>

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
- 🎉[(Intel) llm-scaler x Cache-DiT](https://github.com/intel/llm-scaler/tree/main/omni#cache-dit--torchcompile-acceleration)
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
