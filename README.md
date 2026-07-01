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

Cache-DiT is compatible with compilation, CPU Offloading, and quantization, fully integrates with [SGLang Diffusion](https://docs.sglang.io/diffusion/performance/cache/cache_dit.html), [vLLM-Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion/cache_acceleration/cache_dit/), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/pull/12548), ComfyUI, and runs natively on NVIDIA GPUs, Ascend NPUs and AMD GPUs. Cache-DiT is **fast**, **easy to use**, and **flexible** for various DiTs (online docs at 📘[cache-dit.io](https://cache-dit.readthedocs.io/en/latest/)). Cache-DiT's technical report is available at 🎉[Cache-DiT: A Unified PyTorch-Native Inference Engine for Diffusion Transformers.](docs/papers/cache-dit-tech-report.pdf)

<div align="center">
  <p align="center">
    <i>⚡️9x speedup by Cache-DiT with Cache, Context Parallelism and Compilation</i>
  </p>
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/speedup_v5.png width=800px>
</div>

## 📋Supported DiT Models

Cache-DiT supports **40+ DiT pipeline families (121+ Variants)** from 🤗Diffusers, covering the vast majority of DiT-based pipelines. For full support matrix and detailed usage, please refer to our documentation at 📘[cache-dit.io](https://cache-dit.readthedocs.io/en/latest/).

<div align="center">

| Modality | Pipeline Series | Transformer | Variants | C/P/Q/OF |
|:----------|:----------------|:-------------|:----------|:-------|
| **Image** | FLUX | [`FluxTransformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 10+ | ✔️/✔️/✔️/✔️ |
| Image | FLUX.2 | [`Flux2Transformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✔️/✔️/✔️ |
| Image | FLUX.2 Klein | [`Flux2Transformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 3 | ✔️/✔️/✔️/✔️ |
| Image | ERNIE-Image | [`ErnieImageTransformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✔️/✔️/✔️ |
| Image | Qwen-Image | [`QwenImageTransformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 9+ | ✔️/✔️/✔️/✔️ |
| Image | Z-Image | [`ZImageTransformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 6 | ✔️/✔️/✔️/✔️ |
| Image | Boogu-Image| [`BooguImageTransformer`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✔️/✔️/✔️ |
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
| Image | PRX | [`PRXTransformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 2 | ✔️/✖️/✔️/✔️ |
| Image | OmniGen | [`OmniGenTransformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✖️/✔️/✔️ |
| Image | Lumina | [`LuminaNextDiT2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 2 | ✔️/✖️/✔️/✔️ |
| Image | Lumina 2 | [`Lumina2Transformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 2 | ✔️/✖️/✔️/✔️ |
| Image | VisualCloze | [`FluxTransformer2DModel`](src/cache_dit/caching/block_adapters/adapters.py)  | 2 | ✔️/✔️/✔️/✔️ |
| Image | Amused | [`UVit2DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✖️/✔️/✔️ |
| **Video** | CogVideoX | [`CogVideoXTransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 4 | ✔️/✔️/✔️/✔️ |
| Video | Wan (T2V/I2V) | [`WanTransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 3 | ✔️/✔️/✔️/✔️ |
| Video | Wan-VACE | [`WanVACETransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✔️/✔️/✔️ |
| Video | HunyuanVideo | [`HunyuanVideoTransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 4 | ✔️/✔️/✔️/✔️ |
| Video | HunyuanVideo 1.5 | [`HunyuanVideo15Transformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 2 | ✔️/✖️/✔️/✔️ |
| Video | Mochi | [`MochiTransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✔️/✔️/✔️ |
| Video | Allegro | [`AllegroTransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✖️/✔️/✔️ |
| Video | EasyAnimate | [`EasyAnimateTransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 3 | ✔️/✖️/✔️/✔️ |
| Video | ConsisID | [`ConsisIDTransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✔️/✔️/✔️ |
| Video | Cosmos | [`CosmosTransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 7+ | ✔️/✖️/✔️/✔️ |
| Video | LTX | [`LTXVideoTransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 5 | ✔️/✔️/✔️/✔️ |
| Video | LTX2 | [`LTX2VideoTransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 6 | ✔️/✔️/✔️/✔️ |
| Video | Helios | [`HeliosTransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 2 | ✔️/✔️/✔️/✔️ |
| Video | ChronoEdit | [`ChronoEditTransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✔️/✔️/✔️ |
| Video | SkyReelsV2 | [`SkyReelsV2Transformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 5 | ✔️/✔️/✔️/✔️ |
| Video | Kandinsky5 (T2V/I2V) | [`Kandinsky5Transformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py) | 4 | ✔️/✔️/✔️/✔️ |
| **Audio** | StableAudio | [`StableAudioDiTModel`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✖️/✔️/✔️ |
| **3D/Other** | Shap-E | [`PriorTransformer`](src/cache_dit/caching/block_adapters/adapters.py) | 1 | ✔️/✖️/✔️/✔️ |
| Other | LucyEdit | [`WanTransformer3DModel`](src/cache_dit/caching/block_adapters/adapters.py)  | 1 | ✔️/✔️/✔️/✔️ |

<i> <b>C</b>: Hybrid Cache (DBCache + Calibrator: TaylorSeer/DMD/FoCa/SCM); <b>P</b>: Parallelism (Ulysses/Ring/USP/TP<br>TE-P/VAE-P/2D-P/3D-P); <b>Q</b>: Quantization (W8A8, W4A4); <b>OF</b>: Bucket-style Layerwise Offload (~0 overhead) </i>

</div>

## 🤖Agentic Workflows  

Cache-DiT provides a [cache-dit-model-integration](./.copilot/skills/cache-dit-model-integration/SKILL.md) SKILL to help users integrate new DiT pipelines into Cache-DiT, including **Cache, CP, TP, TE-P, VAE-P** and carefully designed test cases. Users can use it via Coding Agents, e.g, [GitHub Copilot](https://docs.github.com/en/copilot), [Claude Code](https://claude.ai), [Open Code](https://opencode.ai/). 

<div align="center">

<img src=docs/assets/agent.png width=800px>

</div>

> [!NOTE]
> Please note that quantization and layerwise offload in Cache-DiT are generally supported for **nn.Module**, thus no extra integration is needed for new DiT pipelines or transformers. 

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
  title={Cache-DiT: A PyTorch-native Inference Engine with Cache, Parallelism, Quantization and CPU Offload for DiTs.},
  url={https://github.com/vipshop/cache-dit.git},
  note={Open-source software available at https://github.com/vipshop/cache-dit.git},
  author={DefTruth, vipshop.com, etc.},
  year={2025}
}
```
