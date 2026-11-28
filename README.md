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

<p align="justify"><strong>Cache-DiT</strong> is fast, easy to use, and flexible for various DiTs. It is a PyTorch-native inference engine with <a href="https://cache-dit.readthedocs.io/en/latest/user_guide/CACHE_API/">cache</a>, <a href="https://cache-dit.readthedocs.io/en/latest/user_guide/QUANTIZATION/">quantization</a>, <a href="https://cache-dit.readthedocs.io/en/latest/user_guide/OFFLOAD/">offload</a>, <a href="https://cache-dit.readthedocs.io/en/latest/user_guide/CONTEXT_PARALLEL/">parallelism</a> and <a href="https://cache-dit.readthedocs.io/en/latest/user_guide/ATTENTION/">attention backends</a> (FA-2/3, <a href="https://github.com/vipshop/ffpa-attn">ffpa-attn</a>, Sage-2/3), and runs natively on NVIDIA GPUs, Huawei Ascend NPUs and AMD GPUs.</p>

<div align="center">
  <img src=docs/assets/arch_v3.png >
</div>

Cache-DiT is compatible with torch.compile and integrates with [SGLang Diffusion](https://docs.sglang.io/diffusion/performance/cache/cache_dit.html), [vLLM-Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion/cache_acceleration/cache_dit/), [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/models/visual-generation.html#cache-dit) and [xDiT](https://github.com/xdit-project/xDiT/tree/main/xfuser/model_executor/cache/adapters/cache_dit). It covers **60+🎉 DiT pipeline families (150+🎉 Variants)**; see 📘[cache-dit.io](https://cache-dit.readthedocs.io/en/latest/) for the full support matrix.

<div align="center">
  <img src=docs/assets/parallel_speedup_flux.png >
</div>


## 🚀Quick Start

```bash
pip install -U "cache-dit[quantization]" # quantization: torchao/bitsandbytes, etc.
# cu13 wheel: CUDA extensions (SVDQuant-W{2,3,4,6,8}-A{4,6,8}, FP8/FP6/FP4 communication, etc).
pip install -U "cache-dit[quantization]" --extra-index-url https://vipshop.github.io/cache-dit/whl/cu13/
```

```bash
# FP6 Ulysses (CP4) + Cache + FP8 GEMM + FP8 Attn + Compile (PRO 5000: 16.7s -> ⚡️1.57s)
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m cache_dit.generate black-forest-labs/FLUX.1-dev \
  --prompt "A cat holding a sign that says hello world" \
  --parallel-config 'ulysses=4' \
  --quant-communicate-config 'bits=6' \
  --cache-config 'DBCache,rdt=0.15' \
  --calibrator-config 'taylorseer,order=1' \
  --quantize-config 'float8_per_row' \
  --attn-config 'FFPA,bits=8' \
  --compile
```

More commands: 📘[Quick Examples](https://cache-dit.readthedocs.io/en/latest/EXAMPLES/). Other
platforms, the source build and the full install matrix:
📘[Installation](https://cache-dit.readthedocs.io/en/latest/user_guide/INSTALL/).

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
