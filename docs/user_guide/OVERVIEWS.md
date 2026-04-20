<div align="center">
  <p align="center">
    <h2 align="center">
        ⚡️🎉A PyTorch-native Inference Engine with Cache, <br>Parallelism, Quantization for Diffusion Transformers
    </h2>
  </p>
<img src=https://github.com/vipshop/cache-dit/raw/main/assets/speedup_v4.png>
</div>

# Overviews

**🤗Why Cache-DiT❓❓**Cache-DiT is built on top of the 🤗[Diffusers](https://github.com/huggingface/diffusers) library and now supports nearly [ALL](https://cache-dit.readthedocs.io/en/latest/supported_matrix/NVIDIA_GPU/) DiTs from Diffusers. It provides [hybrid cache acceleration](https://cache-dit.readthedocs.io/en/latest/user_guide/CACHE_API/) (DBCache, TaylorSeer, SCM, etc.) and comprehensive [parallelism](https://cache-dit.readthedocs.io/en/latest/user_guide/CONTEXT_PARALLEL/) optimizations, including Context Parallelism, Tensor Parallelism, hybrid 2D or 3D parallelism, and dedicated extra parallelism support for Text Encoder, VAE, and ControlNet.  

<div align="center">
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/arch_v2.png width=815px>
</div>

Cache-DiT is compatible with compilation, CPU Offloading, and quantization, fully integrates with [SGLang Diffusion](https://docs.sglang.io/diffusion/performance/cache/cache_dit.html), [vLLM-Omni](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion/cache_dit_acceleration/), ComfyUI, and runs natively on NVIDIA GPUs, Ascend NPUs and AMD GPUs. Cache-DiT is **fast**, **easy to use**, and **flexible** for various DiTs (online docs at 📘[readthedocs.io](https://cache-dit.readthedocs.io/en/latest/)). Please check [🎉Supported Matrix](../supported_matrix/NVIDIA_GPU.md) for more details.

<div align="center">
  <img src=https://github.com/vipshop/cache-dit/raw/main/assets/speedup_v5.png width=800px>
</div>

[📊Examples](https://github.com/vipshop/cache-dit/tree/main/examples) - The **easiest** way to enable **hybrid cache acceleration** and **parallelism** for DiTs with cache-dit is to start with our examples for popular models: FLUX, Z-Image, Qwen-Image, Wan, etc. [❓FAQ](../FAQ.md) - Frequently asked questions including attention backend configuration, troubleshooting, and optimization tips

## Table of contents

- [Overviews](./OVERVIEWS.md)
- [Installation](./INSTALL.md)
- [Quick Examples](../EXAMPLES.md)
- [Unified Cache APIs](./CACHE_API.md)
- [DBCache Design](./DBCACHE_DESIGN.md)
- [Context Parallelism](./CONTEXT_PARALLEL.md)
- [Tensor Parallelism](./TENSOR_PARALLEL.md)
- [TE-P, VAE-P and CN-P](./EXTRA_PARALLEL.md)
- [2D and 3D Parallelism](./HYBRID_PARALLEL.md)
- [Low-Bits Quantization](./QUANTIZATION.md)
- [Attention Backends](./ATTENTION.md)
- [Use Torch Compile](./COMPILE.md)
- [Use CUDA Graphs](./CUDA_GRAPH.md)
- [Layerwise Offload](./OFFLOAD.md)
- [Ascend NPU Support](./ASCEND_NPU.md)
- [AMD GPU Support](./AMD_GPU.md)
- [Config with YAML](./LOAD_CONFIGS.md)
- [Environment Variables](./ENV.md)
- [Serving Deployment](./SERVING.md)
- [Metrics Tools](./METRICS.md)
- [Profiler Usage](./PROFILER.md)
- [API Documentation](./API_DOCS.md)
- [Supported Matrix](../supported_matrix/NVIDIA_GPU.md)
- [Benchmark](../benchmark/HYBRID_CACHE.md)
- [Developer Guide](../developer_guide/PRE_COMMIT.md)
- [Community Integration](../COMMUNITY.md)
- [FAQ](../FAQ.md)
