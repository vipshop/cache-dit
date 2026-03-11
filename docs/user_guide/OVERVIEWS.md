<div align="center">
  <p align="center">
    <h2 align="center">
        Cache-DiT: A PyTorch-native and Flexible Inference Engine <br>with 🤗🎉 Hybrid Cache Acceleration and Parallelism for DiTs
    </h2>
  </p>
<img src=https://github.com/vipshop/cache-dit/raw/main/assets/speedup_v4.png>
</div>

# Overviews

Currently, **Cache-DiT** library supports almost **Any** Diffusion Transformers (with **Transformer Blocks** that match the specific Input and Output **patterns**). Please check [🎉Supported Matrix](../supported_matrix/NVIDIA_GPU.md) for more details.

- [📊Examples](https://github.com/vipshop/cache-dit/tree/main/examples) - The **easiest** way to enable **hybrid cache acceleration** and **parallelism** for DiTs with cache-dit is to start with our examples for popular models: FLUX, Z-Image, Qwen-Image, Wan, etc.
- [❓FAQ](../FAQ.md) - Frequently asked questions including attention backend configuration, troubleshooting, and optimization tips

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
- [Ascend NPU Support](./ASCEND_NPU.md)
- [AMD GPU Support](./AMD_GPU.md)
- [Config with YAML](./LOAD_CONFIGS.md)
- [Environment Variables](./ENV.md)
- [Serving Deployment](./SERVING.md)
- [Metrics Tools](./METRICS.md)
- [Profiler Usage](./PROFILER.md)
- [API Docmentation](./API_DOCS.md)
- [Supported Matrix](../supported_matrix/NVIDIA_GPU.md)
- [Benchmark](../benchmark/HYBRID_CACHE.md)
- [Developer Guide](../developer_guide/PRE_COMMIT.md)
- [Community Integration](../COMMUNITY.md)
- [FAQ](../FAQ.md)
