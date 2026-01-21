<div align="center">
  <p align="center">
    <h2 align="center">
        CacheDiT: A PyTorch-native and Flexible Inference Engine <br>with 🤗🎉 Hybrid Cache Acceleration and Parallelism for DiTs
    </h2>
  </p>
<img src=https://github.com/vipshop/cache-dit/raw/main/assets/speedup_v4.png>
</div>

# Overviews

Currently, **cache-dit** library supports almost **Any** Diffusion Transformers (with **Transformer Blocks** that match the specific Input and Output **patterns**). Please check [🎉Supported Matrix](../supported_matrix/NVIDIA_GPU.md) for more details.

- [📊Examples](https://github.com/vipshop/cache-dit/tree/main/examples) - The **easiest** way to enable **hybrid cache acceleration** and **parallelism** for DiTs with cache-dit is to start with our examples for popular models: FLUX, Z-Image, Qwen-Image, Wan, etc.
- [🌐HTTP Serving](./SERVING.md) - Deploy cache-dit models with HTTP API for **text-to-image**, **image editing**, **multi-image editing**, and **text-to-video** generation
- [❓FAQ](../FAQ.md) - Frequently asked questions including attention backend configuration, troubleshooting, and optimization tips
