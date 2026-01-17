# Tensor Parallelism

## Hybrid Tensor Parallelism

<div id="tensor-parallelism"></div>

cache-dit is also compatible with tensor parallelism. Currently, we support the use of `Hybrid Cache` + `Tensor Parallelism` scheme (via NATIVE_PYTORCH parallelism backend) in cache-dit. Users can use Tensor Parallelism to further accelerate the speed of inference and **reduce the VRAM usage per GPU**! For more details, please refer to [📚examples/parallelism](https://github.com/vipshop/cache-dit/tree/main/examples). Now, cache-dit supported tensor parallelism for [FLUX.1](https://huggingface.co/black-forest-labs/FLUX.1-dev), 🔥[FLUX.2](https://huggingface.co/black-forest-labs/FLUX.2-dev), [Qwen-Image](https://github.com/QwenLM/Qwen-Image), [Qwen-Image-Lightning](https://github.com/ModelTC/Qwen-Image-Lightning), [Wan2.1](https://github.com/Wan-Video/Wan2.1), [Wan2.2](https://github.com/Wan-Video/Wan2.2), [HunyuanImage-2.1](https://huggingface.co/tencent/HunyuanImage-2.1), [HunyuanVideo](https://huggingface.co/hunyuanvideo-community/HunyuanVideo) and [VisualCloze](https://github.com/lzyhha/VisualCloze), etc. cache-dit will support more models in the future.

```python
# pip3 install "cache-dit[parallelism]"
from cache_dit import ParallelismConfig

cache_dit.enable_cache(
    pipe_or_adapter, 
    cache_config=DBCacheConfig(...),
    # Set tp_size > 1 to enable tensor parallelism.
    parallelism_config=ParallelismConfig(tp_size=2),
)
# torchrun --nproc_per_node=2 parallel_cache.py
```

Please note that in the short term, we have no plans to support Hybrid Parallelism. Please choose to use either Context Parallelism or Tensor Parallelism based on your actual scenario.
