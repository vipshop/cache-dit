# 🎉BlockAdapter 

TODO: more details docs for BlockAdapter.

## 🎉Unified Cache APIs

<div id="unified"></div>  

Currently, for any **Diffusion** models with **Transformer Blocks** that match the specific **Input/Output patterns**, we can use the **Unified Cache APIs** from **cache-dit**, namely, the `cache_dit.enable_cache(...)` API. The **Unified Cache APIs** are currently in the experimental phase; please stay tuned for updates. The supported patterns are listed as follows:

![](https://github.com/vipshop/cache-dit/raw/main/assets/patterns.png)

After the `cache_dit.enable_cache(...)` API is called, you just need to call the pipe as normal. The `pipe` param can be **any** Diffusion Pipeline. Please refer to [Qwen-Image](./examples/run_qwen_image_uapi.py) as an example. 
```python
import cache_dit
from diffusers import DiffusionPipeline 

# can be any diffusion pipeline
pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image")

# one line code with default cache options.
cache_dit.enable_cache(pipe) 

# or, enable cache with custom pattern settings.
from cache_dit import ForwardPattern, BlockAdapter
cache_dit.enable_cache(
    BlockAdapter(
        pipe=pipe,
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        blocks_name="transformer_blocks",
    ),
    forward_pattern=ForwardPattern.Pattern_1,
)

# just call the pipe as normal.
output = pipe(...)

# then, summary the cache stats.
stats = cache_dit.summary(pipe)
```

After finishing each inference of `pipe(...)`, you can call the `cache_dit.summary(...)` API on pipe to get the details of the cache stats for the current inference (markdown table format). You can set `details` param as `True` to show more details of cache stats.

```python
⚡️Cache Steps and Residual Diffs Statistics: QwenImagePipeline

| Cache Steps | Diffs P00 | Diffs P25 | Diffs P50 | Diffs P75 | Diffs P95 | Diffs Min | Diffs Max |
|-------------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| 23          | 0.045     | 0.084     | 0.114     | 0.147     | 0.241     | 0.045     | 0.297     |
...
```
