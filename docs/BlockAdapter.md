# 🎉BlockAdapter 

TODO: more details docs for BlockAdapter.

### 📚Forward Pattern Matching 

Currently, for any **Diffusion** models with **Transformer Blocks** that match the specific **Input/Output patterns**, we can use the **Unified Cache APIs** from **cache-dit**, namely, the `cache_dit.enable_cache(...)` API. The **Unified Cache APIs** are currently in the experimental phase; please stay tuned for updates. The supported patterns are listed as follows:

![](https://github.com/vipshop/cache-dit/raw/main/assets/patterns.png)

### ⚡️Cache Acceleration with One-line Code

In most cases, you only need to call **one-line** of code, that is `cache_dit.enable_cache(...)`. After this API is called, you just need to call the pipe as normal. The `pipe` param can be **any** Diffusion Pipeline. Please refer to [Qwen-Image](https://github.com/vipshop/cache-dit/raw/main/examples/run_qwen_image.py) as an example. 
```python
import cache_dit
from diffusers import DiffusionPipeline 

# Can be any diffusion pipeline
pipe = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image")

# One-line code with default cache options.
cache_dit.enable_cache(pipe) 

# Just call the pipe as normal.
output = pipe(...)
```

### 🔥Automatic Block Adapter: Cache Acceleration for Custom Diffusion Models

But in some cases, you may have a **modified** Diffusion Pipeline or Transformer that is not located in the diffusers library or not officially supported by **cache-dit** at this time. The **BlockAdapter** can help you solve this problems. Please refer to [Qwen-Image w/ BlockAdapter](https://github.com/vipshop/cache-dit/raw/main/examples/run_qwen_image_adapter.py) as an example.

```python
from cache_dit import ForwardPattern, BlockAdapter

# Use BlockAdapter with `auto` mode.
cache_dit.enable_cache(
    BlockAdapter(pipe=pipe, auto=True), # Qwen-Image, etc.  
    # Check `📚Forward Pattern Matching` documentation and hack the code of
    # of Qwen-Image, you will find that it has satisfied `FORWARD_PATTERN_1`.
    forward_pattern=ForwardPattern.Pattern_1,  
)

# Or, manualy setup transformer configurations.
cache_dit.enable_cache(
    BlockAdapter(
        pipe=pipe, # Qwen-Image, etc.
        transformer=pipe.transformer,
        blocks=pipe.transformer.transformer_blocks,
        blocks_name="transformer_blocks",
    ), 
    forward_pattern=ForwardPattern.Pattern_1,
)
```
For such situations, **BlockAdapter** can help you quickly apply various cache acceleration features to your own Diffusion Pipelines and Transformers. Please check the [📚BlockAdapter.md](https://github.com/vipshop/cache-dit/raw/main/docs/BlockAdapter.md) for more details.
