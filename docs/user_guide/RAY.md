# Ray Wrapper

<div id="ray-wrapper"></div>

The Ray Wrapper lets cache-dit create and manage the distributed worker processes for you. After enabling it, user code can still look like normal single-process Diffusers code: load a pipeline, call `cache_dit.enable_cache(...)`, then call the pipeline as usual.

![alt text](../assets/ray_wrapper.png)

This means you do not need to write manual distributed inference code. In the common case, you do not need `torchrun`, `dist.init_process_group`, rank/world-size branching, per-rank device placement, or explicit model sharding code. cache-dit starts Ray actors, places workers on GPUs, initializes the worker process group, transfers the model snapshot, applies cache-dit parallelism, and proxies calls back through the original pipeline object. 

|Baseline|Ray Wrapper with Ulysses 2 + Compile|
|:---:|:---:|
|47.41s|23.86s|
|![](../assets/ray_baseline.png)|![](../assets/ray_ulysses2.png)|

## Minimal Example

```python
import torch
from diffusers import Flux2KleinPipeline

import cache_dit
from cache_dit import ParallelismConfig

pipe = Flux2KleinPipeline.from_pretrained(
  "/path/to/FLUX.2-klein-base-9B",
  torch_dtype=torch.bfloat16,
)

# NOTE: Will auto transfer to cuda inside by ray wrapper for 
# pipeline-level parallelism, so we keep the original pipeline 
# on CPU to avoid redundant GPU memory usage.
cache_dit.enable_cache(
  pipe,
  parallelism_config=ParallelismConfig(
    tp_size=2,
    use_ray=True,
  ),
)

# Call the pipeline as usual; No code changes are needed for
# Ray parallelism to work.
image = pipe(
  prompt="A cat holding a sign that says hello world",
  height=1024,
  width=1024,
  num_inference_steps=28,
).images[0]

image.save("ray_wrapper.png")
cache_dit.disable_cache(pipe)
```

The code above is still a normal single-process script. Run it with `python your_script.py`; cache-dit and Ray handle the distributed execution internally.

## Transformer-Level Wrapper

You can also wrap only the transformer module. This is useful when you want the text encoders, VAE, scheduler, and other pipeline components to stay in the main process while only the transformer is executed by Ray workers.

```python
cache_dit.enable_cache(
  pipe.transformer,
  parallelism_config=ParallelismConfig(
    ulysses_size=2,
    use_ray=True,
  ),
)

# NOTE: Only the transformer is parallelized and transferred to GPU, 
# so we need to move the pipeline to GPU as well for the forward pass.
pipe.to("cuda")
image = pipe(prompt="A cinematic mountain lake at sunrise").images[0]
cache_dit.disable_cache(pipe.transformer)
```

When the transformer-level wrapper is enabled, cache-dit patches the Ray-owned transformer so `pipe.to("cuda")` does not move the main-process transformer copy back onto the GPU. The executable transformer copies live inside the Ray workers.

## Tensor Parallelism and Context Parallelism

Set the normal cache-dit parallelism fields and add `use_ray=True`:

```python
ParallelismConfig(tp_size=2, use_ray=True)
ParallelismConfig(ulysses_size=2, use_ray=True)
ParallelismConfig(ring_size=2, use_ray=True)
```

Use the explicit field names `tp_size`, `ulysses_size`, and `ring_size`. Short aliases such as `tp`, `ulysses`, and `ring` are intentionally not supported.

## Optional Compile

Ray workers can compile the transformer after loading and applying cache-dit parallelism:

```python
cache_dit.enable_cache(
  pipe,
  parallelism_config=ParallelismConfig(
    tp_size=2,
    use_ray=True,
    ray_use_compile=True,
  ),
)
```

If the transformer provides `compile_repeated_blocks()`, cache-dit calls that method first. Otherwise it falls back to `transformer.compile()` when available.

## Cache and Quantization

When `use_ray=True`, cache hooks and quantization are applied inside the Ray workers after the model snapshot is loaded. This preserves the same user-facing `enable_cache` API while avoiding main-process hooks or quantized modules being lost during model transfer.

```python
from cache_dit import DBCacheConfig
from cache_dit import ParallelismConfig
from cache_dit import QuantizeConfig

cache_dit.enable_cache(
  pipe,
  cache_config=DBCacheConfig(...),
  parallelism_config=ParallelismConfig(
    tp_size=2,
    use_ray=True,
  ),
  quantize_config=QuantizeConfig(...),
)
```

## Quick Start

A complete runnable example is available at `examples/ray/ray_wrapper_example.py`.

For example:

```bash
# Baseline
python3 examples/ray/ray_wrapper_example.py \
  --model-path $FLUX_2_KLEIN_BASE_9B_DIR \
  --save-path ./tmp/baseline.png

# Ray wrapper with Ulysses=2 and compile enabled
python3 examples/ray/ray_wrapper_example.py \
  --model-path $FLUX_2_KLEIN_BASE_9B_DIR \
  --ulysses 2 \
  --compile \
  --warmup 1 \
  --repeat 3 \
  --save-path ./tmp/ray.png
```
