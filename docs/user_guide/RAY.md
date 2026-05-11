# Ray Wrapper

<div id="ray-wrapper"></div>

The Ray Wrapper lets Cache-DiT create and manage the distributed worker processes for you. After enabling it, user code can still look like normal single-process Diffusers code: load a pipeline, call <span style="color:#c77dff;">cache_dit.enable_cache(...)</span> with <span style="color:#c77dff;">use_ray=True</span>, then call the pipeline as usual.

![alt text](../assets/ray_wrapper.png)

This means you do not need to write manual distributed inference code. In the common case, you do not need `torchrun`, `dist.init_process_group`, rank/world-size branching, per-rank device placement, or explicit model sharding code. Cache-DiT starts Ray actors, places workers on GPUs, initializes the worker process group, transfers the model snapshot, applies Cache-DiT parallelism, and proxies calls back through the original pipeline object. 

|Baseline|Ray Wrapper with TP=2 + Compile|
|:---:|:---:|
|47.41s|24.57s|
|![](../assets/ray_baseline.png)|![](../assets/ray_tp2.png)|

## Installation

```bash
# Recommended: torch>=2.10, diffusers>=0.38.0, CUDA>=12.9
# Option 1: Install the latest stable release from PyPI (>= 1.3.6)
pip3 install -U "cache-dit[ray,parallelism]" -i https://pypi.org/simple
# Option 2: Install the latest develop version from GitHub
pip3 install git+https://github.com/vipshop/cache-dit.git
```
Please refer to the [Installation](./INSTALL.md) docs for more details and installation options.

## Pipeline Wrapper

We **recommend** the pipeline-level wrapper for the widest support of Cache-DiT features (cache, quantization, parallelism) and the simplest user experience. With it, Ray handles the entire pipeline execution, including text encoders, VAE, scheduler, and all other components.

```python
import cache_dit
from diffusers import Flux2KleinPipeline
from cache_dit import ParallelismConfig

# Just let it load on CPU; Cache-DiT will handle GPU 
# transfer inside the Ray workers.
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
```

The code above is still a normal single-process script. Run it with `python your_script.py`; Cache-DiT and Ray handle the distributed execution internally.

## Transformer Wrapper

You can also wrap only the transformer module. This is useful when you want the text encoders, VAE, scheduler, and other pipeline components to stay in the main process while only the transformer is executed by Ray workers.

The transformer-level wrapper is more generic but may **slightly slower** than the pipeline-level wrapper due to more frequent **main-process <-> worker communication during denoising**. It also does not support some Cache-DiT features such as cache hooks and quantization, which are applied inside the Ray workers and thus only work with the pipeline-level wrapper.

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
image = pipe(prompt="A cat holding a sign that says hello world").images[0]
image.save("ray_transformer_wrapper.png")
```

When the transformer-level wrapper is enabled, Cache-DiT patches the Ray-owned transformer so `pipe.to("cuda")` does not move the main-process transformer copy back onto the GPU. The executable transformer copies live inside the Ray workers.

## How to use LoRAs 

For the best performance and full parallelism features support, we recommend fusing LoRAs into the base model before enabling the Ray wrapper. With unfused LoRAs, the Ray wrapper can still be enabled but only works with CP (Ulysses and Ring), not TP.

```python
# First: fuse the LoRA into the base model. 
pipe.fuse_lora()
# Then: enable the Ray wrapper with TP=2.
cache_dit.enable_cache(
  pipe,
  parallelism_config=ParallelismConfig(
    tp_size=2,
    use_ray=True,
  ),
)
```

## Distributed Inference

Set the normal Cache-DiT parallelism fields and add <span style="color:#c77dff;">use_ray=True</span>. Please note that Tensor Parallelism (TP) will not work with unfused LoRAs due to the way TP shards the model, so we recommend fusing LoRAs into the base model before enabling the Ray wrapper for full parallelism features support. With unfused LoRAs, the Ray wrapper can still be enabled but only works with CP (Ulysses and Ring), not TP.

```python
ParallelismConfig(tp_size=2, use_ray=True)
ParallelismConfig(ulysses_size=2, use_ray=True)
ParallelismConfig(ring_size=2, use_ray=True)
```

## Torch Compile

Ray workers can compile the transformer after loading and applying Cache-DiT parallelism:

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

If the transformer provides `compile_repeated_blocks()`, Cache-DiT calls that method first. Otherwise it falls back to `transformer.compile()` when available.

## Cache and Quantization

When <span style="color:#c77dff;">use_ray=True</span>, cache hooks and quantization are applied inside the Ray workers after the model snapshot is loaded. This preserves the same user-facing `enable_cache` API while avoiding main-process hooks or quantized modules being lost during model transfer.

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

## Ray Runtime Env

When your application code (custom pipeline classes, helper modules) lives in a directory that is **not** part of the default Python path, Ray workers cannot import those modules.  This causes ``ModuleNotFoundError`` during worker-side model loading or pickle deserialization.

Cache-DiT exposes `ParallelismConfig.ray_runtime_env` to pass a Ray `runtime_env` dictionary to `ray.init()`.  The most common and **lightweight** pattern is to use `env_vars` to set `PYTHONPATH`:

```python
cache_dit.enable_cache(
  pipe,
  parallelism_config=ParallelismConfig(
    ulysses_size=2,
    use_ray=True,
    ray_runtime_env={
      # Single or multiple paths can be added, separated by ":" or " "
      "env_vars": {
        "PYTHONPATH": "/path/to/your/app/src",
        # "PYTHONPATH": "xxx/path1:xxx/path2:xxx/path3",
      }
    },
  ),
)
```

### What is PYTHONPATH?

`PYTHONPATH` is a Python environment variable that specifies additional directories where Python searches for modules.  When Python starts, it appends each `:`-separated path in `PYTHONPATH` to `sys.path`, before the standard-library and site-packages entries.  Multiple paths are supported:

```python
"PYTHONPATH": "/app/src:/app/libs:/shared/utils"
```

Ray workers receive `env_vars` entries as process environment variables — **nothing is packaged, zipped, or uploaded**.  Workers resolve `import your_module` directly from the filesystem path. `env_vars` vs `working_dir` / `py_modules` listed as below:

| field | behavior | large files (>2 GB) |
|-------|----------|---------------------|
| `env_vars.PYTHONPATH` | sets an environment variable; **zero packaging** | works |
| `working_dir` | zips the entire directory, uploads to workers | **fails** — Ray packaging times out or errors |
| `py_modules` | packages the module tree, uploads to workers | **fails** — same packaging issue |

> **Rule of thumb:** when your module tree contains model weights (e.g. a 54 GB
> `FLUX.1-dev` checkpoint), always use `env_vars.PYTHONPATH`. The files must be
> accessible from every worker node via a shared filesystem (NFS, Lustre, or
> — on a single-node cluster — the local disk).

### Pre-initialized Ray

If you already called `ray.init()` before `cache_dit.enable_cache()`, Cache-DiT will **not** re-initialize Ray and therefore cannot inject the `runtime_env` from `ParallelismConfig`.  In that case, configure `runtime_env` in your own `ray.init()` call:

```python
ray.init(runtime_env={"env_vars": {"PYTHONPATH": "/path/to/your/app/src"}})
cache_dit.enable_cache(
    pipe,
    parallelism_config=ParallelismConfig(ulysses_size=2, use_ray=True),
)
```

Cache-DiT emits a warning when it detects this situation and the pipeline class is from a user-application module.

## Quick Start

A complete runnable example is available at `examples/ray/ray_wrapper_example.py`. For example:

```bash
# The example is located at cache-dit/examples/ray/ray_wrapper_example.py
git clone https://github.com/vipshop/cache-dit.git && cd cache-dit

# Baseline
python3 examples/ray/ray_wrapper_example.py \
  --model-path $FLUX_2_KLEIN_BASE_9B_DIR \
  --save-path ./tmp/baseline.png

# Ray wrapper with TP=2 and compile enabled
python3 examples/ray/ray_wrapper_example.py \
  --model-path $FLUX_2_KLEIN_BASE_9B_DIR \
  --tp 2 \
  --compile \
  --save-path ./tmp/ray.png
```
