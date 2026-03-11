# Use Yaml Config File

Cache-DiT now supported load the acceleration configs from a custom yaml file. Here are some examples.

## Single GPU inference  

Define a cache only config yaml `cache.yaml` file that contains:

### DBCache + TaylorSeer

```yaml
cache_config:
  max_warmup_steps: 8 
  warmup_interval: 2  
  max_cached_steps: -1
  max_continuous_cached_steps: 2  
  Fn_compute_blocks: 1
  Bn_compute_blocks: 0
  residual_diff_threshold: 0.12
  enable_taylorseer: true
  taylorseer_order: 1
```
Then, apply the acceleration config from yaml.

```python
>>> import cache_dit
>>> cache_dit.enable_cache(pipe, **cache_dit.load_configs("cache.yaml"))
```

### DBCache + TaylorSeer + SCM (Step Computation Mask)

```yaml
cache_config:
  max_warmup_steps: 8 
  warmup_interval: 2  
  max_cached_steps: -1
  max_continuous_cached_steps: 2  
  Fn_compute_blocks: 1
  Bn_compute_blocks: 0
  residual_diff_threshold: 0.12
  enable_taylorseer: true
  taylorseer_order: 1
  # Must set the num_inference_steps for SCM. The SCM will automatically 
  # generate the steps computation mask based on the num_inference_steps.
  num_inference_steps: 28 
  steps_computation_mask: fast
```

### DBCache + TaylorSeer + SCM (Step Computation Mask) + Cache CFG

```yaml
cache_config:
  max_warmup_steps: 8 
  warmup_interval: 2  
  max_cached_steps: -1
  max_continuous_cached_steps: 2  
  Fn_compute_blocks: 1
  Bn_compute_blocks: 0
  residual_diff_threshold: 0.12
  enable_taylorseer: true
  taylorseer_order: 1
  num_inference_steps: 28
  steps_computation_mask: fast
  enable_sperate_cfg: true # e.g, Qwen-Image, Wan, Chroma, Ovis-Image, etc.
```

## Distributed inference  

### 1D Parallelism

Define a parallelism only config yaml `parallel.yaml` file that contains:

```yaml
parallelism_config:
  ulysses_size: auto
  attention_backend: native
```
Then, apply the distributed inference acceleration config from yaml. `ulysses_size: auto` means that cache-dit will auto detect the `world_size` as the ulysses_size. Otherwise, you should manually set it as specific int number, e.g, 4.
```python
>>> import cache_dit
>>> cache_dit.enable_cache(pipe, **cache_dit.load_configs("parallel.yaml"))
```

### 2D Parallelism

You can also define a 2D parallelism config yaml `parallel_2d.yaml` file that contains:

```yaml
parallelism_config:
  ulysses_size: auto
  tp_size: 2
  attention_backend: native
```
Then, apply the 2D parallelism config from yaml. Here `tp_size: 2` means using tensor parallelism with size 2. The `ulysses_size: auto` means that cache-dit will auto detect the `world_size // tp_size` as the ulysses_size.
```python
>>> import cache_dit
>>> cache_dit.enable_cache(pipe, **cache_dit.load_configs("parallel_2d.yaml"))
```

### 3D Parallelism

You can also define a 3D parallelism config yaml `parallel_3d.yaml` file that contains:

```yaml
parallelism_config:
  ulysses_size: 2
  ring_size: 2
  tp_size: 2
  attention_backend: native
```
Then, apply the 3D parallelism config from yaml. Here `ulysses_size: 2`, `ring_size: 2`, `tp_size: 2` means using ulysses parallelism with size 2, ring parallelism with size 2 and tensor parallelism with size 2.
```python
>>> import cache_dit
>>> cache_dit.enable_cache(pipe, **cache_dit.load_configs("parallel_3d.yaml"))
```

### Ulysses Anything Attention

To enable Ulysses Anything Attention, you can define a parallelism config yaml `parallel_uaa.yaml` file that contains:

```yaml
parallelism_config:
  ulysses_size: auto
  attention_backend: native
  ulysses_anything: true
```
Then, apply the config from yaml. Here `ulysses_anything: true` means enabling Ulysses Anything Attention.
```python
>>> import cache_dit
>>> cache_dit.enable_cache(pipe, **cache_dit.load_configs("parallel_uaa.yaml"))
```

### Ulysses FP8 Communcation

For device that don't have NVLink support, you can enable Ulysses FP8 Communication to further reduce the communication overhead. You can define a parallelism config yaml `parallel_fp8.yaml` file that contains:

```yaml
parallelism_config:
  ulysses_size: auto
  attention_backend: native
  ulysses_float8: true
```

### Async Ulysses CP 

You can also enable async ulysses CP to overlap the communication and computation. Define a parallelism config yaml `parallel_async.yaml` file that contains:

```yaml
parallelism_config:
  ulysses_size: auto
  attention_backend: native
  ulysses_async: true
```
Then, apply the config from yaml. Here `ulysses_async: true` means enabling async ulysses CP.

### TE-P and VAE-P

You can also specify the extra parallel modules in the yaml config. For example, define a parallelism config yaml `parallel_extra.yaml` file that contains:

```yaml
parallelism_config:
  ulysses_size: auto
  attention_backend: native
  extra_parallel_modules: ["text_encoder", "vae"]
```

## Hybrid Cache and Parallelism

Define a hybrid cache and parallel acceleration config yaml `hybrid.yaml` file that contains:

```yaml
cache_config:
  max_warmup_steps: 8 
  warmup_interval: 2  
  max_cached_steps: -1
  max_continuous_cached_steps: 2  
  Fn_compute_blocks: 1
  Bn_compute_blocks: 0
  residual_diff_threshold: 0.12
  enable_taylorseer: true
  taylorseer_order: 1
parallelism_config:
  ulysses_size: auto
  attention_backend: native
  extra_parallel_modules: ["text_encoder", "vae"]
```
Then, apply the hybrid cache and parallel acceleration config from yaml. 
```python
>>> import cache_dit
>>> cache_dit.enable_cache(pipe, **cache_dit.load_configs("hybrid.yaml"))
```

## Attention Backend

In some cases, users may want to only specify the attention backend without any other optimazation configs. In this case, you can define a yaml file `attention.yaml` that only contains:

```yaml
attention_backend: "flash" # _flash_3 for Hopper
```
Then, apply the attention backend config from yaml. 
```python
>>> import cache_dit
>>> cache_dit.enable_cache(pipe, **cache_dit.load_configs("attention.yaml"))
```

## Quantization

You can also specify the quantization config in the yaml file. For example, define a yaml file `quantize.yaml` that contains:

```yaml
quantize_config: # quantization configuration for transformer modules
  # float8 (DQ), float8_weight_only, float8_blockwise, int8 (DQ), int8_weight_only, etc.
  quant_type: "float8" 
  # layers to exclude from quantization (transformer). layers that contains any of the 
  # keywords in the exclude_layers list will be excluded from quantization. This is useful 
  # for some sensitive layers that are not robust to quantization, e.g., embedding layers.
  exclude_layers:
    - "embedder"
    - "embed"
  verbose: false # whether to print verbose logs during quantization
```
Then, apply the quantization config from yaml. 
```python
>>> import cache_dit
>>> cache_dit.enable_cache(pipe, **cache_dit.load_configs("quantize.yaml"))
```
Please also enable torch.compile for better performance if you are using quantization.
```python
cache_dit.set_compile_configs()
pipe.transformer = torch.compile(pipe.transformer)
```

## Combined Configs: Cache + Parallelism + Quantization

You can also combine all the above configs together in a single yaml file `combined.yaml` that contains:

```yaml
cache_config:
  max_warmup_steps: 8 
  warmup_interval: 2  
  max_cached_steps: -1
  max_continuous_cached_steps: 2  
  Fn_compute_blocks: 1
  Bn_compute_blocks: 0
  residual_diff_threshold: 0.12
  enable_taylorseer: true
  taylorseer_order: 1
parallelism_config:
  ulysses_size: auto
  attention_backend: native
  extra_parallel_modules: ["text_encoder", "vae"]
quantize_config: 
  quant_type: "float8" 
  exclude_layers: 
    - "embedder"
    - "embed"
  verbose: false 
```
Then, apply the combined cache, parallelism and quantization config from yaml. 
```python
>>> import cache_dit
>>> cache_dit.enable_cache(pipe, **cache_dit.load_configs("combined.yaml"))
```
Please also enable torch.compile for better performance if you are using quantization.
```python
pipe.transformer = torch.compile(pipe.transformer)
```

## Quick Examples 

```bash
# recommend: install latest stable release of torch for better compile compatiblity.
pip3 install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu129 --upgrade
# recommend: install latest torchao nightly due to issue: https://github.com/pytorch/ao/issues/3670
pip3 install --pre torchao --index-url https://download.pytorch.org/whl/cu129
pip3 install transformers accelerate bitsandbytes opencv-python-headless einops imageio-ffmpeg ftfy 
pip3 install git+https://github.com/huggingface/diffusers.git # latest or >= 0.36.0
pip3 install git+https://github.com/vipshop/cache-dit.git # latest

git clone https://github.com/vipshop/cache-dit.git && cd cache-dit/examples/configs

python3 -m cache_dit.generate flux --config cache.yaml
torchrun --nproc_per_node=4 -m cache_dit.generate flux --config hybrid.yaml
torchrun --nproc_per_node=4 -m cache_dit.generate flux --config parallel.yaml
torchrun --nproc_per_node=4 -m cache_dit.generate flux --config parallel_2d.yaml
torchrun --nproc_per_node=8 -m cache_dit.generate flux --config parallel_3d.yaml
torchrun --nproc_per_node=4 -m cache_dit.generate flux --config parallel_usp.yaml
torchrun --nproc_per_node=4 -m cache_dit.generate flux --config combined.yaml --compile
```
