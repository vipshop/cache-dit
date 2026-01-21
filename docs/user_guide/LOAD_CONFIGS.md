# Use Yaml Config File

Cache-DiT now supported load the acceleration configs from a custom yaml file. Here are some examples.

## Single GPU inference  

Define a `config.yaml` file that contains:

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
>>> cache_dit.enable_cache(pipe, **cache_dit.load_configs("config.yaml"))
```

## Distributed inference  

Define a `parallel_config.yaml` file that contains:

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
  parallel_kwargs:
    attention_backend: native
    extra_parallel_modules: ["text_encoder", "vae"]
```
Then, apply the distributed inference acceleration config from yaml. `ulysses_size: auto` means that cache-dit will auto detect the `world_size` as the ulysses_size. Otherwise, you should mannually set it as specific int number, e.g, 4.
```python
>>> import cache_dit
>>> cache_dit.enable_cache(pipe, **cache_dit.load_configs("parallel_config.yaml"))
```

## Quick Examples 

```bash
pip3 install torch==2.9.1 transformers accelerate torchao bitsandbytes torchvision 
pip3 install opencv-python-headless einops imageio-ffmpeg ftfy 
pip3 install git+https://github.com/huggingface/diffusers.git # latest or >= 0.36.0
pip3 install git+https://github.com/vipshop/cache-dit.git # latest

git clone https://github.com/vipshop/cache-dit.git && cd examples

python3 generate.py flux --config config.yaml
torchrun --nproc_per_node=4 --local-ranks-filter=0 generate.py flux --config parallel_config.yaml
```
