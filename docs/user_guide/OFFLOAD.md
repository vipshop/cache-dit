# Layerwise Offload

## Basic Usage

Cache-DiT provides a generic layerwise offload utility for `nn.Module` components. It keeps the
selected submodules on the `offload_device` between forwards, moves them to the `onload_device`
just before execution, and then offloads them again after the layer finishes.

This is useful when a model or component does not fit comfortably in GPU memory, but you still
want to run the forward pass on CUDA one layer at a time. The public APIs are: <span style="color:#c77dff;">layerwise_offload(...)</span>: generic onload/offload wrapper; <span style="color:#c77dff;">layerwise_cpu_offload(...)</span>: convenience wrapper for CPU offload; <span style="color:#c77dff;">remove_layerwise_offload(...)</span>: remove all registered layerwise offload hooks from a root module.

By default, <span style="color:#c77dff;">layerwise_cpu_offload</span> selects leaf modules under the root module. You can narrow the scope with <span style="color:#c77dff;">module_names=[...]</span> or <span style="color:#c77dff;">module_filter=...</span> when you only want to offload part of the
module tree.

```python
from cache_dit.offload import layerwise_cpu_offload

layerwise_cpu_offload(model, onload_device="cuda")
```

## Pipeline Component

In practice, you will usually apply layerwise offload to a large component such as a transformer
module instead of the whole pipeline object. If you only want to offload specific submodules, pass explicit names:

```python
handle = layerwise_cpu_offload(
  pipe.transformer,
  onload_device="cuda",
  module_names=["transformer_blocks.0", "transformer_blocks.1"],
)
```

## Async Transfer

For CUDA onload plus CPU offload, you can enable asynchronous state transfers:

```python
handle = layerwise_cpu_offload(
  pipe.transformer,
  onload_device="cuda",
  async_transfer=True,
  transfer_buckets=2,
  persistent_buckets=2,
)
```

<span style="color:green;">transfer_buckets</span>: How many future targets should be prefetched when async transfer is enabled. A value of 1 still enables async overlap on a single copy lane and preserves the current single-target lookahead behavior. Larger values do not mean "more overlap is always better": in the current design, each prefetched target is already materialized onto CUDA, so increasing `transfer_buckets` also increases the number of future layers whose weights are resident on GPU at the same time.

<span style="color:green;">persistent_buckets</span>: How many leading targets should stay resident on the onload device for the full handle lifetime instead of participating in per-forward onload/offload. These targets are materialized onto the onload device during handle creation, before the first forward starts.

Envrionment: NVIDIA L20, FLUX.1-dev, 28 steps, 1024 x 1024, D=Diffusers.

|w/o offload| sequential (D) | cpu offload (D) | layerwise | + async transfer | + persistent|
|:---:|:---:|:---:|:---:|:---:|:---:|
|~38GiB|~1GiB|~25GiB|~1GiB|~4GiB|~8GiB|
|24s|335s|56s|49s|41s|33s|

Notes: <span style="color:#c77dff;">async_transfer=True</span> currently requires CUDA onload and CPU offload. <span style="color:#c77dff;">transfer_buckets</span> controls how many future target modules are prefetched, but larger values can sharply increase peak/reserved CUDA memory because more future targets are already resident on GPU. In practice, moving from 4 to 8 effective streams/buckets can easily multiply memory usage by 2-3x on large transformers. Prefer starting with <span style="color:#c77dff;">transfer_buckets=1</span> or <span style="color:#c77dff;">2</span>, and only increase further if profiling shows a real latency win and the memory headroom is still acceptable. Pairing a small <span style="color:#c77dff;">transfer_buckets</span> with a modest <span style="color:#c77dff;">persistent_buckets</span> is usually a better tradeoff than aggressively increasing async prefetch depth.

This switch enables cache-dit's generic sequential CPU offload for `nn.Module` components. It is intended for custom non-diffusers modules and is mutually exclusive with the diffusers pipeline offload switches such as `--cpu-offload` and `--sequential-cpu-offload` in Cache-DiT's CLI. If you enable both, the behavior is undefined.
