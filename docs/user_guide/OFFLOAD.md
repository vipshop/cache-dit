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
  max_copy_streams=2,
  max_inflight_prefetch_bytes=2 * 1024**3,
  persistent_buckets=2,
  persistent_bins=2,
)
```

<span style="color:green;">transfer_buckets</span>: Base future-prefetch depth when async transfer is enabled. Runtime expands this into an effective future-target window of `min(4 * transfer_buckets, 8)` pending/ready prefetched targets ahead of the current execution point. A value of 1 still enables async overlap on a single copy lane while allowing a modest widened lookahead. Larger values do not mean "more overlap is always better": in the current design, each prefetched target is already materialized onto CUDA, so increasing the effective window also increases the number of future layers whose weights are resident on GPU at the same time.

<span style="color:green;">max_copy_streams</span>: Maximum number of async CUDA copy streams used by layerwise offload. This caps copy-lane concurrency without changing the logical lookahead depth implied by `transfer_buckets`. When omitted, runtime derives it from `transfer_buckets` and still applies its internal safety cap.

<span style="color:green;">max_inflight_prefetch_bytes</span>: Maximum total CUDA residency budget, in bytes, that async future-target prefetch may consume at once. This caps the combined footprint of both pending transfers and ready-but-not-yet-consumed prefetched targets, even when the effective prefetch window requests a deeper lookahead. When omitted, runtime does not apply an implicit byte-budget cap. Common examples are `1 * 1024**3` for 1 GiB, `4 * 1024**3` for 4 GiB, `8 * 1024**3` for 8 GiB, and `16 * 1024**3` for 16 GiB.

<span style="color:green;">persistent_buckets</span>: How many selected targets should stay resident on the onload device for the full handle lifetime instead of participating in per-forward onload/offload. These targets are materialized onto the onload device during handle creation, before the first forward starts.

<span style="color:green;">persistent_bins</span>: How many evenly distributed bins should be used when placing the `persistent_buckets` budget across the target list. A value of 1 keeps the original prefix behavior. Larger values spread persistent targets across multiple uniformly spaced ranges, which can improve overlap in deeper models without concentrating the full persistent budget only at the beginning.

Concrete example: if the selected target list has 32 targets, `persistent_buckets=16`, and `persistent_bins=4`, runtime keeps four evenly spaced persistent ranges resident on CUDA: `[t0, ..., t3]`, `[t8, ..., t11]`, `[t16, ..., t19]`, and `[t24, ..., t27]`.

Envrionment: NVIDIA L20, FLUX.1-dev, 28 steps, 1024 x 1024, D=Diffusers.

|w/o offload| sequential (D) | cpu offload (D) | layerwise | + async transfer | + persistent|
|:---:|:---:|:---:|:---:|:---:|:---:|
|~38GiB|~1GiB|~25GiB|~1GiB|~4GiB|~8GiB|
|24s|335s|56s|49s|41s|33s|

Notes: <span style="color:#c77dff;">async_transfer=True</span> currently requires CUDA onload and CPU offload. <span style="color:#c77dff;">transfer_buckets</span> controls the base lookahead request, while runtime widens the effective future-target window to <span style="color:#c77dff;">min(4 * transfer_buckets, 8)</span>. <span style="color:#c77dff;">max_copy_streams</span> still limits copy-lane concurrency and <span style="color:#c77dff;">max_inflight_prefetch_bytes</span> limits how much future-target weight state may already be resident on GPU across both pending and ready prefetched targets, but only when you set it explicitly. Larger <span style="color:#c77dff;">transfer_buckets</span> can still sharply increase peak/reserved CUDA memory if the inflight prefetch byte budget is also large, because more future targets may already be materialized on GPU. Prefer starting with <span style="color:#c77dff;">transfer_buckets=1</span> or <span style="color:#c77dff;">2</span>, keeping <span style="color:#c77dff;">max_copy_streams</span> modest, and only adding <span style="color:#c77dff;">max_inflight_prefetch_bytes</span> when profiling shows a real latency win and the memory headroom is still acceptable. Pairing a small <span style="color:#c77dff;">transfer_buckets</span> with a modest <span style="color:#c77dff;">persistent_buckets</span> and a suitable <span style="color:#c77dff;">persistent_bins</span> value is usually a better tradeoff than aggressively increasing async prefetch depth.

This switch enables cache-dit's generic sequential CPU offload for `nn.Module` components. It is intended for custom non-diffusers modules and is mutually exclusive with the diffusers pipeline offload switches such as `--cpu-offload` and `--sequential-cpu-offload` in Cache-DiT's CLI. If you enable both, the behavior is undefined.
