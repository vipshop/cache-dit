# Layerwise Offload

## Basic Usage

Cache-DiT provides a generic layerwise offload utility for `nn.Module` components. It keeps the
selected submodules on the `offload_device` between forwards, moves them to the `onload_device`
just before execution, and then offloads them again after the layer finishes.

This is useful when a model or component does not fit comfortably in GPU memory, but you still
want to run the forward pass on CUDA one layer at a time. The public APIs are:

- `cache_dit.offload.layerwise_offload(...)`: generic onload/offload wrapper.
- `cache_dit.offload.layerwise_cpu_offload(...)`: convenience wrapper for CPU offload.
- `cache_dit.offload.remove_layerwise_offload(...)`: remove all registered layerwise offload
  hooks from a root module.

By default, `layerwise_cpu_offload` selects leaf modules under the root module. You can narrow the
scope with `module_names=[...]` or `module_filter=...` when you only want to offload part of the
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
)
```

Notes: `async_transfer=True` currently requires CUDA onload and CPU offload. `transfer_buckets` controls how many future target modules are prefetched. The async path is most useful when the component has many repeated submodules.

This switch enables cache-dit's generic sequential CPU offload for `nn.Module` components. It is intended for custom non-diffusers modules and is mutually exclusive with the diffusers pipeline offload switches such as `--cpu-offload` and `--sequential-cpu-offload` in Cache-DiT's CLI. If you enable both, the behavior is undefined.
