# Cache Management in cache-dit

This document summarizes how cache-dit manages cache internally, with a focus on:

- whether cache is managed by diffusion step or transformer layer,
- what the unified APIs are,
- and how to reason about behavior when debugging.

It is intended as a practical, reusable engineering note.

## 1) Short Answer

cache-dit uses a **hybrid design**:

- **DBCache** is primarily **step-driven** (decide cache-or-compute per diffusion step),
  while its cached data is produced and consumed at **block/layer segments** (`Fn/Mn/Bn`).
- **DBPrune** is primarily **block/layer-driven** (decide prune-or-compute per block).

So this is not a strict "only step-level" or "only layer-level" design.

## 2) Unified Interface (Public API)

The main unified interface is:

- `cache_dit.enable_cache(...)`
- `cache_dit.refresh_context(...)`
- `cache_dit.disable_cache(...)`
- `cache_dit.summary(...)`

Key source files:

- `src/cache_dit/caching/cache_interface.py`
- `src/cache_dit/caching/__init__.py`
- `src/cache_dit/__init__.py`

## 3) High-Level Architecture

### 3.1 Entry and Adapter Layer

`enable_cache(...)` routes to `CachedAdapter.apply(...)`, which:

1. resolves or validates a `BlockAdapter`,
2. creates cache contexts,
3. replaces model block lists with cache-aware wrappers (`UnifiedBlocks`),
4. installs hooks for call/forward and stats collection.

Main implementation:

- `src/cache_dit/caching/cache_adapters/cache_adapter.py`
- `src/cache_dit/caching/block_adapters/block_adapters.py`
- `src/cache_dit/caching/block_adapters/adapters.py`

### 3.2 Context Layer

Context creation is abstracted by `ContextManager`:

- `CacheType.DBCache` -> `CachedContextManager`
- `CacheType.DBPrune` -> `PrunedContextManager`

Each unique block group gets its own context name (usually generated from block identity), so
multi-block and multi-transformer pipelines can be managed separately.

Main implementation:

- `src/cache_dit/caching/cache_contexts/context_manager.py`
- `src/cache_dit/caching/cache_contexts/cache_manager.py`
- `src/cache_dit/caching/cache_contexts/cache_context.py`

### 3.3 Block Execution Layer

`UnifiedBlocks` dispatches to either `CachedBlocks` (DBCache) or `PrunedBlocks` (DBPrune),
then to pattern-specific implementations according to forward I/O pattern.

Main implementation:

- `src/cache_dit/caching/cache_blocks/__init__.py`
- `src/cache_dit/caching/cache_blocks/pattern_base.py`

## 4) DBCache: Step-Driven + Segment-Level Reuse

### 4.1 Step-level decision signals

DBCache decides whether current step can use cache with context state such as:

- warmup window (`max_warmup_steps`, `warmup_interval`)
- explicit step masking (`steps_computation_mask`, `steps_computation_policy`)
- limits (`max_cached_steps`, `max_continuous_cached_steps`)
- residual criteria (`residual_diff_threshold`, optional accumulated threshold)
- separate CFG/non-CFG bookkeeping (if enabled)

These are maintained in `CachedContext`/`CachedContextManager` and advanced by `mark_step_begin()`.

### 4.2 Segment-level execution (`Fn/Mn/Bn`)

Within one transformer block-list forward:

1. run **Fn** (first N blocks) to obtain stable signal,
2. compute similarity vs previous step,
3. if cacheable:
   - reuse cached Mn/Bn-related states via `apply_cache(...)`,
   - optionally run **Bn** (last N blocks) for correction,
4. otherwise:
   - compute **Mn** (middle blocks),
   - store residual or hidden-state buffers,
   - run **Bn**.

Important takeaway:

- Decision is step-level, but cache payload and approximation are block-segment-level.

## 5) DBPrune: Block-Level Dynamic Pruning

DBPrune stores per-block references and residuals, and for each block in a step:

- decides whether to prune that block,
- if pruned, approximates output from cached values,
- otherwise computes and updates per-block cache.

So DBPrune is more explicitly layer/block-granular than DBCache.

See:

- `src/cache_dit/caching/cache_blocks/pattern_base.py` (prune path)
- `docs/user_guide/CACHE_API.md` (user-facing description)

## 6) Is There One Global Context or Multiple Contexts?

It is **multiple contexts** under one manager:

- one pipeline gets one context manager instance,
- each unique block list (and each transformer in multi-transformer models) gets
  a dedicated context name and state.

This enables:

- different block groups to track independent cache history,
- optional per-block-group config override via `ParamsModifier`.

## 7) Where Configuration Lives

Primary config class:

- `DBCacheConfig` (alias of `BasicCacheConfig`)

Core fields:

- `Fn_compute_blocks`, `Bn_compute_blocks`
- `residual_diff_threshold`
- `max_warmup_steps`, `warmup_interval`
- `max_cached_steps`, `max_continuous_cached_steps`
- `steps_computation_mask`, `steps_computation_policy`
- `num_inference_steps`
- `force_refresh_step_hint`, `force_refresh_step_policy`

Config source:

- `src/cache_dit/caching/cache_contexts/cache_config.py`

## 8) Transformer-only Usage and Refresh

When not using a full `DiffusionPipeline` (for example custom runtime/component execution),
cache context may be persistent and should be refreshed when request-level conditions change
(notably `num_inference_steps`).

Use:

- `cache_dit.refresh_context(transformer, ...)`

This is important to avoid stale step accounting across requests.

## 9) Practical Debug Checklist

If cache behavior looks wrong, check in this order:

1. **Context exists**: transformer should have `_context_manager` after `enable_cache`.
2. **Pattern matched**: target block list matches selected `ForwardPattern`.
3. **Step accounting**: warmup or `steps_computation_mask` may force compute.
4. **Threshold logic**: `residual_diff_threshold <= 0` effectively disables dynamic reuse.
5. **Caps reached**: `max_cached_steps` or `max_continuous_cached_steps` may block caching.
6. **CFG mode**: separate CFG can split cache stats/state.
7. **Context refresh**: for persistent/transformer-only flows, ensure proper `refresh_context`.

`cache_dit.summary(...)` is the fastest way to inspect cached steps and residual diff distributions.

## 10) Mental Model to Reuse

Use this compact mental model for reviews and design discussions:

- **DBCache** = "step scheduler + segment cache (`Fn/Mn/Bn`)"
- **DBPrune** = "block-level dynamic skip with cached approximation"
- **Public API** = `enable_cache / refresh_context / disable_cache / summary`
- **State owner** = context managers with per-block-group contexts

This model is accurate enough for most development, debugging, and extension tasks.
