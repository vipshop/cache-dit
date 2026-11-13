---
name: cache-dit-downstream-compat
description: "Check whether cache-dit's changes break a downstream framework that calls cache-dit (vllm-omni / sglang-diffusion / xDiT / TensorRT-LLM visual_gen, or any new caller). Use when cache-dit has landed refactors, renames, registry additions or behavior changes and the downstreams pin an older version; when a downstream reports `ImportError` / `AttributeError` / `AssertionError` / silent quality drift after a cache-dit bump; when preparing a cache-dit release and you need a blast-radius report per downstream; or when writing a new cache-dit integration and you want to know which APIs are safe to depend on. 触发词：下游兼容性、cache-dit 版本升级、API 不兼容、接口断裂、sglang cache-dit、vllm-omni cache-dit、xDiT cache-dit、TensorRT-LLM cache-dit、升级 cache-dit 后精度下降、注册表派发、BlockAdapterRegister."
user-invocable: true
---

# cache-dit → downstream compatibility audit

Answer one question: **do the cache-dit symbols, signatures and *semantics* a downstream relies on still
hold on the target revision?** Import-level breaks are the easy half; the expensive bugs are semantic
(dispatch gate, CFG accounting, cache-streak counters). Both must be checked.

## When to run this

- cache-dit landed refactors/renames or new model adapters, and a downstream pins an older version.
- A downstream bumps its cache-dit pin and reports a failure or a quality drop.
- Preparing a cache-dit release: produce a per-downstream blast radius.
- Writing a new integration: know which cache-dit surface is contractual vs incidental.

## Step 0 — pin down the two revisions (never skip)

For each downstream, record **the exact cache-dit version it will actually resolve**, not the one you
assume. Note the shape of the specifier, because it decides what the audit means:

| specifier shape | example | meaning for the audit |
|---|---|---|
| exact pin | `cache-dit==1.5.0` | compare that tag vs the target revision |
| range with upper bound | `>=1.5.0,<1.6.0` | pip resolves the newest release; a 1.6 break is blocked |
| **open-ended** | `cache-dit>=1.3.5` | anything can land, including an unreleased `main` — treat as highest risk |

Also resolve the tag boundary of any behavior change you find: `git tag --contains <commit>` tells you
whether a downstream on `1.5.1` already has it.

## Step 1 — enumerate the touch surface (`scripts/scan_refs.py`)

```bash
python scripts/scan_refs.py <downstream-dir>            # human summary
python scripts/scan_refs.py <downstream-dir> --json     # machine readable
```

It AST-walks every `*.py` and reports `from cache_dit…import X`, `import cache_dit` + `cache_dit.<attr>`
uses. Do not rely on `grep` alone: multi-line imports and `getattr(cache_dit, name)` are easy to miss,
and comments/docs produce false positives.

## Step 2 — resolve every symbol on the target revision

Run the probe inside an env where `import cache_dit` resolves to the **target tree**, never to a
pre-installed copy:

```bash
ENV_PY="$(conda info --base)/envs/<env>/bin/python"   # never `conda run` for this: it can route elsewhere
PYTHONPATH=<cache-dit>/src "$ENV_PY" scripts/probe_api.py
```

`probe_api.py` prints the version plus signatures of the surface all four known downstreams use. Diff it
against the downstream's call sites; a symbol that moved package (`cache_dit.quantization.svdq_*` →
`kernels.cuda_ext_*`) or a module that got renamed (`_utils`→`cli`, `parallelism`→`distributed` at
**v1.3.6**, `attention._interface`→`interface`, `kernels._svdquant`→`svdquant`) shows up here.

Verify the interpreter really points at the tree: print `cache_dit.__file__`. An editable install can
outrank `PYTHONPATH` in other setups, so check instead of assuming.

## Step 3 — registry dispatch and gate semantics (`scripts/diff_registry.py`)

The single most productive check for cache-dit. `BlockAdapterRegister` dispatches by **class-name
prefix**, and downstreams keep upstream class names on vendored copies.

```bash
python scripts/diff_registry.py --repo <cache-dit> --base v1.5.1 [--target HEAD]
```

It prints prefixes added/removed and, for the target, which prefixes are marked `owned=True`
(adapter valid only for cache-dit's own vendored pipeline). Then apply the two rules:

1. **Gate loosening**: a downstream that gates on `BlockAdapterRegister.is_supported(transformer)` will
   silently start accepting new families when cache-dit registers them. Enumerate the downstream's model
   class names against both prefix sets (`diff_registry.py --classes <dir>`).
2. **Owned vs family**: before `owned=` existed (PR #1320, in `main` after 2026-11-13), a *foreign* class
   reusing an owned family name was dispatched into an adapter that assumes cache-dit's own
   implementation → hard `AssertionError`, or worse, silent mis-caching (see
   `references/risk-taxonomy.md` §R3). After #1320 such classes are rejected; a downstream relying on the
   silent path must pass an explicit `BlockAdapter`.

## Step 4 — behavior probes (`scripts/probe_behavior.py`)

Symbols existing is not enough. This probes, on a synthetic DiT, every semantic the four known
downstreams depend on:

- `refresh_context(module, num_inference_steps=N)` **merges** into the existing config (Fn/Bn/R kept) —
  `persistent_context` must be `True`, which holds for transformer-only adapters (`BlockAdapter(transformer=…)`)
  but **not** when the adapter carries `pipe=`: then refresh silently warns and no-ops. Downstreams that
  pass a real pipeline must not depend on it.
- `enable_separate_cfg` as a direct refresh kwarg (sequential-CFG models such as Qwen-Image).
- `disable_cache(BlockAdapter)` clears `_is_cached` and `_context_manager`.
- `summary(module)` returns **more than one** `CacheStats` per transformer (observed 2, stable across
  refreshes) — never index `[0]`.
- `steps_mask` on short schedules: `total_steps < 8` accepts only `4` and `6`; anything else asserts.
  Check whether the downstream guards it (TRT-LLM does, xDiT does not).
- `BlockAdapter(auto=True)` without `forward_pattern` asserts inside `auto_block_adapter`.

## Step 5 — run the downstream's own tests, cheapest first

| downstream | command | notes |
|---|---|---|
| sglang-diffusion | `PYTHONPATH=<sglang>/python python python/sglang/multimodal_gen/test/unit/test_cache_dit_integration.py` | 23 CPU cases; most use a **stub** `cache_dit`, one class exercises the real package |
| xDiT | `CUDA_VISIBLE_DEVICES="" PYTHONPATH=<xDiT>:<stub-dir> python -m pytest tests/unit/model_executor/cache/ -q` | needs `CUDA_VISIBLE_DEVICES=""` (else `fp8_comms` hard-imports `yunchang`) plus stubs for `yunchang` / `distvae` |
| vllm-omni | `PYTHONPATH=<vllm-omni> python -c "from vllm_omni.diffusion.cache.cachedit import backend, model_specific"` | needs real `vllm`; if absent, verify the import face symbol-by-symbol instead and say so |
| TensorRT-LLM | `pytest tests/unittest/_torch/visual_gen/test_cache_dit.py` | needs `tensorrt_llm` (compiled libs + `nvtx`); when unavailable, fall back to static + probe evidence and label it |

Report what you could **not** run. "Import face verified, runtime not executed" is an acceptable
conclusion; silently implying a green run is not.

## Step 6 — the e2e sanity run (only when the change can affect numerics)

Baseline vs cache on one GPU, then PSNR/SSIM:

```bash
export FLUX_DIR=<models>/FLUX.1-dev
python -m cache_dit.generate flux --width 1024 --height 1024 --num_inference_steps 28 --seed 42 \
    --warmup 1 --repeat 3 --save-path base.png
python -m cache_dit.generate flux ... --cache --save-path cache.png
python -m cache_dit.generate flux ... --cache --compile --save-path cache_compile.png
cache-dit-metrics psnr ssim -i1 base.png -i2 cache.png
```

Confirm cache really engaged (`Optimization` tag carries `DBCache_…_S<n>`, `Collected Context Config`,
`Match Blocks`) — a silent no-cache run also "passes".

## Verdict levels

| level | meaning | example |
|---|---|---|
| **BROKEN** | import/attribute missing, or the call raises | renamed module, removed helper |
| **SEMANTIC** | runs, but caching decisions changed (step accounting, streak cap) | `#1114` streak reset; `has_separate_cfg` from the family table |
| **GATE** | a previously rejected model is now silently accepted | registry prefix growth |
| **COSMETIC** | extra stats / log noise, no behavior change | `summary` returning 2 entries |

Deliver one table per downstream: `symbol / call site / verdict / evidence (path:line)`, then the fix
options. Never report a verdict without a file:line or a probe transcript.

## Reference material

- `references/risk-taxonomy.md` — the eight failure classes with a real instance of each (R1–R8).
- `references/findings.md` — the 2026-11-13 audit: pins, per-downstream verdicts, what PR #1320 changed,
  and which traps each downstream already guards against.

## Traps that cost time

- `conda run -n <env>` can execute a *different* env; use `$(conda info --base)/envs/<env>/bin/python`.
- A downstream test suite that **stubs** `cache_dit` proves nothing about the real package — find the one
  test that imports the real module, or run the probes yourself.
- Metrics: `cache_dit.metrics.calculate_psnr` assumes a 255 peak; feeding float `[0,1]` adds ~48 dB.
- Do not edit downstream repos during an audit unless asked; produce a pasteable diff instead.
