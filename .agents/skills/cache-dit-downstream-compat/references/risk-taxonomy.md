# Risk taxonomy — eight ways a cache-dit change breaks a downstream

Ordered by how often it bites and how expensive it is to notice. Every entry carries a real instance, so
you can pattern-match instead of re-deriving the failure mode.

## R1 — symbol or module renamed (import-level break)

A downstream importing a path that no longer exists fails loudly, which is the cheap case — unless the
import sits in a `try/except ImportError` fallback, where it turns into a misleading error.

Instances:

- `cache_dit.quantization.svdq_is_available` / `svdq_get_load_error` were **removed**; the only
  availability helper left is `cache_dit.kernels.cuda_ext_is_available`. sglang's docs still teach
  `svdq_is_available()` (`docs/docs/sglang-diffusion/cache_dit.mdx`), its code does not call it.
- package renames: `_utils` → `cli` (#1291), `attention._interface` → `interface`,
  `kernels._svdquant` → `svdquant`, `kernels._*` → no leading underscore (#1292/#1293).
- `parallelism` → `distributed` at **v1.3.5 → v1.3.6** (not 1.5.0; the two packages export the same
  names). A downstream supporting pre-1.5 versions needs an ordered fallback, not a hard rename:
  `try: from cache_dit.distributed import …` / `except ImportError: from cache_dit.parallelism import …`.

## R2 — signature or argument semantics changed

- `enable_cache(..., attention_backend=…)` gained auto-selection when both `attention_backend` and
  `parallelism_config` are `None`. On CUDA it resolves to `None` (harmless); **on NPU it calls
  `set_attn_backend(transformer, "_native_npu")`**, overwriting whatever processors the downstream
  installed. Any downstream with its own attention path (xDiT USP/Ring, sglang) must pass
  `attention_backend` explicitly on NPU.
- `BlockAdapterRegister.register(name, *, supported, owned)` became keyword-only and grew `owned`.
- `refresh_context(module, **kwargs)` treats kwargs as shorthand **only** for cache fields it can parse;
  anything else must go inside a `cache_config` object, otherwise it is silently ignored with a warning.

## R3 — dispatch gate: prefix registry (highest value)

`BlockAdapterRegister` matches **class-name prefixes**, and downstreams keep upstream class names on
their vendored copies. Two distinct sub-failures:

1. **Gate loosening.** A downstream gating on `BlockAdapterRegister.is_supported(transformer)` starts
   accepting families cache-dit registered after its last audit. Without a downstream-side allowlist the
   behaviour changes with no code change on either side.
2. **Wrong adapter bound (before #1320).** Adapters written for cache-dit's *own* vendored pipeline were
   reachable by any same-named foreign class. Consequence is either a bare `AssertionError`
   (`WanS2V`, `LongLive2`, `LongCatVideoAvatar`, `Magi2Preview`, `MOVA`, `BooguImage`) or — worse — silent
   mis-caching for the `_relaxed_assert` ones (`LingBotVideo`, `Nomas`, `HiDreamO1Image`, `MOSSImage`,
   `SanaVideo2`, `MingImage`, `CausVid`). `LongLive2` is the canonical danger: its clean-latent pass must
   replay the raw block stack (`force_full_compute=True`), otherwise a skipped block never writes its
   per-layer KV and the next chunk reads a stale key.

Since #1320 (merged 2026-11-13) `owned=True` adapters are skipped for any object whose MRO is not inside
`cache_dit`, the lookup **stops** instead of falling through to a shorter prefix, and `is_supported`
follows the same rule. A downstream that relied on the silent path must now pass an explicit
`BlockAdapter` — which is the right contract for a vendored fork.

## R4 — fork drift (a downstream copied cache-dit internals)

vllm-omni forked `CachedAdapter.create_context` for Bagel and kept the old early-exit shape: it returns
`block_adapter.pipe` while `Cachify` unpacks `_, contexts_kwargs = cls.create_context(...)`. That is a
`TypeError` waiting for the early exit to be reached; cache-dit later tightened the condition
(`isinstance(pipe._context_manager, _supported_managers)`), which makes it *less* reachable but does not
fix the fork. Any downstream that copies an internal method must be diffed against it on every audit.

## R5 — counter / state-machine semantics (runs, results change)

`#1114` (in **v1.5.3**) rewrote the continuous-cache streak: `mark_step_begin` resets on the previous
step's real hits, `add_cached_step` always `+1`, the old `+=2` hack and the reset-on-cap are gone. Any
downstream that sets `max_continuous_cached_steps` gets a different cached-step sequence for the same
config — TRT-LLM defaults it to `3`, sglang passes `3`, vllm-omni validates a window of `4`. Numbers
measured on ≤1.5.2 are not comparable.

## R6 — short-schedule mask constraint

`steps_mask(mask_policy=…, total_steps=N)` accepts only `4` and `6` when `N < 8`. Downstreams that pass a
user's raw step count must guard it: TRT-LLM downgrades to a plain `refresh_context` with a warning;
xDiT catches only `(ImportError, AttributeError)`, so `--num_inference_steps 5` or `7` dies on an
`AssertionError` from inside cache-dit.

## R7 — `refresh_context` needs a persistent context

`refresh_context` only does something when `context_manager.persistent_context` is `True`, i.e. for
transformer-only adapters (`BlockAdapter(transformer=…)` → `FakeDiffusionPipeline`). An adapter created
with `pipe=` has a non-persistent manager: refresh logs a warning and **returns without changing
anything**. Downstreams that pass a real pipeline must rebuild the adapter or re-`enable_cache` instead.

## R8 — stats / log shape (cosmetic but misleading)

`summary(module)` returns **one entry per context**, observed as 2 for a single transformer and stable
across refreshes — do not index `[0]`. The owned-mismatch warning is de-duplicated per class name, so a
counter that expects one line per `enable_cache` call will undercount.
