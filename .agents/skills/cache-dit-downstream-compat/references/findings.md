# Findings snapshot — 2026-11-13

Audit of four downstreams against cache-dit `dev`/`main` = `d8e592e6` + PR #1320 (`8dbb58b0`).
Re-run the steps in `SKILL.md` before trusting these numbers; the table records **pins as of that date**.

## Pins

| downstream | pin | resolves to (then) | note |
|---|---|---|---|
| sglang-diffusion (main) | `cache-dit==1.5.1` | 1.5.1, no `#1114` | ROCm/XPU/NPU/CPU variants pin 1.1.8 / 1.2.3 / 1.3.0 / 1.3.5 → all still have `cache_dit.parallelism` |
| vllm-omni | `cache-dit==1.5.0` | 1.5.0, no `#1114` | |
| xDiT | `cache-dit>=1.5.0,<1.6.0` | newest release (1.5.3 → has `#1114`) | upper bound is the right shape |
| TensorRT-LLM visual_gen | `cache-dit>=1.3.5` | newest release, **no upper bound** | declared floor is stale: the code uses `force_refresh_step_hint`, `ParamsModifier(calibrator_config=)`, `steps_mask`, which 1.3.5 does not have |

## Verdicts

| downstream | import face | semantics | gate | runnable evidence |
|---|---|---|---|---|
| sglang-diffusion | all symbols present | `similarity` patch still matches (`parallelism` → `distributed` is the only stale path, and only in the `<1.5.0` fallback) | **GATE**: registry 45 → 80 prefixes silently un-gates LingBotVideo / LongLive2 / Nomas / MingImage; after #1320 those are `owned` → rejected with sglang's own `ValueError` | 23 CPU cases green on dev; `test_cache_dit_per_request.py` blocked by an unrelated `sgl_kernel` ABI mismatch |
| vllm-omni | all symbols present (incl. `CachedAdapter`, `CachedBlocks_Pattern_3_4_5`, `BasicCacheConfig`, `CachedContextManager`) | **fork drift** in `BagelCachedAdapter.create_context` (returns one value where `Cachify` unpacks two); `LongCatVideoAvatar` used to hit a hard assert | same gate story; `Anima` is protected by `_NO_CACHE_ACCELERATION` | imports verified symbol-by-symbol; `vllm` is not installed in the audit env, so its runtime tests were not executed |
| xDiT | all symbols present | deep coupling all intact (`Fn_compute_blocks` instance patch, `can_cache(parallelized=)`, arbitrary `:pipefusion_patch_N` context names, `dataclasses.fields/replace`, deepcopy) | no registry use (always explicit `CacheDitAdapterConfig`) → immune to #1320 | `tests/unit/model_executor/cache/` **38 passed** + sd3 pipefusion **2 passed** on dev, CPU (`CUDA_VISIBLE_DEVICES=""` + stubs for `yunchang`/`distvae`) |
| TensorRT-LLM visual_gen | all symbols present | `refresh_context` per-request merge verified (Fn/R kept, `num_inference_steps` updated, `enable_separate_cfg` kwarg honoured); already guards the `<8` steps-mask rule | explicit `BlockAdapter` everywhere → immune to #1320 | `tensorrt_llm` cannot import in the audit env (missing `nvtx`), so its 513-line test file was not run |

## TensorRT-LLM specifics (this audit's new coverage)

Files: `tensorrt_llm/_torch/visual_gen/cache/{cache_dit_accelerator,cache_dit_enablers}.py`
(plus `visual_gen/args.py` for `CacheDiTConfig`).

- Enablers for `WanPipeline` / `WanImageToVideoPipeline` / `FluxPipeline` / `Flux2Pipeline` /
  `LTX2Pipeline` / `QwenImagePipeline` / `QwenImageLayeredPipeline`, dispatched by **exact class name**,
  unknown pipelines raise `ValueError`. No registry use at all.
- `_has_compiled_blocks()` turns off `check_forward_pattern` / `check_num_outputs` when blocks are
  `torch._dynamo.OptimizedModule` — correct, because compiled blocks expose `(*args, **kwargs)`.
- `_refresh_ctx()` already downgrades `scm_steps_mask_policy` to a plain refresh for `N < 8` unless
  `N ∈ {4, 6}` — the R6 trap that xDiT still trips over.
- `refresh_context(module, …)` is called on **every request**; verified on dev that the config merges
  (`Fn=1`, `R=0.12` survive) and that `enable_separate_cfg` can be flipped per request (Qwen-Image).
- `summary(module, details=False)` returns 2 entries per transformer; their `get_stats()` stores the raw
  list, so this is cosmetic, but do not "fix" it by taking `[0]`.
- `max_continuous_cached_steps` defaults to **3** and `residual_diff_threshold` to 0.24 in
  `CacheDiTConfig`, so the `#1114` streak rewrite (present in 1.5.3, which their open-ended pin resolves
  to) changes the cached-step sequence for default settings. Any perf/quality baseline recorded before
  1.5.3 must be re-measured.
- Recommendation: change the pin to `cache-dit>=1.5.0,<1.6.0` so an unreleased 1.6 cannot land silently
  (the declared `1.3.5` floor is already fictional).

## What #1320 changed for downstreams

`owned=True` adapters are 14 registrations / 13 families: `WanS2V`, `WanSpeechToVideo`, `CausVid`,
`LongLive2`, `LongCatVideoAvatar`, `Magi2Preview`, `MOVA`, `BooguImage`, `LingBotVideo`, `Nomas`,
`HiDreamO1Image`, `MOSSImage`, `SanaVideo2`, `MingImage`. For a foreign same-named class,
`get_adapter()` now returns `None` and `is_supported()` returns `False` (plus a one-shot warning); the
lookup does not fall through to a broader prefix (`WanS2V*` no longer becomes a plain `Wan`), and
`has_separate_cfg()` no longer answers from the CFG prefix table for such a class.

Downstreams that were silently using those families must pass an explicit `BlockAdapter` — which is what
sglang's `_CUSTOM_BLOCK_ADAPTER_SPECS`, vllm-omni's `_cache_dit_adapter_config` and xDiT's
`CacheDitAdapterConfig` already do.

## Reported but not fixed downstream-side

- sglang: `from cache_dit.parallelism import …` in the `<1.5.0` fallback (dead on ≥1.3.6, raises a
  misleading `ModuleNotFoundError` if the primary import breaks for another reason). Fix: ordered
  `distributed` → `parallelism` fallback and re-raise the original error. Judged low priority
  (2026-11-13).
- sglang docs still reference `svdq_is_available()`, removed from cache-dit.
- vllm-omni: the Bagel `create_context` fork shape.
- xDiT: `BlockAdapter(pipe=pipe, auto=True)` without `forward_pattern` asserts; `scm_policy` with
  `--num_inference_steps ∈ {5, 7}` asserts.

## Legacy module map (for grepping old downstreams)

| old | new |
|---|---|
| `cache_dit.parallelism` (≤1.3.5) | `cache_dit.distributed` (≥1.3.6) |
| `cache_dit._utils` | `cache_dit.cli` |
| `cache_dit.attention._interface` | `cache_dit.attention.interface` |
| `cache_dit.kernels._svdquant` | `cache_dit.kernels.svdquant` |
| `cache_dit.quantization.svdq_is_available()` | `cache_dit.kernels.cuda_ext_is_available()` |
