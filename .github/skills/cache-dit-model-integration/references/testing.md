# Testing and Verification Reference

When to read this: read this file before running baseline, Cache, CP, TP, TE-P, VAE-P, or final integration tests. It contains local model path setup, command templates, PSNR/SSIM acceptance criteria, result tables, and debugging tips. Read [`./install.md`](./install.md) first when the environment or editable install needs to be prepared. Return to `../SKILL.md` for phase gates.

Companion file: [`./perf_matrix.md`](./perf_matrix.md) owns the **full acceptance matrix** (row checklist, commands, 1/2-GPU default, on-request 4/8-GPU hybrid CP+TP). §7.6 below turns that matrix into pass/fail gates; read both when the deliverable is a performance + accuracy table for an already-supported model.

## 7. Testing & Verification

```text
╔══════════════════════════════════════════════════════════════╗
║  ⚠️  TESTING IS MANDATORY — DO NOT SKIP                      ║
║                                                              ║
║  Every integration feature (Cache, CP, TP, TE-P, VAE-P)      ║
║  MUST be tested against a baseline.  Guessing "it            ║
║  should work" is not acceptable.  Subtle bugs (e.g.,         ║
║  DTensor shard placement issues, incorrect GQA handling)     ║
║  can produce visually plausible but mathematically wrong     ║
║  output that is only caught by quantitative metrics.         ║
║                                                              ║
║  NEVER skip SSIM.  PSNR alone can NOT detect garbled         ║
║  images.  A corrupted image can still have PSNR > 25 dB.     ║
║  If you omit SSIM you WILL ship broken code.                 ║
╚══════════════════════════════════════════════════════════════╝
```

### 7.1 Prerequisites: Local Model Path

**⚠️ CRITICAL: Always set local model paths before testing. Downloading from HuggingFace Hub is extremely slow.**

Models are stored at `/workspace/dev/vipdev/hf_models/`. List available models with `ls /workspace/dev/vipdev/hf_models/`. Set the corresponding environment variable before testing:

```bash
# Find your model's env var in src/cache_dit/_utils/examples.py (_env_path_mapping)
# For example:
export FLUX_DIR=/workspace/dev/vipdev/hf_models/FLUX.1-dev
export QWEN_IMAGE_DIR=/workspace/dev/vipdev/hf_models/Qwen-Image

# Or use --model-path to bypass env vars:
python3 -m cache_dit.generate flux --model-path /workspace/dev/vipdev/hf_models/FLUX.1-dev
```

### 7.2 Installation

Installation is maintained separately in [`./install.md`](./install.md). Read it before running the test command matrix if Python code changed or the environment has not been prepared yet.

### 7.3 Test Command Matrix

**⚠️ IMPORTANT:** The commands below use `flux` as an **example template**. When testing your new model:

- Replace `flux` with your model's registered name (from `ExampleRegister.register("...")`)
- Replace `FLUX_DIR` with your model's environment variable
- Replace `flux_` in save paths with your model name
- These are **templates**, not instructions to test Flux itself.

```bash
# === 0. Set local model path (REQUIRED) ===
export <NEW_MODEL>_DIR=/workspace/dev/vipdev/hf_models/<your-model-dir>
# Alternative: use --model-path directly

# task: task name for temporary logging and output files 
# === 1. Baseline (no optimizations) ===
python3 -m cache_dit.generate <new_model> --save-path .tmp/{task}/<new_model>_base.png

# === 2. Cache acceleration ===
python3 -m cache_dit.generate <new_model> --cache --summary \
    --save-path .tmp/{task}/<new_model>_cache.png

# === 3. Context Parallelism: Ulysses (2 GPUs) ===
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 \
    -m cache_dit.generate <new_model> \
    --parallel ulysses --save-path .tmp/{task}/<new_model>_ulysses2.png

# === 4. Context Parallelism: Ring (2 GPUs) ===
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 \
    -m cache_dit.generate <new_model> \
    --parallel ring --save-path .tmp/{task}/<new_model>_ring2.png

# === 5. Tensor Parallelism (2 GPUs) ===
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 \
    -m cache_dit.generate <new_model> \
    --parallel tp --save-path .tmp/{task}/<new_model>_tp2.png

# === 6. TP + Text Encoder Parallelism ===
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 \
    -m cache_dit.generate <new_model> \
    --parallel tp --parallel-text --save-path .tmp/{task}/<new_model>_tp2_tep2.png

# === 7. TP + VAE Parallelism ===
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 \
    -m cache_dit.generate <new_model> \
    --parallel tp --parallel-vae --save-path .tmp/{task}/<new_model>_tp2_vaep2.png
```

> The commands above are **smoke-level**: none of them passes `--warmup/--repeat`, so their latency
> is a single un-warmed run. Before a number goes into a table, re-run it with `--warmup 2 --repeat 2`
> (§7.5), or use the `run()` helper in [`./perf_matrix.md`](./perf_matrix.md) §4.

### 7.4 Correctness Verification

> ⚠️ **PSNR AND SSIM ARE BOTH MANDATORY — NEVER OMIT SSIM.**
>
> **PSNR alone cannot detect image corruption (garbled output).**  A completely garbled image can still have PSNR > 20 dB or even > 30 dB in some cases, because PSNR only measures per-pixel difference magnitude — it is blind to structural destruction.  SSIM captures perceptual structure and will drop sharply (below 0.5) when the image is corrupted.
>
> **SSIM is not an afterthought — it is the primary guard against silent correctness bugs.**  If you only check PSNR and skip SSIM, you **will** ship broken code.

```bash
# Compare each accelerated result against baseline
cache-dit-metrics psnr ssim -i1 .tmp/{task}/<new_model>_base.png -i2 .tmp/{task}/<new_model>_cache.png
cache-dit-metrics psnr ssim -i1 .tmp/{task}/<new_model>_base.png -i2 .tmp/{task}/<new_model>_ulysses2.png
cache-dit-metrics psnr ssim -i1 .tmp/{task}/<new_model>_base.png -i2 .tmp/{task}/<new_model>_ring2.png
cache-dit-metrics psnr ssim -i1 .tmp/{task}/<new_model>_base.png -i2 .tmp/{task}/<new_model>_tp2.png
cache-dit-metrics psnr ssim -i1 .tmp/{task}/<new_model>_base.png -i2 .tmp/{task}/<new_model>_tp2_tep2.png
cache-dit-metrics psnr ssim -i1 .tmp/{task}/<new_model>_base.png -i2 .tmp/{task}/<new_model>_tp2_vaep2.png
```

**Acceptance criteria:**

- **Cache**: PSNR > 25 dB, SSIM > 0.90 (lossy optimization; exact values depend on `--rdt` threshold).
- **CP/TP/VAE-P**: PSNR > 35 dB, SSIM > 0.90.  These should be numerically near-identical to baseline; the SSIM threshold is set lower than 0.999 to accommodate real-world models like Boogu-Image where CP/TP are verified at ~0.994–0.997.
- **⚠️ Garbled-image red flag**: If PSNR is "reasonable" (e.g., 28 dB) but SSIM is below 0.5, the image is almost certainly visually corrupted. **Do not accept the result** — investigate the root cause (common culprits: missing/wrong `shard_div_attr`, DTensor placement mismatch, incorrect GQA handling).
- Always **visually inspect** at least one output image per configuration.

### 7.5 Results Table (integration phase)

The integration-phase table is deliberately coarse — it exists to prove each feature works, not to
rank configurations:

| Configuration | Latency (s) | PSNR (dB) | SSIM | GPU Mem (GB) |
| --- | --- | --- | --- | --- |
| Baseline | — | ∞ | 1.000 | — |
| Cache | — | — | — | — |
| CP Ulysses (2GPU) | — | — | — | — |
| CP Ring (2GPU) | — | — | — | — |
| TP (2GPU) | — | — | — | — |
| TP + TE-P (2GPU) | — | — | — | — |
| TP + VAE-P (2GPU) | — | — | — | — |

> The **acceptance** table is a different shape (PSNR/SSIM in `mean / min`, cache hit counts, the
> full 1/2-GPU row set, and the on-request 4/8-GPU rows). Its template lives in
> [`./perf_matrix.md`](./perf_matrix.md) §5 — use that one whenever the deliverable is a full matrix.

**Tips for collecting latency:** `--warmup 2 --repeat 2` for any number you intend to publish
(compile cost lands in warmup, and `Inference Time` is the mean over `--repeat`); `--warmup 1
--repeat 1` is only enough for a quick smoke check. Use `--track-memory` to log peak GPU memory.

### 7.6 Full Test Matrix — acceptance standard

Row list and command templates: [`./perf_matrix.md`](./perf_matrix.md) — Tier A = 1/2 GPUs (always),
Tier B = 4 GPUs, Tier C = 8 GPUs (both only when the user explicitly asks). This section defines what
"pass" means for those rows.

> **Authority**: (b) governs an **acceptance / benchmark run** of the full matrix, and for cache and
> parallel rows it supersedes the single-number thresholds in §7.4 and in `../SKILL.md` §0.6 — those
> stay the **integration** gate for a planner you are still writing. Two different products, two
> different gates; mixing them makes a healthy 2-GPU planner "fail" or a broken one "pass".

**(a) Integration gate — while writing a planner.** Unchanged from §7.4: CP / TP / TE-P / VAE-P
against the single-GPU bf16 baseline → **mean PSNR > 35 dB, SSIM > 0.90**. A planner that just
landed should be numerically near-identical; 32 dB on 2 GPUs means go read the plan.

**(b) Acceptance gate — the matrix table.**

Step 0, determinism baseline (mandatory, cheap): run single-GPU bf16 twice with the same seed under
two different tags, then compare.

- `PSNR = inf` → deterministic model. Any **non-quantized, non-cache** row (bf16 parallel / compile /
  offload + compile) below ~25 dB mean is then an integration bug, not drift.
- not `inf` → record `baseline run-to-run = X dB` in the caption and judge rows relative to it.

Per-row-class gates (PSNR mean / min, SSIM mean / min):

| Row class | PSNR mean | PSNR min | SSIM mean | SSIM min | note |
|---|---|---|---|---|---|
| bf16 parallel / compile | ≥ 28 (≈30) | > 20 | > 0.90 | > 0.88 | not bit-exact: reduction order and kernels differ per card count |
| fp8_per_row (+ compile) | > 25 | > 22 | > 0.88 | > 0.86 | GEMM-only quantization; pays off only with `--compile` |
| W4A4 svdq nvfp4 / int4 (+ compile) | > 22 | > 19 | > 0.75 | > 0.60 | min SSIM falls first — a visual check is mandatory |
| layerwise offload (no compile) | inf | inf | 1.000 | 1.000 | bit-exactness **is** its correctness proof |
| layerwise offload + compile | ≥ 28 (≈30) | > 20 | > 0.90 | > 0.88 | compile drift applies; this is **not** an `inf` row |
| cache (lossy), ± compile | — | > 20 | — | > 0.60 | record the effective threshold + hit count; a 0-hit cache is not a speedup |
| TE-P / VAE-P (run-only) | > 30 | > 20 | > 0.90 | > 0.88 | pass/fail plus numbers; never a speedup claim |

These bands are calibrated on one recorded DFR run ([`./perf_matrix.md`](./perf_matrix.md) §5) with
±2 dB model variance — a guide for reading a result, not a CI-validated baseline. Falling below a
band is a finding to report, not a threshold to lower.

**Tier B / Tier C** — 4 GPUs / 8 GPUs, run **only when the user explicitly asks**:

- Tier B must include the hybrid **TP2+CP2** row (`--parallel ulysses_tp`); Tier C must include both
  hybrid rows (**TP2+CP4** = `ulysses_tp`, **TP4+CP2** = `tp_ulysses`), each ± `--compile`, plus one
  quantized hybrid row. Rows, tags, commands: [`./perf_matrix.md`](./perf_matrix.md) §1/§4.
- Same gates, with preflight checks that must pass **before** a failure is called a bug:
  `ulysses_size * tp_size == world_size`, `heads % tp_size == 0`, and post-TP head divisibility
  `(heads // tp_size) % ulysses_size == 0` (or `--ulysses-anything` when the model accepts it) —
  [`./perf_matrix.md`](./perf_matrix.md) §3.

### 7.7 Debugging Tips

1. **Model not found**: Run `python3 -m cache_dit.generate list` to verify your model appears in the example list.
2. **BlockAdapter not activating**: Check that `check_forward_pattern=True` and verify the block forward signature matches the declared pattern.
3. **CP/TP errors**: Verify the transformer class name matches exactly what `@ContextParallelismPlannerRegister.register(...)` expects. Check `transformer.__class__.__name__`.
4. **Image corruption with TP**: Ensure `shard_div_attr(block.attn, "heads", tp_mesh.size())` is called for each attention block.
5. **Cache producing identical output to baseline**: The cache warmup may not have completed. Increase `--max-warmup-steps` or decrease `--residual-diff-threshold`.

---

## More references 

We recommend reading the following files for additional context:

- example CLI readme: `examples/README.md`
